"""
Optional auto-fix step, run after review.py, only if enable-autofix is
turned on AND the review found real bugs/security issues.

Uses LangChain's Deep Agents SDK to propose fixes for validated review
findings. The agent sees only copies of authorized files in a temporary
workspace. The checkout must be clean and at the verified PR head SHA.
Changes are tested with a trusted, shell-free command before a bounded
retry loop; only validated modifications can be committed and pushed.

Fork pull requests are rejected because the action only pushes to a
same-repository head branch.
"""

import base64
import ctypes
import hashlib
import json
import os
import re
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
from urllib.parse import urlparse
import requests

GITHUB_API = "https://api.github.com"
_MAX_GIT_METADATA_FILES = 10000
_MAX_GIT_METADATA_ENTRIES = 20000
_MAX_GIT_METADATA_BYTES = 128 * 1024 * 1024
_MAX_GIT_METADATA_DEPTH = 64


def extract_allowed_paths(issues_text: str) -> list:
    """
    Pulls the exact file paths out of the issues summary written by
    review.py (lines look like "- path/to/file.py:42 [bug] ...").
    These are the ONLY files the agent should be allowed to touch.
    """
    paths = set()
    for line in issues_text.splitlines():
        match = re.match(r"^- (.+):[1-9][0-9]* \[[^\]]+\]", line)
        if match:
            path = match.group(1)
            if _is_authorized_path(path):
                paths.add(path)
    return sorted(paths)


class ValidationError(RuntimeError):
    """A failure whose fixed message is safe to post publicly."""


class PushFailure(RuntimeError):
    def __init__(
        self,
        outcome_uncertain: bool,
        rollback_verified: bool = False,
        rollback_state: tuple = None,
    ):
        super().__init__("The validated fix could not be pushed.")
        self.outcome_uncertain = outcome_uncertain
        self.rollback_verified = rollback_verified
        self.rollback_state = rollback_state


def _is_valid_repo(repo: str) -> bool:
    if not isinstance(repo, str) or repo.count("/") != 1:
        return False
    owner, name = repo.split("/")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", owner):
        return False
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name) or name == ".":
        return False
    return ".." not in repo


def _is_valid_pr_number(pr_number: str) -> bool:
    return (
        isinstance(pr_number, str)
        and re.fullmatch(r"[1-9][0-9]*", pr_number) is not None
    )


def _is_valid_branch(branch: str) -> bool:
    """Reject ref names that Git could misinterpret or refuse."""
    if not isinstance(branch, str) or not branch or len(branch) > 255:
        return False
    if branch.startswith("-") or branch.startswith("/") or branch.endswith("/"):
        return False
    if branch == "@" or branch.endswith("."):
        return False
    if ".." in branch or "@{" in branch or "//" in branch:
        return False
    for part in branch.split("/"):
        if not part or part.startswith(".") or part.endswith(".lock"):
            return False
    for char in branch:
        if char in " ~^:?*[\\" or ord(char) < 32 or ord(char) == 127:
            return False
    return True


def _is_authorized_path(path: str) -> bool:
    if not isinstance(path, str) or not path or "\x00" in path:
        return False
    if path.startswith(("/", "\\")) or "\\" in path:
        return False

    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return False
    if os.name == "nt" and any(":" in part for part in parts):
        return False
    if any(part.casefold() == ".git" for part in parts):
        return False
    if any(part.casefold() == ".github" for part in parts):
        return False

    workspace = os.path.realpath(os.getcwd())
    current = workspace
    for part in parts:
        current = os.path.join(current, part)
        if os.path.islink(current):
            return False
    candidate = os.path.realpath(os.path.join(workspace, *parts))
    try:
        return os.path.commonpath((workspace, candidate)) == workspace
    except ValueError:
        return False


def parse_test_command(raw_command: str) -> list:
    """Parse a trusted JSON argv configuration; no shell is ever used."""
    try:
        command = json.loads(raw_command)
    except (TypeError, json.JSONDecodeError):
        raise ValidationError(
            "the configured test command must be a JSON array of arguments"
        ) from None

    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(part, str) or not part for part in command)
        or any("\x00" in part for part in command)
        or len(command) > 64
        or any(len(part) > 4096 for part in command)
    ):
        raise ValidationError(
            "the configured test command must be a non-empty JSON array of strings"
        )
    return command


def _max_fix_attempts(raw_value: str) -> int:
    if not isinstance(raw_value, str) or not re.fullmatch(r"[1-5]", raw_value):
        raise ValidationError("maximum fix attempts must be between 1 and 5")
    return int(raw_value)


def _test_environment(isolated_temp: str) -> dict:
    safe_names = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "LANG",
        "LC_ALL",
        "VIRTUAL_ENV",
        "PYTHONPATH",
    }
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.upper() in safe_names
    }
    environment["CI"] = "true"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["HOME"] = isolated_temp
    environment["USERPROFILE"] = isolated_temp
    environment["TEMP"] = isolated_temp
    environment["TMP"] = isolated_temp
    environment["TMPDIR"] = isolated_temp
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_CONFIG_COUNT"] = "1"
    environment["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"
    environment["GIT_CONFIG_VALUE_0"] = ""
    return environment


def _remove_checkout_auth_header():
    result = subprocess.run(
        [
            "git", "config", "--local", "--unset-all",
            "http.https://github.com/.extraheader",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 5):
        raise RuntimeError(
            "The checkout authentication configuration could not be sanitized."
        )


def _protect_test_process_environment():
    """Block same-UID Linux test children from reading this process's memory."""
    if sys.platform != "linux":
        raise RuntimeError(
            "Auto-fix tests require Linux process-environment isolation."
        )

    try:
        if os.geteuid() == 0:
            raise RuntimeError("The test runner must not execute as root.")
        with open("/proc/self/status", encoding="ascii") as status_file:
            capabilities = next(
                line.split(":", 1)[1].strip()
                for line in status_file
                if line.startswith("CapEff:")
            )
        if int(capabilities, 16) & (1 << 19):
            raise RuntimeError(
                "Auto-fix tests cannot run with process-inspection capability."
            )
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(4, 0, 0, 0, 0) != 0:
            raise RuntimeError
    except Exception:
        raise RuntimeError(
            "Could not isolate the fixer process from test children."
        ) from None


def run_test_command(command: list, timeout_seconds: int = 600) -> dict:
    """Run configured tests without shell parsing or inherited secrets."""
    _protect_test_process_environment()
    with tempfile.TemporaryDirectory(prefix="ai-autofix-test-home-") as home:
        with tempfile.TemporaryFile() as output:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=os.getcwd(),
                    env=_test_environment(home),
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    start_new_session=(os.name != "nt"),
                )
                try:
                    exit_code = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        process.kill()
                    else:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    process.wait()
                    timed_out = True
                    exit_code = None
                else:
                    timed_out = False
                    if os.name != "nt":
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
            except (OSError, ValueError):
                raise RuntimeError(
                    "The configured test command could not be started."
                ) from None

            output.seek(0, os.SEEK_END)
            output_size = output.tell()
            output.seek(max(0, output_size - 6000))
            captured = output.read(6000).decode("utf-8", errors="replace")

    return {
        "passed": exit_code == 0 and not timed_out,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "output": captured,
    }


def get_pull_request(repo: str, pr_number: str, token: str) -> dict:
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
    except requests.RequestException:
        raise ValidationError(
            "the pull request could not be verified with GitHub"
        ) from None
    if response.status_code != 200:
        raise ValidationError(
            "the pull request could not be verified with GitHub"
        )
    try:
        return response.json()
    except ValueError:
        raise ValidationError(
            "the pull request could not be verified with GitHub"
        ) from None


def validate_pull_request(
    repo: str,
    pr_number: str,
    branch: str,
    token: str,
    expected_head_sha: str = None,
) -> str:
    """Check that the PR targets this repo and still has the same head.

    Returns the pull request's head SHA so the caller can detect later
    changes. Every raised message is fixed text safe for a public comment.
    """
    data = get_pull_request(repo, pr_number, token)
    try:
        base_repo = data["base"]["repo"]["full_name"]
        head = data["head"]
        head_repo = head["repo"]["full_name"]
        head_ref = head["ref"]
        head_sha = head["sha"]
    except (KeyError, TypeError):
        raise ValidationError(
            "the pull request could not be verified with GitHub"
        ) from None

    if not isinstance(base_repo, str) or base_repo.lower() != repo.lower():
        raise ValidationError(
            "the pull request's base repository does not match this repository"
        )
    if data.get("number") is not None and str(data["number"]) != pr_number:
        raise ValidationError(
            "the pull request could not be verified with GitHub"
        )
    if data.get("state") is not None and data["state"] != "open":
        raise ValidationError("the pull request is not open")
    if not isinstance(head_repo, str) or head_repo.lower() != repo.lower():
        raise ValidationError(
            "the pull request comes from a fork, and the auto-fix can only "
            "push to branches in the same repository"
        )
    if head_ref != branch:
        raise ValidationError(
            "the pull request's branch no longer matches the branch being fixed"
        )
    if (
        not isinstance(head_sha, str)
        or re.fullmatch(r"[0-9a-fA-F]{40,64}", head_sha) is None
    ):
        raise ValidationError(
            "the pull request could not be verified with GitHub"
        )
    if expected_head_sha is not None and head_sha != expected_head_sha:
        raise ValidationError(
            "the pull request received new commits while the auto-fix was running"
        )
    return head_sha


def build_agent(allowed_paths: list, root_dir: str = None):
    # Imported here rather than at module level so fixer can be imported
    # (e.g. by the unit-test suite) without installing the full LangChain /
    # Deep Agents stack. Only the auto-fix path needs these packages.
    from langchain_openai import ChatOpenAI
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    model = ChatOpenAI(
        model="nvidia/nemotron-3-super-120b-a12b",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"],
        max_retries=5,
        timeout=180,
    )
    backend = FilesystemBackend(
        root_dir=root_dir or os.getcwd(),
        virtual_mode=True,
    )
    paths_list = "\n".join(f"  - {p}" for p in allowed_paths)
    return create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=(
            "You are a careful senior engineer fixing SPECIFIC issues "
            "that were flagged in a code review.\n\n"
            "You are ONLY allowed to edit these exact file(s):\n"
            f"{paths_list}\n\n"
            "Do NOT create, rename, move, or delete any other files. Do "
            "NOT touch anything under .github/ (workflow configuration) "
            "or access or modify Git metadata. Make the "
            "minimal edits needed to fix exactly the listed issues in "
            "the allowed file(s) only. Summarize exactly what you changed."
        ),
    )


def _create_agent_workspace(allowed_paths: list, root_dir: str) -> dict:
    if os.path.islink(root_dir):
        raise ValidationError("the temporary agent workspace is unsafe")
    os.makedirs(root_dir, exist_ok=True)
    if not os.path.isdir(root_dir):
        raise ValidationError("the temporary agent workspace is not empty")
    with os.scandir(root_dir) as entries:
        if next(entries, None) is not None:
            raise ValidationError("the temporary agent workspace is not empty")

    for path in allowed_paths:
        if not _is_authorized_path(path):
            raise ValidationError("an authorized file path is unsafe")
        source_fd, metadata = _open_repository_file(path, os.O_RDONLY)
        try:
            if not stat.S_ISREG(metadata.st_mode):
                raise ValidationError("an authorized file could not be prepared")
            source_content = bytearray()
            while True:
                chunk = os.read(source_fd, 65536)
                if not chunk:
                    break
                source_content.extend(chunk)
        finally:
            os.close(source_fd)
        destination = os.path.join(root_dir, *path.split("/"))
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "wb") as workspace_file:
            workspace_file.write(source_content)
    return {"root": root_dir, "allowed": set(allowed_paths)}


def _open_repository_parent(path: str):
    if (
        os.name != "posix"
        or os.open not in os.supports_dir_fd
        or os.replace not in os.supports_dir_fd
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
    ):
        raise ValidationError(
            "safe agent changes can only be applied on supported POSIX systems"
        )

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current_fd = os.open(os.path.sep, directory_flags)
    try:
        repository_root = os.path.realpath(os.getcwd())
        for component in repository_root.split(os.path.sep):
            if not component:
                continue
            next_fd = os.open(
                component, directory_flags, dir_fd=current_fd
            )
            previous_fd = current_fd
            current_fd = next_fd
            os.close(previous_fd)

        for component in path.split("/")[:-1]:
            next_fd = os.open(
                component, directory_flags, dir_fd=current_fd
            )
            previous_fd = current_fd
            current_fd = next_fd
            os.close(previous_fd)

        return current_fd, path.split("/")[-1]
    except OSError:
        os.close(current_fd)
        raise ValidationError(
            "an authorized repository path could not be safely opened"
        ) from None


def _open_repository_file(path: str, flags: int):
    parent_fd, filename = _open_repository_parent(path)
    file_fd = None
    try:
        file_fd = os.open(
            filename, flags | os.O_NOFOLLOW, dir_fd=parent_fd
        )
        metadata = os.fstat(file_fd)
        return file_fd, metadata
    except OSError:
        if file_fd is not None:
            os.close(file_fd)
        raise ValidationError(
            "an authorized repository file could not be safely opened"
        ) from None
    finally:
        os.close(parent_fd)


def _apply_authorized_file(path: str, content: bytes) -> bool:
    parent_fd, filename = _open_repository_parent(path)
    current_fd = parent_fd
    try:
        original_fd = os.open(
            filename, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd
        )
        try:
            metadata = os.fstat(original_fd)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or not metadata.st_mode & 0o222
                or metadata.st_mode & 0o7000
            ):
                raise ValidationError(
                    "an authorized destination is not a writable regular file"
                )
            original = bytearray()
            while True:
                chunk = os.read(original_fd, 65536)
                if not chunk:
                    break
                original.extend(chunk)
        finally:
            os.close(original_fd)

        if bytes(original) == content:
            return False

        mode = stat.S_IMODE(metadata.st_mode) & 0o777
        temporary_name = f".ai-autofix-{secrets.token_hex(16)}"
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        try:
            os.fchmod(temporary_fd, mode)
            remaining = memoryview(content)
            while remaining:
                written = os.write(temporary_fd, remaining)
                if written <= 0:
                    raise OSError("file write made no progress")
                remaining = remaining[written:]
            os.fsync(temporary_fd)
        except Exception:
            os.close(temporary_fd)
            os.unlink(temporary_name, dir_fd=parent_fd)
            raise
        os.close(temporary_fd)

        try:
            os.replace(
                temporary_name,
                filename,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        except Exception:
            os.unlink(temporary_name, dir_fd=parent_fd)
            raise
        return True
    except OSError:
        raise ValidationError(
            "an authorized destination could not be safely updated"
        ) from None
    finally:
        os.close(current_fd)


def _apply_agent_workspace(agent_workspace: dict):
    root = agent_workspace["root"]
    allowed = agent_workspace["allowed"]
    found = set()
    for directory, subdirectories, filenames in os.walk(
        root, followlinks=False
    ):
        if any(os.path.islink(os.path.join(directory, name)) for name in subdirectories):
            raise ValidationError("the agent created an unsafe path")
        for filename in filenames:
            agent_file_path = os.path.join(directory, filename)
            path = os.path.relpath(
                agent_file_path, root
            ).replace(os.sep, "/")
            if (
                path not in allowed
                or os.path.islink(agent_file_path)
                or not os.path.isfile(agent_file_path)
            ):
                raise ValidationError("the agent changed files outside the permitted scope")
            found.add(path)

    if found != allowed:
        raise ValidationError("the agent added, removed, or renamed a file")

    changed = []
    for path in sorted(allowed):
        source = os.path.join(root, *path.split("/"))
        with open(source, "rb") as agent_file:
            content = agent_file.read()
        if _apply_authorized_file(path, content):
            changed.append(path)
    return changed


def _git_status_entries() -> list:
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(
        [
            "git", "--no-optional-locks", "status",
            "--porcelain=v1", "-z", "--untracked-files=all",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Could not determine changed files: the current directory is "
            f"not a Git repository or Git is unavailable."
        )

    entries = []
    fields = result.stdout.split("\0")
    i = 0
    while i < len(fields):
        entry = fields[i]
        if not entry:
            i += 1
            continue

        # Each entry is "XY <path>". The path may contain spaces.
        status = entry[:2]
        path = entry[3:]
        if path:
            entries.append((status, path))

        i += 1
        if "R" in status or "C" in status:
            # Renames and copies put the original path in the next field.
            if i < len(fields) and fields[i]:
                entries.append((status, fields[i]))
                i += 1

    return entries


def _verify_repository_checkout(expected_repo: str) -> bool:
    workspace = os.path.realpath(
        os.environ.get("GITHUB_WORKSPACE", os.getcwd())
    )
    working_directory = os.path.realpath(os.getcwd())
    if os.path.normcase(workspace) != os.path.normcase(working_directory):
        return False

    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if root.returncode != 0:
        return False
    if os.path.normcase(os.path.realpath(root.stdout.strip())) != os.path.normcase(workspace):
        return False

    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
    )
    if remote.returncode != 0:
        return False
    rewrites = subprocess.run(
        [
            "git", "config", "--local", "--get-regexp",
            r"^url\..*\.insteadof$",
        ],
        capture_output=True,
        text=True,
    )
    if rewrites.returncode == 0:
        return False
    if rewrites.returncode != 1:
        return False

    remote_url = remote.stdout.strip()
    parsed_remote = urlparse(remote_url)
    if parsed_remote.scheme:
        if (
            parsed_remote.scheme.lower() not in ("https", "ssh")
            or parsed_remote.hostname is None
            or parsed_remote.hostname.lower() != "github.com"
            or parsed_remote.password is not None
            or (
                parsed_remote.scheme.lower() == "https"
                and parsed_remote.username is not None
            )
        ):
            return False
        remote_path = parsed_remote.path.lstrip("/")
    else:
        match = re.fullmatch(
            r"(?:git@)?github\.com:([^:]+)", remote_url, re.IGNORECASE
        )
        if not match:
            return False
        remote_path = match.group(1).lstrip("/")
    remote_repo = re.sub(r"\.git$", "", remote_path, flags=re.IGNORECASE)
    return remote_repo.lower() == expected_repo.lower()


def get_changed_files() -> list:
    return sorted({path for _, path in _git_status_entries()})


def _git_head_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("Could not verify the current Git commit.")
    return result.stdout.strip()


def _metadata_git_directories() -> tuple:
    directories = []
    for label, option in (
        ("worktree", "--absolute-git-dir"),
        ("common", "--git-common-dir"),
    ):
        result = subprocess.run(
            ["git", "rev-parse", option],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError("Could not verify Git metadata.")
        path = result.stdout.strip()
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        directories.append((label, os.path.abspath(path)))
    return tuple(directories)


def _metadata_repository_root() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("Could not verify Git metadata.")
    return os.path.abspath(result.stdout.strip())


def _metadata_fd_digest(
    file_fd: int,
    expected_stat=None,
    max_size: int = _MAX_GIT_METADATA_BYTES,
) -> tuple:
    opened_stat = os.fstat(file_fd)
    if not stat.S_ISREG(opened_stat.st_mode):
        raise RuntimeError("Git metadata contains an unexpected file type.")
    if expected_stat is not None and (
        opened_stat.st_dev != expected_stat.st_dev
        or opened_stat.st_ino != expected_stat.st_ino
    ):
        raise RuntimeError("Git metadata changed while it was inspected.")
    digest = hashlib.sha256()
    size = 0
    while True:
        block = os.read(file_fd, 1024 * 1024)
        if not block:
            break
        size += len(block)
        if size > max_size:
            raise RuntimeError("Git metadata exceeds the safe snapshot limit.")
        digest.update(block)
    final_stat = os.fstat(file_fd)
    if (
        size != opened_stat.st_size
        or final_stat.st_size != opened_stat.st_size
        or final_stat.st_mtime_ns != opened_stat.st_mtime_ns
        or final_stat.st_ctime_ns != opened_stat.st_ctime_ns
    ):
        raise RuntimeError("Git metadata changed while it was inspected.")
    return (
        stat.S_IMODE(final_stat.st_mode),
        size,
        digest.hexdigest(),
    )


def _metadata_file_digest(
    path: str,
    expected_stat=None,
    max_size: int = _MAX_GIT_METADATA_BYTES,
) -> tuple:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_fd = os.open(path, flags)
    try:
        return _metadata_fd_digest(file_fd, expected_stat, max_size)
    finally:
        os.close(file_fd)


def _metadata_directory_digest(path: str, expected_stat=None) -> int:
    if not getattr(os, "O_DIRECTORY", 0):
        current_stat = os.lstat(path)
        if (
            stat.S_ISLNK(current_stat.st_mode)
            or not stat.S_ISDIR(current_stat.st_mode)
            or (
                expected_stat is not None
                and (
                    current_stat.st_dev != expected_stat.st_dev
                    or current_stat.st_ino != expected_stat.st_ino
                )
            )
        ):
            raise RuntimeError("Git metadata changed while it was inspected.")
        return stat.S_IMODE(current_stat.st_mode)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path, flags)
    try:
        opened_stat = os.fstat(directory_fd)
        if not stat.S_ISDIR(opened_stat.st_mode):
            raise RuntimeError("Git metadata contains an unexpected file type.")
        if expected_stat is not None and (
            opened_stat.st_dev != expected_stat.st_dev
            or opened_stat.st_ino != expected_stat.st_ino
        ):
            raise RuntimeError("Git metadata changed while it was inspected.")
        return stat.S_IMODE(opened_stat.st_mode)
    finally:
        os.close(directory_fd)


def _git_metadata_snapshot() -> tuple:
    """Hash bounded regular metadata in both the worktree and common Git dirs.

    The object database is excluded except for objects/info controls: loose
    objects and pack payloads/indexes are content-addressed, potentially large,
    and rebuildable. Mutable refs, packed-refs, index, config, hooks, HEAD,
    reflogs, and worktree linkage files remain in the snapshot.
    """
    snapshot = {}
    file_count = 0
    entry_count = 0
    total_size = 0
    use_directory_fds = (
        os.name == "posix" and os.open in getattr(os, "supports_dir_fd", ())
    )

    def add_entry(key: str, entry_type: str, details: tuple):
        nonlocal entry_count, file_count, total_size
        entry_count += 1
        if entry_count > _MAX_GIT_METADATA_ENTRIES:
            raise RuntimeError("Git metadata exceeds the safe snapshot limit.")
        if entry_type == "file":
            file_count += 1
            total_size += details[1]
            if (
                file_count > _MAX_GIT_METADATA_FILES
                or total_size > _MAX_GIT_METADATA_BYTES
            ):
                raise RuntimeError("Git metadata exceeds the safe snapshot limit.")
        snapshot[key] = (entry_type, *details)

    def open_directory_at(parent_fd: int, name: str, expected_stat):
        directory_fd = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        opened_stat = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(opened_stat.st_mode)
            or opened_stat.st_dev != expected_stat.st_dev
            or opened_stat.st_ino != expected_stat.st_ino
        ):
            os.close(directory_fd)
            raise RuntimeError("Git metadata changed while it was inspected.")
        return directory_fd

    def scan_directory(path: str, key_prefix: str, depth: int):
        if depth > _MAX_GIT_METADATA_DEPTH:
            raise RuntimeError("Git metadata exceeds the safe snapshot limit.")
        directory_stat = os.lstat(path)
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise RuntimeError("Git metadata contains an unexpected file type.")
        mode = _metadata_directory_digest(path, directory_stat)
        add_entry(key_prefix, "directory", (mode,))

        entries = []
        with os.scandir(path) as iterator:
            for entry in iterator:
                if entry_count + len(entries) >= _MAX_GIT_METADATA_ENTRIES:
                    raise RuntimeError(
                        "Git metadata exceeds the safe snapshot limit."
                    )
                entries.append(entry)
        entries.sort(key=lambda entry: entry.name)
        for entry in entries:
            name = entry.name
            if name in (".", "..") or os.sep in name or (
                os.altsep and os.altsep in name
            ):
                raise RuntimeError("Git metadata contains an invalid entry.")
            child_path = os.path.join(path, name)
            child_key = f"{key_prefix}/{name}"
            child_stat = os.lstat(child_path)
            if stat.S_ISLNK(child_stat.st_mode):
                raise RuntimeError("Git metadata contains a symbolic link.")
            if stat.S_ISDIR(child_stat.st_mode):
                if key_prefix.endswith(":") and name == "objects":
                    mode = _metadata_directory_digest(child_path, child_stat)
                    add_entry(child_key, "directory", (mode,))
                    # The object database is content-addressed and can be
                    # large. Capture its small control metadata, not object
                    # payloads or pack files.
                    info_path = os.path.join(child_path, "info")
                    if os.path.lexists(info_path):
                        scan_directory(info_path, child_key + "/info", depth + 1)
                else:
                    scan_directory(child_path, child_key, depth + 1)
            elif stat.S_ISREG(child_stat.st_mode):
                details = _metadata_file_digest(
                    child_path,
                    child_stat,
                    _MAX_GIT_METADATA_BYTES - total_size,
                )
                add_entry(child_key, "file", details)
            else:
                raise RuntimeError("Git metadata contains an unexpected file type.")

    def scan_directory_fd(directory_fd: int, key_prefix: str, depth: int):
        nonlocal entry_count, file_count, total_size
        if depth > _MAX_GIT_METADATA_DEPTH:
            raise RuntimeError("Git metadata exceeds the safe snapshot limit.")
        current_stat = os.fstat(directory_fd)
        if not stat.S_ISDIR(current_stat.st_mode):
            raise RuntimeError("Git metadata contains an unexpected file type.")
        add_entry(key_prefix, "directory", (stat.S_IMODE(current_stat.st_mode),))
        entries = []
        with os.scandir(directory_fd) as iterator:
            for entry in iterator:
                if entry_count + len(entries) >= _MAX_GIT_METADATA_ENTRIES:
                    raise RuntimeError(
                        "Git metadata exceeds the safe snapshot limit."
                    )
                entries.append(entry)
        entries.sort(key=lambda entry: entry.name)
        for entry in entries:
            name = entry.name
            if name in (".", "..") or os.sep in name or (
                os.altsep and os.altsep in name
            ):
                raise RuntimeError("Git metadata contains an invalid entry.")
            child_key = f"{key_prefix}/{name}"
            child_stat = os.stat(
                name, dir_fd=directory_fd, follow_symlinks=False
            )
            if stat.S_ISLNK(child_stat.st_mode):
                raise RuntimeError("Git metadata contains a symbolic link.")
            if stat.S_ISDIR(child_stat.st_mode):
                if key_prefix.endswith(":") and name == "objects":
                    object_fd = open_directory_at(
                        directory_fd, name, child_stat
                    )
                    try:
                        object_stat = os.fstat(object_fd)
                        add_entry(
                            child_key,
                            "directory",
                            (stat.S_IMODE(object_stat.st_mode),),
                        )
                        try:
                            info_stat = os.stat(
                                "info",
                                dir_fd=object_fd,
                                follow_symlinks=False,
                            )
                        except FileNotFoundError:
                            info_stat = None
                        if info_stat is not None:
                            if stat.S_ISLNK(info_stat.st_mode):
                                raise RuntimeError(
                                    "Git metadata contains a symbolic link."
                                )
                            if stat.S_ISDIR(info_stat.st_mode):
                                info_fd = open_directory_at(
                                    object_fd, "info", info_stat
                                )
                                try:
                                    scan_directory_fd(
                                        info_fd, child_key + "/info", depth + 1
                                    )
                                finally:
                                    os.close(info_fd)
                            elif stat.S_ISREG(info_stat.st_mode):
                                info_fd = os.open(
                                    "info",
                                    os.O_RDONLY | os.O_NOFOLLOW
                                    | getattr(os, "O_CLOEXEC", 0),
                                    dir_fd=object_fd,
                                )
                                try:
                                    add_entry(
                                        child_key + "/info",
                                        "file",
                                        _metadata_fd_digest(
                                            info_fd,
                                            info_stat,
                                            _MAX_GIT_METADATA_BYTES
                                            - total_size,
                                        ),
                                    )
                                finally:
                                    os.close(info_fd)
                            else:
                                raise RuntimeError(
                                    "Git metadata contains an unexpected file type."
                                )
                    finally:
                        os.close(object_fd)
                else:
                    child_fd = os.open(
                        name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                        | getattr(os, "O_CLOEXEC", 0),
                        dir_fd=directory_fd,
                    )
                    try:
                        opened_stat = os.fstat(child_fd)
                        if (
                            opened_stat.st_dev != child_stat.st_dev
                            or opened_stat.st_ino != child_stat.st_ino
                        ):
                            raise RuntimeError(
                                "Git metadata changed while it was inspected."
                            )
                        scan_directory_fd(child_fd, child_key, depth + 1)
                    finally:
                        os.close(child_fd)
            elif stat.S_ISREG(child_stat.st_mode):
                flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
                file_fd = os.open(name, flags, dir_fd=directory_fd)
                try:
                    details = _metadata_fd_digest(
                        file_fd,
                        child_stat,
                        _MAX_GIT_METADATA_BYTES - total_size,
                    )
                finally:
                    os.close(file_fd)
                add_entry(child_key, "file", details)
            else:
                raise RuntimeError("Git metadata contains an unexpected file type.")

    for label, directory in _metadata_git_directories():
        try:
            if use_directory_fds:
                directory_fd = os.open(
                    os.sep,
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
                )
                try:
                    for part in os.path.abspath(directory).split(os.sep):
                        if not part:
                            continue
                        next_fd = os.open(
                            part,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                            | getattr(os, "O_CLOEXEC", 0),
                            dir_fd=directory_fd,
                        )
                        os.close(directory_fd)
                        directory_fd = next_fd
                    scan_directory_fd(directory_fd, label + ":", 0)
                finally:
                    os.close(directory_fd)
            else:
                scan_directory(directory, label + ":", 0)
        except OSError:
            raise RuntimeError("Could not safely inspect Git metadata.") from None

    git_entry = os.path.join(_metadata_repository_root(), ".git")
    try:
        git_entry_stat = os.lstat(git_entry)
    except FileNotFoundError:
        git_entry_stat = None
    if git_entry_stat is not None:
        if stat.S_ISLNK(git_entry_stat.st_mode):
            raise RuntimeError("Git metadata contains a symbolic link.")
        if stat.S_ISREG(git_entry_stat.st_mode):
            add_entry(
                "checkout:.git",
                "file",
                _metadata_file_digest(
                    git_entry,
                    git_entry_stat,
                    _MAX_GIT_METADATA_BYTES - total_size,
                ),
            )
        elif not stat.S_ISDIR(git_entry_stat.st_mode):
            raise RuntimeError("Git metadata contains an unexpected file type.")

    return tuple(sorted(snapshot.items()))


def _working_tree_state() -> tuple:
    file_states = []
    for status, path in _git_status_entries():
        if not os.path.lexists(path):
            file_states.append((status, path, None))
        elif os.path.islink(path):
            file_states.append((status, path, os.readlink(path)))
        else:
            with open(path, "rb") as changed_file:
                digest = hashlib.sha256(changed_file.read()).hexdigest()
            file_states.append((status, path, digest))
    index_tree = subprocess.run(
        ["git", "write-tree"],
        capture_output=True,
        text=True,
    )
    if index_tree.returncode != 0:
        raise RuntimeError("Could not verify the Git index.")
    return (
        _git_head_sha(),
        tuple(file_states),
        index_tree.stdout.strip(),
        _git_metadata_snapshot(),
    )


def _verify_changes_authorized(changed_files: list, allowed_paths: list):
    allowed = set(allowed_paths)
    unexpected = [
        path for path in changed_files
        if path not in allowed or not _is_authorized_path(path)
    ]
    if unexpected:
        raise ValidationError(
            "Refusing changes outside the allowed list."
        )


def _staged_status_entries() -> list:
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-status", "-z", "-M"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not inspect staged changes."
        )

    entries = []
    fields = result.stdout.split("\0")
    i = 0
    while i < len(fields):
        status = fields[i]
        if not status:
            i += 1
            continue

        if status[0] in ("R", "C"):
            # Renames and copies are followed by the old and new paths.
            entries.append((status, fields[i + 1]))
            entries.append((status, fields[i + 2]))
            i += 3
        else:
            entries.append((status, fields[i + 1]))
            i += 2

    return entries


def _validate_staged_changes(allowed_paths: list) -> list:
    allowed = set(allowed_paths)
    entries = _staged_status_entries()
    if (
        not entries
        or any(
            status != "M"
            or path not in allowed
            or not _is_authorized_path(path)
            for status, path in entries
        )
    ):
        raise RuntimeError(
            "Refusing to commit: only modifications to existing authorized "
            "files are allowed."
        )

    unstaged = subprocess.run(
        ["git", "diff", "--quiet"],
        capture_output=True,
    )
    if unstaged.returncode != 0:
        raise RuntimeError(
            "Refusing to commit: working-tree changes differ from the staged "
            "changes."
        )

    current_files = set(get_changed_files())
    staged_files = {path for _, path in entries}
    _verify_changes_authorized(sorted(current_files), allowed_paths)
    if current_files != staged_files:
        raise RuntimeError(
            "Refusing to commit: staged changes no longer match the working tree."
        )
    return entries


def discard_changes_since(initial_status: list):
    initial_paths = {path for _, path in initial_status}
    for status, path in _git_status_entries():
        if path in initial_paths:
            continue

        if status == "??":
            if os.path.lexists(path):
                os.remove(path)
            continue

        tracked = subprocess.run(
            [
                "git",
                "--literal-pathspecs",
                "ls-tree",
                "-r",
                "--name-only",
                "-z",
                "HEAD",
                "--",
                path,
            ],
            capture_output=True,
            text=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(
                f"Could not determine whether {path!r} exists in HEAD."
            )
        if path in tracked.stdout.split("\0"):
            subprocess.run(
                [
                    "git",
                    "--literal-pathspecs",
                    "restore",
                    "--source=HEAD",
                    "--staged",
                    "--worktree",
                    "--",
                    path,
                ],
                check=True,
            )
        else:
            subprocess.run(
                ["git", "--literal-pathspecs", "reset", "--", path],
                check=True,
            )
            if os.path.lexists(path):
                os.remove(path)

    if _git_status_entries() != initial_status:
        raise RuntimeError("Rollback could not be verified.")


def _rollback_generated_commit(
    base_sha: str,
    generated_sha: str,
    expected_state: tuple = None,
):
    if _git_head_sha().lower() == base_sha.lower():
        state = _working_tree_state()
        if expected_state is not None and not _rollback_state_matches(
            state, expected_state
        ):
            raise RuntimeError("The generated commit rollback could not be verified.")
        return state
    if _git_head_sha().lower() != generated_sha.lower():
        raise RuntimeError("The generated commit could not be rolled back safely.")
    if _git_status_entries():
        raise RuntimeError("The generated commit could not be rolled back safely.")
    parent = subprocess.run(
        ["git", "rev-parse", "HEAD^"],
        capture_output=True,
        text=True,
    )
    if (
        parent.returncode != 0
        or parent.stdout.strip().lower() != base_sha.lower()
    ):
        raise RuntimeError("The generated commit could not be rolled back safely.")

    before_reset = dict(_git_metadata_snapshot())
    symbolic_ref = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        capture_output=True,
        text=True,
    )
    if symbolic_ref.returncode not in (0, 1):
        raise RuntimeError("The generated commit could not be rolled back safely.")
    expected_reflogs = {
        "worktree:/logs/HEAD",
        "common:/logs/HEAD",
    }
    expected_ref_files = set()
    if symbolic_ref.returncode == 0:
        expected_reflogs.add("common:/logs/" + symbolic_ref.stdout.strip())
        expected_reflogs.add("worktree:/logs/" + symbolic_ref.stdout.strip())
        expected_ref_files.add("common:/" + symbolic_ref.stdout.strip())
        expected_ref_files.add("worktree:/" + symbolic_ref.stdout.strip())
    allowed_commit_changes = {
        "worktree:/HEAD",
        "common:/HEAD",
        "worktree:/index",
        "common:/index",
        "worktree:/COMMIT_EDITMSG",
        "common:/COMMIT_EDITMSG",
    } | expected_reflogs | expected_ref_files
    if expected_state is not None:
        expected_metadata = dict(expected_state[3])
        for key in set(expected_metadata) | set(before_reset):
            if expected_metadata.get(key) != before_reset.get(key):
                if key not in allowed_commit_changes:
                    raise RuntimeError("Git metadata changed before rollback.")
    base_tree = subprocess.run(
        ["git", "rev-parse", f"{base_sha}^{{tree}}"],
        capture_output=True,
        text=True,
    )
    generated_tree = subprocess.run(
        ["git", "rev-parse", f"{generated_sha}^{{tree}}"],
        capture_output=True,
        text=True,
    )
    index_tree = subprocess.run(
        ["git", "write-tree"],
        capture_output=True,
        text=True,
    )
    if (
        base_tree.returncode != 0
        or generated_tree.returncode != 0
        or index_tree.returncode != 0
        or generated_tree.stdout.strip() != index_tree.stdout.strip()
    ):
        raise RuntimeError("The generated commit could not be rolled back safely.")
    subprocess.run(
        ["git", "reset", "--mixed", base_sha],
        check=True,
        capture_output=True,
        text=True,
    )
    discard_changes_since([])
    state = _working_tree_state()
    current_tree = subprocess.run(
        ["git", "write-tree"],
        capture_output=True,
        text=True,
    )
    if (
        _git_head_sha().lower() != base_sha.lower()
        or _git_status_entries()
        or _staged_status_entries()
        or current_tree.returncode != 0
        or current_tree.stdout.strip() != base_tree.stdout.strip()
        or (
            expected_state is not None
            and state[1] != expected_state[1]
        )
        or state[2] != base_tree.stdout.strip()
    ):
        raise RuntimeError("The generated commit rollback could not be verified.")
    after_reset = dict(state[3])
    allowed_reset_changes = allowed_commit_changes | {
        "worktree:/ORIG_HEAD",
        "common:/ORIG_HEAD",
    }
    for key in set(before_reset) | set(after_reset):
        if before_reset.get(key) == after_reset.get(key):
            continue
        if key in allowed_reset_changes:
            continue
        raise RuntimeError("Git metadata changed during rollback.")
    return state


def commit_and_push(
    branch: str,
    token: str,
    repo: str,
    allowed_paths: list,
    pr_number: str = None,
    expected_head_sha: str = None,
    expected_state: tuple = None,
):
    if not _is_valid_repo(repo) or not _is_valid_branch(branch):
        raise ValidationError("the push destination could not be verified")
    if expected_head_sha is not None and (
        not _is_valid_pr_number(pr_number)
        or _git_head_sha().lower() != expected_head_sha.lower()
    ):
        raise ValidationError("the pull request destination changed")

    # Refuse to include changes that were staged before the agent ran.
    if _staged_status_entries():
        raise RuntimeError(
            "Refusing to commit: the Git index already contained staged "
            "changes before the auto-fix was applied."
        )

    allowed = set(allowed_paths)
    changed_files = get_changed_files()
    _verify_changes_authorized(changed_files, allowed_paths)
    if not changed_files:
        raise RuntimeError("Refusing to commit: no changes to commit.")

    # Stage only the authorized paths, never `git add -A`.
    subprocess.run(
        ["git", "--literal-pathspecs", "add", "--", *changed_files],
        check=True,
    )

    staged_entries = _validate_staged_changes(allowed_paths)

    base_sha = _git_head_sha()
    if expected_head_sha is not None:
        if base_sha.lower() != expected_head_sha.lower():
            raise ValidationError("the pull request destination changed")
        validate_pull_request(
            repo, pr_number, branch, token, expected_head_sha
        )
        if _validate_staged_changes(allowed_paths) != staged_entries:
            raise RuntimeError("The staged changes changed before commit.")
        if _git_head_sha().lower() != expected_head_sha.lower():
            raise ValidationError("the checkout head changed")

    hooks = tempfile.TemporaryDirectory(prefix="ai-autofix-hooks-")
    subprocess.run(
        [
            "git",
            "-c", "user.email=ai-fixer@bot.local",
            "-c", "user.name=AI Auto-Fixer",
            "-c", f"core.hooksPath={hooks.name}",
            "-c", "commit.gpgSign=false",
            "commit", "-m", "AI auto-fix: address code review comments",
        ],
        check=True,
    )
    committed_head = _git_head_sha()
    try:
        parent = subprocess.run(
            ["git", "rev-parse", "HEAD^"],
            capture_output=True,
            text=True,
        )
        if (
            committed_head.lower() == base_sha.lower()
            or parent.returncode != 0
            or parent.stdout.strip().lower() != base_sha.lower()
        ):
            raise RuntimeError("The validated commit was not created.")
        if expected_head_sha is not None:
            validate_pull_request(
                repo, pr_number, branch, token, expected_head_sha
            )
    except Exception:
        if committed_head.lower() != base_sha.lower():
            _rollback_generated_commit(base_sha, committed_head, expected_state)
        raise

    # Send the token to Git through environment configuration, not the
    # command line or URL, so it cannot appear in an exception or log.
    credentials = base64.b64encode(
        f"x-access-token:{token}".encode("utf-8")
    ).decode("ascii")
    push_env = os.environ.copy()
    push_env["GIT_TERMINAL_PROMPT"] = "0"
    push_env["GIT_CONFIG_NOSYSTEM"] = "1"
    push_env["GIT_CONFIG_GLOBAL"] = os.devnull
    # `actions/checkout` may have saved its own Authorization header in the
    # repository's Git config. An empty extraheader value resets those
    # inherited values, so only the header below is sent to GitHub.
    push_env["GIT_CONFIG_COUNT"] = "3"
    push_env["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"
    push_env["GIT_CONFIG_VALUE_0"] = ""
    push_env["GIT_CONFIG_KEY_1"] = "http.https://github.com/.extraheader"
    push_env["GIT_CONFIG_VALUE_1"] = f"Authorization: Basic {credentials}"
    push_env["GIT_CONFIG_KEY_2"] = "core.hooksPath"
    push_env["GIT_CONFIG_VALUE_2"] = hooks.name

    try:
        result = subprocess.run(
            [
                "git", "push", f"https://github.com/{repo}.git",
                f"HEAD:refs/heads/{branch}",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=push_env,
        )
    except Exception:
        rollback_state = None
        rollback_verified = False
        if expected_head_sha is not None:
            try:
                rollback_state = _rollback_generated_commit(
                    base_sha, committed_head, expected_state
                )
                rollback_verified = True
            except Exception:
                pass
        raise PushFailure(
            outcome_uncertain=True,
            rollback_verified=rollback_verified,
            rollback_state=rollback_state,
        ) from None
    if result.returncode != 0:
        rollback_state = None
        rollback_verified = False
        if expected_head_sha is not None:
            try:
                rollback_state = _rollback_generated_commit(
                    base_sha, committed_head, expected_state
                )
                rollback_verified = True
            except Exception:
                pass
        raise PushFailure(
            outcome_uncertain=True,
            rollback_verified=rollback_verified,
            rollback_state=rollback_state,
        )


def post_comment(repo: str, pr_number: str, token: str, body: str):
    url = f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    try:
        response = requests.post(
            url, headers=headers, json={"body": body}, timeout=30
        )
        response.raise_for_status()
    except requests.RequestException:
        raise RuntimeError("The auto-fix status could not be reported to GitHub.") from None


def _report(repo: str, pr_number: str, token: str, body: str):
    try:
        post_comment(repo, pr_number, token, body)
    except RuntimeError:
        print("The auto-fix status could not be reported to GitHub.")


def _state_metadata(state: tuple) -> tuple:
    return state[0], state[2], state[3]


def _state_metadata_without_index(state: tuple) -> tuple:
    metadata = tuple(
        (key, value)
        for key, value in state[3]
        if key.split(":", 1)[-1] != "/index"
    )
    return state[0], state[2], metadata


def _with_current_index_metadata(
    expected_state: tuple,
    current_state: tuple,
) -> tuple:
    expected_metadata = dict(expected_state[3])
    current_metadata = dict(current_state[3])
    for key in set(expected_metadata) | set(current_metadata):
        if key.split(":", 1)[-1] == "/index":
            continue
        if expected_metadata.get(key) != current_metadata.get(key):
            raise RuntimeError("Git metadata changed before test execution.")
    if (
        expected_state[0] != current_state[0]
        or expected_state[2] != current_state[2]
    ):
        raise RuntimeError("Git checkout state changed before test execution.")
    for key in list(expected_metadata):
        if key.split(":", 1)[-1] == "/index":
            del expected_metadata[key]
    for key, value in current_metadata.items():
        if key.split(":", 1)[-1] == "/index":
            expected_metadata[key] = value
    return (
        expected_state[0],
        expected_state[1],
        expected_state[2],
        tuple(sorted(expected_metadata.items())),
    )


def _state_status(state: tuple) -> list:
    return [(status, path) for status, path, _ in state[1]]


def _rollback_state_matches(actual_state: tuple, expected_state: tuple) -> bool:
    actual_metadata = {
        key: value
        for key, value in actual_state[3]
        if key.split(":", 1)[-1] != "/index"
    }
    expected_metadata = {
        key: value
        for key, value in expected_state[3]
        if key.split(":", 1)[-1] != "/index"
    }
    return (
        actual_state[0] == expected_state[0]
        and actual_state[1] == expected_state[1]
        and actual_state[2] == expected_state[2]
        and actual_metadata == expected_metadata
    )


def _rollback_attempt(expected_state: tuple):
    current_state = _working_tree_state()
    if (
        current_state[0] != expected_state[0]
        or current_state[2] != expected_state[2]
        or current_state[3] != expected_state[3]
    ):
        raise RuntimeError("Git metadata changed; rollback is unsafe.")
    discard_changes_since(_state_status(expected_state))
    if not _rollback_state_matches(_working_tree_state(), expected_state):
        raise RuntimeError("Rollback could not be verified.")


def _attempt_failure_text(result: dict, attempt: int) -> str:
    if result["timed_out"]:
        reason = "The configured test command timed out."
    else:
        reason = (
            "The configured test command exited with code "
            f"{result['exit_code']}."
        )
    output = result["output"].strip() or "(no test output)"
    return (
        f"Attempt {attempt} validation failure: {reason}\n"
        "Bounded test output follows:\n"
        f"{output}"
    )


def _redact_test_output(output: str, secrets: list) -> str:
    for secret in secrets:
        if secret:
            output = output.replace(secret, "[redacted]")
    return output


def main():
    issues_text = os.environ.get("REVIEW_ISSUES", "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    pr_number = os.environ.get("PR_NUMBER", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    branch = os.environ.get("PR_HEAD_REF", "")

    if not issues_text:
        print("No flagged bugs/security issues -- skipping auto-fix.")
        return

    if not _is_valid_repo(repo) or not _is_valid_pr_number(pr_number):
        print("Invalid repository or pull request number -- skipping auto-fix.")
        return
    if not token:
        print("Missing GitHub token -- skipping auto-fix.")
        return
    if not _is_valid_branch(branch):
        post_comment(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix skipped\n\n"
            "**Reason:** the pull request's branch name is not valid, so "
            "nothing was pushed.",
        )
        print("Invalid branch name -- skipping auto-fix.")
        return

    if not os.environ.get("AUTOFIX_TEST_COMMAND", "").strip():
        _report(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix not started\n\n"
            "No trusted test command is configured. No fix was generated, "
            "tested, committed, or pushed.",
        )
        print("Auto-fix requires a trusted test command; none was configured.")
        return

    try:
        test_command = parse_test_command(
            os.environ["AUTOFIX_TEST_COMMAND"]
        )
        max_attempts = _max_fix_attempts(
            os.environ.get("AUTOFIX_MAX_ATTEMPTS", "2")
        )
    except ValidationError as error:
        _report(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix could not safely continue\n\n"
            f"**Reason:** {error}. Nothing was pushed.",
        )
        print(f"Auto-fix configuration rejected: {error}")
        return

    allowed_paths = extract_allowed_paths(issues_text)
    if not allowed_paths:
        _report(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix not started\n\n"
            "No safe, authorized file paths could be identified. "
            "No fix was generated, tested, committed, or pushed.",
        )
        print("No safe authorized file paths were identified.")
        return

    try:
        expected_head_sha = validate_pull_request(
            repo, pr_number, branch, token
        )
    except ValidationError as e:
        post_comment(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix skipped\n\n"
            f"**Reason:** {e}, so nothing was pushed.",
        )
        print(f"Pull request validation failed: {e}")
        return

    if not _verify_repository_checkout(repo):
        _report(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix could not safely continue\n\n"
            "The checkout does not match the verified PR repository. "
            "No fix was generated, tested, committed, or pushed.",
        )
        print("The checkout does not match the verified PR repository.")
        return

    if _git_head_sha().lower() != expected_head_sha.lower():
        _report(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix could not safely continue\n\n"
            "The checkout is not at the verified pull request head commit. "
            "No fix was generated, tested, committed, or pushed.",
        )
        print("Checkout HEAD does not match the verified PR head.")
        return

    try:
        initial_status = _git_status_entries()
    except Exception:
        print("Could not verify a clean Git working tree.")
        return
    if initial_status:
        print("Git working tree is not clean -- skipping auto-fix.")
        _report(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix skipped\n\n"
            "**Reason:** the Git working tree was not clean before the agent "
            "ran, so no files were changed.",
        )
        return

    try:
        _remove_checkout_auth_header()
    except RuntimeError:
        _report(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix could not safely continue\n\n"
            "The checkout authentication configuration could not be "
            "sanitized. No fix was generated, tested, committed, or pushed.",
        )
        print("Checkout authentication configuration could not be sanitized.")
        return

    try:
        initial_state = _working_tree_state()
        if (
            initial_state[0].lower() != expected_head_sha.lower()
            or initial_state[1]
        ):
            raise RuntimeError("The checkout changed during initialization.")
    except Exception:
        _report(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix could not safely continue\n\n"
            "Git checkout integrity could not be verified. No fix was "
            "generated, tested, committed, or pushed.",
        )
        print("Could not verify Git checkout integrity.")
        return

    try:
        sandbox = tempfile.TemporaryDirectory(prefix="ai-autofix-")
        agent_workspace = _create_agent_workspace(
            allowed_paths, sandbox.name
        )
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        agent = build_agent(allowed_paths, sandbox.name)
    except Exception:
        _report(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix could not safely continue\n\n"
            "The fixing agent could not be initialized. Nothing was tested, "
            "committed, or pushed.",
        )
        print("The fixing agent could not be initialized.")
        return

    # The model client already captured its API key. Keep credentials out
    # of the agent's tools and all later test subprocess environments.
    os.environ.pop("NVIDIA_API_KEY", None)
    os.environ.pop("GITHUB_TOKEN", None)

    feedback = ""
    generated_any = False
    last_test_result = None
    for attempt in range(1, max_attempts + 1):
        task = (
            "Fix these specific issues found in code review. Make only the "
            f"minimal changes needed, in the allowed file(s) only:\n\n{issues_text}"
        )
        if feedback:
            task += (
                "\n\nA previous fix attempt was rolled back after validation "
                "failed. Use this failure information to correct the fix:\n\n"
                f"{feedback}"
            )

        try:
            agent.invoke({"messages": [{"role": "user", "content": task}]})
            _apply_agent_workspace(agent_workspace)
            changed_files = get_changed_files()
            if not changed_files:
                if not generated_any:
                    _report(
                        repo, pr_number, token,
                        "### \U0001F916 Auto-fix completed\n\n"
                        "No applicable fix was generated. Nothing was tested, "
                        "committed, or pushed.",
                    )
                    print("Agent made no file changes.")
                else:
                    _report(
                        repo, pr_number, token,
                        "### \U0001F916 Auto-fix attempts exhausted\n\n"
                        "A fix was generated, but all validation attempts "
                        "failed and were rolled back. Nothing was committed "
                        "or pushed.",
                    )
                return

            generated_any = True
            _verify_changes_authorized(changed_files, allowed_paths)
        except ValidationError:
            try:
                _rollback_attempt(initial_state)
            except Exception:
                _report(
                    repo, pr_number, token,
                    "### \U0001F916 Auto-fix could not safely continue\n\n"
                    "Unauthorized changes were detected, but rollback could "
                    "not be verified. Manual inspection is required; no "
                    "commit or push was attempted.",
                )
                print("Unauthorized changes detected; rollback failed.")
                return
            _report(
                repo, pr_number, token,
                "### \U0001F916 Auto-fix rejected\n\n"
                "The generated changes included files outside the permitted "
                "scope. They were rolled back; no tests, commit, or push "
                "were performed.",
            )
            print("Unauthorized changes detected and rolled back.")
            return
        except Exception:
            try:
                _rollback_attempt(initial_state)
            except Exception:
                _report(
                    repo, pr_number, token,
                    "### \U0001F916 Auto-fix could not safely continue\n\n"
                    "An execution failure occurred and rollback could not be "
                    "verified. Manual inspection is required; no commit or "
                    "push was attempted.",
                )
                print("Execution failed and rollback could not be verified.")
                return
            _report(
                repo, pr_number, token,
                "### \U0001F916 Auto-fix could not safely continue\n\n"
                "The fixing agent failed. Its changes were rolled back; "
                "nothing was tested, committed, or pushed.",
            )
            print("The fixing agent failed; changes were rolled back.")
            return

        try:
            state_before_test = _working_tree_state()
            if (
                _state_metadata_without_index(state_before_test)
                != _state_metadata_without_index(initial_state)
            ):
                _report(
                    repo, pr_number, token,
                    "### \U0001F916 Auto-fix stopped safely\n\n"
                    "Git metadata changed before test execution. Rollback "
                    "could not be performed safely; manual inspection is "
                    "required. No retry, commit, or push was attempted.",
                )
                print("Git metadata changed before test execution.")
                return
            initial_state = _with_current_index_metadata(
                initial_state, state_before_test
            )
            last_test_result = run_test_command(test_command)
        except Exception:
            try:
                _rollback_attempt(initial_state)
            except Exception:
                _report(
                    repo, pr_number, token,
                    "### \U0001F916 Auto-fix could not safely continue\n\n"
                    "The test command could not start and rollback could not "
                    "be verified. Manual inspection is required; no commit "
                    "or push was attempted.",
                )
                print("Test infrastructure failed and rollback was not verified.")
                return
            _report(
                repo, pr_number, token,
                "### \U0001F916 Auto-fix could not safely continue\n\n"
                "The configured test command could not start. This is an "
                "infrastructure failure, not a test failure. Changes were "
                "rolled back; nothing was committed or pushed.",
            )
            print("The configured test command could not start.")
            return

        try:
            state_after_test = _working_tree_state()
        except Exception:
            _report(
                repo, pr_number, token,
                "### \U0001F916 Auto-fix stopped safely\n\n"
                "Git checkout integrity could not be verified after tests. "
                "Rollback could not be performed safely; manual inspection "
                "is required. No retry, commit, or push was attempted.",
            )
            print("Git checkout integrity could not be verified after tests.")
            return

        if (
            _state_metadata(state_after_test)
            != _state_metadata(state_before_test)
        ):
            _report(
                repo, pr_number, token,
                "### \U0001F916 Auto-fix stopped safely\n\n"
                "The test command changed Git metadata or checkout integrity. "
                "Rollback could not be performed safely; manual inspection "
                "is required. No retry, commit, or push was attempted.",
            )
            print("Test command changed Git metadata.")
            return

        if state_after_test != state_before_test:
            try:
                _rollback_attempt(initial_state)
            except Exception:
                _report(
                    repo, pr_number, token,
                    "### \U0001F916 Auto-fix could not safely continue\n\n"
                    "The test command changed files and rollback could not "
                    "be verified. Manual inspection is required; no retry, "
                    "commit, or push was attempted.",
                )
                print("Test command changed checkout state; rollback failed.")
                return
            _report(
                repo, pr_number, token,
                "### \U0001F916 Auto-fix stopped safely\n\n"
                "The test command changed files outside the validated fix. "
                "The attempt was rolled back and will not be "
                "retried, committed, or pushed.",
            )
            print("Test command changed checkout state.")
            return

        if last_test_result["passed"]:
            break

        failure_result = dict(last_test_result)
        failure_result["output"] = _redact_test_output(
            failure_result["output"], [token, api_key]
        )
        feedback = _attempt_failure_text(failure_result, attempt)
        try:
            _rollback_attempt(initial_state)
        except Exception:
            _report(
                repo, pr_number, token,
                "### \U0001F916 Auto-fix could not safely continue\n\n"
                "Validation failed and rollback could not be verified. "
                "Manual inspection is required; no commit or push was "
                "attempted.",
            )
            print("Test validation failed and rollback was not verified.")
            return

        if attempt == max_attempts:
            _report(
                repo, pr_number, token,
                "### \U0001F916 Auto-fix attempts exhausted\n\n"
                "A fix was generated, but tests failed on every attempt. "
                "Each failed attempt was rolled back; nothing was committed "
                "or pushed.",
            )
            print("All configured fix attempts failed validation.")
            return

    try:
        validate_pull_request(repo, pr_number, branch, token, expected_head_sha)
        if _git_head_sha().lower() != expected_head_sha.lower():
            raise ValidationError("the checkout head changed")
        commit_and_push(
            branch, token, repo, allowed_paths,
            pr_number=pr_number,
            expected_head_sha=expected_head_sha,
            expected_state=initial_state,
        )
    except PushFailure as failure:
        try:
            if not failure.rollback_verified or failure.rollback_state is None:
                raise RuntimeError("The local rollback was not verified.")
            current_state = _working_tree_state()
            if (
                current_state != failure.rollback_state
                or current_state[0].lower() != expected_head_sha.lower()
                or current_state[1] != initial_state[1]
            ):
                raise RuntimeError("The local rollback state changed.")
        except Exception:
            _report(
                repo, pr_number, token,
                "### \U0001F916 Auto-fix push outcome unknown\n\n"
                "The push may have been accepted remotely, and local rollback "
                "could not be verified. Inspect the PR branch and checkout "
                "manually before retrying.",
            )
            print("Push outcome is unknown; local rollback was not verified.")
            return
        if failure.outcome_uncertain:
            _report(
                repo, pr_number, token,
                "### \U0001F916 Auto-fix push outcome unknown\n\n"
                "The fix passed tests, but Git did not confirm the push. "
                "The remote may have accepted the commit. The local attempt "
                "was rolled back; inspect the PR branch before retrying.",
            )
            print("Push outcome is unknown; inspect the PR branch before retrying.")
        else:
            _report(
                repo, pr_number, token,
                "### \U0001F916 Auto-fix could not be pushed\n\n"
                "The fix passed tests, but GitHub did not confirm the push. "
                "The local attempt was rolled back.",
            )
            print("The validated fix could not be pushed.")
        return
    except ValidationError:
        try:
            _rollback_attempt(initial_state)
        except Exception:
            _report(
                repo, pr_number, token,
                "### \U0001F916 Auto-fix could not safely continue\n\n"
                "The PR destination changed after tests passed, and safe "
                "rollback could not be verified. Manual inspection is "
                "required; no push was attempted.",
            )
            print("PR destination changed and rollback could not be verified.")
            return
        _report(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix stopped safely\n\n"
            "The PR received new commits or its destination could not be "
            "verified after the fix passed tests. The local fix was rolled "
            "back; nothing was pushed.",
        )
        print("PR destination changed after tests; the fix was rolled back.")
        return
    except Exception:
        try:
            _rollback_attempt(initial_state)
        except Exception:
            _report(
                repo, pr_number, token,
                "### \U0001F916 Auto-fix could not safely continue\n\n"
                "The fix passed tests, but commit/push safety checks failed "
                "and safe rollback could not be verified. Manual inspection "
                "is required; do not retry blindly.",
            )
            print("Commit/push failed and rollback could not be verified.")
            return
        _report(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix could not be pushed\n\n"
            "The fix was generated and passed tests, but could not be "
            "committed and pushed to the verified PR branch. The local "
            "attempt was rolled back.",
        )
        print("The validated fix could not be committed and pushed.")
        return

    _report(
        repo, pr_number, token,
        "### \U0001F916 Auto-fix pushed\n\n"
        "A fix was generated, passed the configured tests, committed, and "
        "successfully pushed to the verified PR branch. Please review it "
        "before merging.",
    )
    print("Auto-fix passed tests, was committed, and pushed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Auto-fix step failed; details were withheld for safety.", file=sys.stderr)
        sys.exit(1)