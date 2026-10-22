"""
Optional auto-fix step, run after review.py, only if enable-autofix is
turned on AND the review found real bugs/security issues.

Uses LangChain's Deep Agents SDK, sandboxed to the checked-out PR
branch, to attempt fixes for the SPECIFIC issues the reviewer flagged.
If it makes changes, it commits and pushes them as a new commit
directly onto the PR's own branch, then comments explaining what
happened.

SAFETY: two layers stop the agent from editing the wrong thing --
(1) the prompt explicitly names the exact files it's allowed to touch
and forbids anything under .github/, and (2) after it runs, a real
`git status` check verifies it only touched those files before anything
is committed or pushed. The worktree must be clean before the agent runs;
if it strays, only changes made since the initial status snapshot are
discarded and a comment explains why.

LIMITATION: pushing still only works for PRs from a branch within the
SAME repo -- GITHUB_TOKEN can't write to a fork's branches.
"""

import base64
import os
import re
import subprocess
import sys
from urllib.parse import urlparse
import requests

GITHUB_API = "https://api.github.com"


def extract_allowed_paths(issues_text: str) -> list:
    """
    Pulls the exact file paths out of the issues summary written by
    review.py (lines look like "- path/to/file.py:42 [bug] ...").
    These are the ONLY files the agent should be allowed to touch.
    """
    paths = set()
    for line in issues_text.splitlines():
        match = re.match(r"^-\s*([^\s:]+):\d+", line.strip())
        if match:
            paths.add(match.group(1))
    return sorted(paths)


class ValidationError(Exception):
    """A failure whose fixed message is safe to post publicly."""


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


def build_agent(allowed_paths: list):
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
    backend = FilesystemBackend(root_dir=os.getcwd(), virtual_mode=True)
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
            "under any circumstances, even if it seems related. Make the "
            "minimal edits needed to fix exactly the listed issues in "
            "the allowed file(s) only. Summarize exactly what you changed."
        ),
    )


def _git_status_entries() -> list:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        capture_output=True,
        text=True,
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

    remote_url = remote.stdout.strip()
    if "://" in remote_url:
        remote_path = urlparse(remote_url).path.lstrip("/")
    else:
        remote_path = remote_url.rsplit(":", 1)[-1]
    remote_repo = re.sub(r"\.git$", "", remote_path, flags=re.IGNORECASE)
    return remote_repo.lower() == expected_repo.lower()


def get_changed_files() -> list:
    return sorted({path for _, path in _git_status_entries()})


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


def commit_and_push(branch: str, token: str, repo: str, allowed_paths: list):
    # Refuse to include changes that were staged before the agent ran.
    if _staged_status_entries():
        raise RuntimeError(
            "Refusing to commit: the Git index already contained staged "
            "changes before the auto-fix was applied."
        )

    allowed = set(allowed_paths)
    changed_files = get_changed_files()
    unexpected = [path for path in changed_files if path not in allowed]
    if unexpected:
        raise RuntimeError(
            "Refusing to commit: changed file(s) outside the allowed list: "
            f"{', '.join(unexpected)}"
        )
    if not changed_files:
        raise RuntimeError("Refusing to commit: no changes to commit.")

    # Stage only the authorized paths, never `git add -A`.
    subprocess.run(["git", "add", "--", *changed_files], check=True)

    # Only modifications to existing authorized files are allowed, so a
    # newly created, deleted, or renamed file aborts the commit.
    for status, path in _staged_status_entries():
        if path not in allowed:
            raise RuntimeError(
                "Refusing to commit: staged file outside the allowed list: "
                f"{path}"
            )
        if status != "M":
            raise RuntimeError(
                "Refusing to commit: only modifications to existing "
                f"authorized files are allowed (found {status} {path})."
            )

    subprocess.run(
        ["git", "-c", "user.email=ai-fixer@bot.local", "-c", "user.name=AI Auto-Fixer",
         "commit", "-m", "AI auto-fix: address code review comments"],
        check=True,
    )
    # Send the token to Git through environment configuration, not the
    # command line or URL, so it cannot appear in an exception or log.
    credentials = base64.b64encode(
        f"x-access-token:{token}".encode("utf-8")
    ).decode("ascii")
    push_env = os.environ.copy()
    push_env["GIT_TERMINAL_PROMPT"] = "0"
    # `actions/checkout` may have saved its own Authorization header in the
    # repository's Git config. An empty extraheader value resets those
    # inherited values, so only the header below is sent to GitHub.
    push_env["GIT_CONFIG_COUNT"] = "2"
    push_env["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"
    push_env["GIT_CONFIG_VALUE_0"] = ""
    push_env["GIT_CONFIG_KEY_1"] = "http.https://github.com/.extraheader"
    push_env["GIT_CONFIG_VALUE_1"] = f"Authorization: Basic {credentials}"

    try:
        result = subprocess.run(
            ["git", "push", f"https://github.com/{repo}.git", f"HEAD:{branch}"],
            capture_output=True,
            text=True,
            timeout=60,
            env=push_env,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "Push timed out -- the auto-fix was not pushed."
        ) from None
    if result.returncode != 0:
        raise RuntimeError(
            "Push failed -- this usually means the PR is from a fork, "
            "which this token can't write to."
        )


def post_comment(repo: str, pr_number: str, token: str, body: str):
    url = f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    requests.post(url, headers=headers, json={"body": body}, timeout=30)


def main():
    issues_text = os.environ.get("REVIEW_ISSUES", "").strip()
    repo = os.environ["GITHUB_REPOSITORY"]
    pr_number = os.environ["PR_NUMBER"]
    token = os.environ["GITHUB_TOKEN"]
    branch = os.environ["PR_HEAD_REF"]

    if not issues_text:
        print("No flagged bugs/security issues -- skipping auto-fix.")
        return

    allowed_paths = extract_allowed_paths(issues_text)
    if not allowed_paths:
        print("Could not determine which files to fix -- skipping auto-fix.")
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

    if not _verify_repository_checkout(repo):
        print(
            "Auto-fix skipped: the current directory is not the expected "
            "GitHub repository checkout."
        )
        return

    # Capture the PR state BEFORE any file is changed, so the check before
    # the push can prove the PR was not updated while the agent worked.
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

    try:
        initial_status = _git_status_entries()
    except RuntimeError as e:
        print(f"Could not verify a clean Git working tree: {e}")
        return
    if initial_status:
        print("Git working tree is not clean -- skipping auto-fix.")
        post_comment(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix skipped\n\n"
            "**Reason:** the Git working tree was not clean before the agent "
            "ran, so no files were changed.",
        )
        return

    print("Attempting auto-fix for the flagged review issues.")

    task = (
        "Fix these specific issues found in code review. Make only the "
        f"minimal changes needed, in the allowed file(s) only:\n\n{issues_text}"
    )

    agent = build_agent(allowed_paths)
    agent.invoke({"messages": [{"role": "user", "content": task}]})

    changed_files = get_changed_files()
    if not changed_files:
        print("Agent made no file changes.")
        post_comment(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix attempted\n\n"
            "The agent made no changes.",
        )
        return

    # SAFETY CHECK: verify it only touched files it was told to.
    unexpected = [f for f in changed_files if f not in allowed_paths]
    if unexpected:
        print(f"Agent touched unexpected files: {unexpected} -- discarding fix.")
        discard_changes_since(initial_status)
        post_comment(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix attempted but was discarded for safety\n\n"
            "The agent changed files outside the permitted scope. "
            "Those changes were discarded, and nothing was pushed.",
        )
        return

    # Re-fetch right before committing or pushing. This cannot make the
    # check and the push atomic, but it keeps the window small. The push
    # is non-force, so a branch that moves in the remaining window is
    # rejected by Git instead of being overwritten.
    try:
        validate_pull_request(
            repo, pr_number, branch, token, expected_head_sha
        )
    except ValidationError as e:
        post_comment(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix attempted but was not pushed\n\n"
            "The pull request received new commits while the auto-fix was "
            "running. "
            "The fix was not pushed.",
        )
        print(f"Pull request changed during auto-fix: {e}")
        return

    try:
        commit_and_push(branch, token, repo, allowed_paths)
        post_comment(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix pushed\n\n"
            "The proposed fix was pushed as a new commit on this PR branch. "
            "Please review it before merging.",
        )
        print("Auto-fix pushed successfully.")
    except Exception:
        post_comment(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix attempted but could not be pushed\n\n"
            "The auto-fix could not be committed or pushed. "
            "Please check the workflow logs.",
        )
        print("Could not commit or push the fix. Check the workflow logs.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Auto-fix step failed; details were withheld for safety.", file=sys.stderr)
        sys.exit(0)