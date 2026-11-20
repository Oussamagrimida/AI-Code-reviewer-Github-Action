"""Tests for fixer.py.

These tests exercise real local Git behaviour against temporary
repositories created with pytest's ``tmp_path``. No test may touch the
network, push to a remote, or operate on the repository the suite itself
is running in.
"""

import base64
import os
import stat
import subprocess
from pathlib import Path
import sys

import pytest

import fixer

_REAL_SUBPROCESS_RUN = subprocess.run
_PROTECT_TEST_PROCESS_ENVIRONMENT = fixer._protect_test_process_environment
_APPLY_AUTHORIZED_FILE = fixer._apply_authorized_file
_OPEN_REPOSITORY_FILE = fixer._open_repository_file


def _git(repo, *args, check=True):
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=check,
    )


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Make accidental GitHub API calls fail loudly."""
    def _blocked(*args, **kwargs):
        raise AssertionError("fixer tests must not perform network requests")

    monkeypatch.setattr(fixer.requests, "post", _blocked)
    monkeypatch.setattr(fixer.requests, "get", _blocked)


@pytest.fixture(autouse=True)
def block_git_push(monkeypatch):
    """Make any accidental ``git push`` fail loudly."""
    real_run = subprocess.run

    def guarded_run(command, *args, **kwargs):
        if "push" in command:
            raise AssertionError("fixer tests must not push to a remote")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded_run)


@pytest.fixture(autouse=True)
def adapt_security_guards_for_nonlinux_workflow_tests(monkeypatch):
    if sys.platform != "linux":
        monkeypatch.setattr(
            fixer, "_protect_test_process_environment", lambda: None
        )
    if os.name != "posix":
        def open_repository_file_for_workflow_tests(path, flags):
            source = Path.cwd().joinpath(*path.split("/"))
            file_fd = os.open(source, flags)
            return file_fd, os.fstat(file_fd)

        def apply_file_for_workflow_tests(path, content):
            destination = Path.cwd().joinpath(*path.split("/"))
            original = destination.read_bytes()
            if original == content:
                return False
            destination.write_bytes(content)
            return True

        monkeypatch.setattr(
            fixer, "_open_repository_file", open_repository_file_for_workflow_tests
        )
        monkeypatch.setattr(
            fixer, "_apply_authorized_file", apply_file_for_workflow_tests
        )


@pytest.fixture
def recorded_git(git_repo, monkeypatch):
    """Record git commands and stop at the commit/push boundary.

    Depends on ``git_repo`` so the repository's own setup commands are
    finished before recording starts.
    """
    real_run = subprocess.run
    commands = []

    def recording_run(command, *args, **kwargs):
        commands.append(list(command))
        if "commit" in command or "push" in command:
            raise AssertionError(f"unexpected git command: {command}")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)
    return git_repo, commands


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    """Create a temporary Git repo with one commit and make it the cwd."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test Author")
    _git(repo, "config", "core.autocrlf", "false")
    _git(repo, "remote", "add", "origin", "https://github.com/owner/repo.git")

    (repo / "tracked.txt").write_text("original\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "initial commit")

    monkeypatch.chdir(repo)
    monkeypatch.setenv("GITHUB_WORKSPACE", str(repo))
    return repo


def test_get_changed_files_detects_modified_tracked_file(git_repo):
    (git_repo / "tracked.txt").write_text("modified\n", encoding="utf-8")

    assert fixer.get_changed_files() == ["tracked.txt"]


def test_get_changed_files_detects_untracked_file(git_repo):
    (git_repo / "new.txt").write_text("new\n", encoding="utf-8")

    assert fixer.get_changed_files() == ["new.txt"]


def test_get_changed_files_detects_modified_and_untracked_files(git_repo):
    (git_repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
    (git_repo / "new.txt").write_text("new\n", encoding="utf-8")

    assert fixer.get_changed_files() == ["new.txt", "tracked.txt"]


def test_get_changed_files_detects_tracked_deletion(git_repo):
    (git_repo / "tracked.txt").unlink()

    assert fixer.get_changed_files() == ["tracked.txt"]


def test_get_changed_files_handles_filename_with_spaces(git_repo):
    (git_repo / "file with spaces.txt").write_text(
        "new\n", encoding="utf-8"
    )

    assert fixer.get_changed_files() == ["file with spaces.txt"]


def test_get_changed_files_detects_renamed_paths(git_repo):
    _git(git_repo, "mv", "tracked.txt", "renamed.txt")

    assert fixer.get_changed_files() == ["renamed.txt", "tracked.txt"]


def test_get_changed_files_raises_outside_git_repository(
    tmp_path, monkeypatch
):
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    monkeypatch.chdir(not_a_repo)

    with pytest.raises(RuntimeError, match="not a Git repository"):
        fixer.get_changed_files()


def test_discard_changes_since_preserves_pre_existing_modification(git_repo):
    tracked = git_repo / "tracked.txt"
    tracked.write_text("modified\n", encoding="utf-8")
    initial_status = fixer._git_status_entries()
    created = git_repo / "created by agent.txt"
    created.write_text("new file\n", encoding="utf-8")

    fixer.discard_changes_since(initial_status)

    assert tracked.read_text(encoding="utf-8") == "modified\n"
    assert not created.exists()


def test_discard_changes_since_preserves_pre_existing_untracked_file(git_repo):
    existing = git_repo / "pre-existing untracked.txt"
    existing.write_text("user file\n", encoding="utf-8")
    initial_status = fixer._git_status_entries()
    tracked = git_repo / "tracked.txt"
    tracked.write_text("changed by agent\n", encoding="utf-8")

    fixer.discard_changes_since(initial_status)

    assert existing.read_text(encoding="utf-8") == "user file\n"
    assert tracked.read_text(encoding="utf-8") == "original\n"


def test_discard_changes_since_removes_unexpected_new_file(git_repo):
    initial_status = fixer._git_status_entries()
    created = git_repo / "created by agent.txt"
    created.write_text("new file\n", encoding="utf-8")

    fixer.discard_changes_since(initial_status)

    assert not created.exists()


def test_discard_changes_since_reverts_unexpected_modification(git_repo):
    tracked = git_repo / "tracked with spaces.txt"
    tracked.write_text("original\n", encoding="utf-8")
    _git(git_repo, "add", "tracked with spaces.txt")
    _git(git_repo, "commit", "-m", "add file with spaces")
    initial_status = fixer._git_status_entries()
    tracked.write_text("changed by agent\n", encoding="utf-8")

    fixer.discard_changes_since(initial_status)

    assert tracked.read_text(encoding="utf-8") == "original\n"


def test_dirty_worktree_stops_before_agent_runs(git_repo, monkeypatch):
    token = "ghs_fake_secret_dirty_tree"
    comments = []
    _prepare_main_test(git_repo, monkeypatch, token, comments)
    tracked = git_repo / "tracked.txt"
    tracked.write_text("pre-existing edit\n", encoding="utf-8")
    agent_calls = []

    class UnexpectedAgent:
        def invoke(self, payload):
            agent_calls.append(payload)
            raise AssertionError("agent must not run on a dirty worktree")

    monkeypatch.setattr(
        fixer, "build_agent", lambda allowed: UnexpectedAgent()
    )

    fixer.main()

    assert agent_calls == []
    assert tracked.read_text(encoding="utf-8") == "pre-existing edit\n"
    assert comments
    assert any("working tree was not clean" in body for body in comments)


def test_change_outside_allowlist_prevents_commit_and_push(
    git_repo, monkeypatch
):
    monkeypatch.setenv("REVIEW_ISSUES", "- tracked.txt:1 [bug] broken")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("PR_NUMBER", "7")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("PR_HEAD_REF", "feature")
    monkeypatch.setenv("AUTOFIX_TEST_COMMAND", '["python", "-c", "pass"]')
    monkeypatch.setattr(fixer, "post_comment", lambda *args, **kwargs: None)
    monkeypatch.setattr(fixer, "post_comment", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        fixer,
        "get_pull_request",
        lambda *args: _pull_request(
            sha=_git(git_repo, "rev-parse", "HEAD").stdout.strip()
        ),
    )

    class FakeMessage:
        content = "summary"

    class FakeAgent:
        def __init__(self, root):
            self.root = root

        def invoke(self, payload):
            (Path(self.root) / "unauthorized.txt").write_text(
                "nope\n", encoding="utf-8"
            )
            return {"messages": [FakeMessage()]}

    monkeypatch.setattr(
        fixer, "build_agent", lambda allowed, root: FakeAgent(root)
    )

    commit_calls = []
    monkeypatch.setattr(
        fixer, "commit_and_push", lambda *args: commit_calls.append(args)
    )

    fixer.main()

    assert commit_calls == []
    assert not (git_repo / "unauthorized.txt").exists()


def test_unauthorized_new_file_blocks_staging_and_commit(recorded_git):
    repo, commands = recorded_git
    (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
    (repo / "unauthorized.txt").write_text("nope\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="outside the allowed list"):
        fixer.commit_and_push(
            "feature", "test-token", "owner/repo", ["tracked.txt"]
        )

    assert not any("add" in command for command in commands)
    assert not any("commit" in command for command in commands)
    assert not any("push" in command for command in commands)

    staged = _git(repo, "diff", "--cached", "--name-only")
    assert staged.stdout.strip() == ""


def test_tracked_deletion_blocks_commit_and_push(recorded_git):
    repo, commands = recorded_git
    (repo / "tracked.txt").unlink()

    with pytest.raises(RuntimeError, match="only modifications to existing"):
        fixer.commit_and_push(
            "feature", "test-token", "owner/repo", ["tracked.txt"]
        )

    assert not any("commit" in command for command in commands)
    assert not any("push" in command for command in commands)


def test_rename_blocks_commit_and_push(recorded_git):
    repo, commands = recorded_git
    (repo / "tracked.txt").rename(repo / "renamed.txt")

    with pytest.raises(RuntimeError, match="only modifications to existing"):
        fixer.commit_and_push(
            "feature",
            "test-token",
            "owner/repo",
            ["renamed.txt", "tracked.txt"],
        )

    assert not any("commit" in command for command in commands)
    assert not any("push" in command for command in commands)


def test_pre_existing_staged_changes_block_commit(recorded_git):
    repo, commands = recorded_git
    (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")

    with pytest.raises(RuntimeError, match="already contained staged changes"):
        fixer.commit_and_push(
            "feature", "test-token", "owner/repo", ["tracked.txt"]
        )

    assert not any("commit" in command for command in commands)
    assert not any("push" in command for command in commands)


def test_allowed_modification_is_committed_and_push_is_mocked(
    git_repo, monkeypatch
):
    (git_repo / "tracked.txt").write_text("modified\n", encoding="utf-8")

    real_run = subprocess.run
    commands = []

    def fake_run(command, *args, **kwargs):
        commands.append(list(command))
        if "push" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    fixer.commit_and_push(
        "feature", "test-token", "owner/repo", ["tracked.txt"]
    )

    assert any("add" in command for command in commands)
    assert any("commit" in command for command in commands)
    assert any("push" in command for command in commands)

    committed = _git(
        git_repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"
    )
    assert committed.stdout.splitlines() == ["tracked.txt"]


def _pull_request(
    base_repo="owner/repo",
    head_repo="owner/repo",
    branch="feature",
    sha="a" * 40,
):
    return {
        "base": {"repo": {"full_name": base_repo}},
        "head": {
            "repo": {"full_name": head_repo},
            "ref": branch,
            "sha": sha,
        },
    }


def _prepare_main_test(
    git_repo, monkeypatch, token, comments, summary="summary", agent_changes=True
):
    head_sha = _git(git_repo, "rev-parse", "HEAD").stdout.strip()
    monkeypatch.setenv("REVIEW_ISSUES", "- tracked.txt:1 [bug] broken")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("PR_NUMBER", "9")
    monkeypatch.setenv("GITHUB_TOKEN", token)
    monkeypatch.setenv("PR_HEAD_REF", "feature")
    monkeypatch.setenv("AUTOFIX_TEST_COMMAND", '["python", "-c", "pass"]')
    monkeypatch.setenv("AUTOFIX_MAX_ATTEMPTS", "2")
    monkeypatch.setattr(
        fixer,
        "post_comment",
        lambda repo, pr_number, tok, body: comments.append(body),
    )
    monkeypatch.setattr(
        fixer,
        "get_pull_request",
        lambda *args: _pull_request(sha=head_sha),
    )

    class FakeMessage:
        content = summary

    class FakeAgent:
        def __init__(self, root):
            self.root = root

        def invoke(self, payload):
            if agent_changes:
                (Path(self.root) / "tracked.txt").write_text(
                    "modified by agent\n", encoding="utf-8"
                )
            return {"messages": [FakeMessage()]}

    monkeypatch.setattr(
        fixer, "build_agent", lambda allowed, root: FakeAgent(root)
    )


@pytest.mark.parametrize(
    "failure_stage",
    [
        "agent invocation",
        "workspace application",
        "changed-file validation",
    ],
)
def test_agent_failures_log_sanitized_context_and_verify_rollback(
    git_repo,
    monkeypatch,
    capsys,
    failure_stage,
):
    token = "github_pat_for_test_abcdefghijklmnopqrstuvwxyz"
    api_key = "nvapi-for-test_key_abcdefghijkl"
    header_token = "authorization-header-secret"
    sensitive_url = (
        "https://user:password@example.test/api?api_key=url-secret"
    )
    failure_message = (
        f"request {sensitive_url} token={token} API_KEY={api_key}\n"
        f"Authorization: Bearer {header_token}\n"
        + "private detail " * 100
    )
    comments = []
    _prepare_main_test(git_repo, monkeypatch, token, comments)
    monkeypatch.setenv("NVIDIA_API_KEY", api_key)
    invocation_count = []

    class FailingAgent:
        def __init__(self, root):
            self.root = root

        def invoke(self, payload):
            invocation_count.append(payload)
            if failure_stage == "agent invocation":
                raise RuntimeError(failure_message)
            (Path(self.root) / "tracked.txt").write_text(
                "modified by agent\n", encoding="utf-8"
            )

    monkeypatch.setattr(
        fixer,
        "build_agent",
        lambda allowed, root: FailingAgent(root),
    )

    if failure_stage == "workspace application":
        def fail_workspace_application(agent_workspace):
            (git_repo / "tracked.txt").write_text(
                "partially applied\n", encoding="utf-8"
            )
            raise RuntimeError(failure_message)

        monkeypatch.setattr(
            fixer, "_apply_agent_workspace", fail_workspace_application
        )
    elif failure_stage == "changed-file validation":
        def fail_changed_file_validation(changed_files, allowed_paths):
            raise RuntimeError(failure_message)

        monkeypatch.setattr(
            fixer, "_verify_changes_authorized", fail_changed_file_validation
        )

    monkeypatch.setattr(
        fixer,
        "run_test_command",
        lambda *args, **kwargs: pytest.fail("Tests must not run"),
    )
    monkeypatch.setattr(
        fixer,
        "commit_and_push",
        lambda *args, **kwargs: pytest.fail("Changes must not be pushed"),
    )

    fixer.main()

    output = capsys.readouterr().out
    failure_log = next(
        line for line in output.splitlines()
        if line.startswith("Auto-fix failure")
    )
    assert f"[stage={failure_stage}]" in failure_log
    assert "[type=RuntimeError]" in failure_log
    assert "[redacted-url]" in failure_log
    assert "Authorization: [redacted]" in failure_log
    assert "private detail" in failure_log
    assert len(failure_log) < 550
    assert token not in output
    assert api_key not in output
    assert header_token not in output
    assert "url-secret" not in output
    assert len(invocation_count) == 1
    assert comments[-1].endswith(
        "The fixing agent failed. Its changes were rolled back; "
        "nothing was tested, committed, or pushed."
    )
    assert all(
        secret not in comments[-1]
        for secret in (token, api_key, header_token, "url-secret")
    )
    assert (git_repo / "tracked.txt").read_text(
        encoding="utf-8"
    ) == "original\n"
    assert fixer._git_status_entries() == []


def test_agent_receives_only_findings_for_allowed_paths(
    git_repo,
    monkeypatch,
):
    backend_paths = ["backend/app/api/routes/items.py"]
    for path in backend_paths:
        source = git_repo.joinpath(*path.split("/"))
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"original\n")
        _git(git_repo, "add", "--", path)
    _git(git_repo, "commit", "-m", "Add backend routes")

    comments = []
    _prepare_main_test(git_repo, monkeypatch, "test-token", comments)
    review_issues = "\n".join(
        [
            "- backend/app/api/routes/items.py:87 [correctness] Update fields safely.",
            "Impact: the update could otherwise overwrite existing data.",
            "Remediation: validate the new value before saving.",
            "- .github/workflows/ai-reviewer.yml:27 [security] Protect workflow.",
            "Impact: this workflow finding must not reach the fixing agent.",
            "Remediation: do not modify any workflow file.",
        ]
    )
    authorized_finding = "\n".join(
        [
            "- backend/app/api/routes/items.py:87 [correctness] Update fields safely.",
            "Impact: the update could otherwise overwrite existing data.",
            "Remediation: validate the new value before saving.",
        ]
    )
    monkeypatch.setenv("REVIEW_ISSUES", review_issues)
    authorized_paths = []
    agent_tasks = []

    class CapturingAgent:
        def __init__(self, root):
            self.root = root

        def invoke(self, payload):
            agent_tasks.append(payload["messages"][0]["content"])

    def build_agent(allowed_paths, root):
        authorized_paths.extend(allowed_paths)
        return CapturingAgent(root)

    monkeypatch.setattr(fixer, "build_agent", build_agent)
    monkeypatch.setattr(
        fixer,
        "run_test_command",
        lambda *args, **kwargs: pytest.fail("Tests must not run"),
    )

    fixer.main()

    assert authorized_paths == backend_paths
    assert len(agent_tasks) == 1
    assert authorized_finding in agent_tasks[0]
    assert ".github/workflows/ai-reviewer.yml" not in agent_tasks[0]
    assert "this workflow finding must not reach the fixing agent" not in agent_tasks[0]
    assert "do not modify any workflow file" not in agent_tasks[0]
    assert os.environ["REVIEW_ISSUES"] == review_issues
    assert fixer._git_status_entries() == []


def test_agent_summary_is_not_published_on_success(git_repo, monkeypatch):
    secret = "simulated-agent-summary-secret"
    comments = []
    _prepare_main_test(
        git_repo, monkeypatch, "test-token", comments, summary=secret
    )
    monkeypatch.setattr(fixer, "commit_and_push", lambda *args: None)

    fixer.main()

    assert comments
    assert all(secret not in body for body in comments)


def test_agent_summary_is_not_published_when_no_changes(git_repo, monkeypatch):
    secret = "simulated-agent-summary-secret"
    comments = []
    _prepare_main_test(
        git_repo,
        monkeypatch,
        "test-token",
        comments,
        summary=secret,
        agent_changes=False,
    )

    fixer.main()

    assert comments
    assert all(secret not in body for body in comments)


def test_agent_summary_is_not_published_when_changes_are_discarded(
    git_repo, monkeypatch
):
    secret = "simulated-agent-summary-secret"
    comments = []
    _prepare_main_test(
        git_repo, monkeypatch, "test-token", comments, summary=secret
    )

    class UnauthorizedAgent:
        def __init__(self, root):
            self.root = root

        def invoke(self, payload):
            (Path(self.root) / "unauthorized.txt").write_text(
                "unexpected\n", encoding="utf-8"
            )

            class Message:
                content = secret

            return {"messages": [Message()]}

    monkeypatch.setattr(
        fixer, "build_agent", lambda allowed, root: UnauthorizedAgent(root)
    )

    fixer.main()

    assert comments
    assert all(secret not in body for body in comments)
    assert not (git_repo / "unauthorized.txt").exists()


def test_agent_summary_is_not_published_when_push_fails(git_repo, monkeypatch):
    secret = "simulated-agent-summary-secret"
    comments = []
    _prepare_main_test(
        git_repo, monkeypatch, "test-token", comments, summary=secret
    )
    monkeypatch.setattr(
        fixer,
        "commit_and_push",
        lambda *args: (_ for _ in ()).throw(RuntimeError("private failure")),
    )

    fixer.main()

    assert comments
    assert all(secret not in body for body in comments)


def test_main_reports_non_git_working_directory_before_agent(
    tmp_path, monkeypatch
):
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    monkeypatch.chdir(not_a_repo)
    monkeypatch.setenv("GITHUB_WORKSPACE", str(not_a_repo))
    monkeypatch.setenv("REVIEW_ISSUES", "- tracked.txt:1 [bug] private issue")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("PR_NUMBER", "9")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("PR_HEAD_REF", "feature")
    monkeypatch.setenv("AUTOFIX_TEST_COMMAND", '["python", "-c", "pass"]')

    comments = []
    monkeypatch.setattr(
        fixer,
        "get_pull_request",
        lambda *args: _pull_request(sha="a" * 40),
    )
    monkeypatch.setattr(
        fixer,
        "post_comment",
        lambda repo, pr, token, body: comments.append(body),
    )
    monkeypatch.setattr(
        fixer, "build_agent",
        lambda *args, **kwargs: pytest.fail("agent must not run"),
    )

    fixer.main()

    assert any("checkout does not match" in body for body in comments)


def test_main_skips_checkout_for_different_repository(git_repo, monkeypatch):
    comments = []
    _prepare_main_test(git_repo, monkeypatch, "test-token", comments)
    monkeypatch.setenv("GITHUB_REPOSITORY", "other/repo")
    monkeypatch.setattr(
        fixer,
        "get_pull_request",
        lambda *args: _pull_request(
            base_repo="other/repo",
            head_repo="other/repo",
            sha=_git(git_repo, "rev-parse", "HEAD").stdout.strip(),
        ),
    )

    fixer.main()

    assert any("checkout does not match" in body for body in comments)

def test_checkout_url_must_target_github(git_repo):
    _git(
        git_repo,
        "remote",
        "set-url",
        "origin",
        "https://github.com.evil.example/owner/repo.git",
    )

    assert fixer._verify_repository_checkout("owner/repo") is False


def test_main_skips_when_working_directory_is_not_workspace(
    git_repo, monkeypatch
):
    comments = []
    _prepare_main_test(git_repo, monkeypatch, "test-token", comments)
    monkeypatch.setenv("GITHUB_WORKSPACE", str(git_repo.parent))

    fixer.main()

    assert any("checkout does not match" in body for body in comments)

def test_push_timeout_does_not_leak_token_into_comment(
    git_repo, monkeypatch
):
    token = "ghs_fake_secret_timeout"
    comments = []
    _prepare_main_test(git_repo, monkeypatch, token, comments)

    real_run = subprocess.run

    def fake_run(command, *args, **kwargs):
        if "push" in command:
            raise subprocess.TimeoutExpired(
                cmd=[
                    "git",
                    "push",
                    f"https://x-access-token:{token}@github.com/owner/repo.git",
                ],
                timeout=60,
            )
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    fixer.main()

    assert comments
    assert any(
        "push outcome unknown" in body
        and "inspect the PR branch before retrying" in body
        for body in comments
    )
    assert all(token not in body for body in comments)


@pytest.mark.parametrize(
    "push_error",
    [subprocess.TimeoutExpired, ConnectionError, OSError],
    ids=["timeout", "connection", "subprocess-os-error"],
)
def test_push_exceptions_are_uncertain_and_roll_back_locally(
    git_repo, monkeypatch, push_error
):
    base_sha = fixer._git_head_sha()
    initial_state = fixer._working_tree_state()
    (git_repo / "tracked.txt").write_text("validated fix\n", encoding="utf-8")
    monkeypatch.setattr(
        fixer, "validate_pull_request", lambda *args: base_sha
    )
    real_run = subprocess.run

    def fail_push(command, *args, **kwargs):
        if "push" in command:
            if push_error is subprocess.TimeoutExpired:
                raise push_error(command, timeout=60)
            raise push_error("connection lost")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail_push)

    with pytest.raises(fixer.PushFailure) as failure:
        fixer.commit_and_push(
            "feature",
            "test-token",
            "owner/repo",
            ["tracked.txt"],
            pr_number="9",
            expected_head_sha=base_sha,
            expected_state=initial_state,
        )

    assert failure.value.outcome_uncertain is True
    assert failure.value.rollback_verified is True
    assert fixer._git_head_sha() == base_sha
    assert fixer._git_status_entries() == []
    assert fixer._staged_status_entries() == []
    assert (git_repo / "tracked.txt").read_text(encoding="utf-8") == "original\n"


def test_nonzero_push_after_remote_acceptance_is_uncertain_and_rolls_back(
    git_repo, tmp_path, monkeypatch
):
    remote = tmp_path / "accepted-remote.git"
    _REAL_SUBPROCESS_RUN(
        ["git", "init", "--bare", str(remote)],
        capture_output=True,
        text=True,
        check=True,
    )
    base_sha = fixer._git_head_sha()
    initial_state = fixer._working_tree_state()
    (git_repo / "tracked.txt").write_text("validated fix\n", encoding="utf-8")
    monkeypatch.setattr(
        fixer, "validate_pull_request", lambda *args: base_sha
    )
    guarded_run = subprocess.run
    captured = {}

    def accepted_then_disconnected(command, *args, **kwargs):
        if "push" in command:
            local_env = kwargs["env"].copy()
            local_env["GIT_CONFIG_COUNT"] = "4"
            local_env["GIT_CONFIG_KEY_3"] = "safe.bareRepository"
            local_env["GIT_CONFIG_VALUE_3"] = "all"
            accepted = _REAL_SUBPROCESS_RUN(
                ["git", "push", str(remote), command[-1]],
                cwd=git_repo,
                env=local_env,
                capture_output=True,
                text=True,
                check=True,
            )
            captured["remote_output"] = accepted.stdout
            captured["remote_sha"] = (
                remote / "refs" / "heads" / "feature"
            ).read_text(encoding="ascii").strip()
            return subprocess.CompletedProcess(
                command, 1, accepted.stdout, "connection closed after acceptance"
            )
        return guarded_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", accepted_then_disconnected)

    with pytest.raises(fixer.PushFailure) as failure:
        fixer.commit_and_push(
            "feature",
            "test-token",
            "owner/repo",
            ["tracked.txt"],
            pr_number="9",
            expected_head_sha=base_sha,
            expected_state=initial_state,
        )

    assert failure.value.outcome_uncertain is True
    assert failure.value.rollback_verified is True
    assert captured["remote_sha"] != base_sha
    assert fixer._git_head_sha() == base_sha
    assert fixer._git_status_entries() == []
    assert fixer._staged_status_entries() == []


def test_confirmed_successful_push_updates_only_the_requested_branch(
    git_repo, tmp_path, monkeypatch
):
    remote = tmp_path / "successful-remote.git"
    _REAL_SUBPROCESS_RUN(
        ["git", "init", "--bare", str(remote)],
        capture_output=True,
        text=True,
        check=True,
    )
    base_sha = fixer._git_head_sha()
    (git_repo / "tracked.txt").write_text("validated fix\n", encoding="utf-8")
    monkeypatch.setattr(
        fixer, "validate_pull_request", lambda *args: base_sha
    )
    guarded_run = subprocess.run
    captured_commands = []

    def accept_local_push(command, *args, **kwargs):
        if "push" in command:
            captured_commands.append(list(command))
            local_env = kwargs["env"].copy()
            local_env["GIT_CONFIG_COUNT"] = "4"
            local_env["GIT_CONFIG_KEY_3"] = "safe.bareRepository"
            local_env["GIT_CONFIG_VALUE_3"] = "all"
            return _REAL_SUBPROCESS_RUN(
                ["git", "push", str(remote), command[-1]],
                cwd=git_repo,
                env=local_env,
                capture_output=True,
                text=True,
                check=True,
            )
        return guarded_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", accept_local_push)

    fixer.commit_and_push(
        "feature",
        "test-token",
        "owner/repo",
        ["tracked.txt"],
        pr_number="9",
        expected_head_sha=base_sha,
    )

    remote_sha = (
        remote / "refs" / "heads" / "feature"
    ).read_text(encoding="ascii").strip()
    assert remote_sha == fixer._git_head_sha()
    assert remote_sha != base_sha
    assert all("--force" not in command for command in captured_commands)
    assert fixer._git_status_entries() == []


def test_mismatched_head_branch_blocks_push(git_repo, monkeypatch):
    comments = []
    _prepare_main_test(git_repo, monkeypatch, "test-token", comments)
    monkeypatch.setattr(
        fixer,
        "get_pull_request",
        lambda *args: _pull_request(
            branch="different-branch",
            sha=_git(git_repo, "rev-parse", "HEAD").stdout.strip(),
        ),
    )
    commits = []
    monkeypatch.setattr(
        fixer,
        "commit_and_push",
        lambda *args, **kwargs: commits.append(args),
    )

    fixer.main()

    assert commits == []
    assert any("branch no longer matches" in body for body in comments)


def test_push_failure_output_does_not_leak_token_into_comment(
    git_repo, monkeypatch
):
    token = "ghs_fake_secret_failure"
    comments = []
    _prepare_main_test(git_repo, monkeypatch, token, comments)

    real_run = subprocess.run

    def fake_run(command, *args, **kwargs):
        if "push" in command:
            return subprocess.CompletedProcess(
                command, 1, "", f"remote rejected: {token}"
            )
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    fixer.main()

    assert comments
    assert any("push outcome unknown" in body for body in comments)
    assert any("inspect the PR branch before retrying" in body for body in comments)
    assert all(token not in body for body in comments)


def test_push_uses_token_via_environment_not_argv(git_repo, monkeypatch):
    token = "ghs_fake_secret_argv"
    (git_repo / "tracked.txt").write_text("modified\n", encoding="utf-8")

    captured = {}
    real_run = subprocess.run

    def fake_run(command, *args, **kwargs):
        if "push" in command:
            captured["command"] = list(command)
            captured["env"] = kwargs.get("env", {})
            return subprocess.CompletedProcess(command, 0, "", "")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    fixer.commit_and_push(
        "feature", token, "owner/repo", ["tracked.txt"]
    )

    assert "push" in captured["command"]
    assert all(token not in part for part in captured["command"])

    credentials = base64.b64encode(
        f"x-access-token:{token}".encode("utf-8")
    ).decode("ascii")
    env = captured["env"]
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_CONFIG_COUNT"] == "3"
    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    assert env["GIT_CONFIG_VALUE_0"] == ""
    assert env["GIT_CONFIG_KEY_1"] == "http.https://github.com/.extraheader"
    assert env["GIT_CONFIG_VALUE_1"] == f"Authorization: Basic {credentials}"
    assert env["GIT_CONFIG_KEY_2"] == "core.hooksPath"
    assert env["GIT_CONFIG_VALUE_2"]


def test_push_resets_header_persisted_by_checkout(git_repo, monkeypatch):
    token = "ghs_fake_secret_checkout"
    checkout_header = "Authorization: Basic b2xkZXItY3JlZGVudGlhbA=="
    _git(
        git_repo,
        "config",
        "--local",
        "http.https://github.com/.extraheader",
        checkout_header,
    )
    (git_repo / "tracked.txt").write_text("modified\n", encoding="utf-8")

    captured = {}
    real_run = subprocess.run

    def fake_run(command, *args, **kwargs):
        if "push" in command:
            captured["command"] = list(command)
            captured["env"] = kwargs.get("env", {})
            return subprocess.CompletedProcess(command, 0, "", "")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)

    fixer.commit_and_push(
        "feature", token, "owner/repo", ["tracked.txt"]
    )

    env = captured["env"]
    credentials = base64.b64encode(
        f"x-access-token:{token}".encode("utf-8")
    ).decode("ascii")
    # The empty value comes first so Git drops any header inherited from
    # the checkout before adding the intended one.
    assert env["GIT_CONFIG_COUNT"] == "3"
    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    assert env["GIT_CONFIG_VALUE_0"] == ""
    assert env["GIT_CONFIG_KEY_1"] == "http.https://github.com/.extraheader"
    assert env["GIT_CONFIG_VALUE_1"] == f"Authorization: Basic {credentials}"
    assert env["GIT_CONFIG_KEY_2"] == "core.hooksPath"
    assert env["GIT_CONFIG_VALUE_2"]
    assert all(token not in part for part in captured["command"])

    # The repository's own configuration is left exactly as it was.
    persisted = _git(
        git_repo,
        "config",
        "--local",
        "--get-all",
        "http.https://github.com/.extraheader",
    )
    assert persisted.stdout.splitlines() == [checkout_header]


def test_checkout_auth_header_can_be_removed_before_untrusted_tests(git_repo):
    header = "Authorization: Basic c2VjcmV0"
    _git(
        git_repo,
        "config",
        "--local",
        "http.https://github.com/.extraheader",
        header,
    )

    fixer._remove_checkout_auth_header()

    result = _git(
        git_repo,
        "config",
        "--local",
        "--get-all",
        "http.https://github.com/.extraheader",
        check=False,
    )
    assert result.returncode != 0


def test_mismatched_base_repository_blocks_push(git_repo, monkeypatch):
    token = "ghs_fake_secret_base"
    comments = []
    _prepare_main_test(git_repo, monkeypatch, token, comments)
    monkeypatch.setattr(
        fixer,
        "get_pull_request",
        lambda *args: _pull_request(base_repo="someone/else"),
    )
    commit_calls = []
    monkeypatch.setattr(
        fixer, "commit_and_push", lambda *args: commit_calls.append(args)
    )

    fixer.main()

    assert commit_calls == []
    assert comments
    assert any("base repository" in body for body in comments)
    assert all(token not in body for body in comments)


def test_fork_pull_request_blocks_push(git_repo, monkeypatch):
    token = "ghs_fake_secret_fork"
    comments = []
    _prepare_main_test(git_repo, monkeypatch, token, comments)
    monkeypatch.setattr(
        fixer,
        "get_pull_request",
        lambda *args: _pull_request(head_repo="contributor/repo"),
    )
    commit_calls = []
    monkeypatch.setattr(
        fixer, "commit_and_push", lambda *args: commit_calls.append(args)
    )

    fixer.main()

    assert commit_calls == []
    assert any("fork" in body for body in comments)
    assert all(token not in body for body in comments)


def test_changed_head_sha_blocks_push(git_repo, monkeypatch):
    token = "ghs_fake_secret_sha"
    comments = []
    _prepare_main_test(git_repo, monkeypatch, token, comments)
    responses = [
        _pull_request(sha=_git(git_repo, "rev-parse", "HEAD").stdout.strip()),
        _pull_request(sha="b" * 40),
    ]
    monkeypatch.setattr(
        fixer, "get_pull_request", lambda *args: responses.pop(0)
    )
    commit_calls = []
    monkeypatch.setattr(
        fixer, "commit_and_push", lambda *args: commit_calls.append(args)
    )

    fixer.main()

    assert commit_calls == []
    assert any("new commits" in body for body in comments)
    assert all(token not in body for body in comments)


def test_invalid_repository_skips_before_api_or_git(git_repo, monkeypatch):
    token = "ghs_fake_secret_repo"
    comments = []
    _prepare_main_test(git_repo, monkeypatch, token, comments)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo/extra")

    def unexpected(*args, **kwargs):
        raise AssertionError("API must not be called for an invalid repository")

    monkeypatch.setattr(fixer, "get_pull_request", unexpected)
    commit_calls = []
    monkeypatch.setattr(
        fixer, "commit_and_push", lambda *args: commit_calls.append(args)
    )

    fixer.main()

    assert commit_calls == []
    assert comments == []


@pytest.mark.parametrize(
    "branch", ["feature..evil", "-feature", "feature name"]
)
def test_invalid_branch_blocks_push(git_repo, monkeypatch, branch):
    token = "ghs_fake_secret_branch"
    comments = []
    _prepare_main_test(git_repo, monkeypatch, token, comments)
    monkeypatch.setenv("PR_HEAD_REF", branch)

    def unexpected(*args, **kwargs):
        raise AssertionError("API must not be called for an invalid branch")

    monkeypatch.setattr(fixer, "get_pull_request", unexpected)
    commit_calls = []
    monkeypatch.setattr(
        fixer, "commit_and_push", lambda *args: commit_calls.append(args)
    )

    fixer.main()

    assert commit_calls == []
    assert any("branch name is not valid" in body for body in comments)
    assert all(token not in body for body in comments)


def test_get_pull_request_hides_api_error_details(monkeypatch):
    token = "ghs_fake_secret_api"

    def failing_get(*args, **kwargs):
        raise fixer.requests.RequestException(f"raw server text {token}")

    monkeypatch.setattr(fixer.requests, "get", failing_get)

    with pytest.raises(fixer.ValidationError) as error:
        fixer.get_pull_request("owner/repo", "9", token)

    message = str(error.value)
    assert "could not be verified" in message
    assert token not in message
    assert "raw server text" not in message


def test_get_pull_request_hides_error_response(monkeypatch):
    token = "ghs_fake_secret_status"

    class FakeResponse:
        status_code = 500

        def json(self):
            return {"message": f"raw server text {token}"}

    monkeypatch.setattr(
        fixer.requests, "get", lambda *args, **kwargs: FakeResponse()
    )

    with pytest.raises(fixer.ValidationError) as error:
        fixer.get_pull_request("owner/repo", "9", token)

    message = str(error.value)
    assert "could not be verified" in message
    assert token not in message
    assert "raw server text" not in message


def test_extract_allowed_paths_supports_spaces_and_rejects_metadata(git_repo):
    assert fixer.extract_allowed_paths(
        "- source file.py:12 [correctness] repair this"
    ) == ["source file.py"]
    assert fixer.extract_allowed_paths(
        "- .git/config:1 [security] unsafe\n"
        "- ../outside.py:2 [security] unsafe\n"
        "- .github/workflows/build.yml:3 [security] unsafe"
    ) == []


def test_agent_workspace_contains_only_authorized_files(git_repo, tmp_path):
    (git_repo / "private.txt").write_text("private\n", encoding="utf-8")
    sandbox = tmp_path / "agent"

    workspace = fixer._create_agent_workspace(["tracked.txt"], str(sandbox))

    assert (sandbox / "tracked.txt").read_text(encoding="utf-8") == "original\n"
    assert not (sandbox / "private.txt").exists()
    assert not (sandbox / ".git").exists()
    assert workspace["allowed"] == {"tracked.txt"}


def test_agent_workspace_rejects_preexisting_contents(git_repo, tmp_path):
    workspace_root = tmp_path / "agent"
    workspace_root.mkdir()
    (workspace_root / "stale-secret.txt").write_text(
        "must not reach the agent\n", encoding="utf-8"
    )

    with pytest.raises(fixer.ValidationError, match="not empty"):
        fixer._create_agent_workspace(["tracked.txt"], str(workspace_root))


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX no-follow handles")
def test_agent_workspace_rejects_source_symlink_even_after_allowlist_check(
    git_repo, tmp_path, monkeypatch
):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside secret\n", encoding="utf-8")
    tracked = git_repo / "tracked.txt"
    tracked.unlink()
    tracked.symlink_to(outside)
    monkeypatch.setattr(fixer, "_is_authorized_path", lambda path: True)

    with pytest.raises(fixer.ValidationError, match="safely opened"):
        fixer._create_agent_workspace(["tracked.txt"], str(tmp_path / "agent"))

    assert not (tmp_path / "agent" / "tracked.txt").exists()
    assert outside.read_text(encoding="utf-8") == "outside secret\n"


def test_agent_workspace_creation_fails_closed_without_secure_path_apis(
    git_repo, tmp_path, monkeypatch
):
    if os.name == "posix":
        pytest.skip("This test covers unsupported platforms")
    monkeypatch.setattr(fixer, "_open_repository_file", _OPEN_REPOSITORY_FILE)

    with pytest.raises(fixer.ValidationError, match="supported POSIX systems"):
        fixer._create_agent_workspace(["tracked.txt"], str(tmp_path / "agent"))


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX no-follow handles")
def test_agent_changes_reject_destination_symlinks(git_repo, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    workspace = fixer._create_agent_workspace(
        ["tracked.txt"], str(tmp_path / "agent")
    )
    (tmp_path / "agent" / "tracked.txt").write_text("agent edit\n", encoding="utf-8")
    (git_repo / "tracked.txt").unlink()
    (git_repo / "tracked.txt").symlink_to(outside)

    with pytest.raises(fixer.ValidationError, match="safely updated"):
        fixer._apply_agent_workspace(workspace)

    assert outside.read_text(encoding="utf-8") == "outside\n"


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX no-follow handles")
def test_agent_changes_reject_symlinked_destination_directories(
    git_repo, tmp_path
):
    nested = git_repo / "src"
    nested.mkdir()
    (nested / "tracked.txt").write_text("original\n", encoding="utf-8")
    _git(git_repo, "add", "src/tracked.txt")
    _git(git_repo, "commit", "-m", "add nested file")
    workspace = fixer._create_agent_workspace(
        ["src/tracked.txt"], str(tmp_path / "agent")
    )
    (tmp_path / "agent" / "src" / "tracked.txt").write_text(
        "agent edit\n", encoding="utf-8"
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "tracked.txt").write_text("outside\n", encoding="utf-8")
    (nested / "tracked.txt").unlink()
    nested.rmdir()
    nested.symlink_to(outside, target_is_directory=True)

    with pytest.raises(fixer.ValidationError, match="safely opened"):
        fixer._apply_agent_workspace(workspace)

    assert (outside / "tracked.txt").read_text(encoding="utf-8") == "outside\n"


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX no-follow handles")
def test_agent_change_replaces_raced_destination_symlink_without_following_it(
    git_repo, tmp_path, monkeypatch
):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")
    workspace = fixer._create_agent_workspace(
        ["tracked.txt"], str(tmp_path / "agent")
    )
    (tmp_path / "agent" / "tracked.txt").write_text("agent edit\n", encoding="utf-8")
    real_replace = os.replace

    def swap_then_replace(source, destination, **kwargs):
        os.unlink(destination, dir_fd=kwargs["dst_dir_fd"])
        os.symlink(str(outside), destination, dir_fd=kwargs["dst_dir_fd"])
        return real_replace(source, destination, **kwargs)

    monkeypatch.setattr(fixer.os, "replace", swap_then_replace)

    assert fixer._apply_agent_workspace(workspace) == ["tracked.txt"]

    assert outside.read_text(encoding="utf-8") == "outside\n"
    assert (git_repo / "tracked.txt").read_text(encoding="utf-8") == "agent edit\n"
    assert not (git_repo / "tracked.txt").is_symlink()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX no-follow handles")
def test_agent_change_preserves_mode_and_does_not_modify_hardlink_target(
    git_repo, tmp_path
):
    outside = tmp_path / "outside.txt"
    outside.write_text("original\n", encoding="utf-8")
    tracked = git_repo / "tracked.txt"
    tracked.unlink()
    os.link(outside, tracked)
    os.chmod(tracked, 0o640)
    workspace = fixer._create_agent_workspace(
        ["tracked.txt"], str(tmp_path / "agent")
    )
    (tmp_path / "agent" / "tracked.txt").write_text("agent edit\n", encoding="utf-8")

    assert fixer._apply_agent_workspace(workspace) == ["tracked.txt"]

    assert outside.read_text(encoding="utf-8") == "original\n"
    assert tracked.read_text(encoding="utf-8") == "agent edit\n"
    assert stat.S_IMODE(tracked.stat().st_mode) == 0o640


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX no-follow handles")
@pytest.mark.parametrize("destination_kind", ["read-only", "directory"])
def test_agent_change_rejects_unwritable_or_nonfile_destinations(
    git_repo, tmp_path, destination_kind
):
    tracked = git_repo / "tracked.txt"
    workspace = fixer._create_agent_workspace(
        ["tracked.txt"], str(tmp_path / "agent")
    )
    (tmp_path / "agent" / "tracked.txt").write_text("agent edit\n", encoding="utf-8")
    if destination_kind == "read-only":
        os.chmod(tracked, 0o444)
    else:
        tracked.unlink()
        tracked.mkdir()

    with pytest.raises(fixer.ValidationError):
        fixer._apply_agent_workspace(workspace)

    if destination_kind == "read-only":
        assert tracked.read_text(encoding="utf-8") == "original\n"
    else:
        assert tracked.is_dir()


def test_agent_change_application_fails_closed_without_secure_path_apis(
    git_repo, tmp_path, monkeypatch
):
    if os.name == "posix":
        pytest.skip("This test covers unsupported platforms")
    monkeypatch.setattr(fixer, "_apply_authorized_file", _APPLY_AUTHORIZED_FILE)
    workspace = fixer._create_agent_workspace(
        ["tracked.txt"], str(tmp_path / "agent")
    )
    (tmp_path / "agent" / "tracked.txt").write_text("agent edit\n", encoding="utf-8")

    with pytest.raises(fixer.ValidationError, match="supported POSIX systems"):
        fixer._apply_agent_workspace(workspace)

    assert (git_repo / "tracked.txt").read_text(encoding="utf-8") == "original\n"


def test_invalid_max_attempt_values_are_rejected():
    assert fixer._max_fix_attempts("1") == 1
    assert fixer._max_fix_attempts("5") == 5
    for value in ("0", "6", "1.5", "unbounded"):
        with pytest.raises(fixer.ValidationError):
            fixer._max_fix_attempts(value)


def test_test_command_requires_json_argv_and_never_uses_shell():
    assert fixer.parse_test_command('["python", "-m", "pytest", "-q"]') == [
        "python", "-m", "pytest", "-q"
    ]

    with pytest.raises(fixer.ValidationError, match="JSON array"):
        fixer.parse_test_command("python -m pytest")
    with pytest.raises(fixer.ValidationError, match="JSON array"):
        fixer.parse_test_command('["python", ""]')


def test_test_command_captures_exit_code_and_bounded_output():
    result = fixer.run_test_command(
        [
            sys.executable, "-c",
            "import sys; print('test output'); sys.exit(7)",
        ]
    )

    assert result["passed"] is False
    assert result["exit_code"] == 7
    assert result["timed_out"] is False
    assert "test output" in result["output"]

    large_output = fixer.run_test_command(
        [
            sys.executable, "-c",
            "print('x' * 12000)",
        ]
    )
    assert len(large_output["output"].encode("utf-8")) <= 6000


def test_test_command_timeout_is_reported():
    result = fixer.run_test_command(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        timeout_seconds=0.1,
    )

    assert result["passed"] is False
    assert result["timed_out"] is True
    assert result["exit_code"] is None


def test_test_command_does_not_inherit_credentials(monkeypatch):
    token = "test-token-secret"
    api_key = "nvidia-api-secret"
    monkeypatch.setenv("GITHUB_TOKEN", token)
    monkeypatch.setenv("NVIDIA_API_KEY", api_key)
    command = [
        sys.executable, "-c",
        "import os; print(os.getenv('GITHUB_TOKEN')); "
        "print(os.getenv('NVIDIA_API_KEY'))",
    ]

    result = fixer.run_test_command(command)

    assert result["passed"] is True
    assert token not in result["output"]
    assert api_key not in result["output"]


def test_test_command_cannot_read_parent_process_environment(
    git_repo, monkeypatch
):
    if sys.platform != "linux":
        pytest.skip("Linux /proc access controls are required")
    if os.geteuid() == 0:
        pytest.skip("root can bypass the process inspection restriction")

    secret = "parent-environment-canary"
    monkeypatch.setenv("PARENT_ENVIRONMENT_CANARY", secret)
    command = [
        sys.executable,
        "-c",
        "import os\n"
        "path = '/proc/%s/environ' % os.getppid()\n"
        "try:\n"
        "    open(path, 'rb').read()\n"
        "except PermissionError:\n"
        "    try_read = 'blocked'\n"
        "else:\n"
        "    try_read = 'readable'\n"
        "print(try_read)",
    ]

    result = fixer.run_test_command(command)

    assert result["passed"] is True
    assert "blocked" in result["output"]
    assert "readable" not in result["output"]
    assert secret not in result["output"]


def test_test_process_protection_fails_closed_off_linux(monkeypatch):
    if sys.platform == "linux":
        pytest.skip("This test covers unsupported platforms")
    monkeypatch.setattr(fixer.sys, "platform", "win32")

    with pytest.raises(RuntimeError, match="require Linux"):
        _PROTECT_TEST_PROCESS_ENVIRONMENT()


def test_main_retries_failed_fix_and_pushes_only_after_success(
    git_repo, monkeypatch
):
    comments = []
    _prepare_main_test(git_repo, monkeypatch, "test-token", comments)
    monkeypatch.setenv("AUTOFIX_MAX_ATTEMPTS", "2")
    outcomes = iter([
        {
            "passed": False,
            "exit_code": 1,
            "timed_out": False,
            "output": "first attempt failed",
        },
        {
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "output": "all good",
        },
    ])
    test_calls = []
    monkeypatch.setattr(
        fixer,
        "run_test_command",
        lambda command: (test_calls.append(command), next(outcomes))[1],
    )

    class RetryAgent:
        def __init__(self, root):
            self.root = root
            self.payloads = []

        def invoke(self, payload):
            self.payloads.append(payload)
            (Path(self.root) / "tracked.txt").write_text(
                "fixed\n", encoding="utf-8"
            )

    agent = RetryAgent("")

    def build_agent(paths, root):
        agent.root = root
        return agent

    monkeypatch.setattr(fixer, "build_agent", build_agent)
    commits = []
    monkeypatch.setattr(
        fixer,
        "commit_and_push",
        lambda *args, **kwargs: commits.append((args, kwargs)),
    )

    fixer.main()

    assert len(agent.payloads) == 2
    assert "first attempt failed" in agent.payloads[1]["messages"][0]["content"]
    assert len(test_calls) == 2
    assert len(commits) == 1
    assert "passed the configured tests" in comments[-1]


def test_main_rolls_back_after_all_attempts_fail(git_repo, monkeypatch):
    comments = []
    _prepare_main_test(git_repo, monkeypatch, "test-token", comments)
    monkeypatch.setenv("AUTOFIX_MAX_ATTEMPTS", "2")
    monkeypatch.setattr(
        fixer,
        "run_test_command",
        lambda command: {
            "passed": False,
            "exit_code": 3,
            "timed_out": False,
            "output": "failure",
        },
    )
    commits = []
    monkeypatch.setattr(
        fixer,
        "commit_and_push",
        lambda *args, **kwargs: commits.append(args),
    )

    fixer.main()

    assert commits == []
    assert (git_repo / "tracked.txt").read_text(encoding="utf-8") == "original\n"
    assert fixer._git_status_entries() == []
    assert any("tests failed on every attempt" in body for body in comments)


def test_main_does_not_retry_infrastructure_failure(git_repo, monkeypatch):
    comments = []
    _prepare_main_test(git_repo, monkeypatch, "test-token", comments)
    calls = []

    def fail_to_start(command):
        calls.append(command)
        raise RuntimeError("private process details")

    monkeypatch.setattr(fixer, "run_test_command", fail_to_start)
    commits = []
    monkeypatch.setattr(
        fixer,
        "commit_and_push",
        lambda *args, **kwargs: commits.append(args),
    )

    fixer.main()

    assert len(calls) == 1
    assert commits == []
    assert fixer._git_status_entries() == []
    assert any("infrastructure failure" in body for body in comments)
    assert all("private process details" not in body for body in comments)


def test_test_command_modifying_validated_files_prevents_push(
    git_repo, monkeypatch
):
    comments = []
    _prepare_main_test(git_repo, monkeypatch, "test-token", comments)

    def mutating_tests(command):
        (git_repo / "tracked.txt").write_text(
            "changed during tests\n", encoding="utf-8"
        )
        return {
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "output": "passed",
        }

    monkeypatch.setattr(fixer, "run_test_command", mutating_tests)
    commits = []
    monkeypatch.setattr(
        fixer,
        "commit_and_push",
        lambda *args, **kwargs: commits.append(args),
    )

    fixer.main()

    assert commits == []
    assert (git_repo / "tracked.txt").read_text(encoding="utf-8") == "original\n"
    assert fixer._git_status_entries() == []
    assert any("changed files outside the validated fix" in body for body in comments)


def test_rollback_failure_prevents_commit_and_push(git_repo, monkeypatch):
    comments = []
    _prepare_main_test(git_repo, monkeypatch, "test-token", comments)
    monkeypatch.setenv("AUTOFIX_MAX_ATTEMPTS", "1")
    monkeypatch.setattr(
        fixer,
        "run_test_command",
        lambda command: {
            "passed": False,
            "exit_code": 1,
            "timed_out": False,
            "output": "failed",
        },
    )
    monkeypatch.setattr(
        fixer,
        "_rollback_attempt",
        lambda status: (_ for _ in ()).throw(RuntimeError("rollback failed")),
    )
    commits = []
    monkeypatch.setattr(
        fixer,
        "commit_and_push",
        lambda *args, **kwargs: commits.append(args),
    )

    fixer.main()

    assert commits == []
    assert any("rollback could not be verified" in body for body in comments)


def test_working_tree_rollback_restores_head_files_and_index(git_repo):
    baseline = fixer._working_tree_state()
    head = fixer._git_head_sha()
    index_tree = baseline[2]
    (git_repo / "tracked.txt").write_text("failed attempt\n", encoding="utf-8")
    (git_repo / "attempt.txt").write_text("temporary\n", encoding="utf-8")

    fixer._rollback_attempt(baseline)

    restored = fixer._working_tree_state()
    assert restored[0] == head
    assert restored[1] == baseline[1]
    assert restored[2] == index_tree
    assert fixer._rollback_state_matches(restored, baseline)
    assert fixer._git_status_entries() == []
    assert (git_repo / "tracked.txt").read_text(encoding="utf-8") == "original\n"
    assert not (git_repo / "attempt.txt").exists()


def test_working_tree_rollback_preserves_preexisting_user_files(git_repo):
    user_change = git_repo / "tracked.txt"
    user_change.write_text("user modification\n", encoding="utf-8")
    user_file = git_repo / "user-untracked.txt"
    user_file.write_text("user data\n", encoding="utf-8")
    baseline = fixer._working_tree_state()
    (git_repo / "attempt.txt").write_text("temporary\n", encoding="utf-8")

    fixer._rollback_attempt(baseline)

    assert user_change.read_text(encoding="utf-8") == "user modification\n"
    assert user_file.read_text(encoding="utf-8") == "user data\n"
    assert not (git_repo / "attempt.txt").exists()
    assert fixer._git_status_entries() == [
        (" M", "tracked.txt"),
        ("??", "user-untracked.txt"),
    ]


def test_working_tree_rollback_refuses_unexpected_head_change(git_repo):
    baseline = fixer._working_tree_state()
    original_head = baseline[0]
    _git(git_repo, "commit", "--allow-empty", "-m", "unexpected commit")

    with pytest.raises(RuntimeError, match="Git metadata changed"):
        fixer._rollback_attempt(baseline)

    assert fixer._git_head_sha() != original_head


def test_working_tree_rollback_refuses_metadata_changes(git_repo):
    baseline = fixer._working_tree_state()
    git_config = git_repo / ".git" / "config"
    original_config = git_config.read_bytes()
    git_config.write_bytes(original_config + b"\n# unexpected metadata change\n")

    with pytest.raises(RuntimeError, match="Git metadata changed"):
        fixer._rollback_attempt(baseline)

    assert git_config.read_bytes() != original_config


def test_generated_commit_rollback_refuses_unexpected_metadata(
    git_repo,
):
    baseline = fixer._working_tree_state()
    base_sha = fixer._git_head_sha()
    (git_repo / "tracked.txt").write_text("generated fix\n", encoding="utf-8")
    _git(git_repo, "add", "tracked.txt")
    _git(git_repo, "commit", "-m", "generated fix")
    generated_sha = fixer._git_head_sha()
    git_config = git_repo / ".git" / "config"
    git_config.write_bytes(git_config.read_bytes() + b"\n# unexpected change\n")

    with pytest.raises(RuntimeError, match="Git metadata changed before rollback"):
        fixer._rollback_generated_commit(base_sha, generated_sha, baseline)

    assert fixer._git_head_sha() == generated_sha


def _git_metadata_path(repo, name):
    path = Path(_git(repo, "rev-parse", "--git-path", name).stdout.strip())
    return path if path.is_absolute() else repo / path


@pytest.mark.parametrize("kind", ["ref", "hook", "config", "reflog"])
def test_git_metadata_snapshot_detects_mutable_git_metadata(git_repo, kind):
    if kind == "ref":
        branch = _git(git_repo, "symbolic-ref", "--short", "HEAD").stdout.strip()
        ref = _git_metadata_path(git_repo, f"refs/heads/{branch}")
        initial = fixer._git_head_sha()
        _git(git_repo, "commit", "--allow-empty", "-m", "second commit")
        updated = fixer._git_head_sha()
        _git(git_repo, "update-ref", "refs/heads/autofix-snapshot-ref", initial)
        ref = _git_metadata_path(git_repo, "refs/heads/autofix-snapshot-ref")
        before = fixer._git_metadata_snapshot()
        _git(git_repo, "update-ref", "refs/heads/autofix-snapshot-ref", updated)
    elif kind == "hook":
        ref = _git_metadata_path(git_repo, "hooks/pre-commit")
        before = fixer._git_metadata_snapshot()
        ref.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    elif kind == "config":
        ref = _git_metadata_path(git_repo, "config")
        before = fixer._git_metadata_snapshot()
        _git(git_repo, "config", "--local", "autofix.snapshot", "changed")
    else:
        ref = _git_metadata_path(git_repo, "logs/HEAD")
        before = fixer._git_metadata_snapshot()
        ref.write_bytes(ref.read_bytes() + b"\n")

    assert fixer._git_metadata_snapshot() != before


def test_git_metadata_snapshot_detects_created_and_deleted_files(git_repo):
    metadata_file = _git_metadata_path(git_repo, "ai-autofix-test-metadata")
    before = fixer._git_metadata_snapshot()

    metadata_file.write_bytes(b"created")
    created = fixer._git_metadata_snapshot()
    assert created != before

    metadata_file.unlink()
    assert fixer._git_metadata_snapshot() == before


def test_git_metadata_snapshot_rejects_symlink_entries(git_repo):
    if sys.platform != "linux":
        pytest.skip("Symlink metadata integration test requires Linux")
    git_dir = Path(
        _git(git_repo, "rev-parse", "--absolute-git-dir").stdout.strip()
    )
    link = git_dir / "autofix-metadata-symlink"
    os.symlink(git_dir / "config", link)

    with pytest.raises(RuntimeError, match="symbolic link"):
        fixer._git_metadata_snapshot()


def test_git_metadata_snapshot_rejects_unexpected_filesystem_objects(git_repo):
    if sys.platform != "linux":
        pytest.skip("FIFO metadata integration test requires Linux")
    git_dir = Path(
        _git(git_repo, "rev-parse", "--absolute-git-dir").stdout.strip()
    )
    fifo = git_dir / "autofix-metadata-fifo"
    os.mkfifo(fifo)

    with pytest.raises(RuntimeError, match="unexpected file type"):
        fixer._git_metadata_snapshot()


@pytest.mark.parametrize(
    ("limit", "value"),
    [
        ("_MAX_GIT_METADATA_FILES", 1),
        ("_MAX_GIT_METADATA_BYTES", 1),
    ],
)
def test_git_metadata_snapshot_enforces_limits(
    git_repo, monkeypatch, limit, value
):
    monkeypatch.setattr(fixer, limit, value)

    with pytest.raises(RuntimeError, match="safe snapshot limit"):
        fixer._git_metadata_snapshot()


def test_git_metadata_snapshot_fails_closed_when_root_is_unavailable(
    tmp_path, monkeypatch
):
    missing = str(tmp_path / "missing-git-metadata")
    monkeypatch.setattr(
        fixer,
        "_metadata_git_directories",
        lambda: (("worktree", missing), ("common", missing)),
    )

    with pytest.raises(RuntimeError, match="safely inspect Git metadata"):
        fixer._git_metadata_snapshot()


def test_git_metadata_snapshot_handles_worktree_and_common_directories(
    git_repo, tmp_path, monkeypatch
):
    worktree = tmp_path / "linked-worktree"
    _git(git_repo, "worktree", "add", "--detach", str(worktree), "HEAD")
    worktree_git_dir = None
    try:
        monkeypatch.chdir(worktree)
        worktree_git_dir = Path(
            _git(worktree, "rev-parse", "--absolute-git-dir").stdout.strip()
        )
        common_git_dir = Path(
            _git(worktree, "rev-parse", "--git-common-dir").stdout.strip()
        )
        assert worktree_git_dir != common_git_dir
        before = fixer._git_metadata_snapshot()

        (worktree_git_dir / "autofix-worktree-metadata").write_bytes(b"changed")

        assert fixer._git_metadata_snapshot() != before
    finally:
        monkeypatch.chdir(git_repo)
        if worktree_git_dir is not None:
            (worktree_git_dir / "autofix-worktree-metadata").unlink(
                missing_ok=True
            )
        _git(git_repo, "worktree", "remove", str(worktree))


def test_metadata_change_during_test_stops_without_commit_or_rollback(
    git_repo, monkeypatch
):
    comments = []
    _prepare_main_test(git_repo, monkeypatch, "test-token", comments)
    test_calls = []
    commits = []

    def mutate_git_config(command):
        test_calls.append(command)
        with (git_repo / ".git" / "config").open("ab") as config:
            config.write(b"\n# changed by test command\n")
        return {
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "output": "passed",
        }

    monkeypatch.setattr(fixer, "run_test_command", mutate_git_config)
    monkeypatch.setattr(
        fixer, "commit_and_push", lambda *args, **kwargs: commits.append(args)
    )

    fixer.main()

    assert len(test_calls) == 1
    assert commits == []
    assert (git_repo / "tracked.txt").read_text(encoding="utf-8") == (
        "modified by agent\n"
    )
    assert any("manual inspection is required" in body for body in comments)


def test_head_change_during_test_is_not_reset_or_pushed(git_repo, monkeypatch):
    comments = []
    _prepare_main_test(git_repo, monkeypatch, "test-token", comments)
    original_head = fixer._git_head_sha()
    test_calls = []
    commits = []

    def change_head(command):
        test_calls.append(command)
        _git(git_repo, "commit", "--allow-empty", "-m", "test-side commit")
        return {
            "passed": True,
            "exit_code": 0,
            "timed_out": False,
            "output": "passed",
        }

    monkeypatch.setattr(fixer, "run_test_command", change_head)
    monkeypatch.setattr(
        fixer, "commit_and_push", lambda *args, **kwargs: commits.append(args)
    )

    fixer.main()

    assert len(test_calls) == 1
    assert commits == []
    assert fixer._git_head_sha() != original_head
    assert any("manual inspection is required" in body for body in comments)


def test_main_skips_when_no_test_command_is_configured(git_repo, monkeypatch):
    # Legacy test: without Mode 2 (allow-unverified-push absent/false),
    # the agent still runs and generates a fix, but the fix is NOT pushed.
    # The comment must explain that no test command is configured.
    comments = []
    _prepare_main_test(git_repo, monkeypatch, "test-token", comments)
    monkeypatch.delenv("AUTOFIX_TEST_COMMAND")
    # allow-unverified-push defaults to false (env var absent)
    monkeypatch.delenv("AUTOFIX_ALLOW_UNVERIFIED_PUSH", raising=False)

    commits = []
    monkeypatch.setattr(
        fixer, "commit_and_push",
        lambda *args, **kwargs: commits.append(args),
    )

    fixer.main()

    # Agent ran (no "agent must not run" failure); but nothing was committed
    assert commits == []
    # The comment must mention that no test suite was configured
    assert any(
        "no test" in body.lower() or "allow-unverified-push" in body.lower()
        or "basic" in body.lower()
        for body in comments
    )


def test_mode2_fix_not_pushed_when_allow_unverified_push_is_false(
    git_repo, monkeypatch
):
    """Mode 2: agent runs, basic validation passes, but push is suppressed."""
    comments = []
    _prepare_main_test(git_repo, monkeypatch, "test-token", comments)
    monkeypatch.delenv("AUTOFIX_TEST_COMMAND")
    monkeypatch.setenv("AUTOFIX_ALLOW_UNVERIFIED_PUSH", "false")

    basic_calls = []

    def fake_basic_validation(changed_files):
        basic_calls.append(changed_files)
        return {"passed": True, "exit_code": 0, "timed_out": False, "output": "ok"}

    monkeypatch.setattr(fixer, "_run_basic_validation", fake_basic_validation)
    commits = []
    monkeypatch.setattr(
        fixer, "commit_and_push",
        lambda *args, **kwargs: commits.append(args),
    )

    fixer.main()

    # Basic validation ran
    assert basic_calls, "Expected _run_basic_validation to be called"
    # Nothing was committed / pushed
    assert commits == []
    # Comment explains the situation
    assert any(
        "allow-unverified-push" in body.lower() or "basic" in body.lower()
        for body in comments
    )


def test_mode2_fix_pushed_when_allow_unverified_push_is_true(
    git_repo, monkeypatch
):
    """Mode 2: agent runs, basic validation passes, push is allowed."""
    comments = []
    _prepare_main_test(git_repo, monkeypatch, "test-token", comments)
    monkeypatch.delenv("AUTOFIX_TEST_COMMAND")
    monkeypatch.setenv("AUTOFIX_ALLOW_UNVERIFIED_PUSH", "true")

    basic_calls = []

    def fake_basic_validation(changed_files):
        basic_calls.append(changed_files)
        return {"passed": True, "exit_code": 0, "timed_out": False, "output": "ok"}

    monkeypatch.setattr(fixer, "_run_basic_validation", fake_basic_validation)
    commits = []
    monkeypatch.setattr(
        fixer, "commit_and_push",
        lambda *args, **kwargs: commits.append(args),
    )

    fixer.main()

    # Basic validation ran
    assert basic_calls, "Expected _run_basic_validation to be called"
    # Changes were committed
    assert len(commits) == 1
    # Comment mentions basic validation only
    assert any(
        "basic" in body.lower() or "basic validation" in body.lower()
        for body in comments
    )


def test_mode2_fix_rolled_back_on_syntax_failure(
    git_repo, monkeypatch
):
    """Mode 2: agent runs, but basic validation fails -> rollback, no push."""
    comments = []
    _prepare_main_test(git_repo, monkeypatch, "test-token", comments)
    monkeypatch.delenv("AUTOFIX_TEST_COMMAND")
    monkeypatch.setenv("AUTOFIX_ALLOW_UNVERIFIED_PUSH", "true")

    def fake_basic_validation(changed_files):
        return {
            "passed": False,
            "exit_code": 1,
            "timed_out": False,
            "output": "SyntaxError: invalid syntax",
        }

    monkeypatch.setattr(fixer, "_run_basic_validation", fake_basic_validation)
    commits = []
    monkeypatch.setattr(
        fixer, "commit_and_push",
        lambda *args, **kwargs: commits.append(args),
    )
    monkeypatch.setenv("AUTOFIX_MAX_ATTEMPTS", "1")

    fixer.main()

    # Nothing was committed / pushed
    assert commits == []
    # Rollback happened: working tree should be clean
    assert not fixer.get_changed_files()
