"""Tests for fixer.py.

These tests exercise real local Git behaviour against temporary
repositories created with pytest's ``tmp_path``. No test may touch the
network, push to a remote, or operate on the repository the suite itself
is running in.
"""

import base64
import subprocess

import pytest

import fixer


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
    monkeypatch.setattr(fixer, "post_comment", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        fixer, "get_pull_request", lambda *args: _pull_request()
    )

    class FakeMessage:
        content = "summary"

    class FakeAgent:
        def invoke(self, payload):
            (git_repo / "unauthorized.txt").write_text(
                "nope\n", encoding="utf-8"
            )
            return {"messages": [FakeMessage()]}

    monkeypatch.setattr(fixer, "build_agent", lambda allowed: FakeAgent())

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
    monkeypatch.setenv("REVIEW_ISSUES", "- tracked.txt:1 [bug] broken")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("PR_NUMBER", "9")
    monkeypatch.setenv("GITHUB_TOKEN", token)
    monkeypatch.setenv("PR_HEAD_REF", "feature")
    monkeypatch.setattr(
        fixer,
        "post_comment",
        lambda repo, pr_number, tok, body: comments.append(body),
    )
    monkeypatch.setattr(
        fixer, "get_pull_request", lambda *args: _pull_request()
    )

    class FakeMessage:
        content = summary

    class FakeAgent:
        def invoke(self, payload):
            if agent_changes:
                (git_repo / "tracked.txt").write_text(
                    "modified by agent\n", encoding="utf-8"
                )
            return {"messages": [FakeMessage()]}

    monkeypatch.setattr(fixer, "build_agent", lambda allowed: FakeAgent())


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
        def invoke(self, payload):
            (git_repo / "unauthorized.txt").write_text(
                "unexpected\n", encoding="utf-8"
            )

            class Message:
                content = secret

            return {"messages": [Message()]}

    monkeypatch.setattr(
        fixer, "build_agent", lambda allowed: UnauthorizedAgent()
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


def test_main_skips_non_git_working_directory_before_api(
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

    def unexpected(*args, **kwargs):
        raise AssertionError("GitHub API must not be called outside checkout")

    monkeypatch.setattr(fixer, "get_pull_request", unexpected)
    monkeypatch.setattr(fixer, "build_agent", unexpected)

    fixer.main()


def test_main_skips_checkout_for_different_repository(git_repo, monkeypatch):
    _prepare_main_test(git_repo, monkeypatch, "test-token", [])
    monkeypatch.setenv("GITHUB_REPOSITORY", "other/repo")

    def unexpected(*args, **kwargs):
        raise AssertionError("GitHub API must not be called for wrong checkout")

    monkeypatch.setattr(fixer, "get_pull_request", unexpected)

    fixer.main()


def test_main_skips_when_working_directory_is_not_workspace(
    git_repo, monkeypatch
):
    _prepare_main_test(git_repo, monkeypatch, "test-token", [])
    monkeypatch.setenv("GITHUB_WORKSPACE", str(git_repo.parent))

    def unexpected(*args, **kwargs):
        raise AssertionError("GitHub API must not be called from wrong directory")

    monkeypatch.setattr(fixer, "get_pull_request", unexpected)

    fixer.main()


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
    assert any("could not be pushed" in body for body in comments)
    assert all(token not in body for body in comments)


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
    assert any("could not be pushed" in body for body in comments)
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
    assert env["GIT_CONFIG_COUNT"] == "2"
    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    assert env["GIT_CONFIG_VALUE_0"] == ""
    assert env["GIT_CONFIG_KEY_1"] == "http.https://github.com/.extraheader"
    assert env["GIT_CONFIG_VALUE_1"] == f"Authorization: Basic {credentials}"


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
    assert env["GIT_CONFIG_COUNT"] == "2"
    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    assert env["GIT_CONFIG_VALUE_0"] == ""
    assert env["GIT_CONFIG_KEY_1"] == "http.https://github.com/.extraheader"
    assert env["GIT_CONFIG_VALUE_1"] == f"Authorization: Basic {credentials}"
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
        _pull_request(sha="a" * 40),
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
