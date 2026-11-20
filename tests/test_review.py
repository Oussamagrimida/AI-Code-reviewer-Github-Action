import pytest
import requests
import review
import os
import subprocess
import sys
import json
from requests.structures import CaseInsensitiveDict

def test_post_review_sends_correct_github_payload(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 201

        def raise_for_status(self):
            pass

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["payload"] = kwargs["json"]
        return FakeResponse()

    monkeypatch.setattr(review.requests, "post", fake_post)

    result = {
        "summary": "Found a potential bug.",
        "issues": [
            {
                "path": "src/example.py",
                "line": 2,
                "category": "correctness",
                "severity": "high",
                "confidence": 0.95,
                "body": "This can return an incorrect result.",
            }
        ],
    }

    review.post_review(
        "owner/repo",
        "12",
        "test-token",
        "abc123",
        result,
    )

    assert captured["url"].endswith("/repos/owner/repo/pulls/12/reviews")

    payload = captured["payload"]
    assert payload["commit_id"] == "abc123"
    assert payload["event"] == "COMMENT"
    assert payload["body"].endswith("Found a potential bug.")
    assert len(payload["comments"]) == 1
    assert payload["comments"][0]["path"] == "src/example.py"
    assert payload["comments"][0]["line"] == 2
    assert payload["comments"][0]["side"] == "RIGHT"


def test_post_review_falls_back_when_inline_comments_are_rejected(
    monkeypatch,
):
    payloads = []

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code

        def json(self):
            return {
                "message": "Validation Failed",
                "errors": [
                    {
                        "resource": "PullRequestReviewComment",
                        "field": "line",
                        "code": "invalid",
                    }
                ],
            }

        def raise_for_status(self):
            if self.status_code >= 400:
                raise review.requests.HTTPError(
                    f"HTTP {self.status_code}"
                )

    def fake_post(url, **kwargs):
        payloads.append(kwargs["json"])
        return FakeResponse(422 if len(payloads) == 1 else 201)

    monkeypatch.setattr(review.requests, "post", fake_post)

    result = {
        "summary": "One issue found.",
        "issues": [
            {
                "path": "src/example.py",
                "line": 2,
                "category": "correctness",
                "severity": "high",
                "confidence": 0.95,
                "body": "Incorrect result.",
            }
        ],
    }

    review.post_review(
        "owner/repo",
        "12",
        "test-token",
        "abc123",
        result,
    )

    assert len(payloads) == 2
    assert len(payloads[0]["comments"]) == 1
    assert payloads[1]["comments"] == []
    assert "Flagged issues" in payloads[1]["body"]
    assert "src/example.py:2" in payloads[1]["body"]


@pytest.mark.parametrize("status_code", [400, 422])
def test_post_review_does_not_fallback_for_unrelated_validation_error(
    monkeypatch,
    status_code,
):
    calls = []

    class FakeResponse:
        def __init__(self):
            self.status_code = status_code

        def json(self):
            return {
                "message": "Validation Failed",
                "errors": [
                    {
                        "resource": "PullRequest",
                        "field": "title",
                        "code": "missing",
                    }
                ],
            }

        def raise_for_status(self):
            raise requests.HTTPError(f"HTTP {status_code}")

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"])
        return FakeResponse()

    monkeypatch.setattr(review.requests, "post", fake_post)

    result = {
        "summary": "One issue found.",
        "issues": [
            {
                "path": "src/example.py",
                "line": 2,
                "category": "correctness",
                "severity": "high",
                "confidence": 0.95,
                "body": "Incorrect result.",
            }
        ],
    }

    with pytest.raises(review.GitHubAPIError, match=f"HTTP {status_code}"):
        review.post_review(
            "owner/repo",
            "12",
            "test-token",
            "abc123",
            result,
        )

    assert len(calls) == 1
    assert len(calls[0]["comments"]) == 1


@pytest.mark.parametrize("status_code", [400, 422])
def test_post_review_does_not_fallback_without_inline_comments(
    monkeypatch,
    status_code,
):
    calls = []

    class FakeResponse:
        def __init__(self):
            self.status_code = status_code

        def json(self):
            return {
                "message": "Validation Failed",
                "errors": [
                    {
                        "resource": "PullRequestReviewComment",
                        "field": "line",
                        "code": "invalid",
                    }
                ],
            }

        def raise_for_status(self):
            raise requests.HTTPError(f"HTTP {status_code}")

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"])
        return FakeResponse()

    monkeypatch.setattr(review.requests, "post", fake_post)

    with pytest.raises(review.GitHubAPIError, match=f"HTTP {status_code}"):
        review.post_review(
            "owner/repo",
            "12",
            "test-token",
            "abc123",
            {"summary": "No issues found.", "issues": []},
        )

    assert len(calls) == 1
    assert calls[0]["comments"] == []


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
def test_post_review_does_not_retry_on_server_error(
    monkeypatch,
    status_code,
):
    calls = []

    class FakeResponse:
        def __init__(self):
            self.status_code = status_code

        def raise_for_status(self):
            raise requests.HTTPError(f"HTTP {status_code}")

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"])
        return FakeResponse()

    monkeypatch.setattr(review.requests, "post", fake_post)

    result = {
        "summary": "One issue found.",
        "issues": [
            {
                "path": "src/example.py",
                "line": 2,
                "category": "correctness",
                "severity": "high",
                "confidence": 0.95,
                "body": "Incorrect result.",
            }
        ],
    }

    with pytest.raises(review.GitHubAPIError, match=f"HTTP {status_code}"):
        review.post_review(
            "owner/repo",
            "12",
            "test-token",
            "abc123",
            result,
        )

    assert len(calls) == 1


def test_post_review_raises_on_authentication_error(monkeypatch):
    class FakeResponse:
        status_code = 401

        def raise_for_status(self):
            raise requests.HTTPError("401 Unauthorized")

    monkeypatch.setattr(
        review.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(review.GitHubAPIError, match="HTTP 401"):
        review.post_review(
            "owner/repo",
            "12",
            "test-token",
            "abc123",
            {"summary": "Review complete.", "issues": []},
        )


def test_post_review_raises_on_timeout(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.Timeout("Request timed out")

    monkeypatch.setattr(review.requests, "post", fake_post)

    with pytest.raises(review.GitHubAPIError, match="timed out"):
        review.post_review(
            "owner/repo",
            "12",
            "test-token",
            "abc123",
            {"summary": "Review complete.", "issues": []},
        )


def test_post_review_does_not_retry_on_timeout(monkeypatch):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs.get('json'))
        raise requests.Timeout("Request timed out")

    monkeypatch.setattr(review.requests, "post", fake_post)

    with pytest.raises(review.GitHubAPIError, match="timed out"):
        review.post_review(
            "owner/repo",
            "12",
            "test-token",
            "abc123",
            {"summary": "Review complete.", "issues": []},
        )

    assert len(calls) == 1


def test_post_review_does_not_retry_on_connection_error(monkeypatch):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(kwargs.get('json'))
        raise requests.ConnectionError("Connection failed")

    monkeypatch.setattr(review.requests, "post", fake_post)

    with pytest.raises(review.GitHubAPIError, match="lost its connection"):
        review.post_review(
            "owner/repo",
            "12",
            "test-token",
            "abc123",
            {"summary": "Review complete.", "issues": []},
        )

    assert len(calls) == 1


def test_post_review_prints_inline_comment_count_on_success(monkeypatch, capsys):
    captured_payloads = []

    class FakeResponse:
        status_code = 201

        def raise_for_status(self):
            pass

    def fake_post(url, **kwargs):
        captured_payloads.append(kwargs["json"])
        return FakeResponse()

    monkeypatch.setattr(review.requests, "post", fake_post)

    result = {
        "summary": "Found a potential bug.",
        "issues": [
            {
                "path": "src/example.py",
                "line": line,
                "category": "correctness",
                "severity": "high",
                "confidence": 0.95,
                "body": f"This can return an incorrect result on line {line}.",
            } for line in range(2, 7)
        ],
    }

    review.post_review(
        "owner/repo",
        "12",
        "test-token",
        "abc123",
        result,
    )

    captured = capsys.readouterr()
    assert "Review posted: 5 inline comment(s)." in captured.out
    assert len(captured_payloads[0]["comments"]) == 5


def test_post_review_prints_zero_on_fallback(monkeypatch, capsys):
    payloads = []

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code

        def json(self):
            return {
                "message": "Validation Failed",
                "errors": [
                    {
                        "resource": "PullRequestReviewComment",
                        "field": "position",
                        "code": "invalid",
                    }
                ],
            }

        def raise_for_status(self):
            if self.status_code >= 400:
                raise review.requests.HTTPError(f"HTTP {self.status_code}")

    def fake_post(url, **kwargs):
        payloads.append(kwargs["json"])
        return FakeResponse(422 if len(payloads) == 1 else 201)

    monkeypatch.setattr(review.requests, "post", fake_post)

    result = {
        "summary": "One issue found.",
        "issues": [
            {
                "path": "src/example.py",
                "line": 2,
                "category": "correctness",
                "severity": "high",
                "confidence": 0.95,
                "body": "Incorrect result.",
            }
        ],
    }

    review.post_review(
        "owner/repo",
        "12",
        "test-token",
        "abc123",
        result,
    )

    captured = capsys.readouterr()
    assert "Inline comments rejected (422), falling back to summary-only." in captured.out
    assert "Review posted: 0 inline comment(s)." in captured.out
    assert payloads[1]["comments"] == []


def test_post_review_fallback_failure_raises(monkeypatch, capsys):
    calls = []

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code

        def json(self):
            return {
                "message": "Validation Failed",
                "errors": [
                    {
                        "resource": "PullRequestReviewComment",
                        "field": "line",
                        "code": "invalid",
                    }
                ],
            }

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(
                    f"HTTP {self.status_code}"
                )

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"])
        return FakeResponse(422 if len(calls) == 1 else 500)

    monkeypatch.setattr(review.requests, "post", fake_post)

    result = {
        "summary": "One issue found.",
        "issues": [
            {
                "path": "src/example.py",
                "line": 2,
                "category": "correctness",
                "severity": "high",
                "confidence": 0.95,
                "body": "Incorrect result.",
            }
        ],
    }

    with pytest.raises(review.GitHubAPIError, match="HTTP 500"):
        review.post_review(
            "owner/repo",
            "12",
            "test-token",
            "abc123",
            result,
        )

    assert len(calls) == 2
    assert calls[1]["comments"] == []
    assert "Review posted:" not in capsys.readouterr().out


def test_post_review_without_issues_sends_summary_only(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 201

        def raise_for_status(self):
            pass

    def fake_post(url, **kwargs):
        captured["payload"] = kwargs["json"]
        return FakeResponse()

    monkeypatch.setattr(review.requests, "post", fake_post)

    review.post_review(
        "owner/repo",
        "12",
        "test-token",
        "abc123",
        {"summary": "No issues found.", "issues": []},
    )

    assert captured["payload"]["commit_id"] == "abc123"
    assert captured["payload"]["comments"] == []
    assert "No issues found." in captured["payload"]["body"]


def test_post_review_retries_explicit_rate_limit(monkeypatch):
    payloads = []
    delays = []

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code
            self.headers = CaseInsensitiveDict({"retry-after": "3"})

    def fake_post(url, **kwargs):
        payloads.append(kwargs["json"])
        return FakeResponse(429 if len(payloads) == 1 else 201)

    monkeypatch.setattr(review.requests, "post", fake_post)
    monkeypatch.setattr(review.time, "sleep", delays.append)

    review.post_review(
        "owner/repo",
        "12",
        "test-token",
        "abc123",
        {"summary": "No issues found.", "issues": []},
    )

    assert len(payloads) == 2
    assert delays == [3]
    assert all(payload["comments"] == [] for payload in payloads)


def test_main_skips_review_if_pr_head_changes(monkeypatch):
    posted_reviews = []

    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("PR_NUMBER", "12")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")

    pr_info_responses = iter([
        {"head": {"sha": "original-sha"}},
        {"head": {"sha": "new-sha"}},
    ])

    monkeypatch.setattr(
        review,
        "get_pr_info",
        lambda *args: next(pr_info_responses),
    )

    monkeypatch.setattr(
        review,
        "get_changed_files",
        lambda *args: [{"path": "src/example.py", "patch": "+new code"}],
    )

    monkeypatch.setattr(
        review.llm_review,
        "review_diff",
        lambda files: {"summary": "Found an issue.", "issues": []},
    )

    monkeypatch.setattr(
        review,
        "post_review",
        lambda *args: posted_reviews.append(args),
    )

    review.main()

    assert posted_reviews == []


def test_main_posts_review_when_pr_head_sha_is_unchanged(
    monkeypatch,
    tmp_path,
):
    output_file = tmp_path / "github_output.txt"
    posted_reviews = []
    pr_info_calls = []
    review_result = {
        "summary": "Found a correctness issue.",
        "issues": [
            {
                "path": "src/example.py",
                "line": 2,
                "category": "correctness",
                "severity": "high",
                "confidence": 0.95,
                "body": "Handle this changed value.",
            }
        ],
    }

    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("PR_NUMBER", "12")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    def fake_get_pr_info(*args):
        pr_info_calls.append(args)
        return {"head": {"sha": "abc123"}}

    monkeypatch.setattr(review, "get_pr_info", fake_get_pr_info)
    monkeypatch.setattr(
        review,
        "get_changed_files",
        lambda *args: [
            {"path": "src/example.py", "patch": "+new code"}
        ],
    )
    monkeypatch.setattr(
        review.llm_review,
        "review_diff",
        lambda files: review_result,
    )
    monkeypatch.setattr(
        review,
        "post_review",
        lambda *args: posted_reviews.append(args),
    )

    review.main()

    assert len(pr_info_calls) == 2
    assert len(posted_reviews) == 1
    assert posted_reviews[0] == (
        "owner/repo",
        "12",
        "test-token",
        "abc123",
        review_result,
    )

    output = output_file.read_text(encoding="utf-8")
    assert "has_bugs=true" in output
    assert "src/example.py:2" in output
    assert "Handle this changed value." in output


def test_run_exits_when_second_pr_info_fetch_fails(
    monkeypatch,
    capsys,
    tmp_path,
):
    output_file = tmp_path / "github_output.txt"
    pr_info_calls = []
    posted_reviews = []

    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("PR_NUMBER", "12")
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    def fake_get_pr_info(*args):
        pr_info_calls.append(args)
        if len(pr_info_calls) == 1:
            return {"head": {"sha": "abc123"}}
        raise review.GitHubAPIError(
            "GitHub API request failed (HTTP 500)."
        )

    monkeypatch.setattr(review, "get_pr_info", fake_get_pr_info)
    monkeypatch.setattr(
        review,
        "get_changed_files",
        lambda *args: [
            {"path": "src/example.py", "patch": "+new code"}
        ],
    )
    monkeypatch.setattr(
        review.llm_review,
        "review_diff",
        lambda files: {
            "summary": "Review completed.",
            "issues": [],
        },
    )
    monkeypatch.setattr(
        review,
        "post_review",
        lambda *args: posted_reviews.append(args),
    )

    with pytest.raises(SystemExit) as exc:
        review.run()

    assert exc.value.code == 1
    assert len(pr_info_calls) == 2
    assert posted_reviews == []
    assert not output_file.exists()

    captured = capsys.readouterr()
    assert "GitHub API failure: GitHub API request failed (HTTP 500)." in captured.err
    assert "Posting review..." not in captured.out


def test_get_changed_files_fetches_multiple_pages(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, data):
            self.data = data
            self.status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return self.data

    def fake_get(url, **kwargs):
        page = kwargs["params"]["page"]
        calls.append(page)

        if page == 1:
            return FakeResponse(
                [
                    {
                        "filename": f"src/file_{i}.py",
                        "patch": f"@@ -1 +1 @@\n-old\n+new_{i}\n",
                    }
                    for i in range(100)
                ]
            )

        if page == 2:
            return FakeResponse(
                [
                    {
                        "filename": "src/final.py",
                        "patch": "@@ -1 +1 @@\n-old\n+new\n",
                    }
                ]
            )

        return FakeResponse([])

    monkeypatch.setattr(review.requests, "get", fake_get)
    monkeypatch.setattr(review.time, "sleep", lambda _: None)

    files = review.get_changed_files(
        "owner/repo",
        "12",
        "test-token",
    )

    assert calls == [1, 2, 3]
    assert len(files) == 101
    assert files[-1]["path"] == "src/final.py"


def test_write_github_outputs_with_fixable_issue(monkeypatch, tmp_path):
    output_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    result = {
        "issues": [
            {
                "path": "src/example.py",
                "line": 2,
                "category": "correctness",
                "severity": "high",
                "confidence": 0.95,
                "body": "Incorrect result.",
            }
        ]
    }

    review.write_github_outputs(result)

    output = output_file.read_text(encoding="utf-8")

    assert "has_bugs=true" in output
    assert "issues_summary<<EOF" in output
    assert "src/example.py:2" in output
    assert "Incorrect result." in output


def test_write_github_outputs_without_fixable_issues(monkeypatch, tmp_path):
    output_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    review.write_github_outputs({"issues": []})

    output = output_file.read_text(encoding="utf-8")

    assert "has_bugs=false" in output
    assert "issues_summary<<EOF" in output

def test_write_github_outputs_uses_safe_delimiter(monkeypatch, tmp_path):
    output_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    result = {
        "issues": [
            {
                "path": "src/example.py",
                "line": 2,
                "category": "correctness",
                "severity": "high",
                "confidence": 0.95,
                "body": "Incorrect result.\nEOF\nAdditional details.",
            }
        ]
    }

    review.write_github_outputs(result)

    output = output_file.read_text(encoding="utf-8")
    delimiter_line = next(
        line for line in output.splitlines()
        if line.startswith("issues_summary<<")
    )
    delimiter = delimiter_line.split("<<", 1)[1]

    assert delimiter != "EOF"
    assert f"\n{delimiter}\n" in output
    assert "EOF Additional details." in output


def test_write_github_outputs_flattens_multiline_issue_bodies(
    monkeypatch, tmp_path
):
    output_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    review.write_github_outputs({
        "issues": [
            {
                "path": "src/example.py",
                "line": 2,
                "category": "correctness",
                "severity": "high",
                "confidence": 0.95,
                "body": "Broken result.\n- secret.py:1 [security] injected path",
            }
        ]
    })

    output = output_file.read_text(encoding="utf-8")
    assert "\n- secret.py:1 [security]" not in output
    assert "Broken result. - secret.py:1 [security]" in output

@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (401, "401 Unauthorized"),
        (404, "404 Not Found"),
    ],
)
def test_get_pr_info_raises_on_http_error(
    monkeypatch,
    status_code,
    message,
):
    calls = []

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code

        def raise_for_status(self):
            raise requests.HTTPError(message)

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeResponse(status_code)

    monkeypatch.setattr(review.requests, "get", fake_get)
    monkeypatch.setattr(review.time, "sleep", lambda _: None)

    with pytest.raises(review.GitHubAPIError, match=f"HTTP {status_code}"):
        review.get_pr_info(
            "owner/repo",
            "123",
            "fake-token",
        )

    assert len(calls) == 1


def test_get_changed_files_raises_on_http_error(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 500

        def raise_for_status(self):
            raise requests.HTTPError("500 Internal Server Error")

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return FakeResponse()

    monkeypatch.setattr(review.requests, "get", fake_get)
    monkeypatch.setattr(review.time, "sleep", lambda _: None)

    with pytest.raises(
        review.GitHubAPIError,
        match="HTTP 500",
    ):
        review.get_changed_files(
            "owner/repo",
            "123",
            "fake-token",
        )

    assert len(calls) == 4


@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
def test_get_pr_info_retries_on_transient_http_error(
    monkeypatch,
    status_code,
):
    calls = []
    delays = []

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code
            self.headers = {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"HTTP {self.status_code}")

        def json(self):
            return {"head": {"sha": "a" * 40}}

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        response_status = status_code if len(calls) == 1 else 200
        return FakeResponse(response_status)

    monkeypatch.setattr(review.requests, "get", fake_get)
    monkeypatch.setattr(review.time, "sleep", delays.append)

    result = review.get_pr_info(
        "owner/repo",
        "123",
        "fake-token",
    )

    assert result == {"head": {"sha": "a" * 40}}
    assert len(calls) == 2
    assert delays == [5]


@pytest.mark.parametrize(
    "error_type",
    [requests.Timeout, requests.ConnectionError],
)
def test_get_pr_info_retries_on_network_error(monkeypatch, error_type):
    calls = []
    delays = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"head": {"sha": "a" * 40}}

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise error_type("Temporary network failure")
        return FakeResponse()

    monkeypatch.setattr(review.requests, "get", fake_get)
    monkeypatch.setattr(review.time, "sleep", delays.append)

    result = review.get_pr_info(
        "owner/repo",
        "123",
        "fake-token",
    )

    assert result == {"head": {"sha": "a" * 40}}
    assert len(calls) == 2
    assert delays == [5]


def test_get_pr_info_raises_on_timeout(monkeypatch):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        raise requests.Timeout("Request timed out")

    monkeypatch.setattr(review.requests, "get", fake_get)
    monkeypatch.setattr(review.time, "sleep", lambda _: None)

    with pytest.raises(review.GitHubAPIError, match="timed out"):
        review.get_pr_info(
            "owner/repo",
            "123",
            "fake-token",
        )

    assert len(calls) == 4


def test_github_rate_limit_honors_retry_after(monkeypatch):
    calls = []
    delays = []

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code
            self.headers = CaseInsensitiveDict({"retry-after": "4"})

        def json(self):
            return {"head": {"sha": "a" * 40}}

    def fake_get(*args, **kwargs):
        calls.append(1)
        return FakeResponse(429 if len(calls) == 1 else 200)

    monkeypatch.setattr(review.requests, "get", fake_get)
    monkeypatch.setattr(review.time, "sleep", delays.append)

    review.get_pr_info("owner/repo", "123", "test-token")

    assert len(calls) == 2
    assert delays == [4]


def test_github_rate_limit_remaining_header_is_case_insensitive(monkeypatch):
    calls = []
    delays = []

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code
            self.headers = CaseInsensitiveDict(
                {"x-ratelimit-remaining": "0"}
            )

        def json(self):
            return {"head": {"sha": "a" * 40}}

    def fake_get(*args, **kwargs):
        calls.append(1)
        return FakeResponse(403 if len(calls) == 1 else 200)

    monkeypatch.setattr(review.requests, "get", fake_get)
    monkeypatch.setattr(review.time, "sleep", delays.append)

    review.get_pr_info("owner/repo", "123", "test-token")

    assert len(calls) == 2
    assert delays == [5]


def test_github_malformed_retry_after_uses_bounded_backoff(monkeypatch):
    calls = []
    delays = []

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code
            self.headers = CaseInsensitiveDict({"Retry-After": "later"})

        def json(self):
            return {"head": {"sha": "a" * 40}}

    def fake_get(*args, **kwargs):
        calls.append(1)
        return FakeResponse(429 if len(calls) == 1 else 200)

    monkeypatch.setattr(review.requests, "get", fake_get)
    monkeypatch.setattr(review.time, "sleep", delays.append)

    review.get_pr_info("owner/repo", "123", "test-token")

    assert len(calls) == 2
    assert delays == [5]


def test_invalid_pull_request_sha_is_rejected(monkeypatch):
    class FakeResponse:
        status_code = 200

        def json(self):
            return {"head": {"sha": "not-a-commit"}}

    monkeypatch.setattr(
        review.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(review.GitHubMetadataError, match="head SHA"):
        review.get_pr_info("owner/repo", "123", "test-token")


def test_get_changed_files_raises_on_timeout(monkeypatch):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        raise requests.Timeout("Request timed out")

    monkeypatch.setattr(review.requests, "get", fake_get)
    monkeypatch.setattr(review.time, "sleep", lambda _: None)

    with pytest.raises(review.GitHubAPIError, match="timed out"):
        review.get_changed_files(
            "owner/repo",
            "123",
            "fake-token",
        )

    assert len(calls) == 4



def test_run_exits_when_github_api_fails(monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("PR_NUMBER", "123")
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token")

    def fake_get_pr_info(*args, **kwargs):
        raise review.GitHubAPIError(
            "GitHub API request failed (HTTP 404)."
        )

    monkeypatch.setattr(review, "get_pr_info", fake_get_pr_info)

    with pytest.raises(SystemExit) as exc:
        review.run()

    assert exc.value.code == 1

    captured = capsys.readouterr()
    assert "GitHub API failure: GitHub API request failed (HTTP 404)." in captured.err


def _run_controlled_review_pipeline(monkeypatch, model_result):
    head_sha = "a" * 40
    files = [
        {
            "filename": "src/one.py",
            "patch": (
                "@@ -1,2 +1,3 @@\n"
                " context one\n"
                "+added line\n"
                " context two\n"
                "@@ -10,1 +11,2 @@\n"
                "+added in second hunk\n"
                " context after second addition\n"
            ),
        },
        {
            "filename": "src/two.py",
            "patch": "@@ -0,0 +1 @@\n+new file line\n",
        },
    ]
    nvidia_calls = []
    review_posts = []

    class FakeResponse:
        def __init__(self, status_code, data=None):
            self.status_code = status_code
            self._data = data
            self.headers = {}

        def json(self):
            return self._data

    def fake_get(url, **kwargs):
        if url.endswith("/pulls/7/files"):
            page = kwargs["params"]["page"]
            return FakeResponse(200, files if page == 1 else [])
        if url.endswith("/pulls/7"):
            return FakeResponse(200, {"head": {"sha": head_sha}})
        pytest.fail(f"Unexpected GitHub GET URL: {url}")

    def fake_post(url, **kwargs):
        if url == review.llm_review.NVIDIA_API_URL:
            nvidia_calls.append((url, kwargs))
            return FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(model_result),
                            },
                        }
                    ]
                },
            )
        if url.endswith("/pulls/7/reviews"):
            review_posts.append(kwargs["json"])
            return FakeResponse(201)
        pytest.fail(f"Unexpected POST URL: {url}")

    monkeypatch.setattr(review.requests, "get", fake_get)
    monkeypatch.setattr(review.requests, "post", fake_post)
    monkeypatch.setattr(review.time, "sleep", lambda _: None)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("PR_NUMBER", "7")
    monkeypatch.setenv("GITHUB_TOKEN", "github-test-token")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-test-key")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

    review.main()
    return nvidia_calls, review_posts


@pytest.mark.parametrize(
    ("model_result", "expected_locations"),
    [
        (
            {
                "summary": "Two valid issues; invalid locations were ignored.",
                "issues": [
                    {
                        "path": "src/one.py",
                        "line": 2,
                        "category": "correctness",
                        "severity": "high",
                        "confidence": 0.9,
                        "body": "Valid issue on an added line.",
                    },
                    {
                        "path": "src/one.py",
                        "line": 1,
                        "category": "correctness",
                        "severity": "high",
                        "confidence": 0.9,
                        "body": "Context line is not eligible.",
                    },
                    {
                        "path": "src/one.py",
                        "line": 99,
                        "category": "correctness",
                        "severity": "high",
                        "confidence": 0.9,
                        "body": "Out-of-range line is not eligible.",
                    },
                    {
                        "path": "src/outside.py",
                        "line": 1,
                        "category": "correctness",
                        "severity": "high",
                        "confidence": 0.9,
                        "body": "A file outside the PR is not eligible.",
                    },
                    {
                        "path": "src/two.py",
                        "line": 1,
                        "category": "security",
                        "severity": "medium",
                        "confidence": 0.8,
                        "body": "Valid issue in the second file.",
                    },
                ],
            },
            [("src/one.py", 2), ("src/two.py", 1)],
        ),
        (
            {
                "summary": "No valid inline finding was available.",
                "issues": [
                    {
                        "path": "src/one.py",
                        "line": 1,
                        "category": "correctness",
                        "severity": "medium",
                        "confidence": 0.8,
                        "body": "This unchanged line must be dropped.",
                    }
                ],
            },
            [],
        ),
    ],
)
def test_controlled_end_to_end_review_pipeline(
    monkeypatch,
    model_result,
    expected_locations,
):
    nvidia_calls, review_posts = _run_controlled_review_pipeline(
        monkeypatch,
        model_result,
    )

    assert len(nvidia_calls) == 1
    request_url, request = nvidia_calls[0]
    assert request_url == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer nvidia-test-key"
    assert request["timeout"] == 120
    assert request["json"]["model"] == (
        "nvidia/nemotron-3-super-120b-a12b"
    )
    assert request["json"]["stream"] is False
    assert request["json"]["max_tokens"] == 8192
    assert len(review_posts) == 1

    comments = review_posts[0]["comments"]
    assert [(comment["path"], comment["line"]) for comment in comments] == (
        expected_locations
    )
    assert all(comment["side"] == "RIGHT" for comment in comments)


def test_pr_without_changed_files_skips_model_and_publishes_output(
    monkeypatch,
    tmp_path,
):
    output_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("PR_NUMBER", "7")
    monkeypatch.setenv("GITHUB_TOKEN", "github-test-token")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setattr(
        review,
        "get_pr_info",
        lambda *args: {"head": {"sha": "a" * 40}},
    )
    monkeypatch.setattr(review, "get_changed_files", lambda *args: [])
    monkeypatch.setattr(
        review.llm_review,
        "review_diff",
        lambda *args: pytest.fail("No model call expected"),
    )
    monkeypatch.setattr(
        review,
        "post_review",
        lambda *args: pytest.fail("No review should be posted"),
    )

    review.main()

    assert "has_bugs=false" in output_file.read_text(encoding="utf-8")