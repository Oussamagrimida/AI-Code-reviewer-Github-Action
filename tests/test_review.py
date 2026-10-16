import pytest
import requests
import review
import os
import subprocess
import sys

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

    with pytest.raises(requests.HTTPError, match=f"HTTP {status_code}"):
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

    with pytest.raises(requests.HTTPError, match=f"HTTP {status_code}"):
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

    with pytest.raises(requests.HTTPError, match=f"HTTP {status_code}"):
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

    with pytest.raises(requests.HTTPError, match="401 Unauthorized"):
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

    with pytest.raises(requests.Timeout, match="Request timed out"):
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

    with pytest.raises(requests.Timeout, match="Request timed out"):
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

    with pytest.raises(requests.ConnectionError, match="Connection failed"):
        review.post_review(
            "owner/repo",
            "12",
            "test-token",
            "abc123",
            {"summary": "Review complete.", "issues": []},
        )

    assert len(calls) == 1


def test_post_review_prints_inline_comment_count_on_success(monkeypatch, capsys):
    class FakeResponse:
        status_code = 201

        def raise_for_status(self):
            pass

    def fake_post(url, **kwargs):
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

    captured = capsys.readouterr()
    assert "Review posted: 1 inline comment(s)." in captured.out


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


def test_post_review_fallback_failure_raises(monkeypatch):
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

    with pytest.raises(requests.HTTPError, match="HTTP 500"):
        review.post_review(
            "owner/repo",
            "12",
            "test-token",
            "abc123",
            result,
        )

    assert len(calls) == 2
    assert calls[1]["comments"] == []


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
        raise requests.HTTPError("500 PR state unavailable")

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
    assert "Review failed: 500 PR state unavailable" in captured.err
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
    assert "EOF\nAdditional details." in output

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

    with pytest.raises(requests.HTTPError, match=message):
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
        requests.HTTPError,
        match="500 Internal Server Error",
    ):
        review.get_changed_files(
            "owner/repo",
            "123",
            "fake-token",
        )

    assert len(calls) == 1


@pytest.mark.parametrize("status_code", [502, 503, 504])
def test_get_pr_info_retries_on_transient_http_error(
    monkeypatch,
    status_code,
):
    calls = []
    delays = []

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"HTTP {self.status_code}")

        def json(self):
            return {"head": {"sha": "abc123"}}

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

    assert result == {"head": {"sha": "abc123"}}
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
            return {"head": {"sha": "abc123"}}

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

    assert result == {"head": {"sha": "abc123"}}
    assert len(calls) == 2
    assert delays == [5]


def test_get_pr_info_raises_on_timeout(monkeypatch):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        raise requests.Timeout("Request timed out")

    monkeypatch.setattr(review.requests, "get", fake_get)
    monkeypatch.setattr(review.time, "sleep", lambda _: None)

    with pytest.raises(
        requests.Timeout,
        match="Request timed out",
    ):
        review.get_pr_info(
            "owner/repo",
            "123",
            "fake-token",
        )

    assert len(calls) == 4


def test_get_changed_files_raises_on_timeout(monkeypatch):
    calls = []

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        raise requests.Timeout("Request timed out")

    monkeypatch.setattr(review.requests, "get", fake_get)
    monkeypatch.setattr(review.time, "sleep", lambda _: None)

    with pytest.raises(
        requests.Timeout,
        match="Request timed out",
    ):
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
        raise requests.HTTPError("404 Not Found")

    monkeypatch.setattr(review, "get_pr_info", fake_get_pr_info)

    with pytest.raises(SystemExit) as exc:
        review.run()

    assert exc.value.code == 1

    captured = capsys.readouterr()
    assert "Review failed: 404 Not Found" in captured.err