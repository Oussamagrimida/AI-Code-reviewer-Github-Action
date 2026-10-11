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


def test_get_changed_files_fetches_multiple_pages(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, data):
            self.data = data

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

def test_get_pr_info_raises_on_http_error(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            raise requests.HTTPError("404 Not Found")

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(review.requests, "get", fake_get)

    with pytest.raises(requests.HTTPError, match="404 Not Found"):
        review.get_pr_info(
            "owner/repo",
            "123",
            "fake-token",
        )

def test_get_changed_files_raises_on_http_error(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            raise requests.HTTPError("500 Internal Server Error")

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(review.requests, "get", fake_get)

    with pytest.raises(
        requests.HTTPError,
        match="500 Internal Server Error",
    ):
        review.get_changed_files(
            "owner/repo",
            "123",
            "fake-token",
        )

def test_get_pr_info_raises_on_timeout(monkeypatch):
    def fake_get(*args, **kwargs):
        raise requests.Timeout("Request timed out")

    monkeypatch.setattr(review.requests, "get", fake_get)

    with pytest.raises(
        requests.Timeout,
        match="Request timed out",
    ):
        review.get_pr_info(
            "owner/repo",
            "123",
            "fake-token",
        )

def test_get_changed_files_raises_on_timeout(monkeypatch):
    def fake_get(*args, **kwargs):
        raise requests.Timeout("Request timed out")

    monkeypatch.setattr(review.requests, "get", fake_get)

    with pytest.raises(
        requests.Timeout,
        match="Request timed out",
    ):
        review.get_changed_files(
            "owner/repo",
            "123",
            "fake-token",
        )



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