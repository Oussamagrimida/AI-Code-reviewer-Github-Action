import json
import pytest
import requests
import re
from requests.structures import CaseInsensitiveDict
from unittest.mock import patch, MagicMock

from llm_review import (
    NVIDIAAPIError,
    ReviewValidationError,
    _added_lines,
    _call_nvidia_with_retry,
    _parse_json_response,
    validate_review,
)
import llm_review


@pytest.fixture(autouse=True)
def nvidia_api_key(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("ENABLE_MULTI_PASS", "false")


PATCH = """@@ -1,4 +1,5 @@
 old line
+new line
 unchanged
@@ -10,2 +11,3 @@
-old
+replacement
+another line
"""


FILES = [
    {
        "path": "src/example.py",
        "patch": PATCH,
    }
]


def valid_review():
    return {
        "summary": (
            "The change is generally sound, "
            "with one correctness issue."
        ),
        "issues": [
            {
                "path": "src/example.py",
                "line": 2,
                "category": "correctness",
                "severity": "high",
                "confidence": 0.94,
                "body": (
                    "Handle the new value before "
                    "using it here."
                ),
                "explanation": (
                    "Handle the new value before "
                    "using it here."
                ),
                "impact": "",
                "remediation": "",
            }
        ],
    }


def test_added_lines_extracts_new_file_lines():
    assert _added_lines(PATCH) == {2, 11, 12}


def test_valid_review_is_accepted():
    assert (
        validate_review(valid_review(), FILES)
        == valid_review()
    )


def test_valid_findings_survive_invalid_references_across_files():
    files = [
        {
            "path": "src/one.py",
            "patch": (
                "@@ -1,3 +1,4 @@\n"
                " context\n"
                " +source text begins with a plus\n"
                "-deleted\n"
                "+added\n"
                " unchanged\n"
                "@@ -10 +11,2 @@\n"
                "+second hunk\n"
                "+third hunk line\n"
            ),
        },
        {
            "path": "src/two.py",
            "patch": "@@ -0,0 +1 @@\n+new file line\n",
        },
    ]
    assert _added_lines(files[0]["patch"]) == {3, 11, 12}

    def finding(path, line, body):
        return {
            "path": path,
            "line": line,
            "category": "correctness",
            "severity": "medium",
            "confidence": 0.8,
            "body": body,
        }

    review = {
        "summary": "The diff contains two actionable issues.",
        "issues": [
            finding("src/one.py", 3, "Issue on a changed line."),
            finding("src/one.py", 2, "Context line must not be posted."),
            finding("src/one.py", 4, "Unchanged line must not be posted."),
            finding("src/two.py", 1, "Issue in another changed file."),
            finding("src/outside.py", 1, "File is not part of this PR."),
        ],
    }

    validated = validate_review(review, files)
    assert validated["summary"] == review["summary"]
    assert [
        (issue["path"], issue["line"])
        for issue in validated["issues"]
    ] == [("src/one.py", 3), ("src/two.py", 1)]


def test_review_with_only_invalid_findings_keeps_safe_summary():
    review = valid_review()
    review["issues"][0]["line"] = 3

    assert validate_review(review, FILES) == {
        "summary": review["summary"],
        "issues": [],
    }


def test_invalid_finding_reports_added_line_rejection_reason(capsys):
    review = valid_review()
    review["issues"][0]["line"] = 3

    validate_review(review, FILES)

    assert (
        "Discarded invalid finding 1: "
        "Issue 1 does not point to an added line."
    ) in capsys.readouterr().out


def test_empty_issue_list_is_valid():
    review = {
        "summary": "No actionable issues found.",
        "issues": [],
    }

    assert validate_review(review, FILES) == review


def test_malformed_json_raises():
    with pytest.raises(
        ReviewValidationError,
        match="not valid JSON",
    ):
        _parse_json_response("{not-json}")


def test_non_object_json_raises():
    with pytest.raises(
        ReviewValidationError,
        match="JSON object",
    ):
        _parse_json_response("[]")


def test_missing_required_issue_field_is_discarded(capsys):
    review = valid_review()

    del review["issues"][0]["confidence"]

    assert validate_review(review, FILES)["issues"] == []
    assert (
        "Issue 1 is missing required field(s): confidence."
        in capsys.readouterr().out
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("category", "bug"),
        ("severity", "bug"),
    ],
)
def test_invalid_enum_is_discarded(field, value):
    review = valid_review()

    review["issues"][0][field] = value

    assert validate_review(review, FILES)["issues"] == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("category", []),
        ("category", {}),
        ("severity", []),
        ("severity", {}),
    ],
)
def test_unhashable_enum_values_are_discarded(field, value):
    review = valid_review()
    review["issues"][0][field] = value

    assert validate_review(review, FILES)["issues"] == []


def test_invalid_confidence_is_discarded():
    review = valid_review()

    review["issues"][0]["confidence"] = 1.1

    assert validate_review(review, FILES)["issues"] == []


def test_unknown_file_is_discarded():
    review = valid_review()

    review["issues"][0]["path"] = "src/missing.py"

    assert validate_review(review, FILES)["issues"] == []


def test_control_characters_in_path_are_discarded():
    review = valid_review()
    review["issues"][0]["path"] = "src/example.py\n- secret.py"
    files = [{"path": review["issues"][0]["path"], "patch": PATCH}]

    assert validate_review(review, files)["issues"] == []


def test_context_line_is_discarded():
    review = valid_review()

    review["issues"][0]["line"] = 3

    assert validate_review(review, FILES)["issues"] == []


def test_added_line_anchor_does_not_ground_unchanged_permission_claim():
    files = [
        {
            "path": "app/routes/items.py",
            "patch": (
                "@@ -20,4 +20,5 @@\n"
                " def update_item(item_id, payload, current_user, db):\n"
                "     if item.owner_id != current_user.id:\n"
                "         raise HTTPException(status_code=403)\n"
                "+    item.name = payload.name\n"
                "     return item\n"
            ),
        }
    ]
    review = valid_review()
    finding = review["issues"][0]
    finding.update(
        {
            "path": "app/routes/items.py",
            "line": 23,
            "category": "security",
            "explanation": (
                "The update_item permission check is inverted by this change."
            ),
        }
    )

    assert llm_review._added_lines(files[0]["patch"]) == {23}
    # Line validation accepts this anchor; it cannot establish the claim's truth.
    assert validate_review(review, files)["issues"] == [finding]


def test_deleted_line_is_discarded():
    review = valid_review()

    review["issues"][0]["line"] = 10

    assert validate_review(review, FILES)["issues"] == []


def test_unexpected_issue_field_is_discarded(capsys):
    review = valid_review()

    review["issues"][0]["extra"] = "unexpected"

    assert validate_review(review, FILES)["issues"] == []
    assert (
        "Issue 1 has unexpected field(s): extra."
        in capsys.readouterr().out
    )


def test_unexpected_top_level_field_is_rejected():
    review = valid_review()

    review["raw"] = "model output"

    with pytest.raises(
        ReviewValidationError,
        match="top-level",
    ):
        validate_review(review, FILES)


def test_invalid_summary_is_rejected():
    review = valid_review()

    review["summary"] = ""

    with pytest.raises(
        ReviewValidationError,
        match="summary",
    ):
        validate_review(review, FILES)

def test_added_lines_multiple_hunks():
    patch = (
        "@@ -10,3 +10,4 @@\n"
        " def hello():\n"
        " x = 1\n"
        "+y = 2\n"
        "@@ -50,3 +51,5 @@\n"
        " def goodbye():\n"
        "+a = 10\n"
        "+b = 20\n"
        " return True\n"
    )

    files = [
        {
            "path": "example.py",
            "patch": patch,
        }
    ]

    assert llm_review._added_lines(patch) == {12, 52, 53}
def test_added_lines_multiple_hunks_with_deleted_and_context_lines():
    patch = (
        "@@ -10,4 +10,5 @@\n"
        " def hello():\n"
        " x = 1\n"
        "-old = 2\n"
        "+new = 2\n"
        " return x\n"
        "@@ -30,4 +31,6 @@\n"
        " def goodbye():\n"
        "-old_a = 10\n"
        "+new_a = 10\n"
        "+new_b = 20\n"
        " return True\n"
    )

    assert llm_review._added_lines(patch) == {12, 32, 33}

def test_diff_is_split_into_chunks():
    files = [
        {
            "path": "a.py",
            "patch": "a" * 100,
        },
        {
            "path": "b.py",
            "patch": "b" * 100,
        },
        {
            "path": "c.py",
            "patch": "c" * 100,
        },
    ]

    chunks = llm_review._split_diff_into_chunks(
        files,
        max_chars=250,
    )

    assert len(chunks) == 2
    assert chunks[0][0]["path"] == "a.py"
    assert chunks[1][0]["path"] == "c.py"

def test_review_diff_processes_multiple_chunks(monkeypatch):
    calls = []

    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"summary":"First chunk reviewed.",'
                            '"issues":[]}'
                        )
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"summary":"Second chunk reviewed.",'
                            '"issues":[]}'
                        )
                    }
                }
            ]
        },
    ]

    def fake_call(payload, headers):
        calls.append(payload)
        return responses[len(calls) - 1]

    monkeypatch.setattr(llm_review, "_call_nvidia_with_retry", fake_call)

    files = [
        {
            "path": "a.py",
            "patch": "a" * 100,
        },
        {
            "path": "b.py",
            "patch": "b" * 100,
        },
        {
            "path": "c.py",
            "patch": "c" * 100,
        },
    ]

    result = llm_review.review_diff(files)

    assert len(calls) == 2
    assert len(result["issues"]) == 0
    assert "First chunk reviewed." in result["summary"]
    assert "Second chunk reviewed." in result["summary"]


def test_review_diff_merges_and_deduplicates_chunk_findings(monkeypatch):
    calls = []
    first_finding = {
        "path": "src/shared.py",
        "line": 1,
        "category": "correctness",
        "severity": "high",
        "confidence": 0.9,
        "body": "Handle this changed value safely.",
        "explanation": "Handle this changed value safely.",
        "impact": "",
        "remediation": "",
    }
    second_finding = {
        "path": "src/last.py",
        "line": 1,
        "category": "security",
        "severity": "medium",
        "confidence": 0.8,
        "body": "Validate this changed input.",
        "explanation": "Validate this changed input.",
        "impact": "",
        "remediation": "",
    }
    reviews = [
        {
            "summary": "First chunk summary.",
            "issues": [first_finding],
        },
        {
            "summary": "Second chunk summary.",
            "issues": [second_finding, first_finding],
        },
    ]

    def fake_call(payload, headers):
        calls.append(payload)
        review = reviews[len(calls) - 1]
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(review),
                    }
                }
            ]
        }

    monkeypatch.setattr(
        llm_review,
        "_call_nvidia_with_retry",
        fake_call,
    )

    files = [
        {
            "path": "src/shared.py",
            "patch": "@@ -1 +1 @@\n-old\n+new_shared\n",
        },
        {
            "path": "src/other.py",
            "patch": "@@ -1 +1 @@\n-old\n+new_other\n",
        },
        {
            "path": "src/shared.py",
            "patch": "@@ -1 +1 @@\n-old\n+new_shared_again\n",
        },
        {
            "path": "src/last.py",
            "patch": "@@ -1 +1 @@\n-old\n+new_last\n",
        },
    ]

    result = llm_review.review_diff(files)

    assert len(calls) == 2
    assert result["summary"] == (
        "First chunk summary. Second chunk summary."
    )
    assert result["issues"] == [first_finding, second_finding]


def test_review_diff_keeps_finding_only_valid_in_its_chunk(monkeypatch):
    calls = []

    def fake_call(payload, headers):
        calls.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": "Review completed.",
                                "issues": [
                                    {
                                        "path": "src/shared.py",
                                        "line": 2,
                                        "category": "correctness",
                                        "severity": "medium",
                                        "confidence": 0.9,
                                        "body": (
                                            "This line is only in "
                                            "another chunk."
                                        ),
                                    }
                                ],
                            }
                        ),
                    }
                }
            ]
        }

    monkeypatch.setattr(
        llm_review,
        "_call_nvidia_with_retry",
        fake_call,
    )

    files = [
        {
            "path": "src/shared.py",
            "patch": "@@ -1 +1 @@\n-old\n+first_chunk_line\n",
        },
        {
            "path": "src/other.py",
            "patch": "@@ -1 +1 @@\n-old\n+other_chunk_line\n",
        },
        {
            "path": "src/shared.py",
            "patch": "@@ -1,0 +2 @@\n+second_chunk_line\n",
        },
    ]
    chunks = llm_review._split_diff_into_chunks(files)

    assert len(chunks) == 2
    assert llm_review._added_lines(chunks[0][0]["patch"]) == {1}
    assert llm_review._added_lines(chunks[1][0]["patch"]) == {2}

    result = llm_review.review_diff(files)

    assert len(calls) == 2
    assert [
        (issue["path"], issue["line"])
        for issue in result["issues"]
    ] == [("src/shared.py", 2)]


def test_duplicate_issues_are_removed():
    issues = [
        {
            "path": "a.py",
            "line": 10,
            "category": "correctness",
            "severity": "high",
            "confidence": 0.9,
            "body": "Fix this bug.",
        },
        {
            "path": "a.py",
            "line": 10,
            "category": "correctness",
            "severity": "high",
            "confidence": 0.8,
            "body": "Fix this bug.",
        },
        {
            "path": "a.py",
            "line": 20,
            "category": "security",
            "severity": "medium",
            "confidence": 0.9,
            "body": "Validate this input.",
        },
    ]

    result = llm_review._deduplicate_issues(issues)

    assert len(result) == 2
    assert result[0]["line"] == 10
    assert result[1]["line"] == 20

def test_all_valid_severities_are_accepted():
    for severity in llm_review.VALID_SEVERITIES:
        review = valid_review()
        review["issues"][0]["severity"] = severity

        assert validate_review(review, FILES) == review


def test_invalid_severity_is_discarded():
    review = valid_review()
    review["issues"][0]["severity"] = "urgent"

    assert validate_review(review, FILES)["issues"] == []


@pytest.mark.parametrize(
    "severity",
    [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
        "INFO",
        "",
        None,
        1,
    ],
)
def test_invalid_severity_values_are_discarded(severity):
    review = valid_review()
    review["issues"][0]["severity"] = severity

    assert validate_review(review, FILES)["issues"] == []

def test_system_prompt_contains_severity_definitions():
    assert "Severity definitions:" in llm_review.SYSTEM_PROMPT
    assert "critical:" in llm_review.SYSTEM_PROMPT
    assert "high:" in llm_review.SYSTEM_PROMPT
    assert "medium:" in llm_review.SYSTEM_PROMPT
    assert "low:" in llm_review.SYSTEM_PROMPT
    assert "info:" in llm_review.SYSTEM_PROMPT

@pytest.mark.parametrize(
    "confidence",
    [0.0, 0.5, 1.0],
)
def test_valid_confidence_values_are_accepted(confidence):
    review = valid_review()
    review["issues"][0]["confidence"] = confidence

    assert validate_review(review, FILES) == review


@pytest.mark.parametrize(
    "confidence",
    [-0.01, 1.01, 2, -1],
)
def test_out_of_range_confidence_is_discarded(confidence):
    review = valid_review()
    review["issues"][0]["confidence"] = confidence

    assert validate_review(review, FILES)["issues"] == []


@pytest.mark.parametrize(
    "confidence",
    [True, False, "0.9", "high", None, [], {}],
)
def test_invalid_confidence_types_are_discarded(confidence):
    review = valid_review()
    review["issues"][0]["confidence"] = confidence

    assert validate_review(review, FILES)["issues"] == []

def test_system_prompt_contains_confidence_guidance():
    assert "Confidence scoring:" in llm_review.SYSTEM_PROMPT
    assert "between 0 and 1" in llm_review.SYSTEM_PROMPT
    assert "confidence represents how certain" in llm_review.SYSTEM_PROMPT
    assert "Report issues only" in llm_review.SYSTEM_PROMPT
    assert "Never report issues on unchanged/context lines" in llm_review.SYSTEM_PROMPT

def test_markdown_json_code_fence_is_accepted():
    raw = """```json
{"summary":"No issues found.","issues":[]}
```"""

    result = _parse_json_response(raw)

    assert result == {
        "summary": "No issues found.",
        "issues": [],
    }


def test_plain_markdown_code_fence_is_accepted():
    raw = """```
{"summary":"No issues found.","issues":[]}
```"""

    result = _parse_json_response(raw)

    assert result == {
        "summary": "No issues found.",
        "issues": [],
    }


def test_empty_response_is_rejected():
    with pytest.raises(
        ReviewValidationError,
        match="response is empty",
    ):
        _parse_json_response("")


def test_whitespace_only_response_is_rejected():
    with pytest.raises(
        ReviewValidationError,
        match="response is empty",
    ):
        _parse_json_response("   ")


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "{",
        '{"summary":}',
        '{"issues": [}',
    ],
)
def test_malformed_json_is_rejected(raw):
    with pytest.raises(
        ReviewValidationError,
        match="not valid JSON",
    ):
        _parse_json_response(raw)

def test_missing_summary_is_rejected():
    review = valid_review()
    del review["summary"]

    with pytest.raises(
        ReviewValidationError,
        match="summary",
    ):
        validate_review(review, FILES)


def test_missing_issues_is_rejected():
    review = valid_review()
    del review["issues"]

    with pytest.raises(
        ReviewValidationError,
        match="issues",
    ):
        validate_review(review, FILES)


def test_issues_must_be_a_list():
    review = valid_review()
    review["issues"] = {}

    with pytest.raises(
        ReviewValidationError,
        match="issues.*array",
    ):
        validate_review(review, FILES)


def test_non_object_issue_is_discarded():
    review = valid_review()
    review["issues"] = ["invalid issue"]

    assert validate_review(review, FILES)["issues"] == []

@pytest.mark.parametrize(
    "field,value",
    [
        ("path", 123),
        ("line", "2"),
        ("category", 123),
        ("severity", 123),
        ("body", 123),
    ],
)
def test_invalid_issue_field_types_are_discarded(field, value):
    review = valid_review()
    review["issues"][0][field] = value

    assert validate_review(review, FILES)["issues"] == []

def test_nvidia_retry_succeeds_after_temporary_failure(monkeypatch):
    calls = []

    responses = [
        requests.exceptions.Timeout("timeout"),
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"summary":"Success.","issues":[]}'
                        )
                    }
                }
            ]
        },
    ]

    def fake_post(*args, **kwargs):
        calls.append(1)
        result = responses[len(calls) - 1]

        if isinstance(result, Exception):
            raise result

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return result

        return FakeResponse()

    monkeypatch.setattr(
        llm_review.requests,
        "post",
        fake_post,
    )

    monkeypatch.setattr(
        llm_review.time,
        "sleep",
        lambda _: None,
    )

    result = _call_nvidia_with_retry(
        {"model": "test"},
        {"Authorization": "Bearer test"},
        max_attempts=2,
    )

    assert result["choices"][0]["message"]["content"] == (
        '{"summary":"Success.","issues":[]}'
    )

    assert len(calls) == 2


def test_nvidia_connection_error_retries_then_succeeds(monkeypatch):
    calls = []
    delays = []
    success_result = {
        "choices": [
            {
                "message": {
                    "content": '{"summary":"Recovered.","issues":[]}'
                }
            }
        ]
    }

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return success_result

    def fake_post(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.exceptions.ConnectionError(
                "Connection interrupted"
            )
        return FakeResponse()

    monkeypatch.setattr(llm_review.requests, "post", fake_post)
    monkeypatch.setattr(llm_review.time, "sleep", delays.append)

    result = _call_nvidia_with_retry(
        {"model": "test"},
        {},
        max_attempts=3,
    )

    assert result == success_result
    assert len(calls) == 2
    assert delays == [5]


def test_nvidia_connection_error_exhaustion_raises_and_sleeps(
    monkeypatch,
):
    calls = []
    delays = []

    def fake_post(*args, **kwargs):
        calls.append(1)
        raise requests.exceptions.ConnectionError(
            "Connection interrupted"
        )

    monkeypatch.setattr(llm_review.requests, "post", fake_post)
    monkeypatch.setattr(llm_review.time, "sleep", delays.append)

    with pytest.raises(NVIDIAAPIError, match="connection error"):
        _call_nvidia_with_retry(
            {},
            {},
            max_attempts=3,
        )

    assert len(calls) == 3
    assert delays == [5, 10]


def test_nvidia_retryable_http_error_exhaustion_raises_and_sleeps(
    monkeypatch,
):
    calls = []
    delays = []

    class FakeResponse:
        status_code = 503
        reason = "Service Unavailable"

    def fake_post(*args, **kwargs):
        calls.append(1)
        return FakeResponse()

    monkeypatch.setattr(llm_review.requests, "post", fake_post)
    monkeypatch.setattr(llm_review.time, "sleep", delays.append)

    with pytest.raises(NVIDIAAPIError, match="HTTP 503"):
        _call_nvidia_with_retry(
            {},
            {},
            max_attempts=3,
        )

    assert len(calls) == 3
    assert delays == [5, 10]



@pytest.mark.parametrize(
    "status_code",
    [429, 500, 502, 503, 504],
)
def test_nvidia_retries_server_errors(
    monkeypatch,
    status_code,
):
    calls = []

    class FakeResponse:
        def __init__(self, status):
            self.status_code = status
            self.reason = "Server Error"

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(
                f"{self.status_code} Server Error"
            )

    def fake_post(*args, **kwargs):
        calls.append(1)

        if len(calls) == 1:
            return FakeResponse(status_code)

        class SuccessResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"summary":"OK.","issues":[]}'
                                )
                            }
                        }
                    ]
                }

        return SuccessResponse()

    monkeypatch.setattr(
        llm_review.requests,
        "post",
        fake_post,
    )

    monkeypatch.setattr(
        llm_review.time,
        "sleep",
        lambda _: None,
    )

    result = _call_nvidia_with_retry(
        {},
        {},
        max_attempts=2,
    )

    assert len(calls) == 2
    assert result["choices"][0]["message"]["content"] == (
        '{"summary":"OK.","issues":[]}'
    )


def test_nvidia_gone_response_is_not_retried(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 410

    monkeypatch.setattr(
        llm_review.requests,
        "post",
        lambda *args, **kwargs: calls.append(1) or FakeResponse(),
    )

    with pytest.raises(NVIDIAAPIError, match="HTTP 410"):
        _call_nvidia_with_retry({}, {}, max_attempts=4)

    assert len(calls) == 1


def test_nvidia_pending_response_is_reported_as_api_failure(monkeypatch):
    class FakeResponse:
        status_code = 202

    monkeypatch.setattr(
        llm_review.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(NVIDIAAPIError, match="asynchronous HTTP 202"):
        _call_nvidia_with_retry({}, {})


def test_nvidia_rate_limit_honors_retry_after(monkeypatch):
    calls = []
    delays = []

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code
            self.headers = CaseInsensitiveDict({"retry-after": "3"})

        def json(self):
            return {"choices": []}

    def fake_post(*args, **kwargs):
        calls.append(1)
        return FakeResponse(429 if len(calls) == 1 else 200)

    monkeypatch.setattr(llm_review.requests, "post", fake_post)
    monkeypatch.setattr(llm_review.time, "sleep", delays.append)

    _call_nvidia_with_retry({}, {}, max_attempts=2)
    assert len(calls) == 2
    assert delays == [3]


def test_nvidia_rate_limit_retry_after_is_capped(monkeypatch):
    class FakeResponse:
        status_code = 429
        headers = CaseInsensitiveDict({"Retry-After": "300"})

    monkeypatch.setattr(
        llm_review.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(),
    )
    delays = []
    monkeypatch.setattr(llm_review.time, "sleep", delays.append)

    with pytest.raises(NVIDIAAPIError, match="rate limit"):
        _call_nvidia_with_retry({}, {}, max_attempts=2)

    assert delays == [60]


def test_nvidia_malformed_retry_after_uses_bounded_backoff(monkeypatch):
    calls = []
    delays = []

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code
            self.headers = CaseInsensitiveDict(
                {"Retry-After": "not-a-number"}
            )

        def json(self):
            return {"choices": []}

    def fake_post(*args, **kwargs):
        calls.append(1)
        return FakeResponse(429 if len(calls) == 1 else 200)

    monkeypatch.setattr(llm_review.requests, "post", fake_post)
    monkeypatch.setattr(llm_review.time, "sleep", delays.append)

    _call_nvidia_with_retry({}, {}, max_attempts=2)

    assert len(calls) == 2
    assert delays == [5]


def test_nvidia_malformed_http_json_has_safe_error(monkeypatch):
    class FakeResponse:
        status_code = 200

        def json(self):
            raise ValueError("private response body")

    monkeypatch.setattr(
        llm_review.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(),
    )

    with pytest.raises(NVIDIAAPIError, match="invalid JSON") as error:
        _call_nvidia_with_retry({}, {})
    assert "private response body" not in str(error.value)


def test_nvidia_retry_exhaustion_raises(monkeypatch):
    calls = []

    def fake_post(*args, **kwargs):
        calls.append(1)
        raise requests.exceptions.Timeout("timeout")

    monkeypatch.setattr(
        llm_review.requests,
        "post",
        fake_post,
    )

    monkeypatch.setattr(
        llm_review.time,
        "sleep",
        lambda _: None,
    )

    with pytest.raises(NVIDIAAPIError, match="network timeout"):
        _call_nvidia_with_retry(
            {},
            {},
            max_attempts=3,
        )

    assert len(calls) == 3

def test_nvidia_does_not_retry_client_errors(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 401
        reason = "Unauthorized"

        def raise_for_status(self):
            raise requests.exceptions.HTTPError(
                "401 Client Error"
            )

    def fake_post(*args, **kwargs):
        calls.append(1)
        return FakeResponse()

    monkeypatch.setattr(
        llm_review.requests,
        "post",
        fake_post,
    )

    with pytest.raises(NVIDIAAPIError, match="HTTP 401"):
        _call_nvidia_with_retry(
            {},
            {},
            max_attempts=4,
        )

    assert len(calls) == 1


def test_large_single_file_diff_is_split_into_chunks():
    patch = (
        "@@ -1,6 +1,6 @@\n"
        + "".join(
            f" line_{i}\n+added_line_{i}\n"
            for i in range(1, 7)
        )
    )

    files = [{"path": "src/large.py", "patch": patch}]

    chunks = llm_review._split_diff_into_chunks(
        files,
        max_chars=100,
        max_files_per_chunk=2,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk) == 1
        assert chunk[0]["path"] == "src/large.py"
        assert chunk[0]["patch"]

def test_large_single_file_chunks_preserve_added_line_numbers():
    patch = (
        "@@ -1,12 +1,12 @@\n"
        + "".join(
            f" line_{i}\n+added_line_{i}\n"
            for i in range(1, 7)
        )
    )

    files = [{"path": "src/large.py", "patch": patch}]

    chunks = llm_review._split_diff_into_chunks(
        files,
        max_chars=100,
        max_files_per_chunk=2,
    )

    assert len(chunks) > 1

    added_lines = set()

    for chunk in chunks:
        added_lines.update(
            llm_review._added_lines(chunk[0]["patch"])
        )

    assert added_lines == {2, 4, 6, 8, 10, 12}

def test_large_hunk_preserves_each_added_line_exactly_once():
    from llm_review import _split_diff_into_chunks

    added_lines = [
        f"+added_line_{i:02d}_" + "x" * 40 + "\n"
        for i in range(10)
    ]

    context_lines = [
        f" context_line_{i:02d}_" + "y" * 40 + "\n"
        for i in range(10)
    ]

    body = []
    for context, added in zip(context_lines, added_lines):
        body.extend([context, added])

    patch = "@@ -1,10 +1,20 @@\n" + "".join(body)

    chunks = _split_diff_into_chunks(
        [{"path": "example.py", "patch": patch}],
        max_chars=300,
        max_files_per_chunk=2,
    )

    actual_added_lines = []

    for chunk in chunks:
        for file in chunk:
            for line in file["patch"].splitlines():
                if line.startswith("+") and not line.startswith("+++"):
                    actual_added_lines.append(line + "\n")

    assert actual_added_lines == added_lines

def test_large_hunk_with_malformed_header_stays_within_chunk_limit():
    from llm_review import _split_diff_into_chunks

    patch = "@@ invalid header @@\n" + ("+added_line\n" * 50)

    chunks = _split_diff_into_chunks(
        [{"path": "example.py", "patch": patch}],
        max_chars=100,
        max_files_per_chunk=2,
    )

    assert chunks
    assert all(
        len(f"--- FILE: {file['path']} ---\n{file['patch']}") + 2 <= 100
        for chunk in chunks
        for file in chunk
    )


def test_large_hunk_reconstructed_headers_match_body_counts():
    import re

    from llm_review import _split_diff_into_chunks

    body = []
    for i in range(10):
        body.append(f" context_{i}\n")
        body.append(f"-removed_{i}\n")
        body.append(f"+added_{i}\n")
        body.append(f"+added_extra_{i}\n")

    patch = "@@ -1,20 +1,30 @@\n" + "".join(body)

    chunks = _split_diff_into_chunks(
        [{"path": "example.py", "patch": patch}],
        max_chars=300,
        max_files_per_chunk=2,
    )

    header_re = re.compile(
        r"^@@ -\d+,(\d+) \+\d+,(\d+) @@"
    )

    for chunk in chunks:
        for file in chunk:
            lines = file["patch"].splitlines()
            header = header_re.match(lines[0])

            assert header is not None

            old_count = int(header.group(1))
            new_count = int(header.group(2))

            body_lines = lines[1:]
            actual_old_count = sum(
                line.startswith((" ", "-")) for line in body_lines
            )
            actual_new_count = sum(
                line.startswith((" ", "+")) for line in body_lines
            )

            assert old_count == actual_old_count
            assert new_count == actual_new_count

def test_large_hunk_preserves_original_added_line_numbers():
    from llm_review import _added_lines, _split_diff_into_chunks

    body = []
    for i in range(10):
        body.append(f" context_{i}\n")
        body.append(f"+added_{i}\n")

    patch = "@@ -1,10 +1,20 @@\n" + "".join(body)

    chunks = _split_diff_into_chunks(
        [{"path": "example.py", "patch": patch}],
        max_chars=300,
        max_files_per_chunk=2,
    )

    actual_lines = []

    for chunk in chunks:
        for file in chunk:
            actual_lines.extend(_added_lines(file["patch"]))

    assert actual_lines == list(range(2, 22, 2))


def test_split_chunks_have_valid_hunk_headers():
    patch = (
        "@@ -1,12 +1,12 @@\n"
        + "".join(
            f" line_{i}\n+added_line_{i}\n"
            for i in range(1, 7)
        )
    )

    files = [{"path": "src/large.py", "patch": patch}]

    chunks = llm_review._split_diff_into_chunks(
        files,
        max_chars=100,
        max_files_per_chunk=2,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        chunk_patch = chunk[0]["patch"]
        lines = chunk_patch.splitlines()

        headers = [
            line for line in lines
            if line.startswith("@@ ")
        ]

        assert headers

        for header in headers:
            match = llm_review.HUNK_HEADER_RE.match(header)
            assert match is not None

def test_split_chunk_hunk_counts_match_content():
    patch = (
        "@@ -1,12 +1,12 @@\n"
        + "".join(
            f" line_{i}\n+added_line_{i}\n"
            for i in range(1, 7)
        )
    )

    files = [{"path": "src/large.py", "patch": patch}]

    chunks = llm_review._split_diff_into_chunks(
        files,
        max_chars=100,
        max_files_per_chunk=2,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        for line in chunk[0]["patch"].splitlines():
            if not line.startswith("@@ "):
                continue

            match = re.match(
                r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@",
                line,
            )
            assert match is not None

            old_count = int(match.group(1) or 1)
            new_count = int(match.group(2) or 1)

            chunk_lines = chunk[0]["patch"].splitlines()
            header_index = chunk_lines.index(line)
            body = chunk_lines[header_index + 1:]

            actual_old = sum(
                1 for item in body
                if item.startswith((" ", "-"))
            )
            actual_new = sum(
                1 for item in body
                if item.startswith((" ", "+"))
            )

            assert actual_old == old_count
            assert actual_new == new_count

def test_split_chunk_review_validation_uses_correct_line_numbers():
    patch = (
        "@@ -1,12 +1,12 @@\n"
        + "".join(
            f" line_{i}\n+added_line_{i}\n"
            for i in range(1, 7)
        )
    )

    files = [{"path": "src/large.py", "patch": patch}]

    chunks = llm_review._split_diff_into_chunks(
        files,
        max_chars=100,
        max_files_per_chunk=2,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        chunk_file = chunk[0]
        added_lines = llm_review._added_lines(chunk_file["patch"])

        for line_number in added_lines:
            review = {
                "summary": "Review completed",
                "issues": [
                    {
                        "path": "src/large.py",
                        "line": line_number,
                        "category": "correctness",
                        "severity": "medium",
                        "confidence": 0.9,
                        "body": "Potential issue on an added line.",
                    }
                ],
            }

            llm_review.validate_review(review, [chunk_file])

def test_split_chunk_review_rejects_context_line():
    patch = (
        "@@ -1,12 +1,12 @@\n"
        + "".join(
            f" line_{i}\n+added_line_{i}\n"
            for i in range(1, 7)
        )
    )

    files = [{"path": "src/large.py", "patch": patch}]

    chunks = llm_review._split_diff_into_chunks(
        files,
        max_chars=100,
        max_files_per_chunk=2,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        chunk_file = chunk[0]
        added_lines = llm_review._added_lines(chunk_file["patch"])

        context_lines = {
            line_number
            for line_number in range(1, 13)
            if line_number not in added_lines
        }

        if context_lines:
            review = {
                "summary": "Review completed",
                "issues": [
                    {
                        "path": "src/large.py",
                        "line": min(context_lines),
                        "category": "correctness",
                        "severity": "medium",
                        "confidence": 0.9,
                        "body": "This finding should be rejected.",
                    }
                ],
            }

            assert llm_review.validate_review(
                review,
                [chunk_file],
            )["issues"] == []
            return

    pytest.fail("No context line was found to test")

def test_large_diff_preserves_all_added_lines_across_chunks():
    patch = (
        "@@ -1,12 +1,12 @@\n"
        + "".join(
            f" line_{i}\n+added_line_{i}\n"
            for i in range(1, 7)
        )
    )

    files = [{"path": "src/large.py", "patch": patch}]

    chunks = llm_review._split_diff_into_chunks(
        files,
        max_chars=100,
        max_files_per_chunk=2,
    )

    original_added_lines = llm_review._added_lines(patch)
    combined_added_lines = set()
    total_added_lines = 0

    for chunk in chunks:
        chunk_added_lines = llm_review._added_lines(
            chunk[0]["patch"]
        )
        total_added_lines += len(chunk_added_lines)
        combined_added_lines.update(chunk_added_lines)

    assert combined_added_lines == original_added_lines
    assert total_added_lines == len(original_added_lines)

def test_large_diff_preserves_lines_with_multiple_hunks_and_deletions():
    patch = (
        "@@ -1,4 +1,4 @@\n"
        " context_one\n"
        "-old_one\n"
        "+new_one\n"
        " context_two\n"
        "@@ -20,4 +20,5 @@\n"
        " context_three\n"
        "-old_two\n"
        "+new_two\n"
        "+new_three\n"
        " context_four\n"
    )

    files = [{"path": "src/large.py", "patch": patch}]

    original_added_lines = llm_review._added_lines(patch)

    chunks = llm_review._split_diff_into_chunks(
        files,
        max_chars=100,
        max_files_per_chunk=2,
    )

    combined_added_lines = set()
    total_added_lines = 0

    for chunk in chunks:
        chunk_added_lines = llm_review._added_lines(
            chunk[0]["patch"]
        )
        total_added_lines += len(chunk_added_lines)
        combined_added_lines.update(chunk_added_lines)

    assert combined_added_lines == original_added_lines
    assert total_added_lines == len(original_added_lines)


def test_review_diff_with_no_patches_does_not_call_api(monkeypatch):
    def unexpected_api_call(*args, **kwargs):
        pytest.fail("API should not be called when there are no patches")

    monkeypatch.setattr(
        llm_review,
        "_call_nvidia_with_retry",
        unexpected_api_call,
    )

    result = llm_review.review_diff([])

    assert result == {
        "summary": "No reviewable diff content was available.",
        "issues": [],
    }


@pytest.mark.parametrize(
    "response_json",
    [
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": None}}]},
    ],
)
def test_review_diff_rejects_malformed_api_responses(
    monkeypatch,
    response_json,
):
    monkeypatch.setattr(
        llm_review,
        "_call_nvidia_with_retry",
        lambda *args, **kwargs: response_json,
    )

    files = [
        {
            "path": "src/example.py",
            "patch": "@@ -1 +1 @@\n-old\n+new\n",
        }
    ]

    with pytest.raises(ReviewValidationError):
        llm_review.review_diff(files)


def test_review_diff_rejects_truncated_completion(monkeypatch):
    monkeypatch.setattr(
        llm_review,
        "_call_nvidia_with_retry",
        lambda *args, **kwargs: {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": '{"summary":"partial'},
                }
            ]
        },
    )

    with pytest.raises(ReviewValidationError, match="truncated"):
        llm_review.review_diff(
            [{"path": "src/example.py", "patch": "@@ -1 +1 @@\n+new\n"}]
        )


def test_review_diff_requires_nvidia_api_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY")
    monkeypatch.setattr(
        llm_review,
        "_call_nvidia_with_retry",
        lambda *args, **kwargs: pytest.fail("No request expected"),
    )

    with pytest.raises(NVIDIAAPIError, match="NVIDIA_API_KEY"):
        llm_review.review_diff(
            [{"path": "src/example.py", "patch": "@@ -1 +1 @@\n+new\n"}]
        )


@pytest.mark.parametrize(
    "content",
    [
        "",
        "   ",
        None,
        123,
        [],
        {},
    ],
)
def test_review_diff_rejects_invalid_api_content(
    monkeypatch,
    content,
):
    monkeypatch.setattr(
        llm_review,
        "_call_nvidia_with_retry",
        lambda *args, **kwargs: {
            "choices": [
                {
                    "message": {
                        "content": content,
                    }
                }
            ]
        },
    )

    files = [
        {
            "path": "src/example.py",
            "patch": "@@ -1 +1 @@\n-old\n+new\n",
        }
    ]

    with pytest.raises(ReviewValidationError):
        llm_review.review_diff(files)


def test_semantic_dedup_exact_duplicates():
    issue1 = {
        "path": "calculator.py",
        "line": 42,
        "category": "correctness",
        "severity": "high",
        "confidence": 0.95,
        "body": "Division by zero is not handled when b == 0.",
        "explanation": "Division by zero is not handled when b == 0.",
        "impact": "Triggers ZeroDivisionError",
        "remediation": "Check if b == 0 before dividing.",
    }
    issue2 = dict(issue1)
    deduped = llm_review._semantic_deduplicate_issues([issue1, issue2])
    assert len(deduped) == 1
    assert deduped[0]["path"] == "calculator.py"
    assert deduped[0]["line"] == 42


def test_semantic_dedup_same_defect_different_wording():
    issue1 = {
        "path": "calculator.py",
        "line": 42,
        "category": "correctness",
        "severity": "medium",
        "confidence": 0.85,
        "body": "Potential division by zero when divisor parameter is zero.",
        "explanation": "Potential division by zero when divisor parameter is zero.",
        "impact": "ZeroDivisionError crashes execution.",
        "remediation": "Add guard: if divisor == 0: raise ValueError.",
    }
    issue2 = {
        "path": "calculator.py",
        "line": 42,
        "category": "correctness",
        "severity": "high",
        "confidence": 0.92,
        "body": "ZeroDivisionError occurs if divisor argument equals zero.",
        "explanation": "ZeroDivisionError occurs if divisor argument equals zero in divide calculation.",
        "impact": "Application crash with uncaught ZeroDivisionError exception.",
        "remediation": "Validate divisor is non-zero before performing division.",
    }
    deduped = llm_review._semantic_deduplicate_issues([issue1, issue2])
    assert len(deduped) == 1
    assert deduped[0]["severity"] == "high"
    assert deduped[0]["confidence"] == 0.92


def test_semantic_dedup_same_defect_adjacent_lines():
    issue1 = {
        "path": "server.py",
        "line": 50,
        "category": "resource_management",
        "severity": "medium",
        "confidence": 0.80,
        "body": "Unclosed database socket leaks network resource descriptors.",
        "explanation": "Unclosed database socket leaks network resource descriptors.",
        "impact": "Socket descriptor exhaustion under load.",
        "remediation": "Use with closing(db.connect()) context manager.",
    }
    issue2 = {
        "path": "server.py",
        "line": 51,
        "category": "resource_management",
        "severity": "high",
        "confidence": 0.90,
        "body": "Database connection descriptor leak: socket handle is not closed.",
        "explanation": "Database connection descriptor leak: socket handle is opened but never closed in finally block.",
        "impact": "Descriptor leak exhausts available file handles.",
        "remediation": "Wrap connection in try/finally to close socket.",
    }
    deduped = llm_review._semantic_deduplicate_issues([issue1, issue2])
    assert len(deduped) == 1
    assert deduped[0]["line"] == 50
    assert deduped[0]["severity"] == "high"
    assert deduped[0]["confidence"] == 0.90


def test_semantic_dedup_same_defect_two_lines_apart():
    issue1 = {
        "path": "server.py",
        "line": 50,
        "category": "resource_management",
        "severity": "medium",
        "confidence": 0.80,
        "body": "Unclosed database socket leaks network resource descriptors.",
        "explanation": "Unclosed database socket leaks network resource descriptors.",
        "impact": "Socket descriptor exhaustion under load.",
    }
    issue2 = {
        "path": "server.py",
        "line": 52,
        "category": "resource_management",
        "severity": "high",
        "confidence": 0.90,
        "body": "Database socket resource leak leaves the descriptor unclosed.",
        "explanation": "Database socket resource leak leaves the descriptor unclosed.",
        "impact": "Leaked descriptors exhaust available network resources.",
    }

    deduped = llm_review._semantic_deduplicate_issues([issue1, issue2])

    assert len(deduped) == 1
    assert deduped[0]["line"] == 50


def test_semantic_dedup_two_independent_defects_on_adjacent_lines():
    issue1 = {
        "path": "app.py",
        "line": 20,
        "category": "correctness",
        "severity": "high",
        "confidence": 0.95,
        "body": "Division by zero if total count is zero.",
        "explanation": "Division by zero if total count is zero.",
        "impact": "Application crash.",
        "remediation": "Check if count > 0.",
    }
    issue2 = {
        "path": "app.py",
        "line": 21,
        "category": "security",
        "severity": "critical",
        "confidence": 0.99,
        "body": "SQL injection: unsanitized raw query parameter concatenated into SQL string.",
        "explanation": "SQL injection: unsanitized raw query parameter concatenated into SQL string.",
        "impact": "Remote database exfiltration or tampering.",
        "remediation": "Use parameterized query placeholders.",
    }
    deduped = llm_review._semantic_deduplicate_issues([issue1, issue2])
    assert len(deduped) == 2
    assert deduped[0]["category"] == "correctness"
    assert deduped[1]["category"] == "security"


def test_semantic_dedup_preserves_distinct_same_line_defects():
    average_issue = {
        "path": "calculator.py",
        "line": 42,
        "category": "edge_case",
        "severity": "medium",
        "confidence": 0.90,
        "body": (
            "calculate_average() raises ZeroDivisionError when the list is "
            "empty because its length is zero."
        ),
        "explanation": (
            "calculate_average() raises ZeroDivisionError when the list is "
            "empty because its length is zero."
        ),
    }
    divide_issue = {
        "path": "calculator.py",
        "line": 42,
        "category": "error_handling",
        "severity": "high",
        "confidence": 0.90,
        "body": "divide() raises ZeroDivisionError when divisor b is zero.",
        "explanation": "divide() raises ZeroDivisionError when divisor b is zero.",
    }
    same_text_security_issue = {
        **average_issue,
        "category": "security",
    }

    deduped = llm_review._semantic_deduplicate_issues(
        [average_issue, divide_issue, same_text_security_issue]
    )

    assert len(deduped) == 2
    assert {issue["category"] for issue in deduped} == {
        "correctness",
        "error_handling",
    }


def test_semantic_dedup_different_categories_and_root_causes_same_line():
    issue1 = {
        "path": "service.py",
        "line": 30,
        "category": "performance",
        "severity": "medium",
        "confidence": 0.85,
        "body": "O(N^2) nested loop over unindexed collection causes latency spike.",
        "explanation": "O(N^2) nested loop over unindexed collection causes latency spike.",
        "impact": "High CPU utilization on large payloads.",
        "remediation": "Build a lookup set beforehand.",
    }
    issue2 = {
        "path": "service.py",
        "line": 30,
        "category": "security",
        "severity": "critical",
        "confidence": 0.95,
        "body": "Hardcoded secret API token exposed directly in source.",
        "explanation": "Hardcoded secret API token exposed directly in source.",
        "impact": "Credential leakage across repository clones.",
        "remediation": "Load secret from environment variable.",
    }
    deduped = llm_review._semantic_deduplicate_issues([issue1, issue2])
    assert len(deduped) == 2
    assert {d["category"] for d in deduped} == {"performance", "security"}


def test_semantic_dedup_merges_logic_reports_mislabeled_as_security():
    issue1 = {
        "path": "calculator.py",
        "line": 10,
        "category": "correctness",
        "severity": "high",
        "confidence": 0.95,
        "explanation": "add() subtracts b instead of adding it.",
        "impact": "The arithmetic sum is incorrect.",
    }
    issue2 = {
        **issue1,
        "category": "security",
        "explanation": "Incorrect arithmetic: add() subtracts b instead of adding it.",
    }

    deduped = llm_review._semantic_deduplicate_issues([issue1, issue2])

    assert len(deduped) == 1
    assert deduped[0]["category"] == "correctness"


def test_semantic_dedup_keeps_path_traversal_in_security_category():
    issue1 = {
        "path": "storage.py",
        "line": 20,
        "category": "correctness",
        "severity": "high",
        "confidence": 0.90,
        "explanation": "Path traversal lets an untrusted filename escape the export directory.",
        "impact": "An attacker can overwrite files outside the export directory.",
    }
    issue2 = {
        **issue1,
        "category": "security",
        "explanation": "Path traversal lets an untrusted filename escape the export directory.",
    }

    deduped = llm_review._semantic_deduplicate_issues([issue1, issue2])

    assert len(deduped) == 1
    assert deduped[0]["category"] == "security"


def test_semantic_dedup_preserves_highest_severity():
    issue1 = {
        "path": "math_ops.py",
        "line": 15,
        "category": "correctness",
        "severity": "low",
        "confidence": 0.60,
        "body": "Off by one error in loop termination bound.",
        "explanation": "Off by one error in loop termination bound.",
        "impact": "Skips the last element.",
        "remediation": "Use range(len(items)).",
    }
    issue2 = {
        "path": "math_ops.py",
        "line": 15,
        "category": "correctness",
        "severity": "critical",
        "confidence": 0.95,
        "body": "Off by one boundary defect causing IndexError in loop termination.",
        "explanation": "Off by one boundary defect causing IndexError in loop termination index.",
        "impact": "System crash on last loop iteration.",
        "remediation": "Change boundary condition to i < length.",
    }
    deduped = llm_review._semantic_deduplicate_issues([issue1, issue2])
    assert len(deduped) == 1
    assert deduped[0]["severity"] == "critical"
    assert deduped[0]["confidence"] == 0.95


def test_semantic_dedup_remediation_selection():
    issue_without_rem = {
        "path": "auth.py",
        "line": 100,
        "category": "security",
        "severity": "high",
        "confidence": 0.90,
        "body": "Insecure cryptographic hashing algorithm MD5 is used for password verification.",
        "explanation": "Insecure cryptographic hashing algorithm MD5 is used for password verification.",
        "impact": "Pre-image attacks and trivial hash collisions.",
        "remediation": "",
    }
    issue_with_rem = {
        "path": "auth.py",
        "line": 100,
        "category": "security",
        "severity": "high",
        "confidence": 0.92,
        "body": "MD5 hash used for password check. Remediation: Use argon2 or bcrypt.",
        "explanation": "Insecure MD5 hashing used for user password verification.",
        "impact": "Hash collision vulnerability.",
        "remediation": "Replace hashlib.md5 with argon2-cffi or bcrypt.hashpw.",
    }
    deduped = llm_review._semantic_deduplicate_issues([issue_without_rem, issue_with_rem])
    assert len(deduped) == 1
    assert deduped[0]["remediation"] == "Replace hashlib.md5 with argon2-cffi or bcrypt.hashpw."


def test_legacy_finding_compatibility():
    legacy = {
        "path": "src/example.py",
        "line": 2,
        "category": "correctness",
        "severity": "medium",
        "confidence": 0.88,
        "body": "Handle null pointer dereference safely.",
    }
    review = {"summary": "Safe legacy review.", "issues": [legacy]}
    validated = llm_review.validate_review(review, FILES)
    assert len(validated["issues"]) == 1
    issue = validated["issues"][0]
    assert issue["body"] == "Handle null pointer dereference safely."
    assert issue["explanation"] == "Handle null pointer dereference safely."
    assert issue["impact"] == ""
    assert issue["remediation"] == ""


def test_enriched_finding_compatibility():
    enriched = {
        "path": "src/example.py",
        "line": 2,
        "category": "security",
        "severity": "high",
        "confidence": 0.95,
        "explanation": "Unvalidated input passed into system call.",
        "impact": "Command execution with permissions of running process.",
        "remediation": "Use subprocess.run with argv array and shell=False.",
    }
    review = {"summary": "Enriched review.", "issues": [enriched]}
    validated = llm_review.validate_review(review, FILES)
    assert len(validated["issues"]) == 1
    issue = validated["issues"][0]
    assert issue["explanation"] == "Unvalidated input passed into system call."
    assert issue["impact"] == "Command execution with permissions of running process."
    assert issue["remediation"] == "Use subprocess.run with argv array and shell=False."
    assert "**Impact:** Command execution" in issue["body"]
    assert "**Remediation:** Use subprocess.run" in issue["body"]


def test_logic_pass_security_category_is_corrected(monkeypatch):
    calls = []

    def fake_call(prompt, diff_text, headers, budget_tracker):
        calls.append(prompt)
        if prompt == llm_review.CORRECTNESS_PROMPT:
            return {
                "summary": "A logic issue was found.",
                "issues": [
                    {
                        "path": "src/example.py",
                        "line": 2,
                        "category": "security",
                        "severity": "high",
                        "confidence": 0.95,
                        "body": "The changed arithmetic returns the wrong result.",
                    }
                ],
            }
        return {"summary": "No security issue found.", "issues": []}

    monkeypatch.setattr(llm_review, "_call_model_for_prompt", fake_call)

    result = llm_review.review_diff(FILES, multi_pass=True)

    assert calls == [
        llm_review.CORRECTNESS_PROMPT,
        llm_review.SECURITY_RELIABILITY_PROMPT,
    ]
    assert len(result["issues"]) == 1
    assert result["issues"][0]["category"] == "correctness"


def test_all_review_passes_share_the_same_output_schema():
    prompts = (
        llm_review.SYSTEM_PROMPT,
        llm_review.CORRECTNESS_PROMPT,
        llm_review.SECURITY_RELIABILITY_PROMPT,
        llm_review.CROSS_FILE_PROMPT,
    )

    assert all(
        prompt.endswith(llm_review.FINDING_OUTPUT_CONTRACT)
        for prompt in prompts
    )
    assert '"explanation", "impact", and "remediation"' in (
        llm_review.FINDING_OUTPUT_CONTRACT
    )


def test_prompts_require_before_after_and_uniqueness_reasoning():
    contract = " ".join(llm_review.FINDING_OUTPUT_CONTRACT.split())
    assert (
        "compare the removed (-) code with the added (+) code"
        in contract
    )
    assert (
        "distinguish excluding the current user's own record from allowing "
        "another user's duplicate value"
    ) in contract
    security_prompt = " ".join(
        llm_review.SECURITY_RELIABILITY_PROMPT.split()
    )
    assert "an added line introduces or weakens the vulnerable behavior" in (
        security_prompt
    )
    assert (
        "do not attribute an unchanged permission check to the change"
        in security_prompt
    )


def test_multi_pass_keeps_independent_correctness_defects(monkeypatch):
    calls = []

    def fake_call(prompt, diff_text, headers, budget_tracker):
        calls.append(prompt)
        if prompt == llm_review.CORRECTNESS_PROMPT:
            return {
                "summary": "Two independent logic defects were found.",
                "issues": [
                    {
                        "path": "src/example.py",
                        "line": 2,
                        "category": "correctness",
                        "severity": "medium",
                        "confidence": 0.95,
                        "explanation": "is_even() returns true for odd values.",
                        "impact": "Even values are reported as odd.",
                        "remediation": "Compare the remainder with zero.",
                    },
                    {
                        "path": "src/example.py",
                        "line": 11,
                        "category": "edge_case",
                        "severity": "medium",
                        "confidence": 0.95,
                        "explanation": (
                            "calculate_average() divides by zero for an empty list."
                        ),
                        "impact": "An empty list raises ZeroDivisionError.",
                        "remediation": "Handle an empty list before division.",
                    },
                ],
            }
        return {"summary": "No security defect found.", "issues": []}

    monkeypatch.setattr(llm_review, "_call_model_for_prompt", fake_call)

    result = llm_review.review_diff(FILES, multi_pass=True)

    assert calls == [
        llm_review.CORRECTNESS_PROMPT,
        llm_review.SECURITY_RELIABILITY_PROMPT,
    ]
    assert len(result["issues"]) == 2
    assert {issue["line"] for issue in result["issues"]} == {2, 11}
    explanations = " ".join(
        issue["explanation"] for issue in result["issues"]
    )
    assert "is_even" in explanations
    assert "calculate_average" in explanations


def test_multi_pass_deduplicates_and_normalizes_across_passes(monkeypatch):
    duplicate_text = {
        2: "add() subtracts b instead of adding it.",
        11: "is_even() has an inverted condition and returns true for odd values.",
        12: "Path traversal lets an untrusted filename escape its directory.",
    }
    correctness_findings = [
        {
            "path": "src/example.py",
            "line": line,
            "category": "correctness",
            "severity": "high",
            "confidence": 0.95,
            "explanation": text,
            "impact": "",
            "remediation": "",
        }
        for line, text in duplicate_text.items()
    ]
    security_findings = [
        {
            **finding,
            "category": "security",
        }
        for finding in correctness_findings
    ]
    calls = []

    def fake_call(prompt, diff_text, headers, budget_tracker):
        calls.append(prompt)
        if prompt == llm_review.CORRECTNESS_PROMPT:
            return {"summary": "Logic pass.", "issues": correctness_findings}
        return {"summary": "Security pass.", "issues": security_findings}

    monkeypatch.setattr(llm_review, "_call_model_for_prompt", fake_call)

    result = llm_review.review_diff(FILES, multi_pass=True)

    assert len(calls) == 2
    assert len(result["issues"]) == 3
    categories = {
        issue["line"]: issue["category"] for issue in result["issues"]
    }
    assert categories == {
        2: "correctness",
        11: "correctness",
        12: "security",
    }


@pytest.mark.parametrize("category", ["error_handling", "security"])
def test_semantic_dedup_discards_speculative_open_error_handling(
    capsys,
    category,
):
    issue = {
        "path": "src/example.py",
        "line": 2,
        "category": category,
        "severity": "low",
        "confidence": 0.75,
        "explanation": "open() may raise PermissionError when access is denied.",
        "impact": "The file operation can fail.",
        "remediation": "Wrap open() in try/except and catch PermissionError.",
    }

    assert llm_review._semantic_deduplicate_issues([issue]) == []
    assert (
        "Discarded speculative optional error-handling finding at "
        "src/example.py:2."
    ) in capsys.readouterr().out


def test_review_prompts_discourage_speculative_error_handling():
    assert "Do not recommend optional validation" in llm_review.SYSTEM_PROMPT
    assert "Do not report that open() may raise" in llm_review.SYSTEM_PROMPT
    assert "accepted zero divisor" in llm_review.CORRECTNESS_PROMPT
    assert "Do not report ordinary logic errors as security findings" in (
        llm_review.SECURITY_RELIABILITY_PROMPT
    )


def test_malformed_findings_in_dedup_and_validation():
    # Deduplication gracefully ignores non-dicts
    deduped = llm_review._semantic_deduplicate_issues(["not a dict", None, 123])
    assert deduped == []

    # Validation rejects malformed issues
    malformed_review = {
        "summary": "Malformed test.",
        "issues": [
            "string issue",
            {"missing_fields": True},
            {"path": "src/example.py", "line": 2, "category": "invalid_cat", "severity": "high", "confidence": 0.9, "body": "test"},
        ],
    }
    validated = llm_review.validate_review(malformed_review, FILES)
    assert validated["issues"] == []


# ---------------------------------------------------------------------------
# _parse_json_response — extended tests for multi-stage parser
# ---------------------------------------------------------------------------

class TestParseJsonResponseExtended:
    """Tests for the multi-stage JSON parser added in the JSON-fix pass."""

    def test_valid_plain_json_accepted(self):
        raw = '{"summary": "All good.", "issues": []}'
        result = _parse_json_response(raw)
        assert result == {"summary": "All good.", "issues": []}

    def test_json_fence_with_language_tag_accepted(self):
        raw = '```json\n{"summary": "Fenced.", "issues": []}\n```'
        result = _parse_json_response(raw)
        assert result == {"summary": "Fenced.", "issues": []}

    def test_json_fence_without_language_tag_accepted(self):
        raw = '```\n{"summary": "Plain fence.", "issues": []}\n```'
        result = _parse_json_response(raw)
        assert result == {"summary": "Plain fence.", "issues": []}

    def test_json_with_preamble_and_postamble_accepted(self):
        """Model emits conversational text before/after the JSON object."""
        raw = 'Here is my review:\n{"summary": "Preamble test.", "issues": []}\nThat is all.'
        result = _parse_json_response(raw)
        assert result == {"summary": "Preamble test.", "issues": []}

    def test_unescaped_newline_in_string_accepted(self):
        """LLM emits raw \\n inside a JSON string — strict=False must handle it."""
        raw = '{"summary": "Line 1\nLine 2", "issues": []}'
        result = _parse_json_response(raw)
        assert "summary" in result
        assert "Line 1" in result["summary"]

    def test_unescaped_tab_in_string_accepted(self):
        """LLM emits raw \\t inside a JSON string — strict=False must handle it."""
        raw = '{"summary": "Col1\tCol2", "issues": []}'
        result = _parse_json_response(raw)
        assert "summary" in result
        assert "Col1" in result["summary"]

    def test_malformed_json_raises_validation_error(self):
        """Structural JSON breakage (unclosed brace) must raise ReviewValidationError."""
        with pytest.raises(ReviewValidationError, match="not valid JSON"):
            _parse_json_response('{"summary": "broken"')

    def test_json_array_top_level_raises_validation_error(self):
        """A top-level JSON array is invalid; must raise ReviewValidationError."""
        with pytest.raises(ReviewValidationError, match="JSON object"):
            _parse_json_response('[{"summary": "array"}]')


# ---------------------------------------------------------------------------
# _call_model_for_prompt — budget tracker and repair tests
# ---------------------------------------------------------------------------

def _make_nvidia_response(content: str, finish_reason: str = "stop") -> dict:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ]
    }


class TestCallModelBudgetAndRepair:
    """Tests for the budget_tracker and bounded JSON repair in _call_model_for_prompt."""

    HEADERS = {"Authorization": "Bearer test-key"}
    DIFF = "--- FILE: calc.py ---\n@@ -1,2 +1,2 @@\n-x = 1\n+x = 2\n"
    VALID_JSON = '{"summary": "ok", "issues": []}'

    def test_budget_exhausted_before_call_raises(self):
        """When budget is already at max, no API call should be made."""
        budget = {"calls": 16, "max_calls": 16}
        with pytest.raises(ReviewValidationError, match="budget exhausted"):
            llm_review._call_model_for_prompt(
                llm_review.SYSTEM_PROMPT, self.DIFF, self.HEADERS, budget
            )

    def test_successful_call_increments_budget(self, monkeypatch):
        """A successful API call must increment budget_tracker['calls'] by 1."""
        budget = {"calls": 0, "max_calls": 16}
        monkeypatch.setattr(
            llm_review,
            "_call_nvidia_with_retry",
            lambda payload, headers, **kw: _make_nvidia_response(self.VALID_JSON),
        )
        result = llm_review._call_model_for_prompt(
            llm_review.SYSTEM_PROMPT, self.DIFF, self.HEADERS, budget
        )
        assert result == {"summary": "ok", "issues": []}
        assert budget["calls"] == 1

    def test_repair_attempted_on_bad_json(self, monkeypatch):
        """When the first response is malformed JSON, one repair call is made."""
        call_log = []
        bad_json = '{"summary": broken'
        good_json = '{"summary": "repaired", "issues": []}'

        def fake_retry(payload, headers, **kw):
            call_log.append(payload)
            if len(call_log) == 1:
                return _make_nvidia_response(bad_json)
            return _make_nvidia_response(good_json)

        monkeypatch.setattr(llm_review, "_call_nvidia_with_retry", fake_retry)
        budget = {"calls": 0, "max_calls": 16}
        result = llm_review._call_model_for_prompt(
            llm_review.SYSTEM_PROMPT, self.DIFF, self.HEADERS, budget
        )
        assert result == {"summary": "repaired", "issues": []}
        assert budget["calls"] == 2  # original + repair
        assert len(call_log) == 2

    def test_repair_not_attempted_when_budget_at_limit(self, monkeypatch):
        """When budget is exactly 1 slot left, the original call uses it; no repair."""
        bad_json = '{"summary": broken'

        monkeypatch.setattr(
            llm_review,
            "_call_nvidia_with_retry",
            lambda payload, headers, **kw: _make_nvidia_response(bad_json),
        )
        budget = {"calls": 15, "max_calls": 16}
        with pytest.raises(ReviewValidationError):
            llm_review._call_model_for_prompt(
                llm_review.SYSTEM_PROMPT, self.DIFF, self.HEADERS, budget
            )
        # Original call was made (budget goes to 16), repair blocked
        assert budget["calls"] == 16

    def test_repair_failure_propagates_validation_error(self, monkeypatch):
        """If both original and repair calls produce bad JSON, raise ReviewValidationError."""
        bad_json = '{"summary": broken'

        monkeypatch.setattr(
            llm_review,
            "_call_nvidia_with_retry",
            lambda payload, headers, **kw: _make_nvidia_response(bad_json),
        )
        budget = {"calls": 0, "max_calls": 16}
        with pytest.raises(ReviewValidationError):
            llm_review._call_model_for_prompt(
                llm_review.SYSTEM_PROMPT, self.DIFF, self.HEADERS, budget
            )
        assert budget["calls"] == 2  # original + one repair attempt

    def test_none_budget_creates_local_tracker(self, monkeypatch):
        """Passing budget_tracker=None should not raise and uses a local budget."""
        monkeypatch.setattr(
            llm_review,
            "_call_nvidia_with_retry",
            lambda payload, headers, **kw: _make_nvidia_response(self.VALID_JSON),
        )
        result = llm_review._call_model_for_prompt(
            llm_review.SYSTEM_PROMPT, self.DIFF, self.HEADERS, None
        )
        assert result["summary"] == "ok"

    def test_response_format_field_in_payload(self, monkeypatch):
        """The outgoing payload must include response_format json_object."""
        captured = {}

        def capture(payload, headers, **kw):
            captured["payload"] = payload
            return _make_nvidia_response(self.VALID_JSON)

        monkeypatch.setattr(llm_review, "_call_nvidia_with_retry", capture)
        budget = {"calls": 0, "max_calls": 16}
        llm_review._call_model_for_prompt(
            llm_review.SYSTEM_PROMPT, self.DIFF, self.HEADERS, budget
        )
        assert captured["payload"].get("response_format") == {"type": "json_object"}

    def test_invalid_schema_after_valid_json_raises(self, monkeypatch):
        """Valid JSON that fails schema validation (missing 'summary') must raise."""
        no_summary = '{"issues": []}'
        monkeypatch.setattr(
            llm_review,
            "_call_nvidia_with_retry",
            lambda payload, headers, **kw: _make_nvidia_response(no_summary),
        )
        budget = {"calls": 0, "max_calls": 16}
        raw_result = llm_review._call_model_for_prompt(
            llm_review.SYSTEM_PROMPT, self.DIFF, self.HEADERS, budget
        )
        # _call_model_for_prompt returns the raw parsed dict; schema validation
        # is done by validate_review, which must reject it.
        files = [{"path": "calc.py", "patch": self.DIFF}]
        with pytest.raises(ReviewValidationError, match="summary"):
            validate_review(raw_result, files)
