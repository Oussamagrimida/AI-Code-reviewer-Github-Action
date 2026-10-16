import json
import pytest
import requests
import re

from llm_review import (
    ReviewValidationError,
    _added_lines,
    _call_nvidia_with_retry,
    _parse_json_response,
    validate_review,
)
import llm_review


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


def test_missing_required_issue_field_raises():
    review = valid_review()

    del review["issues"][0]["confidence"]

    with pytest.raises(
        ReviewValidationError,
        match="confidence",
    ):
        validate_review(review, FILES)


@pytest.mark.parametrize(
    "field,value",
    [
        ("category", "bug"),
        ("severity", "bug"),
    ],
)
def test_invalid_enum_raises(field, value):
    review = valid_review()

    review["issues"][0][field] = value

    with pytest.raises(
        ReviewValidationError,
        match="invalid",
    ):
        validate_review(review, FILES)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("category", []),
        ("category", {}),
        ("severity", []),
        ("severity", {}),
    ],
)
def test_unhashable_enum_values_raise_validation_error(field, value):
    review = valid_review()
    review["issues"][0][field] = value

    with pytest.raises(ReviewValidationError):
        validate_review(review, FILES)


def test_invalid_confidence_raises():
    review = valid_review()

    review["issues"][0]["confidence"] = 1.1

    with pytest.raises(
        ReviewValidationError,
        match="between 0 and 1",
    ):
        validate_review(review, FILES)


def test_unknown_file_raises():
    review = valid_review()

    review["issues"][0]["path"] = "src/missing.py"

    with pytest.raises(
        ReviewValidationError,
        match="not in the PR diff",
    ):
        validate_review(review, FILES)


def test_context_line_is_rejected():
    review = valid_review()

    review["issues"][0]["line"] = 3

    with pytest.raises(
        ReviewValidationError,
        match="not an added line",
    ):
        validate_review(review, FILES)


def test_deleted_line_is_rejected():
    review = valid_review()

    review["issues"][0]["line"] = 10

    with pytest.raises(
        ReviewValidationError,
        match="not an added line",
    ):
        validate_review(review, FILES)


def test_unexpected_issue_field_is_rejected():
    review = valid_review()

    review["issues"][0]["extra"] = "unexpected"

    with pytest.raises(
        ReviewValidationError,
        match="unexpected field",
    ):
        validate_review(review, FILES)


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
    patch = """@@ -10,3 +10,4 @@
    def hello():
        x = 1
        +y = 2
    @@ -50,3 +51,5 @@
    def goodbye():
        +a = 10
        +b = 20
        return True
    """

    files = [
        {
            "path": "example.py",
            "patch": patch,
        }
    ]

    assert llm_review._added_lines(patch) == {12, 52, 53}
def test_added_lines_multiple_hunks_with_deleted_and_context_lines():
    patch = """@@ -10,4 +10,5 @@
    def hello():
        x = 1
        -old = 2
        +new = 2
        return x

    @@ -30,4 +31,6 @@
    def goodbye():
        -old_a = 10
        +new_a = 10
        +new_b = 20
        return True
    """

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
    }
    second_finding = {
        "path": "src/last.py",
        "line": 1,
        "category": "security",
        "severity": "medium",
        "confidence": 0.8,
        "body": "Validate this changed input.",
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


def test_review_diff_rejects_finding_from_another_chunk(monkeypatch):
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

    with pytest.raises(
        ReviewValidationError,
        match="not an added line",
    ):
        llm_review.review_diff(files)

    assert len(calls) == 1


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


def test_invalid_severity_is_rejected():
    review = valid_review()
    review["issues"][0]["severity"] = "urgent"

    with pytest.raises(
        ReviewValidationError,
        match="invalid severity",
    ):
        validate_review(review, FILES)


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
def test_invalid_severity_values_are_rejected(severity):
    review = valid_review()
    review["issues"][0]["severity"] = severity

    with pytest.raises(
        ReviewValidationError,
        match="invalid severity",
    ):
        validate_review(review, FILES)

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
def test_out_of_range_confidence_is_rejected(confidence):
    review = valid_review()
    review["issues"][0]["confidence"] = confidence

    with pytest.raises(
        ReviewValidationError,
        match="between 0 and 1",
    ):
        validate_review(review, FILES)


@pytest.mark.parametrize(
    "confidence",
    [True, False, "0.9", "high", None, [], {}],
)
def test_invalid_confidence_types_are_rejected(confidence):
    review = valid_review()
    review["issues"][0]["confidence"] = confidence

    with pytest.raises(
        ReviewValidationError,
        match="invalid.*confidence",
    ):
        validate_review(review, FILES)

def test_system_prompt_contains_confidence_guidance():
    assert "Confidence scoring:" in llm_review.SYSTEM_PROMPT
    assert "between 0 and 1" in llm_review.SYSTEM_PROMPT
    assert "confidence represents how certain" in llm_review.SYSTEM_PROMPT

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


def test_issue_must_be_an_object():
    review = valid_review()
    review["issues"] = ["invalid issue"]

    with pytest.raises(
        ReviewValidationError,
        match="must be a JSON object",
    ):
        validate_review(review, FILES)

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
def test_invalid_issue_field_types_are_rejected(field, value):
    review = valid_review()
    review["issues"][0][field] = value

    with pytest.raises(ReviewValidationError):
        validate_review(review, FILES)

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

    with pytest.raises(
        requests.exceptions.ConnectionError,
        match="Connection interrupted",
    ):
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

    with pytest.raises(
        requests.exceptions.HTTPError,
        match="503 Server Error: Service Unavailable",
    ):
        _call_nvidia_with_retry(
            {},
            {},
            max_attempts=3,
        )

    assert len(calls) == 3
    assert delays == [5, 10]



@pytest.mark.parametrize(
    "status_code",
    [502, 503, 504],
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

    with pytest.raises(
        requests.exceptions.Timeout,
        match="timeout",
    ):
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

    with pytest.raises(
        requests.exceptions.HTTPError,
        match="401",
    ):
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

            with pytest.raises(ReviewValidationError):
                llm_review.validate_review(review, [chunk_file])
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
        "summary": "",
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
