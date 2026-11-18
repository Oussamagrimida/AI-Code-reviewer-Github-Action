"""
Multi-defect benchmark tests for the AI code reviewer.

These tests verify that the multi-pass review engine can discover multiple
independent defects in a single execution. They use deterministic mocked
NVIDIA responses that mirror what the LLM would plausibly report.

No real network calls are made.
"""

import json
import pytest
import requests
from requests.structures import CaseInsensitiveDict

import llm_review
from llm_review import validate_review, review_diff


# ---------------------------------------------------------------------------
# Shared calculator benchmark fixture diff
# ---------------------------------------------------------------------------

CALC_PATCH_CALC_PY = """\
@@ -1,30 +1,37 @@
+def add(a, b):
+    \"\"\"Return the sum of a and b.\"\"\"
+    return a - b  # BUG: should be a + b
+
+def subtract(a, b):
+    \"\"\"Return a minus b.\"\"\"
+    return a - b
+
+def divide(a, b):
+    \"\"\"Return a divided by b.\"\"\"
+    return a / b  # BUG: no division-by-zero guard
+
+def is_even(n):
+    \"\"\"Return True if n is even.\"\"\"
+    return n % 2 == 1  # BUG: inverted condition, True for odd
+
+def average(numbers):
+    \"\"\"Return the average of a list of numbers.\"\"\"
+    return sum(numbers) / len(numbers)  # BUG: ZeroDivisionError on empty list
"""

CALC_PATCH_UTILS_PY = """\
@@ -1,15 +1,22 @@
+import os
+
+def read_config(filename):
+    \"\"\"Read a configuration file from the config directory.\"\"\"
+    # BUG: path traversal - user-controlled filename is not sanitized
+    path = os.path.join("configs", filename)
+    with open(path) as f:
+        return f.read()
"""

BENCHMARK_FILES = [
    {
        "path": "calculator.py",
        "patch": CALC_PATCH_CALC_PY,
    },
    {
        "path": "utils.py",
        "patch": CALC_PATCH_UTILS_PY,
    },
]


def _make_response(issues: list, summary: str = "Issues found.") -> dict:
    """Build a fake NVIDIA API response for the given issues list."""
    content = json.dumps({"summary": summary, "issues": issues})
    return {
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": "stop",
            }
        ]
    }


# ---------------------------------------------------------------------------
# Benchmark: single review run must find all 5 independent defects
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def set_nvidia_key(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")


def test_multi_defect_benchmark_single_review_run(monkeypatch):
    """
    A single call to review_diff(files, multi_pass=True) must discover all 5
    independent defects planted in the benchmark calculator diff:

    1. Incorrect addition (a - b instead of a + b)
    2. Division by zero (no guard in divide())
    3. Inverted even/odd condition (% 2 == 1 instead of == 0)
    4. Empty list average (ZeroDivisionError on empty input)
    5. Path traversal security defect in read_config()
    """
    correctness_issues = [
        {
            "path": "calculator.py",
            "line": 3,
            "category": "correctness",
            "severity": "high",
            "confidence": 0.95,
            "explanation": "add() uses subtraction operator instead of addition.",
            "impact": "Every addition call returns a - b instead of a + b.",
            "remediation": "Change 'return a - b' to 'return a + b'.",
        },
        {
            "path": "calculator.py",
            "line": 11,
            "category": "error_handling",
            "severity": "high",
            "confidence": 0.95,
            "explanation": "divide() performs a / b without checking for b == 0.",
            "impact": "Raises ZeroDivisionError when the divisor is zero.",
            "remediation": "Guard with 'if b == 0: raise ValueError(...)' before dividing.",
        },
        {
            "path": "calculator.py",
            "line": 15,
            "category": "correctness",
            "severity": "medium",
            "confidence": 0.90,
            "explanation": "is_even() checks n % 2 == 1 which is true for odd numbers, not even.",
            "impact": "is_even() returns True for odd numbers and False for even numbers.",
            "remediation": "Change the condition to 'n % 2 == 0'.",
        },
        {
            "path": "calculator.py",
            "line": 19,
            "category": "edge_case",
            "severity": "medium",
            "confidence": 0.90,
            "explanation": "average() calls len(numbers) without checking for an empty list.",
            "impact": "Raises ZeroDivisionError when numbers is empty.",
            "remediation": "Return None or raise ValueError if numbers is empty.",
        },
    ]

    security_issues = [
        {
            "path": "utils.py",
            "line": 6,
            "category": "security",
            "severity": "high",
            "confidence": 0.95,
            "explanation": "read_config() uses a user-supplied filename directly in os.path.join() without sanitization.",
            "impact": "An attacker can read arbitrary files by using path traversal sequences like '../../etc/passwd'.",
            "remediation": "Validate that filename contains no path separators and is in an allowlist.",
        },
    ]

    # Map: (prompt_keyword, call_index) -> response
    # Pass 2 (CORRECTNESS_PROMPT) => correctness issues
    # Pass 3 (SECURITY_RELIABILITY_PROMPT) => security issues
    call_sequence = [
        _make_response(correctness_issues, "Correctness pass: found 4 bugs."),
        _make_response(security_issues, "Security pass: found path traversal."),
    ]
    call_iter = iter(call_sequence)

    def fake_post(url, headers=None, json=None, timeout=None):
        resp = type("R", (), {})()
        resp.status_code = 200
        resp.headers = CaseInsensitiveDict({"content-type": "application/json"})
        resp.json = lambda: next(call_iter)
        resp.raise_for_status = lambda: None
        return resp

    monkeypatch.setattr(requests, "post", fake_post)

    result = review_diff(BENCHMARK_FILES, multi_pass=True)

    issues = result["issues"]
    paths = {i["path"] for i in issues}
    categories = {i["category"] for i in issues}

    # Must cover both files
    assert "calculator.py" in paths, "calculator.py issues not found"
    assert "utils.py" in paths, "utils.py security issue not found"

    # Must have at least one security finding and correctness finding
    assert "security" in categories, "Security defect not detected"
    assert "correctness" in categories, "Correctness defect not detected"

    # Must find at least 5 independent issues (4 bugs + 1 security)
    assert len(issues) >= 5, (
        f"Expected at least 5 independent defects in a single run, got {len(issues)}: "
        + str([i.get("explanation", i.get("body", "")) for i in issues])
    )

    # Verify specific defects are reported
    explanations = " ".join(
        i.get("explanation", i.get("body", "")) for i in issues
    ).lower()
    assert "subtraction" in explanations or "a - b" in explanations or "addition" in explanations, \
        "Incorrect addition defect not found"
    assert "divide" in explanations or "zero" in explanations or "b == 0" in explanations or "divisor" in explanations, \
        "Division-by-zero defect not found"
    assert "odd" in explanations or "even" in explanations or "% 2" in explanations or "inverted" in explanations, \
        "Inverted even/odd condition not found"
    assert "empty" in explanations or "average" in explanations, \
        "Empty list average defect not found"
    assert "traversal" in explanations or "path" in explanations or "sanitiz" in explanations, \
        "Path traversal security defect not found"


# ---------------------------------------------------------------------------
# Benchmark: deduplication preserves 5 independent defects
# ---------------------------------------------------------------------------

def test_benchmark_deduplication_preserves_independent_defects():
    """
    When both passes report the same defect AND two different defects,
    deduplication must eliminate only the genuine duplicate and keep all
    independent findings.
    """
    from llm_review import _semantic_deduplicate_issues

    issues = [
        # Duplicate pair: same division-by-zero reported in both passes
        {
            "path": "calculator.py",
            "line": 11,
            "category": "error_handling",
            "severity": "high",
            "confidence": 0.95,
            "explanation": "divide() performs a / b without checking for b == 0.",
            "impact": "ZeroDivisionError when b is zero.",
            "remediation": "Guard before division.",
        },
        {
            "path": "calculator.py",
            "line": 11,
            "category": "error_handling",
            "severity": "high",
            "confidence": 0.80,
            "explanation": "No zero-check guard before dividing a by b in divide().",
            "impact": "Raises ZeroDivisionError when divisor is 0.",
            "remediation": "Check if b == 0 before dividing.",
        },
        # Independent: incorrect addition
        {
            "path": "calculator.py",
            "line": 3,
            "category": "correctness",
            "severity": "high",
            "confidence": 0.95,
            "explanation": "add() returns a - b instead of a + b.",
            "impact": "Wrong result for every addition.",
            "remediation": "Change operator to +.",
        },
        # Independent: inverted condition
        {
            "path": "calculator.py",
            "line": 15,
            "category": "correctness",
            "severity": "medium",
            "confidence": 0.90,
            "explanation": "is_even() checks n % 2 == 1 which returns True for odd numbers.",
            "impact": "is_even() logic is inverted.",
            "remediation": "Change to n % 2 == 0.",
        },
        # Independent: empty-list average
        {
            "path": "calculator.py",
            "line": 19,
            "category": "edge_case",
            "severity": "medium",
            "confidence": 0.90,
            "explanation": "average() will raise ZeroDivisionError on an empty list.",
            "impact": "Crashes when called with no elements.",
            "remediation": "Guard against empty list.",
        },
        # Independent: path traversal
        {
            "path": "utils.py",
            "line": 6,
            "category": "security",
            "severity": "high",
            "confidence": 0.95,
            "explanation": "read_config() joins unsanitized filename enabling path traversal.",
            "impact": "Arbitrary file read by attacker.",
            "remediation": "Validate filename before use.",
        },
    ]

    result = _semantic_deduplicate_issues(issues)

    # The duplicate division-by-zero pair collapses to 1; all others survive
    assert len(result) == 5, (
        f"Expected 5 findings after deduplication (1 duplicate removed), got {len(result)}: "
        + str([i.get("explanation", "") for i in result])
    )

    paths = {i["path"] for i in result}
    assert "calculator.py" in paths
    assert "utils.py" in paths

    categories = {i["category"] for i in result}
    assert "correctness" in categories
    assert "security" in categories

    # The preserved division-by-zero finding should have the highest confidence
    divzero = [
        i for i in result
        if "divide" in i.get("explanation", "").lower()
        or i.get("line") == 11
    ]
    assert len(divzero) == 1, "Expected exactly one division-by-zero finding"
    assert divzero[0]["confidence"] == 0.95


# ---------------------------------------------------------------------------
# Cross-file consistency test
# ---------------------------------------------------------------------------

def test_cross_file_broken_caller_detected(monkeypatch):
    """
    Pass 4 must detect when a function signature changes in one file and a
    caller in another file is not updated to match.
    """
    # File A: function signature changed from (x, y) to (x, y, z)
    file_a_patch = """\
@@ -5,3 +5,3 @@
-def compute(x, y):
+def compute(x, y, z=None):
     return x + y
"""
    # File B: caller still uses the old 2-argument form (fine here, z is optional)
    # But in this test we simulate a case where the call breaks the new required param
    file_b_patch = """\
@@ -10,3 +10,3 @@
-    result = compute(a, b)
+    result = compute(a)  # BUG: missing required argument b
"""

    cross_files = [
        {"path": "engine.py", "patch": file_a_patch},
        {"path": "runner.py", "patch": file_b_patch},
    ]

    cross_issue = {
        "path": "runner.py",
        "line": 10,
        "category": "correctness",
        "severity": "high",
        "confidence": 0.95,
        "explanation": "runner.py calls compute(a) with only one argument; compute() requires at least x and y.",
        "impact": "TypeError at runtime when runner.py calls compute().",
        "remediation": "Pass both required arguments: compute(a, b).",
    }

    call_sequence = [
        # Pass 2: correctness (finds the broken call in runner.py)
        _make_response([cross_issue], "Correctness: broken caller found."),
        # Pass 3: security (nothing notable)
        _make_response([], "Security: no issues."),
        # Pass 4: cross-file (confirms the broken interface)
        _make_response([cross_issue], "Cross-file: signature mismatch confirmed."),
    ]
    call_iter = iter(call_sequence)

    def fake_post(url, headers=None, json=None, timeout=None):
        resp = type("R", (), {})()
        resp.status_code = 200
        resp.headers = CaseInsensitiveDict({"content-type": "application/json"})
        resp.json = lambda: next(call_iter)
        resp.raise_for_status = lambda: None
        return resp

    monkeypatch.setattr(requests, "post", fake_post)
    # Force cross-file analysis to run by patching structure detection
    orig_analyze = llm_review._analyze_diff_structure

    def patched_analyze(files):
        result = orig_analyze(files)
        # Simulate that the engine detected a cross-file relationship
        result["has_cross_file_relationships"] = True
        return result

    monkeypatch.setattr(llm_review, "_analyze_diff_structure", patched_analyze)

    result = review_diff(cross_files, multi_pass=True)
    issues = result["issues"]

    assert len(issues) >= 1, "Expected at least one cross-file issue to be reported"
    runner_issues = [i for i in issues if i["path"] == "runner.py"]
    assert runner_issues, "Expected runner.py broken caller to be reported"
    assert any(
        "compute" in i.get("explanation", i.get("body", "")).lower()
        for i in runner_issues
    ), "Expected the broken compute() call to be mentioned"


# ---------------------------------------------------------------------------
# Mode 2 basic validation unit tests
# ---------------------------------------------------------------------------

def test_run_basic_validation_valid_python(tmp_path):
    """Valid Python file must pass."""
    import fixer
    f = tmp_path / "calc.py"
    f.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    result = fixer._run_basic_validation([str(f)])
    assert result["passed"] is True


def test_run_basic_validation_invalid_python(tmp_path):
    """Python file with a syntax error must fail."""
    import fixer
    f = tmp_path / "broken.py"
    f.write_text("def broken(\n", encoding="utf-8")
    result = fixer._run_basic_validation([str(f)])
    assert result["passed"] is False
    assert "broken.py" in result["output"]


def test_run_basic_validation_valid_json(tmp_path):
    """Valid JSON file must pass."""
    import fixer
    f = tmp_path / "config.json"
    f.write_text('{"key": "value"}', encoding="utf-8")
    result = fixer._run_basic_validation([str(f)])
    assert result["passed"] is True


def test_run_basic_validation_invalid_json(tmp_path):
    """Malformed JSON file must fail."""
    import fixer
    f = tmp_path / "config.json"
    f.write_text("{key: value}", encoding="utf-8")
    result = fixer._run_basic_validation([str(f)])
    assert result["passed"] is False
    assert "config.json" in result["output"]


def test_run_basic_validation_nonexistent_file():
    """Non-existent file must be skipped without error (file may be deleted)."""
    import fixer
    result = fixer._run_basic_validation(["/nonexistent/path/to/file.py"])
    assert result["passed"] is True


def test_run_basic_validation_empty_list():
    """Empty file list must pass immediately."""
    import fixer
    result = fixer._run_basic_validation([])
    assert result["passed"] is True


def test_run_basic_validation_mixed_files(tmp_path):
    """When one file fails, the entire validation fails fast."""
    import fixer
    good = tmp_path / "good.py"
    good.write_text("x = 1\n", encoding="utf-8")
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{broken", encoding="utf-8")
    result = fixer._run_basic_validation([str(good), str(bad_json)])
    assert result["passed"] is False
    assert "bad.json" in result["output"]
