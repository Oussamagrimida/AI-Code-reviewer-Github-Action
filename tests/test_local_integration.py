"""
Local integration test for the multi-pass review engine.

Exercises the real review orchestration pipeline against a temporary Git
repository containing five independent defects:
1. Incorrect addition (operator flaw)
2. Incorrect division logic (missing division-by-zero validation)
3. Inverted even/odd boolean condition
4. Empty-list average calculation failure (ZeroDivisionError)
5. Path traversal vulnerability in file export

Tests:
- Deterministic orchestration with mocked external NVIDIA HTTP calls
- Cross-file conditional pass gating
- Line anchoring and schema validation
- Live execution against the real NVIDIA hosted API (when NVIDIA_API_KEY is provided)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Ensure repository root is on sys.path when running as a standalone script
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pytest
import requests
from requests.structures import CaseInsensitiveDict

import llm_review
from llm_review import (
    review_diff,
    validate_review,
    _added_lines,
    _analyze_diff_structure,
    CORRECTNESS_PROMPT,
    SECURITY_RELIABILITY_PROMPT,
    CROSS_FILE_PROMPT,
)


def create_temporary_buggy_project(repo_dir: Path, with_cross_file: bool = False) -> list[dict]:
    """
    Initializes a temporary Git repository, commits clean baseline files,
    then introduces the five independent defects into the working tree and
    generates realistic Git diff patches.
    """
    def run_git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=True,
        )

    # 1. Initialize local temporary Git repo
    run_git("init")
    run_git("config", "user.name", "Local Integration Tester")
    run_git("config", "user.email", "tester@local.test")
    run_git("config", "core.autocrlf", "false")

    calc_file = repo_dir / "calculator.py"
    storage_file = repo_dir / "storage.py"

    # 2. Baseline commit
    calc_file.write_text(
        "# Calculator utilities\n"
        "def version():\n"
        "    return '1.0.0'\n",
        encoding="utf-8",
    )
    storage_file.write_text(
        "# Storage utilities\n"
        "import os\n"
        "EXPORT_DIR = 'exports'\n",
        encoding="utf-8",
    )

    tracked_files = ["calculator.py", "storage.py"]
    if with_cross_file:
        service_file = repo_dir / "service.py"
        service_file.write_text(
            "# Service utilities\n"
            "def compute_total(values: list) -> float:\n"
            "    return 0.0\n",
            encoding="utf-8",
        )
        tracked_files.append("service.py")

    run_git("add", *tracked_files)
    run_git("commit", "-m", "Initial clean baseline")

    # 3. Introduce the 5 independent defects
    # File 1: calculator.py (4 defects)
    calc_buggy = (
        "# Calculator utilities\n"
        "def version():\n"
        "    return '1.0.0'\n\n"
        "def add(a: float, b: float) -> float:\n"
        '    """Return sum of a and b."""\n'
        "    return a - b  # Defect 1: Wrong arithmetic operator (- instead of +)\n\n"
        "def divide(a: float, b: float) -> float:\n"
        '    """Return quotient of a and b."""\n'
        "    # Defect 2: Missing division-by-zero check\n"
        "    return a / b\n\n"
        "def is_even(n: int) -> bool:\n"
        '    """Return True if n is even."""\n'
        "    # Defect 3: Inverted condition (returns True for odd numbers)\n"
        "    return n % 2 == 1\n\n"
        "def calculate_average(numbers: list) -> float:\n"
        '    """Return mean of numbers."""\n'
        "    # Defect 4: ZeroDivisionError on empty list\n"
        "    return sum(numbers) / len(numbers)\n"
    )
    calc_file.write_text(calc_buggy, encoding="utf-8")

    # File 2: storage.py (1 security defect)
    storage_buggy = (
        "# Storage utilities\n"
        "import os\n"
        "EXPORT_DIR = 'exports'\n\n"
        "def export_user_data(filename: str, payload: str) -> str:\n"
        '    """Write payload to export file."""\n'
        "    # Defect 5: Path traversal - unsanitized filename joined directly\n"
        "    destination = os.path.join(EXPORT_DIR, filename)\n"
        '    with open(destination, "w", encoding="utf-8") as f:\n'
        "        f.write(payload)\n"
        "    return destination\n"
    )
    storage_file.write_text(storage_buggy, encoding="utf-8")

    # Optional File 3: service.py (imports calculator and calls add)
    if with_cross_file:
        service_file.write_text(
            "# Service utilities\n"
            "import calculator\n\n"
            "def compute_total(values: list) -> float:\n"
            "    total = 0.0\n"
            "    for v in values:\n"
            "        total = calculator.add(total, v)\n"
            "    return total\n",
            encoding="utf-8",
        )

    # 4. Extract realistic Git unified diffs
    files = []
    for rel_path in tracked_files:
        diff_raw = run_git("diff", "-U3", "HEAD", "--", rel_path).stdout
        lines = diff_raw.splitlines()
        hunk_start = next((i for i, line in enumerate(lines) if line.startswith("@@")), None)
        if hunk_start is not None:
            patch = "\n".join(lines[hunk_start:]) + "\n"
            files.append({"path": rel_path, "patch": patch})

    return files


def _make_mock_response(issues: list, summary: str = "Review pass completed.") -> dict:
    return {
        "choices": [
            {
                "message": {"content": json.dumps({"summary": summary, "issues": issues})},
                "finish_reason": "stop",
            }
        ]
    }


# ===========================================================================
# Automated Deterministic Tests
# ===========================================================================

def test_real_orchestration_independent_files(tmp_path, monkeypatch):
    """
    Exercises the real multi-pass review orchestration against the temporary
    buggy project. Verifies:
    1. Real Git diffs are created and parsed.
    2. Pass 1 analyzes AST structure with zero API calls.
    3. Pass 2 and Pass 3 execute.
    4. Pass 4 is skipped because calculator.py and storage.py are independent.
    5. All 5 independent defects are reported and line-anchored to the real diff.
    6. Semantic deduplication merges duplicate reports of the same defect.
    7. All findings have enriched schema fields and backward-compatible body.
    """
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    files = create_temporary_buggy_project(repo_dir, with_cross_file=False)

    assert len(files) == 2, "Expected 2 diff files (calculator.py, storage.py)"
    calc_added = _added_lines(files[0]["patch"])
    storage_added = _added_lines(files[1]["patch"])

    # Determine real added line numbers from the Git patch
    # Defect 1: add return line
    # Defect 2: divide return line
    # Defect 3: is_even return line
    # Defect 4: calculate_average return line
    # Defect 5: export_user_data path join line
    add_line = min(calc_added) + 2
    divide_line = min(calc_added) + 6
    even_line = min(calc_added) + 10
    average_line = min(calc_added) + 14
    traversal_line = min(storage_added) + 3

    assert add_line in calc_added
    assert divide_line in calc_added
    assert even_line in calc_added
    assert average_line in calc_added
    assert traversal_line in storage_added

    correctness_findings = [
        {
            "path": "calculator.py",
            "line": add_line,
            "category": "correctness",
            "severity": "high",
            "confidence": 0.95,
            "explanation": "add() subtracts b from a instead of performing addition.",
            "impact": "Calculates incorrect arithmetic sum.",
            "remediation": "Change 'return a - b' to 'return a + b'.",
        },
        {
            "path": "calculator.py",
            "line": divide_line,
            "category": "error_handling",
            "severity": "high",
            "confidence": 0.95,
            "explanation": "divide() does not check if divisor b == 0.",
            "impact": "Raises ZeroDivisionError on b=0.",
            "remediation": "Check for b == 0 and raise ValueError.",
        },
        {
            "path": "calculator.py",
            "line": even_line,
            "category": "correctness",
            "severity": "medium",
            "confidence": 0.90,
            "explanation": "is_even() condition n % 2 == 1 is inverted.",
            "impact": "Returns True for odd integers and False for even.",
            "remediation": "Change condition to 'n % 2 == 0'.",
        },
        {
            "path": "calculator.py",
            "line": average_line,
            "category": "edge_case",
            "severity": "medium",
            "confidence": 0.90,
            "explanation": "calculate_average() fails when numbers list is empty.",
            "impact": "Raises ZeroDivisionError on empty collection.",
            "remediation": "Check if numbers is empty before dividing by len(numbers).",
        },
    ]

    security_findings = [
        # Defect 5: path traversal
        {
            "path": "storage.py",
            "line": traversal_line,
            "category": "security",
            "severity": "high",
            "confidence": 0.95,
            "explanation": "export_user_data() introduces a path traversal vulnerability by joining unsanitized filename.",
            "impact": "Path traversal vulnerability allowing arbitrary file overwrite.",
            "remediation": "Sanitize filename or validate against allowlist.",
        },
        # Duplicate report of divide-by-zero from security pass to test Pass 5 deduplication
        {
            "path": "calculator.py",
            "line": divide_line,
            "category": "error_handling",
            "severity": "high",
            "confidence": 0.85,
            "explanation": "divide() unhandled division by zero exception.",
            "impact": "Denial of service via crash on zero divisor.",
            "remediation": "Guard with zero check before division.",
        },
    ]

    recorded_calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        prompt_used = json["messages"][0]["content"]
        recorded_calls.append({
            "prompt": prompt_used,
            "diff_text": json["messages"][1]["content"],
        })
        if prompt_used == CORRECTNESS_PROMPT:
            return type("R", (), {
                "status_code": 200,
                "headers": CaseInsensitiveDict({"content-type": "application/json"}),
                "json": lambda *args, **kwargs: _make_mock_response(correctness_findings, "Pass 2 Correctness done."),
                "raise_for_status": lambda *args, **kwargs: None,
            })()
        elif prompt_used == SECURITY_RELIABILITY_PROMPT:
            return type("R", (), {
                "status_code": 200,
                "headers": CaseInsensitiveDict({"content-type": "application/json"}),
                "json": lambda *args, **kwargs: _make_mock_response(security_findings, "Pass 3 Security done."),
                "raise_for_status": lambda *args, **kwargs: None,
            })()
        else:
            return type("R", (), {
                "status_code": 200,
                "headers": CaseInsensitiveDict({"content-type": "application/json"}),
                "json": lambda *args, **kwargs: _make_mock_response([], "Other pass done."),
                "raise_for_status": lambda *args, **kwargs: None,
            })()

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-api-key")

    # Run the actual public multi-pass review entry point
    result = review_diff(files, multi_pass=True)

    issues = result["issues"]
    summary = result["summary"]

    # Verify Pass 4 was NOT called (files are independent)
    assert not any(call["prompt"] == CROSS_FILE_PROMPT for call in recorded_calls), \
        "Pass 4 (Cross-file) should NOT run for independent files"

    # Verify API calls: 1 call for Pass 2, 1 call for Pass 3 = 2 calls total
    assert len(recorded_calls) == 2, f"Expected exactly 2 model calls, got {len(recorded_calls)}"

    # Verify all 5 independent defects are in the output (deduplication merged the duplicate divide-by-zero)
    assert len(issues) == 5, f"Expected 5 findings (4 bugs + 1 security), got {len(issues)}"

    # Verify all 5 defects by category and explanation keywords
    reported_explanations = " ".join(i["explanation"] for i in issues).lower()
    assert "subtract" in reported_explanations, "Defect 1 (addition) missing"
    assert "divisor" in reported_explanations or "zero" in reported_explanations, "Defect 2 (division) missing"
    assert "inverted" in reported_explanations or "even" in reported_explanations, "Defect 3 (is_even) missing"
    assert "average" in reported_explanations or "empty" in reported_explanations, "Defect 4 (average) missing"
    assert "traversal" in reported_explanations, "Defect 5 (path traversal) missing"

    # Verify all findings have enriched fields and valid body
    for issue in issues:
        assert issue["path"] in ("calculator.py", "storage.py")
        assert isinstance(issue["line"], int) and issue["line"] > 0
        assert issue["category"] in ("correctness", "error_handling", "edge_case", "security")
        assert issue["severity"] in ("high", "medium")
        assert 0.0 <= issue["confidence"] <= 1.0
        assert issue["explanation"]
        assert issue["impact"]
        assert issue["remediation"]
        assert issue["body"]  # backward-compatible body for review.py / fixer.py


def test_real_orchestration_cross_file_triggering(tmp_path, monkeypatch):
    """
    Verifies that Pass 4 (Cross-file analysis) runs automatically when
    one file imports and calls symbols defined in another changed file.
    """
    repo_dir = tmp_path / "cross_repo"
    repo_dir.mkdir()
    files = create_temporary_buggy_project(repo_dir, with_cross_file=True)

    assert len(files) == 3, "Expected 3 diff files (calculator.py, storage.py, service.py)"

    recorded_prompts = []

    def fake_post(url, headers=None, json=None, timeout=None):
        prompt = json["messages"][0]["content"]
        recorded_prompts.append(prompt)
        return type("R", (), {
            "status_code": 200,
            "headers": CaseInsensitiveDict({"content-type": "application/json"}),
            "json": lambda *args, **kwargs: _make_mock_response([], "pass completed"),
            "raise_for_status": lambda *args, **kwargs: None,
        })()

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-api-key")

    review_diff(files, multi_pass=True)

    # Pass 4 (CROSS_FILE_PROMPT) must have run
    assert CROSS_FILE_PROMPT in recorded_prompts, "Pass 4 Cross-file analysis must execute when cross-file dependencies exist"
    # Total calls must stay well within safe limits (capped at 16)
    assert len(recorded_prompts) <= 16


# ===========================================================================
# Live API Integration Test (runs only when NVIDIA_API_KEY is present)
# ===========================================================================

@pytest.mark.live
def test_live_nvidia_review_with_real_api(tmp_path):
    """
    Sends the 5-defect buggy project diffs directly to the real hosted NVIDIA API
    using the model `nvidia/nemotron-3.5-lightning-30b-a3b`.
    Skipped if NVIDIA_API_KEY is not set in the environment.
    """
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        pytest.skip(
            "NVIDIA_API_KEY not found in environment. "
            "Set $env:NVIDIA_API_KEY='...' to run live integration against NVIDIA Nemotron."
        )

    repo_dir = tmp_path / "live_repo"
    repo_dir.mkdir()
    files = create_temporary_buggy_project(repo_dir, with_cross_file=False)

    result = review_diff(files, multi_pass=True)

    assert isinstance(result, dict)
    assert "summary" in result and result["summary"]
    assert "issues" in result and isinstance(result["issues"], list)

    print(f"\n[LIVE TEST] Real NVIDIA review completed with {len(result['issues'])} findings:")
    for idx, issue in enumerate(result["issues"], start=1):
        print(f"  {idx}. [{issue.get('severity')}] {issue.get('path')}:{issue.get('line')} - {issue.get('category')}")
        print(f"     Explanation: {issue.get('explanation', issue.get('body', ''))[:100]}...")


# ===========================================================================
# Standalone CLI entry point for manual testing
# ===========================================================================

def run_cli():
    parser = argparse.ArgumentParser(
        description="Local Integration Test for AI Code Reviewer Multi-Pass Engine"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute review against the REAL NVIDIA hosted API using $env:NVIDIA_API_KEY",
    )
    parser.add_argument(
        "--cross-file",
        action="store_true",
        help="Include cross-file dependency (service.py -> calculator.py) to trigger Pass 4",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("AI Code Reviewer — Local Multi-Pass Engine Integration Test")
    print("=" * 70)

    with tempfile.TemporaryDirectory(prefix="ai-reviewer-test-") as temp_dir:
        repo_dir = Path(temp_dir)
        print(f"\n[1] Initializing temporary project at: {repo_dir}")
        files = create_temporary_buggy_project(repo_dir, with_cross_file=args.cross_file)

        print(f"[2] Generated {len(files)} realistic Git diffs:")
        for f in files:
            added = _added_lines(f["patch"])
            print(f"    - {f['path']} ({len(added)} added lines)")

        # Structural AST check (Pass 1)
        structure = _analyze_diff_structure(files)
        print(f"[3] Pass 1 (Structural AST Analysis — 0 API calls):")
        print(f"    - Languages detected: {structure['languages']}")
        print(f"    - Cross-file relationships: {structure['has_cross_file_relationships']}")

        if args.live:
            api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
            if not api_key:
                print("\n[ERROR] --live requires NVIDIA_API_KEY in environment!")
                print("Set it via PowerShell:  $env:NVIDIA_API_KEY = 'nvapi-...'")
                sys.exit(1)

            # Masked API key display
            masked_key = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
            print(f"\n[4] Calling REAL hosted NVIDIA API (Key: {masked_key}, Model: {llm_review.MODEL})...")
            print("    Executing Pass 2 (Correctness), Pass 3 (Security)" + (", Pass 4 (Cross-file)" if structure['has_cross_file_relationships'] else "") + ", Pass 5 (Synthesis)...")
            result = review_diff(files, multi_pass=True)
            print("\n[5] Multi-Pass Review Results (Real NVIDIA Model):")
            print("-" * 70)
            print(f"Summary: {result.get('summary', '')}\n")
            print(f"Total Validated Findings: {len(result['issues'])}")
            for idx, issue in enumerate(result["issues"], start=1):
                print(f"\n--- Finding #{idx} ---")
                print(f"  Location:    {issue['path']}:{issue['line']}")
                print(f"  Category:    {issue.get('category')}  |  Severity: {issue.get('severity')}  |  Confidence: {issue.get('confidence')}")
                print(f"  Explanation: {issue.get('explanation')}")
                print(f"  Impact:      {issue.get('impact')}")
                print(f"  Remediation: {issue.get('remediation')}")
            print("-" * 70)
            print("[SUCCESS] Real multi-pass review completed successfully!")
        else:
            print("\n[4] Running deterministic test (mocking external NVIDIA responses)...")
            print("    To run against the real NVIDIA API, pass --live with $env:NVIDIA_API_KEY set.")
            print("    Running pytest on tests/test_local_integration.py...")
            cmd = [sys.executable, "-m", "pytest", "tests/test_local_integration.py", "-k", "not live", "-v"]
            res = subprocess.run(cmd)
            sys.exit(res.returncode)


if __name__ == "__main__":
    run_cli()
