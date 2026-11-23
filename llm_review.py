import ast
import os
import json
import re
import time
from collections.abc import Mapping

import requests


NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
RETRYABLE_NVIDIA_STATUSES = {429, 500, 502, 503, 504}

VALID_CATEGORIES = {
    "correctness",
    "security",
    "concurrency",
    "performance",
    "resource_management",
    "error_handling",
    "edge_case",
    "missing_test",
}

VALID_SEVERITIES = {
    "critical",
    "high",
    "medium",
    "low",
    "info",
}

SEVERITY_ORDER = {
    "critical": 5,
    "high": 4,
    "medium": 3,
    "low": 2,
    "info": 1,
}

ALLOWED_ISSUE_FIELDS = {
    "path",
    "line",
    "category",
    "severity",
    "confidence",
    "body",
    "explanation",
    "impact",
    "remediation",
    "suggested_remediation",
}

CORE_REQUIRED_FIELDS = {
    "path",
    "line",
    "category",
    "severity",
    "confidence",
}

HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)


class ReviewValidationError(ValueError):
    """Raised when the LLM response is not safe to publish as a review."""


class NVIDIAResponseTruncatedError(ReviewValidationError):
    """Raised when NVIDIA reports that a completion was truncated."""


class NVIDIAAPIError(RuntimeError):
    """Raised when NVIDIA cannot return a usable API response."""


FINDING_OUTPUT_CONTRACT = """Review every changed file and every changed function or method before
writing findings. Inspect all changed statements and conditions in each one,
and continue after finding an issue. Report independent defects separately;
do not stop after the first defect or limit findings to one per function.

Use the same strict output schema for every review pass. The top-level object
must contain exactly "summary" and "issues". Each issue must contain exactly
these fields: "path", "line", "category", "severity", "confidence",
"explanation", "impact", and "remediation". All fields are required; use an
empty string for "impact" or "remediation" when there is no additional detail.
Do not add fields or substitute "body" or "suggested_remediation". Keep the
category within the values allowed for this pass. Continue to report only
added-line locations.

Every concrete defect stated in the summary must also appear as a corresponding
structured issue in "issues"; do not report an actionable defect only in the
summary.

Before reporting a regression, compare the removed (-) code with the added (+)
code. Verify that an added line causes the claimed behavior; a nearby changed
line is not sufficient evidence. Do not describe unchanged context as modified.
When a regression is caused by deleted or replaced code, anchor it to the
corresponding added replacement line using its new-file line number, not to the
deleted line or unchanged context. If there is no corresponding added line, do
not report the finding.
For example, if a removed owner filter from a count query is replaced by an
added unfiltered count query, anchor the finding to the added replacement
query line (line 29 in this example), not later query-use lines (lines 30 or
33), unchanged context, or the deleted filter.
For uniqueness checks, trace the full predicate and distinguish excluding the
current user's own record from allowing another user's duplicate value. Report
a duplicate-value vulnerability only if the modified code fails to reject a
conflicting record belonging to someone else.
"""


SYSTEM_PROMPT = """You are a senior software engineer doing a pull request code review.

You will be given file diffs in unified diff format. Report issues only
on added lines (lines prefixed with "+" in a diff hunk), using their
new-file line numbers. Never report issues on unchanged/context lines,
deleted lines, or files not included in the supplied diff. Review only
the changed code, but reason about how it BEHAVES AT RUNTIME, not just
whether it reads correctly.

Actively check EVERY one of these categories:

1. CORRECTNESS
2. SECURITY
3. CONCURRENCY
4. PERFORMANCE / SCALABILITY
5. RESOURCE MANAGEMENT
6. ERROR HANDLING
7. EDGE CASES
8. MISSING TESTS

List every distinct real issue you find.

Do not invent filler comments.

Report only concrete defects supported by the changed code and a realistic
input or state. Do not recommend optional validation, try/except blocks,
fallbacks, or logging unless their absence causes a demonstrated failure for
an input the code accepts. Do not speculate about undocumented callers.
Do not report that open() may raise an I/O or permission exception unless
the changed code demonstrates a required recovery behavior.

Severity definitions:

        - critical: A severe issue that can cause catastrophic impact, major
        security compromise, data loss, system-wide failure, or makes the
        application fundamentally unsafe to deploy.

        - high: A serious issue that can cause significant incorrect behavior,
        security vulnerabilities, data corruption, major reliability problems,
        or substantial production impact.

        - medium: A real issue with meaningful but limited impact, such as
        incorrect behavior in realistic edge cases, moderate performance
        problems, or incomplete error handling.

        - low: A minor real issue with limited impact that should be fixed but
        is unlikely to cause significant production problems.

        - info: A valid observation or improvement suggestion that does not
        represent a meaningful defect or risk.

        Choose the lowest severity that accurately represents the actual impact.
        Do not inflate severity merely because an issue exists.

Confidence scoring:

    - confidence represents how certain you are that the reported issue is
    a real issue in the changed code.
    - 0.9-1.0: Very strong evidence; the issue is clearly real.
    - 0.7-0.89: Strong evidence; the issue is likely real.
    - 0.5-0.69: Moderate evidence; the issue is plausible but has some uncertainty.
    - 0.0-0.49: Weak evidence; do not report the issue unless it is still
    sufficiently actionable.

    Use a number between 0 and 1.
    Do not use confidence to inflate severity.

CRITICAL JSON FORMATTING REQUIREMENTS:
- Output ONLY a single raw JSON object. No markdown fences (no ```json or ```).
- No comments (no // or # or /* */).
- No trailing commas after the last element in any array or object.
- All string values must have newlines escaped as \\n, tabs as \\t, double-quotes as \\".
- Never include raw unescaped control characters (ASCII 0x00-0x1F) inside JSON strings.
- Output nothing before the opening { and nothing after the closing }.

Respond in exactly this shape:

{
  "summary": "2-3 sentence overall assessment of this PR",
  "issues": [
    {
      "path": "relative/file/path.py",
      "line": 42,
      "category": "correctness" | "security" | "concurrency" | "performance" | "resource_management" | "error_handling" | "edge_case" | "missing_test",
      "severity": "critical" | "high" | "medium" | "low" | "info",
      "confidence": 0.94,
      "explanation": "specific explanation of the defect",
      "impact": "concrete consequence",
      "remediation": "actionable fix"
    }
  ]
}

The path MUST be one of the files in the supplied diff.

The line MUST be an ADDED line in the supplied diff
and use its new-file line number.

The confidence MUST be a number between 0 and 1.

If there are no real issues, return an empty issues list.
""" + FINDING_OUTPUT_CONTRACT


CORRECTNESS_PROMPT = """You are a senior software engineer conducting a deep CORRECTNESS AND LOGIC code review.

You will be given unified diffs of pull request changes.
Your mission is to find ALL defects in calculations, logic, algorithms, control flow, error states, and boundary conditions.
Examine EVERY added line (lines starting with "+"). Do not stop after finding the first issue. If there are multiple independent defects, report all of them.
Report only concrete failures for inputs the code accepts. Do not suggest
optional error handling for hypothetical callers or undocumented contracts.
Concrete failures such as division by zero for an accepted zero divisor remain
in scope.
Do not recommend catching exceptions from open() solely because the operation
may fail due to permissions or I/O.

Actively check for:
1. Arithmetic errors: wrong operators, incorrect calculations, operator precedence, division by zero (/ 0, ZeroDivisionError, NaN, modulo by zero).
2. Conditions and comparisons: inverted boolean logic, wrong operators (e.g. `>` vs `>=`, `==` vs `!=`), missing parentheses, incorrect short-circuiting.
3. Boundary & edge cases: empty strings, empty lists/dicts/arrays, 0, -1, None/null/undefined dereference, off-by-one loop/slice boundaries.
4. Return values & state: wrong return type/value, uninitialized variables, mutated arguments, incomplete dictionary lookups (KeyError), missing await.
5. Error handling: unhandled exceptions, incorrect except/catch blocks, suppressing critical exceptions, unhandled None returns.

Severity definitions:
- critical: Application crash, data corruption, calculation error in financial/core transactions.
- high: Serious incorrect behavior, broken business logic, uncaught exceptions under realistic inputs.
- medium: Edge-case failures, subtle boundary bugs, incomplete error handling.
- low: Minor logic flaws, redundant conditions, non-breaking discrepancies.
- info: Minor observation or suggestion.

CRITICAL JSON FORMATTING REQUIREMENTS:
- Output ONLY a single raw JSON object. No markdown fences (no ```json or ```).
- No comments (no // or # or /* */).
- No trailing commas after the last element in any array or object.
- All string values must have newlines escaped as \\n, tabs as \\t, double-quotes as \\".
- Never include raw unescaped control characters (ASCII 0x00-0x1F) inside JSON strings.
- Output nothing before the opening { and nothing after the closing }.

Respond in exactly this shape:
{
  "summary": "2-3 sentence assessment of code correctness and logic",
  "issues": [
    {
      "path": "relative/file/path.py",
      "line": 42,
      "category": "correctness" | "error_handling" | "edge_case",
      "severity": "critical" | "high" | "medium" | "low" | "info",
      "confidence": 0.95,
      "explanation": "Clear, specific technical explanation of the defect.",
      "impact": "Concrete runtime failure or consequence.",
      "remediation": "Exact suggested code correction."
    }
  ]
}

The path MUST be one of the files in the supplied diff.
The line MUST be an ADDED line in the supplied diff and use its new-file line number.
If no issues exist, return an empty issues list.
""" + FINDING_OUTPUT_CONTRACT


SECURITY_RELIABILITY_PROMPT = """You are a senior security engineer and DevSecOps specialist conducting a SECURITY AND RELIABILITY code review.

You will be given unified diffs of pull request changes.
Your mission is to detect ALL security vulnerabilities, injection flaws, data exposure, resource leaks, and concurrency problems.
Examine EVERY added line (lines starting with "+"). Report all distinct vulnerabilities.
Do not report ordinary logic errors as security findings, or recommend
defensive error handling without a demonstrated vulnerability or reliability
failure in the changed code.
Do not report catching open() failures unless the changed code establishes a
required recovery behavior.
For security findings, verify that an added line introduces or weakens the
vulnerable behavior; do not attribute an unchanged permission check to the
change.

Actively check for:
1. Injections: SQL injection, command injection, path traversal (e.g. unvalidated file paths, `../`), template injection, regex DoS (ReDoS), XSS.
2. Authentication & Authorization: missing access checks, broken object level authorization, insecure token validation, privilege escalation.
3. Sensitive Data Exposure: hardcoded secrets/passwords/tokens/API keys, logging passwords or PII, unmasked credentials.
4. Cryptographic issues: insecure hashing (MD5/SHA1 for passwords), hardcoded IVs, weak random number generators.
5. Resource Leaks: unclosed files, database connections, sockets, missing `with` statements or try/finally blocks.
6. Concurrency & Race conditions: non-thread-safe state modification, deadlocks, race conditions in file/database operations.
7. Unsafe APIs: `eval()`, `exec()`, `pickle.loads()`, `shell=True`, unvalidated redirects, unpinned external requests without timeout.

Severity definitions:
- critical: Remote code execution, SQL injection, authentication bypass, hardcoded master credentials.
- high: Path traversal, privilege escalation, resource leak causing denial of service, sensitive data exposure.
- medium: Missing CSRF protection, missing request timeout, missing rate limit, insecure default.
- low: Minor security hardening suggestion, verbose error messages.
- info: Security observation or best practice.

CRITICAL JSON FORMATTING REQUIREMENTS:
- Output ONLY a single raw JSON object. No markdown fences (no ```json or ```).
- No comments (no // or # or /* */).
- No trailing commas after the last element in any array or object.
- All string values must have newlines escaped as \\n, tabs as \\t, double-quotes as \\".
- Never include raw unescaped control characters (ASCII 0x00-0x1F) inside JSON strings.
- Output nothing before the opening { and nothing after the closing }.

Respond in exactly this shape:
{
  "summary": "2-3 sentence assessment of security and reliability",
  "issues": [
    {
      "path": "relative/file/path.py",
      "line": 42,
      "category": "security" | "concurrency" | "resource_management" | "performance",
      "severity": "critical" | "high" | "medium" | "low" | "info",
      "confidence": 0.95,
      "explanation": "Clear, specific vulnerability description.",
      "impact": "Potential security impact or exploitability.",
      "remediation": "Exact remediation or secure coding fix."
    }
  ]
}

The path MUST be one of the files in the supplied diff.
The line MUST be an ADDED line in the supplied diff and use its new-file line number.
If no issues exist, return an empty issues list.
""" + FINDING_OUTPUT_CONTRACT


CROSS_FILE_PROMPT = """You are a senior software architect reviewing CROSS-FILE CONSISTENCY and INTERFACE CONTRACTS across a pull request.

You will be given diffs from multiple related files modified in this pull request.
Your mission is to detect interface mismatches, broken caller-callee relationships, import errors, and inconsistent assumptions between modified files.
Report interface and logic contract defects as correctness or error handling,
not security, unless the mismatch creates a concrete vulnerability.

Actively check for:
1. Changed function/method signatures in one file where callers in another changed file pass wrong arguments, wrong count, or wrong types.
2. Renamed or deleted functions/classes/constants that break imports or calls in another changed file.
3. Inconsistent data models or schemas between API endpoints, services, and database callers.
4. Mismatched configuration settings and code expectations.

CRITICAL JSON FORMATTING REQUIREMENTS:
- Output ONLY a single raw JSON object. No markdown fences (no ```json or ```).
- No comments (no // or # or /* */).
- No trailing commas after the last element in any array or object.
- All string values must have newlines escaped as \\n, tabs as \\t, double-quotes as \\".
- Never include raw unescaped control characters (ASCII 0x00-0x1F) inside JSON strings.
- Output nothing before the opening { and nothing after the closing }.

Respond in exactly this shape:
{
  "summary": "2-3 sentence assessment of cross-file consistency",
  "issues": [
    {
      "path": "relative/file/path.py",
      "line": 42,
      "category": "correctness" | "error_handling",
      "severity": "critical" | "high" | "medium" | "low" | "info",
      "confidence": 0.90,
      "explanation": "Specific interface or cross-file inconsistency description.",
      "impact": "Runtime AttributeError, TypeError, ImportError, or broken contract.",
      "remediation": "How to align the caller and callee."
    }
  ]
}

The path MUST be one of the files in the supplied diff.
The line MUST be an ADDED line in the supplied diff and use its new-file line number.
If no issues exist, return an empty issues list.
""" + FINDING_OUTPUT_CONTRACT


def _analyze_diff_structure(files: list) -> dict:
    structure = {
        "languages": set(),
        "files_info": {},
        "has_cross_file_relationships": False,
        "is_multi_file": len(files) > 1,
        "defined_symbols": {},
        "called_symbols": {},
        "imported_symbols": {},
    }

    extension_map = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".php": "php",
        ".go": "go",
        ".java": "java",
        ".rs": "rust",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".rb": "ruby",
        ".sh": "shell",
        ".bash": "shell",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".md": "markdown",
    }

    for file in files:
        if not isinstance(file, dict):
            continue
        path = file.get("path", "")
        patch = file.get("patch", "")
        _, ext = os.path.splitext(path.lower())
        lang = extension_map.get(ext, "unknown")
        structure["languages"].add(lang)

        added_lines = _added_lines(patch) if patch else set()
        file_info = {
            "path": path,
            "language": lang,
            "added_line_count": len(added_lines),
            "added_lines": sorted(added_lines),
            "modified_functions": [],
            "classes": [],
            "is_config": lang in ("json", "yaml", "toml"),
            "is_doc": lang in ("markdown", "unknown") and ext in (".md", ".rst", ".txt"),
        }

        if lang == "python":
            defined_funcs = set()
            defined_classes = set()
            calls = set()
            imports = set()

            disk_content = None
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        disk_content = f.read()
                except OSError:
                    pass

            if disk_content:
                try:
                    tree = ast.parse(disk_content, filename=path)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            defined_funcs.add(node.name)
                            end_lineno = getattr(node, "end_lineno", node.lineno)
                            func_lines = set(range(node.lineno, end_lineno + 1))
                            if func_lines & added_lines:
                                file_info["modified_functions"].append(node.name)
                        elif isinstance(node, ast.ClassDef):
                            defined_classes.add(node.name)
                        elif isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Name):
                                calls.add(node.func.id)
                            elif isinstance(node.func, ast.Attribute):
                                calls.add(node.func.attr)
                        elif isinstance(node, ast.Import):
                            for alias in node.names:
                                imports.add(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.add(node.module)
                            for alias in node.names:
                                imports.add(alias.name)
                except SyntaxError:
                    pass

            if not file_info["modified_functions"] and patch:
                for line in patch.splitlines():
                    fn_match = re.search(r"^\+?\s*def\s+([A-Za-z0-9_]+)\s*\(", line)
                    if fn_match:
                        file_info["modified_functions"].append(fn_match.group(1))
                        defined_funcs.add(fn_match.group(1))
                    cls_match = re.search(r"^\+?\s*class\s+([A-Za-z0-9_]+)", line)
                    if cls_match:
                        file_info["classes"].append(cls_match.group(1))
                        defined_classes.add(cls_match.group(1))
                    imp_match = re.search(r"^\+?\s*(?:from\s+([A-Za-z0-9_]+)\s+import|import\s+([A-Za-z0-9_]+))", line)
                    if imp_match:
                        imports.add(imp_match.group(1) or imp_match.group(2))
                    if not re.search(r"^\+?\s*def\s+", line):
                        for c in re.findall(r"([A-Za-z0-9_]+)\s*\(", line):
                            if c not in ("def", "class", "if", "for", "while", "return", "with"):
                                calls.add(c)

            structure["defined_symbols"][path] = defined_funcs | defined_classes
            structure["called_symbols"][path] = calls
            structure["imported_symbols"][path] = imports

        elif lang in ("javascript", "typescript", "php", "go", "java"):
            if patch:
                for line in patch.splitlines():
                    fn_match = re.search(r"^\+?\s*(?:function|def|func|public|private)?\s*([A-Za-z0-9_]+)\s*\(", line)
                    if fn_match and fn_match.group(1) not in ("if", "for", "while", "switch", "catch"):
                        file_info["modified_functions"].append(fn_match.group(1))

        structure["files_info"][path] = file_info

    if len(files) > 1:
        for p, calls in structure["called_symbols"].items():
            other_defined = set()
            for other_p, syms in structure["defined_symbols"].items():
                if other_p != p:
                    other_defined |= syms
            if calls & other_defined:
                structure["has_cross_file_relationships"] = True
                break

        if not structure["has_cross_file_relationships"]:
            for p, imps in structure["imported_symbols"].items():
                other_bases = {
                    os.path.splitext(os.path.basename(f["path"]))[0].lower()
                    for f in files
                    if isinstance(f, dict) and "path" in f and f["path"] != p
                }
                if {imp.lower() for imp in imps} & other_bases:
                    structure["has_cross_file_relationships"] = True
                    break

    return structure



def _retry_delay(response, attempt):
    headers = getattr(response, "headers", {})
    retry_after = headers.get("Retry-After") if isinstance(headers, Mapping) else None
    try:
        return min(60, max(0, int(retry_after)))
    except (TypeError, ValueError):
        return 5 * attempt


def _call_nvidia_with_retry(payload, headers, max_attempts=4):
    last_status = None
    last_failure = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                NVIDIA_API_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )
        except requests.exceptions.Timeout:
            last_failure = "network timeout"
            response = None
            last_status = None
        except requests.exceptions.ConnectionError:
            last_failure = "network connection error"
            response = None
            last_status = None
        except requests.RequestException:
            last_failure = "network transport error"
            response = None
            last_status = None
        else:
            last_status = response.status_code
            if last_status == 202:
                raise NVIDIAAPIError(
                    "NVIDIA API returned an asynchronous HTTP 202 response; "
                    "this client supports synchronous completions only."
                )
            retryable = last_status in RETRYABLE_NVIDIA_STATUSES
            if not retryable and last_status >= 400:
                if last_status == 410:
                    detail = (
                        "Verify that the configured model is available to "
                        "this API key on NVIDIA's hosted endpoint."
                    )
                elif last_status in (401, 403):
                    detail = "Check the NVIDIA API key and its endpoint access."
                elif last_status == 429:
                    detail = "NVIDIA rate limits persisted after retries."
                else:
                    detail = "Check the NVIDIA model and request configuration."
                raise NVIDIAAPIError(
                    f"NVIDIA API request failed (HTTP {last_status}). {detail}"
                )
            if not retryable:
                try:
                    return response.json()
                except (ValueError, requests.exceptions.JSONDecodeError):
                    raise NVIDIAAPIError(
                        "NVIDIA API returned an invalid JSON response."
                    ) from None
            last_failure = f"HTTP {last_status}"

        print(
            f"NVIDIA API request failed ({last_failure}), "
            f"attempt {attempt}/{max_attempts}."
        )
        if attempt < max_attempts:
            wait = (
                _retry_delay(response, attempt)
                if response is not None
                else 5 * attempt
            )
            time.sleep(wait)

    if last_status == 429:
        raise NVIDIAAPIError(
            "NVIDIA API rate limit persisted after retries; try again later."
        ) from None
    if last_status is not None:
        raise NVIDIAAPIError(
            f"NVIDIA API temporary failure persisted after retries "
            f"(HTTP {last_status})."
        ) from None
    raise NVIDIAAPIError(
        f"NVIDIA API {last_failure or 'request failure'} persisted after retries."
    ) from None



def _added_lines(patch: str) -> set[int]:
    added = set()
    current_line = None
    new_remaining = 0

    for raw_line in patch.splitlines():
        if raw_line.startswith("@@"):
            match = HUNK_HEADER_RE.match(raw_line)

            if match:
                current_line = int(match.group(3))
                new_remaining = int(match.group(4) or 1)
            else:
                current_line = None

            continue

        if current_line is None:
            continue

        if raw_line.startswith("\\"):
            continue

        if raw_line.startswith("+"):
            if new_remaining > 0:
                added.add(current_line)
                new_remaining -= 1
            current_line += 1
        elif raw_line.startswith("-"):
            continue
        elif raw_line.startswith(" "):
            current_line += 1
            new_remaining = max(0, new_remaining - 1)
        else:
            current_line = None

    return added


def _rejected_finding_location(issue, files_by_path) -> str:
    if not isinstance(issue, dict):
        return "<unknown>:<invalid>"

    path = issue.get("path")
    if not isinstance(path, str) or path not in files_by_path:
        path = "<unknown>"

    line = issue.get("line")
    if isinstance(line, bool) or not isinstance(line, int) or line < 1:
        line = "<invalid>"

    return f"{path}:{line}"


def validate_review(review: dict, files: list) -> dict:
    if not isinstance(review, dict):
        raise ReviewValidationError(
            "Top-level response must be a JSON object."
        )

    summary = review.get("summary")
    issues = review.get("issues")

    if not isinstance(summary, str) or not summary.strip():
        raise ReviewValidationError(
            "'summary' must be a non-empty string."
        )

    if not isinstance(issues, list):
        raise ReviewValidationError(
            "'issues' must be an array."
        )

    allowed_top_level_fields = {
        "summary",
        "issues",
    }

    unexpected_top_level = set(review) - allowed_top_level_fields

    if unexpected_top_level:
        fields = ", ".join(sorted(unexpected_top_level))

        raise ReviewValidationError(
            f"Unexpected top-level field(s): {fields}."
        )

    files_by_path = {
        file["path"]: file
        for file in files
        if (
            isinstance(file, dict)
            and isinstance(file.get("path"), str)
        )
    }

    added_lines_by_path = {
        path: _added_lines(file.get("patch", ""))
        for path, file in files_by_path.items()
    }

    valid_issues = []
    rejected_count = 0

    for index, issue in enumerate(issues):
        try:
            valid_issues.append(
                _validate_issue(
                    issue,
                    index + 1,
                    files_by_path,
                    added_lines_by_path,
                )
            )
        except ReviewValidationError as exc:
            rejected_count += 1
            location = _rejected_finding_location(issue, files_by_path)
            print(
                f"Discarded invalid finding {index + 1} at {location}: "
                f"{exc}"
            )

    if rejected_count:
        print(
            f"Discarded {rejected_count} invalid finding(s) from "
            "the model response; valid findings were retained."
        )

    return {
        "summary": summary.strip(),
        "issues": valid_issues,
    }


def _validate_issue(issue, index, files_by_path, added_lines_by_path):
    if not isinstance(issue, dict):
        raise ReviewValidationError(
            f"Issue {index} must be a JSON object."
        )

    missing = CORE_REQUIRED_FIELDS - issue.keys()
    if missing:
        raise ReviewValidationError(
            f"Issue {index} is missing required field(s): "
            f"{', '.join(sorted(missing))}."
        )

    unexpected = issue.keys() - ALLOWED_ISSUE_FIELDS
    if unexpected:
        raise ReviewValidationError(
            f"Issue {index} has unexpected field(s): "
            f"{', '.join(sorted(unexpected))}."
        )

    if "body" not in issue and "explanation" not in issue:
        raise ReviewValidationError(
            f"Issue {index} must include either 'body' or 'explanation'."
        )

    path = issue["path"]
    line = issue["line"]
    category = issue["category"]
    severity = issue["severity"]
    confidence = issue["confidence"]
    raw_body = issue.get("body")
    raw_explanation = issue.get("explanation")
    raw_impact = issue.get("impact")
    raw_remediation = (
        issue.get("remediation")
        if "remediation" in issue
        else issue.get("suggested_remediation")
    )

    if (
        not isinstance(path, str)
        or not path.strip()
        or any(ord(char) < 32 or ord(char) == 127 for char in path)
    ):
        raise ReviewValidationError(f"Issue {index} has an invalid path.")

    if path not in files_by_path:
        raise ReviewValidationError(f"Issue {index} references an unknown file.")

    if isinstance(line, bool) or not isinstance(line, int) or line < 1:
        raise ReviewValidationError(f"Issue {index} has an invalid line.")

    if line not in added_lines_by_path[path]:
        raise ReviewValidationError(
            f"Issue {index} does not point to an added line."
        )

    if not isinstance(category, str) or category not in VALID_CATEGORIES:
        raise ReviewValidationError(f"Issue {index} has an invalid category.")

    if not isinstance(severity, str) or severity not in VALID_SEVERITIES:
        raise ReviewValidationError(f"Issue {index} has an invalid severity.")

    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        raise ReviewValidationError(f"Issue {index} has invalid confidence.")

    if raw_body is not None and (
        not isinstance(raw_body, str) or not raw_body.strip()
    ):
        raise ReviewValidationError(f"Issue {index} has an invalid body.")

    if raw_explanation is not None and (
        not isinstance(raw_explanation, str) or not raw_explanation.strip()
    ):
        raise ReviewValidationError(f"Issue {index} has an invalid explanation.")

    if raw_impact is not None and not isinstance(raw_impact, str):
        raise ReviewValidationError(f"Issue {index} has an invalid impact.")

    if raw_remediation is not None and not isinstance(raw_remediation, str):
        raise ReviewValidationError(f"Issue {index} has an invalid remediation.")

    impact = raw_impact.strip() if raw_impact else ""
    remediation = raw_remediation.strip() if raw_remediation else ""

    if raw_explanation is not None and raw_explanation.strip():
        explanation = raw_explanation.strip()
    else:
        explanation = raw_body.strip()

    if raw_body is not None and raw_body.strip():
        body = raw_body.strip()
    else:
        body_parts = [explanation]
        if impact:
            body_parts.append(f"**Impact:** {impact}")
        if remediation:
            body_parts.append(f"**Remediation:** {remediation}")
        body = "\n\n".join(body_parts)

    return {
        "path": path,
        "line": line,
        "category": category,
        "severity": severity,
        "confidence": confidence,
        "body": body,
        "explanation": explanation,
        "impact": impact,
        "remediation": remediation,
    }


def _split_diff_into_chunks(files, max_chars=20000, max_files_per_chunk=2):
    hunk_header_re = re.compile(
        r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
    )

    def split_patch(path, patch):
        lines = patch.splitlines(keepends=True)
        pieces = []
        prefix = []
        hunks = []
        current_hunk = None

        for line in lines:
            if line.startswith("@@ "):
                if current_hunk is not None:
                    hunks.append(current_hunk)
                current_hunk = [line]
            elif current_hunk is None:
                prefix.append(line)
            else:
                current_hunk.append(line)

        if current_hunk is not None:
            hunks.append(current_hunk)

        if not hunks:
            return [patch]

        current_piece = "".join(prefix)
        for hunk in hunks:
            hunk_text = "".join(hunk)

            if len(hunk_text) > max_chars:
                if current_piece.strip():
                    pieces.append(current_piece)
                    current_piece = ""

                match = hunk_header_re.match(hunk[0].rstrip("\r\n"))
                if not match:
                    max_piece_chars = max(
                        1,
                        max_chars - len(f"--- FILE: {path} ---\n") - 2,
                    )

                    for start in range(0, len(hunk_text), max_piece_chars):
                        pieces.append(hunk_text[start:start + max_piece_chars])

                    continue

                old_line = int(match.group(1))
                old_count = int(match.group(2) or 1)
                new_line = int(match.group(3))
                new_count = int(match.group(4) or 1)
                suffix = match.group(5)

                body = hunk[1:]
                segment = []
                segment_old = old_line
                segment_new = new_line
                segment_old_count = 0
                segment_new_count = 0
                segment_size = 0

                def flush_segment():
                    nonlocal segment, segment_old, segment_new
                    nonlocal segment_old_count, segment_new_count
                    nonlocal segment_size

                    if not segment:
                        return

                    old_start = (
                        segment_old
                        if segment_old_count
                        else max(0, segment_old - 1)
                    )
                    new_start = (
                        segment_new
                        if segment_new_count
                        else max(0, segment_new - 1)
                    )

                    pieces.append(
                        f"@@ -{old_start},{segment_old_count} "
                        f"+{new_start},{segment_new_count} @@{suffix}\n"
                        + "".join(segment)
                    )

                    segment = []
                    segment_old = old_line
                    segment_new = new_line
                    segment_old_count = 0
                    segment_new_count = 0
                    segment_size = 0

                for body_line in body:
                    if segment and segment_size + len(body_line) > max_chars - 100:
                        flush_segment()
                        segment_old = old_line
                        segment_new = new_line

                    segment.append(body_line)
                    segment_size += len(body_line)

                    if body_line.startswith((" ", "-")):
                        segment_old_count += 1
                        old_line += 1

                    if body_line.startswith((" ", "+")):
                        segment_new_count += 1
                        new_line += 1

                    if body_line.startswith("\\"):
                        continue

                    segment_old = old_line - segment_old_count
                    segment_new = new_line - segment_new_count

                flush_segment()
                continue

            if current_piece and len(current_piece) + len(hunk_text) > max_chars:
                pieces.append(current_piece)
                current_piece = ""

            current_piece += hunk_text

        if current_piece:
            pieces.append(current_piece)

        return pieces

    chunks = []
    current_files = []
    current_size = 0

    for file in files:
        patch = file.get("patch", "")
        if not patch:
            continue

        path = file["path"]
        pieces = split_patch(path, patch)

        for piece in pieces:
            file_piece = {**file, "patch": piece}
            file_text_size = len(f"--- FILE: {path} ---\n{piece}") + 2

            if current_files and (
                len(current_files) >= max_files_per_chunk
                or current_size + file_text_size > max_chars
            ):
                chunks.append(current_files)
                current_files = []
                current_size = 0

            current_files.append(file_piece)
            current_size += file_text_size

            if file_text_size > max_chars:
                chunks.append(current_files)
                current_files = []
                current_size = 0

    if current_files:
        chunks.append(current_files)

    return chunks


_STOP_WORDS = {
    "the", "and", "this", "that", "with", "from", "for", "are", "was",
    "were", "has", "have", "had", "not", "but", "can", "could", "should",
    "would", "may", "might", "must", "will", "shall", "does", "did", "doing",
    "been", "being", "here", "there", "when", "where", "which", "while",
    "about", "into", "through", "after", "before", "above", "below", "between",
    "each", "every", "some", "any", "all", "both", "such", "than", "then",
    "too", "very", "also", "line", "code", "issue", "defect", "problem", "bug",
    "file", "function", "method", "class", "value", "using", "used", "call",
    "error", "exception", "crash", "raise", "raises", "zerodivision",
}


def _tokenize_text(text: str) -> set[str]:
    if not isinstance(text, str):
        return set()
    words = re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())
    stems = set()
    for w in words:
        if w in _STOP_WORDS:
            continue
        for suffix in ("error", "ing", "tion", "sion", "s", "ed", "ment"):
            if len(w) > len(suffix) + 3 and w.endswith(suffix):
                w = w[:-len(suffix)]
                break
        if w not in _STOP_WORDS:
            stems.add(w)
    return stems


_SECURITY_CATEGORY_TERMS = (
    "path traversal",
    "directory traversal",
    "traversal",
    "sql injection",
    "command injection",
    "template injection",
    "injection vulnerability",
    "cross-site scripting",
    "xss",
    "authentication bypass",
    "authorization bypass",
    "privilege escalation",
    "hardcoded secret",
    "hard-coded secret",
    "credential exposure",
    "credential leakage",
    "arbitrary file",
    "remote code execution",
    "command execution",
    "unsafe deserialization",
    "unvalidated redirect",
    "shell=true",
)

_LOGIC_CATEGORY_TERMS = (
    "subtract",
    "subtraction",
    "instead of adding",
    "wrong arithmetic",
    "incorrect arithmetic",
    "wrong operator",
    "inverted",
    "is_even",
    "calculate_average",
    "division by zero",
    "divide by zero",
    "zero divisor",
    "empty list",
    "incorrect result",
    "wrong result",
    "off-by-one",
    "off by one",
    "returns true for odd",
    "returns false for even",
    "logic error",
    "a - b",
)


def _normalize_issue_category(issue: dict) -> str:
    category = issue.get("category", "")
    text = " ".join(
        value
        for field in ("body", "explanation", "impact", "remediation")
        if isinstance((value := issue.get(field)), str)
    ).lower()

    if any(term in text for term in _SECURITY_CATEGORY_TERMS):
        return "security"

    if (
        category in ("security", "correctness", "error_handling", "edge_case")
        and any(term in text for term in _LOGIC_CATEGORY_TERMS)
    ):
        return "correctness"

    return category


def _is_speculative_open_error_handling(issue: dict) -> bool:
    if issue.get("category") not in ("error_handling", "security"):
        return False

    text = " ".join(
        value
        for field in ("body", "explanation", "impact", "remediation")
        if isinstance((value := issue.get(field)), str)
    ).lower()
    if any(term in text for term in _SECURITY_CATEGORY_TERMS):
        return False
    if not re.search(r"\bopen\s*\(", text):
        return False

    return any(
        term in text
        for term in (
            "permissionerror",
            "oserror",
            "ioerror",
            "try/except",
            "try except",
            "catch the",
            "catch this",
            "catch exception",
            "handle the exception",
            "handle this exception",
        )
    )


def _are_duplicate_issues(issue1: dict, issue2: dict) -> bool:
    if not isinstance(issue1, dict) or not isinstance(issue2, dict):
        return False

    if issue1.get("path") != issue2.get("path"):
        return False

    line1 = issue1.get("line")
    line2 = issue2.get("line")
    if not isinstance(line1, int) or not isinstance(line2, int):
        return False

    cat1 = _normalize_issue_category(issue1)
    cat2 = _normalize_issue_category(issue2)

    same_reported_category = issue1.get("category") == issue2.get("category")
    same_category = cat1 == cat2
    line_diff = abs(line1 - line2)
    if line_diff > 2:
        return False

    body1 = issue1.get("body", "").strip()
    body2 = issue2.get("body", "").strip()
    exp1 = issue1.get("explanation", "").strip()
    exp2 = issue2.get("explanation", "").strip()

    if same_category and (
        (body1 and body1 == body2) or (exp1 and exp1 == exp2)
    ):
        return True

    tokens1 = _tokenize_text(f"{exp1} {body1} {issue1.get('impact', '')}")
    tokens2 = _tokenize_text(f"{exp2} {body2} {issue2.get('impact', '')}")

    if not tokens1 or not tokens2:
        return False

    inter = tokens1 & tokens2
    min_len = min(len(tokens1), len(tokens2)) if min(len(tokens1), len(tokens2)) > 0 else 1
    overlap = len(inter) / min_len

    if line_diff == 0 and same_reported_category:
        return len(inter) >= 2 and overlap >= 0.25

    if line_diff == 1 and same_reported_category:
        return len(inter) >= 2 and overlap >= 0.40

    if line_diff == 2 and same_reported_category:
        return len(inter) >= 3 and overlap >= 0.50

    if line_diff == 0:
        return len(inter) >= 3 and overlap >= 0.55

    if line_diff == 1:
        return len(inter) >= 4 and overlap >= 0.60

    return len(inter) >= 5 and overlap >= 0.65


def _merge_duplicate_issues(issue1: dict, issue2: dict) -> dict:
    sev1 = SEVERITY_ORDER.get(issue1.get("severity", "info"), 1)
    sev2 = SEVERITY_ORDER.get(issue2.get("severity", "info"), 1)
    chosen_severity = issue1.get("severity") if sev1 >= sev2 else issue2.get("severity")

    try:
        conf1 = float(issue1.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf1 = 0.0
    try:
        conf2 = float(issue2.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf2 = 0.0
    chosen_confidence = max(conf1, conf2)

    category1 = _normalize_issue_category(issue1)
    category2 = _normalize_issue_category(issue2)
    if category1 == category2:
        chosen_category = category1
    else:
        combined_text = " ".join(
            issue.get(field, "")
            for issue in (issue1, issue2)
            for field in ("body", "explanation", "impact", "remediation")
            if isinstance(issue.get(field), str)
        ).lower()
        if any(term in combined_text for term in _SECURITY_CATEGORY_TERMS):
            chosen_category = "security"
        elif any(term in combined_text for term in _LOGIC_CATEGORY_TERMS):
            chosen_category = "correctness"
        else:
            chosen_category = category1

    line1 = issue1.get("line", 1)
    line2 = issue2.get("line", 1)
    chosen_line = min(line1, line2)

    exp1 = issue1.get("explanation", "").strip()
    exp2 = issue2.get("explanation", "").strip()
    if len(exp2) > len(exp1):
        chosen_explanation = exp2
    else:
        chosen_explanation = exp1 or exp2

    imp1 = issue1.get("impact", "").strip()
    imp2 = issue2.get("impact", "").strip()
    chosen_impact = imp2 if len(imp2) > len(imp1) else imp1

    rem1 = issue1.get("remediation", "").strip()
    rem2 = issue2.get("remediation", "").strip()
    chosen_remediation = rem2 if len(rem2) > len(rem1) else rem1

    body_parts = [chosen_explanation]
    if chosen_impact and f"**Impact:**" not in chosen_explanation:
        body_parts.append(f"**Impact:** {chosen_impact}")
    if chosen_remediation and f"**Remediation:**" not in chosen_explanation:
        body_parts.append(f"**Remediation:** {chosen_remediation}")
    chosen_body = "\n\n".join(body_parts)

    return {
        "path": issue1.get("path"),
        "line": chosen_line,
        "category": chosen_category,
        "severity": chosen_severity,
        "confidence": chosen_confidence,
        "body": chosen_body,
        "explanation": chosen_explanation,
        "impact": chosen_impact,
        "remediation": chosen_remediation,
    }


def _semantic_deduplicate_issues(issues: list) -> list:
    if not isinstance(issues, list):
        return []

    unique = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        issue = dict(issue)
        issue["category"] = _normalize_issue_category(issue)
        if _is_speculative_open_error_handling(issue):
            print(
                "Discarded speculative optional error-handling finding at "
                f"{issue.get('path')}:{issue.get('line')}."
            )
            continue
        merged = False
        for idx, existing in enumerate(unique):
            if _are_duplicate_issues(existing, issue):
                unique[idx] = _merge_duplicate_issues(existing, issue)
                merged = True
                break
        if not merged:
            unique.append(issue)

    return unique


def _deduplicate_issues(issues: list) -> list:
    return _semantic_deduplicate_issues(issues)

def review_diff(files: list) -> dict:
    chunks = _split_diff_into_chunks(
        files,
        max_chars=20000,
        max_files_per_chunk=2,
    )

    if not chunks:
        return {
            "summary": "No reviewable diff content was available.",
            "issues": [],
        }

    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise NVIDIAAPIError(
            "NVIDIA_API_KEY is not configured; cannot request a review."
        )

def _call_model_for_prompt(
    prompt: str,
    diff_text: str,
    headers: dict,
    budget_tracker: dict | None = None,
) -> dict:
    """Call the NVIDIA model and parse the JSON response.

    Parameters
    ----------
    prompt:
        System-role prompt text.
    diff_text:
        The unified diff to review.
    headers:
        HTTP request headers (must include Authorization).
    budget_tracker:
        A mutable dict ``{"calls": int, "max_calls": int}`` shared across all
        calls within a single review run.  If *None*, a local tracker scoped
        to this single call is created.  Every API call (including repair
        attempts) is counted against this budget.  When the budget is
        exhausted the function raises ``ReviewValidationError`` immediately
        rather than making another API request.
    """
    if budget_tracker is None:
        budget_tracker = {"calls": 0, "max_calls": 16}

    if budget_tracker["calls"] >= budget_tracker["max_calls"]:
        raise ReviewValidationError(
            "API call budget exhausted; no further model calls will be made."
        )

    payload = {
        "model": MODEL,
        "temperature": 0.1,
        "max_tokens": 8192,
        "stream": False,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {
            "enable_thinking": False
        },
        "messages": [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": (
                    "Review this pull request diff:\n\n"
                    f"{diff_text}"
                ),
            },
        ],
    }

    budget_tracker["calls"] += 1
    try:
        response_json = _call_nvidia_with_retry(payload, headers)
    except NVIDIAAPIError as exc:
        # If the endpoint rejected response_format, retry without it
        err_str = str(exc).lower()
        if "400" in err_str or "422" in err_str or "response_format" in err_str:
            payload_no_rf = {k: v for k, v in payload.items() if k != "response_format"}
            # This retry does NOT count against the budget (it replaces the failed call)
            budget_tracker["calls"] -= 1
            if budget_tracker["calls"] >= budget_tracker["max_calls"]:
                raise
            budget_tracker["calls"] += 1
            response_json = _call_nvidia_with_retry(payload_no_rf, headers)
        else:
            raise

    choices = (
        response_json.get("choices")
        if isinstance(response_json, dict)
        else None
    )
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], dict)
    ):
        raise ReviewValidationError(
            "NVIDIA response does not contain choices[0].message.content."
        )

    choice = choices[0]
    message = choice.get("message")
    raw_text = (
        message.get("content")
        if isinstance(message, dict)
        else None
    )
    finish_reason = choice.get("finish_reason")
    if raw_text is None:
        raise ReviewValidationError(
            "NVIDIA response does not contain choices[0].message.content."
        )

    if finish_reason == "length":
        raise NVIDIAResponseTruncatedError(
            "NVIDIA response was truncated; reduce the diff size and retry."
        )

    try:
        return _parse_json_response(raw_text)
    except ReviewValidationError as parse_err:
        # Bounded repair: one additional call to ask the model to fix its output
        if budget_tracker["calls"] >= budget_tracker["max_calls"]:
            raise ReviewValidationError(
                "LLM response could not be parsed and the API call budget is "
                "exhausted; cannot attempt repair."
            ) from parse_err

        # Only repair string responses; non-string content is a structural
        # protocol error from the API — re-raise immediately.
        if not isinstance(raw_text, str):
            raise

        truncated_output = raw_text[:2000]
        repair_messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    "Review this pull request diff:\n\n"
                    f"{diff_text}"
                ),
            },
            {"role": "assistant", "content": truncated_output},
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON. "
                    "Return ONLY a valid JSON object with no markdown fences, "
                    "no comments, no trailing commas, and all control characters "
                    "escaped (newlines as \\\\n, tabs as \\\\t). "
                    "Output nothing except the JSON object."
                ),
            },
        ]
        repair_payload = {
            "model": MODEL,
            "temperature": 0.0,
            "max_tokens": 8192,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": repair_messages,
        }
        budget_tracker["calls"] += 1
        repair_response = _call_nvidia_with_retry(repair_payload, headers)

        repair_choices = (
            repair_response.get("choices")
            if isinstance(repair_response, dict)
            else None
        )
        if (
            not isinstance(repair_choices, list)
            or not repair_choices
            or not isinstance(repair_choices[0], dict)
        ):
            raise ReviewValidationError(
                "Repair response does not contain choices[0].message.content."
            ) from parse_err

        repair_message = repair_choices[0].get("message")
        repair_raw = (
            repair_message.get("content")
            if isinstance(repair_message, dict)
            else None
        )
        if repair_raw is None:
            raise ReviewValidationError(
                "Repair response does not contain choices[0].message.content."
            ) from parse_err

        return _parse_json_response(repair_raw)


def _run_single_pass_review(chunks: list, headers: dict) -> dict:
    all_issues = []
    summaries = []

    budget_tracker = {"calls": 0, "max_calls": 16}

    for chunk in chunks:
        diff_text = "\n\n".join(
            f"--- FILE: {f['path']} ---\n{f['patch']}"
            for f in chunk
            if f.get("patch")
        )
        if not diff_text.strip():
            continue

        review = _call_model_for_prompt(SYSTEM_PROMPT, diff_text, headers, budget_tracker)
        validated = validate_review(review, chunk)
        summaries.append(validated["summary"])
        all_issues.extend(validated["issues"])

    return {
        "summary": " ".join(summaries),
        "issues": _deduplicate_issues(all_issues),
    }


def _run_multi_pass_review(files: list, chunks: list, headers: dict) -> dict:
    # Pass 1: Structural Analysis (0 API calls)
    structure = _analyze_diff_structure(files)

    all_issues = []
    summaries = []

    # Shared budget tracker: counts every API call including repair attempts
    budget_tracker = {"calls": 0, "max_calls": 16}

    for chunk_index, chunk in enumerate(chunks, start=1):
        diff_text = "\n\n".join(
            f"--- FILE: {f['path']} ---\n{f['patch']}"
            for f in chunk
            if f.get("patch")
        )
        if not diff_text.strip():
            continue

        chunk_has_code = any(
            structure["files_info"].get(f["path"], {}).get("language") not in ("markdown", "unknown")
            or not structure["files_info"].get(f["path"], {}).get("is_doc", False)
            for f in chunk
        )

        if not chunk_has_code:
            primary_passes = [("general", SYSTEM_PROMPT)]
        else:
            primary_passes = [
                ("correctness", CORRECTNESS_PROMPT),
                ("security and reliability", SECURITY_RELIABILITY_PROMPT),
            ]

        completed_primary_passes = 0
        truncated_passes = []
        for pass_name, prompt in primary_passes:
            if budget_tracker["calls"] >= budget_tracker["max_calls"]:
                continue
            try:
                pass_review = _call_model_for_prompt(
                    prompt, diff_text, headers, budget_tracker
                )
            except NVIDIAResponseTruncatedError as exc:
                truncated_passes.append(exc)
                print(
                    f"Skipping {pass_name} pass for diff chunk "
                    f"{chunk_index}/{len(chunks)}: response was truncated."
                )
                continue

            validated_pass = validate_review(pass_review, chunk)
            completed_primary_passes += 1
            summaries.append(validated_pass["summary"])
            all_issues.extend(validated_pass["issues"])

        if completed_primary_passes == 0:
            if truncated_passes:
                raise NVIDIAResponseTruncatedError(
                    f"No primary pass completed for diff chunk "
                    f"{chunk_index}/{len(chunks)}; all attempted primary "
                    "passes were truncated."
                ) from truncated_passes[-1]
            raise ReviewValidationError(
                f"No primary pass completed for diff chunk "
                f"{chunk_index}/{len(chunks)} because the API call budget "
                "was exhausted."
            )

    # Pass 4: Conditional Cross-File Analysis
    if structure["has_cross_file_relationships"] and len(files) > 1 and budget_tracker["calls"] < budget_tracker["max_calls"]:
        cross_diff = "\n\n".join(
            f"--- FILE: {f['path']} ---\n{f['patch']}"
            for f in files
            if f.get("patch")
        )
        if len(cross_diff) > 30000:
            cross_diff = cross_diff[:30000] + "\n\n[Diff truncated for cross-file scan]"

        try:
            review_cross = _call_model_for_prompt(CROSS_FILE_PROMPT, cross_diff, headers, budget_tracker)
            validated_cross = validate_review(review_cross, files)
            all_issues.extend(validated_cross["issues"])
            if validated_cross.get("summary"):
                summaries.append(validated_cross["summary"])
        except NVIDIAResponseTruncatedError:
            print(
                "Skipping cross-file pass for all files: "
                "NVIDIA response was truncated."
            )

    # Pass 5: Synthesis & Semantic Deduplication
    deduped_issues = _semantic_deduplicate_issues(all_issues)

    combined_summary = " ".join(summaries).strip()
    if not combined_summary:
        combined_summary = "Multi-pass automated review completed."

    return {
        "summary": combined_summary,
        "issues": deduped_issues,
    }


def review_diff(files: list, multi_pass: bool | None = None) -> dict:
    if multi_pass is None:
        multi_pass = os.environ.get("ENABLE_MULTI_PASS", "false").strip().lower() in ("true", "1", "yes")

    chunks = _split_diff_into_chunks(
        files,
        max_chars=20000,
        max_files_per_chunk=2,
    )

    if not chunks:
        return {
            "summary": "No reviewable diff content was available.",
            "issues": [],
        }

    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise NVIDIAAPIError(
            "NVIDIA_API_KEY is not configured; cannot request a review."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if not multi_pass:
        return _run_single_pass_review(chunks, headers)

    return _run_multi_pass_review(files, chunks, headers)



def _parse_json_response(raw_text: str) -> dict:
    """Multi-stage JSON parser that handles common LLM output quirks.

    Parsing ladder (each step is tried in order):
      1. Strip markdown fences if present.
      2. Standard strict json.loads.
      3. json.loads with strict=False — accepts unescaped \\n, \\r, \\t inside
         string values (the most common cause of live parse failures).
      4. Extract the outermost {...} and retry with strict=False — handles
         conversational preamble or postamble added by the model.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ReviewValidationError("LLM response is empty.")

    cleaned = raw_text.strip()

    # Step 1: strip markdown fences
    if "```" in cleaned:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, flags=re.IGNORECASE)
        if m:
            cleaned = m.group(1).strip()
        else:
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    # Step 2: standard strict parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
        raise ReviewValidationError("LLM response must be a JSON object.")
    except json.JSONDecodeError:
        pass

    # Step 3: non-strict parse (tolerates unescaped control chars in strings)
    try:
        data = json.loads(cleaned, strict=False)
        if isinstance(data, dict):
            return data
        raise ReviewValidationError("LLM response must be a JSON object.")
    except json.JSONDecodeError:
        pass

    # Step 4: extract outermost {...} to skip preamble/postamble
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last > first:
        candidate = cleaned[first : last + 1]
        try:
            data = json.loads(candidate, strict=False)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # All stages failed — report the structural reason without leaking content
    try:
        json.loads(cleaned, strict=False)
    except json.JSONDecodeError as exc:
        raise ReviewValidationError(
            f"LLM response is not valid JSON: {exc.msg}."
        ) from exc

    raise ReviewValidationError("LLM response must be a JSON object.")

