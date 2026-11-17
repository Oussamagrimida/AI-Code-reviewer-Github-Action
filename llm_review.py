import os
import json
import re
import time
from collections.abc import Mapping

import requests


NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
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

HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@"
)


class ReviewValidationError(ValueError):
    """Raised when the LLM response is not safe to publish as a review."""


class NVIDIAAPIError(RuntimeError):
    """Raised when NVIDIA cannot return a usable API response."""


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

Respond with ONLY valid JSON, no markdown fences,
in exactly this shape:

{
  "summary": "2-3 sentence overall assessment of this PR",
  "issues": [
    {
      "path": "relative/file/path.py",
      "line": 42,
      "category": "correctness" | "security" | "concurrency" | "performance" | "resource_management" | "error_handling" | "edge_case" | "missing_test",
      "severity": "critical" | "high" | "medium" | "low" | "info",
      "confidence": 0.94,
      "body": "specific, actionable comment about this exact changed line"
    }
  ]
}

The path MUST be one of the files in the supplied diff.

The line MUST be an ADDED line in the supplied diff
and use its new-file line number.

The confidence MUST be a number between 0 and 1.

If there are no real issues, return an empty issues list.
"""



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
                    index,
                    files_by_path,
                    added_lines_by_path,
                )
            )
        except ReviewValidationError:
            rejected_count += 1

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

    required_fields = {
        "path",
        "line",
        "category",
        "severity",
        "confidence",
        "body",
    }

    missing = required_fields - issue.keys()
    if missing:
        raise ReviewValidationError(
            f"Issue {index} is missing required fields."
        )

    if issue.keys() - required_fields:
        raise ReviewValidationError(
            f"Issue {index} has unexpected fields."
        )

    path = issue["path"]
    line = issue["line"]
    category = issue["category"]
    severity = issue["severity"]
    confidence = issue["confidence"]
    body = issue["body"]

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

    if not isinstance(body, str) or not body.strip():
        raise ReviewValidationError(f"Issue {index} has an invalid body.")

    return issue


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


def _deduplicate_issues(issues: list) -> list:
    seen = set()
    unique = []

    for issue in issues:
        key = (
            issue["path"],
            issue["line"],
            issue["body"].strip(),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(issue)

    return unique

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

    all_issues = []
    summaries = []

    for chunk in chunks:
        diff_text = "\n\n".join(
            f"--- FILE: {f['path']} ---\n{f['patch']}"
            for f in chunk
            if f.get("patch")
        )

        payload = {
            "model": MODEL,
            "temperature": 0.1,
            "max_tokens": 8192,
            "stream": False,
            "chat_template_kwargs": {
                "enable_thinking": False
            },
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
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

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response_json = _call_nvidia_with_retry(
            payload,
            headers,
        )

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
                "NVIDIA response does not contain "
                "choices[0].message.content."
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
                "NVIDIA response does not contain "
                "choices[0].message.content."
            )

        if finish_reason == "length":
            raise ReviewValidationError(
                "NVIDIA response was truncated; reduce the diff size and retry."
            )

        review = _parse_json_response(raw_text)

        validated = validate_review(
            review,
            chunk,
        )

        summaries.append(validated["summary"])
        all_issues.extend(validated["issues"])

    return {
        "summary": " ".join(summaries),
        "issues": _deduplicate_issues(all_issues),
    }



def _parse_json_response(raw_text: str) -> dict:
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ReviewValidationError(
            "LLM response is empty."
        )

    cleaned = raw_text.strip()

    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

    try:
        data = json.loads(cleaned)

    except json.JSONDecodeError as exc:
        raise ReviewValidationError(
            f"LLM response is not valid JSON: {exc.msg}."
        ) from exc

    if not isinstance(data, dict):
        raise ReviewValidationError(
            "LLM response must be a JSON object."
        )

    return data

