import os
import json
import re
import time
import requests


NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

MODEL = "nvidia/nemotron-3-super-120b-a12b"

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
    r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@"
)


class ReviewValidationError(ValueError):
    """Raised when the LLM response is not safe to publish as a review."""


SYSTEM_PROMPT = """You are a senior software engineer doing a pull request code review.

You will be given a set of file diffs (unified diff format, showing only
the CHANGED lines with their new line numbers). Review only the changed
code, but reason about how it BEHAVES AT RUNTIME, not just whether it
reads correctly.

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



def _call_nvidia_with_retry(payload, headers, max_attempts=4):
    last_error = None
    retryable_statuses = {502, 503, 504}

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                NVIDIA_API_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
        ) as error:
            last_error = error
        else:
            if response.status_code in retryable_statuses:
                last_error = requests.exceptions.HTTPError(
                    f"{response.status_code} Server Error: {response.reason}"
                )
            else:
                response.raise_for_status()
                return response.json()

        print(
            f"NVIDIA API call failed (attempt {attempt}/{max_attempts}): "
            f"{last_error}"
        )

        if attempt < max_attempts:
            wait = 5 * attempt
            print(f"Retrying in {wait}s...")
            time.sleep(wait)

    raise last_error



def _added_lines(patch: str) -> set[int]:
    added = set()
    current_line = None

    for raw_line in patch.splitlines():
        line = raw_line.lstrip()

        if line.startswith("@@"):
            match = HUNK_HEADER_RE.match(line)

            if match:
                current_line = int(match.group(1))
            else:
                current_line = None

            continue

        if current_line is None:
            continue

        if line.startswith("\\"):
            continue

        if line.startswith("+++"):
            continue

        if line.startswith("+"):
            added.add(current_line)
            current_line += 1

        elif line.startswith("-"):
            continue

        else:
            current_line += 1

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

    required_fields = {
        "path",
        "line",
        "category",
        "severity",
        "confidence",
        "body",
    }

    for index, issue in enumerate(issues):
        if not isinstance(issue, dict):
            raise ReviewValidationError(
                f"Issue {index} must be a JSON object."
            )

        missing = required_fields - issue.keys()

        if missing:
            fields = ", ".join(sorted(missing))

            raise ReviewValidationError(
                f"Issue {index} is missing required field(s): {fields}."
            )

        unexpected = issue.keys() - required_fields

        if unexpected:
            fields = ", ".join(sorted(unexpected))

            raise ReviewValidationError(
                f"Issue {index} has unexpected field(s): {fields}."
            )

        path = issue["path"]
        line = issue["line"]
        category = issue["category"]
        severity = issue["severity"]
        confidence = issue["confidence"]
        body = issue["body"]

        if not isinstance(path, str) or not path.strip():
            raise ReviewValidationError(
                f"Issue {index} has an invalid 'path'."
            )

        if path not in files_by_path:
            raise ReviewValidationError(
                f"Issue {index} references file '{path}', "
                "which is not in the PR diff."
            )

        if (
            isinstance(line, bool)
            or not isinstance(line, int)
            or line < 1
        ):
            raise ReviewValidationError(
                f"Issue {index} has an invalid 'line'."
            )

        if line not in added_lines_by_path[path]:
            raise ReviewValidationError(
                f"Issue {index} points to {path}:{line}, "
                "which is not an added line in the PR diff."
            )

        if (
            not isinstance(category, str)
            or category not in VALID_CATEGORIES
        ):
            allowed = ", ".join(sorted(VALID_CATEGORIES))

            raise ReviewValidationError(
                f"Issue {index} has invalid category '{category}'. "
                f"Allowed: {allowed}."
            )

        if (
            not isinstance(severity, str)
            or severity not in VALID_SEVERITIES
        ):
            allowed = ", ".join(sorted(VALID_SEVERITIES))

            raise ReviewValidationError(
                f"Issue {index} has invalid severity '{severity}'. "
                f"Allowed: {allowed}."
            )

        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
        ):
            raise ReviewValidationError(
                f"Issue {index} has an invalid 'confidence'."
            )

        if not 0 <= confidence <= 1:
            raise ReviewValidationError(
                f"Issue {index} confidence must be between 0 and 1."
            )

        if not isinstance(body, str) or not body.strip():
            raise ReviewValidationError(
                f"Issue {index} has an invalid 'body'."
            )

    return review


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

        api_key = os.environ.get("NVIDIA_API_KEY", "test-key")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        response_json = _call_nvidia_with_retry(
            payload,
            headers,
        )

        try:
            raw_text = response_json["choices"][0]["message"]["content"]

        except (KeyError, IndexError, TypeError) as exc:
            raise ReviewValidationError(
                "NVIDIA response does not contain "
                "choices[0].message.content."
            ) from exc

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

