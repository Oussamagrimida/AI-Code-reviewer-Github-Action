"""
Calls NVIDIA's Nemotron endpoint to review a set of PR file diffs and
returns structured feedback: a summary plus a list of issues, each tied
to a specific file and line number.
"""

import os
import json
import re
import time
import requests

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "nvidia/nemotron-3-super-120b-a12b"

# The checklist below exists because early testing showed the model
# reliably catches correctness/edge-case bugs but can miss categories
# that require reasoning about RUNTIME BEHAVIOR rather than just code
# structure (e.g. it fixed a None-handling edge case but walked past an
# N+1 query in the same function). Listing categories explicitly, and
# instructing it to reason about execution cost/scale/timing, closes
# that gap for future bugs in any of these categories -- not just the
# one that was observed missing.
SYSTEM_PROMPT = """You are a senior software engineer doing a pull request code review.

You will be given a set of file diffs (unified diff format, showing only
the CHANGED lines with their new line numbers). Review only the changed
code, but reason about how it BEHAVES AT RUNTIME, not just whether it
reads correctly -- some of the most important bugs only appear when you
think about execution: how many times a loop runs, what happens under
concurrent access, what happens with unexpected or malicious input.

Actively check EVERY one of these categories on every review, not just
the ones that seem obvious at a glance:

1. CORRECTNESS -- logic errors, off-by-one errors, wrong operators,
   incorrect formulas, wrong variable used.
2. SECURITY -- injection (SQL, command, template), insecure
   deserialization (pickle, yaml.load, eval/exec on external input),
   hardcoded secrets/credentials, missing input validation on anything
   that reaches a database, shell, or file system.
3. CONCURRENCY -- race conditions (non-atomic read-modify-write on
   shared state), missing locks, deadlock potential.
4. PERFORMANCE / SCALABILITY -- for every loop, ask: "if this runs
   1,000 or 100,000 times, what does it cost?" Flag database queries,
   network calls, or other I/O placed INSIDE a loop when they could be
   batched into one call outside it (the N+1 pattern). Flag obviously
   inefficient algorithms on data that could be large.
5. RESOURCE MANAGEMENT -- unclosed files, connections, or other
   resources that should use a context manager or explicit cleanup.
6. ERROR HANDLING -- bare `except:` or `except Exception: pass` that
   silently swallows failures; missing handling for operations that can
   fail (network calls, parsing, lookups that may return None/empty).
7. EDGE CASES -- empty inputs, None values, boundary values (first/last
   page, zero, negative numbers), unexpected types.
8. MISSING TESTS -- new logic with no accompanying test coverage.

Do not stop checking after finding one issue -- a single changed
function can have problems in more than one category above (for
example, a loop's body can have BOTH a performance problem AND a
missing edge case). List every distinct issue you find, not just the
first one.

Do not invent issues in unchanged code you cannot see. Do not comment on
trivial style preferences with no real impact.

Respond with ONLY valid JSON, no markdown fences, no explanation outside
the JSON, in exactly this shape:

{
  "summary": "2-3 sentence overall assessment of this PR",
  "issues": [
    {
      "path": "relative/file/path.py",
      "line": 42,
      "severity": "bug" | "security" | "performance" | "concurrency" | "style" | "missing_test",
      "body": "specific, actionable comment about this exact line"
    }
  ]
}

The "line" number MUST be a line number that appears in the diff you were
given (a line that was added/changed), not a made-up number. If you find
no real issues, return an empty "issues" list -- do not invent filler
comments just to have something to say.
"""


def _call_nvidia_with_retry(payload: dict, headers: dict, max_attempts: int = 4) -> dict:
    """
    NVIDIA's free endpoint can occasionally return transient errors
    (502/503/504) under load. Retries with a short backoff before
    giving up, instead of failing the whole review on one bad request.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(NVIDIA_API_URL, json=payload, headers=headers, timeout=120)
            if resp.status_code in (502, 503, 504):
                raise requests.exceptions.HTTPError(
                    f"{resp.status_code} Server Error: {resp.reason}"
                )
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            last_error = e
            print(f"NVIDIA API call failed (attempt {attempt}/{max_attempts}): {e}")
            if attempt < max_attempts:
                wait = 5 * attempt
                print(f"Retrying in {wait}s...")
                time.sleep(wait)
    raise last_error


def review_diff(files: list) -> dict:
    diff_text = "\n\n".join(
        f"--- FILE: {f['path']} ---\n{f['patch']}" for f in files if f.get("patch")
    )

    MAX_CHARS = 24000
    if len(diff_text) > MAX_CHARS:
        diff_text = diff_text[:MAX_CHARS] + "\n\n[... diff truncated for length ...]"

    payload = {
        "model": MODEL,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Review this pull request diff:\n\n{diff_text}"},
        ],
    }
    headers = {
        "Authorization": f"Bearer {os.environ['NVIDIA_API_KEY']}",
        "Content-Type": "application/json",
    }

    response_json = _call_nvidia_with_retry(payload, headers)
    raw_text = response_json["choices"][0]["message"]["content"]

    return _parse_json_response(raw_text)


def _parse_json_response(raw_text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\n?|```$", "", raw_text.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(cleaned)
        data.setdefault("summary", "Automated review completed.")
        data.setdefault("issues", [])
        return data
    except json.JSONDecodeError:
        return {
            "summary": "Automated review ran, but the response couldn't be parsed as structured feedback.",
            "issues": [],
            "_raw": raw_text,
        }