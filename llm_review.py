"""
Calls NVIDIA's Nemotron endpoint to review a set of PR file diffs and
returns structured feedback: a summary plus a list of issues, each tied
to a specific file and line number.

We ask the model for JSON, not prose, because the output has to be
programmatically posted as inline PR comments -- "review this code" as
free text isn't usable by the GitHub API, which needs an exact file path
and line number per comment.
"""

import os
import json
import re
import requests

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "nvidia/nemotron-3-super-120b-a12b"

SYSTEM_PROMPT = """You are a senior software engineer doing a pull request code review.

You will be given a set of file diffs (unified diff format, showing only
the CHANGED lines with their new line numbers). Review only the changed
code. Flag real problems: bugs, security issues, missing error handling,
missing tests for new logic, and clear style/readability issues. Do not
invent issues in unchanged code you cannot see. Do not comment on trivial
style preferences with no real impact.

Respond with ONLY valid JSON, no markdown fences, no explanation outside
the JSON, in exactly this shape:

{
  "summary": "2-3 sentence overall assessment of this PR",
  "issues": [
    {
      "path": "relative/file/path.py",
      "line": 42,
      "severity": "bug" | "security" | "style" | "missing_test",
      "body": "specific, actionable comment about this exact line"
    }
  ]
}

The "line" number MUST be a line number that appears in the diff you were
given (a line that was added/changed), not a made-up number. If you find
no real issues, return an empty "issues" list -- do not invent filler
comments just to have something to say.
"""


def review_diff(files: list[dict]) -> dict:
    """
    files: list of {"path": str, "patch": str} -- one entry per changed
    file, where "patch" is the unified diff text for that file (as
    returned by the GitHub API).

    Returns: {"summary": str, "issues": [{"path", "line", "severity", "body"}]}
    """
    diff_text = "\n\n".join(
        f"--- FILE: {f['path']} ---\n{f['patch']}" for f in files if f.get("patch")
    )

    # Keep the prompt within a sane size -- very large PRs get truncated
    # rather than failing outright. A production version would chunk and
    # review in batches instead.
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

    resp = requests.post(NVIDIA_API_URL, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    raw_text = resp.json()["choices"][0]["message"]["content"]

    return _parse_json_response(raw_text)


def _parse_json_response(raw_text: str) -> dict:
    """
    Models sometimes wrap JSON in markdown fences despite instructions.
    Strip those defensively before parsing, and fail soft (empty review)
    rather than crashing the whole CI run if parsing still fails.
    """
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
