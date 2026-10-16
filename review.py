"""
Entry point run by the GitHub Action on every pull_request event.
"""

import os
import sys
import uuid
import requests
import time

import llm_review

GITHUB_API = "https://api.github.com"


def gh_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def github_get_with_retry(
    url: str,
    headers: dict,
    params: dict | None = None,
    timeout: int = 30,
    max_attempts: int = 4,
    retry_delay: int = 5,
):
    for attempt in range(max_attempts):
        try:
            resp = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=timeout,
            )

            if resp.status_code in (502, 503, 504):
                if attempt < max_attempts - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue

            resp.raise_for_status()
            return resp

        except (requests.Timeout, requests.ConnectionError):
            if attempt == max_attempts - 1:
                raise

            time.sleep(retry_delay * (attempt + 1))

    raise RuntimeError("GitHub GET request failed unexpectedly")



def get_pr_info(repo: str, pr_number: str, token: str) -> dict:
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"

    resp = github_get_with_retry(
        url,
        headers=gh_headers(token),
        timeout=30,
    )
    resp.raise_for_status()

    return resp.json()


def get_changed_files(
    repo: str,
    pr_number: str,
    token: str,
) -> list:
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/files"

    files = []
    page = 1

    while True:
        resp = github_get_with_retry(
            url,
            headers=gh_headers(token),
            params={
                "per_page": 100,
                "page": page,
            },
            timeout=30,
        )

        resp.raise_for_status()

        batch = resp.json()

        if not batch:
            break

        files.extend(batch)
        page += 1

    return [
        {
            "path": file["filename"],
            "patch": file.get("patch", ""),
        }
        for file in files
        if file.get("patch")
    ]


def _is_inline_comment_validation_error(response) -> bool:
    json_method = getattr(response, "json", None)
    if not callable(json_method):
        return False

    try:
        error_data = json_method()
    except ValueError:
        return False

    if not isinstance(error_data, dict):
        return False

    errors = error_data.get("errors")
    if not isinstance(errors, list):
        return False

    inline_comment_fields = {
        "line",
        "path",
        "position",
        "side",
        "subject_type",
    }

    return any(
        isinstance(error, dict)
        and error.get("resource") == "PullRequestReviewComment"
        and isinstance(error.get("field"), str)
        and error["field"] in inline_comment_fields
        for error in errors
    )


def post_review(
    repo: str,
    pr_number: str,
    token: str,
    commit_id: str,
    review: dict,
):
    valid_comments = []

    for issue in review.get("issues", []):
        valid_comments.append(
            {
                "path": issue["path"],
                "line": issue["line"],
                "side": "RIGHT",
                "body": (
                    f"**[{issue.get('severity', 'info').upper()} · "
                    f"{issue.get('category', 'unknown')} · "
                    f"{issue.get('confidence', 0):.0%} confidence]** "
                    f"{issue['body']}"
                ),
            }
        )

    body = (
        "### 🤖 Automated review\n\n"
        f"{review.get('summary', 'Automated review completed.')}"
    )

    url = (
        f"{GITHUB_API}/repos/{repo}/pulls/"
        f"{pr_number}/reviews"
    )

    payload = {
        "commit_id": commit_id,
        "body": body,
        "event": "COMMENT",
        "comments": valid_comments,
    }
    inline_comments_posted = len(valid_comments)

    resp = requests.post(
        url,
        headers=gh_headers(token),
        json=payload,
        timeout=30,
    )

    if (
        resp.status_code in (400, 422)
        and valid_comments
        and _is_inline_comment_validation_error(resp)
    ):
        print(
            f"Inline comments rejected ({resp.status_code}), "
            "falling back to summary-only."
        )

        fallback_notes = [
            (
                f"- **{issue['path']}:{issue['line']}** "
                f"[{issue.get('severity', 'info').upper()} · "
                f"{issue.get('category', 'unknown')} · "
                f"{issue.get('confidence', 0):.0%}] "
                f"{issue['body']}"
            )
            for issue in review.get("issues", [])
        ]

        body += (
            "\n\n**Flagged issues:**\n"
            + "\n".join(fallback_notes)
        )

        payload = {
            "commit_id": commit_id,
            "body": body,
            "event": "COMMENT",
            "comments": [],
        }

        resp = requests.post(
            url,
            headers=gh_headers(token),
            json=payload,
            timeout=30,
        )
        inline_comments_posted = 0

    resp.raise_for_status()

    posted_inline_comments = len(valid_comments)

    if payload["comments"] == []:
        posted_inline_comments = 0

    print(
        f"Review posted: "
        f"{posted_inline_comments} inline comment(s)."
    )


def write_github_outputs(review: dict):
    """
    Expose validated review results to the optional
    auto-fix step through GitHub Actions outputs.
    """

    fixable_categories = {
        "correctness",
        "security",
        "performance",
        "concurrency",
        "resource_management",
        "error_handling",
        "edge_case",
    }

    fixable_severities = {
        "critical",
        "high",
        "medium",
    }

    fixable = [
        issue
        for issue in review.get("issues", [])
        if (
            issue.get("category") in fixable_categories
            and issue.get("severity") in fixable_severities
        )
    ]

    gh_output_path = os.environ.get("GITHUB_OUTPUT")

    if not gh_output_path:
        return

    summary = "\n".join(
        (
            f"- {issue['path']}:{issue['line']} "
            f"[{issue.get('category', 'unknown')}] "
            f"[{issue.get('severity', 'info')}] "
            f"[{issue.get('confidence', 0):.0%}] "
            f"{issue['body']}"
        )
        for issue in fixable
    )

    delimiter = f"EOF_{uuid.uuid4().hex}"

    while delimiter in summary:
        delimiter = f"EOF_{uuid.uuid4().hex}"

    with open(
        gh_output_path,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"has_bugs={'true' if fixable else 'false'}\n"
        )

        f.write(f"issues_summary<<{delimiter}\n")
        f.write(summary)
        f.write(f"\n{delimiter}\n")



def main():
    repo = os.environ["GITHUB_REPOSITORY"]
    pr_number = os.environ["PR_NUMBER"]
    token = os.environ["GITHUB_TOKEN"]

    print(
        f"Reviewing {repo} PR #{pr_number}..."
    )

    pr_info = get_pr_info(
        repo,
        pr_number,
        token,
    )

    commit_id = pr_info["head"]["sha"]

    files = get_changed_files(
        repo,
        pr_number,
        token,
    )

    if not files:
        print(
            "No reviewable file diffs found -- skipping."
        )
        write_github_outputs({"issues": []})
        return

    print(
        f"Reviewing {len(files)} changed file(s)..."
    )

    review = llm_review.review_diff(files)

    latest_pr_info = get_pr_info(
        repo,
        pr_number,
        token,
    )

    latest_commit_id = latest_pr_info["head"]["sha"]

    if latest_commit_id != commit_id:
        print(
            "PR head changed during review. "
            "Skipping review."
        )
        write_github_outputs({"issues": []})
        return

    print(
        f"Found {len(review.get('issues', []))} "
        "issue(s). Posting review..."
    )

    post_review(
        repo,
        pr_number,
        token,
        commit_id,
        review,
    )

    write_github_outputs(review)



def run():
    try:
        main()

    except llm_review.ReviewValidationError as e:
        print(
            f"Review validation failed: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    except Exception as e:
        print(
            f"Review failed: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    run()