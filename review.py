"""
Entry point run by the GitHub Action on every pull_request event.
"""

import os
import re
import sys
import time
import uuid
from collections.abc import Mapping

import requests

import llm_review

GITHUB_API = "https://api.github.com"
RETRYABLE_GITHUB_STATUSES = {429, 500, 502, 503, 504}


class GitHubAPIError(RuntimeError):
    """Raised when a GitHub API operation fails."""


class GitHubMetadataError(ValueError):
    """Raised when GitHub returns unusable pull request data."""


def _github_retry_delay(response, attempt, default_delay):
    headers = getattr(response, "headers", {})
    retry_after = (
        headers.get("Retry-After")
        if isinstance(headers, Mapping)
        else None
    )
    try:
        return min(60, max(0, int(retry_after)))
    except (TypeError, ValueError):
        return default_delay * (attempt + 1)


def _is_github_rate_limited(response) -> bool:
    headers = getattr(response, "headers", {})
    return (
        response.status_code == 429
        or (
            response.status_code == 403
            and isinstance(headers, Mapping)
            and headers.get("X-RateLimit-Remaining") == "0"
        )
    )


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
        except requests.Timeout:
            if attempt == max_attempts - 1:
                raise GitHubAPIError(
                    "GitHub API request timed out after retries."
                ) from None
            time.sleep(retry_delay * (attempt + 1))
            continue
        except requests.ConnectionError:
            if attempt == max_attempts - 1:
                raise GitHubAPIError(
                    "GitHub API connection failed after retries."
                ) from None
            time.sleep(retry_delay * (attempt + 1))
            continue
        except requests.RequestException:
            if attempt == max_attempts - 1:
                raise GitHubAPIError(
                    "GitHub API transport failed after retries."
                ) from None
            time.sleep(retry_delay * (attempt + 1))
            continue

        rate_limited = _is_github_rate_limited(resp)
        if (
            resp.status_code in RETRYABLE_GITHUB_STATUSES
            or rate_limited
        ):
            if attempt == max_attempts - 1:
                reason = (
                    "rate limit persisted"
                    if rate_limited
                    else "request failed"
                )
                raise GitHubAPIError(
                    f"GitHub API {reason} after retries "
                    f"(HTTP {resp.status_code})."
                )
            time.sleep(
                _github_retry_delay(resp, attempt, retry_delay)
            )
            continue

        if resp.status_code >= 400:
            raise GitHubAPIError(
                f"GitHub API request failed (HTTP {resp.status_code})."
            )

        return resp

    raise RuntimeError("GitHub GET request failed unexpectedly")


def _validate_pr_identifiers(repo: str, pr_number: str):
    if (
        not isinstance(repo, str)
        or re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
            repo,
        ) is None
    ):
        raise GitHubMetadataError(
            "GITHUB_REPOSITORY must be in owner/repository format."
        )
    if (
        not isinstance(pr_number, str)
        or re.fullmatch(r"[1-9]\d*", pr_number) is None
    ):
        raise GitHubMetadataError("PR_NUMBER must be a positive integer.")


def get_pr_info(repo: str, pr_number: str, token: str) -> dict:
    _validate_pr_identifiers(repo, pr_number)
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"

    resp = github_get_with_retry(
        url,
        headers=gh_headers(token),
        timeout=30,
    )
    try:
        data = resp.json()
    except ValueError:
        raise GitHubMetadataError(
            "GitHub returned invalid JSON for pull request metadata."
        ) from None
    if not isinstance(data, dict):
        raise GitHubMetadataError(
            "GitHub returned invalid pull request metadata."
        )
    head = data.get("head")
    sha = head.get("sha") if isinstance(head, dict) else None
    if (
        not isinstance(sha, str)
        or re.fullmatch(r"[0-9a-fA-F]{40}", sha) is None
    ):
        raise GitHubMetadataError(
            "GitHub pull request metadata is missing a valid head SHA."
        )
    return data


def get_changed_files(
    repo: str,
    pr_number: str,
    token: str,
) -> list:
    _validate_pr_identifiers(repo, pr_number)
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

        try:
            batch = resp.json()
        except ValueError:
            raise GitHubMetadataError(
                "GitHub returned invalid JSON for the changed-file list."
            ) from None

        if not batch:
            break
        if not isinstance(batch, list):
            raise GitHubMetadataError(
                "GitHub returned an invalid changed-file list."
            )

        for file in batch:
            if not isinstance(file, dict):
                raise GitHubMetadataError(
                    "GitHub returned invalid changed-file metadata."
                )
            path = file.get("filename")
            patch = file.get("patch", "")
            if (
                not isinstance(path, str)
                or not path
                or (patch is not None and not isinstance(patch, str))
            ):
                raise GitHubMetadataError(
                    "GitHub returned invalid changed-file metadata."
                )
            files.append({"path": path, "patch": patch or ""})
        page += 1

    return files


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


def _post_github_review(url: str, headers: dict, payload: dict):
    for attempt in range(3):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=30,
            )
        except requests.Timeout:
            raise GitHubAPIError(
                "GitHub review publication timed out; the submission outcome "
                "may be uncertain. Check the pull request before rerunning."
            ) from None
        except requests.ConnectionError:
            raise GitHubAPIError(
                "GitHub review publication lost its connection; the submission "
                "outcome may be uncertain. Check the pull request before rerunning."
            ) from None
        except requests.RequestException:
            raise GitHubAPIError(
                "GitHub review publication encountered a transport failure."
            ) from None

        rate_limited = _is_github_rate_limited(response)
        if not rate_limited or attempt == 2:
            return response
        time.sleep(_github_retry_delay(response, attempt, 5))

    raise GitHubAPIError("GitHub review publication failed unexpectedly.")


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
    resp = _post_github_review(
        url,
        gh_headers(token),
        payload,
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

        resp = _post_github_review(
            url,
            gh_headers(token),
            payload,
        )
    if resp.status_code >= 400:
        if _is_github_rate_limited(resp):
            raise GitHubAPIError(
                "GitHub review publication was rate limited "
                f"(HTTP {resp.status_code})."
            )
        raise GitHubAPIError(
            f"GitHub review publication failed (HTTP {resp.status_code})."
        )

    posted_inline_comments = len(payload["comments"])

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
            f"{' '.join(issue['body'].splitlines())}"
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
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    pr_number = os.environ.get("PR_NUMBER", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    _validate_pr_identifiers(repo, pr_number)
    if not token:
        raise GitHubMetadataError("GITHUB_TOKEN is not configured.")

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
        print("No changed files were returned by GitHub; skipping review.")
        write_github_outputs({"issues": []})
        return

    reviewable_files = [
        file for file in files
        if isinstance(file.get("patch"), str) and file["patch"].strip()
    ]
    if not reviewable_files:
        print(
            "Changed files were returned, but none contained a reviewable "
            "diff; skipping model review."
        )
        write_github_outputs({"issues": []})
        return

    print(
        f"Reviewing {len(reviewable_files)} changed file(s)..."
    )

    review_result = llm_review.review_diff(reviewable_files)

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

    if not review_result["issues"]:
        print(
            "No valid findings remain after validation; "
            "skipping review publication and auto-fix."
        )
        write_github_outputs(review_result)
        return

    print(
        f"Found {len(review_result.get('issues', []))} "
        "issue(s). Posting review..."
    )

    post_review(
        repo,
        pr_number,
        token,
        commit_id,
        review_result,
    )

    write_github_outputs(review_result)



def run():
    try:
        main()

    except llm_review.ReviewValidationError as e:
        print(
            f"Invalid NVIDIA model output: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    except llm_review.NVIDIAAPIError as e:
        print(
            f"NVIDIA API failure: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    except GitHubMetadataError as e:
        print(
            f"Invalid pull request metadata: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    except GitHubAPIError as e:
        print(
            f"GitHub API failure: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    except Exception as e:
        print(
            f"Review failed due to an internal error "
            f"({type(e).__name__}).",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    run()