"""
Entry point run by the GitHub Action on every pull_request event.

Reads context from environment variables that GitHub Actions provides
automatically, fetches the PR's changed files, sends them to Nemotron for
review, and posts the result as a real PR review with inline comments.

Required environment variables (set by action.yml):
    GITHUB_TOKEN       - provided automatically by GitHub Actions
    GITHUB_REPOSITORY  - "owner/repo", provided automatically
    PR_NUMBER          - the pull request number being reviewed
    NVIDIA_API_KEY      - your NVIDIA API key, passed in as a repo secret
"""

import os
import sys
import requests

import llm_review

GITHUB_API = "https://api.github.com"


def gh_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def get_pr_info(repo: str, pr_number: str, token: str) -> dict:
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}"
    resp = requests.get(url, headers=gh_headers(token), timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_changed_files(repo: str, pr_number: str, token: str) -> list[dict]:
    """
    Returns changed files with their diff ("patch") text. Skips files
    GitHub doesn't provide a patch for (binary files, or diffs too large
    for the API to include) since there's nothing meaningful to review.
    """
    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/files"
    files = []
    page = 1
    while True:
        resp = requests.get(
            url, headers=gh_headers(token),
            params={"per_page": 100, "page": page}, timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        files.extend(batch)
        page += 1

    return [
        {"path": f["filename"], "patch": f.get("patch", "")}
        for f in files if f.get("patch")
    ]


def post_review(repo: str, pr_number: str, token: str, commit_id: str, review: dict):
    """
    Posts one PR review containing a summary + inline comments.

    If GitHub rejects a specific inline comment (most commonly because
    the LLM picked a line number outside the diff), we don't fail the
    whole run -- we drop that one comment and fold it into the summary
    text instead, so the reviewer's other valid feedback still gets
    posted.
    """
    valid_comments = []
    fallback_notes = []

    for issue in review.get("issues", []):
        valid_comments.append({
            "path": issue["path"],
            "line": issue["line"],
            "side": "RIGHT",
            "body": f"**[{issue.get('severity', 'note')}]** {issue['body']}",
        })

    body = f"### 🤖 Automated review\n\n{review['summary']}"
    if review.get("_raw"):
        body += f"\n\n<details><summary>raw model output</summary>\n\n```\n{review['_raw'][:2000]}\n```\n</details>"

    url = f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}/reviews"
    payload = {
        "commit_id": commit_id,
        "body": body,
        "event": "COMMENT",
        "comments": valid_comments,
    }

    resp = requests.post(url, headers=gh_headers(token), json=payload, timeout=30)

    if resp.status_code >= 300 and valid_comments:
        # Likely a bad line number in one of the comments. Retry with a
        # summary-only review, and list the flagged issues in the body
        # instead of failing the whole review.
        print(f"Inline comments rejected ({resp.status_code}), falling back to summary-only.")
        fallback_notes = [
            f"- **{i['path']}:{i['line']}** [{i.get('severity','note')}] {i['body']}"
            for i in review.get("issues", [])
        ]
        body += "\n\n**Flagged issues:**\n" + "\n".join(fallback_notes)
        payload = {"commit_id": commit_id, "body": body, "event": "COMMENT", "comments": []}
        resp = requests.post(url, headers=gh_headers(token), json=payload, timeout=30)

    resp.raise_for_status()
    print(f"Review posted: {len(valid_comments)} inline comment(s).")


def main():
    repo = os.environ["GITHUB_REPOSITORY"]
    pr_number = os.environ["PR_NUMBER"]
    token = os.environ["GITHUB_TOKEN"]

    print(f"Reviewing {repo} PR #{pr_number}...")

    pr_info = get_pr_info(repo, pr_number, token)
    commit_id = pr_info["head"]["sha"]

    files = get_changed_files(repo, pr_number, token)
    if not files:
        print("No reviewable file diffs found -- skipping.")
        return

    print(f"Reviewing {len(files)} changed file(s)...")
    review = llm_review.review_diff(files)

    print(f"Found {len(review.get('issues', []))} issue(s). Posting review...")
    post_review(repo, pr_number, token, commit_id, review)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Review failed: {e}", file=sys.stderr)
        sys.exit(1)
