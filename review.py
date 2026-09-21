"""
Entry point run by the GitHub Action on every pull_request event.
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


def get_changed_files(repo: str, pr_number: str, token: str) -> list:
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
    valid_comments = []

    for issue in review.get("issues", []):
        valid_comments.append({
            "path": issue["path"],
            "line": issue["line"],
            "side": "RIGHT",
            "body": f"**[{issue.get('severity', 'note')}]** {issue['body']}",
        })

    body = f"### \U0001F916 Automated review\n\n{review['summary']}"
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

    # Expose results to the optional auto-fix step via GitHub Actions
    # outputs. Only "bug"/"security" issues are passed along.
    fixable = [i for i in review.get("issues", []) if i.get("severity") in ("bug", "security")]
    gh_output_path = os.environ.get("GITHUB_OUTPUT")
    if gh_output_path:
        with open(gh_output_path, "a") as f:
            f.write(f"has_bugs={'true' if fixable else 'false'}\n")
            f.write("issues_summary<<EOF\n")
            for issue in fixable:
                f.write(f"- {issue['path']}:{issue['line']} [{issue['severity']}] {issue['body']}\n")
            f.write("EOF\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Review failed: {e}", file=sys.stderr)
        sys.exit(1)