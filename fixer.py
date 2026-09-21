"""
Optional auto-fix step, run after review.py, only if enable-autofix is
turned on AND the review found real bugs/security issues.

Uses LangChain's Deep Agents SDK, sandboxed to the checked-out PR
branch, to attempt fixes for the SPECIFIC issues the reviewer flagged.
If it makes changes, it commits and pushes them as a new commit
directly onto the PR's own branch, then comments explaining what
happened.

LIMITATION: only works for PRs from a branch within the SAME repo.
GITHUB_TOKEN can't write to a fork's branches -- on a fork PR the push
fails safely and posts an explanatory comment instead of crashing.
"""

import os
import subprocess
import sys
import requests

from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

GITHUB_API = "https://api.github.com"
REPO_DIR = os.getcwd()


def build_agent():
    model = ChatOpenAI(
        model="nvidia/nemotron-3-super-120b-a12b",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"],
        max_retries=5,
        timeout=180,
    )
    backend = FilesystemBackend(root_dir=REPO_DIR, virtual_mode=True)
    return create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=(
            "You are a careful senior engineer fixing SPECIFIC issues "
            "that were flagged in a code review. Make the minimal edits "
            "needed to fix exactly the listed issues -- do not refactor "
            "or touch unrelated code. Summarize exactly what you changed."
        ),
    )


def has_changes() -> bool:
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return bool(result.stdout.strip())


def commit_and_push(branch: str, token: str, repo: str):
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(
        ["git", "-c", "user.email=ai-fixer@bot.local", "-c", "user.name=AI Auto-Fixer",
         "commit", "-m", "AI auto-fix: address code review comments"],
        check=True,
    )
    push_url = f"https://x-access-token:{token}@github.com/{repo}.git"
    result = subprocess.run(
        ["git", "push", push_url, f"HEAD:{branch}"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        safe_err = result.stderr.replace(token, "***")
        raise RuntimeError(
            f"Push failed -- this usually means the PR is from a fork, "
            f"which this token can't write to. Details: {safe_err}"
        )


def post_comment(repo: str, pr_number: str, token: str, body: str):
    url = f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    requests.post(url, headers=headers, json={"body": body}, timeout=30)


def main():
    issues_text = os.environ.get("REVIEW_ISSUES", "").strip()
    repo = os.environ["GITHUB_REPOSITORY"]
    pr_number = os.environ["PR_NUMBER"]
    token = os.environ["GITHUB_TOKEN"]
    branch = os.environ["PR_HEAD_REF"]

    if not issues_text:
        print("No flagged bugs/security issues -- skipping auto-fix.")
        return

    print(f"Attempting auto-fix for:\n{issues_text}")
    task = (
        "Fix these specific issues found in code review. Make only the "
        f"minimal changes needed:\n\n{issues_text}"
    )

    agent = build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": task}]})
    summary = result["messages"][-1].content
    print(f"Agent summary:\n{summary}")

    if not has_changes():
        print("Agent made no file changes.")
        post_comment(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix attempted\n\nThe agent looked at the flagged "
            f"issues but made no changes.\n\n{summary}",
        )
        return

    try:
        commit_and_push(branch, token, repo)
        post_comment(
            repo, pr_number, token,
            f"### \U0001F916 Auto-fix pushed\n\n{summary}\n\n"
            "This fix was pushed as a new commit on this PR branch. Please review it before merging.",
        )
        print("Auto-fix pushed successfully.")
    except Exception as e:
        post_comment(
            repo, pr_number, token,
            f"### \U0001F916 Auto-fix attempted but could not be pushed\n\n{summary}\n\n"
            f"**Reason:** {e}",
        )
        print(f"Could not push fix: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Auto-fix step failed: {e}", file=sys.stderr)
        sys.exit(0)