"""
Optional auto-fix step, run after review.py, only if enable-autofix is
turned on AND the review found real bugs/security issues.

Uses LangChain's Deep Agents SDK, sandboxed to the checked-out PR
branch, to attempt fixes for the SPECIFIC issues the reviewer flagged.
If it makes changes, it commits and pushes them as a new commit
directly onto the PR's own branch, then comments explaining what
happened.

SAFETY: two layers stop the agent from editing the wrong thing --
(1) the prompt explicitly names the exact files it's allowed to touch
and forbids anything under .github/, and (2) after it runs, a real
`git diff --name-only` check verifies it only touched those files
before anything is committed or pushed. If it strayed, the fix is
discarded (git checkout --) and a comment explains why, instead of
pushing an unexpected change -- or worse, failing on GitHub's own
workflow-file protection like this project did during testing.

LIMITATION: pushing still only works for PRs from a branch within the
SAME repo -- GITHUB_TOKEN can't write to a fork's branches.
"""

import os
import re
import subprocess
import sys
import requests

from langchain_openai import ChatOpenAI
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

GITHUB_API = "https://api.github.com"
REPO_DIR = os.getcwd()


def extract_allowed_paths(issues_text: str) -> list:
    """
    Pulls the exact file paths out of the issues summary written by
    review.py (lines look like "- path/to/file.py:42 [bug] ...").
    These are the ONLY files the agent should be allowed to touch.
    """
    paths = set()
    for line in issues_text.splitlines():
        match = re.match(r"^-\s*([^\s:]+):\d+", line.strip())
        if match:
            paths.add(match.group(1))
    return sorted(paths)


def build_agent(allowed_paths: list):
    model = ChatOpenAI(
        model="nvidia/nemotron-3-super-120b-a12b",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"],
        max_retries=5,
        timeout=180,
    )
    backend = FilesystemBackend(root_dir=REPO_DIR, virtual_mode=True)
    paths_list = "\n".join(f"  - {p}" for p in allowed_paths)
    return create_deep_agent(
        model=model,
        backend=backend,
        system_prompt=(
            "You are a careful senior engineer fixing SPECIFIC issues "
            "that were flagged in a code review.\n\n"
            "You are ONLY allowed to edit these exact file(s):\n"
            f"{paths_list}\n\n"
            "Do NOT create, rename, move, or delete any other files. Do "
            "NOT touch anything under .github/ (workflow configuration) "
            "under any circumstances, even if it seems related. Make the "
            "minimal edits needed to fix exactly the listed issues in "
            "the allowed file(s) only. Summarize exactly what you changed."
        ),
    )


def get_changed_files() -> list:
    result = subprocess.run(
        ["git", "diff", "--name-only"], capture_output=True, text=True
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def discard_all_changes():
    subprocess.run(["git", "checkout", "--", "."], check=False)
    subprocess.run(["git", "clean", "-fd"], check=False)


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

    allowed_paths = extract_allowed_paths(issues_text)
    if not allowed_paths:
        print("Could not determine which files to fix -- skipping auto-fix.")
        return

    print(f"Attempting auto-fix for:\n{issues_text}")
    print(f"Allowed files: {allowed_paths}")

    task = (
        "Fix these specific issues found in code review. Make only the "
        f"minimal changes needed, in the allowed file(s) only:\n\n{issues_text}"
    )

    agent = build_agent(allowed_paths)
    result = agent.invoke({"messages": [{"role": "user", "content": task}]})
    summary = result["messages"][-1].content
    print(f"Agent summary:\n{summary}")

    changed_files = get_changed_files()
    if not changed_files:
        print("Agent made no file changes.")
        post_comment(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix attempted\n\nThe agent looked at the flagged "
            f"issues but made no changes.\n\n{summary}",
        )
        return

    # SAFETY CHECK: verify it only touched files it was told to.
    unexpected = [f for f in changed_files if f not in allowed_paths]
    if unexpected:
        print(f"Agent touched unexpected files: {unexpected} -- discarding fix.")
        discard_all_changes()
        post_comment(
            repo, pr_number, token,
            "### \U0001F916 Auto-fix attempted but was discarded for safety\n\n"
            f"{summary}\n\n"
            f"**Reason:** the agent modified file(s) outside what was flagged "
            f"({', '.join(unexpected)}), so the fix was discarded rather than "
            "pushed. Please fix this manually or refine the issue description.",
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