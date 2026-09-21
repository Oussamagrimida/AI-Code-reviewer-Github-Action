# AI Code Reviewer

A GitHub Action that automatically reviews every pull request — reads
the diff, flags bugs and security issues, and posts real inline
comments — using NVIDIA Nemotron 3. Optionally, it can also attempt to
**fix** the flagged issues and push the fix directly onto the PR.

Drop it into any repo's workflow and it runs automatically the moment a
PR is opened. No server to host, no manual command to run.

## Why this matters

Code review is a real bottleneck: PRs sit waiting for a free reviewer,
senior engineers spend hours a week catching routine issues (missing
null checks, obvious bugs, security holes), and review quality varies
with how rushed the reviewer is. This doesn't replace human review — it
catches the obvious stuff automatically so humans can focus on judgment
calls: architecture, intent, tradeoffs.

**Real companies doing versions of this today:** CodeRabbit, Qodo
(formerly CodiumAI), Amazon CodeGuru, GitHub's own Copilot code review,
and Greptile all automatically review PRs and post inline feedback the
same way this project does.

## How it works

```
PR opened/updated on a repo using this Action
        │
        ▼
GitHub Actions runner starts (free, GitHub's own infrastructure)
        │
        ├─ 1. Fetch the PR's changed files + diffs (GitHub REST API)
        ├─ 2. Send diffs to Nemotron 3, asking for structured JSON:
        │      { summary, issues: [{path, line, severity, body}] }
        ├─ 3. Parse the response
        ├─ 4. Post a real PR review: inline comments at the flagged
        │      lines, plus a summary — visible to every human reviewer
        │
        │      ── optional, off by default ──
        │
        ├─ 5. If real bugs/security issues were found AND auto-fix is
        │      enabled: a Deep Agent (LangChain SDK) attempts to fix
        │      exactly those issues, sandboxed to the checked-out repo
        └─ 6. If it made changes: commit + push a new commit onto the
               PR branch, and comment explaining what it did
```

If the model picks a line number that isn't actually part of the diff,
or the auto-fix step can't push (e.g. the PR is from a fork), the run
degrades gracefully — it falls back to a summary comment instead of
failing the whole workflow.

## Use it on your own repo

1. Get a free NVIDIA API key at [build.nvidia.com](https://build.nvidia.com)
2. Add it as a repo secret: **Settings → Secrets and variables → Actions
   → New repository secret** → name it `NVIDIA_API_KEY`
3. Add this workflow file to your repo at `.github/workflows/ai-review.yml`:

```yaml
name: AI Review

on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  pull-requests: write
  contents: write   # only needed if you enable auto-fix

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.ref }}
          repository: ${{ github.event.pull_request.head.repo.full_name }}
          token: ${{ secrets.GITHUB_TOKEN }}
      - uses: YOUR_USERNAME/ai-code-reviewer@main
        with:
          nvidia-api-key: ${{ secrets.NVIDIA_API_KEY }}
          enable-autofix: 'true'   # optional, defaults to 'false'
```

4. Open a PR — the review appears automatically within a minute or two.

## Auto-fix (optional)

Off by default. When `enable-autofix: 'true'`, after posting the review,
the Action attempts to actually fix any flagged **bug** or **security**
issues (not style nitpicks) and pushes them as a new commit on the same
PR branch, with a comment explaining what changed. A human still reviews
and decides whether to keep it — this never auto-merges anything.

**Limitation:** only works for PRs from a branch within the same repo.
GitHub's default token can't push to a fork's branch, so on fork PRs the
fix attempt fails safely with an explanatory comment instead of crashing.

## Testing it

See [`TESTING.md`](./TESTING.md) for three graded test cases (easy,
medium, hard — including a SQL injection scenario) to check what the
reviewer actually catches, not just whether it runs.

## Project structure

```
action.yml                Defines the reusable GitHub Action
review.py                 Fetches diff → calls LLM → posts review comments
llm_review.py              Prompt + NVIDIA API call + structured JSON parsing
fixer.py                   Optional: fixes flagged issues, pushes a commit
requirements.txt           Deps for the review step
requirements-autofix.txt   Extra deps (deepagents SDK) for the fix step
TESTING.md                 Graded test cases to validate real performance
.github/workflows/
  self-test.yml             Runs this action on this repo's own PRs (live demo)
```

## Known limitations (honest, not hidden)

- Large PRs get truncated to stay within prompt limits — a production
  version would chunk the diff and review in batches instead.
- Only reviews lines that changed, since that's what the diff API
  provides — it can't reason about unrelated code it wasn't shown.
- No automated test execution before an auto-fix is pushed — the
  agent's own judgment is the only check right now. Running the repo's
  existing test suite before pushing is the natural next improvement.
- Auto-fix processes all flagged issues in one pass rather than one at a
  time, which risks one fix overwriting another on the same file.
- No memory across PRs — each review starts fresh.

## Background

This project grew out of a learning path: first a hand-rolled Python
prototype (plan/edit/test loop with sandboxed pytest execution), then a
migration to LangChain's real Deep Agents SDK integrated with NVIDIA
Nemotron, then this — applying the same skills to automated PR review,
with an optional agent-driven fix step layered on top.