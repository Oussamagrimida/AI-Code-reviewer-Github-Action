# AI Code Reviewer

A GitHub Action that automatically reviews every pull request — reads the
diff, flags bugs and security issues, and posts real inline comments —
using NVIDIA Nemotron 3. Drop it into any repo's workflow and it reviews
PRs the moment they're opened, no server to host or run.

This follows the same pattern as production tools in this space:
**CodeRabbit**, **Qodo (CodiumAI)**, **Amazon CodeGuru**, and GitHub's own
**Copilot code review** all automatically review PRs and post inline
feedback the same way this project does.

## Why this matters

Code review is a real bottleneck: PRs sit waiting for a free reviewer,
senior engineers spend hours a week catching routine issues (missing
null checks, obvious bugs, missing tests), and review quality varies with
how rushed the reviewer is. This doesn't replace human review — it
catches the obvious stuff automatically so humans can focus on judgment
calls: architecture, intent, tradeoffs.

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
        └─ 4. Post a real PR review: inline comments at the flagged
              lines, plus a summary — visible to every human reviewer
```

If the model picks a line number that isn't actually part of the diff
(GitHub rejects out-of-diff line comments), the run doesn't fail — it
falls back to listing that issue in the summary comment instead, so one
bad line number never loses the whole review.

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

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: YOUR_USERNAME/ai-code-reviewer@main
        with:
          nvidia-api-key: ${{ secrets.NVIDIA_API_KEY }}
```

4. Open a PR — the review appears automatically within a minute or two.

## Project structure

```
action.yml              Defines the reusable GitHub Action
review.py                Orchestration: fetch diff → call LLM → post review
llm_review.py            Prompt + NVIDIA API call + structured JSON parsing
requirements.txt
.github/workflows/
  self-test.yml           Runs this action on this repo's own PRs (live demo)
```

## Known limitations (honest, not hidden)

- Large PRs get truncated to stay within prompt limits — a production
  version would chunk the diff and review in batches instead.
- Only reviews lines that changed, since that's what the diff API
  provides — it can't reason about unrelated code it wasn't shown.
- No memory across PRs yet — each review starts fresh, unlike a human
  reviewer who remembers past feedback on the same codebase.
- Advisory only, by design — it never approves, requests changes, or
  blocks a merge. It comments; a human still decides.

## Background

This project builds on an earlier learning exercise: a hand-rolled
plan/edit/test coding agent, then a migration to LangChain's Deep Agents
SDK with real GitHub PR creation. This project applies the same
diff-reading and structured-output skills to the review side of the
workflow instead of the fix side.
