# AI Code Reviewer

A GitHub Action that reviews pull request diffs with NVIDIA Nemotron and
publishes a GitHub pull request review with a summary and, when possible,
inline comments. Review results are checked before submission; they are
not a substitute for human review.

## How it works

For a pull request workflow, the Action:

1. Fetches the pull request head and changed-file patches from the GitHub
   REST API. The changed-files request is paginated.
2. Splits diffs into review chunks, targeting 20,000 characters and no
   more than two file pieces per chunk. Oversized hunks are split where
   possible. Each chunk is sent to the NVIDIA API for a structured JSON
   review.
3. Parses and validates each response. A finding must use a file and an
   added line from that chunk, and must satisfy the required schema,
   category, severity, and confidence constraints. Malformed responses
   or invalid findings fail the review run; they are not silently
   converted into summary-only reviews. Duplicate findings across chunks
   are removed.
4. Fetches the pull request head again. If the head SHA changed while the
   review was being generated, it skips submission. Otherwise, it submits
   the review against the original head SHA.

The SHA check reduces the chance of posting stale feedback, but cannot
eliminate a push that happens after the final check and before submission.

## API retries and review submission

- GitHub GET requests retry timeouts, connection errors, and HTTP 502,
  503, or 504 responses, up to four total attempts, waiting 5, 10, then
  15 seconds between attempts. Other HTTP errors propagate.
- NVIDIA API requests use the same retryable error categories, also up
  to four total attempts with 5, 10, then 15 second delays. A timeout or
  connection failure can be ambiguous: the provider may have received
  the request, so retries may result in repeated inference work or
  charges.
- GitHub review POST requests are not automatically retried after
  timeouts, connection errors, or server errors. A lost response can
  leave the submission outcome uncertain; manually rerunning the
  workflow can post another review.
- If a review with inline comments receives HTTP 400 or 422 and GitHub's
  structured error response identifies an invalid
  `PullRequestReviewComment` field (such as `line`, `path`, `position`,
  `side`, or `subject_type`), the Action makes one summary-only fallback
  submission with the findings listed in its body. Unrelated or
  unrecognized errors propagate instead of triggering that fallback.
  Failure of the fallback also propagates.

## Use the Action

Create a workflow such as `.github/workflows/ai-review.yml` in the
repository where pull requests will be reviewed:

```yaml
name: AI Review

on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  pull-requests: write
  contents: read

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - name: Run AI review
        uses: Oussamagrimida/AI-Code-reviewer-Github-Action@main
        with:
          nvidia-api-key: ${{ secrets.NVIDIA_API_KEY }}
          github-token: ${{ github.token }}
          enable-autofix: 'false'
```

For production workflows, pin `uses:` to a reviewed release or commit
instead of a moving branch.

### Required credentials and permissions

- Add `NVIDIA_API_KEY` as an Actions repository secret. The Action
  requires this input to call the NVIDIA API.
- `github-token` defaults to `${{ github.token }}`; the example passes
  it explicitly. No personal access token is required for review
  submission when the generated token has the permissions below.
- Grant `pull-requests: write` so the Action can fetch pull request
  details and create reviews. The repository's self-test workflow also
  grants `contents: read` to check out the trusted Action source.
- Secrets and write permissions are restricted for many fork-originated
  `pull_request` workflows. Test with a same-repository PR unless the
  workflow is deliberately designed to handle fork contributions safely.

Auto-fix is optional and disabled by default. Do not enable it for this
review-only setup. The self-test workflow intentionally keeps it disabled.

### Optional auto-fix

Auto-fix is supported only for same-repository pull requests; fork PRs are
rejected before the fixing agent or test command runs. The checkout must be
at the PR's exact head SHA, not GitHub's synthetic merge commit. For example,
configure a trusted workflow as follows:

```yaml
name: AI Review and Auto-Fix

on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  contents: write
  pull-requests: write

jobs:
  review:
    if: github.event.pull_request.head.repo.full_name == github.repository
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}
          persist-credentials: false

      # Install the project and test dependencies here, using Python 3.12.

      - name: Run AI review and auto-fix
        uses: Oussamagrimida/AI-Code-reviewer-Github-Action@main
        with:
          nvidia-api-key: ${{ secrets.NVIDIA_API_KEY }}
          github-token: ${{ github.token }}
          enable-autofix: 'true'
          test-command: '["python", "-m", "pytest", "-q"]'
          max-fix-attempts: '2'
```

Only enable this in a default-branch workflow trusted by repository
maintainers. Do not use `pull_request_target` to run the auto-fixer or tests
on PR-controlled code. Restrict eligible PR authors to trusted collaborators:
tests execute code from the PR, so repository workflow configuration and the
test command must be trusted.

`test-command` is a trusted JSON array of executable and arguments; it is
invoked without a shell. It must not be derived from PR content. If it is
missing or invalid, auto-fix stops without generating or pushing a fix.
The test command has a 10-minute timeout; test output is bounded and only
passed to the fixing agent for retries. A nonzero exit or timeout rolls back
the current attempt and permits another attempt if any remain. Infrastructure
failures, unauthorized file changes, and failed rollback stop immediately.
`max-fix-attempts` accepts 1 through 5 and defaults to 2.

The agent receives only copies of the review-authorized files in a temporary
workspace. The generated edits are checked against that allowlist before
being copied into the checkout. Tests run with a reduced environment that
omits `GITHUB_TOKEN` and `NVIDIA_API_KEY`; the checkout's persisted GitHub
authentication header is removed before tests, and push credentials are
provided only to the Git push subprocess without placing the token in its
arguments or remote URL. On Linux, the fixer process is made non-dumpable
before tests start so an unprivileged test child cannot read its initial
environment through `/proc`. Auto-fix fails closed if this protection is
unavailable or the runner is not Linux. Applying edits uses no-follow
directory handles and atomic replacement, and fails closed where those
filesystem primitives are unavailable. These restrictions are not a general
OS sandbox and do not protect against privileged processes.
Failed tests are rolled back and verified. Only validated modifications to
existing authorized files are staged, committed, and pushed. The push is
non-forced and is guarded by PR head-SHA checks.

Auto-fix requires `contents: write` to push and `pull-requests: write` to
validate the PR and report the outcome. Keep `enable-autofix` false unless
those permissions, the same-repository restriction, checkout SHA, trusted
test command, and eligible authors are all configured deliberately.

## Tests

Run the unit suite locally from the repository root:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
```

The GitHub Actions unit-test workflow uses Python 3.12 and runs the same
pytest command. The tests mock external API calls; passing them does not
verify live credentials, GitHub permissions, NVIDIA availability, or
real PR comment placement.

For focused tests, run:

```bash
python -m pytest -q tests/test_review.py
python -m pytest -q tests/test_llm_review.py
```

## Known limitations

- Reviews cover only diff content returned by GitHub. Files without a
  patch are skipped, and the model cannot inspect unrelated code unless
  it is present in the supplied diff.
- The model can miss real issues or produce invalid findings. Invalid
  findings are rejected by validation, which fails the run rather than
  guaranteeing a fallback review.
- There is no cross-run review deduplication. Separate workflow runs may
  submit similar or duplicate reviews.
- The optional auto-fix path is separate from review submission and is
  not enabled by default. It should be tested and permissioned separately.
