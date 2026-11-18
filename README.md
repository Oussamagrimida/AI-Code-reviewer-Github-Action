# AI Code Reviewer

A GitHub Action that reviews pull request diffs with NVIDIA Nemotron and
publishes a GitHub pull request review with a summary and, when possible,
inline comments. Review results are checked before submission; they are
not a substitute for human review.

## How it works

For a pull request workflow, the Action executes an intelligent multi-pass
review pipeline:

1. **Diff ingestion & chunking**: Fetches pull request head and changed-file
   patches from the GitHub REST API (paginated). Diffs are grouped into review
   chunks (up to 20,000 characters and no more than two files per chunk).
2. **Pass 1 — Structural analysis (zero API calls)**: Analyzes file types,
   AST definitions (Python classes, functions, calls), and cross-file dependencies
   locally before making any model calls.
3. **Pass 2 — Correctness & logic**: Specialized prompt targeting algorithmic
   flaws, arithmetic mistakes, inverted conditions, off-by-one errors, empty
   inputs, and boundary conditions.
4. **Pass 3 — Security & reliability**: Specialized prompt targeting injection,
   path traversal, secret exposure, unhandled exceptions, and resource leaks.
5. **Pass 4 — Conditional cross-file analysis**: When structural relationships
   exist across multiple changed files, scans for broken interfaces, changed
   signatures, and unadjusted callers.
6. **Pass 5 — Synthesis & semantic deduplication**: Merges multi-pass findings,
   resolving duplicate reports of the same underlying defect while preserving
   independent issues. Chooses the highest severity and most actionable remediation.
7. **Safe submission**: Verifies that findings anchor exclusively to added lines.
   Checks pull request head SHA to avoid stale reviews and posts summary and inline
   comments.

The SHA check reduces the chance of posting stale feedback, but cannot
eliminate a push that happens after the final check and before submission.



## Enriched finding schema

Findings provide structured, actionable technical feedback:

- `path` & `line`: Exact file and line in the pull request diff.
- `category`: `correctness`, `security`, `concurrency`, `performance`,
  `resource_management`, `error_handling`, `edge_case`, or `missing_test`.
- `severity`: `critical`, `high`, `medium`, `low`, or `info`.
- `confidence`: Calibrated certainty score between 0.0 and 1.0.
- `explanation`: Detailed root-cause technical analysis.
- `impact`: Concrete runtime failure or security consequence.
- `remediation`: Explicit code fix or architectural correction.
- `body`: Backward-compatible summary formatted for GitHub review comments.

## API retries and review submission

- GitHub GET requests retry timeouts, connection errors, and HTTP 429,
  500, 502, 503, or 504 responses, up to four total attempts, waiting 5,
  10, then 15 seconds between attempts unless a numeric `Retry-After`
  header is present (capped at 60 seconds). Other HTTP errors are
  reported without exposing response bodies or credentials.
- NVIDIA API requests retry network timeouts, connection errors, and
  HTTP 429, 500, 502, 503, or 504 responses, up to four total attempts.
  Numeric `Retry-After` values are honored up to 60 seconds; otherwise
  requests wait 5, 10, then 15 seconds. Permanent errors such as HTTP
  410 are not retried. A timeout or connection failure can be ambiguous:
  the provider may have received the request, so retries may result in
  repeated inference work or charges.
- GitHub review POST requests retry only explicit rate-limit responses
  (HTTP 429 or an HTTP 403 with the rate-limit header), at most twice,
  honoring numeric `Retry-After` values up to 60 seconds. They are not
  retried after timeouts, connection errors, or server errors because a
  lost response can leave the submission outcome uncertain; manually
  rerunning the workflow can post another review.
- If a review with inline comments receives HTTP 400 or 422 and GitHub's
  structured error response identifies an invalid
  `PullRequestReviewComment` field (such as `line`, `path`, `position`,
  `side`, or `subject_type`), the Action makes one summary-only fallback
  submission with the findings listed in its body. Unrelated or
  unrecognized errors propagate instead of triggering that fallback.
  Failure of the fallback also propagates.

## NVIDIA hosted API configuration

The configured hosted model is
`nvidia/nemotron-3.5-lightning-30b-a3b`, called through
`https://integrate.api.nvidia.com/v1/chat/completions` with Bearer
authentication. The request uses the OpenAI-compatible `messages` format,
`temperature`, `max_tokens: 8192`, `stream: false`, and
`chat_template_kwargs.enable_thinking: false`. NVIDIA's current
[model API reference](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b/apireference)
documents this model identifier, endpoint, Bearer authentication, request
format, and `choices[0].message.content` response shape.

An identifier in code or in NVIDIA's catalog does not guarantee that the
hosted endpoint is enabled for a particular API key, account, or region.
Before rollout, verify the model is currently available to the Action's
NVIDIA key in the hosted API catalog and make a real authenticated
non-streaming request. HTTP 410 should be treated as a model/endpoint
availability or configuration issue, not retried. This project does not
make live NVIDIA requests during unit or integration tests.

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

### Dual-mode auto-fix execution

The auto-fix engine operates in two distinct modes depending on project setup:

- **Mode 1 (Configured test suite)**: When `test-command` is provided (e.g.
  `["python", "-m", "pytest", "-q"]`), the agent attempts fixes, executes tests
  without a shell in a bounded retry loop (up to `max-fix-attempts`), and verifies
  clean test passes before committing and pushing. If tests fail, attempts are
  cleanly rolled back.
- **Mode 2 (Basic deterministic validation)**: When `test-command` is omitted,
  the agent proposes a fix and executes deterministic language syntax checks
  (e.g., Python AST/compilation, JSON validation, JavaScript syntax checks).
  - By default (`allow-unverified-push: 'false'`), changes are rolled back and
    an informative PR comment is posted explaining the fix passed basic syntax
    checks but was not pushed because no automated test suite was configured.
  - When explicitly opted-in (`allow-unverified-push: 'true'`), the validated fix
    is committed and pushed with an honest notice that only basic syntax checks ran.
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
pytest command. Tests include a controlled end-to-end pass through the
GitHub diff fetch, NVIDIA response parsing, finding validation, and review
publication boundaries. They do not verify live credentials, GitHub
permissions, NVIDIA account/model availability, or real PR comment
placement.

For focused tests, run:

```bash
python -m pytest -q tests/test_review.py
python -m pytest -q tests/test_llm_review.py
python -m pytest -q tests/test_fixer.py
python -m pytest -q tests/test_benchmark.py
```

## Known limitations

- Reviews cover only diff content returned by GitHub. Files without a
  patch are skipped, and the model cannot inspect unrelated code unless
  it is present in the supplied diff.
- The model can miss real issues or produce invalid findings. Invalid
  individual findings are discarded; if no valid findings remain but the
  model returned a usable summary, the Action publishes a summary-only
  review. If the full model response is malformed, the run fails instead of
  publishing unvalidated output.
- NVIDIA's API can return an asynchronous HTTP 202 response with a request
  identifier. The current client handles synchronous completions only and
  reports HTTP 202 as an API failure; it does not poll asynchronous jobs.
- There is no cross-run review deduplication. Separate workflow runs may
  submit similar or duplicate reviews.
- The optional auto-fix path is separate from review submission and is
  not enabled by default. It should be tested and permissioned separately.
