# AI Code Reviewer

This repository contains a GitHub Action that reviews pull-request diffs with
NVIDIA Nemotron and can optionally propose and push bounded fixes. Review
findings are validated before publication, and Auto-Fix is disabled by
default. Neither model output nor passing tests replace human review.

## 1. Purpose and operating model

The Reviewer and Auto-Fixer are separate stages connected by a validated JSON
finding contract. The Reviewer decides what changed and reports concrete
defects. The Auto-Fixer receives only eligible findings on authorized paths;
it does not re-review the repository or expand its own permissions.

Auto-Fix is intended for carefully controlled, same-repository pull requests.
It is not a general-purpose sandbox or an unattended merge mechanism.

## 2. End-to-end pipeline

The workflow is:

1. **Collect context:** `review.py` fetches pull-request metadata and changed
   file patches from GitHub. It skips files without a reviewable patch.
2. **Review:** `llm_review.py` chunks the diff and runs correctness and
   security/reliability passes, with conditional cross-file analysis.
3. **Validate findings:** each finding is checked against the supplied diff,
   including strict added-line anchoring, required values, category and
   severity, and safe output shape. Similar reports are deduplicated without
   intentionally merging independent defects.
4. **Publish and hand off:** `review.py` posts validated findings and emits a
   separate machine-readable Auto-Fix contract through `GITHUB_OUTPUT`.
   `action.yml` passes that contract to `fixer.py`.
5. **Plan and generate a patch:** the fixer parses the contract, confirms the
   PR and checkout identity, and gives the coding agent copies of authorized
   files plus only the validated findings.
6. **Validate and test:** changes are copied back only after workspace
   validation, then checked for path, file, and patch bounds. A configured
   test command runs without a shell.
7. **Verify and decide:** a separate model call checks each finding against
   before/after code and the exact patch. Commit and push are allowed only
   after the required gates succeed.

Failures are fail-closed: invalid contracts, unsafe changes, failed tests,
uncertain verification, changed PR heads, and unverified rollback prevent a
push.

## 3. Reviewer behavior and finding schema

The Reviewer operates on the changed diff, not the entire repository. It uses
specialized prompts to inspect correctness and logic, security and
reliability, and (when relevant) cross-file interface consistency. The
multi-pass prompts share required output fields and instruct the model to
continue through changed functions rather than stop after the first defect.

A validated review issue includes:

- `path` and `line`: the file and validated added-line location.
- `category` and `severity`: the issue classification and impact level.
- `confidence`: the model's confidence, bounded from 0 to 1.
- `explanation`, `impact`, and `remediation`: the root cause, consequence,
  and suggested correction.
- `body`: a formatted, backward-compatible comment body for GitHub review.

The model must anchor a regression caused by a removed or replaced statement
to the corresponding added replacement line. Deleted lines and unchanged
context are rejected as locations. Invalid individual findings are discarded;
malformed responses and API failures remain review failures.

## 4. Review chunking, validation, and outputs

Diffs are grouped into chunks of at most 20,000 characters and at most two
files per chunk. The multi-pass review has a 16-call logical model budget per
review run, including output-repair calls. A truncated primary pass is skipped
when another pass can still review the chunk; successful findings are retained.
If all primary passes for a chunk are truncated, the review fails rather than
claiming a complete review.

Finding validation checks that paths belong to supplied diffs and line
numbers are added lines. GitHub publication also checks that the PR head has
not changed during review. If no valid findings remain, the Action skips
review publication and emits `has_bugs=false`; invalid findings are not
published.

The Auto-Fix output is not the human-readable markdown summary. It is a
single-line JSON array written as the `autofix_findings` output. This avoids
parsing display text as instructions or as a source of permissions.

## 5. Model responsibilities and NVIDIA API

All three inference roles use the same model identifier:

- **Reviewer:** `nvidia/nemotron-3.5-super-vl-120b-a12b`, configured by
  `llm_review.py`.
- **Independent verification:** uses the Reviewer model through
  `llm_review.py`; it receives the finding, before/after excerpts, and exact
  patch and returns a strict structured decision.
- **Auto-Fixer coding agent:** `nvidia/nemotron-3.5-super-vl-120b-a12b`,
  configured in `fixer.py`.

The calls use NVIDIA's OpenAI-compatible endpoint
`https://integrate.api.nvidia.com/v1/chat/completions` with Bearer
authentication. Review and verification use the shared client configuration;
the coding agent is configured through LangChain's OpenAI-compatible client.
Model identifiers do not guarantee that a model is enabled for a particular
key, account, or region. Confirm availability for the Action's NVIDIA key
before rollout. Unit tests do not make live NVIDIA requests.

The reviewer retries selected transient API failures with bounded backoff.
HTTP 410 is not retried. Network failures can have an ambiguous outcome and
may cause repeated inference work or charges. GitHub review publication
retries only explicit rate limits; it does not blindly retry after an
ambiguous timeout or connection loss.

## 6. Review-only setup

Add a workflow in the repository to be reviewed:

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

Pin the Action to a reviewed release or commit instead of a moving branch for
production workflows. Add `NVIDIA_API_KEY` as an Actions secret. Review
publication requires `pull-requests: write`; `contents: read` is needed when
the workflow checks out the Action or project files.

## 7. Auto-Fix trust requirements and permissions

Auto-Fix is off by default. It requires a same-repository PR and a checkout
whose repository root, branch, and `HEAD` match the verified PR head SHA.
Fork-originated PRs are rejected. The workflow must grant `contents: write`
to push and `pull-requests: write` to validate the PR and report status.

The workflow invoking Auto-Fix must check out the PR head, not a synthetic
merge commit, and should disable checkout credential persistence. A trusted
workflow can use this pattern:

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

      # Install the project and test dependencies using a trusted workflow.

      - name: Run AI review and auto-fix
        uses: Oussamagrimida/AI-Code-reviewer-Github-Action@main
        with:
          nvidia-api-key: ${{ secrets.NVIDIA_API_KEY }}
          github-token: ${{ github.token }}
          enable-autofix: 'true'
          test-command: '["python", "-m", "pytest", "-q"]'
          max-fix-attempts: '2'
```

Do not use `pull_request_target` to run the fixer or PR-controlled tests with
privileged credentials. Workflow files and test commands must be trusted by
maintainers. Restrict eligible contributors according to the repository's
risk policy; same-repository origin alone does not make PR code trustworthy.

## 8. Reviewer-to-Fixer contract and eligibility

The Reviewer builds `autofix_findings` only from validated issues. Each
contract entry contains `id`, `path`, `line`, `category`, `severity`,
`confidence`, `body`, `explanation`, `impact`, `remediation`, and
`fixable=true`. IDs are deterministic for the finding identity and are
revalidated by the Fixer.

Only findings with an authorized path, actionable explanation and
remediation, confidence of at least 0.8, and severity `medium`, `high`, or
`critical` are eligible. `missing_test`, informational/low-severity items,
and findings without remediation are not auto-fixed. Paths under `.git` or
`.github`, absolute paths, traversal paths, and symlinked paths are excluded.
`has_bugs` reflects the presence of these eligible findings, not simply any
published review comment.

The Fixer rejects a missing, empty, malformed, oversized, or incorrectly
shaped contract. Unauthorized entries are filtered by the shared path policy
and never sent to the agent. If no authorized findings remain, Auto-Fix
stops without generating a patch.

## 9. Patch generation and validation

The coding agent receives only the validated findings and copies of their
authorized files in a temporary workspace. The workspace is limited to 2 MiB
per file and 8 MiB total; each agent invocation is limited to 12 graph steps.
Every attempt resets the workspace from the same original file-content
snapshot. Unexpected files and directories are removed without following
symlinks, and deleted authorized copies are restored before the next attempt.

The agent cannot write directly to the repository through its workspace.
Before applying its output, the fixer rejects symlinks, unexpected files,
deletions, renames, and paths outside the authorized set. Applying changes
uses no-follow directory handles and atomic replacement, and fails closed if
the required filesystem primitives are unavailable.

Only modifications to existing authorized files are allowed. The generated
patch is bounded to 200 changed lines and 64 KiB, checked before tests, and
checked again before staging. `.git`, `.github`, path traversal, and symlink
protections are maintained throughout.

## 10. Tests, independent verification, and retries

When `test-command` is set, it must be a JSON array of arguments. It is
executed without shell parsing and has a 10-minute timeout. Test output is
bounded and redacted before being passed to a retry. A failed test attempt is
rolled back before retrying.

After tests pass, independent verification checks the original finding
against the before/after code and exact patch. It must return a result for
every finding, confirm each fix with confidence of at least 0.8, and report
no regressions. Missing, malformed, failed, uncertain, or negative
verification prevents a commit or push. This is a separate inference, not an
independent provider or deterministic proof.

`max-fix-attempts` accepts 1 through 5 and defaults to 2. The same limit
applies when test validation or verification rejects a patch. Infrastructure
failures, unsafe file changes, or an unverifiable rollback stop the process
instead of retrying.

When `test-command` is omitted, Mode 2 runs deterministic syntax/structure
checks where supported (including Python compilation and JSON parsing).
By default, changes are rolled back and not pushed. Setting
`allow-unverified-push: 'true'` explicitly permits a push only after basic
validation **and** independent verification; the resulting notice makes clear
that a project test suite did not run.

## 11. Rollback, Git operations, and credential handling

The fixer requires a clean working tree before it starts and snapshots the
checkout, index, and bounded Git metadata. It checks that tests did not alter
the checkout or Git metadata. Rollback is verified; if verification fails,
the fixer stops and does not commit or push.

The fixer stages only the validated changed paths. It rejects staged
pre-existing changes, unauthorized modifications, additions/deletions, and
staged/unstaged mismatches. It rechecks the PR head before commit and push.
Pushes are non-forced and target the verified PR branch.

The checkout's persisted GitHub authorization header is removed before tests.
`GITHUB_TOKEN` and `NVIDIA_API_KEY` are omitted from the test subprocess
environment. Push credentials are supplied only to the Git push subprocess;
the token is not placed in the command arguments or remote URL. Diagnostic
logs redact known credentials and sensitive URLs, while public PR status
comments remain generic.

## 12. Tests, limitations, and rollout readiness

Run the unit suite from the repository root:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
```

For focused suites:

```bash
python -m pytest -q tests/test_llm_review.py
python -m pytest -q tests/test_review.py
python -m pytest -q tests/test_fixer.py
python -m pytest -q tests/test_benchmark.py
```

The tests use controlled responses and temporary Git repositories. They cover
review parsing and validation, contract transfer, path restrictions,
workspace reset, patch validation, tests, verification, rollback, and push
gates. They do not prove live NVIDIA model behavior, real GitHub permissions,
or real PR comment placement.

Known limitations:

- Review coverage is limited to GitHub-returned diff patches. Files without
  patches are skipped, and code outside the supplied diff may not be
  available to the model.
- Models can miss defects or produce invalid output. Invalid individual
  findings are discarded; if no valid findings remain, review publication
  and Auto-Fix are skipped. Malformed full responses and genuine API errors
  fail the review.
- The NVIDIA client handles synchronous completions. An asynchronous HTTP
  202 response is treated as an API failure; the client does not poll jobs.
- There is no cross-run deduplication, so separate workflow runs may post
  similar reviews.
- The test process has a reduced environment and Linux process protections,
  but it is not an OS sandbox. Test code may still access the runner's
  ordinary filesystem and network; privileged processes are outside this
  protection.
- The model verifier can make mistakes. Tests passing and model verification
  succeeding are not guarantees that a fix is correct.

Begin with supervised, same-repository integration testing. Do not treat
passing unit tests as evidence that the Auto-Fixer is ready for unattended
production use.
