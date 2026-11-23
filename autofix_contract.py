"""Shared, validated finding contract for the review and auto-fix steps."""

import hashlib
import json
import os
import re


AUTOFIX_MIN_CONFIDENCE = 0.8
_AUTOFIX_CATEGORIES = {
    "correctness",
    "security",
    "performance",
    "concurrency",
    "resource_management",
    "error_handling",
    "edge_case",
}
_AUTOFIX_SEVERITIES = {"critical", "high", "medium"}
_REVIEW_ISSUE_FIELD_ORDER = (
    "path",
    "line",
    "category",
    "severity",
    "confidence",
    "body",
    "explanation",
    "impact",
    "remediation",
)
_REVIEW_ISSUE_FIELDS = set(_REVIEW_ISSUE_FIELD_ORDER)
_CONTRACT_FIELDS = _REVIEW_ISSUE_FIELDS | {"id", "fixable"}
_MAX_CONTRACT_BYTES = 1024 * 1024


class FindingContractError(ValueError):
    """Raised when the reviewer-to-fixer finding contract is invalid."""


def is_authorized_path(path: str, root_dir: str = None) -> bool:
    if not isinstance(path, str) or not path or "\x00" in path:
        return False
    if path.startswith(("/", "\\")) or "\\" in path:
        return False

    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return False
    if os.name == "nt" and any(":" in part for part in parts):
        return False
    if any(part.casefold() in (".git", ".github") for part in parts):
        return False

    workspace = os.path.realpath(root_dir or os.getcwd())
    current = workspace
    for part in parts:
        current = os.path.join(current, part)
        if os.path.islink(current):
            return False
    candidate = os.path.realpath(os.path.join(workspace, *parts))
    try:
        return os.path.commonpath((workspace, candidate)) == workspace
    except ValueError:
        return False


def _finding_id(issue: dict) -> str:
    identity = json.dumps(
        [
            issue["path"],
            issue["line"],
            issue["category"],
            issue["explanation"],
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"finding-{digest}"


def _is_fixable_finding(issue: dict) -> bool:
    if not isinstance(issue, dict):
        return False
    confidence = issue.get("confidence")
    return (
        isinstance(issue.get("category"), str)
        and issue["category"] in _AUTOFIX_CATEGORIES
        and isinstance(issue.get("severity"), str)
        and issue["severity"] in _AUTOFIX_SEVERITIES
        and isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and AUTOFIX_MIN_CONFIDENCE <= confidence <= 1
        and isinstance(issue.get("line"), int)
        and not isinstance(issue.get("line"), bool)
        and issue["line"] > 0
        and all(
            isinstance(issue.get(field), str) and issue[field].strip()
            for field in ("path", "body", "explanation", "remediation")
        )
        and isinstance(issue.get("impact"), str)
    )


def build_autofix_findings(issues: list, root_dir: str = None) -> list:
    if not isinstance(issues, list):
        raise FindingContractError("validated findings must be an array")

    findings = []
    for issue in issues:
        if not _is_fixable_finding(issue):
            continue
        if not is_authorized_path(issue["path"], root_dir):
            continue
        finding = {
            field: issue[field]
            for field in _REVIEW_ISSUE_FIELD_ORDER
        }
        finding["id"] = _finding_id(finding)
        finding["fixable"] = True
        findings.append(finding)
    return findings


def serialize_autofix_findings(findings: list) -> str:
    return json.dumps(findings, ensure_ascii=True, separators=(",", ":"))


def parse_autofix_findings(raw_findings: str, root_dir: str = None) -> list:
    if not isinstance(raw_findings, str) or not raw_findings.strip():
        raise FindingContractError("the validated finding contract is missing")
    if len(raw_findings.encode("utf-8")) > _MAX_CONTRACT_BYTES:
        raise FindingContractError("the validated finding contract is too large")
    try:
        findings = json.loads(raw_findings)
    except (TypeError, json.JSONDecodeError):
        raise FindingContractError("the validated finding contract is invalid") from None
    if not isinstance(findings, list):
        raise FindingContractError("the validated finding contract must be an array")

    validated = []
    seen_ids = set()
    for finding in findings:
        if not isinstance(finding, dict) or set(finding) != _CONTRACT_FIELDS:
            raise FindingContractError("a validated finding has an invalid shape")
        if finding.get("fixable") is not True:
            raise FindingContractError("a finding is not marked as fixable")
        if not _is_fixable_finding(finding):
            raise FindingContractError("a finding does not meet fixability requirements")
        if (
            not isinstance(finding.get("id"), str)
            or re.fullmatch(r"finding-[0-9a-f]{16}", finding["id"]) is None
            or finding["id"] != _finding_id(finding)
            or finding["id"] in seen_ids
        ):
            raise FindingContractError("a finding identifier is invalid")
        seen_ids.add(finding["id"])
        if not is_authorized_path(finding["path"], root_dir):
            continue
        validated.append(dict(finding))
    return validated
