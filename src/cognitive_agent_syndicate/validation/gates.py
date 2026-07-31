"""Deterministic gate evaluation for pipeline outputs."""

from __future__ import annotations

import time
from collections.abc import Callable

from cognitive_agent_syndicate.config import Settings
from cognitive_agent_syndicate.paths import canonical_path_key, normalize_relative_posix_path
from cognitive_agent_syndicate.schemas import (
    ArchitectureSpec,
    ArtifactBundle,
    GateResult,
    GateStatus,
    ReviewCategory,
    ReviewReport,
    ReviewSeverity,
    ReviewStatus,
    SystemBrief,
)

GateFn = Callable[
    [
        SystemBrief,
        ArchitectureSpec,
        ArtifactBundle,
        ReviewReport,
        Settings,
        list[str],
        list[str],
    ],
    GateResult,
]


def _gate(
    gate_name: str,
    evaluate: Callable[[], tuple[bool, str]],
) -> GateResult:
    started = time.perf_counter()
    passed, message = evaluate()
    duration_ms = (time.perf_counter() - started) * 1000.0
    return GateResult(
        gate_name=gate_name,
        status=GateStatus.PASSED if passed else GateStatus.FAILED,
        message=message,
        duration_ms=duration_ms,
    )


def gate_artifact_bundle_non_empty(
    _brief: SystemBrief,
    _architecture: ArchitectureSpec,
    bundle: ArtifactBundle,
    _review: ReviewReport,
    _settings: Settings,
    _permitted_paths: list[str],
    _required_project_files: list[str],
) -> GateResult:
    return _gate(
        "artifact_bundle_non_empty",
        lambda: (len(bundle.files) > 0, f"Artifact bundle contains {len(bundle.files)} file(s)."),
    )


def gate_file_count_within_limit(
    _brief: SystemBrief,
    _architecture: ArchitectureSpec,
    bundle: ArtifactBundle,
    _review: ReviewReport,
    settings: Settings,
    _permitted_paths: list[str],
    _required_project_files: list[str],
) -> GateResult:
    count = len(bundle.files)
    limit = settings.max_generated_files

    def evaluate() -> tuple[bool, str]:
        if count > limit:
            return False, f"Generated {count} files, exceeding limit of {limit}."
        return True, f"Generated file count {count} is within limit of {limit}."

    return _gate("file_count_within_limit", evaluate)


def gate_total_content_within_limit(
    _brief: SystemBrief,
    _architecture: ArchitectureSpec,
    bundle: ArtifactBundle,
    _review: ReviewReport,
    settings: Settings,
    _permitted_paths: list[str],
    _required_project_files: list[str],
) -> GateResult:
    total_chars = sum(len(generated_file.content) for generated_file in bundle.files)
    limit = settings.max_output_chars

    def evaluate() -> tuple[bool, str]:
        if total_chars > limit:
            return False, f"Generated {total_chars} characters, exceeding limit of {limit}."
        return True, f"Generated content size {total_chars} is within limit of {limit}."

    return _gate("total_content_within_limit", evaluate)


def gate_paths_comply_with_permitted_prefixes(
    _brief: SystemBrief,
    _architecture: ArchitectureSpec,
    bundle: ArtifactBundle,
    _review: ReviewReport,
    _settings: Settings,
    permitted_paths: list[str],
    _required_project_files: list[str],
) -> GateResult:
    normalized_prefixes = [normalize_relative_posix_path(prefix) for prefix in permitted_paths]

    def evaluate() -> tuple[bool, str]:
        for generated_file in bundle.files:
            path = generated_file.path
            allowed = any(
                path == prefix or path.startswith(f"{prefix}/") for prefix in normalized_prefixes
            )
            if not allowed:
                return False, f"Path {path!r} is outside permitted prefixes."
        return True, "All generated paths comply with permitted prefixes."

    return _gate("paths_comply_with_permitted_prefixes", evaluate)


def gate_acceptance_criteria_represented_in_review(
    brief: SystemBrief,
    architecture: ArchitectureSpec,
    _bundle: ArtifactBundle,
    review: ReviewReport,
    _settings: Settings,
    _permitted_paths: list[str],
    _required_project_files: list[str],
) -> GateResult:
    required_ids = {criterion.id for criterion in brief.acceptance_criteria if criterion.must_pass}
    required_ids.update(
        criterion.id for criterion in architecture.acceptance_criteria if criterion.must_pass
    )
    acceptance_outcomes = {
        finding.criterion_id: finding.passed
        for finding in review.findings
        if finding.category == ReviewCategory.ACCEPTANCE_CRITERION
    }

    def evaluate() -> tuple[bool, str]:
        missing = sorted(required_ids - acceptance_outcomes.keys())
        if missing:
            return False, f"Missing acceptance criterion findings for: {', '.join(missing)}."

        failed = sorted(
            criterion_id
            for criterion_id in required_ids
            if acceptance_outcomes.get(criterion_id) is False
        )
        if failed:
            return False, f"Acceptance criteria marked failed: {', '.join(failed)}."

        not_passed = sorted(
            criterion_id
            for criterion_id in required_ids
            if acceptance_outcomes.get(criterion_id) is not True
        )
        if not_passed:
            return False, f"Acceptance criteria not marked passed: {', '.join(not_passed)}."

        return True, "All required acceptance criteria passed in reviewer findings."

    return _gate("acceptance_criteria_represented_in_review", evaluate)


def gate_reviewer_status_consistent_with_findings(
    _brief: SystemBrief,
    _architecture: ArchitectureSpec,
    _bundle: ArtifactBundle,
    review: ReviewReport,
    _settings: Settings,
    _permitted_paths: list[str],
    _required_project_files: list[str],
) -> GateResult:
    blocking = [
        finding
        for finding in review.findings
        if finding.severity in {ReviewSeverity.ERROR, ReviewSeverity.CRITICAL}
    ]
    failed_criteria = [
        finding.criterion_id
        for finding in review.findings
        if finding.category == ReviewCategory.ACCEPTANCE_CRITERION and finding.passed is False
    ]

    def evaluate() -> tuple[bool, str]:
        if review.status == ReviewStatus.APPROVED:
            if blocking:
                return False, "Reviewer status is approved but error or critical findings exist."
            if failed_criteria:
                return (
                    False,
                    "Reviewer status is approved but failed acceptance criteria exist: "
                    f"{', '.join(sorted(failed_criteria))}.",
                )
        return True, "Reviewer status is consistent with findings."

    return _gate("reviewer_status_consistent_with_findings", evaluate)


def gate_required_common_project_files(
    _brief: SystemBrief,
    _architecture: ArchitectureSpec,
    bundle: ArtifactBundle,
    _review: ReviewReport,
    _settings: Settings,
    _permitted_paths: list[str],
    required_project_files: list[str],
) -> GateResult:
    if not required_project_files:
        return GateResult(
            gate_name="required_common_project_files",
            status=GateStatus.SKIPPED,
            message="No required common project files configured.",
            duration_ms=0.0,
        )

    generated_keys = {canonical_path_key(generated_file.path) for generated_file in bundle.files}
    normalized_required = [normalize_relative_posix_path(path) for path in required_project_files]

    def evaluate() -> tuple[bool, str]:
        missing = [
            path for path in normalized_required if canonical_path_key(path) not in generated_keys
        ]
        if missing:
            return False, f"Missing required project files: {', '.join(missing)}."
        return True, "All required common project files are present."

    return _gate("required_common_project_files", evaluate)


DEFAULT_GATES: tuple[GateFn, ...] = (
    gate_artifact_bundle_non_empty,
    gate_file_count_within_limit,
    gate_total_content_within_limit,
    gate_paths_comply_with_permitted_prefixes,
    gate_acceptance_criteria_represented_in_review,
    gate_reviewer_status_consistent_with_findings,
    gate_required_common_project_files,
)


class GateRunner:
    """Executes deterministic gates in a fixed order."""

    def __init__(self, gates: tuple[GateFn, ...] | None = None) -> None:
        self._gates = gates or DEFAULT_GATES

    def run(
        self,
        *,
        brief: SystemBrief,
        architecture: ArchitectureSpec,
        bundle: ArtifactBundle,
        review: ReviewReport,
        settings: Settings,
        permitted_paths: list[str],
        required_project_files: list[str],
    ) -> list[GateResult]:
        results: list[GateResult] = []
        for gate in self._gates:
            results.append(
                gate(
                    brief,
                    architecture,
                    bundle,
                    review,
                    settings,
                    permitted_paths,
                    required_project_files,
                )
            )
        return results

    @staticmethod
    def all_required_passed(results: list[GateResult]) -> bool:
        return all(result.status in {GateStatus.PASSED, GateStatus.SKIPPED} for result in results)
