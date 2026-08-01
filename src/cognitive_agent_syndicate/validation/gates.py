"""Deterministic gate evaluation for pipeline outputs."""

from __future__ import annotations

import ast
import time
from collections.abc import Callable

from cognitive_agent_syndicate.config import Settings
from cognitive_agent_syndicate.paths import (
    canonical_path_key,
    find_path_hierarchy_collision,
    normalize_relative_posix_path,
)
from cognitive_agent_syndicate.schemas import (
    ArchitectureSpec,
    ArtifactBundle,
    GateRepairability,
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

_FORBIDDEN_CONTENT_MESSAGE = (
    "Limited static policy check detected disallowed execution pattern. "
    "This is not a complete security scanner."
)


def _gate(
    *,
    gate_id: str,
    gate_name: str,
    required: bool = True,
    repairable: GateRepairability = GateRepairability.REPAIRABLE,
    evaluate: Callable[[], tuple[bool, str]],
) -> GateResult:
    started = time.perf_counter()
    passed, message = evaluate()
    duration_ms = (time.perf_counter() - started) * 1000.0
    return GateResult(
        gate_id=gate_id,
        gate_name=gate_name,
        status=GateStatus.PASSED if passed else GateStatus.FAILED,
        message=message,
        duration_ms=duration_ms,
        required=required,
        repairable=repairable,
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
        gate_id="artifact_bundle_non_empty",
        gate_name="artifact_bundle_non_empty",
        evaluate=lambda: (
            len(bundle.files) > 0,
            f"Artifact bundle contains {len(bundle.files)} file(s).",
        ),
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

    return _gate(
        gate_id="file_count_within_limit", gate_name="file_count_within_limit", evaluate=evaluate
    )


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

    return _gate(
        gate_id="total_content_within_limit",
        gate_name="total_content_within_limit",
        evaluate=evaluate,
    )


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

    return _gate(
        gate_id="paths_comply_with_permitted_prefixes",
        gate_name="paths_comply_with_permitted_prefixes",
        repairable=GateRepairability.NON_REPAIRABLE,
        evaluate=evaluate,
    )


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

    return _gate(
        gate_id="acceptance_criteria_represented_in_review",
        gate_name="acceptance_criteria_represented_in_review",
        evaluate=evaluate,
    )


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

    return _gate(
        gate_id="reviewer_status_consistent_with_findings",
        gate_name="reviewer_status_consistent_with_findings",
        evaluate=evaluate,
    )


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
            gate_id="required_common_project_files",
            gate_name="required_common_project_files",
            status=GateStatus.SKIPPED,
            message="No required common project files configured.",
            duration_ms=0.0,
            required=False,
            repairable=GateRepairability.REPAIRABLE,
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

    return _gate(
        gate_id="required_common_project_files",
        gate_name="required_common_project_files",
        evaluate=evaluate,
    )


def gate_python_syntax(
    _brief: SystemBrief,
    _architecture: ArchitectureSpec,
    bundle: ArtifactBundle,
    _review: ReviewReport,
    _settings: Settings,
    _permitted_paths: list[str],
    _required_project_files: list[str],
) -> GateResult:
    def evaluate() -> tuple[bool, str]:
        for generated_file in bundle.files:
            if not generated_file.path.endswith(".py"):
                continue
            try:
                ast.parse(generated_file.content, filename=generated_file.path)
            except SyntaxError as exc:
                location = f"line {exc.lineno}, column {exc.offset or 0}"
                return (
                    False,
                    f"Syntax error in {generated_file.path} at {location}: {exc.msg}.",
                )
        return True, "All generated Python files parsed successfully."

    return _gate(gate_id="python_syntax", gate_name="python_syntax", evaluate=evaluate)


def gate_architecture_data_model_consistency(
    _brief: SystemBrief,
    architecture: ArchitectureSpec,
    _bundle: ArtifactBundle,
    _review: ReviewReport,
    _settings: Settings,
    _permitted_paths: list[str],
    _required_project_files: list[str],
) -> GateResult:
    model_names = {model.name for model in architecture.data_models}

    def evaluate() -> tuple[bool, str]:
        missing: list[str] = []
        for endpoint in architecture.endpoints:
            if endpoint.request_model and endpoint.request_model not in model_names:
                missing.append(
                    f"{endpoint.method} {endpoint.path} request_model={endpoint.request_model}"
                )
            if endpoint.response_model and endpoint.response_model not in model_names:
                missing.append(
                    f"{endpoint.method} {endpoint.path} response_model={endpoint.response_model}"
                )
        if missing:
            return False, f"Missing data model references: {'; '.join(missing)}."
        return True, "All endpoint model references exist in architecture data_models."

    return _gate(
        gate_id="architecture_data_model_consistency",
        gate_name="architecture_data_model_consistency",
        evaluate=evaluate,
    )


def gate_file_hierarchy_collision(
    _brief: SystemBrief,
    _architecture: ArchitectureSpec,
    bundle: ArtifactBundle,
    _review: ReviewReport,
    _settings: Settings,
    _permitted_paths: list[str],
    _required_project_files: list[str],
) -> GateResult:
    paths = [generated_file.path for generated_file in bundle.files]

    def evaluate() -> tuple[bool, str]:
        collision = find_path_hierarchy_collision(paths)
        if collision is not None:
            left, right = collision
            return False, f"Path hierarchy collision between {left!r} and {right!r}."
        return True, "No file hierarchy collisions detected."

    return _gate(
        gate_id="file_hierarchy_collision",
        gate_name="file_hierarchy_collision",
        repairable=GateRepairability.REPAIRABLE,
        evaluate=evaluate,
    )


_FORBIDDEN_DIRECT_CALLS = frozenset({"eval", "exec", "__import__"})
_FORBIDDEN_ATTRIBUTE_CALLS = frozenset(
    {
        "os.system",
        "subprocess.run",
        "subprocess.call",
        "subprocess.Popen",
        "subprocess.check_call",
        "subprocess.check_output",
        "importlib.import_module",
    }
)


def _resolve_call_target(func: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(func, ast.Name):
        if func.id in _FORBIDDEN_DIRECT_CALLS:
            return func.id
        resolved = aliases.get(func.id)
        if resolved in _FORBIDDEN_DIRECT_CALLS:
            return resolved
        if resolved in _FORBIDDEN_ATTRIBUTE_CALLS:
            return resolved
        return None

    if isinstance(func, ast.Attribute):
        chain = _attribute_chain(func)
        if chain in _FORBIDDEN_ATTRIBUTE_CALLS:
            return chain
        if "." in chain:
            module, _, attr = chain.rpartition(".")
            alias_target = aliases.get(module)
            if alias_target is not None:
                resolved = f"{alias_target}.{attr}"
                if resolved in _FORBIDDEN_ATTRIBUTE_CALLS:
                    return resolved
        return None

    return None


def _collect_import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".")[-1]
                aliases[local_name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    continue
                local_name = alias.asname or alias.name
                full_name = f"{module}.{alias.name}" if module else alias.name
                aliases[local_name] = full_name
    return aliases


def _find_forbidden_patterns(content: str, path: str) -> list[str]:
    violations: list[str] = []
    if not path.endswith(".py"):
        return violations

    try:
        tree = ast.parse(content, filename=path)
    except SyntaxError:
        return violations

    import_aliases = _collect_import_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = _resolve_call_target(node.func, import_aliases)
        if target is not None:
            violations.append(f"{path}: call to {target}(...)")

    return violations


def _attribute_chain(node: ast.Attribute) -> str:
    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def gate_forbidden_generated_content(
    _brief: SystemBrief,
    _architecture: ArchitectureSpec,
    bundle: ArtifactBundle,
    _review: ReviewReport,
    _settings: Settings,
    _permitted_paths: list[str],
    _required_project_files: list[str],
) -> GateResult:
    def evaluate() -> tuple[bool, str]:
        all_violations: list[str] = []
        for generated_file in bundle.files:
            if not generated_file.path.endswith(".py"):
                continue
            all_violations.extend(
                _find_forbidden_patterns(generated_file.content, generated_file.path)
            )
        if all_violations:
            detail = "; ".join(all_violations[:5])
            return False, f"{_FORBIDDEN_CONTENT_MESSAGE} Violations: {detail}."
        return True, f"{_FORBIDDEN_CONTENT_MESSAGE} No violations detected."

    return _gate(
        gate_id="forbidden_generated_content",
        gate_name="forbidden_generated_content",
        evaluate=evaluate,
    )


DEFAULT_GATES: tuple[GateFn, ...] = (
    gate_artifact_bundle_non_empty,
    gate_file_count_within_limit,
    gate_total_content_within_limit,
    gate_paths_comply_with_permitted_prefixes,
    gate_python_syntax,
    gate_required_common_project_files,
    gate_architecture_data_model_consistency,
    gate_file_hierarchy_collision,
    gate_forbidden_generated_content,
    gate_acceptance_criteria_represented_in_review,
    gate_reviewer_status_consistent_with_findings,
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
        return all(
            result.status in {GateStatus.PASSED, GateStatus.SKIPPED}
            for result in results
            if result.required
        )

    @staticmethod
    def failed_required(results: list[GateResult]) -> list[GateResult]:
        return [
            result for result in results if result.required and result.status == GateStatus.FAILED
        ]

    @staticmethod
    def has_non_repairable_failure(results: list[GateResult]) -> bool:
        return any(
            result.required
            and result.status == GateStatus.FAILED
            and result.repairable == GateRepairability.NON_REPAIRABLE
            for result in results
        )
