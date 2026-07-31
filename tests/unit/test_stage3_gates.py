"""Unit tests for expanded deterministic gates."""

import ast
from unittest.mock import patch

import pytest

from cognitive_agent_syndicate.config import Settings
from cognitive_agent_syndicate.schemas import (
    ArchitectureSpec,
    ArtifactBundle,
    DataModelSpec,
    EndpointSpec,
    GateRepairability,
    GateStatus,
    GeneratedFile,
)
from cognitive_agent_syndicate.validation.gates import (
    DEFAULT_GATES,
    GateRunner,
    gate_forbidden_generated_content,
    gate_python_syntax,
)
from tests.fixtures.pipeline_fixtures import (
    sample_architecture,
    sample_brief,
    sample_bundle,
    sample_review_approved,
)


def _run_gates(
    *,
    bundle: ArtifactBundle | None = None,
    architecture: ArchitectureSpec | None = None,
    settings: Settings | None = None,
    permitted_paths: list[str] | None = None,
    required_project_files: list[str] | None = None,
) -> list:
    runner = GateRunner()
    return runner.run(
        brief=sample_brief(),
        architecture=architecture or sample_architecture(),
        bundle=bundle or sample_bundle(),
        review=sample_review_approved(),
        settings=settings or Settings(_env_file=None),
        permitted_paths=permitted_paths or ["pyproject.toml", "src", "tests"],
        required_project_files=required_project_files or ["pyproject.toml"],
    )


def test_valid_python_passes_syntax_gate() -> None:
    results = _run_gates()
    gate = next(item for item in results if item.gate_id == "python_syntax")
    assert gate.status == GateStatus.PASSED


def test_invalid_python_fails_with_location() -> None:
    bundle = ArtifactBundle(
        files=[GeneratedFile(path="src/demo/broken.py", content="def broken(:\n    pass\n")]
    )
    results = _run_gates(bundle=bundle)
    gate = next(item for item in results if item.gate_id == "python_syntax")
    assert gate.status == GateStatus.FAILED
    assert "src/demo/broken.py" in gate.message
    assert "line" in gate.message


def test_python_syntax_gate_does_not_import_or_execute() -> None:
    bundle = ArtifactBundle(
        files=[
            GeneratedFile(
                path="src/demo/evil.py",
                content="import os\nos.remove('/tmp/should-not-run')\n",
            )
        ]
    )

    with patch(
        "cognitive_agent_syndicate.validation.gates.ast.parse", wraps=ast.parse
    ) as parse_mock:
        result = gate_python_syntax(
            sample_brief(),
            sample_architecture(),
            bundle,
            sample_review_approved(),
            Settings(_env_file=None),
            ["src"],
            [],
        )

    assert result.status == GateStatus.PASSED
    parse_mock.assert_called_once()


def test_endpoint_missing_data_model_fails() -> None:
    architecture = sample_architecture().model_copy(
        update={
            "endpoints": [
                EndpointSpec(
                    path="/items",
                    method="GET",
                    description="List items.",
                    response_model="MissingModel",
                )
            ],
            "data_models": [],
        }
    )
    results = _run_gates(architecture=architecture)
    gate = next(item for item in results if item.gate_id == "architecture_data_model_consistency")
    assert gate.status == GateStatus.FAILED
    assert "MissingModel" in gate.message


def test_hierarchy_collision_fails_gate() -> None:
    bundle = ArtifactBundle(
        files=[
            GeneratedFile(path="src", content="dir marker\n"),
            GeneratedFile(path="src/demo/service.py", content="def run() -> None:\n    pass\n"),
        ]
    )
    results = _run_gates(bundle=bundle)
    gate = next(item for item in results if item.gate_id == "file_hierarchy_collision")
    assert gate.status == GateStatus.FAILED


def test_case_insensitive_hierarchy_collision_fails_gate() -> None:
    bundle = ArtifactBundle(
        files=[
            GeneratedFile(path="SRC", content="dir marker\n"),
            GeneratedFile(path="src/service.py", content="def run() -> None:\n    pass\n"),
        ]
    )
    results = _run_gates(bundle=bundle)
    gate = next(item for item in results if item.gate_id == "file_hierarchy_collision")
    assert gate.status == GateStatus.FAILED


def test_backslash_hierarchy_collision_fails_gate() -> None:
    bundle = ArtifactBundle(
        files=[
            GeneratedFile(path="A/B.py", content="print('file')\n"),
            GeneratedFile(path="a", content="dir marker\n"),
        ]
    )
    results = _run_gates(bundle=bundle)
    gate = next(item for item in results if item.gate_id == "file_hierarchy_collision")
    assert gate.status == GateStatus.FAILED


@pytest.mark.parametrize(
    ("content", "label"),
    [
        ("result = eval('1+1')\n", "eval"),
        ("exec('pass')\n", "exec"),
        ("import subprocess\nsubprocess.run(['ls'])\n", "subprocess"),
        ("import os\nos.system('echo hi')\n", "os.system"),
        ("import importlib\nimportlib.import_module('os')\n", "importlib"),
        ("mod = __import__('os')\n", "__import__"),
    ],
)
def test_forbidden_patterns_fail(content: str, label: str) -> None:
    bundle = ArtifactBundle(files=[GeneratedFile(path="src/demo/risky.py", content=content)])
    result = gate_forbidden_generated_content(
        sample_brief(),
        sample_architecture(),
        bundle,
        sample_review_approved(),
        Settings(_env_file=None),
        ["src"],
        [],
    )
    assert result.status == GateStatus.FAILED, label
    assert "limited static policy check" in result.message.lower()


@pytest.mark.parametrize(
    "content",
    [
        "def retrieval(url: str) -> str:\n    return url\n",
        "evaluation_result = 1\n",
        "def execute_plan() -> None:\n    pass\n",
    ],
)
def test_safe_identifiers_do_not_trigger_forbidden_gate(content: str) -> None:
    bundle = ArtifactBundle(files=[GeneratedFile(path="src/demo/safe.py", content=content)])
    result = gate_forbidden_generated_content(
        sample_brief(),
        sample_architecture(),
        bundle,
        sample_review_approved(),
        Settings(_env_file=None),
        ["src"],
        [],
    )
    assert result.status == GateStatus.PASSED


def test_whitespace_eval_call_fails_forbidden_gate() -> None:
    bundle = ArtifactBundle(
        files=[GeneratedFile(path="src/demo/risky.py", content="eval (value)\n")]
    )
    result = gate_forbidden_generated_content(
        sample_brief(),
        sample_architecture(),
        bundle,
        sample_review_approved(),
        Settings(_env_file=None),
        ["src"],
        [],
    )
    assert result.status == GateStatus.FAILED
    assert "eval" in result.message


def test_aliased_subprocess_call_fails_forbidden_gate() -> None:
    content = "import subprocess as sp\nsp.run(['ls'])\n"
    bundle = ArtifactBundle(files=[GeneratedFile(path="src/demo/risky.py", content=content)])
    result = gate_forbidden_generated_content(
        sample_brief(),
        sample_architecture(),
        bundle,
        sample_review_approved(),
        Settings(_env_file=None),
        ["src"],
        [],
    )
    assert result.status == GateStatus.FAILED
    assert "subprocess.run" in result.message


def test_ordinary_safe_imports_do_not_fail_forbidden_gate() -> None:
    bundle = ArtifactBundle(
        files=[
            GeneratedFile(path="src/demo/safe.py", content="import re\nfrom typing import List\n")
        ]
    )
    result = gate_forbidden_generated_content(
        sample_brief(),
        sample_architecture(),
        bundle,
        sample_review_approved(),
        Settings(_env_file=None),
        ["src"],
        [],
    )
    assert result.status == GateStatus.PASSED


def test_gate_repairability_metadata_is_present() -> None:
    results = _run_gates()
    for gate in results:
        assert gate.gate_id
        assert gate.repairable in {GateRepairability.REPAIRABLE, GateRepairability.NON_REPAIRABLE}

    path_gate = next(
        item for item in results if item.gate_id == "paths_comply_with_permitted_prefixes"
    )
    assert path_gate.repairable == GateRepairability.NON_REPAIRABLE


def test_ast_parse_used_for_python_syntax_gate() -> None:
    bundle = ArtifactBundle(files=[GeneratedFile(path="src/demo/ok.py", content="x = 1\n")])
    with patch("ast.parse", wraps=ast.parse) as parse_mock:
        gate_python_syntax(
            sample_brief(),
            sample_architecture(),
            bundle,
            sample_review_approved(),
            Settings(_env_file=None),
            ["src"],
            [],
        )
    parse_mock.assert_called_once()


def test_required_file_missing_fails_gate() -> None:
    bundle = ArtifactBundle(
        files=[GeneratedFile(path="src/demo/service.py", content="def run() -> None:\n    pass\n")]
    )
    results = _run_gates(bundle=bundle, required_project_files=["pyproject.toml"])
    gate = next(item for item in results if item.gate_id == "required_common_project_files")
    assert gate.status == GateStatus.FAILED


def test_data_model_consistency_passes_with_models() -> None:
    architecture = sample_architecture().model_copy(
        update={
            "endpoints": [
                EndpointSpec(
                    path="/health",
                    method="GET",
                    description="Health check.",
                    response_model="HealthResponse",
                )
            ],
            "data_models": [
                DataModelSpec(
                    name="HealthResponse", description="Health payload.", fields=["status: str"]
                )
            ],
        }
    )
    results = _run_gates(architecture=architecture)
    gate = next(item for item in results if item.gate_id == "architecture_data_model_consistency")
    assert gate.status == GateStatus.PASSED


def test_gate_ordering_includes_new_gates() -> None:
    names = [gate.__name__ for gate in DEFAULT_GATES]
    assert names.index("gate_python_syntax") < names.index("gate_forbidden_generated_content")
