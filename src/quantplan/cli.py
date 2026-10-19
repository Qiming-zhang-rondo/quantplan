from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from quantplan import __version__
from quantplan.adapters.modelslim import inspect_modelslim_yaml
from quantplan.doctor import inspect_environment
from quantplan.p0.evaluation import evaluate_ranking
from quantplan.run_config import init_run_config, load_run_config
from quantplan.runtime import build_torch_runtime
from quantplan.workflow import execute_workflow


def _read_module_inventory(path: Path | None) -> tuple[str, ...] | None:
    if path is None:
        return None
    modules = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not modules:
        raise ValueError("module inventory is empty")
    return modules


def _read_score_mapping(path: Path) -> Mapping[str, float]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    try:
        return {str(key): float(score) for key, score in value.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} contains a non-numeric score") from exc


def _inspect_yaml(args: argparse.Namespace) -> int:
    result = inspect_modelslim_yaml(
        args.yaml,
        module_names=_read_module_inventory(args.modules),
    )
    payload = {
        "source_path": result.source_path,
        "source_sha256": result.source_sha256,
        "manifest_id": result.manifest_id,
        "static_policy_passed": result.static_policy_passed,
        "execution_readiness": result.execution_readiness,
        "execution_ready": result.execution_ready,
        "issues": [
            {
                "code": issue.code,
                "path": issue.path,
                "message": issue.message,
                "severity": issue.severity.value,
            }
            for issue in result.issues
        ],
        "pattern_expansions": [
            {
                "field_path": expansion.field_path,
                "pattern": expansion.pattern,
                "matched_modules": list(expansion.matched_modules),
            }
            for expansion in result.pattern_expansions
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.static_policy_passed else 2


def _evaluate_ranking(args: argparse.Namespace) -> int:
    result = evaluate_ranking(
        _read_score_mapping(args.scores),
        _read_score_mapping(args.truth),
        top_k=args.top_k,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _init_run(args: argparse.Namespace) -> int:
    destination = init_run_config(
        args.config,
        model_path=args.model_path,
        data_path=args.data_path,
        original_yaml=args.original_yaml,
        output_dir=args.output_dir,
        device=args.device,
        overwrite=args.force,
    )
    config = load_run_config(destination)
    payload = {
        "config": str(destination.resolve()),
        "runnable": not config.blocking_placeholders,
        "blocking_placeholders": list(config.blocking_placeholders),
        "next_command": f"quantplan doctor --config {destination}",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _doctor(args: argparse.Namespace) -> int:
    config = load_run_config(args.config)
    report = inspect_environment(config)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.ready_for_analysis else 2


def _run(args: argparse.Namespace) -> int:
    config = load_run_config(args.config)
    if args.latency_mode is not None:
        config = replace(config, latency=replace(config.latency, mode=args.latency_mode))
    preflight = inspect_environment(config)
    if not preflight.ready_for_analysis:
        print(
            json.dumps(
                {
                    "status": "preflight_blocked",
                    "doctor": preflight.to_dict(),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    runtime = build_torch_runtime(config)
    result = execute_workflow(
        config,
        runtime,
        overwrite=args.overwrite,
        progress=_print_progress,
        analysis_mode=args.mode,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    successful = {"complete_analysis_only", "complete_partial_analysis"}
    return 0 if result.status in successful else 3


def _print_progress(stage: str, details: Mapping[str, Any]) -> None:
    detail_text = " ".join(f"{key}={value}" for key, value in details.items())
    print(f"[quantplan] {stage} {detail_text}".rstrip(), file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantplan",
        description="Evidence-driven mixed-precision planning prototype",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-yaml",
        help="inspect a ModelSlim YAML without modifying it",
    )
    inspect_parser.add_argument("yaml", type=Path)
    inspect_parser.add_argument(
        "--modules",
        type=Path,
        help="newline-delimited named_modules inventory for include/exclude checks",
    )
    inspect_parser.set_defaults(handler=_inspect_yaml)

    ranking_parser = subparsers.add_parser(
        "evaluate-ranking",
        help="compare proposal scores with held-out intervention damage",
    )
    ranking_parser.add_argument("--scores", type=Path, required=True)
    ranking_parser.add_argument("--truth", type=Path, required=True)
    ranking_parser.add_argument("--top-k", type=int)
    ranking_parser.set_defaults(handler=_evaluate_ranking)

    init_parser = subparsers.add_parser(
        "init",
        help="create a run config with conservative defaults",
    )
    init_parser.add_argument("config", type=Path, nargs="?", default=Path("quantplan.yaml"))
    init_parser.add_argument("--model-path", type=Path)
    init_parser.add_argument("--data-path", type=Path)
    init_parser.add_argument("--original-yaml", type=Path)
    init_parser.add_argument("--output-dir", type=Path)
    init_parser.add_argument("--device")
    init_parser.add_argument("--force", action="store_true")
    init_parser.set_defaults(handler=_init_run)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="check paths and runtime dependencies without loading model weights",
    )
    doctor_parser.add_argument("--config", type=Path, required=True)
    doctor_parser.set_defaults(handler=_doctor)

    run_parser = subparsers.add_parser(
        "run",
        help="analyze, plan, validate and emit a guarded ModelSlim YAML",
    )
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument(
        "--mode",
        choices=("exact", "fast", "fast-stats", "fast-taylor"),
        default="exact",
        help=(
            "exact full-model interventions, fast local block replay, or experimental "
            "fast-stats factorized gradient statistics / fast-taylor shared projection "
            "signed first-order ranking without candidate affine or block replay"
        ),
    )
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace artifacts already present in output_dir",
    )
    run_parser.add_argument(
        "--latency-mode",
        choices=("off", "static", "calibrated"),
        help="override the configured fast latency model for this run",
    )
    run_parser.set_defaults(handler=_run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2
