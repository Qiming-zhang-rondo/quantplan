from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from quantplan import __version__
from quantplan.adapters.modelslim import inspect_modelslim_yaml
from quantplan.p0.evaluation import evaluate_ranking


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2
