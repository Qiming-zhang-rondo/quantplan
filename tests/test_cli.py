from __future__ import annotations

import json
from pathlib import Path

from quantplan.cli import main


def test_cli_evaluates_ranking(tmp_path: Path, capsys) -> None:
    scores = tmp_path / "scores.json"
    truth = tmp_path / "truth.json"
    scores.write_text(json.dumps({"a": 3, "b": 2, "c": 1}), encoding="utf-8")
    truth.write_text(json.dumps({"a": 9, "b": 4, "c": 1}), encoding="utf-8")

    exit_code = main(
        [
            "evaluate-ranking",
            "--scores",
            str(scores),
            "--truth",
            str(truth),
            "--top-k",
            "1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["spearman_rho"] == 1.0
    assert payload["dangerous_recall"] == 1.0


def test_cli_yaml_guard_uses_blocking_exit_code(tmp_path: Path, capsys) -> None:
    source = tmp_path / "unknown.yaml"
    source.write_text(
        "apiversion: modelslim_v1\nspec:\n  future_execution_mode: true\n",
        encoding="utf-8",
    )

    exit_code = main(["inspect-yaml", str(source)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["static_policy_passed"] is False
    assert payload["execution_ready"] is False
    assert payload["execution_readiness"] == "not_assessed"
    assert payload["issues"][0]["severity"] == "blocking"
