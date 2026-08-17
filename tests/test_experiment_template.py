from pathlib import Path

import yaml

EXPERIMENT = Path(__file__).parents[1] / "experiments" / "qwen3_5_35b_p0.yaml"
CANONICAL_STATUSES = {
    "SUPPORTED_VERIFIED",
    "SUPPORTED_EXPERIMENTAL",
    "KNOWN_UNSUPPORTED",
    "UNKNOWN",
    "OUT_OF_SCOPE",
}


def test_p0_template_is_explicitly_non_runnable() -> None:
    document = yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))

    assert document["template"] is True
    assert document["runnable"] is False
    assert document["blocking_placeholders"]


def test_p0_candidate_statuses_use_canonical_vocabulary() -> None:
    document = yaml.safe_load(EXPERIMENT.read_text(encoding="utf-8"))
    statuses = {item["status"] for item in document["candidates"].values()}

    assert statuses <= CANONICAL_STATUSES
