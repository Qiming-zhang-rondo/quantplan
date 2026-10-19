"""Move verbose probe reports to a recoverable archive and retain concise summaries.

Requires an explicitly supplied existing archive directory, typically from
mktemp -d. Original reports are moved, never deleted. Files from other schemas,
including fast-stats-trained.json, are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
from pathlib import Path


def reproduction_command(arguments, filename):
    command = [".venv/bin/python", "experiments/probe_linear_taylor.py", "--output",
               f"/tmp/{Path(filename).stem}-reproduced.json"]
    for key, value in arguments.items():
        if key == "output":
            continue
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(option)
        elif isinstance(value, list):
            command.extend((option, *(str(item) for item in value)))
        else:
            command.extend((option, str(value)))
    return shlex.join(command)


def compact_report(source, archive_dir):
    raw = source.read_bytes()
    report = json.loads(raw)
    if report.get("schema") != "quantplan.linear-taylor-probe/v1":
        raise ValueError(f"Refusing to archive non-probe or already summarized report: {source}")
    archived = archive_dir / source.name
    if archived.exists():
        raise FileExistsError(f"Archive destination already exists: {archived}")
    grouped = {}
    for run in report["runs"]:
        grouped.setdefault((run["training_steps"], run["seed"]), []).append(run)
    seeds = []
    for (training_steps, seed), runs in sorted(grouped.items()):
        first = runs[0]
        quality_identical = all(
            run["ranking"] == first["ranking"] and run["plans"] == first["plans"] for run in runs
        )
        counts_identical = all(
            {method: data["actual_counts"] for method, data in run["methods"].items()}
            == {method: data["actual_counts"] for method, data in first["methods"].items()}
            for run in runs
        )
        row = {
            "seed": seed, "training_steps": training_steps, "repeat_count": len(runs),
            "quality_identical_across_repeats": quality_identical,
            "counts_identical_across_repeats": counts_identical,
            "training": first["training"], "ranking": first["ranking"],
            "plans": first["plans"],
            "actual_counts": {method: data["actual_counts"]
                              for method, data in first["methods"].items()},
            "analysis_seconds_by_repeat": {
                method: [run["methods"][method]["seconds"] for run in runs]
                for method in first["methods"]
            },
        }
        if not quality_identical or not counts_identical:
            row["repeat_details"] = [{
                "repeat": run["repeat"], "ranking": run["ranking"], "plans": run["plans"],
                "actual_counts": {method: data["actual_counts"]
                                  for method, data in run["methods"].items()},
            } for run in runs]
        seeds.append(row)
    summary = {
        "schema": "quantplan.linear-taylor-probe-summary/v1",
        "source_schema": report["schema"],
        "platform": report["platform"], "python_version": report["python_version"],
        "torch_version": report["torch_version"], "devices": report["devices"],
        "arguments": report["arguments"], "median_seconds": report["median_seconds"],
        "notes": report["notes"], "full_report_path": str(archived),
        "full_report_sha256": hashlib.sha256(raw).hexdigest(), "full_report_bytes": len(raw),
        "reproduction_command": reproduction_command(report["arguments"], source.name),
        "seeds": seeds,
    }
    # Construct and validate the summary before moving the full report.
    encoded = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    shutil.move(str(source), str(archived))
    source.write_text(encoded, encoding="utf-8")
    if hashlib.sha256(archived.read_bytes()).hexdigest() != summary["full_report_sha256"]:
        raise RuntimeError(f"Archive checksum mismatch: {archived}")
    print(f"{source}: {len(raw)} -> {len(encoded.encode())} bytes; full report: {archived}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    archive_dir = args.archive_dir.resolve()
    if not archive_dir.is_dir():
        parser.error("archive-dir must already exist (create it with mktemp -d)")
    reports = [source.resolve(strict=True) for source in args.reports]
    if len(set(reports)) != len(reports):
        parser.error("reports must be distinct")
    for source in reports:
        if source.parent == archive_dir:
            parser.error("archive-dir must differ from source report directory")
    for source in reports:
        compact_report(source, archive_dir)


if __name__ == "__main__":
    main()
