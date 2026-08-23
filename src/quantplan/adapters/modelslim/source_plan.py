"""Extract the immutable linear-quant boundary from an existing ModelSlim YAML."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import yaml

from quantplan.domain.models import PrecisionFormat


class SourcePlanError(ValueError):
    """The source YAML cannot define an unambiguous P0 planning boundary."""


@dataclass(frozen=True)
class SourceTemplate:
    precision: PrecisionFormat
    qconfig: Mapping[str, Any]
    source_hash: str


@dataclass(frozen=True)
class LinearSourceProfile:
    source_sha256: str
    eligible_modules: tuple[str, ...]
    templates: tuple[SourceTemplate, ...]
    linear_process_count: int

    def template_map(self) -> dict[PrecisionFormat, dict[str, Any]]:
        return {
            item.precision: copy.deepcopy(dict(item.qconfig))
            for item in self.templates
        }


@dataclass(frozen=True)
class TemplateResolution:
    templates: Mapping[PrecisionFormat, Mapping[str, Any]]
    origins: Mapping[PrecisionFormat, str]


def profile_linear_source(
    source: str | Path,
    candidate_module_names: Iterable[str],
) -> LinearSourceProfile:
    """Resolve source linear targets against a typed proxy target inventory."""

    source_path = Path(source)
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    try:
        document = yaml.safe_load(source_bytes)
    except yaml.YAMLError as exc:
        raise SourcePlanError(f"source ModelSlim YAML cannot be parsed: {exc}") from exc
    if not isinstance(document, Mapping):
        raise SourcePlanError("source ModelSlim YAML root must be a mapping")
    spec = document.get("spec")
    if not isinstance(spec, Mapping):
        raise SourcePlanError("source ModelSlim YAML must contain a mapping at $.spec")
    processes = spec.get("process")
    if not isinstance(processes, list):
        raise SourcePlanError("source ModelSlim YAML must contain $.spec.process as a list")

    candidates = tuple(candidate_module_names)
    if not candidates or any(not isinstance(name, str) or not name for name in candidates):
        raise SourcePlanError("candidate module inventory must contain non-empty names")
    if len(set(candidates)) != len(candidates):
        raise SourcePlanError("candidate module inventory contains duplicates")

    covered: set[str] = set()
    templates: dict[PrecisionFormat, dict[str, Any]] = {}
    template_hashes: dict[PrecisionFormat, str] = {}
    linear_process_count = 0
    for process in _walk_processes(processes):
        if process.get("type") != "linear_quant":
            continue
        linear_process_count += 1
        include = _pattern_list(process.get("include"), "linear_quant.include")
        exclude = _pattern_list(
            process.get("exclude", []),
            "linear_quant.exclude",
            allow_empty=True,
        )
        matched = {
            name
            for name in candidates
            if any(fnmatchcase(name, pattern) for pattern in include)
            and not any(fnmatchcase(name, pattern) for pattern in exclude)
        }
        covered.update(matched)

        qconfig = process.get("qconfig")
        precision = _qconfig_precision(qconfig)
        if precision is None:
            continue
        plain = _plain_mapping(qconfig, "linear_quant.qconfig")
        digest = _content_hash(plain)
        previous = template_hashes.get(precision)
        if previous is not None and previous != digest:
            raise SourcePlanError(
                f"source YAML contains multiple {precision.value} qconfig recipes; "
                "a single bitwidth change cannot preserve both strategies"
            )
        templates[precision] = plain
        template_hashes[precision] = digest

    if linear_process_count == 0:
        raise SourcePlanError("source YAML contains no linear_quant processor")
    eligible = tuple(name for name in candidates if name in covered)
    if not eligible:
        raise SourcePlanError(
            "source linear_quant rules cover none of the supported proxy targets"
        )
    if not templates:
        raise SourcePlanError(
            "source YAML contains no recognizable W4A8/W8A8 qconfig template"
        )
    return LinearSourceProfile(
        source_sha256=source_sha256,
        eligible_modules=eligible,
        templates=tuple(
            SourceTemplate(precision, templates[precision], template_hashes[precision])
            for precision in sorted(templates, key=lambda item: item.value)
        ),
        linear_process_count=linear_process_count,
    )


def resolve_precision_templates(
    profile: LinearSourceProfile,
    requested: Iterable[PrecisionFormat],
) -> TemplateResolution:
    """Reuse source recipes, changing only weight dtype when derivation is unique."""

    requested_formats = tuple(dict.fromkeys(requested))
    source_templates = profile.template_map()
    resolved: dict[PrecisionFormat, Mapping[str, Any]] = {}
    origins: dict[PrecisionFormat, str] = {}
    for precision in requested_formats:
        if precision is PrecisionFormat.BF16:
            continue
        if precision not in {PrecisionFormat.W4A8, PrecisionFormat.W8A8}:
            raise SourcePlanError(
                f"source-derived template resolution does not support {precision.value}"
            )
        if precision in source_templates:
            resolved[precision] = copy.deepcopy(source_templates[precision])
            origins[precision] = f"source_exact:{precision.value}"
            continue
        structural = {
            _template_structure_hash(template): source_precision
            for source_precision, template in source_templates.items()
        }
        if len(structural) != 1:
            raise SourcePlanError(
                f"cannot derive {precision.value}: source recipes differ beyond weight dtype"
            )
        source_precision = next(iter(structural.values()))
        template = copy.deepcopy(source_templates[source_precision])
        weight = template.get("weight")
        act = template.get("act")
        if (
            not isinstance(weight, dict)
            or not isinstance(act, Mapping)
            or act.get("dtype") != "int8"
        ):
            raise SourcePlanError(
                f"cannot derive {precision.value} from a non-WxA8 source recipe"
            )
        weight["dtype"] = "int4" if precision is PrecisionFormat.W4A8 else "int8"
        resolved[precision] = template
        origins[precision] = (
            f"source_derived:{source_precision.value}->"
            f"{precision.value}:$.weight.dtype_only"
        )
    return TemplateResolution(templates=resolved, origins=origins)


def _walk_processes(
    processes: Sequence[Any],
    *,
    depth: int = 0,
    active: set[int] | None = None,
) -> Iterator[Mapping[str, Any]]:
    if depth > 32:
        raise SourcePlanError("nested process depth exceeds 32")
    active = set() if active is None else active
    identity = id(processes)
    if identity in active:
        raise SourcePlanError("recursive process aliases are not supported")
    active.add(identity)
    try:
        for process in processes:
            if not isinstance(process, Mapping):
                continue
            yield process
            if process.get("type") == "group" and isinstance(
                process.get("configs"), list
            ):
                yield from _walk_processes(
                    process["configs"],
                    depth=depth + 1,
                    active=active,
                )
    finally:
        active.remove(identity)


def _pattern_list(value: Any, path: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise SourcePlanError(f"{path} must be a list of non-empty strings")
    patterns = tuple(value)
    if not patterns and not allow_empty:
        raise SourcePlanError(f"{path} must not be empty")
    if any("{" in pattern or "}" in pattern for pattern in patterns):
        raise SourcePlanError(
            f"{path} uses brace expansion without a pinned backend pattern adapter"
        )
    return patterns


def _qconfig_precision(value: Any) -> PrecisionFormat | None:
    if not isinstance(value, Mapping):
        return None
    act = value.get("act")
    weight = value.get("weight")
    if not isinstance(act, Mapping) or not isinstance(weight, Mapping):
        return None
    if act.get("dtype") != "int8":
        return None
    if weight.get("dtype") == "int4":
        return PrecisionFormat.W4A8
    if weight.get("dtype") == "int8":
        return PrecisionFormat.W8A8
    return None


def _plain_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SourcePlanError(f"{path} must be a mapping")
    try:
        payload = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SourcePlanError(f"{path} must contain plain JSON-compatible data") from exc
    if not isinstance(payload, dict):
        raise SourcePlanError(f"{path} must be a mapping")
    return payload


def _template_structure_hash(template: Mapping[str, Any]) -> str:
    normalized = copy.deepcopy(dict(template))
    weight = normalized.get("weight")
    if isinstance(weight, dict):
        weight["dtype"] = "<weight-bitwidth>"
    return _content_hash(normalized)


def _content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "LinearSourceProfile",
    "SourcePlanError",
    "SourceTemplate",
    "TemplateResolution",
    "profile_linear_source",
    "resolve_precision_templates",
]
