"""Smell-family selection and finding filter helpers for the UI and pipeline."""

from __future__ import annotations

from typing import Any

from code_review_agent.reporter import SEVERITY_ORDER, filter_by_min_severity

PYTHON_SMELL_FAMILIES = ("code", "architectural", "structural")
ML_SMELL_FAMILIES = ("framework", "huggingface", "general")
ALL_SMELL_FAMILIES = PYTHON_SMELL_FAMILIES + ML_SMELL_FAMILIES

SMELL_FAMILY_LABELS: dict[str, str] = {
    "code": "Code smells",
    "architectural": "Architectural smells",
    "structural": "Structural smells",
    "framework": "ML framework anti-patterns",
    "huggingface": "Hugging Face anti-patterns",
    "general": "General ML anti-patterns",
}

PREFIX_TO_FAMILY: dict[str, str] = {
    "PYCS": "code",
    "PYAS": "architectural",
    "PYSS": "structural",
    "MLFW": "framework",
    "MLHF": "huggingface",
    "MLGN": "general",
}

TOOL_SOURCE_LABELS: dict[str, str] = {
    "python_smells": "Python smells",
    "ml_smells": "ML smells",
}


def smell_family_choices() -> list[tuple[str, str]]:
    return [(SMELL_FAMILY_LABELS[k], k) for k in ALL_SMELL_FAMILIES]


def default_smell_families() -> list[str]:
    return list(ALL_SMELL_FAMILIES)


def normalize_smell_families(selected: list[str] | tuple[str, ...] | None) -> list[str]:
    if selected is None:
        return default_smell_families()
    out: list[str] = []
    seen: set[str] = set()
    for item in selected:
        key = str(item).strip()
        if key in ALL_SMELL_FAMILIES and key not in seen:
            out.append(key)
            seen.add(key)
    return out


def split_smell_families(
    selected: list[str] | tuple[str, ...] | None,
) -> tuple[list[str], list[str]]:
    normalized = normalize_smell_families(selected)
    py = [f for f in normalized if f in PYTHON_SMELL_FAMILIES]
    ml = [f for f in normalized if f in ML_SMELL_FAMILIES]
    return py, ml


def infer_smell_family(finding: Any) -> str | None:
    extra = getattr(finding, "extra", None) or {}
    if isinstance(extra, dict) and extra.get("smell_family"):
        return str(extra["smell_family"])
    if isinstance(finding, dict):
        extra = finding.get("extra") or {}
        if isinstance(extra, dict) and extra.get("smell_family"):
            return str(extra["smell_family"])
    fid = str(getattr(finding, "finding_id", None) or finding.get("finding_id") or "")
    prefix = fid.split("-", 1)[0] if fid else ""
    return PREFIX_TO_FAMILY.get(prefix)


def infer_tool_source(finding: Any) -> str:
    tool = getattr(finding, "tool", None) or (
        finding.get("tool") if isinstance(finding, dict) else ""
    )
    return str(tool or "")


def _severity_value(finding: Any) -> str:
    sev = getattr(finding, "severity", None)
    if sev is None and isinstance(finding, dict):
        sev = finding.get("severity")
    if hasattr(sev, "value"):
        return str(sev.value)
    return str(sev or "medium").lower()


def category_choices(findings: list[Any]) -> list[tuple[str, str]]:
    cats: dict[str, int] = {}
    for finding in findings:
        cat = getattr(finding, "category", None) or (
            finding.get("category") if isinstance(finding, dict) else ""
        )
        cat = str(cat or "").strip()
        if cat:
            cats[cat] = cats.get(cat, 0) + 1
    ranked = sorted(cats.items(), key=lambda x: (-x[1], x[0].lower()))
    return [(f"{name} ({count})", name) for name, count in ranked]


def filter_findings(
    findings: list[Any],
    *,
    min_severity: str = "info",
    tools: list[str] | None = None,
    families: list[str] | None = None,
    categories: list[str] | None = None,
    search: str = "",
) -> list[Any]:
    """Filter Finding objects or finding dicts from saved JSON."""
    if not findings:
        return []

    min_sev = (min_severity or "info").lower()
    if min_sev not in SEVERITY_ORDER:
        min_sev = "info"

    tool_set = {t.strip() for t in (tools or []) if t.strip()}
    family_set = {f.strip() for f in (families or []) if f.strip()}
    category_set = {c.strip() for c in (categories or []) if c.strip()}
    needle = (search or "").strip().lower()

    # Reuse severity helper when we have Finding objects.
    if hasattr(findings[0], "severity"):
        from code_review_agent.reporter import Finding

        typed: list[Finding] = findings  # type: ignore[assignment]
        typed = filter_by_min_severity(typed, min_sev)
        out: list[Any] = []
        for finding in typed:
            if tool_set and infer_tool_source(finding) not in tool_set:
                continue
            fam = infer_smell_family(finding)
            if family_set and fam not in family_set:
                continue
            if category_set and finding.category not in category_set:
                continue
            if needle:
                blob = " ".join(
                    [
                        finding.finding_id,
                        finding.file,
                        finding.category,
                        finding.symbol or "",
                        finding.message or "",
                    ]
                ).lower()
                if needle not in blob:
                    continue
            out.append(finding)
        return out

    cutoff = SEVERITY_ORDER.index(min_sev)
    out_dicts: list[Any] = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        sev = _severity_value(finding)
        if sev not in SEVERITY_ORDER:
            continue
        if SEVERITY_ORDER.index(sev) > cutoff:
            continue
        if tool_set and infer_tool_source(finding) not in tool_set:
            continue
        fam = infer_smell_family(finding)
        if family_set and fam not in family_set:
            continue
        cat = str(finding.get("category") or "")
        if category_set and cat not in category_set:
            continue
        if needle:
            blob = " ".join(
                [
                    str(finding.get("finding_id") or ""),
                    str(finding.get("file") or ""),
                    cat,
                    str(finding.get("symbol") or ""),
                    str(finding.get("message") or ""),
                ]
            ).lower()
            if needle not in blob:
                continue
        out_dicts.append(finding)
    return out_dicts


def filter_td_predictions(
    predictions: list[dict[str, Any]],
    *,
    categories: list[str] | None = None,
    search: str = "",
    min_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    category_set = {c.strip() for c in (categories or []) if c.strip()}
    needle = (search or "").strip().lower()
    out: list[dict[str, Any]] = []
    for pred in predictions:
        if not isinstance(pred, dict):
            continue
        conf = float(pred.get("confidence") or pred.get("predicted_probability") or 0)
        if conf < min_confidence:
            continue
        cat = str(pred.get("category") or "")
        if category_set and cat not in category_set:
            continue
        text = str(pred.get("text") or "")
        if needle and needle not in text.lower() and needle not in cat.lower():
            continue
        out.append(pred)
    return out


def render_filtered_summary(
    payload: dict[str, Any],
    filtered: list[Any],
    *,
    td_filtered: list[dict[str, Any]] | None = None,
) -> str:
    """Compact markdown for the filtered Findings / Report view."""
    total = len(payload.get("findings") or [])
    shown = len(filtered)
    target = payload.get("target") or "?"
    lines = [
        f"## Filtered findings ({shown} of {total})",
        "",
        f"**Target:** `{target}`",
    ]
    if payload.get("health_score") is not None:
        lines.append(f"**Health score (full run):** {payload.get('health_score')}/100")
    lines.append("")

    if not filtered and not td_filtered:
        lines.append("_No findings match the current filters._")
        return "\n".join(lines)

    if filtered:
        lines.extend(
            [
                "| Sev | ID | Location | Category | Message |",
                "|-----|-----|----------|----------|---------|",
            ]
        )
        for finding in filtered[:200]:
            if hasattr(finding, "severity"):
                sev = finding.severity.value.upper()
                fid = finding.finding_id
                loc = f"{finding.file}:{finding.line}"
                cat = finding.category
                msg = (finding.message or "")[:100]
            else:
                sev = _severity_value(finding).upper()
                fid = str(finding.get("finding_id") or "")
                loc = f"{finding.get('file')}:{finding.get('line')}"
                cat = str(finding.get("category") or "")
                msg = str(finding.get("message") or "")[:100]
            lines.append(f"| {sev} | {fid} | `{loc}` | {cat} | {msg} |")
        if shown > 200:
            lines.append("")
            lines.append(f"_… and {shown - 200} more (narrow filters to see them)._")

    if td_filtered:
        lines.extend(["", "### Filtered technical debt", ""])
        lines.extend(
            [
                "| Snippet | Category | Confidence |",
                "|---------|----------|------------|",
            ]
        )
        for pred in td_filtered[:40]:
            text = str(pred.get("text") or "")[:90].replace("|", "\\|")
            cat = str(pred.get("category") or "")
            conf = pred.get("confidence") or pred.get("predicted_probability") or 0
            try:
                conf_s = f"{float(conf):.0%}"
            except (TypeError, ValueError):
                conf_s = "—"
            lines.append(f"| {text} | {cat} | {conf_s} |")

    return "\n".join(lines)
