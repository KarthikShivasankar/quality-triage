"""
Structured report generator.
Converts raw tool output into normalised Finding objects and renders
detailed Markdown (or JSON) reports with exact file:line:col locations.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TD_LABEL_MAP: dict[int, str] = {
    0:  "Architecture Debt",
    1:  "Build Debt",
    2:  "Code Debt",
    3:  "Defect Debt",
    4:  "Design Debt",
    5:  "Documentation Debt",
    6:  "Infrastructure Debt",
    7:  "People Debt",
    8:  "Process Debt",
    9:  "Requirement Debt",
    10: "Service Debt",
    11: "Test Automation Debt",
    12: "Test Debt",
    13: "Versioning Debt",
    14: "Security Debt",
    15: "Performance Debt",
    16: "Usability Debt",
    17: "No Debt",
}

SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]

SEVERITY_EMOJI = {
    "critical": "🔴",
    "high":     "🟠",
    "medium":   "🟡",
    "low":      "🟢",
    "info":     "🔵",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"
    INFO     = "info"


@dataclass
class Finding:
    finding_id: str
    tool: str
    category: str
    severity: Severity
    file: str             # relative path
    file_abs: str         # absolute path
    line: int
    col: int | None
    end_line: int | None = None
    symbol: str | None = None
    message: str = ""
    how_to_fix: str | None = None
    code_snippet: str | None = None
    confidence: float | None = None
    framework: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class ReportData:
    target: str
    analyzed_at: str
    provider: str
    model: str
    files_analyzed: int
    findings: list[Finding] = field(default_factory=list)
    td_predictions: list[dict] = field(default_factory=list)
    intel_summary: dict | None = None
    tools_run: list[str] = field(default_factory=list)
    duration_s: float | None = None
    config_source: str | None = None


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------

_SEVERITY_ML = {
    "data_leakage":        Severity.CRITICAL,
    "missing_random_seed": Severity.CRITICAL,
    "reproducibility":     Severity.CRITICAL,
}

_SEVERITY_STR = {
    "critical": Severity.CRITICAL,
    "high":     Severity.HIGH,
    "medium":   Severity.MEDIUM,
    "low":      Severity.LOW,
    "info":     Severity.INFO,
    "":         Severity.MEDIUM,
}


def _rel(abs_path: str, root: str) -> str:
    try:
        return os.path.relpath(abs_path, root)
    except ValueError:
        return abs_path


class FindingNormalizer:
    def __init__(self, project_root: str):
        self.root = project_root
        self._counter: dict[str, int] = {}

    def _next_id(self, prefix: str) -> str:
        self._counter[prefix] = self._counter.get(prefix, 0) + 1
        return f"{prefix}-{self._counter[prefix]:03d}"

    # ---- ML smells --------------------------------------------------------

    def normalize_ml_smells(self, raw: dict) -> list[Finding]:
        findings: list[Finding] = []
        for group_key, prefix in [
            ("framework_smells", "MLFW"),
            ("huggingface_smells", "MLHF"),
            ("general_ml_smells", "MLGN"),
        ]:
            for file_entry in raw.get(group_key, []):
                file_abs = file_entry.get("file", "")
                smells = file_entry.get("smells", [])
                if isinstance(smells, list):
                    for smell in smells:
                        self._norm_ml_smell(smell, file_abs, prefix, findings)
                elif isinstance(smells, dict):
                    # some versions return a dict keyed by smell name
                    for name, items in smells.items():
                        if isinstance(items, list):
                            for item in items:
                                item.setdefault("name", name)
                                self._norm_ml_smell(item, file_abs, prefix, findings)
        return findings

    def _norm_ml_smell(self, smell: Any, file_abs: str, prefix: str, out: list[Finding]):
        if not isinstance(smell, dict):
            return
        name = smell.get("name", "Unknown")
        fp = smell.get("file_path") or file_abs
        line = int(smell.get("line_number", 0) or 0)
        sev_key = name.lower().replace(" ", "_")
        sev = _SEVERITY_ML.get(sev_key, Severity.HIGH)
        out.append(Finding(
            finding_id=self._next_id(prefix),
            tool="ml_smells",
            category=name,
            severity=sev,
            file=_rel(fp, self.root),
            file_abs=str(fp),
            line=line,
            col=None,
            symbol=None,
            message=smell.get("description", smell.get("name", "")),
            how_to_fix=smell.get("how_to_fix"),
            code_snippet=smell.get("code_snippet"),
            framework=smell.get("framework"),
            extra={
                "benefits": smell.get("benefits", ""),
                "strategies": smell.get("strategies", ""),
            },
        ))

    # ---- Python smells ----------------------------------------------------

    def normalize_python_smells(self, raw: dict) -> list[Finding]:
        findings: list[Finding] = []
        for key, prefix in [
            ("code_smells", "PYCS"),
            ("architectural_smells", "PYAS"),
            ("structural_smells", "PYSS"),
        ]:
            smells = raw.get(key, {})
            items = smells if isinstance(smells, list) else []

            # try .smells attr format (wrapped dict with report key)
            if isinstance(smells, dict):
                if "report" in smells:
                    # text report fallback - skip normalisation
                    continue
                # try common container keys
                for container_key in ("smells", "results", "items", key):
                    if container_key in smells and isinstance(smells[container_key], list):
                        items = smells[container_key]
                        break

            for smell in items:
                self._norm_py_smell(smell, prefix, findings)
        return findings

    def _norm_py_smell(self, smell: Any, prefix: str, out: list[Finding]):
        if not isinstance(smell, dict):
            # Try treating as a dataclass/object
            try:
                smell = smell.__dict__
            except Exception:
                return

        name = smell.get("name", "Unknown")
        fp = smell.get("file_path", "")
        line = int(smell.get("line_number", 0) or 0)
        sev_raw = str(smell.get("severity", "medium") or "medium").lower()
        sev = _SEVERITY_STR.get(sev_raw, Severity.MEDIUM)
        out.append(Finding(
            finding_id=self._next_id(prefix),
            tool="python_smells",
            category=name,
            severity=sev,
            file=_rel(fp, self.root) if fp else "",
            file_abs=str(fp),
            line=line,
            col=None,
            symbol=smell.get("module_class"),
            message=smell.get("description", name),
            how_to_fix=smell.get("how_to_fix"),
            code_snippet=smell.get("code_snippet"),
        ))

    # ---- TD classifier ----------------------------------------------------

    def normalize_td_predictions(self, raw: dict) -> list[dict]:
        out = []
        for pred in raw.get("predictions", []):
            if not isinstance(pred, dict):
                continue
            cls_idx = pred.get("predicted_class")
            label = TD_LABEL_MAP.get(cls_idx, f"Class-{cls_idx}") if cls_idx is not None else "Unknown"
            prob = pred.get("predicted_probability", 0.0)
            if label == "No Debt":
                continue
            out.append({
                "text": pred.get("text", ""),
                "category": label,
                "confidence": round(float(prob), 3),
                "class_index": cls_idx,
                "error": pred.get("error"),
            })
        return out


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class ReportRenderer:
    def __init__(self, include_code_snippets: bool = True, max_snippet_lines: int = 10):
        self.include_code_snippets = include_code_snippets
        self.max_snippet_lines = max_snippet_lines

    # ---- Markdown ---------------------------------------------------------

    def render_markdown(self, data: ReportData) -> str:
        lines: list[str] = []
        w = lines.append

        target_display = data.target
        w(f"# Code Review Report: `{target_display}`")
        w("")
        w(f"**Analyzed:** {data.analyzed_at}  ")
        w(f"**Provider:** {data.provider} — `{data.model}`  ")
        w(f"**Files analyzed:** {data.files_analyzed}  ")
        w(f"**Tools run:** {', '.join(data.tools_run)}  ")
        if data.duration_s is not None:
            w(f"**Duration:** {data.duration_s:.1f}s  ")
        if data.config_source:
            w(f"**Config:** `{data.config_source}`  ")
        w("")
        w("---")
        w("")

        # --- Severity breakdown ---
        by_sev = {s: 0 for s in SEVERITY_ORDER}
        for f in data.findings:
            by_sev[f.severity.value] = by_sev.get(f.severity.value, 0) + 1

        w("## Summary")
        w("")
        w("| Severity | Count |")
        w("|----------|-------|")
        for sev in SEVERITY_ORDER:
            emoji = SEVERITY_EMOJI[sev]
            w(f"| {emoji} **{sev.upper()}** | {by_sev[sev]} |")
        w(f"| **TOTAL** | **{len(data.findings)}** |")
        w("")

        # TD summary
        if data.td_predictions:
            td_by_cat: dict[str, list[float]] = {}
            for p in data.td_predictions:
                td_by_cat.setdefault(p["category"], []).append(p["confidence"])
            w("### Technical Debt Summary")
            w("")
            w("| Debt Category | Snippets | Avg Confidence |")
            w("|---------------|----------|----------------|")
            for cat, probs in sorted(td_by_cat.items(), key=lambda x: -len(x[1])):
                avg = sum(probs) / len(probs)
                w(f"| {cat} | {len(probs)} | {avg:.0%} |")
            w("")

        # --- Findings by severity ---
        for sev_level in SEVERITY_ORDER:
            sev_findings = [f for f in data.findings if f.severity.value == sev_level]
            if not sev_findings:
                continue

            emoji = SEVERITY_EMOJI[sev_level]
            label = sev_level.upper()
            w(f"## {emoji} {label} Findings ({len(sev_findings)})")
            w("")

            for finding in sev_findings:
                self._render_finding(finding, lines)

        # --- Findings by file ---
        w("## Findings by File")
        w("")
        by_file: dict[str, list[Finding]] = {}
        for f in sorted(data.findings, key=lambda x: (x.file, x.line)):
            by_file.setdefault(f.file, []).append(f)

        for file_path, file_findings in sorted(by_file.items()):
            w(f"### `{file_path}` ({len(file_findings)} findings)")
            w("")
            w("| ID | Sev | Line:Col | Category | Symbol | Message |")
            w("|----|-----|----------|----------|--------|---------|")
            for f in sorted(file_findings, key=lambda x: x.line):
                loc_str = f"{f.line}" + (f":{f.col}" if f.col else "")
                sym = f"`{f.symbol}`" if f.symbol else "—"
                msg = f.message[:80] + "…" if len(f.message) > 80 else f.message
                w(f"| {f.finding_id} | {SEVERITY_EMOJI[f.severity.value]} | {loc_str} | {f.category} | {sym} | {msg} |")
            w("")

        # --- Technical Debt snippets ---
        if data.td_predictions:
            w("## Technical Debt Snippets")
            w("")
            high_conf = [p for p in data.td_predictions if p["confidence"] >= 0.7]
            if high_conf:
                w("### High-Confidence Debt (≥ 70%)")
                w("")
                w("| Snippet | Category | Confidence |")
                w("|---------|----------|------------|")
                for p in sorted(high_conf, key=lambda x: -x["confidence"]):
                    text_short = p["text"][:80].replace("|", "\\|") + ("…" if len(p["text"]) > 80 else "")
                    w(f"| {text_short} | {p['category']} | {p['confidence']:.0%} |")
                w("")

        # --- Code Intelligence ---
        if data.intel_summary:
            w("## Code Intelligence")
            w("")
            s = data.intel_summary
            w(f"**Files:** {s.get('files_analyzed', '?')}  ")
            w(f"**Symbols:** {s.get('total_symbols', '?')} "
              f"({s.get('total_functions', '?')} functions, "
              f"{s.get('total_classes', '?')} classes)  ")
            if s.get("parse_errors"):
                w(f"**Parse errors:** {len(s['parse_errors'])} files could not be parsed  ")
            w("")

            hotspots = s.get("complexity_hotspots", [])
            if hotspots:
                w("### Complexity Hotspots")
                w("")
                w("| Function | File | Line | CC | LOC | Params | Nesting |")
                w("|----------|------|------|----|-----|--------|---------|")
                for h in hotspots:
                    parent = f"{h['parent_class']}." if h.get("parent_class") else ""
                    loc_str = f"{h['file']}:{h['line']}:{h['col']}"
                    w(f"| `{parent}{h['name']}` | `{loc_str}` | {h['line']} | **{h['cyclomatic_complexity']}** | {h['loc']} | {h['param_count']} | {h['nesting_depth']} |")
                w("")

        w("---")
        w(f"*Generated by code-review-agent v0.2.0*")
        return "\n".join(lines)

    def _render_finding(self, f: Finding, lines: list[str]):
        loc = f"{f.file}:{f.line}" + (f":{f.col}" if f.col else "")
        symbol_str = f" in `{f.symbol}`" if f.symbol else ""
        lines.append(f"### [{f.finding_id}] {f.category}{symbol_str}")
        lines.append("")
        lines.append(f"**Location:** `{loc}`  ")
        if f.framework:
            lines.append(f"**Framework:** {f.framework}  ")
        lines.append(f"**Severity:** {SEVERITY_EMOJI[f.severity.value]} {f.severity.value.upper()}  ")
        lines.append("")
        if f.message:
            lines.append(f"{f.message}")
            lines.append("")
        if self.include_code_snippets and f.code_snippet:
            snippet = "\n".join(f.code_snippet.splitlines()[:self.max_snippet_lines])
            lines.append("```python")
            lines.append(snippet)
            lines.append("```")
            lines.append("")
        if f.how_to_fix:
            lines.append("**How to fix:**")
            lines.append("")
            lines.append(f.how_to_fix)
            lines.append("")
        if f.confidence is not None:
            lines.append(f"**Confidence:** {f.confidence:.0%}")
            lines.append("")

    # ---- JSON -------------------------------------------------------------

    def render_json(self, data: ReportData) -> str:
        def _enc(obj):
            if isinstance(obj, Enum):
                return obj.value
            if hasattr(obj, "__dict__"):
                return obj.__dict__
            return str(obj)

        return json.dumps(
            {
                "target": data.target,
                "analyzed_at": data.analyzed_at,
                "provider": data.provider,
                "model": data.model,
                "files_analyzed": data.files_analyzed,
                "tools_run": data.tools_run,
                "duration_s": data.duration_s,
                "findings": [f.__dict__ for f in data.findings],
                "td_predictions": data.td_predictions,
                "intel_summary": data.intel_summary,
            },
            default=_enc,
            indent=2,
        )


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(
    target: str,
    provider: str,
    model: str,
    ml_raw: dict | None = None,
    py_raw: dict | None = None,
    td_raw: dict | None = None,
    intel_summary: dict | None = None,
    files_analyzed: int = 0,
    tools_run: list[str] | None = None,
    duration_s: float | None = None,
    config_source: str | None = None,
) -> ReportData:
    """Build a ReportData from raw tool outputs."""
    normalizer = FindingNormalizer(project_root=target)
    all_findings: list[Finding] = []

    if ml_raw:
        all_findings.extend(normalizer.normalize_ml_smells(ml_raw))
    if py_raw:
        all_findings.extend(normalizer.normalize_python_smells(py_raw))

    td_preds: list[dict] = []
    if td_raw:
        td_preds = normalizer.normalize_td_predictions(td_raw)

    # Sort: severity first, then file+line
    sev_order = {s: i for i, s in enumerate(SEVERITY_ORDER)}
    all_findings.sort(key=lambda f: (sev_order.get(f.severity.value, 99), f.file, f.line))

    return ReportData(
        target=target,
        analyzed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        provider=provider,
        model=model,
        files_analyzed=files_analyzed,
        findings=all_findings,
        td_predictions=td_preds,
        intel_summary=intel_summary,
        tools_run=tools_run or [],
        duration_s=duration_s,
        config_source=config_source,
    )


def save_report(
    data: ReportData,
    output_dir: str = "./reports",
    fmt: str = "markdown",
) -> list[str]:
    """
    Save report to output_dir. Returns list of written file paths.
    fmt: "markdown" | "json" | "both"
    """
    renderer = ReportRenderer()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    target_slug = Path(data.target).name.replace(" ", "_")[:30]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(output_dir) / f"review-{target_slug}-{ts}"

    written: list[str] = []
    if fmt in ("markdown", "both"):
        md_path = str(base) + ".md"
        Path(md_path).write_text(renderer.render_markdown(data), encoding="utf-8")
        written.append(md_path)
    if fmt in ("json", "both"):
        json_path = str(base) + ".json"
        Path(json_path).write_text(renderer.render_json(data), encoding="utf-8")
        written.append(json_path)
    return written
