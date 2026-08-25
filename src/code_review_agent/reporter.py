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
    0: "Architecture Debt",
    1: "Build Debt",
    2: "Code Debt",
    3: "Defect Debt",
    4: "Design Debt",
    5: "Documentation Debt",
    6: "Infrastructure Debt",
    7: "People Debt",
    8: "Process Debt",
    9: "Requirement Debt",
    10: "Service Debt",
    11: "Test Automation Debt",
    12: "Test Debt",
    13: "Versioning Debt",
    14: "Security Debt",
    15: "Performance Debt",
    16: "Usability Debt",
    17: "No Debt",
}


def td_class_label(pred: dict[str, Any]) -> str:
    """Human label for a TD prediction (binary ONNX or 18-class)."""
    idx = pred.get("predicted_class")
    probs = pred.get("class_probabilities")
    if isinstance(probs, list) and len(probs) == 2:
        return "Technical Debt" if idx == 1 else "No Debt"
    if idx is None:
        return "Unknown"
    try:
        return TD_LABEL_MAP.get(int(idx), f"Class-{idx}")
    except (TypeError, ValueError):
        return str(idx)


SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]

_HEALTH_WEIGHTS = {
    "critical": 18,
    "high": 8,
    "medium": 3,
    "low": 1,
    "info": 0,
}

_SARIF_LEVEL = {
    "critical": "error",
    "high": "error",
    "medium": "warning",
    "low": "note",
    "info": "note",
}

SEVERITY_EMOJI = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🟢",
    "info": "🔵",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Finding:
    finding_id: str
    tool: str
    category: str
    severity: Severity
    file: str  # relative path
    file_abs: str  # absolute path
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
    narrative: str | None = None

    @property
    def health_score(self) -> int:
        return compute_health_score(self.findings)


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------

_SEVERITY_ML = {
    "data_leakage": Severity.CRITICAL,
    "missing_random_seed": Severity.CRITICAL,
    "reproducibility": Severity.CRITICAL,
}

_SEVERITY_STR = {
    "critical": Severity.CRITICAL,
    "high": Severity.HIGH,
    "medium": Severity.MEDIUM,
    "low": Severity.LOW,
    "info": Severity.INFO,
    "": Severity.MEDIUM,
}


def compute_health_score(findings: list[Finding]) -> int:
    """0–100 score: 100 is clean, deductions scale with severity."""
    score = 100
    for finding in findings:
        score -= _HEALTH_WEIGHTS.get(finding.severity.value, 0)
    return max(0, min(100, score))


def count_by_severity(findings: list[Finding]) -> dict[str, int]:
    counts = {s: 0 for s in SEVERITY_ORDER}
    for finding in findings:
        counts[finding.severity.value] = counts.get(finding.severity.value, 0) + 1
    return counts


def health_label(score: int) -> str:
    if score >= 80:
        return "healthy"
    if score >= 50:
        return "needs attention"
    return "at risk"


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

    def _norm_ml_smell(
        self, smell: Any, file_abs: str, prefix: str, out: list[Finding]
    ):
        if not isinstance(smell, dict):
            return
        name = smell.get("name", "Unknown")
        fp = smell.get("file_path") or file_abs
        line = int(smell.get("line_number", 0) or 0)
        sev_key = name.lower().replace(" ", "_")
        sev = _SEVERITY_ML.get(sev_key, Severity.HIGH)
        out.append(
            Finding(
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
            )
        )

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
                    if container_key in smells and isinstance(
                        smells[container_key], list
                    ):
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
        out.append(
            Finding(
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
            )
        )

    # ---- TD classifier ----------------------------------------------------

    def normalize_td_predictions(self, raw: dict) -> list[dict]:
        out = []
        for pred in raw.get("predictions", []):
            if not isinstance(pred, dict):
                continue
            cls_idx = pred.get("predicted_class")
            label = td_class_label(pred)
            prob = pred.get("predicted_probability", 0.0)
            if label == "No Debt":
                continue
            out.append(
                {
                    "text": pred.get("text", ""),
                    "category": label,
                    "confidence": round(float(prob), 3),
                    "class_index": cls_idx,
                    "error": pred.get("error"),
                }
            )
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
        score = data.health_score
        label = health_label(score)
        w(f"# Code Review Report: `{target_display}`")
        w("")
        w(f"**Health score:** {score}/100 ({label})  ")
        w(f"**Analyzed:** {data.analyzed_at}  ")
        w(f"**Provider:** {data.provider} — `{data.model}`  ")
        w(f"**Files analyzed:** {data.files_analyzed}  ")
        w(f"**Tools run:** {', '.join(data.tools_run) or '—'}  ")
        if data.duration_s is not None:
            w(f"**Duration:** {data.duration_s:.1f}s  ")
        if data.config_source:
            w(f"**Config:** `{data.config_source}`  ")
        w("")
        w("---")
        w("")

        if data.narrative:
            w("## AI Synthesis")
            w("")
            w(data.narrative.strip())
            w("")
            w("---")
            w("")

        by_sev = count_by_severity(data.findings)

        w("## Summary")
        w("")
        w("| Severity | Count |")
        w("|----------|-------|")
        for sev in SEVERITY_ORDER:
            emoji = SEVERITY_EMOJI[sev]
            w(f"| {emoji} **{sev.upper()}** | {by_sev[sev]} |")
        w(f"| **TOTAL** | **{len(data.findings)}** |")
        w("")

        if not data.findings:
            w("No findings at or above the configured severity. Detectors still ran ")
            w("on the listed tools; residual risk is undocumented behaviour and ")
            w("unscanned non-Python files.")
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
        if data.findings:
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
                    w(
                        f"| {f.finding_id} | {SEVERITY_EMOJI[f.severity.value]} | {loc_str} | {f.category} | {sym} | {msg} |"
                    )
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
                    text_short = p["text"][:80].replace("|", "\\|") + (
                        "…" if len(p["text"]) > 80 else ""
                    )
                    w(f"| {text_short} | {p['category']} | {p['confidence']:.0%} |")
                w("")

        # --- Code Intelligence ---
        if data.intel_summary:
            w("## Code Intelligence")
            w("")
            s = data.intel_summary
            w(f"**Files:** {s.get('files_analyzed', '?')}  ")
            w(
                f"**Symbols:** {s.get('total_symbols', '?')} "
                f"({s.get('total_functions', '?')} functions, "
                f"{s.get('total_classes', '?')} classes)  "
            )
            if s.get("parse_errors"):
                w(
                    f"**Parse errors:** {len(s['parse_errors'])} files could not be parsed  "
                )
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
                    w(
                        f"| `{parent}{h['name']}` | `{loc_str}` | {h['line']} | **{h['cyclomatic_complexity']}** | {h['loc']} | {h['param_count']} | {h['nesting_depth']} |"
                    )
                w("")

        w("---")
        try:
            from code_review_agent import __version__
        except Exception:
            __version__ = "0.2.0"
        w(f"*Generated by code-review-agent v{__version__}*")
        return "\n".join(lines)

    def _render_finding(self, f: Finding, lines: list[str]):
        loc = f"{f.file}:{f.line}" + (f":{f.col}" if f.col else "")
        symbol_str = f" in `{f.symbol}`" if f.symbol else ""
        lines.append(f"### [{f.finding_id}] {f.category}{symbol_str}")
        lines.append("")
        lines.append(f"**Location:** `{loc}`  ")
        if f.framework:
            lines.append(f"**Framework:** {f.framework}  ")
        lines.append(
            f"**Severity:** {SEVERITY_EMOJI[f.severity.value]} {f.severity.value.upper()}  "
        )
        lines.append("")
        if f.message:
            lines.append(f"{f.message}")
            lines.append("")
        if self.include_code_snippets and f.code_snippet:
            snippet = "\n".join(f.code_snippet.splitlines()[: self.max_snippet_lines])
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
                "health_score": data.health_score,
                "severity_counts": count_by_severity(data.findings),
                "findings": [f.__dict__ for f in data.findings],
                "td_predictions": data.td_predictions,
                "intel_summary": data.intel_summary,
                "narrative": data.narrative,
            },
            default=_enc,
            indent=2,
        )

    def render_html(self, data: ReportData) -> str:
        from code_review_agent.dashboard import render_html_report

        return render_html_report(data, include_snippets=self.include_code_snippets)

    def render_sarif(self, data: ReportData) -> str:
        """SARIF 2.1.0 document for GitHub Code Scanning / other CI consumers."""
        rules: dict[str, dict[str, Any]] = {}
        results: list[dict[str, Any]] = []
        for finding in data.findings:
            rule_id = f"{finding.tool}/{finding.category}".replace(" ", "_")
            if rule_id not in rules:
                rules[rule_id] = {
                    "id": rule_id,
                    "name": finding.category,
                    "shortDescription": {"text": finding.category},
                    "fullDescription": {"text": finding.message or finding.category},
                    "defaultConfiguration": {
                        "level": _SARIF_LEVEL.get(finding.severity.value, "warning"),
                    },
                }
            region: dict[str, Any] = {"startLine": max(int(finding.line or 0), 1)}
            if finding.col:
                region["startColumn"] = int(finding.col)
            if finding.end_line:
                region["endLine"] = int(finding.end_line)
            results.append(
                {
                    "ruleId": rule_id,
                    "level": _SARIF_LEVEL.get(finding.severity.value, "warning"),
                    "message": {"text": finding.message or finding.category},
                    "locations": [
                        {
                            "physicalLocation": {
                                "artifactLocation": {
                                    "uri": finding.file.replace("\\", "/")
                                },
                                "region": region,
                            }
                        }
                    ],
                }
            )
        try:
            from code_review_agent import __version__
        except Exception:
            __version__ = "0.2.0"
        doc = {
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "quality-triage",
                            "version": __version__,
                            "informationUri": "https://github.com/KarthikShivasankar/quality-triage",
                            "rules": list(rules.values()),
                        }
                    },
                    "results": results,
                }
            ],
        }
        return json.dumps(doc, indent=2)


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
    all_findings.sort(
        key=lambda f: (sev_order.get(f.severity.value, 99), f.file, f.line)
    )

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
        narrative=None,
    )


def filter_by_min_severity(findings: list[Finding], min_severity: str) -> list[Finding]:
    """Keep findings at or above min_severity."""
    if min_severity not in SEVERITY_ORDER:
        return findings
    cutoff = SEVERITY_ORDER.index(min_severity)
    return [f for f in findings if SEVERITY_ORDER.index(f.severity.value) <= cutoff]


def findings_meet_threshold(findings: list[Finding], fail_on: str | None) -> bool:
    """True when any finding is at or above the fail-on severity."""
    if not fail_on or fail_on == "none":
        return False
    if fail_on not in SEVERITY_ORDER:
        return False
    cutoff = SEVERITY_ORDER.index(fail_on)
    return any(SEVERITY_ORDER.index(f.severity.value) <= cutoff for f in findings)


def save_report(
    data: ReportData,
    output_dir: str = "./reports",
    fmt: str = "markdown",
    include_code_snippets: bool = True,
    max_snippet_lines: int = 10,
) -> list[str]:
    """
    Save report to output_dir. Returns list of written file paths.
    fmt: "markdown" | "json" | "sarif" | "html" | "both" | "archive"
    archive writes markdown + json + html for the UI results library.
    """
    renderer = ReportRenderer(
        include_code_snippets=include_code_snippets,
        max_snippet_lines=max_snippet_lines,
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    target_slug = Path(data.target).name.replace(" ", "_")[:30]
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(output_dir) / f"review-{target_slug}-{ts}"

    written: list[str] = []
    if fmt in ("markdown", "both", "archive"):
        md_path = str(base) + ".md"
        Path(md_path).write_text(renderer.render_markdown(data), encoding="utf-8")
        written.append(md_path)
    if fmt in ("json", "both", "archive"):
        json_path = str(base) + ".json"
        Path(json_path).write_text(renderer.render_json(data), encoding="utf-8")
        written.append(json_path)
    if fmt == "sarif":
        sarif_path = str(base) + ".sarif"
        Path(sarif_path).write_text(renderer.render_sarif(data), encoding="utf-8")
        written.append(sarif_path)
    if fmt in ("html", "archive"):
        html_path = str(base) + ".html"
        Path(html_path).write_text(renderer.render_html(data), encoding="utf-8")
        written.append(html_path)
    return written


@dataclass
class StoredReport:
    path: str
    label: str
    target: str = ""
    health_score: int | None = None
    finding_count: int | None = None
    saved_at: str = ""


def list_stored_reports(output_dir: str, limit: int = 40) -> list[StoredReport]:
    """Newest-first markdown reports in output_dir (``review-*.md``)."""
    root = Path(output_dir)
    if not root.is_dir():
        return []
    files = sorted(
        root.glob("review-*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    out: list[StoredReport] = []
    for md in files[:limit]:
        payload: dict[str, Any] = {}
        json_path = md.with_suffix(".json")
        if json_path.is_file():
            try:
                loaded = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload = loaded
            except (OSError, json.JSONDecodeError):
                payload = {}
        target = str(payload.get("target") or "")
        health = payload.get("health_score")
        try:
            health_i = int(health) if health is not None else None
        except (TypeError, ValueError):
            health_i = None
        findings = payload.get("findings")
        n_findings = len(findings) if isinstance(findings, list) else None
        mtime = datetime.fromtimestamp(md.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        bits = [md.name, mtime]
        if health_i is not None:
            bits.append(f"health {health_i}")
        if n_findings is not None:
            bits.append(f"{n_findings} findings")
        if target:
            bits.append(Path(target).name)
        out.append(
            StoredReport(
                path=str(md.resolve()),
                label=" · ".join(bits),
                target=target,
                health_score=health_i,
                finding_count=n_findings,
                saved_at=mtime,
            )
        )
    return out


def stored_report_choices(output_dir: str) -> list[tuple[str, str]]:
    """Gradio dropdown choices: (label, markdown path)."""
    return [(item.label, item.path) for item in list_stored_reports(output_dir)]


def load_stored_markdown(path: str) -> str:
    """Read a saved markdown report (or sibling .md next to a .json/.html file)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".md":
        return p.read_text(encoding="utf-8")
    md = p.with_suffix(".md")
    if md.is_file():
        return md.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return f"```json\n{p.read_text(encoding='utf-8')}\n```"
    return p.read_text(encoding="utf-8")


def stored_report_payload(path: str) -> dict[str, Any]:
    json_path = Path(path).with_suffix(".json")
    if not json_path.is_file():
        return {}
    try:
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def target_from_stored(path: str) -> str:
    return str(stored_report_payload(path).get("target") or "")
