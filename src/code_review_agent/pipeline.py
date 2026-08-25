"""
Deterministic analysis pipeline.

Runs detectors in a fixed order, normalises findings, optionally asks
LiteLLM to write a narrative on top of the structured report.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from code_review_agent.config import AppConfig, get_config
from code_review_agent.llm import LLMClient
from code_review_agent.prompts import SYNTHESIS_PROMPT
from code_review_agent.reporter import (
    ReportData,
    build_report,
    compute_health_score,
    count_by_severity,
    filter_by_min_severity,
)
from code_review_agent.tools import (
    _python_files,
    analyze_code_intelligence,
    classify_technical_debt,
    detect_ml_smells,
    detect_python_smells,
    list_python_files,
)

_TODO_RE = re.compile(
    r"\b(TODO|FIXME|HACK|XXX|NOTE)\b[:\s\-].*",
    re.IGNORECASE,
)


def extract_debt_comments(
    path: str,
    ignore: set[str],
    limit: int = 50,
) -> list[str]:
    """Pull TODO/FIXME/HACK comments from Python sources."""
    texts: list[str] = []
    target = Path(path)
    files = _python_files(target, ignore)
    for py_file in files:
        try:
            lines = py_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if _TODO_RE.search(stripped):
                texts.append(stripped[:200])
                if len(texts) >= limit:
                    return texts
    return texts


def _compact_report_payload(data: ReportData, max_findings: int = 80) -> dict[str, Any]:
    findings = []
    for f in data.findings[:max_findings]:
        findings.append(
            {
                "id": f.finding_id,
                "tool": f.tool,
                "category": f.category,
                "severity": f.severity.value,
                "file": f.file,
                "line": f.line,
                "col": f.col,
                "symbol": f.symbol,
                "message": f.message[:400],
                "how_to_fix": (f.how_to_fix or "")[:400] or None,
            }
        )
    return {
        "target": data.target,
        "files_analyzed": data.files_analyzed,
        "tools_run": data.tools_run,
        "finding_count": len(data.findings),
        "health_score": compute_health_score(data.findings),
        "severity_counts": count_by_severity(data.findings),
        "findings": findings,
        "td_predictions": data.td_predictions[:40],
        "intel_summary": {
            k: v for k, v in (data.intel_summary or {}).items() if k != "large_files"
        },
    }


@dataclass
class PipelineResult:
    report: ReportData
    ml_raw: dict[str, Any]
    py_raw: dict[str, Any]
    td_raw: dict[str, Any]
    intel_raw: dict[str, Any]
    files_raw: dict[str, Any]
    synthesis_error: str | None = None


def run_pipeline(
    path: str,
    *,
    cfg: AppConfig | None = None,
    parallel: bool = True,
    on_step: Callable[[str], None] | None = None,
    issue_texts: list[str] | None = None,
) -> PipelineResult:
    """Run the fixed analysis sequence against a local path."""
    cfg = cfg or get_config()
    target = str(Path(path).resolve())
    ignore = set(cfg.tools.ignore_dirs)

    def note(msg: str) -> None:
        if on_step:
            on_step(msg)

    note("Listing Python files…")
    files_raw = list_python_files(target)

    def _intel():
        return analyze_code_intelligence(target, import_graph=True)

    def _py():
        return detect_python_smells(target, analysis_type="all")

    def _ml():
        return detect_ml_smells(target)

    note("Code intelligence, Python smells, ML smells…")
    if parallel:
        with ThreadPoolExecutor(max_workers=3) as pool:
            fut_intel = pool.submit(_intel)
            fut_py = pool.submit(_py)
            fut_ml = pool.submit(_ml)
            intel_raw = fut_intel.result()
            py_raw = fut_py.result()
            ml_raw = fut_ml.result()
    else:
        intel_raw = _intel()
        py_raw = _py()
        ml_raw = _ml()

    note("Extracting TODO/FIXME comments…")
    td_texts = extract_debt_comments(target, ignore)
    extra = [t.strip() for t in (issue_texts or []) if isinstance(t, str) and t.strip()]
    if extra:
        note(f"Including {len(extra)} GitHub issue(s) in TD classification…")
        td_texts = td_texts + extra
    td_raw: dict[str, Any] = {}
    if td_texts:
        note("Classifying technical debt…")
        td_raw = classify_technical_debt(td_texts)

    files_analyzed = int(files_raw.get("total_files") or 0)
    if not files_analyzed and isinstance(intel_raw.get("summary"), dict):
        files_analyzed = int(intel_raw["summary"].get("files_analyzed") or 0)

    tools_run = [
        "list_python_files",
        "analyze_code_intelligence",
        "detect_python_smells",
        "detect_ml_smells",
    ]
    if td_texts:
        tools_run.append("classify_technical_debt")

    intel_summary = intel_raw.get("summary") if isinstance(intel_raw, dict) else None
    if not cfg.code_intel.metrics_enabled and isinstance(intel_summary, dict):
        intel_summary = dict(intel_summary)
        intel_summary.pop("complexity_hotspots", None)

    report = build_report(
        target=target,
        provider=cfg.provider,
        model=cfg.llm.model,
        ml_raw=ml_raw if "error" not in ml_raw else None,
        py_raw=py_raw if "error" not in py_raw else None,
        td_raw=td_raw if td_raw and "error" not in td_raw else None,
        intel_summary=intel_summary,
        files_analyzed=files_analyzed,
        tools_run=tools_run,
        config_source=cfg._source,
    )
    return PipelineResult(
        report=report,
        ml_raw=ml_raw,
        py_raw=py_raw,
        td_raw=td_raw,
        intel_raw=intel_raw,
        files_raw=files_raw,
    )


def synthesize_report(
    report: ReportData,
    client: LLMClient,
    extra_context: str = "",
) -> str:
    """Ask LiteLLM for a narrative over already-normalised findings."""
    payload = _compact_report_payload(report)
    user = (
        "Synthesise the following structured code-review findings into the "
        "required report sections. Do not invent files or line numbers that "
        "are not in the JSON.\n\n"
        f"```json\n{json.dumps(payload, default=str)}\n```"
    )
    if extra_context:
        user += f"\n\nAdditional reviewer context: {extra_context}"
    result = client.complete_text(
        [
            {"role": "system", "content": SYNTHESIS_PROMPT},
            {"role": "user", "content": user},
        ]
    )
    return result.content


def stream_synthesis(
    report: ReportData,
    client: LLMClient,
    extra_context: str = "",
) -> Iterator[str]:
    payload = _compact_report_payload(report)
    user = (
        "Synthesise the following structured code-review findings into the "
        "required report sections. Do not invent files or line numbers that "
        "are not in the JSON.\n\n"
        f"```json\n{json.dumps(payload, default=str)}\n```"
    )
    if extra_context:
        user += f"\n\nAdditional reviewer context: {extra_context}"
    yield from client.stream_text(
        [
            {"role": "system", "content": SYNTHESIS_PROMPT},
            {"role": "user", "content": user},
        ]
    )


def stamp_duration(report: ReportData, started: datetime) -> ReportData:
    report.duration_s = (datetime.now(timezone.utc) - started).total_seconds()
    return report


def execute_hybrid_review(
    path: str,
    cfg: AppConfig,
    *,
    model: str | None = None,
    provider: str | None = None,
    api_base: str | None = None,
    fallbacks: list[str] | None = None,
    no_llm: bool = False,
    extra_context: str = "",
    issue_texts: list[str] | None = None,
    parallel: bool = True,
    on_step: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Run detectors, optionally synthesise a narrative, stamp duration."""
    from code_review_agent.llm import LLMClient, resolve_llm_model

    def note(msg: str) -> None:
        if on_step:
            on_step(msg)

    started = datetime.now(timezone.utc)
    result = run_pipeline(
        path,
        cfg=cfg,
        parallel=parallel,
        on_step=on_step,
        issue_texts=issue_texts,
    )
    result.report.model = resolve_llm_model(model=model, provider=provider, cfg=cfg)
    result.report.provider = result.report.model.split("/", 1)[0]
    result.report.findings = filter_by_min_severity(
        result.report.findings,
        cfg.report.min_severity,
    )

    if not no_llm:
        note(f"Synthesising with {result.report.model}…")
        client = LLMClient(
            cfg.llm,
            model=result.report.model,
            api_base=api_base,
            fallbacks=fallbacks,
        )
        try:
            result.report.narrative = synthesize_report(
                result.report,
                client,
                extra_context=extra_context,
            )
        except Exception as exc:
            result.synthesis_error = str(exc)
            result.report.narrative = None

    stamp_duration(result.report, started)
    note("Done.")
    return result
