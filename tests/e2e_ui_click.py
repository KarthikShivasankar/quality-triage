#!/usr/bin/env python3
"""Manual E2E UI driver for pallets/click via Gradio Client API."""

from __future__ import annotations

import json
import sys
import time

CLICK = "https://github.com/pallets/click"
CLICK_SRC = "https://github.com/pallets/click/tree/main/src/click"
ALL_TOOLS = [
    "list-files",
    "code-intel",
    "python-smells",
    "ml-smells",
    "classify-td",
]
SUBSET_TOOLS = ["list-files", "classify-td"]


def main() -> int:
    from gradio_client import Client

    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7866"
    print(f"Connecting to {base}…")
    client = Client(base)

    checks: list[tuple[str, bool, str]] = []

    def ok(name: str, cond: bool, detail: str = "") -> None:
        checks.append((name, cond, detail))
        mark = "PASS" if cond else "FAIL"
        print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))

    # --- Tools: list-files on click src ---
    print("\n=== Tools: list-files ===")
    t0 = time.time()
    tool_out = client.predict(
        "list-files",
        CLICK_SRC,
        None,
        "",
        api_name="/run_tool",
    )
    tool_payload = json.loads(tool_out)
    ok(
        "tools/list-files",
        tool_payload.get("total_files", 0) >= 5,
        f"{tool_payload.get('total_files')} files in {time.time() - t0:.1f}s",
    )

    # --- Tools: td-from-comments (TODOs + GitHub issues on clone) ---
    print("\n=== Tools: td-from-comments ===")
    t0 = time.time()
    td_out = client.predict(
        "td-from-comments",
        CLICK,
        None,
        "",
        api_name="/run_tool",
    )
    td_payload = json.loads(td_out)
    texts = [p.get("text", "") for p in td_payload.get("predictions", [])]
    issue_hits = [t for t in texts if t.startswith("#")]
    ok(
        "tools/td-from-comments issues",
        len(issue_hits) >= 1,
        f"{len(issue_hits)} issue snippets, {len(texts)} total in {time.time() - t0:.1f}s",
    )
    if issue_hits:
        print(f"  sample: {issue_hits[0][:100]}…")

    # --- Tools: classify-td offline snippets ---
    print("\n=== Tools: classify-td ===")
    td_off = client.predict(
        "classify-td",
        "",
        None,
        "TODO: refactor parser\nFIXME: edge case in shell completion",
        api_name="/run_tool",
    )
    off_payload = json.loads(td_off)
    ok(
        "tools/classify-td",
        len(off_payload.get("predictions", [])) >= 2,
        f"{len(off_payload.get('predictions', []))} predictions",
    )

    # --- Review: pipeline-only on src/click (faster than full repo) ---
    print("\n=== Review: pipeline-only (src/click) ===")
    t0 = time.time()
    review_out = client.predict(
        CLICK,
        None,
        "local",
        True,
        "E2E UI test",
        ALL_TOOLS,
        api_name="/_run_review_ui_1",
    )
    status, report_md, _rows, json_text, saved = review_out[:5]
    ok("review/status ok", not str(status).startswith("Failed:"), str(status)[:120])
    ok("review/markdown", "# Code Review Report" in str(report_md), "")
    payload = json.loads(json_text) if json_text else {}
    ok(
        "review/json health",
        "health_score" in payload,
        f"score={payload.get('health_score')}",
    )
    td = payload.get("td_predictions") or []
    issues = [x for x in td if str(x.get("text", "")).startswith("#")]
    ok(
        "review/github issues in TD",
        len(issues) >= 1,
        f"{len(issues)} issue snippets in {time.time() - t0:.1f}s",
    )
    ok(
        "review/saved archive",
        ".md" in str(saved) and ".json" in str(saved),
        str(saved)[:100],
    )

    # --- Results: open saved report ---
    print("\n=== Results: view saved report ===")
    md_path = next((p.strip() for p in str(saved).split(", ") if p.endswith(".md")), "")
    if md_path:
        open_out = client.predict(md_path, api_name="/_open_saved_ui")
        archive_status, archive_md = open_out[0], open_out[1]
        ok(
            "results/view report",
            "Opened" in str(archive_status)
            and "# Code Review Report" in str(archive_md),
            str(archive_status)[:80],
        )
        rerun_path = client.predict(md_path, api_name="/rerun_target_from_report")
        ok(
            "results/re-run target",
            "click" in str(rerun_path).lower() or "github" in str(rerun_path).lower(),
            str(rerun_path)[:80],
        )
    else:
        ok("results/view report", False, "no md path from review")
        ok("results/re-run target", False, "skipped")

    # --- Review: subset detectors ---
    print("\n=== Review: subset detectors ===")
    subset_out = client.predict(
        CLICK,
        None,
        "local",
        True,
        "",
        SUBSET_TOOLS,
        api_name="/_run_review_ui_1",
    )
    subset_status, _, _, subset_json, _ = subset_out[:5]
    subset_payload = json.loads(subset_json) if subset_json else {}
    tools_run = subset_payload.get("tools_run") or []
    ok(
        "review/subset tools",
        not str(subset_status).startswith("Failed:")
        and "list_python_files" in tools_run
        and "classify_technical_debt" in tools_run
        and "detect_python_smells" not in tools_run,
        str(tools_run),
    )

    # --- Ask tab (completion-only model; smoke only) ---
    print("\n=== Ask: smoke ===")
    ask_out = client.predict(
        "Reply with the single word: pong",
        "local",
        api_name="/run_ask",
    )
    ok("ask/non-empty", bool(str(ask_out).strip()), str(ask_out)[:60])

    print("\n=== Summary ===")
    failed = [c for c in checks if not c[1]]
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        for name, _, detail in failed:
            print(f"  FAILED: {name} — {detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
