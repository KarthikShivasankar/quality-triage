"""
Analysis selection layer.

Lets the user control WHICH detector families run (and which technical-debt
categories are used) for a review, and translates that selection into:

  * the subset of tool schemas the agent is allowed to see,
  * the set of tool names the agent is allowed to actually call,
  * the ``analysis_type`` for the python-smell detector,
  * a human-readable note injected into the review prompt so the LLM knows
    exactly what to run and what to mark as skipped.

This module is pure/deterministic and has no heavy imports, so it is cheap to
unit-test offline.
"""

from __future__ import annotations

from code_review_agent.tools import TD_CATEGORY_MODELS

# Detector families the user can select. ``code``/``architectural``/``structural``
# all map onto the single python-smell detector via its ``analysis_type``.
ALL_FAMILIES: list[str] = ["ml", "code", "architectural", "structural", "td", "code-intel"]

# Navigation/inspection tools are always available regardless of selection —
# the agent needs them to map the project and read files.
_NAV_TOOLS: set[str] = {"list_python_files", "read_file"}

_FAMILY_TOOLS: dict[str, set[str]] = {
    "ml": {"detect_ml_smells"},
    "code": {"detect_python_smells"},
    "architectural": {"detect_python_smells"},
    "structural": {"detect_python_smells"},
    "td": {
        "classify_technical_debt",
        "classify_technical_debt_all",
        "classify_technical_debt_ensemble",
        "list_td_categories",
    },
    "code-intel": {"analyze_code_intelligence"},
}

_FAMILY_LABELS: dict[str, str] = {
    "ml": "ML-specific smells (pandas/numpy/sklearn/torch/TF/HuggingFace)",
    "code": "Python code smells (long methods, large classes, duplication, …)",
    "architectural": "Architectural smells (cyclic deps, god objects, hubs)",
    "structural": "Structural smells (complexity, cohesion, coupling)",
    "td": "Technical-debt classification of comments/docstrings",
    "code-intel": "AST code intelligence (symbols, metrics, import graph)",
}


def resolve_families(values: list[str] | tuple[str, ...] | None) -> list[str]:
    """Validate + normalise a list of family names. Empty/None → all families.

    Unknown names are dropped (defensive); order follows ``ALL_FAMILIES``.
    """
    if not values:
        return list(ALL_FAMILIES)
    wanted = {v.strip().lower() for v in values if v and v.strip()}
    resolved = [f for f in ALL_FAMILIES if f in wanted]
    return resolved or list(ALL_FAMILIES)


def skipped_families(families: list[str]) -> list[str]:
    """Families NOT selected (for 'skipped' reporting)."""
    selected = set(families)
    return [f for f in ALL_FAMILIES if f not in selected]


def resolve_td_categories(values: list[str] | tuple[str, ...] | None) -> tuple[list[str], list[str]]:
    """Return (valid_categories, unknown_categories). Empty/None → ['general']."""
    if not values:
        return ["general"], []
    valid: list[str] = []
    unknown: list[str] = []
    for v in values:
        key = (v or "").strip().lower()
        if not key:
            continue
        if key in TD_CATEGORY_MODELS:
            if key not in valid:
                valid.append(key)
        else:
            unknown.append(v)
    return (valid or ["general"]), unknown


def allowed_tool_names(families: list[str]) -> set[str]:
    """All tool names the agent may call for this family selection."""
    allowed = set(_NAV_TOOLS)
    for fam in families:
        allowed |= _FAMILY_TOOLS.get(fam, set())
    return allowed


def filter_openai_tools(defs: list[dict], families: list[str]) -> list[dict]:
    """Return the subset of OpenAI tool schemas permitted by the selection."""
    allowed = allowed_tool_names(families)
    return [d for d in defs if d.get("function", {}).get("name") in allowed]


def python_analysis_type(families: list[str]) -> str | None:
    """Translate the code/architectural/structural selection into an analysis_type.

    Returns ``None`` if the python-smell detector should not run at all.
    """
    py = [f for f in ("code", "architectural", "structural") if f in families]
    if not py:
        return None
    if len(py) == 3:
        return "all"
    if len(py) == 1:
        return py[0]
    # A 2-of-3 subset: there is no single enum for it, so fall back to "all"
    # and let the prompt note tell the agent which categories to keep.
    return "all"


def selection_note(
    families: list[str],
    td_categories: list[str],
    include_fix_instructions: bool = False,
) -> str:
    """Build the prompt context describing the user's analysis selection."""
    selected = resolve_families(families)
    skipped = skipped_families(selected)

    lines: list[str] = []
    lines.append("## Analysis Selection (user-controlled)")
    lines.append("")
    lines.append("Run ONLY the following analysis families and base your report on them:")
    for fam in selected:
        lines.append(f"- **{fam}** — {_FAMILY_LABELS.get(fam, fam)}")

    if "td" in selected:
        cats = ", ".join(td_categories) or "general"
        lines.append("")
        lines.append(
            f"For technical-debt classification, use these categories only: {cats}. "
            "Pass the matching `category` to `classify_technical_debt` (or use "
            "`classify_technical_debt_all` restricted to these categories)."
        )

    pa = python_analysis_type(selected)
    if pa and pa != "all":
        lines.append("")
        lines.append(
            f"When calling `detect_python_smells`, use analysis_type=\"{pa}\"."
        )
    elif {"code", "architectural", "structural"} & set(selected) and pa == "all" and \
            len([f for f in ("code", "architectural", "structural") if f in selected]) < 3:
        keep = [f for f in ("code", "architectural", "structural") if f in selected]
        lines.append("")
        lines.append(
            "Only report these python-smell categories: " + ", ".join(keep) +
            " (ignore the other python-smell categories even if the detector returns them)."
        )

    if skipped:
        lines.append("")
        lines.append(
            "The following families were NOT selected — do NOT run their tools, and "
            "in the report mark their sections as **Skipped (not selected)** rather "
            "than reporting zero findings: " + ", ".join(skipped) + "."
        )

    if include_fix_instructions:
        from code_review_agent.fixes import FIX_BLOCK_INSTRUCTIONS
        lines.append("")
        lines.append(FIX_BLOCK_INSTRUCTIONS)

    return "\n".join(lines)
