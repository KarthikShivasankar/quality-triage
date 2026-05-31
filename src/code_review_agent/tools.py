"""
Tool implementations wrapping the three analyser packages + AST code intelligence.
All outputs use the canonical findings shape with exact file:line:col.
"""

from __future__ import annotations

import contextlib
import json
import os
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rel(abs_path: str, root: str) -> str:
    try:
        return os.path.relpath(abs_path, root)
    except ValueError:
        return abs_path


def _enrich_column(file_path: str, line: int, snippet: str) -> int | None:
    """Return 1-based column of the first line of snippet in the file, or None."""
    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        if not (1 <= line <= len(lines)):
            return None
        src_line = lines[line - 1]
        needle = snippet.split("\n")[0].strip()
        idx = src_line.find(needle)
        return idx + 1 if idx >= 0 else None
    except Exception:
        return None


def _python_files(path: Path, ignore: set[str]) -> list[Path]:
    if path.is_file() and path.suffix == ".py":
        return [path]
    return [f for f in path.rglob("*.py") if not any(part in ignore for part in f.parts)]


def _get_cfg():
    """Lazy import of config to avoid circular imports."""
    from code_review_agent.config import get_config

    return get_config()


def _smell_to_dict(obj: Any) -> dict:
    """Convert a smell dataclass/object to a plain dict."""
    if isinstance(obj, dict):
        return obj
    try:
        return dict(obj.__dict__)
    except Exception:
        return {"raw": str(obj)}


# ---------------------------------------------------------------------------
# Threshold resolution (adaptive — always provides a complete set of keys)
# ---------------------------------------------------------------------------

_PKG_THRESHOLD_CACHE: dict[str, dict] | None = None


def _package_default_thresholds() -> dict[str, dict]:
    """
    Load the *complete* default threshold set bundled with code_quality_analyzer.

    The detectors raise ``CodeAnalysisError`` if a threshold key they need is
    missing, so we always start from the package's own defaults and let the
    user's config.yaml override individual values. This keeps us working even
    if a newer commit of the analyzer adds new threshold keys.
    """
    global _PKG_THRESHOLD_CACHE
    if _PKG_THRESHOLD_CACHE is not None:
        return _PKG_THRESHOLD_CACHE

    result: dict[str, dict] = {
        "code_smells": {},
        "architectural_smells": {},
        "structural_smells": {},
    }
    try:
        import code_quality_analyzer
        import yaml as _yaml

        pkg_dir = Path(code_quality_analyzer.__file__).parent
        for cfg_name in ("code_quality_config.yaml", "config.yaml"):
            cfg_file = pkg_dir / cfg_name
            if cfg_file.exists():
                raw = _yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
                for smell_type in result:
                    section = raw.get(smell_type, {}) or {}
                    for k, v in section.items():
                        result[smell_type][k] = v.get("value") if isinstance(v, dict) else v
                break
    except Exception:
        pass

    _PKG_THRESHOLD_CACHE = result
    return result


# Threshold keys that some shipped code_quality_analyzer detectors REQUIRE but
# the package's bundled config omits (so the detectors KeyError). We supply
# sensible defaults so the methods run; users can still override in config.yaml.
_AGENT_EXTRA_THRESHOLDS: dict[str, dict[str, Any]] = {
    "code_smells": {
        "LAZY_CLASS_LINES": 15,
        "DATA_CLASS_METHODS": 5,
    },
}


def _resolve_thresholds(cfg, smell_type: str) -> dict:
    """Merge package default thresholds with the user's (flattened) overrides.

    Agent-level fallbacks (``_AGENT_EXTRA_THRESHOLDS``) fill in keys the upstream
    package config omits, so detectors that reference them never KeyError.
    """
    from code_review_agent.config import get_thresholds_flat

    merged = dict(_package_default_thresholds().get(smell_type, {}))
    for key, value in _AGENT_EXTRA_THRESHOLDS.get(smell_type, {}).items():
        merged.setdefault(key, value)
    merged.update(get_thresholds_flat(cfg, smell_type))
    return merged


# ---------------------------------------------------------------------------
# ML smell normalisation (keys differ between the three detectors)
# ---------------------------------------------------------------------------


def _parse_line_number(value: Any) -> int:
    """Extract an int line number from an int or a string like 'Line 5'."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        import re as _re

        m = _re.search(r"\d+", value)
        if m:
            return int(m.group())
    return 0


def _normalize_ml_smell(smell: Any) -> dict:
    """
    Normalise a raw ML smell dict into a canonical shape.

    The three detectors disagree on key names:
      FrameworkSpecificSmellDetector: name, framework, how_to_fix, benefits,
                                      strategies, line_number, code_snippet, file_path
      HuggingFaceSmellDetector:       framework, name, fix, benefits, location
      ML_SmellDetector:               smell, line_number, code_snippet, file_path
    """
    if not isinstance(smell, dict):
        return {"name": "Unknown", "description": str(smell)}

    raw_name = smell.get("name") or smell.get("smell") or ""
    # Derive a short canonical name from the first words of a description.
    name = smell.get("name")
    if not name:
        words = str(raw_name).split()
        name = " ".join(words[:8]) if words else "ML Smell"

    description = smell.get("description") or smell.get("smell") or smell.get("name") or ""

    line_number = smell.get("line_number")
    if line_number is None and smell.get("location"):
        line_number = _parse_line_number(smell.get("location"))
    line_number = _parse_line_number(line_number) if line_number is not None else 0

    out = {
        "name": name,
        "description": description,
        "how_to_fix": smell.get("how_to_fix") or smell.get("fix"),
        "benefits": smell.get("benefits"),
        "strategies": smell.get("strategies"),
        "framework": smell.get("framework"),
        "line_number": line_number,
        "code_snippet": smell.get("code_snippet"),
        "file_path": smell.get("file_path"),
    }
    # Drop None values for cleaner output.
    return {k: v for k, v in out.items() if v is not None}


# ---------------------------------------------------------------------------
# Tool 1: detect_ml_smells
# ---------------------------------------------------------------------------


def detect_ml_smells(
    path: str,
    ignore_dirs: list[str] | None = None,
) -> dict[str, Any]:
    """Detect ML-specific code smells using ml_code_smell_detector."""
    try:
        from ml_code_smell_detector import (
            FrameworkSpecificSmellDetector,
            HuggingFaceSmellDetector,
            ML_SmellDetector,
        )
    except ImportError as e:
        return {"error": f"ml_code_smell_detector not available: {e}"}

    target = Path(path).resolve()
    if not target.exists():
        return {"error": f"Path does not exist: {path}"}

    cfg = _get_cfg()
    ignore = set(ignore_dirs or cfg.tools.ignore_dirs)
    py_files = _python_files(target, ignore)
    if not py_files:
        return {"error": "No Python files found", "path": str(target)}

    results: dict[str, Any] = {
        "tool": "ml_smells",
        "target": str(target),
        "framework_smells": [],
        "huggingface_smells": [],
        "general_ml_smells": [],
        "errors": [],
    }

    detectors = [
        (FrameworkSpecificSmellDetector, "framework_smells"),
        (HuggingFaceSmellDetector, "huggingface_smells"),
        (ML_SmellDetector, "general_ml_smells"),
    ]

    for DetectorCls, key in detectors:
        for py_file in py_files:
            file_str = str(py_file)
            # IMPORTANT: a fresh detector PER FILE. The detectors accumulate into
            # self.smells and detect_smells() returns the *accumulated* list, so
            # reusing one instance across files duplicates earlier results.
            try:
                detector = DetectorCls()
                raw = detector.detect_smells(file_str)
                # Prefer get_results() if available (canonical accumulated list).
                if hasattr(detector, "get_results"):
                    with contextlib.suppress(Exception):
                        raw = detector.get_results()

                if isinstance(raw, list):
                    smell_list = [_normalize_ml_smell(s) for s in raw]
                elif isinstance(raw, dict):
                    smell_list = [_normalize_ml_smell(s) for s in raw.values()]
                else:
                    smell_list = []

                # Enrich column info where missing.
                for smell in smell_list:
                    if smell.get("line_number") and not smell.get("col"):
                        snippet = smell.get("code_snippet", "")
                        if snippet:
                            col = _enrich_column(file_str, smell["line_number"], snippet)
                            if col:
                                smell["col"] = col
                    # Make sure file_path is populated for the reporter.
                    smell.setdefault("file_path", file_str)

                if smell_list:
                    results[key].append({"file": file_str, "smells": smell_list})
            except Exception as exc:
                results["errors"].append({"file": file_str, "error": str(exc)})

    total_smells = sum(
        len(e["smells"])
        for key in ("framework_smells", "huggingface_smells", "general_ml_smells")
        for e in results[key]
    )
    results["summary"] = {
        "files_analyzed": len(py_files),
        "total_smells": total_smells,
        "files_with_framework_smells": len(results["framework_smells"]),
        "files_with_hf_smells": len(results["huggingface_smells"]),
        "files_with_general_ml_smells": len(results["general_ml_smells"]),
    }
    return results


# ---------------------------------------------------------------------------
# Tool 2: detect_python_smells
# ---------------------------------------------------------------------------


def detect_python_smells(
    path: str,
    analysis_type: str = "all",
    ignore_dirs: list[str] | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Detect code, architectural, and structural smells using code_quality_analyzer."""
    try:
        from code_quality_analyzer import (
            ArchitecturalSmellDetector,
            CodeSmellDetector,
            StructuralSmellDetector,
        )
    except ImportError as e:
        return {"error": f"code_quality_analyzer not available: {e}"}

    target = Path(path).resolve()
    if not target.exists():
        return {"error": f"Path does not exist: {path}"}

    cfg = _get_cfg()
    ignore = list(ignore_dirs or cfg.tools.ignore_dirs)

    # Always pass a COMPLETE flat threshold set (package defaults + user overrides).
    code_thresh = _resolve_thresholds(cfg, "code_smells")
    arch_thresh = _resolve_thresholds(cfg, "architectural_smells")
    struct_thresh = _resolve_thresholds(cfg, "structural_smells")

    results: dict[str, Any] = {
        "tool": "python_smells",
        "target": str(target),
        "analysis_type": analysis_type,
        "errors": [],
    }

    # ---- Code smells (per-file detect_smells + one cross-file pass) ----
    if analysis_type in ("code", "all"):
        try:
            det = CodeSmellDetector(code_thresh)
            py_files = _python_files(target, set(ignore))
            for py_file in py_files:
                try:
                    det.detect_smells(str(py_file))
                except Exception as exc:
                    results["errors"].append(
                        {"file": str(py_file), "phase": "code_smells", "error": str(exc)}
                    )
                # Three catalog detectors exist upstream but are NOT in the
                # dispatch list, so they never fire. Invoke them explicitly per
                # file (they append to det.code_smells, like the dispatched ones).
                _run_unwired_code_detectors(det, py_file, results)
            with contextlib.suppress(Exception):
                det.detect_cross_file_smells()
            code_smells = _extract_smell_list(det, "code_smells")
            # Supplement: correct Switch-Statements detection (upstream caps at 2
            # branches because it doesn't recurse elif chains).
            try:
                from code_review_agent.cqa_supplement import merge_dedup, supplemental_switch_smells

                switch_smells = supplemental_switch_smells(
                    py_files, code_thresh.get("COMPLEX_CONDITIONAL", 3)
                )
                code_smells = merge_dedup(code_smells, switch_smells)
            except Exception as exc:
                results["errors"].append({"phase": "switch_supplement", "error": str(exc)})
            results["code_smells"] = code_smells
        except Exception as exc:
            results["code_smells"] = {"error": str(exc)}

    # ---- Architectural smells (directory only; detect_smells runs detection) ----
    if analysis_type in ("architectural", "all"):
        if target.is_dir():
            try:
                det = ArchitecturalSmellDetector(arch_thresh)
                det.detect_smells(str(target), ignore_dirs=ignore)
                results["architectural_smells"] = _extract_smell_list(det, "architectural_smells")
            except Exception as exc:
                results["architectural_smells"] = {"error": str(exc)}
        else:
            results["architectural_smells"] = {
                "note": "Architectural smells require a directory target; skipped for single file."
            }

    # ---- Structural smells (detect_smells walks a directory) ----
    if analysis_type in ("structural", "all"):
        try:
            det = StructuralSmellDetector(struct_thresh)
            if target.is_dir():
                det.detect_smells(str(target), ignore_dirs=ignore)
            else:
                # detect_smells needs a directory: copy the single file into a
                # fresh temp dir, analyse it there, then clean up.
                import shutil
                import tempfile

                tmp = tempfile.mkdtemp(prefix="cra_struct_")
                try:
                    shutil.copy2(str(target), os.path.join(tmp, target.name))
                    det.detect_smells(tmp, ignore_dirs=ignore)
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
            structural_smells = _extract_smell_list(det, "structural_smells")
            # Supplement: correct Deep-Inheritance-Tree detection (upstream's
            # graph uses bare base names vs qualified node names, so it never
            # exceeds the threshold and mislabels classes as "Isolated").
            try:
                from code_review_agent.cqa_supplement import merge_dedup, supplemental_dit_smells

                struct_files = _python_files(target, set(ignore))
                dit_smells = supplemental_dit_smells(
                    struct_files, struct_thresh.get("DIT_THRESHOLD", 3)
                )
                structural_smells = merge_dedup(structural_smells, dit_smells)
            except Exception as exc:
                results["errors"].append({"phase": "dit_supplement", "error": str(exc)})
            results["structural_smells"] = structural_smells
        except Exception as exc:
            results["structural_smells"] = {"error": str(exc)}

    return results


# Upstream CodeSmellDetector defines these but omits them from its dispatch
# list, so they never fire through detect_smells(). We call them explicitly.
_UNWIRED_CODE_DETECTORS = ("detect_data_class", "detect_dead_code", "detect_lazy_class")


def _run_unwired_code_detectors(det: Any, py_file: Any, results: dict) -> None:
    """Invoke the three never-dispatched code-smell detectors on one file.

    They take an astroid module + path and append to ``det.code_smells``.
    Defensive: a missing astroid, parse error, or detector error never crashes
    the review — it is recorded under ``results['errors']`` instead.
    """
    try:
        import astroid
    except ImportError:
        return
    try:
        source = Path(str(py_file)).read_text(encoding="utf-8", errors="replace")
        module = astroid.parse(source, path=str(py_file))
    except Exception as exc:
        results["errors"].append(
            {"file": str(py_file), "phase": "astroid_parse", "error": str(exc)}
        )
        return
    for meth_name in _UNWIRED_CODE_DETECTORS:
        meth = getattr(det, meth_name, None)
        if meth is None:
            continue
        try:
            meth(module, str(py_file))
        except Exception as exc:
            results["errors"].append({"file": str(py_file), "phase": meth_name, "error": str(exc)})


def _extract_smell_list(detector: Any, attr: str) -> list[dict]:
    """Extract smell list from a detector object in various formats."""
    # Try common attribute names
    for a in (attr, "smells", "results", "_smells", "detected_smells"):
        val = getattr(detector, a, None)
        if val is not None:
            if isinstance(val, list):
                return [_smell_to_dict(s) for s in val]
            if isinstance(val, dict):
                return list(val.values())
    # Fallback: print_report to string
    try:
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            detector.print_report()
        return [{"report": buf.getvalue()}]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Tool 3: classify_technical_debt
# ---------------------------------------------------------------------------

# Binary, per-category TD models on the HuggingFace Hub (karths/ namespace).
# Each model outputs predicted_class 0/1 (1 = that category of debt present).
#
# Two model families live under the namespace and were verified against the
# live HuggingFace Hub listing:
#   * SATD debt-type models use full-word suffixes (..._code, ..._design, …)
#   * ISO/IEC 25010 quality-attribute models use abbreviated suffixes
#     (..._secu, ..._perf, ..._usab, ..._main, ..._reli, ..._port, ..._comp).
# Friendly aliases (e.g. "security" -> "secu") are provided for both families.
TD_CATEGORY_MODELS: dict[str, str] = {
    # General technical debt
    "general": "karths/binary_classification_train_TD",
    "td": "karths/binary_classification_train_TD",
    "technical_debt": "karths/binary_classification_train_TD",
    # SATD debt-type categories (full-word model ids)
    "code": "karths/binary_classification_train_code",
    "design": "karths/binary_classification_train_design",
    "documentation": "karths/binary_classification_train_documentation",
    "docs": "karths/binary_classification_train_documentation",
    "test": "karths/binary_classification_train_test",
    "defect": "karths/binary_classification_train_defect",
    "requirement": "karths/binary_classification_train_requirement",
    "build": "karths/binary_classification_train_build",
    "automation": "karths/binary_classification_train_automation",
    "test_automation": "karths/binary_classification_train_automation",
    "people": "karths/binary_classification_train_people",
    "process": "karths/binary_classification_train_process",
    "infrastructure": "karths/binary_classification_train_infrastructure",
    "infra": "karths/binary_classification_train_infrastructure",
    "architecture": "karths/binary_classification_train_architecture",
    "arch": "karths/binary_classification_train_architecture",
    "service": "karths/binary_classification_train_service",
    # ISO/IEC 25010 quality-attribute models (abbreviated model ids)
    "security": "karths/binary_classification_train_secu",
    "secu": "karths/binary_classification_train_secu",
    "performance": "karths/binary_classification_train_perf",
    "perf": "karths/binary_classification_train_perf",
    "usability": "karths/binary_classification_train_usab",
    "usab": "karths/binary_classification_train_usab",
    "maintainability": "karths/binary_classification_train_main",
    "main": "karths/binary_classification_train_main",
    "reliability": "karths/binary_classification_train_reli",
    "reli": "karths/binary_classification_train_reli",
    "portability": "karths/binary_classification_train_port",
    "port": "karths/binary_classification_train_port",
    "compatibility": "karths/binary_classification_train_comp",
    "comp": "karths/binary_classification_train_comp",
}

# Canonical (de-duplicated) category keys, in a friendly order. Used for the
# ``--all-categories`` sweep and for listing available categories.
TD_PRIMARY_CATEGORIES: list[str] = [
    "general",
    "code",
    "design",
    "documentation",
    "test",
    "defect",
    "requirement",
    "build",
    "automation",
    "people",
    "process",
    "infrastructure",
    "architecture",
    "service",
    "security",
    "performance",
    "usability",
    "maintainability",
    "reliability",
    "portability",
    "compatibility",
]

# Map a model id back to a human-readable debt label.
TD_MODEL_LABELS: dict[str, str] = {
    "karths/binary_classification_train_TD": "Technical Debt",
    "karths/binary_classification_train_code": "Code Debt",
    "karths/binary_classification_train_design": "Design Debt",
    "karths/binary_classification_train_documentation": "Documentation Debt",
    "karths/binary_classification_train_test": "Test Debt",
    "karths/binary_classification_train_defect": "Defect Debt",
    "karths/binary_classification_train_requirement": "Requirement Debt",
    "karths/binary_classification_train_build": "Build Debt",
    "karths/binary_classification_train_automation": "Test Automation Debt",
    "karths/binary_classification_train_people": "People Debt",
    "karths/binary_classification_train_process": "Process Debt",
    "karths/binary_classification_train_infrastructure": "Infrastructure Debt",
    "karths/binary_classification_train_architecture": "Architecture Debt",
    "karths/binary_classification_train_service": "Service Debt",
    "karths/binary_classification_train_secu": "Security Debt",
    "karths/binary_classification_train_perf": "Performance Debt",
    "karths/binary_classification_train_usab": "Usability Debt",
    "karths/binary_classification_train_main": "Maintainability Debt",
    "karths/binary_classification_train_reli": "Reliability Debt",
    "karths/binary_classification_train_port": "Portability Debt",
    "karths/binary_classification_train_comp": "Compatibility Debt",
}


def td_label_for_model(model_path: str | None) -> str:
    """Human-readable debt label for a model id (default 'Technical Debt')."""
    if not model_path:
        return "Technical Debt"
    return TD_MODEL_LABELS.get(model_path, "Technical Debt")


def _download_onnx_from_hub(model_path: str) -> tuple[str | None, str | None]:
    """
    Download a model's ONNX export + tokenizer from the HuggingFace Hub.

    Uses the plain model API + ``/resolve/main/`` URLs over ``requests`` (the
    huggingface_hub httpx client is unreliable in some networks). Returns
    ``(onnx_file, tokenizer_dir)`` or ``(None, None)`` when the repo has no ONNX
    weights. Files are cached under the system temp dir so repeat runs are fast.
    """
    import tempfile

    try:
        import requests
    except ImportError:
        return None, None

    api = f"https://huggingface.co/api/models/{model_path}"
    resp = requests.get(api, timeout=30)
    if resp.status_code != 200:
        return None, None
    siblings = [s.get("rfilename") for s in resp.json().get("siblings", [])]
    siblings = [s for s in siblings if s]

    onnx_files = [f for f in siblings if f.endswith(".onnx")]
    if not onnx_files:
        return None, None
    onnx_files.sort(key=lambda f: (0 if f.endswith("model.onnx") else 1, len(f)))
    chosen = onnx_files[0]

    def _is_tokenizer_file(f: str) -> bool:
        return (
            f.endswith((".json", ".txt", ".model"))
            or "vocab" in f
            or "merges" in f
            or "tokenizer" in f
            or "sentencepiece" in f
        )

    wanted: list[str] = [chosen] + [f for f in siblings if _is_tokenizer_file(f)]

    local_dir = Path(tempfile.gettempdir()) / "cra_td_onnx" / model_path.replace("/", "__")
    local_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    for fname in dict.fromkeys(wanted):
        dest = local_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = f"https://huggingface.co/{model_path}/resolve/main/{fname}"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=180) as r:
                if r.status_code != 200:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
        except Exception:
            continue

    onnx_local = local_dir / chosen
    if not onnx_local.exists():
        return None, None
    return str(onnx_local), str(local_dir)


def _load_onnx_engine_class():
    """
    Resolve tdsuite's ``OnnxInferenceEngine`` class.

    The ``tdsuite.utils`` package __init__ eagerly imports the torch-backed
    ``InferenceEngine``, so on a torch-less CPU box ``from tdsuite.utils import
    OnnxInferenceEngine`` fails. We then load ``onnx_inference.py`` directly by
    file path — it only needs numpy/pandas/onnxruntime/transformers — so ONNX
    CPU inference keeps working.
    """
    try:
        from tdsuite.utils import OnnxInferenceEngine

        return OnnxInferenceEngine
    except Exception:
        pass
    try:
        import importlib.util

        import tdsuite

        fp = Path(tdsuite.__file__).parent / "utils" / "onnx_inference.py"
        if not fp.exists():
            return None
        spec = importlib.util.spec_from_file_location("tdsuite_onnx_inference_standalone", fp)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "OnnxInferenceEngine", None)
    except Exception:
        return None


def _load_onnx_ensemble_engine_class():
    """
    Resolve tdsuite's ``OnnxEnsembleInferenceEngine`` class (torch-free).

    Same loading strategy as :func:`_load_onnx_engine_class`: the eager
    ``tdsuite.utils.__init__`` pulls torch/``datasets``-backed modules, so the
    direct import fails on a CPU-only box. We then load ``onnx_inference.py``
    by file path — it only needs numpy/pandas/onnxruntime/transformers — so the
    native weighted ONNX ensemble keeps working without torch. Returns ``None``
    when the installed tdsuite predates the engine.
    """
    try:
        from tdsuite.utils import OnnxEnsembleInferenceEngine

        return OnnxEnsembleInferenceEngine
    except Exception:
        pass
    try:
        import importlib.util

        import tdsuite

        fp = Path(tdsuite.__file__).parent / "utils" / "onnx_inference.py"
        if not fp.exists():
            return None
        spec = importlib.util.spec_from_file_location("tdsuite_onnx_inference_standalone", fp)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "OnnxEnsembleInferenceEngine", None)
    except Exception:
        return None


def _build_td_engine(model_path: str, onnx_path: str | None, device: str, backend: str):
    """
    Build a tdsuite inference engine ADAPTIVELY.

    Order: explicit local ONNX path → ONNX from_pretrained (if the installed
    build exposes it) → ONNX downloaded straight from the HF Hub (CPU, no torch)
    → PyTorch InferenceEngine (auto-downloads from HF Hub). All engines share
    the same predict_single()/predict_batch() return shape. Returns the engine
    instance, or raises on unrecoverable failure.
    """
    backend = (backend or "auto").lower()
    OnnxEngine = _load_onnx_engine_class()

    if onnx_path:
        if OnnxEngine is None:
            raise ImportError("OnnxInferenceEngine unavailable")
        return OnnxEngine(onnx_path)

    if backend in ("onnx", "auto") and OnnxEngine is not None:
        if hasattr(OnnxEngine, "from_pretrained"):
            try:
                return OnnxEngine.from_pretrained(model_path)  # type: ignore[attr-defined]
            except Exception:
                if backend == "onnx":
                    raise
        # CPU-only path: pull the ONNX export + tokenizer from the Hub. This is
        # what lets TD classification run without a working torch install.
        try:
            onnx_file, tok_dir = _download_onnx_from_hub(model_path)
            if onnx_file:
                return OnnxEngine(onnx_file, tokenizer_path=tok_dir)
        except Exception:
            if backend == "onnx":
                raise

    # PyTorch fallback (requires torch + transformers; auto-downloads model).
    from tdsuite.utils import InferenceEngine

    return InferenceEngine(model_path=model_path, device=device)


def classify_technical_debt(
    texts: list[str],
    model_path: str | None = None,
    category: str | None = None,
    onnx_path: str | None = None,
    device: str | None = None,
    backend: str | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """
    Classify text snippets with a BINARY, per-category technical-debt model.

    The model returns ``predicted_class`` 0/1 (1 = that category of debt is
    present) and ``predicted_probability``. Choose a category with ``category``
    (e.g. "security", "code", "design") or pass an explicit ``model_path`` /
    local ``onnx_path``. Inference needs network + model download, so this is
    intentionally lazy and defensive — it returns ``{"error": ...}`` rather than
    crashing if tdsuite / torch / onnxruntime are unavailable.
    """
    if not texts:
        return {"error": "No texts provided"}

    cfg = _get_cfg()
    if category and not model_path:
        key = category.strip().lower()
        if key not in TD_CATEGORY_MODELS:
            return {
                "error": f"Unknown TD category: {category}",
                "available_categories": sorted(TD_CATEGORY_MODELS),
            }
        model_path = TD_CATEGORY_MODELS[key]

    model_path = model_path or cfg.tools.td_classifier.model_path
    device = device or cfg.tools.td_classifier.device
    backend = backend or getattr(cfg.tools.td_classifier, "backend", "auto")
    onnx_path = onnx_path or getattr(cfg.tools.td_classifier, "onnx_path", None)
    batch_size = batch_size or getattr(cfg.tools.td_classifier, "batch_size", 32)

    try:
        import tdsuite  # noqa: F401  (lazy: heavy import, may pull torch)
    except ImportError as e:
        return {"error": f"tdsuite not available: {e}"}

    try:
        engine = _build_td_engine(model_path, onnx_path, device, backend)
    except ImportError as e:
        return {"error": f"TD inference backend unavailable: {e}"}
    except Exception as exc:
        return {"error": f"Failed to load TD model '{model_path}': {exc}"}

    predictions = _run_td_predictions(engine, texts, batch_size)

    return {
        "tool": "td_classify",
        "model": model_path,
        "label": td_label_for_model(model_path),
        "category": category,
        "predictions": predictions,
    }


def _run_td_predictions(engine: Any, texts: list[str], batch_size: int) -> list[dict]:
    """
    Run predictions over ``texts`` using a tdsuite engine.

    Uses the engine's vectorised ``predict_batch`` when there is more than one
    text and the method is available (much faster); otherwise falls back to
    per-text ``predict_single``. Every prediction's ``text`` is truncated to a
    preview length so results stay compact, and individual failures are
    captured per text rather than aborting the whole run.
    """
    predictions: list[dict] = []

    if len(texts) > 1 and hasattr(engine, "predict_batch"):
        try:
            raw = engine.predict_batch(texts, batch_size=batch_size)
            for text, result in zip(texts, raw, strict=False):
                if isinstance(result, dict):
                    result["text"] = text[:200]
                    predictions.append(result)
                else:
                    predictions.append({"text": text[:200], "raw": str(result)})
            return predictions
        except Exception:
            # Fall back to per-text inference on any batch failure.
            predictions = []

    for text in texts:
        try:
            result = engine.predict_single(text)
            if isinstance(result, dict):
                result["text"] = text[:200]
                predictions.append(result)
            else:
                predictions.append({"text": text[:200], "raw": str(result)})
        except Exception as exc:
            predictions.append({"text": text[:200], "error": str(exc)})

    return predictions


def list_td_categories() -> dict[str, Any]:
    """Return the available technical-debt categories and their model ids."""
    return {
        "tool": "td_categories",
        "categories": [
            {
                "category": cat,
                "model": TD_CATEGORY_MODELS[cat],
                "label": td_label_for_model(TD_CATEGORY_MODELS[cat]),
            }
            for cat in TD_PRIMARY_CATEGORIES
        ],
        "aliases": {k: v for k, v in TD_CATEGORY_MODELS.items() if k not in TD_PRIMARY_CATEGORIES},
    }


# ---------------------------------------------------------------------------
# Tool 3b: classify_technical_debt_all — sweep every category model
# ---------------------------------------------------------------------------


def classify_technical_debt_all(
    texts: list[str],
    categories: list[str] | None = None,
    device: str | None = None,
    backend: str | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """
    Classify ``texts`` against EVERY (or a chosen subset of) TD category model.

    Returns, per text, the list of categories whose binary model predicted the
    debt as present (class==1) plus the probability. This is the multi-label
    view tdsuite supports by running several binary models. Each category model
    downloads on first use, so this is intentionally lazy and defensive.
    """
    if not texts:
        return {"error": "No texts provided"}

    cats = [c.strip().lower() for c in (categories or TD_PRIMARY_CATEGORIES)]
    unknown = [c for c in cats if c not in TD_CATEGORY_MODELS]
    if unknown:
        return {
            "error": f"Unknown TD categories: {unknown}",
            "available_categories": sorted(TD_CATEGORY_MODELS),
        }

    per_text: list[dict[str, Any]] = [
        {"text": t[:200], "positive_categories": [], "scores": {}} for t in texts
    ]
    errors: list[dict] = []

    for cat in cats:
        result = classify_technical_debt(
            texts, category=cat, device=device, backend=backend, batch_size=batch_size
        )
        if "error" in result:
            errors.append({"category": cat, "error": result["error"]})
            continue
        label = result.get("label", cat)
        for i, pred in enumerate(result.get("predictions", [])):
            if not isinstance(pred, dict) or pred.get("error"):
                continue
            try:
                cls = int(pred.get("predicted_class"))
            except (TypeError, ValueError):
                continue
            # Record the probability that this category of debt is PRESENT
            # (class==1), so the score stays consistent with positive_categories.
            class_probs = pred.get("class_probabilities")
            if isinstance(class_probs, (list, tuple)) and len(class_probs) >= 2:
                present_prob = float(class_probs[1] or 0.0)
            else:
                present_prob = (
                    float(pred.get("predicted_probability", 0.0) or 0.0) if cls == 1 else 0.0
                )
            per_text[i]["scores"][label] = round(present_prob, 3)
            if cls == 1:
                per_text[i]["positive_categories"].append(label)

    return {
        "tool": "td_classify_all",
        "categories": cats,
        "results": per_text,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Tool 3c: classify_technical_debt_ensemble — weighted multi-model ensemble
# ---------------------------------------------------------------------------


def classify_technical_debt_ensemble(
    texts: list[str],
    model_names: list[str] | None = None,
    model_paths: list[str] | None = None,
    categories: list[str] | None = None,
    weights: list[float] | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """
    Classify ``texts`` with a WEIGHTED ENSEMBLE of TD models.

    Models can be given as HuggingFace ids (``model_names``), local checkpoint
    dirs (``model_paths``), or friendly category names (``categories``, resolved
    via the category->model map). ``weights`` are optional per-model weights
    (normalised to sum to 1).

    Backend resolution (``backend`` = ``auto`` | ``onnx`` | ``torch``):

    1. **onnx/auto** — tdsuite's native, torch-free ``OnnxEnsembleInferenceEngine``
       (weighted mean of per-model softmax probabilities; CPU by default, GPU via
       ``onnxruntime-gpu``). This is the default.
    2. **torch** (or when ONNX is unavailable) — the PyTorch
       ``EnsembleInferenceEngine``.
    3. Last resort — a hand-rolled CPU ensemble that runs each binary model
       independently and combines the present-probabilities.

    Stays defensive about missing heavy deps, returning ``{"error": ...}`` rather
    than crashing.
    """
    if not texts:
        return {"error": "No texts provided"}

    cfg = _get_cfg()
    device = device or cfg.tools.td_classifier.device
    batch_size = batch_size or getattr(cfg.tools.td_classifier, "batch_size", 32)
    backend = (backend or getattr(cfg.tools.td_classifier, "backend", "auto")).lower()

    names = list(model_names or [])
    if categories:
        for cat in categories:
            key = cat.strip().lower()
            if key not in TD_CATEGORY_MODELS:
                return {
                    "error": f"Unknown TD category: {cat}",
                    "available_categories": sorted(TD_CATEGORY_MODELS),
                }
            names.append(TD_CATEGORY_MODELS[key])

    if not names and not model_paths:
        return {"error": "Provide at least two models via model_names, model_paths, or categories"}

    try:
        import tdsuite  # noqa: F401
    except ImportError as e:
        return {"error": f"tdsuite not available: {e}"}

    all_models = (model_paths or []) + names

    # Preferred path: tdsuite's native torch-free ONNX ensemble engine.
    if backend in ("onnx", "auto"):
        OnnxEnsemble = _load_onnx_ensemble_engine_class()
        if OnnxEnsemble is not None:
            try:
                engine = OnnxEnsemble(
                    model_paths=model_paths or None,
                    model_names=names or None,
                    device=device,
                    weights=weights,
                )
                predictions = _run_td_predictions(engine, texts, batch_size)
                return {
                    "tool": "td_classify_ensemble",
                    "backend": "onnx",
                    "models": all_models,
                    "weights": getattr(engine, "weights", weights),
                    "label": "Technical Debt (ensemble)",
                    "predictions": predictions,
                }
            except Exception:
                # Fall through to torch / manual CPU ensemble below.
                pass
        elif backend == "onnx":
            return {
                "error": "OnnxEnsembleInferenceEngine unavailable: update tdsuite or use backend=torch"
            }

    # PyTorch ensemble: explicit opt-in, or auto fallback when ONNX is missing.
    if backend in ("torch", "auto"):
        try:
            from tdsuite.utils import EnsembleInferenceEngine

            engine = EnsembleInferenceEngine(
                model_paths=model_paths or None,
                model_names=names or None,
                device=device,
                weights=weights,
            )
            predictions = _run_td_predictions(engine, texts, batch_size)
            return {
                "tool": "td_classify_ensemble",
                "backend": "torch",
                "models": all_models,
                "weights": getattr(engine, "weights", weights),
                "label": "Technical Debt (ensemble)",
                "predictions": predictions,
            }
        except Exception:
            if backend == "torch":
                return {
                    "error": "PyTorch EnsembleInferenceEngine unavailable: install the torch extra"
                }
            # auto: fall through to the manual CPU ensemble below.

    return _ensemble_predict_cpu(
        all_models, texts, weights, device, backend="auto", batch_size=batch_size
    )


def _ensemble_predict_cpu(
    models: list[str],
    texts: list[str],
    weights: list[float] | None,
    device: str,
    backend: str,
    batch_size: int,
) -> dict[str, Any]:
    """
    CPU/ONNX weighted ensemble that needs no torch.

    Runs each model's binary classifier independently (via the same ONNX path
    used by ``classify_technical_debt``) and combines the per-text probability
    that debt is PRESENT (class==1) as a weighted average. ``predicted_class``
    is 1 when that weighted probability is >= 0.5.
    """
    if weights and len(weights) != len(models):
        return {"error": f"weights length ({len(weights)}) must match models ({len(models)})"}
    norm_weights = list(weights) if weights else [1.0] * len(models)
    total_w = sum(norm_weights) or 1.0
    norm_weights = [w / total_w for w in norm_weights]

    accum = [0.0] * len(texts)
    per_model: list[dict] = []
    model_errors: list[dict] = []

    for model, weight in zip(models, norm_weights, strict=False):
        result = classify_technical_debt(
            texts, model_path=model, device=device, backend=backend, batch_size=batch_size
        )
        if "error" in result:
            model_errors.append({"model": model, "error": result["error"]})
            continue
        contributions = []
        for i, pred in enumerate(result.get("predictions", [])):
            if not isinstance(pred, dict) or pred.get("error"):
                contributions.append(None)
                continue
            class_probs = pred.get("class_probabilities")
            if isinstance(class_probs, (list, tuple)) and len(class_probs) >= 2:
                present = float(class_probs[1] or 0.0)
            else:
                try:
                    cls = int(pred.get("predicted_class"))
                except (TypeError, ValueError):
                    cls = 0
                present = float(pred.get("predicted_probability", 0.0) or 0.0) if cls == 1 else 0.0
            accum[i] += weight * present
            contributions.append(round(present, 3))
        per_model.append(
            {"model": model, "weight": round(weight, 3), "present_probabilities": contributions}
        )

    if not per_model:
        return {
            "error": "TD ensemble backend unavailable: no models could be loaded",
            "model_errors": model_errors,
        }

    predictions = []
    for i, text in enumerate(texts):
        prob = round(accum[i], 3)
        predictions.append(
            {
                "text": text[:200],
                "predicted_class": 1 if prob >= 0.5 else 0,
                "predicted_probability": prob if prob >= 0.5 else round(1.0 - prob, 3),
                "ensemble_present_probability": prob,
            }
        )

    return {
        "tool": "td_classify_ensemble",
        "backend": "cpu-manual",
        "models": models,
        "weights": norm_weights,
        "label": "Technical Debt (ensemble)",
        "predictions": predictions,
        "per_model": per_model,
        "model_errors": model_errors,
    }


# ---------------------------------------------------------------------------
# Tool 3d: GitHub issues -> TD classification pipeline
# (mirrors tdsuite's fetch_github_issues.py + extract_issue_bodies.py scripts)
# ---------------------------------------------------------------------------


def fetch_github_issues(
    repo: str,
    state: str = "all",
    limit: int = 100,
    fetch_all: bool = False,
    token: str | None = None,
) -> dict[str, Any]:
    """
    Fetch issues from a public GitHub repo via the REST API.

    Mirrors tdsuite's ``fetch_github_issues.py`` script. Pull requests are
    filtered out (GitHub returns them through the issues endpoint). Returns a
    list of ``{number, title, body, state, labels, created_at}`` dicts.
    """
    if "/" not in repo:
        return {"error": "repo must be in 'owner/repo' format"}

    try:
        import requests
    except ImportError as e:
        return {"error": f"'requests' is required to fetch issues: {e}"}

    token = token or os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    issues: list[dict] = []
    per_page = 100 if fetch_all else min(limit, 100)
    page = 1
    base = f"https://api.github.com/repos/{repo}/issues"

    try:
        while True:
            resp = requests.get(
                base,
                headers=headers,
                params={
                    "state": state,
                    "per_page": per_page,
                    "page": page,
                    "sort": "created",
                    "direction": "desc",
                },
                timeout=30,
            )
            if resp.status_code == 403 and "rate limit" in resp.text.lower():
                return {"error": "GitHub rate limit exceeded; set GITHUB_TOKEN", "repo": repo}
            if resp.status_code != 200:
                return {
                    "error": f"GitHub API error {resp.status_code}: {resp.text[:200]}",
                    "repo": repo,
                }

            batch = resp.json()
            if not batch:
                break
            for it in batch:
                if "pull_request" in it:
                    continue  # skip PRs
                issues.append(
                    {
                        "number": it.get("number"),
                        "title": it.get("title", ""),
                        "body": it.get("body") or "",
                        "state": it.get("state", ""),
                        "labels": [lbl.get("name") for lbl in it.get("labels", [])],
                        "created_at": it.get("created_at", ""),
                    }
                )
                if not fetch_all and len(issues) >= limit:
                    break
            if not fetch_all and len(issues) >= limit:
                break
            page += 1
    except Exception as exc:
        return {"error": f"Failed to fetch issues: {exc}", "repo": repo}

    return {"tool": "github_issues", "repo": repo, "count": len(issues), "issues": issues}


def extract_issue_bodies(
    issues: list[dict],
    min_length: int = 20,
    drop_duplicates: bool = True,
) -> list[dict]:
    """
    Extract usable issue body texts (mirrors ``extract_issue_bodies.py``).

    Keeps bodies of at least ``min_length`` characters and optionally drops
    duplicate texts, preserving issue ``number``/``title`` metadata.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for it in issues:
        body = (it.get("body") or "").strip()
        if len(body) < min_length:
            continue
        if drop_duplicates and body in seen:
            continue
        seen.add(body)
        out.append({"number": it.get("number"), "title": it.get("title", ""), "text": body})
    return out


def classify_github_issues(
    repo: str,
    category: str | None = None,
    state: str = "all",
    limit: int = 50,
    fetch_all: bool = False,
    min_length: int = 20,
    token: str | None = None,
    device: str | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """
    End-to-end GitHub-issues technical-debt pipeline.

    Fetches issues from ``repo`` (owner/repo), extracts their body text, and
    classifies each body with a binary per-category TD model. Surfaces the
    full upstream tdsuite pipeline (fetch -> extract -> classify) as one tool.
    """
    fetched = fetch_github_issues(repo, state=state, limit=limit, fetch_all=fetch_all, token=token)
    if "error" in fetched:
        return fetched

    bodies = extract_issue_bodies(fetched["issues"], min_length=min_length)
    if not bodies:
        return {
            "tool": "github_issues_td",
            "repo": repo,
            "issues_fetched": fetched["count"],
            "classified": 0,
            "results": [],
            "note": "No issue bodies met the minimum length.",
        }

    texts = [b["text"] for b in bodies]
    td = classify_technical_debt(texts, category=category, device=device, backend=backend)
    if "error" in td:
        return {
            "tool": "github_issues_td",
            "repo": repo,
            "issues_fetched": fetched["count"],
            "error": td["error"],
        }

    results = []
    for meta, pred in zip(bodies, td.get("predictions", []), strict=False):
        results.append(
            {
                "number": meta["number"],
                "title": meta["title"],
                "text": (meta["text"][:200]),
                "label": td.get("label"),
                "predicted_class": pred.get("predicted_class"),
                "predicted_probability": pred.get("predicted_probability"),
                "error": pred.get("error"),
            }
        )

    return {
        "tool": "github_issues_td",
        "repo": repo,
        "category": category,
        "label": td.get("label"),
        "model": td.get("model"),
        "issues_fetched": fetched["count"],
        "classified": len(results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Tool 3e: tdsuite data-splitting / training / ONNX export wrappers
# Thin, defensive wrappers around the installed tdsuite package APIs. Heavy
# optional deps (torch, onnx) are guarded with clear error messages.
# ---------------------------------------------------------------------------


def _load_td_split_data_fn():
    """
    Resolve tdsuite's ``split_data`` function.

    The normal ``tdsuite.data`` package __init__ imports torch-backed modules,
    so on a torch-less CPU box we load ``data_splitter.py`` directly by file
    path (it only needs pandas/numpy/sklearn/datasets) to keep splitting usable.
    """
    try:
        from tdsuite.data.data_splitter import split_data

        return split_data
    except Exception:
        pass
    try:
        import importlib.util
        import sys
        import types

        import tdsuite

        fp = Path(tdsuite.__file__).parent / "data" / "data_splitter.py"
        if not fp.exists():
            return None
        # ``data_splitter.py`` does a top-level ``from datasets import ...`` but
        # only uses it for HuggingFace datasets. If the optional ``datasets``
        # dep is unavailable, install a thin shim so local-file splitting works.
        if "datasets" not in sys.modules:
            try:
                import datasets  # noqa: F401
            except Exception:
                shim = types.ModuleType("datasets")
                shim.Dataset = object

                def _load_dataset_unavailable(*_a, **_k):
                    raise RuntimeError(
                        "the 'datasets' package is required for HuggingFace datasets"
                    )

                shim.load_dataset = _load_dataset_unavailable
                sys.modules["datasets"] = shim
        spec = importlib.util.spec_from_file_location("tdsuite_data_splitter_standalone", fp)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "split_data", None)
    except Exception:
        return None


def td_split_data(
    data_file: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
    repo_column: str | None = None,
    is_huggingface_dataset: bool = False,
    is_numeric_labels: bool = False,
) -> dict[str, Any]:
    """Split/save a dataset for TD training (wraps ``tdsuite.data.data_splitter.split_data``)."""
    split_data = _load_td_split_data_fn()
    if split_data is None:
        return {"error": "tdsuite data splitter unavailable"}

    try:
        train_df, test_df, top_repo_df = split_data(
            data_file=data_file,
            output_dir=output_dir,
            test_size=test_size,
            random_state=random_state,
            repo_column=repo_column,
            is_huggingface_dataset=is_huggingface_dataset,
            is_numeric_labels=is_numeric_labels,
        )
    except Exception as exc:
        return {"error": f"Data split failed: {exc}"}

    return {
        "tool": "td_split_data",
        "output_dir": output_dir,
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "top_repo_samples": int(len(top_repo_df)) if top_repo_df is not None else 0,
        "files": ["train.csv", "test.csv"] + (["top_repos.csv"] if top_repo_df is not None else []),
    }


def td_export_onnx(
    output: str,
    model_name: str | None = None,
    model_path: str | None = None,
    max_length: int = 512,
    opset: int = 14,
) -> dict[str, Any]:
    """
    Export a transformer TD model to ONNX for CPU inference.

    Mirrors tdsuite's ``export_onnx.py`` script (not installed as a console
    entry point). Requires ``torch`` and the ``onnx`` package; both are guarded.
    The tokenizer is saved next to the ``.onnx`` file so OnnxInferenceEngine
    can load it.
    """
    source = model_path or model_name
    if not source:
        return {"error": "Provide model_name (HF id) or model_path (local dir)"}

    try:
        import torch  # noqa: F401
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        return {"error": f"torch + transformers required for ONNX export: {e}"}
    try:
        import onnx  # noqa: F401
    except ImportError:
        return {"error": "The 'onnx' package is required: uv pip install onnx onnxruntime"}

    try:
        import torch

        out_dir = os.path.dirname(os.path.abspath(output))
        os.makedirs(out_dir, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForSequenceClassification.from_pretrained(source)
        model.eval()

        dummy = tokenizer(
            "dummy input for onnx export",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        input_names = list(dummy.keys())
        dynamic_axes = {n: {0: "batch", 1: "sequence"} for n in input_names}
        dynamic_axes["logits"] = {0: "batch"}

        torch.onnx.export(
            model,
            tuple(dummy[n] for n in input_names),
            output,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )
        tokenizer.save_pretrained(out_dir)
    except Exception as exc:
        return {"error": f"ONNX export failed: {exc}"}

    return {
        "tool": "td_export_onnx",
        "source": source,
        "onnx_path": output,
        "tokenizer_dir": out_dir,
        "opset": opset,
    }


def td_train(
    data_file: str,
    model_name: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    positive_category: str | None = None,
    numeric_labels: bool = False,
    text_column: str = "text",
    label_column: str = "label",
    is_huggingface_dataset: bool = False,
    cross_validation: bool = False,
    device: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Train a binary TD classifier (wraps tdsuite's ``tdsuite-train`` entry point).

    Builds the same argv that ``tdsuite.train.main`` expects and invokes it.
    Requires ``torch`` (training on CPU is extremely slow and only attempted if
    explicitly requested); the dep is guarded with a clear message.
    """
    try:
        import torch  # noqa: F401
    except ImportError as e:
        return {"error": f"torch is required for training: {e}"}

    try:
        from tdsuite.train import main as train_main
    except ImportError as e:
        return {"error": f"tdsuite training entry point unavailable: {e}"}

    argv = [
        "--data_file",
        str(data_file),
        "--model_name",
        str(model_name),
        "--output_dir",
        str(output_dir),
        "--num_epochs",
        str(num_epochs),
        "--batch_size",
        str(batch_size),
        "--learning_rate",
        str(learning_rate),
        "--max_length",
        str(max_length),
        "--text_column",
        text_column,
        "--label_column",
        label_column,
        "--seed",
        str(seed),
    ]
    if positive_category:
        argv += ["--positive_category", positive_category]
    if numeric_labels:
        argv.append("--numeric_labels")
    if is_huggingface_dataset:
        argv.append("--is_huggingface_dataset")
    if cross_validation:
        argv.append("--cross_validation")
    if device:
        argv += ["--device", device]

    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["tdsuite-train"] + argv
        train_main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            return {"error": f"Training exited with code {exc.code}", "argv": argv}
    except Exception as exc:
        return {"error": f"Training failed: {exc}", "argv": argv}
    finally:
        sys.argv = old_argv

    return {
        "tool": "td_train",
        "output_dir": output_dir,
        "model_name": model_name,
        "status": "completed",
    }


# ---------------------------------------------------------------------------
# Tool 4: read_file
# ---------------------------------------------------------------------------


def read_file(file_path: str, max_lines: int | None = None) -> dict[str, Any]:
    """Read a Python file and return its contents with line numbers."""
    cfg = _get_cfg()
    max_lines = max_lines or cfg.tools.read_file_max_lines
    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}
    if not path.is_file():
        return {"error": f"Not a file: {file_path}"}

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        truncated = len(lines) > max_lines
        # Add line numbers
        numbered = "\n".join(f"{i + 1:4d} | {line}" for i, line in enumerate(lines[:max_lines]))
        return {
            "tool": "read_file",
            "file": str(path.resolve()),
            "total_lines": len(lines),
            "shown_lines": min(len(lines), max_lines),
            "content": numbered,
            "truncated": truncated,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 5: list_python_files
# ---------------------------------------------------------------------------


def list_python_files(
    directory: str,
    ignore_dirs: list[str] | None = None,
) -> dict[str, Any]:
    """List all Python files in a project directory with sizes."""
    target = Path(directory).resolve()
    if not target.exists():
        return {"error": f"Directory not found: {directory}"}
    if not target.is_dir():
        return {"error": f"Not a directory: {directory}"}

    cfg = _get_cfg()
    ignore = set(ignore_dirs or cfg.tools.ignore_dirs)

    files = []
    for f in sorted(target.rglob("*.py")):
        if any(part in ignore for part in f.parts):
            continue
        try:
            size = f.stat().st_size
        except Exception:
            size = 0
        files.append(
            {
                "path": str(f.relative_to(target)),
                "abs_path": str(f),
                "size_bytes": size,
                "size_kb": round(size / 1024, 1),
            }
        )

    return {
        "tool": "list_python_files",
        "directory": str(target),
        "total_files": len(files),
        "files": files,
    }


# ---------------------------------------------------------------------------
# Tool 6: analyze_code_intelligence
# ---------------------------------------------------------------------------


def analyze_code_intelligence(
    path: str,
    symbol: str | None = None,
    find_usages_of: str | None = None,
    metrics_only: bool = False,
    import_graph: bool = False,
    ignore_dirs: list[str] | None = None,
    top_n: int | None = None,
) -> dict[str, Any]:
    """
    AST-based code intelligence: symbols, metrics, import graph, usages.
    Returns exact file:line:col for every symbol and usage.
    """
    from code_review_agent.code_intel import CodeIntelligence

    target = Path(path).resolve()
    if not target.exists():
        return {"error": f"Path does not exist: {path}"}

    cfg = _get_cfg()
    ignore = ignore_dirs or cfg.tools.ignore_dirs
    n = top_n or cfg.code_intel.top_complexity_n

    ci = CodeIntelligence(
        include_private=cfg.code_intel.include_private_symbols,
        max_file_size_kb=cfg.code_intel.max_file_size_kb,
    )

    if target.is_file():
        intel_map = {str(target): ci.analyze_file(str(target))}
        root = str(target.parent)
    else:
        intel_map = ci.analyze_project(str(target), ignore)
        root = str(target)

    result: dict[str, Any] = {
        "tool": "code_intel",
        "target": str(target),
    }

    # Symbol lookup
    if symbol:
        defs = ci.lookup_symbol(symbol, intel_map)
        result["symbol_definitions"] = [
            {
                "name": d.name,
                "kind": d.kind,
                "parent": d.parent,
                "signature": d.signature,
                "docstring": d.docstring,
                "file": os.path.relpath(d.location.file, root),
                "line": d.location.line,
                "col": d.location.col,
                "end_line": d.location.end_line,
            }
            for d in defs
        ]

    # Find usages
    if find_usages_of:
        usages = ci.find_usages(find_usages_of, intel_map)
        result["usages"] = [
            {
                "file": os.path.relpath(u.location.file, root),
                "line": u.location.line,
                "col": u.location.col,
                "context": u.context_line,
            }
            for u in usages
        ]

    # Import graph
    if import_graph:
        graph = ci.build_import_graph(intel_map)
        result["import_graph"] = {
            os.path.relpath(fp, root): [
                {"module": e.module, "names": e.names, "line": e.line} for e in edges
            ]
            for fp, edges in graph.items()
        }

    # Always include project summary + metrics
    summary = ci.project_summary(intel_map, root, top_n=n)
    result["summary"] = summary

    if not metrics_only and not symbol and not find_usages_of and not import_graph:
        # Default: full summary
        pass

    return result


# ---------------------------------------------------------------------------
# OpenAI / Ollama tool schemas
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS_OPENAI: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "detect_ml_smells",
            "description": (
                "Detect ML-specific anti-patterns (Pandas, NumPy, Scikit-learn, PyTorch, "
                "TensorFlow, HuggingFace). Finds data leakage, magic numbers, reproducibility "
                "issues, improper API usage. Returns exact file:line:col locations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to .py file or project directory",
                    },
                    "ignore_dirs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Dirs to skip",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_python_smells",
            "description": (
                "Detect general Python code quality issues: code smells (long methods, large classes, "
                "duplicate code, feature envy), architectural smells (cyclic deps, god objects, "
                "hub-like modules), structural smells (high cyclomatic complexity, deep inheritance, "
                "low cohesion). Returns exact file:line:col locations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to .py file or project directory",
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["code", "architectural", "structural", "all"],
                        "description": "Which category to detect (default: 'all')",
                    },
                    "ignore_dirs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_technical_debt",
            "description": (
                "Classify text snippets (code comments, docstrings, commit messages, issue bodies) "
                "for technical debt using a BINARY, per-category transformer model. Returns "
                "predicted_class 0/1 (1 = the chosen category of debt is present) plus a "
                "probability. Use 'category' to pick a debt type. Available categories: general, "
                "code, design, documentation, test, defect, requirement, build, automation, people, "
                "process, infrastructure, architecture, service, security, performance, usability, "
                "maintainability, reliability, portability, compatibility. Inference downloads the "
                "model on first use."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Text snippets to classify",
                    },
                    "category": {
                        "type": "string",
                        "description": "Debt category: general (default), code, design, test, "
                        "security, documentation, defect, requirement, build, "
                        "performance, usability, maintainability, reliability, etc.",
                    },
                    "model_path": {
                        "type": "string",
                        "description": "HuggingFace model ID or local path",
                    },
                    "device": {"type": "string", "enum": ["cpu", "cuda", "mps"]},
                },
                "required": ["texts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "classify_github_issues",
            "description": (
                "Fetch issues from a public GitHub repo (owner/repo), extract their body text, and "
                "classify each for technical debt with a binary per-category model. Surfaces the "
                "full tdsuite GitHub-issues pipeline (fetch -> extract -> classify). Needs network; "
                "set GITHUB_TOKEN to avoid rate limits."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "Repository in 'owner/repo' format"},
                    "category": {
                        "type": "string",
                        "description": "Debt category (see classify_technical_debt)",
                    },
                    "state": {"type": "string", "enum": ["open", "closed", "all"]},
                    "limit": {"type": "integer", "description": "Max issues to fetch (default 50)"},
                },
                "required": ["repo"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a Python file with line numbers for detailed code review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "max_lines": {
                        "type": "integer",
                        "description": "Max lines to return (default: 500)",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_python_files",
            "description": "List all Python files in a project directory with sizes. Use this first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string"},
                    "ignore_dirs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["directory"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_code_intelligence",
            "description": (
                "AST-based code intelligence: symbol lookup, find usages, import dependency graph, "
                "per-function metrics (cyclomatic complexity, LOC, nesting depth). "
                "Returns exact file:line:col for every symbol and usage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File or project directory"},
                    "symbol": {
                        "type": "string",
                        "description": "Look up definitions of this symbol",
                    },
                    "find_usages_of": {
                        "type": "string",
                        "description": "Find all usages of this symbol",
                    },
                    "import_graph": {
                        "type": "boolean",
                        "description": "Include import dependency graph",
                    },
                    "metrics_only": {
                        "type": "boolean",
                        "description": "Return only function metrics",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Limit metrics to top N by complexity",
                    },
                    "ignore_dirs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["path"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Any] = {
    "detect_ml_smells": detect_ml_smells,
    "detect_python_smells": detect_python_smells,
    "classify_technical_debt": classify_technical_debt,
    "classify_technical_debt_all": classify_technical_debt_all,
    "classify_technical_debt_ensemble": classify_technical_debt_ensemble,
    "classify_github_issues": classify_github_issues,
    "list_td_categories": list_td_categories,
    "td_split_data": td_split_data,
    "td_export_onnx": td_export_onnx,
    "td_train": td_train,
    "read_file": read_file,
    "list_python_files": list_python_files,
    "analyze_code_intelligence": analyze_code_intelligence,
}


def execute_tool(name: str, inputs: dict[str, Any]) -> str:
    """Dispatch a tool call and return JSON-serialised result."""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = fn(**inputs)
        return json.dumps(result, default=str, indent=2)
    except Exception as exc:
        return json.dumps(
            {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
