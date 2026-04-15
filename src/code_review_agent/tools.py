"""
Tool implementations wrapping the three analyser packages + AST code intelligence.
All outputs use the canonical findings shape with exact file:line:col.
"""

from __future__ import annotations

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
    return [
        f for f in path.rglob("*.py")
        if not any(part in ignore for part in f.parts)
    ]


def _get_cfg():
    """Lazy import of config to avoid circular imports."""
    from code_review_agent.config import get_config
    return get_config()


def _smell_to_dict(obj: Any) -> dict:
    """Convert a smell dataclass/object to a plain dict."""
    if isinstance(obj, dict):
        return obj
    try:
        return obj.__dict__
    except Exception:
        return {"raw": str(obj)}


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
        detector = DetectorCls()
        for py_file in py_files:
            file_str = str(py_file)
            try:
                raw = detector.detect_smells(file_str)
                if raw:
                    # raw may be list[dict] or dict
                    if isinstance(raw, list):
                        smell_list = [_smell_to_dict(s) for s in raw]
                    elif isinstance(raw, dict):
                        smell_list = list(raw.values()) if raw else []
                    else:
                        smell_list = []

                    # Enrich column info where missing
                    for smell in smell_list:
                        if isinstance(smell, dict) and smell.get("line_number") and not smell.get("col"):
                            snippet = smell.get("code_snippet", "")
                            if snippet:
                                col = _enrich_column(file_str, smell["line_number"], snippet)
                                if col:
                                    smell["col"] = col

                    if smell_list:
                        results[key].append({"file": file_str, "smells": smell_list})
            except Exception as exc:
                results["errors"].append({"file": file_str, "error": str(exc)})

    total_smells = sum(len(e["smells"]) for key in ("framework_smells", "huggingface_smells", "general_ml_smells") for e in results[key])
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
            CodeSmellDetector,
            ArchitecturalSmellDetector,
            StructuralSmellDetector,
        )
    except ImportError as e:
        return {"error": f"code_quality_analyzer not available: {e}"}

    target = Path(path).resolve()
    if not target.exists():
        return {"error": f"Path does not exist: {path}"}

    cfg = _get_cfg()
    ignore = ignore_dirs or cfg.tools.ignore_dirs

    from code_review_agent.config import get_thresholds
    code_thresh = get_thresholds(cfg, "code_smells")
    arch_thresh  = get_thresholds(cfg, "architectural_smells")
    struct_thresh = get_thresholds(cfg, "structural_smells")

    results: dict[str, Any] = {
        "tool": "python_smells",
        "target": str(target),
        "analysis_type": analysis_type,
        "errors": [],
    }

    # ---- Code smells ----
    if analysis_type in ("code", "all"):
        try:
            det = CodeSmellDetector(thresholds=code_thresh) if code_thresh else CodeSmellDetector()
            py_files = _python_files(target, set(ignore))
            for py_file in py_files:
                try:
                    det.detect_smells(str(py_file))
                except Exception as exc:
                    results["errors"].append({"file": str(py_file), "phase": "code_smells", "error": str(exc)})
            try:
                det.detect_cross_file_smells()
            except Exception:
                pass
            results["code_smells"] = _extract_smell_list(det, "code_smells")
        except Exception as exc:
            results["code_smells"] = {"error": str(exc)}

    # ---- Architectural smells ----
    if analysis_type in ("architectural", "all") and target.is_dir():
        try:
            det = ArchitecturalSmellDetector(thresholds=arch_thresh) if arch_thresh else ArchitecturalSmellDetector()
            det.analyze_directory(str(target), ignore)
            results["architectural_smells"] = _extract_smell_list(det, "architectural_smells")
        except Exception as exc:
            results["architectural_smells"] = {"error": str(exc)}

    # ---- Structural smells ----
    if analysis_type in ("structural", "all"):
        try:
            det = StructuralSmellDetector(thresholds=struct_thresh) if struct_thresh else StructuralSmellDetector()
            if target.is_dir():
                det.detect_smells(str(target), ignore)
            else:
                det.analyze_file(str(target))
            results["structural_smells"] = _extract_smell_list(det, "structural_smells")
        except Exception as exc:
            results["structural_smells"] = {"error": str(exc)}

    return results


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
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            detector.print_report()
        return [{"report": buf.getvalue()}]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# tdsuite loader helpers
# tdsuite.utils.__init__ eagerly imports data_utils → datasets (optional dep).
# We load onnx_inference.py directly from disk to bypass the broken __init__.
# ---------------------------------------------------------------------------

def _load_tdsuite_onnx_cls():
    """Return OnnxInferenceEngine class, bypassing tdsuite.utils.__init__."""
    import importlib.util
    import importlib
    spec = importlib.util.find_spec("tdsuite")
    if spec is None:
        raise ImportError("tdsuite is not installed")
    import pathlib
    tdsuite_root = pathlib.Path(spec.origin).parent
    onnx_file = tdsuite_root / "utils" / "onnx_inference.py"
    if not onnx_file.exists():
        raise ImportError(f"tdsuite ONNX module not found at {onnx_file}")
    mod_spec = importlib.util.spec_from_file_location("_tdsuite_onnx_inference", onnx_file)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return mod.OnnxInferenceEngine


def _load_tdsuite_torch_cls():
    """Return InferenceEngine (PyTorch) class."""
    import importlib.util
    import importlib
    spec = importlib.util.find_spec("tdsuite")
    if spec is None:
        raise ImportError("tdsuite is not installed")
    import pathlib
    tdsuite_root = pathlib.Path(spec.origin).parent
    torch_file = tdsuite_root / "utils" / "inference.py"
    if not torch_file.exists():
        raise ImportError(f"tdsuite torch inference module not found at {torch_file}")
    mod_spec = importlib.util.spec_from_file_location("_tdsuite_torch_inference", torch_file)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return mod.InferenceEngine


def _export_hf_model_to_onnx(model_path: str) -> str:
    """Export a HuggingFace model to ONNX; return path to the exported model.onnx.

    Used when the model's HF-hosted model.onnx has broken external data references.
    Caches the exported ONNX under ~/.cache/td_onnx_export/.
    """
    import torch
    from pathlib import Path
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cache_dir = Path.home() / ".cache" / "td_onnx_export" / model_path.replace("/", "__")
    onnx_path = cache_dir / "model.onnx"

    if not onnx_path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()

        dummy = tokenizer(
            "test input", return_tensors="pt",
            padding="max_length", max_length=512, truncation=True,
        )
        input_names = list(dummy.keys())

        torch.onnx.export(
            model,
            tuple(dummy[k] for k in input_names),
            str(onnx_path),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes={k: {0: "batch", 1: "seq"} for k in input_names},
            opset_version=14,
        )
        tokenizer.save_pretrained(str(cache_dir))

    return str(cache_dir)


def _load_td_onnx_engine(model_path: str, device: str):
    """Build and return an OnnxInferenceEngine, or a dict error.

    First tries tdsuite's OnnxInferenceEngine.from_pretrained (works when the
    HF repo ships a self-contained model.onnx).  If that fails because the
    hosted model.onnx uses external weight files that weren't uploaded to HF,
    we export the model to ONNX from its safetensors weights and try again.
    """
    try:
        OnnxInferenceEngine = _load_tdsuite_onnx_cls()
    except ImportError as e:
        return {"error": f"tdsuite ONNX inference not available: {e}"}

    try:
        return OnnxInferenceEngine.from_pretrained(model_path, device=device)
    except Exception as first_exc:
        # Detect broken external-data ONNX and fall back to local export
        if "cannot get file size" in str(first_exc) or "onnx.data" in str(first_exc):
            try:
                exported_dir = _export_hf_model_to_onnx(model_path)
                onnx_file = str(Path(exported_dir) / "model.onnx")
                return OnnxInferenceEngine(
                    onnx_path=onnx_file,
                    tokenizer_path=exported_dir,
                    device=device,
                )
            except Exception as export_exc:
                return {
                    "error": f"ONNX export failed: {export_exc}",
                    "traceback": traceback.format_exc(),
                }
        return {"error": str(first_exc), "traceback": traceback.format_exc()}


def _load_td_torch_engine(model_path: str, device: str):
    """Build and return a PyTorch InferenceEngine, or a dict error."""
    try:
        InferenceEngine = _load_tdsuite_torch_cls()
    except ImportError as e:
        return {"error": f"tdsuite PyTorch inference not available (install tdsuite[gpu]): {e}"}
    try:
        return InferenceEngine(model_path=model_path, device=device)
    except Exception as exc:
        return {"error": str(exc), "traceback": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Tool 3: classify_technical_debt
# ---------------------------------------------------------------------------

def classify_technical_debt(
    texts: list[str],
    model_path: str | None = None,
    device: str | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Classify text snippets into technical debt categories using tdsuite.

    backend: "onnx" (default, CPU-only, no PyTorch required) | "torch"
    """
    if not texts:
        return {"error": "No texts provided"}

    cfg = _get_cfg()
    model_path = model_path or cfg.tools.td_classifier.model_path
    device = device or cfg.tools.td_classifier.device
    backend = backend or cfg.tools.td_classifier.backend

    if backend == "torch":
        engine = _load_td_torch_engine(model_path, device)
        if isinstance(engine, dict):
            return engine
    else:
        # ONNX backend (default) — downloads model.onnx from HF Hub on first run
        engine = _load_td_onnx_engine(model_path, device)
        if isinstance(engine, dict):
            return engine

    try:
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
        return {
            "tool": "td_classify",
            "backend": backend,
            "model": model_path,
            "predictions": predictions,
        }
    except Exception as exc:
        return {"error": str(exc), "traceback": traceback.format_exc()}


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
        numbered = "\n".join(f"{i+1:4d} | {line}" for i, line in enumerate(lines[:max_lines]))
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
        files.append({
            "path": str(f.relative_to(target)),
            "abs_path": str(f),
            "size_bytes": size,
            "size_kb": round(size / 1024, 1),
        })

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
                {"module": e.module, "names": e.names, "line": e.line}
                for e in edges
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
                    "path": {"type": "string", "description": "Path to .py file or project directory"},
                    "ignore_dirs": {"type": "array", "items": {"type": "string"}, "description": "Dirs to skip"},
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
                    "path": {"type": "string", "description": "Path to .py file or project directory"},
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
                "into 18 technical debt categories using a transformer ML model."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "texts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Text snippets to classify",
                    },
                    "model_path": {"type": "string", "description": "HuggingFace model ID or local path"},
                    "device": {"type": "string", "enum": ["cpu", "cuda", "mps"]},
                    "backend": {"type": "string", "enum": ["onnx", "torch"], "description": "Inference backend: 'onnx' (default, CPU, no PyTorch) or 'torch'"},
                },
                "required": ["texts"],
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
                    "max_lines": {"type": "integer", "description": "Max lines to return (default: 500)"},
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
                    "symbol": {"type": "string", "description": "Look up definitions of this symbol"},
                    "find_usages_of": {"type": "string", "description": "Find all usages of this symbol"},
                    "import_graph": {"type": "boolean", "description": "Include import dependency graph"},
                    "metrics_only": {"type": "boolean", "description": "Return only function metrics"},
                    "top_n": {"type": "integer", "description": "Limit metrics to top N by complexity"},
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
        return json.dumps({
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
