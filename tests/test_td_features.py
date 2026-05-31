"""
Offline, deterministic tests for the technical-debt feature integrations added
on top of tdsuite: the corrected category->model map, batch inference,
multi-category sweep, weighted ensemble, the GitHub-issues pipeline, and the
data/ONNX lifecycle wrappers.

No model is ever downloaded and no network call is made: tdsuite engines and
``requests`` are mocked, and the only real dependency exercised is the local
data splitter (sklearn-backed) on a tiny in-memory CSV.
"""

from __future__ import annotations

import sys
import types

import pytest

from code_review_agent.config import reset_config


@pytest.fixture(autouse=True)
def reset_cfg():
    reset_config()
    yield
    reset_config()


# ---------------------------------------------------------------------------
# Corrected category -> model id map (verified against the live HF Hub)
# ---------------------------------------------------------------------------

class TestCategoryModelMap:
    def test_quality_attribute_abbreviations_are_correct(self):
        from code_review_agent.tools import TD_CATEGORY_MODELS as M
        assert M["performance"] == "karths/binary_classification_train_perf"
        assert M["usability"] == "karths/binary_classification_train_usab"
        assert M["security"] == "karths/binary_classification_train_secu"
        assert M["maintainability"] == "karths/binary_classification_train_main"
        assert M["reliability"] == "karths/binary_classification_train_reli"
        assert M["portability"] == "karths/binary_classification_train_port"
        assert M["compatibility"] == "karths/binary_classification_train_comp"

    def test_nonexistent_versioning_category_removed(self):
        from code_review_agent.tools import TD_CATEGORY_MODELS as M
        assert "versioning" not in M
        # The old wrong full-word ids must not be referenced anywhere.
        assert "karths/binary_classification_train_performance" not in M.values()
        assert "karths/binary_classification_train_usability" not in M.values()
        assert "karths/binary_classification_train_versioning" not in M.values()

    def test_every_model_has_a_label(self):
        from code_review_agent.tools import TD_CATEGORY_MODELS, TD_MODEL_LABELS
        for model in set(TD_CATEGORY_MODELS.values()):
            assert model in TD_MODEL_LABELS

    def test_list_td_categories(self):
        from code_review_agent.tools import list_td_categories
        result = list_td_categories()
        assert len(result["categories"]) == 21
        cats = {c["category"] for c in result["categories"]}
        assert {"general", "security", "performance", "maintainability", "service"} <= cats


# ---------------------------------------------------------------------------
# Fake tdsuite engines (single + ensemble) with batch support
# ---------------------------------------------------------------------------

class _FakeEngine:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def predict_single(self, text):
        return {"text": text, "predicted_class": 1, "predicted_probability": 0.88,
                "class_probabilities": [0.12, 0.88]}

    def predict_batch(self, texts, batch_size=32):
        return [
            {"text": t, "predicted_class": 1, "predicted_probability": 0.77,
             "class_probabilities": [0.23, 0.77]}
            for t in texts
        ]


@pytest.fixture
def fake_tdsuite(monkeypatch):
    tdsuite_mod = types.ModuleType("tdsuite")
    utils_mod = types.ModuleType("tdsuite.utils")

    class OnnxInferenceEngine(_FakeEngine):
        pass

    class InferenceEngine(_FakeEngine):
        pass

    class EnsembleInferenceEngine(_FakeEngine):
        def __init__(self, model_paths=None, model_names=None, device="cpu", weights=None):
            super().__init__()
            self.weights = weights or [1.0]

    utils_mod.OnnxInferenceEngine = OnnxInferenceEngine
    utils_mod.InferenceEngine = InferenceEngine
    utils_mod.EnsembleInferenceEngine = EnsembleInferenceEngine
    tdsuite_mod.utils = utils_mod

    monkeypatch.setitem(sys.modules, "tdsuite", tdsuite_mod)
    monkeypatch.setitem(sys.modules, "tdsuite.utils", utils_mod)
    return {"onnx": OnnxInferenceEngine, "torch": InferenceEngine, "ensemble": EnsembleInferenceEngine}


# ---------------------------------------------------------------------------
# Batch inference path
# ---------------------------------------------------------------------------

class TestBatchClassification:
    def test_multiple_texts_use_predict_batch(self, fake_tdsuite):
        from code_review_agent.tools import classify_technical_debt
        result = classify_technical_debt(["a", "b", "c"], backend="torch")
        assert len(result["predictions"]) == 3
        # Batch engine returns 0.77 (single would be 0.88).
        assert all(p["predicted_probability"] == 0.77 for p in result["predictions"])

    def test_single_text_uses_predict_single(self, fake_tdsuite):
        from code_review_agent.tools import classify_technical_debt
        result = classify_technical_debt(["only one"], backend="torch")
        assert result["predictions"][0]["predicted_probability"] == 0.88


# ---------------------------------------------------------------------------
# Multi-category sweep
# ---------------------------------------------------------------------------

class TestClassifyAll:
    def test_sweep_subset_of_categories(self, fake_tdsuite):
        from code_review_agent.tools import classify_technical_debt_all
        result = classify_technical_debt_all(["x"], categories=["security", "design"], backend="torch")
        assert result["categories"] == ["security", "design"]
        positives = result["results"][0]["positive_categories"]
        assert "Security Debt" in positives and "Design Debt" in positives
        assert result["results"][0]["scores"]["Security Debt"] == 0.88

    def test_unknown_category_errors(self, fake_tdsuite):
        from code_review_agent.tools import classify_technical_debt_all
        result = classify_technical_debt_all(["x"], categories=["nope"])
        assert "error" in result

    def test_empty_texts_errors(self):
        from code_review_agent.tools import classify_technical_debt_all
        assert "error" in classify_technical_debt_all([])


# ---------------------------------------------------------------------------
# Weighted ensemble
# ---------------------------------------------------------------------------

class TestEnsemble:
    def test_ensemble_from_categories(self, fake_tdsuite):
        from code_review_agent.tools import classify_technical_debt_ensemble
        result = classify_technical_debt_ensemble(
            ["a", "b"], categories=["security", "design"], weights=[0.6, 0.4]
        )
        assert result["tool"] == "td_classify_ensemble"
        assert len(result["predictions"]) == 2
        assert "karths/binary_classification_train_secu" in result["models"]

    def test_ensemble_requires_models(self, fake_tdsuite):
        from code_review_agent.tools import classify_technical_debt_ensemble
        assert "error" in classify_technical_debt_ensemble(["a"])

    def test_ensemble_unknown_category(self, fake_tdsuite):
        from code_review_agent.tools import classify_technical_debt_ensemble
        assert "error" in classify_technical_debt_ensemble(["a"], categories=["bogus"])

    def test_ensemble_cpu_fallback_when_torch_engine_missing(self, fake_tdsuite, monkeypatch):
        # Drop the torch-backed ensemble engine so the import fails and the
        # CPU/ONNX manual weighted ensemble path is exercised instead.
        import sys as _sys
        utils_mod = _sys.modules["tdsuite.utils"]
        monkeypatch.delattr(utils_mod, "EnsembleInferenceEngine", raising=False)
        from code_review_agent.tools import classify_technical_debt_ensemble
        result = classify_technical_debt_ensemble(
            ["a"], categories=["security", "design"], weights=[0.7, 0.3]
        )
        assert result["tool"] == "td_classify_ensemble"
        # Each fake model reports class-1 prob 0.88, so weighted avg is 0.88 -> class 1.
        pred = result["predictions"][0]
        assert pred["predicted_class"] == 1
        assert pred["ensemble_present_probability"] == 0.88
        assert len(result["per_model"]) == 2


# ---------------------------------------------------------------------------
# GitHub issues pipeline (requests mocked)
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = "ok"

    def json(self):
        return self._payload


@pytest.fixture
def fake_requests(monkeypatch):
    import requests

    pages = {
        1: [
            {"number": 1, "title": "Bug", "body": "This is a real issue body long enough", "labels": [], "state": "open", "created_at": "x"},
            {"number": 2, "title": "PR", "body": "ignore me", "pull_request": {"url": "..."}, "labels": [], "state": "open", "created_at": "x"},
            {"number": 3, "title": "Short", "body": "tiny", "labels": [], "state": "open", "created_at": "x"},
        ],
    }

    def fake_get(url, headers=None, params=None, timeout=None):
        page = (params or {}).get("page", 1)
        return _FakeResp(pages.get(page, []))

    monkeypatch.setattr(requests, "get", fake_get)
    return pages


class TestGithubIssues:
    def test_fetch_filters_prs(self, fake_requests):
        from code_review_agent.tools import fetch_github_issues
        result = fetch_github_issues("owner/repo", limit=10)
        assert result["count"] == 2  # PR filtered out
        numbers = {i["number"] for i in result["issues"]}
        assert 2 not in numbers

    def test_bad_repo_format(self):
        from code_review_agent.tools import fetch_github_issues
        assert "error" in fetch_github_issues("not-a-repo")

    def test_extract_bodies_min_length_and_dedup(self):
        from code_review_agent.tools import extract_issue_bodies
        issues = [
            {"number": 1, "title": "a", "body": "long enough body text here"},
            {"number": 2, "title": "b", "body": "long enough body text here"},  # dup
            {"number": 3, "title": "c", "body": "short"},
        ]
        out = extract_issue_bodies(issues, min_length=10, drop_duplicates=True)
        assert len(out) == 1

    def test_classify_github_issues_end_to_end(self, fake_requests, fake_tdsuite):
        from code_review_agent.tools import classify_github_issues
        result = classify_github_issues("owner/repo", category="defect", backend="torch")
        assert result["tool"] == "github_issues_td"
        assert result["issues_fetched"] == 2
        assert result["classified"] == 1
        assert result["results"][0]["predicted_class"] == 1


# ---------------------------------------------------------------------------
# Data split (real, sklearn-backed) + ONNX export / training guards
# ---------------------------------------------------------------------------

class TestLifecycleWrappers:
    def test_split_data_real(self, tmp_path):
        from code_review_agent.tools import td_split_data
        csv = tmp_path / "data.csv"
        rows = ["text,label"]
        for i in range(20):
            rows.append(f"sample text number {i},{i % 2}")
        csv.write_text("\n".join(rows), encoding="utf-8")
        out = tmp_path / "out"
        result = td_split_data(str(csv), str(out), test_size=0.25, is_numeric_labels=True)
        if "error" in result:
            pytest.skip(f"data splitter unavailable: {result['error']}")
        assert result["train_samples"] + result["test_samples"] == 20
        assert (out / "train.csv").exists()
        assert (out / "test.csv").exists()

    def test_export_onnx_guards_missing_deps(self, tmp_path):
        """With torch/onnx unavailable, export returns a clean error, not a crash."""
        from code_review_agent.tools import td_export_onnx
        result = td_export_onnx(str(tmp_path / "m.onnx"), model_name="some/model")
        assert "error" in result
        assert ("onnx" in result["error"].lower()) or ("torch" in result["error"].lower())

    def test_export_onnx_requires_a_source(self, tmp_path):
        from code_review_agent.tools import td_export_onnx
        result = td_export_onnx(str(tmp_path / "m.onnx"))
        assert "error" in result

    def test_train_invokes_entry_point(self, tmp_path, monkeypatch):
        """td_train should build argv and call tdsuite.train.main (both mocked)."""
        captured = {}

        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        train_mod = types.ModuleType("tdsuite.train")

        def fake_main():
            captured["argv"] = list(sys.argv)

        train_mod.main = fake_main
        monkeypatch.setitem(sys.modules, "tdsuite.train", train_mod)

        data = tmp_path / "data.csv"
        data.write_text("text,label\nx,0\n", encoding="utf-8")
        from code_review_agent.tools import td_train
        result = td_train(str(data), "roberta-base", str(tmp_path / "out"),
                           num_epochs=1, positive_category="security")
        assert result.get("status") == "completed"
        assert "--model_name" in captured["argv"]
        assert "roberta-base" in captured["argv"]
        assert "--positive_category" in captured["argv"]

    def test_train_captures_errors(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        train_mod = types.ModuleType("tdsuite.train")

        def boom():
            raise RuntimeError("kaboom")

        train_mod.main = boom
        monkeypatch.setitem(sys.modules, "tdsuite.train", train_mod)

        data = tmp_path / "data.csv"
        data.write_text("text,label\nx,0\n", encoding="utf-8")
        from code_review_agent.tools import td_train
        result = td_train(str(data), "roberta-base", str(tmp_path / "out"))
        assert "error" in result

    def test_train_guards_missing_torch(self, tmp_path, monkeypatch):
        """With torch genuinely absent, td_train returns a clean error (no crash)."""
        import importlib.util
        if importlib.util.find_spec("torch") is not None:
            try:
                import torch  # noqa: F401
                pytest.skip("torch importable; guard path not exercised")
            except Exception:
                pass
        data = tmp_path / "data.csv"
        data.write_text("text,label\nx,0\n", encoding="utf-8")
        from code_review_agent.tools import td_train
        result = td_train(str(data), "roberta-base", str(tmp_path / "out"))
        assert "error" in result and "torch" in result["error"].lower()
