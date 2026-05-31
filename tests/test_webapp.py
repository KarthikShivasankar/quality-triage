"""Offline tests for the FastAPI web UI (skipped if the web extra is absent)."""

from __future__ import annotations

import importlib.util

import pytest

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi (web extra) not installed")


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from code_review_agent.webapp.app import create_app

    return TestClient(create_app())


@pytest.fixture
def proj(tmp_path):
    (tmp_path / "mod.py").write_text("import numpy as np\nmodel = train()\n", encoding="utf-8")
    return tmp_path


class TestIndex:
    def test_index_renders_controls(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert 'name="target"' in r.text
        assert 'name="checks"' in r.text
        assert 'value="ml"' in r.text
        assert 'name="td_categories"' in r.text

    def test_healthz(self, client):
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestReviewRoute:
    def test_review_maps_controls_to_service(self, client, monkeypatch, proj):
        from code_review_agent import service
        from code_review_agent.fixes import FixSuggestion

        captured = {}

        def fake_run_review(cfg, **kwargs):
            captured.update(kwargs)
            return {
                "text": "MOCK REPORT BODY",
                "fixes": [{"file": "mod.py", "start_line": 2, "end_line": 2,
                           "original": "model = train()", "replacement": "model = train(seed=42)",
                           "description": "seed", "finding_id": None, "source": "agent"}],
                "fix_objects": [FixSuggestion("mod.py", 2, 2, "model = train()",
                                              "model = train(seed=42)", "seed")],
                "selection": {"families": ["ml"], "skipped": ["code", "td"],
                              "td_categories": ["general"]},
                "provider": "ollama", "model": None, "target": str(proj),
            }

        monkeypatch.setattr(service, "run_review", fake_run_review)
        r = client.post("/review", data={
            "target": str(proj), "provider": "ollama", "model": "qwen3.5:4b",
            "checks": ["ml"], "suggest_fixes": "1", "fmt": "html",
        })
        assert r.status_code == 200
        # Controls reached the service untouched.
        assert captured["checks"] == ["ml"]
        assert captured["suggest_fixes"] is True
        # Report + selection + fix preview rendered.
        assert "MOCK REPORT BODY" in r.text
        assert "ml" in r.text and "skipped" in r.text.lower()
        assert "Suggested Fixes" in r.text

    def test_review_json_format(self, client, monkeypatch, proj):
        from code_review_agent import service

        def fake_run_review(cfg, **kwargs):
            return {
                "text": "JSON REPORT", "fixes": [], "fix_objects": [],
                "selection": {"families": ["code"], "skipped": [], "td_categories": ["general"]},
                "provider": "ollama", "model": None, "target": str(proj),
            }

        monkeypatch.setattr(service, "run_review", fake_run_review)
        r = client.post("/review", data={"target": str(proj), "fmt": "json", "checks": ["code"]})
        assert r.status_code == 200
        payload = r.json()
        assert payload["report_markdown"] == "JSON REPORT"
        assert payload["selection"]["families"] == ["code"]


class TestApplyRoute:
    def test_apply_writes_only_with_confirm(self, client, proj):
        import json

        f = proj / "mod.py"
        fixes_json = json.dumps([{
            "file": "mod.py", "start_line": 2, "end_line": 2,
            "original": "model = train()", "replacement": "model = retrain()",
            "description": "rename", "finding_id": None,
        }])

        # Without confirm: nothing is written.
        r0 = client.post("/fixes/apply", data={"target": str(proj), "fixes_json": fixes_json, "confirm": ""})
        assert r0.status_code == 200
        assert "model = train()" in f.read_text(encoding="utf-8")

        # With confirm: the fix is applied.
        r1 = client.post("/fixes/apply", data={"target": str(proj), "fixes_json": fixes_json, "confirm": "1"})
        assert r1.status_code == 200
        assert "model = retrain()" in f.read_text(encoding="utf-8")

    def test_apply_refuses_outside_root(self, client, proj):
        import json

        fixes_json = json.dumps([{
            "file": "../escape.py", "start_line": 1, "end_line": 1,
            "original": "x = 1", "replacement": "x = 2", "description": "", "finding_id": None,
        }])
        r = client.post("/fixes/apply", data={"target": str(proj), "fixes_json": fixes_json, "confirm": "1"})
        assert r.status_code == 200
        assert "skipped" in r.text.lower()
