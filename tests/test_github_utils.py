"""
Tests for code_review_agent.github_utils
"""

import pytest

from code_review_agent.github_utils import (
    ClonedRepo,
    GithubIssue,
    clamp_issue_limit,
    cleanup_repo,
    fetch_recent_open_issues,
    github_repo_name,
    is_github_url,
    issue_snippets,
    parse_github_issues,
    parse_github_url,
    resolve_review_target,
)

# ---------------------------------------------------------------------------
# is_github_url
# ---------------------------------------------------------------------------


class TestIsGithubUrl:
    def test_https_url(self):
        assert is_github_url("https://github.com/owner/repo") is True

    def test_https_url_with_tree(self):
        assert is_github_url("https://github.com/owner/repo/tree/main") is True

    def test_https_url_with_dotgit(self):
        assert is_github_url("https://github.com/owner/repo.git") is True

    def test_ssh_url(self):
        assert is_github_url("git@github.com:owner/repo.git") is True

    def test_local_path(self):
        assert is_github_url("/home/user/project") is False

    def test_relative_path(self):
        assert is_github_url("./my_project") is False

    def test_non_github_url(self):
        assert is_github_url("https://gitlab.com/owner/repo") is False

    def test_empty_string(self):
        assert is_github_url("") is False


# ---------------------------------------------------------------------------
# parse_github_url
# ---------------------------------------------------------------------------


class TestParseGithubUrl:
    def test_basic_https(self):
        clone_url, branch, subpath = parse_github_url("https://github.com/owner/repo")
        assert clone_url == "https://github.com/owner/repo.git"
        assert branch is None
        assert subpath is None

    def test_https_with_dotgit(self):
        clone_url, branch, _ = parse_github_url("https://github.com/owner/repo.git")
        assert clone_url == "https://github.com/owner/repo.git"

    def test_https_with_tree_branch(self):
        clone_url, branch, subpath = parse_github_url(
            "https://github.com/owner/repo/tree/dev"
        )
        assert clone_url == "https://github.com/owner/repo.git"
        assert branch == "dev"
        assert subpath is None

    def test_https_with_tree_branch_and_subpath(self):
        clone_url, branch, subpath = parse_github_url(
            "https://github.com/owner/repo/tree/main/src/lib"
        )
        assert branch == "main"
        assert subpath == "src/lib"

    def test_ssh_format(self):
        clone_url, branch, subpath = parse_github_url("git@github.com:owner/repo.git")
        assert clone_url == "https://github.com/owner/repo.git"
        assert branch is None

    def test_non_github_raises(self):
        with pytest.raises(ValueError, match="Not a GitHub URL"):
            parse_github_url("https://gitlab.com/owner/repo")

    def test_incomplete_url_raises(self):
        with pytest.raises(ValueError):
            parse_github_url("https://github.com/onlyowner")


# ---------------------------------------------------------------------------
# cleanup_repo
# ---------------------------------------------------------------------------


class TestCleanupRepo:
    def test_cleanup_deletes_temp_dir(self, tmp_path):
        """cleanup_repo removes directory when is_temp=True."""
        fake_dir = tmp_path / "cloned_repo"
        fake_dir.mkdir()
        (fake_dir / "file.py").write_text("x = 1")

        repo = ClonedRepo(
            url="https://github.com/owner/repo",
            local_path=str(fake_dir),
            repo_name="owner/repo",
            branch="main",
            commit_sha="abc1234",
            is_temp=True,
        )
        cleanup_repo(repo)
        assert not fake_dir.exists()

    def test_cleanup_skips_non_temp_dir(self, tmp_path):
        """cleanup_repo does NOT remove directory when is_temp=False."""
        fake_dir = tmp_path / "persistent_repo"
        fake_dir.mkdir()

        repo = ClonedRepo(
            url="https://github.com/owner/repo",
            local_path=str(fake_dir),
            repo_name="owner/repo",
            branch="main",
            commit_sha="abc1234",
            is_temp=False,
        )
        cleanup_repo(repo)
        assert fake_dir.exists()

    def test_review_path_appends_subpath(self, tmp_path):
        repo = ClonedRepo(
            url="https://github.com/owner/repo/tree/main/src",
            local_path=str(tmp_path),
            repo_name="owner/repo",
            branch="main",
            commit_sha="abc",
            is_temp=True,
            subpath="src",
        )
        assert repo.review_path == str(tmp_path / "src")

    def test_cleanup_nonexistent_dir_is_safe(self, tmp_path):
        """cleanup_repo on a path that does not exist should not raise."""
        repo = ClonedRepo(
            url="https://github.com/owner/repo",
            local_path=str(tmp_path / "ghost"),
            repo_name="owner/repo",
            branch="main",
            commit_sha="abc1234",
            is_temp=True,
        )
        cleanup_repo(repo)  # should not raise


def test_resolve_review_target_local(tmp_path):
    path, cloned = resolve_review_target(str(tmp_path))
    assert cloned is None
    assert path == str(tmp_path.resolve())


def test_resolve_review_target_missing():
    with pytest.raises(FileNotFoundError):
        resolve_review_target("/definitely/missing/qt-repo")


def test_clone_repo_git_missing(monkeypatch):
    import shutil

    from code_review_agent.github_utils import clone_repo

    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="git is not installed"):
        clone_repo("https://github.com/owner/repo")


def test_clone_repo_success(monkeypatch, tmp_path):
    import shutil
    import subprocess
    from types import SimpleNamespace

    from code_review_agent.github_utils import clone_repo

    fake_issues = [
        GithubIssue(number=3, title="Fix hacks", body="remove deprecated code")
    ]
    monkeypatch.setattr(
        "code_review_agent.github_utils.fetch_recent_open_issues",
        lambda *a, **k: fake_issues,
    )
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/git")

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "clone"]:
            dest = __import__("pathlib").Path(cmd[-1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "README.md").write_text("ok")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        joined = " ".join(cmd)
        if "rev-parse" in joined and "--abbrev-ref" not in joined:
            return SimpleNamespace(returncode=0, stdout="deadbeef\n", stderr="")
        if "--abbrev-ref" in joined:
            return SimpleNamespace(returncode=0, stdout="main\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    repo = clone_repo(
        "https://github.com/acme/demo", target_dir=str(tmp_path), persist=True
    )
    assert repo.commit_sha == "deadbeef"
    assert repo.repo_name == "acme/demo"
    assert repo.branch == "main"
    assert repo.issues == fake_issues


def test_clone_repo_failure_cleans_dest(monkeypatch, tmp_path):
    import shutil
    import subprocess
    from types import SimpleNamespace

    from code_review_agent.github_utils import clone_repo

    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/git")

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "clone"]:
            dest = __import__("pathlib").Path(cmd[-1])
            dest.mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(returncode=1, stdout="", stderr="denied")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="git clone failed"):
        clone_repo(
            "https://github.com/acme/demo", target_dir=str(tmp_path), persist=True
        )


def test_clone_repo_missing_subpath(monkeypatch, tmp_path):
    import shutil
    import subprocess
    from types import SimpleNamespace

    from code_review_agent.github_utils import clone_repo

    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/git")

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "clone"]:
            dest = __import__("pathlib").Path(cmd[-1])
            dest.mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        joined = " ".join(cmd)
        if "rev-parse" in joined and "--abbrev-ref" not in joined:
            return SimpleNamespace(returncode=0, stdout="abc\n", stderr="")
        if "--abbrev-ref" in joined:
            return SimpleNamespace(returncode=0, stdout="main\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="subpath"):
        clone_repo(
            "https://github.com/acme/demo/tree/main/src",
            target_dir=str(tmp_path),
            persist=True,
        )


def test_clone_repo_injects_github_token(monkeypatch, tmp_path):
    import shutil
    import subprocess
    from types import SimpleNamespace

    from code_review_agent.github_utils import clone_repo

    captured: dict = {}
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/git")
    monkeypatch.setattr(
        "code_review_agent.github_utils.fetch_recent_open_issues",
        lambda *a, **k: [],
    )

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "clone"]:
            captured["url"] = cmd[-2]
            dest = __import__("pathlib").Path(cmd[-1])
            dest.mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        joined = " ".join(cmd)
        if "rev-parse" in joined and "--abbrev-ref" not in joined:
            return SimpleNamespace(returncode=0, stdout="abc\n", stderr="")
        if "--abbrev-ref" in joined:
            return SimpleNamespace(returncode=0, stdout="main\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    clone_repo("https://github.com/acme/demo", target_dir=str(tmp_path), persist=True)
    assert "secret-token@" in captured["url"]


def test_github_repo_name():
    assert github_repo_name("https://github.com/pallets/click") == "pallets/click"
    assert github_repo_name("https://github.com/pallets/click/tree/main/src") == (
        "pallets/click"
    )


def test_clamp_issue_limit():
    assert clamp_issue_limit(None) == 10
    assert clamp_issue_limit(10) == 10
    assert clamp_issue_limit(30) == 30
    assert clamp_issue_limit(99) == 30
    assert clamp_issue_limit(0) == 0
    assert clamp_issue_limit(-3) == 0
    assert clamp_issue_limit("nope") == 10


def test_parse_github_issues_skips_prs_and_empty():
    payload = [
        {
            "number": 2,
            "title": "PR",
            "body": "merge me",
            "pull_request": {"url": "https://api.github.com/repos/a/b/pulls/2"},
            "html_url": "https://github.com/a/b/pull/2",
            "created_at": "2026-01-02T00:00:00Z",
        },
        {
            "number": 1,
            "title": "Bug in CLI",
            "body": "  please fix the hacks  ",
            "html_url": "https://github.com/a/b/issues/1",
            "created_at": "2026-01-01T00:00:00Z",
        },
        {"number": 3, "title": "", "body": "", "html_url": "", "created_at": ""},
        {
            "number": 4,
            "title": "Docs",
            "body": None,
            "html_url": "https://github.com/a/b/issues/4",
            "created_at": "2026-01-04T00:00:00Z",
        },
    ]
    issues = parse_github_issues(payload, limit=10)
    assert [i.number for i in issues] == [1, 4]
    assert issues[0].body == "please fix the hacks"
    snippets = issue_snippets(issues)
    assert snippets[0].startswith("#1 Bug in CLI")
    assert "please fix the hacks" in snippets[0]


def test_parse_github_issues_uses_what_exists():
    payload = [{"number": i, "title": f"Issue {i}", "body": "x"} for i in range(1, 4)]
    assert len(parse_github_issues(payload, limit=10)) == 3
    assert len(parse_github_issues(payload, limit=2)) == 2
    assert parse_github_issues([], limit=10) == []
    assert parse_github_issues("nope", limit=10) == []


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        import json

        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_fetch_recent_open_issues_filters_and_auth(monkeypatch):
    captured: dict = {}
    payload = [
        {
            "number": 9,
            "title": "Open bug",
            "body": "broken",
            "html_url": "https://github.com/acme/demo/issues/9",
            "created_at": "2026-08-01T00:00:00Z",
        },
        {
            "number": 8,
            "title": "A PR",
            "body": "pr",
            "pull_request": {"url": "https://api.github.com/repos/acme/demo/pulls/8"},
        },
    ]

    def fake_urlopen(req, timeout=20):
        captured["url"] = req.full_url
        captured["auth"] = req.get_header("Authorization")
        return _FakeResp(payload)

    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setattr(
        "code_review_agent.github_utils.urllib.request.urlopen", fake_urlopen
    )
    issues = fetch_recent_open_issues("acme/demo", limit=10)
    assert len(issues) == 1
    assert issues[0].number == 9
    assert "state=open" in captured["url"]
    assert "sort=created" in captured["url"]
    assert "per_page=30" in captured["url"]
    assert captured["auth"] == "Bearer tok"


def test_fetch_recent_open_issues_404(monkeypatch):
    import urllib.error

    def fake_urlopen(req, timeout=20):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", hdrs=None, fp=None)

    monkeypatch.setattr(
        "code_review_agent.github_utils.urllib.request.urlopen", fake_urlopen
    )
    assert fetch_recent_open_issues("missing/repo", limit=10) == []


def test_fetch_recent_open_issues_skips_network_when_disabled(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not call GitHub")

    monkeypatch.setattr("code_review_agent.github_utils.urllib.request.urlopen", boom)
    assert fetch_recent_open_issues("acme/demo", limit=0) == []
    assert fetch_recent_open_issues("", limit=10) == []
    assert fetch_recent_open_issues("not-a-repo", limit=10) == []


def test_clone_repo_skips_issue_fetch_when_disabled(monkeypatch, tmp_path):
    import shutil
    import subprocess
    from types import SimpleNamespace

    from code_review_agent.github_utils import clone_repo

    called = {"n": 0}

    def boom(*a, **k):
        called["n"] += 1
        raise AssertionError("should not fetch issues")

    monkeypatch.setattr("code_review_agent.github_utils.fetch_recent_open_issues", boom)
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/git")

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["git", "clone"]:
            dest = __import__("pathlib").Path(cmd[-1])
            dest.mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        joined = " ".join(cmd)
        if "rev-parse" in joined and "--abbrev-ref" not in joined:
            return SimpleNamespace(returncode=0, stdout="abc\n", stderr="")
        if "--abbrev-ref" in joined:
            return SimpleNamespace(returncode=0, stdout="main\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    repo = clone_repo(
        "https://github.com/acme/demo",
        target_dir=str(tmp_path),
        persist=True,
        fetch_issues=False,
    )
    assert called["n"] == 0
    assert repo.issues == []
