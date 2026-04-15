"""
Tests for code_review_agent.github_utils
"""

import shutil
from pathlib import Path

import pytest

from code_review_agent.github_utils import (
    ClonedRepo,
    cleanup_repo,
    is_github_url,
    parse_github_url,
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
