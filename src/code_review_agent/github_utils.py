"""
GitHub repository cloning utilities.
Accepts GitHub URLs, clones to a temp directory, returns the local path.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass
class ClonedRepo:
    url: str
    local_path: str       # absolute path to the cloned directory
    repo_name: str        # "owner/repo"
    branch: str | None
    commit_sha: str       # HEAD SHA after clone
    is_temp: bool         # True if we should delete on cleanup


def is_github_url(path_or_url: str) -> bool:
    """Return True if the string is a GitHub URL (not a local path)."""
    return bool(re.match(r"https?://(www\.)?github\.com/", path_or_url.strip())) or \
           bool(re.match(r"git@github\.com:", path_or_url.strip()))


def parse_github_url(url: str) -> tuple[str, str | None, str | None]:
    """
    Parse a GitHub URL and return (clone_url, branch, subpath).

    Supports:
      https://github.com/owner/repo
      https://github.com/owner/repo.git
      https://github.com/owner/repo/tree/branch
      https://github.com/owner/repo/tree/branch/subdir
      git@github.com:owner/repo.git
    """
    url = url.strip()

    # SSH format
    ssh_match = re.match(r"git@github\.com:([^/]+/[^.]+)(\.git)?$", url)
    if ssh_match:
        return f"https://github.com/{ssh_match.group(1)}.git", None, None

    parsed = urlparse(url)
    if "github.com" not in parsed.netloc:
        raise ValueError(f"Not a GitHub URL: {url}")

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Could not parse owner/repo from: {url}")

    owner, repo = parts[0], parts[1].replace(".git", "")
    clone_url = f"https://github.com/{owner}/{repo}.git"

    branch = None
    subpath = None

    # /tree/branch[/subpath...]
    if len(parts) > 3 and parts[2] == "tree":
        branch = parts[3]
        if len(parts) > 4:
            subpath = "/".join(parts[4:])

    return clone_url, branch, subpath


def clone_repo(
    url: str,
    target_dir: str | None = None,
    depth: int = 1,
    timeout: int = 120,
    token: str | None = None,
) -> ClonedRepo:
    """
    Clone a GitHub repository.

    Args:
        url:        GitHub URL (https or git@)
        target_dir: Where to clone. If None, clones to a temp directory.
        depth:      Git shallow clone depth (1 = latest commit only).
        timeout:    Clone timeout in seconds.
        token:      GitHub personal access token for private repos.

    Returns ClonedRepo with is_temp=True if we created the directory.
    """
    if not shutil.which("git"):
        raise RuntimeError("git is not installed or not in PATH")

    clone_url, branch, _ = parse_github_url(url)

    # Inject token for private repos
    if token:
        clone_url = clone_url.replace("https://", f"https://{token}@")
    else:
        env_token = __import__("os").environ.get("GITHUB_TOKEN")
        if env_token:
            clone_url = clone_url.replace("https://", f"https://{env_token}@")

    # Derive repo name
    m = re.search(r"github\.com[:/](.+?/[^/]+?)(\.git)?$", url)
    repo_name = m.group(1) if m else "unknown/repo"
    repo_slug = repo_name.replace("/", "__")

    is_temp = target_dir is None
    if is_temp:
        target_dir = str(Path(tempfile.mkdtemp(prefix=f"cra_{repo_slug}_")))
    else:
        Path(target_dir).mkdir(parents=True, exist_ok=True)

    cmd = ["git", "clone", f"--depth={depth}", "--single-branch"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [clone_url, target_dir]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        if is_temp:
            shutil.rmtree(target_dir, ignore_errors=True)
        raise RuntimeError(
            f"git clone failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )

    # Get HEAD SHA
    sha_result = subprocess.run(
        ["git", "-C", target_dir, "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=10,
    )
    commit_sha = sha_result.stdout.strip() if sha_result.returncode == 0 else "unknown"

    # Resolve actual branch
    branch_result = subprocess.run(
        ["git", "-C", target_dir, "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, timeout=10,
    )
    actual_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else branch

    return ClonedRepo(
        url=url,
        local_path=str(Path(target_dir).resolve()),
        repo_name=repo_name,
        branch=actual_branch,
        commit_sha=commit_sha,
        is_temp=is_temp,
    )


def cleanup_repo(repo: ClonedRepo) -> None:
    """Delete the cloned directory if it was created as a temp directory."""
    if repo.is_temp and Path(repo.local_path).exists():
        shutil.rmtree(repo.local_path, ignore_errors=True)
