"""
GitHub repository cloning utilities.
Accepts GitHub URLs, clones to a temp directory, returns the local path.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

GITHUB_ISSUE_LIMIT_DEFAULT = 10
GITHUB_ISSUE_LIMIT_MAX = 30


@dataclass
class GithubIssue:
    number: int
    title: str
    body: str
    html_url: str = ""
    created_at: str = ""

    def snippet(self, max_chars: int = 800) -> str:
        """Single-line title + body for the TD classifier."""
        body = " ".join((self.body or "").split())
        text = f"#{self.number} {self.title}".strip()
        if body:
            text = f"{text}. {body}"
        return text[:max_chars]


@dataclass
class ClonedRepo:
    url: str
    local_path: str  # absolute path to the cloned directory
    repo_name: str  # "owner/repo"
    branch: str | None
    commit_sha: str  # HEAD SHA after clone
    is_temp: bool  # True if we should delete on cleanup
    subpath: str | None = None
    issues: list[GithubIssue] = field(default_factory=list)

    @property
    def review_path(self) -> str:
        """Directory the analysis should run against (clone + optional subpath)."""
        if self.subpath:
            return str(Path(self.local_path) / self.subpath)
        return self.local_path


def is_github_url(path_or_url: str) -> bool:
    """Return True if the string is a GitHub URL (not a local path)."""
    return bool(
        re.match(r"https?://(www\.)?github\.com/", path_or_url.strip())
    ) or bool(re.match(r"git@github\.com:", path_or_url.strip()))


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


def github_repo_name(url: str) -> str:
    """Return ``owner/repo`` from a GitHub URL."""
    clone_url, _, _ = parse_github_url(url)
    match = re.search(r"github\.com[:/](.+?/.+?)(\.git)?$", clone_url)
    return match.group(1) if match else ""


def clamp_issue_limit(limit: int | None) -> int:
    """Default 10 recently opened issues; cap at 30; 0 disables fetch."""
    if limit is None:
        return GITHUB_ISSUE_LIMIT_DEFAULT
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return GITHUB_ISSUE_LIMIT_DEFAULT
    if value <= 0:
        return 0
    return min(value, GITHUB_ISSUE_LIMIT_MAX)


def parse_github_issues(payload: list, *, limit: int) -> list[GithubIssue]:
    """Keep native issues (not PRs). Use however many exist, up to ``limit``."""
    limit = clamp_issue_limit(limit)
    if limit == 0 or not isinstance(payload, list):
        return []
    out: list[GithubIssue] = []
    for item in payload:
        if not isinstance(item, dict) or item.get("pull_request"):
            continue
        title = (item.get("title") or "").strip()
        body = item.get("body")
        body_text = body.strip() if isinstance(body, str) else ""
        if not title and not body_text:
            continue
        try:
            number = int(item.get("number") or 0)
        except (TypeError, ValueError):
            number = 0
        out.append(
            GithubIssue(
                number=number,
                title=title,
                body=body_text,
                html_url=str(item.get("html_url") or ""),
                created_at=str(item.get("created_at") or ""),
            )
        )
        if len(out) >= limit:
            break
    return out


def issue_snippets(issues: list[GithubIssue] | None) -> list[str]:
    if not issues:
        return []
    return [issue.snippet() for issue in issues if issue.snippet()]


def fetch_recent_open_issues(
    repo_name: str,
    *,
    limit: int = GITHUB_ISSUE_LIMIT_DEFAULT,
    timeout: int = 20,
    token: str | None = None,
) -> list[GithubIssue]:
    """Most recently opened issues. Never raises; returns [] if none/unavailable."""
    limit = clamp_issue_limit(limit)
    repo_name = (repo_name or "").strip().strip("/")
    if not limit or not repo_name or "/" not in repo_name:
        return []
    token = token or os.environ.get("GITHUB_TOKEN")
    # Request the cap so PR rows can be skipped and we still fill ``limit``.
    per_page = GITHUB_ISSUE_LIMIT_MAX
    url = (
        f"https://api.github.com/repos/{repo_name}/issues"
        f"?state=open&sort=created&direction=desc&per_page={per_page}"
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "quality-triage",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        try:
            exc.close()
        except Exception:
            pass
        return []
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []
    if not isinstance(payload, list):
        return []
    return parse_github_issues(payload, limit=limit)


def clone_repo(
    url: str,
    target_dir: str | None = None,
    depth: int = 1,
    timeout: int = 120,
    token: str | None = None,
    persist: bool = False,
    fetch_issues: bool = True,
    issue_limit: int = GITHUB_ISSUE_LIMIT_DEFAULT,
) -> ClonedRepo:
    """
    Clone a GitHub repository.

    Args:
        url:        GitHub URL (https or git@)
        target_dir: Parent directory to clone into. If None, uses a temp directory.
        depth:      Git shallow clone depth (1 = latest commit only).
        timeout:    Clone timeout in seconds.
        token:      GitHub personal access token for private repos.
        persist:    If True, skip cleanup (is_temp=False).

    Returns ClonedRepo. `review_path` points at optional /tree/.../subpath.
    """
    if not shutil.which("git"):
        raise RuntimeError("git is not installed or not in PATH")

    clone_url, branch, subpath = parse_github_url(url)

    # Inject token for private repos
    if token:
        clone_url = clone_url.replace("https://", f"https://{token}@")
    else:
        env_token = os.environ.get("GITHUB_TOKEN")
        if env_token:
            clone_url = clone_url.replace("https://", f"https://{env_token}@")

    # Derive repo name from the canonical clone URL (ignores /tree/...)
    m = re.search(r"github\.com[:/](.+?/.+?)(\.git)?$", clone_url)
    repo_name = m.group(1) if m else "unknown/repo"
    repo_slug = repo_name.replace("/", "__")

    is_temp = not persist
    if target_dir is None:
        dest = str(Path(tempfile.mkdtemp(prefix=f"cra_{repo_slug}_")))
    else:
        parent = Path(target_dir)
        parent.mkdir(parents=True, exist_ok=True)
        dest = str(parent / repo_slug)
        if Path(dest).exists():
            shutil.rmtree(dest, ignore_errors=True)

    cmd = ["git", "clone", f"--depth={depth}", "--single-branch"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [clone_url, dest]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError(
            f"git clone failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )

    # Get HEAD SHA
    sha_result = subprocess.run(
        ["git", "-C", dest, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    commit_sha = sha_result.stdout.strip() if sha_result.returncode == 0 else "unknown"

    # Resolve actual branch
    branch_result = subprocess.run(
        ["git", "-C", dest, "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    actual_branch = (
        branch_result.stdout.strip() if branch_result.returncode == 0 else branch
    )

    repo = ClonedRepo(
        url=url,
        local_path=str(Path(dest).resolve()),
        repo_name=repo_name,
        branch=actual_branch,
        commit_sha=commit_sha,
        is_temp=is_temp,
        subpath=subpath,
    )
    if subpath and not Path(repo.review_path).exists():
        if is_temp:
            shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError(f"GitHub subpath does not exist in clone: {subpath}")
    if fetch_issues:
        repo.issues = fetch_recent_open_issues(
            repo.repo_name,
            limit=issue_limit,
            token=token or os.environ.get("GITHUB_TOKEN"),
        )
    return repo


def cleanup_repo(repo: ClonedRepo) -> None:
    """Delete the cloned directory if it was created as a temp directory."""
    if repo.is_temp and Path(repo.local_path).exists():
        shutil.rmtree(repo.local_path, ignore_errors=True)


def resolve_review_target(
    target: str,
    clone_dir: str | None = None,
    depth: int = 1,
    timeout: int = 120,
    persist: bool = False,
    fetch_issues: bool = True,
    issue_limit: int = GITHUB_ISSUE_LIMIT_DEFAULT,
) -> tuple[str, ClonedRepo | None]:
    """Return (local review path, cloned repo or None)."""
    if is_github_url(target):
        cloned = clone_repo(
            target,
            target_dir=clone_dir,
            depth=depth,
            timeout=timeout,
            persist=persist,
            fetch_issues=fetch_issues,
            issue_limit=issue_limit,
        )
        return cloned.review_path, cloned
    path = Path(target)
    if not path.exists():
        raise FileNotFoundError(target)
    return str(path.resolve()), None
