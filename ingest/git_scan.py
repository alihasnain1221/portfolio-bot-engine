"""Discover git repos under CODE_ROOT and compute authorship for each.

Uses git as the lens (git log --all), which automatically ignores
node_modules/venv/dist. Emits, per repo: commit share, role, first/last dates
of the user's work, authored LOC, and (internal) the top collaborators.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .identities import is_me, is_noise

# Record/field separators for robust --format parsing (won't appear in names).
_RS = "\x1e"
_FS = "\x1f"

# Dirs to prune during traversal (heavy/generated; avoids long broken paths).
_PRUNE_DIRS = {"node_modules", ".terraform", "dist", "build", ".next",
               "venv", ".venv", "__pycache__", "vendor", "coverage"}


def discover_repos(root: Path | None = None) -> list[Path]:
    """Return repo root dirs (dirs containing a .git), pruning node_modules and
    the configured excludes DURING traversal so we never descend into them."""
    root = root or config.CODE_ROOT
    repos: list[Path] = []
    for dirpath, dirnames, _ in os.walk(root, onerror=lambda e: None):
        dirnames[:] = [d for d in dirnames if d not in _PRUNE_DIRS]
        if ".git" in dirnames:
            repo = Path(dirpath)
            rel = str(repo).replace("\\", "/")
            if not any(sub in rel for sub in config.EXCLUDE_REPO_SUBSTRINGS):
                repos.append(repo)
            dirnames.remove(".git")   # don't descend into git internals
    return sorted(set(repos))


def _overrides_for(repo: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(extra_name_fragments, extra_email_fragments) for this repo, if any."""
    rel = str(repo).replace("\\", "/")
    names: list[str] = []
    emails: list[str] = []
    for key, ov in config.REPO_IDENTITY_OVERRIDES.items():
        if key in rel:
            names.extend(ov.get("name_fragments", ()))
            emails.extend(ov.get("email_fragments", ()))
    return tuple(names), tuple(emails)


@dataclass
class Authorship:
    slug: str
    path: str
    commits_me: int = 0
    commits_total: int = 0            # excludes bots/shared noise
    loc_me_added: int = 0
    loc_total_added: int = 0
    first_me: str | None = None       # ISO date of my earliest commit
    last_me: str | None = None        # ISO date of my latest commit
    role: str = "exposure"
    identities_me: dict[str, int] = field(default_factory=dict)   # "name <email>" -> commits
    top_others: list[tuple[str, int]] = field(default_factory=list)

    @property
    def share(self) -> float:
        return (self.commits_me / self.commits_total) if self.commits_total else 0.0

    def public_authorship(self) -> dict:
        return {
            "your_commits": self.commits_me,
            "total": self.commits_total,
            "share": round(self.share, 3),
            "first": self.first_me,
            "last": self.last_me,
            "loc_you": self.loc_me_added,
        }


def _role_for(share: float, commits_me: int, commits_total: int) -> str:
    """Volume-aware role. A 1-commit repo must not read as "solo owner": ownership
    roles require a minimum real commit count, and near-empty histories get the
    honest 'snapshot' tag instead of an inflated share."""
    if commits_me == 0:
        return "exposure"
    if commits_total <= 2:
        return "snapshot"          # import / single-commit; too little to characterize
    if share >= 0.90 and commits_me >= 5:
        return "solo"
    if share >= 0.50 and commits_me >= 3:
        return "lead"
    if share >= 0.15:
        return "contributor"
    return "minor"


def _git_log(repo: Path) -> str:
    fmt = f"{_RS}%an{_FS}%ae{_FS}%aI"
    cmd = [
        "git", "-C", str(repo), "log", "--all", "--no-merges",
        "--numstat", f"--format={fmt}",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=300)
    return proc.stdout.decode("utf-8", errors="replace")


def scan_repo(repo: Path) -> Authorship:
    slug = str(repo.relative_to(config.CODE_ROOT)).replace("\\", "/")
    a = Authorship(slug=slug, path=str(repo))
    x_names, x_emails = _overrides_for(repo)

    others: dict[str, int] = {}
    cur_me = False
    cur_key = ""

    for line in _git_log(repo).split("\n"):
        if line.startswith(_RS):
            name, email, date = (line[1:].split(_FS) + ["", "", ""])[:3]
            date = date[:10]  # YYYY-MM-DD
            if is_noise(name, email):
                cur_me = False
                cur_key = ""
                continue
            a.commits_total += 1
            cur_me = is_me(name, email, x_names, x_emails)
            cur_key = f"{name} <{email}>"
            if cur_me:
                a.commits_me += 1
                a.identities_me[cur_key] = a.identities_me.get(cur_key, 0) + 1
                if date:
                    if a.first_me is None or date < a.first_me:
                        a.first_me = date
                    if a.last_me is None or date > a.last_me:
                        a.last_me = date
            else:
                others[cur_key] = others.get(cur_key, 0) + 1
        elif line.strip() and cur_key:
            # numstat line: "<adds>\t<dels>\t<path>"  ("-" for binary files)
            cols = line.split("\t")
            if len(cols) >= 1 and cols[0].isdigit():
                adds = int(cols[0])
                a.loc_total_added += adds
                if cur_me:
                    a.loc_me_added += adds

    a.role = _role_for(a.share, a.commits_me, a.commits_total)
    a.top_others = sorted(others.items(), key=lambda kv: -kv[1])[:8]
    return a
