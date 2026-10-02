"""Provenance fields for results/ and derived files (git commit + dirty flag, UTC time, command line).

"dirty" means modified tracked files outside results/; outputs (untracked, or tracked artifacts under
results/ being regenerated) do not count (W1-B09). ``code_sha256`` hashes the script and B's packages
(wuji_cap/common, wuji_cap/sim), so an artifact names the exact code even when made from a dirty tree.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def git_commit(repo: Path = REPO_ROOT) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"))}


CODE_DIRS = ("wuji_cap/common", "wuji_cap/sim")


def code_sha256(script: str) -> dict:
    files = [REPO_ROOT / script] + sorted(f for d in CODE_DIRS for f in (REPO_ROOT / d).glob("*.py"))
    return {str(f.relative_to(REPO_ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()}


def provenance(script: str) -> dict:
    return {
        "script": script,
        "command": " ".join(sys.argv),
        "git": git_commit(),
        "code_sha256": code_sha256(script),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
