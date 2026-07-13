"""Metadata-only ingestion pass.

Walks every repo under CODE_ROOT, computes authorship + stack, and writes a
raw card set to out/projects_raw.json (full, with _internal grounding).
No LLM yet -- this is the eyeball-the-data checkpoint. LLM card generation
(summary_public / workflow_brief) is layered on in a later step.

Run:  python -m ingest.run_ingest
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from . import config, git_scan, stack_detect


def _slug_to_name(slug: str) -> str:
    tail = slug.split("/")[-1]
    return tail.replace("-", " ").replace("_", " ").strip()


def build_card(repo: Path) -> dict:
    auth = git_scan.scan_repo(repo)
    stk = stack_detect.detect(repo)
    return {
        "slug": auth.slug,
        "name": _slug_to_name(auth.slug),
        "role": auth.role,
        "authorship": auth.public_authorship(),
        "stack": {
            "languages": stk.languages,
            "frameworks": stk.frameworks,
            "testing": stk.testing,
            "iac": stk.iac,
            "tags": stk.tags,
        },
        "signals": stk.signals,
        "featured_rank": None,
        "disclosure_tier": config.DEFAULT_DISCLOSURE_TIER,
        "summary_public": None,       # filled by LLM step
        "workflow_brief": None,       # filled by LLM step
        "skills_evidenced": [],       # filled by LLM step
        "_internal": {
            "path": auth.path,
            "readme_excerpt": stk.readme_text,
            "key_files": stk.key_files,
            "identities_me": auth.identities_me,
            "top_others": auth.top_others,
        },
    }


def main() -> int:
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    repos = git_scan.discover_repos()
    print(f"Discovered {len(repos)} repos under {config.CODE_ROOT}\n")

    cards: list[dict] = []
    t0 = time.time()
    for i, repo in enumerate(repos, 1):
        try:
            card = build_card(repo)
        except Exception as e:  # keep going; report at the end
            print(f"  [{i:>3}/{len(repos)}] !! {repo}: {e}")
            continue
        cards.append(card)
        a = card["authorship"]
        print(f"  [{i:>3}/{len(repos)}] {card['role']:<11} "
              f"{a['share']*100:5.1f}%  ({a['your_commits']}/{a['total']})  {card['slug']}")

    # Sort: most-owned first (role rank, then commits).
    role_rank = {"solo": 0, "lead": 1, "contributor": 2, "minor": 3,
                 "snapshot": 4, "exposure": 5}
    cards.sort(key=lambda c: (role_rank.get(c["role"], 9), -c["authorship"]["your_commits"]))

    out = config.OUT_DIR / "projects_raw.json"
    out.write_text(json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8")

    mine = [c for c in cards if c["authorship"]["your_commits"] > 0]
    print(f"\nDone in {time.time()-t0:.0f}s. {len(cards)} repos, "
          f"{len(mine)} with your commits.")
    print(f"Wrote {out}")

    # quick rollups
    def n(role): return sum(1 for c in cards if c["role"] == role)
    print(f"  roles: solo={n('solo')} lead={n('lead')} "
          f"contributor={n('contributor')} minor={n('minor')} "
          f"snapshot={n('snapshot')} exposure={n('exposure')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
