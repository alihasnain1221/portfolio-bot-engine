"""Merge profile.yaml + raw cards, LLM-generate summaries at the correct
disclosure altitude, and emit the deploy-safe bundle.

Outputs (in out/):
  - projects.json  : full cards incl. _internal grounding   -> LOCAL ONLY
  - skills.json    : skill -> weighted project evidence
  - profile.json   : DEPLOY-SAFE bundle (public fields only; no _internal,
                     no README text, no key files)          -> deployed

Grounding policy (confirmed with user):
  - Featured projects: summary_public = the vetted profile.yaml narrative.
    No repo text is sent to the LLM; workflow_brief/skills are grounded only in
    the narrative + declared tech + lead_focus.
  - Non-featured meaningful repos (role solo/lead/contributor): the LLM also
    sees a secret-stripped README excerpt for grounding.

Run:  python -m ingest.build_bundle
"""
from __future__ import annotations

import json
import re
import sys
import time

import yaml

from . import config, git_scan
from .llm import get_llm

ROLE_WEIGHT = {"solo": 1.0, "lead": 0.85, "contributor": 0.55,
               "minor": 0.25, "snapshot": 0.1, "exposure": 0.0}

# Roles worth auto-including in the public bundle when NOT featured.
_INCLUDE_ROLES = {"solo", "lead", "contributor"}

# ---------------------------------------------------------------------------
# Secret stripping (README excerpts before any egress to the LLM)
# ---------------------------------------------------------------------------
_SECRET_LINE = re.compile(
    r"(?i)(api[_-]?key|secret|token|password|passwd|pwd|access[_-]?key|"
    r"private[_-]?key|client[_-]?secret|bearer|authorization|sheet\s*id|"
    r"connection[_-]?string|mongodb\+srv|postgres://|mysql://)")
_ASSIGN = re.compile(r"""(?i)([A-Z0-9_]{3,})\s*[:=]\s*['"]?([A-Za-z0-9_\-./+=]{16,})['"]?""")
_LONGTOKEN = re.compile(r"\b(AKIA[0-9A-Z]{12,}|[A-Za-z0-9_\-]{28,})\b")
_URL_CREDS = re.compile(r"://[^/\s:@]+:[^/\s@]+@")


def strip_secrets(text: str) -> str:
    out = []
    for line in (text or "").splitlines():
        if _SECRET_LINE.search(line):
            out.append("[redacted]")
            continue
        line = _ASSIGN.sub(r"\1=[redacted]", line)
        line = _URL_CREDS.sub("://[redacted]@", line)
        line = _LONGTOKEN.sub("[redacted]", line)
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# LLM card generation
# ---------------------------------------------------------------------------
_SYSTEM = """You write concise, FACTUAL project cards for a software engineer's \
portfolio chatbot. The engineer is a Senior Software & AI Engineer / Technical Lead.

Hard rules:
1. GROUND ONLY in the evidence provided. Never invent features, clients, metrics,
   dates, or technologies. If evidence is thin, keep it general -- do not guess.
2. Respect the disclosure tier:
   - mention-only: refer to it only as "a <domain> platform"; no name, no specifics.
   - summary: one factual paragraph; you may name it and list tech categories.
   - open: fuller detail is allowed.
3. NEVER output secrets, credentials, file paths, internal URLs, or client PII.
4. Be HONEST about role. If role is contributor/minor, do NOT imply sole authorship
   ("contributed to", "worked on"), never "built" or "led".
5. When role is lead/solo, surface the relevant technical-leadership / ownership
   angles from lead_focus (architecture, IaC, CI/CD & testing, AI pipelines,
   observability, DB design, team leadership) -- but only those supported by evidence.
6. Neutral third-person-free factual prose (the chat persona/voice is applied later).

Return ONLY JSON:
{"domain": "<short domain, e.g. fintech / healthcare / sports / devtools>",
 "summary_public": "<one paragraph at the disclosure altitude>",
 "workflow_brief": "<2-3 sentence concept-level 'how it works'; empty string if mention-only>",
 "skills_evidenced": ["<skill>", ...]}"""


def _evidence_block(card: dict, *, narrative: str | None, lead_focus: list[str],
                    tier: str, readme: str | None) -> str:
    s = card["stack"]
    a = card.get("authorship")
    role_line = (f"role: {card['role']} (self-reported; no local git to verify)"
                 if not a else
                 f"role: {card['role']} (your commit share {a['share']*100:.0f}%, "
                 f"{a['your_commits']}/{a['total']} commits)")
    parts = [
        f"disclosure_tier: {tier}",
        f"name: {card['name']}",
        role_line,
        f"languages: {', '.join(s['languages']) or 'n/a'}",
        f"frameworks: {', '.join(s['frameworks']) or 'n/a'}",
        f"testing: {', '.join(s['testing']) or 'n/a'}",
        f"iac: {', '.join(s['iac']) or 'n/a'}",
        f"tags: {', '.join(s['tags']) or 'n/a'}",
    ]
    if lead_focus:
        parts.append("lead_focus (tech-lead angles to surface): " + "; ".join(lead_focus))
    if narrative:
        parts.append("AUTHORITATIVE narrative (use as the basis for summary_public):\n" + narrative)
    if readme:
        parts.append("README excerpt (grounding; secret-stripped):\n" + readme[:3500])
    return "\n".join(parts)


def _gen_fields(llm, evidence: str) -> dict:
    try:
        data = llm.complete_json(_SYSTEM, evidence, max_tokens=900)
        return {
            "domain": str(data.get("domain", "")).strip(),
            "summary_public": str(data.get("summary_public", "")).strip(),
            "workflow_brief": str(data.get("workflow_brief", "")).strip(),
            "skills_evidenced": [str(x).strip() for x in data.get("skills_evidenced", []) if str(x).strip()],
        }
    except Exception as e:
        return {"domain": "", "summary_public": "", "workflow_brief": "",
                "skills_evidenced": [], "_gen_error": str(e)}


# ---------------------------------------------------------------------------
# Featured aggregation (a featured project can span several repos)
# ---------------------------------------------------------------------------
def _aggregate(cards: list[dict]) -> dict:
    me = sum(c["authorship"]["your_commits"] for c in cards)
    tot = sum(c["authorship"]["total"] for c in cards)
    loc = sum(c["authorship"]["loc_you"] for c in cards)
    firsts = [c["authorship"]["first"] for c in cards if c["authorship"]["first"]]
    lasts = [c["authorship"]["last"] for c in cards if c["authorship"]["last"]]
    share = (me / tot) if tot else 0.0

    def uni(key):
        out = []
        for c in cards:
            for v in c["stack"][key]:
                if v not in out:
                    out.append(v)
        return sorted(out)

    signals = {}
    for c in cards:
        for k, v in c["signals"].items():
            signals[k] = signals.get(k, False) or bool(v)
    return {
        "role": git_scan._role_for(share, me, tot),
        "authorship": {"your_commits": me, "total": tot, "share": round(share, 3),
                       "first": min(firsts) if firsts else None,
                       "last": max(lasts) if lasts else None, "loc_you": loc},
        "stack": {"languages": uni("languages"), "frameworks": uni("frameworks"),
                  "testing": uni("testing"), "iac": uni("iac"), "tags": uni("tags")},
        "signals": signals,
        "_internal": {
            "repos": [c["slug"] for c in cards],
            "readme_excerpt": next((c["_internal"]["readme_excerpt"]
                                    for c in cards if c["_internal"]["readme_excerpt"]), ""),
            "key_files": [f for c in cards for f in c["_internal"]["key_files"]][:15],
        },
    }


def _public_view(card: dict) -> dict:
    """Deploy-safe projection: drop _internal and any raw text."""
    return {k: v for k, v in card.items() if k != "_internal"}


def refresh_meta() -> int:
    """Fast path: re-emit profile.json from existing cards + current profile.yaml
    profile meta (cta, emerging_skills, headline, ...). No LLM calls."""
    profile = yaml.safe_load(config.PROFILE_YAML.read_text(encoding="utf-8"))
    full = json.load(open(config.OUT_DIR / "projects.json", encoding="utf-8"))
    skills = json.load(open(config.OUT_DIR / "skills.json", encoding="utf-8"))
    bundle = {
        "profile": {k: v for k, v in profile["profile"].items()},
        "projects": [_public_view(c) for c in full],
        "skills": skills,
        "generated_with": "meta-refresh",
    }
    (config.OUT_DIR / "profile.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Refreshed out/profile.json profile-meta (no LLM). "
          f"projects={len(bundle['projects'])} skills={len(skills)}")
    return 0


def main() -> int:
    if "--meta-only" in sys.argv:
        return refresh_meta()
    llm = get_llm()
    print(f"LLM: {llm.provider} / {llm.model}")
    profile = yaml.safe_load((config.PROFILE_YAML).read_text(encoding="utf-8"))
    raw = json.load(open(config.OUT_DIR / "projects_raw.json", encoding="utf-8"))
    by_slug = {c["slug"]: c for c in raw}

    featured_cards: list[dict] = []
    used_slugs: set[str] = set()
    t0 = time.time()

    # ---- Featured (curated) ----
    for f in profile["featured"]:
        member_slugs = [s for s in f.get("repos", []) if s in by_slug]
        used_slugs.update(member_slugs)
        members = [by_slug[s] for s in member_slugs]
        agg = _aggregate(members) if members else {
            "role": "exposure",
            "authorship": {"your_commits": 0, "total": 0, "share": 0.0,
                           "first": None, "last": None, "loc_you": 0},
            "stack": {"languages": [], "frameworks": [], "testing": [], "iac": [], "tags": []},
            "signals": {}, "_internal": {"repos": [], "readme_excerpt": "", "key_files": []},
        }
        # honest role override (e.g. BuzzMi -> contributor)
        resume_only = bool(f.get("resume_only"))
        if resume_only:
            role = f.get("role_override", "self-reported")   # no local git to verify
        else:
            role = f.get("role_override", agg["role"])
        tier = f.get("disclosure_tier", config.DEFAULT_DISCLOSURE_TIER)
        card = {
            "key": f["key"], "slug": member_slugs[0] if member_slugs else f["key"],
            "name": f["name"], "role": role,
            "self_reported": resume_only,
            "authorship": None if resume_only else agg["authorship"],
            "stack": {**agg["stack"]},
            "signals": agg["signals"], "featured_rank": f["rank"],
            "disclosure_tier": tier, "link": f.get("link"),
            "ai_showcase": bool(f.get("ai_showcase")),
            "source": f.get("source", "resume"),
            "_internal": agg["_internal"],
        }
        # merge declared tech into frameworks/tags for visibility
        for t in f.get("tech", []):
            if t not in card["stack"]["frameworks"] and t not in card["stack"]["tags"]:
                card["stack"]["tags"].append(t)
        # featured never send repo text; ground in the vetted narrative only
        ev = _evidence_block(card, narrative=f.get("narrative"),
                             lead_focus=f.get("lead_focus", []), tier=tier, readme=None)
        gen = _gen_fields(llm, ev)
        # summary_public is authoritative from yaml (fallback to gen if missing)
        card["summary_public"] = " ".join((f.get("narrative") or gen["summary_public"]).split())
        card["workflow_brief"] = gen["workflow_brief"]
        card["domain"] = gen["domain"]
        # include curated declared tech so it's a scannable skill, not just
        # buried in the narrative (helps terse "what about X?" questions land).
        card["skills_evidenced"] = sorted(set(gen["skills_evidenced"]) |
                                          set(card["stack"]["frameworks"]) |
                                          set(f.get("tech", [])))
        featured_cards.append(card)
        print(f"  featured  #{f['rank']:>2} {role:<11} {f['name']}")

    # ---- Non-featured meaningful repos ----
    other_cards: list[dict] = []
    for c in raw:
        if c["slug"] in used_slugs or c["role"] not in _INCLUDE_ROLES:
            continue
        tier = config.DEFAULT_DISCLOSURE_TIER
        readme = strip_secrets(c["_internal"]["readme_excerpt"])
        ev = _evidence_block(c, narrative=None, lead_focus=[], tier=tier, readme=readme)
        gen = _gen_fields(llm, ev)
        c = dict(c)
        c["disclosure_tier"] = tier
        c["summary_public"] = gen["summary_public"]
        c["workflow_brief"] = gen["workflow_brief"]
        c["domain"] = gen["domain"]
        c["skills_evidenced"] = sorted(set(gen["skills_evidenced"]) | set(c["stack"]["frameworks"]))
        other_cards.append(c)
        print(f"  other        {c['role']:<11} {c['slug']}")

    all_cards = featured_cards + other_cards

    # ---- skills.json ----
    skills: dict[str, list] = {}
    for c in all_cards:
        w = ROLE_WEIGHT.get(c["role"], 0.0)
        for sk in c.get("skills_evidenced", []):
            skills.setdefault(sk, []).append(
                {"project": c.get("key", c["slug"]), "role": c["role"], "weight": round(w, 2)})
    skills_sorted = dict(sorted(skills.items(),
                                key=lambda kv: -sum(e["weight"] for e in kv[1])))

    # ---- write outputs ----
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUT_DIR / "projects.json").write_text(
        json.dumps(all_cards, indent=2, ensure_ascii=False), encoding="utf-8")
    (config.OUT_DIR / "skills.json").write_text(
        json.dumps(skills_sorted, indent=2, ensure_ascii=False), encoding="utf-8")

    bundle = {
        "profile": {k: v for k, v in profile["profile"].items()},
        "projects": [_public_view(c) for c in all_cards],
        "skills": skills_sorted,
        "generated_with": f"{llm.provider}/{llm.model}",
    }
    (config.OUT_DIR / "profile.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")

    errs = [c for c in all_cards if c.get("_gen_error")]
    print(f"\nDone in {time.time()-t0:.0f}s. featured={len(featured_cards)} "
          f"other={len(other_cards)} skills={len(skills_sorted)}")
    if errs:
        print(f"  !! {len(errs)} cards had generation errors")
    print("  wrote out/projects.json, out/skills.json, out/profile.json (deploy-safe)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
