"""Scrape the CV PDF into profile.yaml (the CV-derived keys only).

Run:  python -m ingest.scrape_cv          (uses whatever LLM key is available)

Workflow: update the CV PDF, run this, then rebuild the bundle
(`python -m ingest.build_bundle --meta-only`). This keeps the bot able to speak
to CV skills / experience / projects even when ingest never saw a git repo for
them.

It writes ONLY these keys under `profile:` and leaves everything hand-curated
(summary, featured, voice, tech_lead_areas, ...) untouched:
    cv_skills      grouped additional-skills lists
    cv_experience  work-experience entries with quantified highlights
    cv_projects    project list (name/summary/tech/link) as a fallback

Extraction is LLM-based so it survives CV reformatting; nothing is invented
(the model is instructed to copy faithfully from the text).
"""
from __future__ import annotations

import io
import sys

from pypdf import PdfReader
from ruamel.yaml import YAML

from ingest import config
from ingest.llm import get_llm

_SYSTEM = (
    "You extract structured data from a resume/CV. Copy content faithfully from "
    "the provided text. Never invent skills, employers, metrics, projects, or "
    "links that are not present. Return ONLY JSON."
)

_SCHEMA = """\
Extract the CV into JSON with EXACTLY these keys:
{
  "skill_groups": [{"label": string, "skills": [string]}],
  "experience":   [{"company": string, "title": string, "period": string, "highlights": [string]}],
  "projects":     [{"name": string, "summary": string, "tech": [string], "link": string|null}],
  "education":    [string],
  "languages":    [string]
}
Rules:
- skill_groups: mirror the CV's own skill categories and their items verbatim.
- experience.highlights: one entry per bullet, trimmed but faithful (keep metrics).
- projects.summary: the project's description as written; tech = its listed technologies.
- Use null for a missing link. Do not add commentary.
CV TEXT:
"""


def _pdf_text(path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages).strip()


def _yaml() -> YAML:
    y = YAML()                       # round-trip: preserves comments + block scalars
    y.preserve_quotes = True
    y.width = 4096                   # don't hard-wrap long lines
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def scrape() -> int:
    if not config.RESUME_PDF.exists():
        print(f"CV not found at {config.RESUME_PDF} "
              f"(override with PORTFOLIO_RESUME_PDF).", file=sys.stderr)
        return 1

    text = _pdf_text(config.RESUME_PDF)
    if len(text) < 200:
        print(f"Extracted only {len(text)} chars from the PDF; is it text-based "
              f"(not a scan)?", file=sys.stderr)
        return 1

    llm = get_llm()
    print(f"LLM: {llm.provider} / {llm.model}  |  CV chars: {len(text)}")
    data = llm.complete_json(_SYSTEM, _SCHEMA + text, max_tokens=6000, temperature=0.0)

    groups = data.get("skill_groups") or []
    cv_skills = {g["label"]: g["skills"] for g in groups if g.get("label") and g.get("skills")}
    cv_experience = data.get("experience") or []
    cv_projects = data.get("projects") or []

    # Round-trip profile.yaml, replace ONLY the cv_* keys, preserve the rest.
    y = _yaml()
    doc = y.load(config.PROFILE_YAML.read_text(encoding="utf-8"))
    prof = doc["profile"]
    # Assigning existing keys replaces values in place (no reordering / comment
    # drift on re-runs). The keys' one-time placement + note live in profile.yaml.
    prof["cv_skills"] = cv_skills
    prof["cv_experience"] = cv_experience
    prof["cv_projects"] = cv_projects

    buf = io.StringIO()
    y.dump(doc, buf)
    config.PROFILE_YAML.write_text(buf.getvalue(), encoding="utf-8")

    print(f"Updated {config.PROFILE_YAML.name}: "
          f"skill groups={len(cv_skills)}, experience={len(cv_experience)}, "
          f"projects={len(cv_projects)}.")
    print("Next: python -m ingest.build_bundle --meta-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(scrape())
