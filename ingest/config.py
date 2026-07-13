"""Central configuration for the portfolio-chatbot ingestion pipeline.

Everything here is LOCAL-only. Nothing in this module is deployed; the
deploy-safe bundle is produced by run_ingest.py (profile.json).
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
# Root that holds all the git repos to scan (~112 repos as of the scan).
CODE_ROOT = Path(os.environ.get("PORTFOLIO_CODE_ROOT", r"E:\React-JS"))

PROJECT_DIR = Path(__file__).resolve().parent.parent          # portfolio-chatbot/
DATA_DIR = PROJECT_DIR / "data"
OUT_DIR = PROJECT_DIR / "out"                                 # generated artifacts
PROFILE_YAML = DATA_DIR / "profile.yaml"                      # curation (featured/tiers/narratives)
RESUME_PDF = Path(os.environ.get("PORTFOLIO_RESUME_PDF",       # source CV for scrape_cv.py
                                 str(PROJECT_DIR / "Ali_s Resume.pdf")))

# Repos to skip entirely (vendored third-party history, not the user's work).
EXCLUDE_REPO_SUBSTRINGS = (
    "chromiumLayers/chromium",   # vendored Chromium build layer
)

# --- Per-repo identity overrides -------------------------------------------
# Identity is mostly global (see identities.py), but some projects the user
# LED or OWNS were committed largely under a collaborator's name. For those,
# list extra name/email fragments that count as "me" FOR THAT REPO ONLY.
# Key = substring matched against the repo's path under CODE_ROOT.
# Extend this freely as more such projects surface.
_AMJAD = {
    "name_fragments": ("amjad", "islamamjad"),
    "email_fragments": (
        "amjadislam",                       # matches amjadislamkhan@ and amjadislamkhn@
        "amjad@totalpresent.com",
        "amjad@pinpointworks.com",
        "amjad.khan@cooperativecomputing.com",
    ),
}
_NAJAM = {
    "name_fragments": ("najam",),
    "email_fragments": (
        "najam.bashir@diginatives.io",
        "najambashir1@gmail.com",
        "najambashir@rapidzzsolutions.com",
    ),
}
_FURQAN = {  # throwaway identity on the user's own take-home
    "name_fragments": ("furqan",),
    "email_fragments": ("furqan.anwar@mail.com", "choudharyfurqan0@gmail.com"),
}
REPO_IDENTITY_OVERRIDES = {
    "Buzz-mi": _AMJAD,                 # BuzzMi:   Amjad Islam's work here is the user's
    "Trove": _AMJAD,                   # Trove:    Amjad Islam's work here is the user's
    "CastingPAX": _NAJAM,              # CastingPAX: Najam Bashir's work here is the user's
    "llm-knowledge-extractor": _FURQAN,  # user's take-home under a throwaway identity
}


# Role (solo/lead/contributor/minor/snapshot/exposure) is derived volume-aware
# in git_scan._role_for -- share alone is not enough (a 1-commit repo is not
# "solo ownership").

# --- Disclosure tiers ------------------------------------------------------
# mention-only : refer to it only as "a <domain> platform"
# summary      : may name it + one-paragraph general summary + tech categories (DEFAULT)
# open         : full detail (own / open-source projects only)
DEFAULT_DISCLOSURE_TIER = "summary"

# README byte cap read into internal grounding (never deployed raw).
README_MAX_BYTES = 8000
