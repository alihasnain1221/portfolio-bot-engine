"""Identity resolution: does a given git (name, email) count as "me"?

Confirmed from the author-clustering scan (2026-07-11):
  ME  = Ali Hasnain core  +  Uzair Rahim  +  Abdullah Akbar
  NOT = Najam Bashir, Amjad Islam (collaborators); "Ali Hassan" (different
        person); bots and shared repo accounts.

Design: a pure ALLOWLIST of specific email fragments / name+email pairs, plus a
bot/noise filter. No denylist -- the allowlist is specific (e.g. "ali.hasnain@"
never matches "ali.hassan@"), so a denylist would be redundant maintenance
surface as the code dir grows.

Some projects the user led were committed under a collaborator's name; those are
handled by PER-REPO overrides (config.REPO_IDENTITY_OVERRIDES), passed in here
as extra_name_fragments / extra_email_fragments by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass


# Email fragments that, on their own, identify one of "me".
_ME_EMAIL_FRAGMENTS = (
    # --- Ali Hasnain (core) ---
    "ali.hasnain@rapidzzsolutions.com",
    "ali-husnain@rapidzz",            # rapidzzl.com / rapidzzsol.com / rapidzz.com
    "ali-hasnain-rapidzz@rapidzz",
    "ali-husnain-rapidzz@rapidzz",
    "165882691+ali-hasnain-rapidzz",  # GitHub noreply
    "ali-hasnain-rapidzz@github.com",
    "ali.hasnain.nextek@gmail.com",
    "alibutthasnain@gmail.com",       # personal / resume contact
    "alihasnain1221",
    "ali@spinsports.ai",
    "ali@spninc.ai",
    # --- Uzair Rahim ---
    "uxairahim@gmail.com",
    "uzair.rapidzz@gmail.com",
    "uzair_dev@rapidzzsolutions.com",
    "uzair@rapidzz.com",
    "uzair@intelliscence.com",
    "uzair.raheem987@gmail.com",
    "153824630+uxairrahim",
    # --- Abdullah Akbar ---
    "muhammedabdullahakbar@gmail.com",
    # --- Team output the user claims as their own (role tags keep it honest) ---
    "jeff@spninc.ai",                 # Jeff Marston
    "jeff@spinsports.ai",             # Jeff (jeff-spin2)
    "bax@spninc.ai",                  # Rob / Robby Bax
    "sokainakanwal@rapidzzsolutions.com",  # Sokaina Kanwal
)

# Name+email pairs to force-attribute to me even though the email alone is
# ambiguous (Ali committed on a teammate's machine). Kept tight and explicit.
_ME_NAME_EMAIL_PAIRS = {
    ("ali-hasnain-rapidzz", "hassam.ullah@rapidzzsolutions.com"),
    ("ali-hasnain-rapidzz", "akbarsafiullah32@gmail.com"),
    ("ali-hasnain-rapidzz", "amjadislamkhan@gmail.com"),
}

# Names that are always bots/noise regardless of email.
_NOISE_NAME_FRAGMENTS = ("github-actions", "dependabot", "system administrator")


@dataclass(frozen=True)
class Author:
    name: str
    email: str

    @property
    def n(self) -> str:
        return (self.name or "").strip().lower()

    @property
    def e(self) -> str:
        return (self.email or "").strip().lower()


def is_me(
    name: str,
    email: str,
    extra_name_fragments: tuple[str, ...] = (),
    extra_email_fragments: tuple[str, ...] = (),
) -> bool:
    """True if this git author should be attributed to the user.

    extra_*_fragments come from a per-repo override (config.REPO_IDENTITY_OVERRIDES)
    and only apply for the repo the caller is currently scanning.
    """
    a = Author(name, email)

    if any(frag in a.n for frag in _NOISE_NAME_FRAGMENTS):
        return False
    if (a.n, a.e) in _ME_NAME_EMAIL_PAIRS:
        return True
    if any(frag in a.e for frag in _ME_EMAIL_FRAGMENTS):
        return True
    # Per-repo overrides (e.g. Amjad Islam counts as me inside BuzzMi).
    if any(frag in a.e for frag in extra_email_fragments):
        return True
    if any(frag in a.n for frag in extra_name_fragments):
        return True
    return False


def is_noise(name: str, email: str) -> bool:
    """Bots / shared accounts that should not count toward repo totals."""
    a = Author(name, email)
    if any(frag in a.n for frag in _NOISE_NAME_FRAGMENTS):
        return True
    if a.e == "repos@rapidzzsolutions.com":
        return True
    if a.e.startswith("root@"):
        return True
    return False
