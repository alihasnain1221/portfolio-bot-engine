"""Local REPL to chat with the portfolio bot and tune the persona.

Run:  python -m app.cli
      python -m app.cli "can you do AWS infrastructure work?"   # one-shot
"""
from __future__ import annotations

import sys

from .ask import answer, load_bundle


def _one(q: str) -> None:
    res = answer(q)
    print(f"\n{res['answer']}\n")
    print(f"  \033[90m[sources: {', '.join(res['projects_used'][:6])}"
          f"{' …' if len(res['projects_used']) > 6 else ''}  |  {res['model']}]\033[0m")


def main() -> int:
    n = len(load_bundle()["projects"])
    if len(sys.argv) > 1:
        _one(" ".join(sys.argv[1:]))
        return 0
    print(f"Portfolio bot ready ({n} projects loaded). Ask away — Ctrl-C or 'exit' to quit.\n")
    while True:
        try:
            q = input("\033[96myou > \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if q.lower() in ("exit", "quit", ":q"):
            return 0
        if q:
            _one(q)


if __name__ == "__main__":
    raise SystemExit(main())
