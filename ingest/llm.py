"""Pluggable LLM provider layer.

Supports Anthropic Claude, OpenAI, Groq, and xAI Grok. Auto-detects whichever
API key is present (in the environment or a local .env), so ingestion works
with "whatever is available at the moment". Override the choice with
PORTFOLIO_LLM_PROVIDER=anthropic|openai|groq|xai.

The OpenAI SDK is reused for OpenAI, Groq, and xAI via base_url override;
Anthropic uses its own SDK. SDKs are imported lazily so a missing one never
breaks the others.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (no dependency). Does not overwrite existing env."""
    path = path or (_PROJECT_DIR / ".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


# provider -> (env key, default model, openai-compatible base_url or None)
_PROVIDERS = {
    "anthropic": ("ANTHROPIC_API_KEY", "claude-sonnet-5", None),
    "openai":    ("OPENAI_API_KEY",   "gpt-4o",           None),
    "groq":      ("GROQ_API_KEY",     "llama-3.3-70b-versatile", "https://api.groq.com/openai/v1"),
    "xai":       ("XAI_API_KEY",      "grok-2-latest",    "https://api.x.ai/v1"),
}
# Preference order when PORTFOLIO_LLM_PROVIDER is not set.
_PREFERENCE = ["anthropic", "openai", "groq", "xai"]


@dataclass
class LLM:
    provider: str
    model: str

    def complete(self, system: str, user: str, *,
                 max_tokens: int = 1200, temperature: float = 0.2,
                 as_json: bool = False) -> str:
        """Single-turn completion. Returns the model's text (or JSON string)."""
        env_key, _, base_url = _PROVIDERS[self.provider]
        api_key = os.environ[env_key]

        if self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            kwargs = {}
            if as_json:
                system = system + "\n\nRespond with ONLY valid JSON, no prose."
            msg = client.messages.create(
                model=self.model, max_tokens=max_tokens, temperature=temperature,
                system=system, messages=[{"role": "user", "content": user}], **kwargs,
            )
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

        # OpenAI-compatible (openai, groq, xai)
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        kwargs = {"response_format": {"type": "json_object"}} if as_json else {}
        resp = client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}], **kwargs,
        )
        return resp.choices[0].message.content or ""

    def complete_json(self, system: str, user: str, **kw) -> dict:
        raw = self.complete(system, user, as_json=True, **kw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # tolerate a fenced block or leading prose
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1:
                return json.loads(raw[start:end + 1])
            raise


def available_providers() -> list[str]:
    load_dotenv()
    return [p for p in _PREFERENCE if os.environ.get(_PROVIDERS[p][0])]


def get_llm(provider: str | None = None, model: str | None = None) -> LLM:
    """Pick a provider: explicit arg > PORTFOLIO_LLM_PROVIDER > first available."""
    load_dotenv()
    chosen = provider or os.environ.get("PORTFOLIO_LLM_PROVIDER")
    if chosen:
        chosen = chosen.lower()
        if chosen not in _PROVIDERS:
            raise ValueError(f"Unknown provider '{chosen}'. Options: {list(_PROVIDERS)}")
        if not os.environ.get(_PROVIDERS[chosen][0]):
            raise RuntimeError(f"{chosen} selected but {_PROVIDERS[chosen][0]} is not set.")
    else:
        avail = available_providers()
        if not avail:
            raise RuntimeError(
                "No LLM API key found. Set one of ANTHROPIC_API_KEY / OPENAI_API_KEY / "
                "GROQ_API_KEY / XAI_API_KEY (in the environment or a .env file).")
        chosen = avail[0]
    default_model = _PROVIDERS[chosen][1]
    return LLM(provider=chosen, model=model or os.environ.get("PORTFOLIO_LLM_MODEL") or default_model)
