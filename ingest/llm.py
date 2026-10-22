"""Pluggable LLM provider layer.

Supports Anthropic Claude, OpenAI, Groq, xAI Grok, and Google Gemini.
Auto-detects whichever API key is present (in the environment or a local .env),
so ingestion works with "whatever is available at the moment". Override the
choice with PORTFOLIO_LLM_PROVIDER=anthropic|openai|groq|xai|gemini.

The OpenAI SDK is reused for OpenAI, Groq, xAI, and Gemini via base_url
override; Anthropic uses its own SDK. SDKs are imported lazily so a missing one never
breaks the others.
"""
from __future__ import annotations

import json
import os
import time
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
    "anthropic": ("ANTHROPIC_API_KEY", "claude-opus-5-5", None),
    "openai":    ("OPENAI_API_KEY",   "gpt-4o",           None),
    "groq":      ("GROQ_API_KEY",     "llama-3.3-70b-versatile", "https://api.groq.com/openai/v1"),
    "xai":       ("XAI_API_KEY",      "grok-2-latest",    "https://api.x.ai/v1"),
    "gemini":    ("GEMINI_API_KEY",   "gemini-3.5-flash-lite", "https://generativelanguage.googleapis.com/v1beta/openai/"),
}
# Preference order when PORTFOLIO_LLM_PROVIDER is not set.
_PREFERENCE = ["anthropic", "openai", "groq", "xai", "gemini"]

# Claude models that accept server-side refusal fallbacks ("default" routing).
_CLAUDE_FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")
# Thinking tokens count against max_tokens, so callers' answer budgets get this on top.
_THINKING_HEADROOM = 8000
# Tried after the configured Gemini model; each has its own free-tier quota
# (flash-lite: 500 requests/day vs 20 for full flash).
_GEMINI_FALLBACK_MODELS = ("gemini-3.1-flash-lite",)
# A model that just returned 5xx/429 or timed out is skipped for this long, so
# later requests go straight to a healthy one. Kept per process (resets on restart).
_COOLDOWN_SEC = 300
_down_until: dict[str, float] = {}


@dataclass
class LLM:
    provider: str
    model: str
    served_model: str | None = None   # model that answered the last call

    def complete(self, system: str, user: str, *,
                 max_tokens: int = 1200, temperature: float = 0.2,
                 as_json: bool = False, effort: str | None = None,
                 timeout: float = 120) -> str:
        """Single-turn completion. Returns the model's text (or JSON string).

        `temperature` applies to OpenAI-compatible providers only (current Claude
        models reject it). `effort` applies to Claude (low|medium|high|xhigh|max)
        and Gemini (minimal|low|medium|high, sent as reasoning_effort).
        `timeout` (seconds, per attempt) applies to OpenAI-compatible providers.
        """
        env_key, _, base_url = _PROVIDERS[self.provider]
        api_key = os.environ[env_key]
        effort = os.environ.get("PORTFOLIO_LLM_EFFORT") or effort

        if self.provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            if as_json:
                system = system + "\n\nRespond with ONLY valid JSON, no prose."
            kwargs = {}
            if effort:
                kwargs["output_config"] = {"effort": effort}
            if self.model.startswith(_CLAUDE_FALLBACK_MODELS):
                kwargs["betas"] = ["server-side-fallback-2026-07-01"]
                kwargs["fallbacks"] = "default"
            msg = client.beta.messages.create(
                model=self.model, max_tokens=max_tokens + _THINKING_HEADROOM,
                system=system, messages=[{"role": "user", "content": user}], **kwargs,
            )
            if msg.stop_reason == "refusal":
                raise RuntimeError(f"Claude declined the request ({getattr(msg.stop_details, 'category', None)})")
            self.served_model = msg.model
            return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

        # OpenAI-compatible (openai, groq, xai, gemini)
        import openai
        kwargs = {}
        models = [self.model]
        if self.provider == "gemini":
            # The fallback chain replaces SDK retries, which would only burn the
            # free-tier quota re-hitting a model that's overloaded or rate-limited.
            client = openai.OpenAI(api_key=api_key, base_url=base_url,
                                   timeout=timeout, max_retries=0)
            # Gemini 3.x thinking can't be disabled and its compat layer doesn't
            # document json_object mode, so JSON is requested via the prompt.
            if as_json:
                system = system + "\n\nRespond with ONLY valid JSON, no prose."
            if effort:
                kwargs["reasoning_effort"] = effort
            max_tokens += _THINKING_HEADROOM
            chain = [self.model] + [m for m in _GEMINI_FALLBACK_MODELS if m != self.model]
            now = time.monotonic()
            models = [m for m in chain if _down_until.get(m, 0) <= now] or chain[:1]
        else:
            client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
            if as_json:
                kwargs["response_format"] = {"type": "json_object"}

        for i, model in enumerate(models):
            try:
                resp = client.chat.completions.create(
                    model=model, max_tokens=max_tokens, temperature=temperature,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}], **kwargs,
                )
            except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError):
                _down_until[model] = time.monotonic() + _COOLDOWN_SEC
                if i == len(models) - 1:
                    raise
                continue
            _down_until.pop(model, None)
            self.served_model = model
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
                "GROQ_API_KEY / XAI_API_KEY / GEMINI_API_KEY (in the environment or a .env file).")
        chosen = avail[0]
    default_model = _PROVIDERS[chosen][1]
    return LLM(provider=chosen, model=model or os.environ.get("PORTFOLIO_LLM_MODEL") or default_model)
