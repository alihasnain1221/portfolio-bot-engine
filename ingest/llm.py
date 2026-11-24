"""Pluggable LLM provider layer.

Supports Anthropic Claude, OpenAI, Groq, xAI Grok, and Google Gemini.
Auto-detects whichever API key is present (in the environment or a local .env),
so ingestion works with "whatever is available at the moment". Override the
choice with PORTFOLIO_LLM_PROVIDER=anthropic|openai|groq|xai|gemini.

Each provider takes several keys: <NAME>_API_KEY first, then <NAME>_API_KEY_2,
_3, ... in number order. A key that's rejected, out of quota or not answering
hands the request to the next one.

The OpenAI SDK is reused for OpenAI, Groq, xAI, and Gemini via base_url
override; Anthropic uses its own SDK. SDKs are imported lazily so a missing one never
breaks the others.
"""
from __future__ import annotations

import json
import os
import re
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
# A key or (key, model) that just failed is skipped for this long, so later
# requests go straight to a healthy one. Kept per process (resets on restart).
_COOLDOWN_SEC = 300
_down_until: dict[tuple[str, str | None], float] = {}   # (key, model or None for the whole key)
# Below this many seconds of budget left, another attempt can't finish in time.
_MIN_ATTEMPT_SEC = 5


def api_keys(env_key: str) -> list[str]:
    """Keys for one provider: ENV_KEY, then ENV_KEY_2, ENV_KEY_3, ... (duplicates dropped)."""
    pattern = re.compile(rf"{re.escape(env_key)}(?:_(\d+))?")
    slots = sorted((int(m.group(1) or 1), val.strip()) for name, val in os.environ.items()
                   if (m := pattern.fullmatch(name)) and val.strip())
    keys: list[str] = []
    for _, val in slots:
        if val not in keys:
            keys.append(val)
    return keys


def _failure_scope(exc: Exception) -> str | None:
    """'key' when this key can't be used, 'model' when this model is busy on it,
    None when another key or model wouldn't change the outcome."""
    status = getattr(exc, "status_code", None)
    msg = str(exc).lower()
    if status in (401, 403) or any(s in msg for s in ("api key", "api_key", "credit balance", "billing")):
        return "key"
    if status is None or status == 429 or status >= 500:   # no status: connection error or timeout
        return "model"
    return None


@dataclass
class LLM:
    provider: str
    model: str
    served_model: str | None = None   # model that answered the last call
    served_key: int | None = None     # 1 = <NAME>_API_KEY, 2 = <NAME>_API_KEY_2, ...

    def complete(self, system: str, user: str, *,
                 max_tokens: int = 1200, temperature: float = 0.2,
                 as_json: bool = False, effort: str | None = None,
                 timeout: float = 120, budget: float | None = None) -> str:
        """Single-turn completion. Returns the model's text (or JSON string).

        `temperature` applies to OpenAI-compatible providers only (current Claude
        models reject it). `effort` applies to Claude (low|medium|high|xhigh|max)
        and Gemini (minimal|low|medium|high, sent as reasoning_effort).
        `timeout` is seconds per attempt; `budget` caps the total across all
        keys and models.
        """
        env_key, _, base_url = _PROVIDERS[self.provider]
        keys = api_keys(env_key)
        if not keys:
            raise RuntimeError(f"{env_key} is not set.")
        effort = os.environ.get("PORTFOLIO_LLM_EFFORT") or effort
        deadline = time.monotonic() + budget if budget else None
        kwargs = {}

        if self.provider == "anthropic":
            import anthropic
            api_error, make_client = anthropic.APIError, (lambda k: anthropic.Anthropic(api_key=k))
            if as_json:
                system = system + "\n\nRespond with ONLY valid JSON, no prose."
            if effort:
                kwargs["output_config"] = {"effort": effort}
            if self.model.startswith(_CLAUDE_FALLBACK_MODELS):
                kwargs["betas"] = ["server-side-fallback-2026-07-01"]
                kwargs["fallbacks"] = "default"
            chain = [self.model]   # model fallback happens server-side
        else:
            # OpenAI-compatible (openai, groq, xai, gemini)
            import openai
            api_error = openai.APIError
            chain = [self.model]
            if self.provider == "gemini":
                # Key and model fallback replace SDK retries, which would only burn
                # the free-tier quota re-hitting a model that's overloaded or rate-limited.
                make_client = lambda k: openai.OpenAI(api_key=k, base_url=base_url, max_retries=0)
                # Gemini 3.x thinking can't be disabled and its compat layer doesn't
                # document json_object mode, so JSON is requested via the prompt.
                if as_json:
                    system = system + "\n\nRespond with ONLY valid JSON, no prose."
                if effort:
                    kwargs["reasoning_effort"] = effort
                max_tokens += _THINKING_HEADROOM
                chain += [m for m in _GEMINI_FALLBACK_MODELS if m != self.model]
            else:
                make_client = lambda k: openai.OpenAI(api_key=k, base_url=base_url)
                if as_json:
                    kwargs["response_format"] = {"type": "json_object"}

        # Every model on the first key, then every model on the next key, and so on.
        now = time.monotonic()
        attempts = [(slot, key, model) for slot, key in enumerate(keys, 1) for model in chain]
        healthy = [a for a in attempts
                   if _down_until.get((a[1], None), 0) <= now and _down_until.get((a[1], a[2]), 0) <= now]
        last_error: Exception | None = None
        dead_keys: set[str] = set()
        clients: dict[str, object] = {}
        for slot, key, model in healthy or attempts[:1]:
            if key in dead_keys:
                continue
            attempt_timeout = timeout
            if deadline is not None:
                attempt_timeout = min(timeout, deadline - time.monotonic())
                if attempt_timeout < _MIN_ATTEMPT_SEC and last_error is not None:
                    break
            if key not in clients:
                clients[key] = make_client(key)
            client = clients[key]
            try:
                if self.provider == "anthropic":
                    msg = client.beta.messages.create(
                        model=model, max_tokens=max_tokens + _THINKING_HEADROOM,
                        system=system, messages=[{"role": "user", "content": user}],
                        timeout=attempt_timeout, **kwargs,
                    )
                    if msg.stop_reason == "refusal":
                        raise RuntimeError(f"Claude declined the request ({getattr(msg.stop_details, 'category', None)})")
                    text, served = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text"), msg.model
                else:
                    resp = client.chat.completions.create(
                        model=model, max_tokens=max_tokens, temperature=temperature,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": user}],
                        timeout=attempt_timeout, **kwargs,
                    )
                    text, served = resp.choices[0].message.content or "", model
            except api_error as e:
                scope = _failure_scope(e)
                if scope is None:
                    raise
                last_error = e
                if scope == "key":
                    dead_keys.add(key)
                _down_until[(key, None if scope == "key" else model)] = time.monotonic() + _COOLDOWN_SEC
                continue
            _down_until.pop((key, None), None)
            _down_until.pop((key, model), None)
            self.served_model, self.served_key = served, slot
            return text
        raise last_error or RuntimeError(f"No {self.provider} key or model is available right now.")

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
    return [p for p in _PREFERENCE if api_keys(_PROVIDERS[p][0])]


def get_llm(provider: str | None = None, model: str | None = None) -> LLM:
    """Pick a provider: explicit arg > PORTFOLIO_LLM_PROVIDER > first available."""
    load_dotenv()
    chosen = provider or os.environ.get("PORTFOLIO_LLM_PROVIDER")
    if chosen:
        chosen = chosen.lower()
        if chosen not in _PROVIDERS:
            raise ValueError(f"Unknown provider '{chosen}'. Options: {list(_PROVIDERS)}")
        if not api_keys(_PROVIDERS[chosen][0]):
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
