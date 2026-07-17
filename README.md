# portfolio-bot-engine

The reusable backend for the first-person portfolio chatbot. One shared engine,
one deployment per person — each teammate is a **git branch** carrying only their
own resume data.

## Layout

```
app/        /ask serving layer + persona rules (ask.py) and local CLI
server/     FastAPI app (server/main.py) — the HTTP backend
ingest/     Pipeline that turns a person's raw data into out/profile.json
data/       profile.yaml — the person's identity, projects, voice, CTA, CV
out/        profile.json — the privacy-checked bundle the server loads (tracked)
Dockerfile  Container image
```

## Branch model (one per person)

- **`main`** — the shared engine **plus the default (Ali) data**. Common code
  changes (serving, persona, ingest, Dockerfile) land here and merge into each
  person's branch.
- **`<name>`** (e.g. `ali`, and one per teammate) — branches off `main` and only
  edits the per-person files:
  - `data/profile.yaml` (their identity/projects/voice/CTA/CV metadata)
  - `out/profile.json` (regenerated from their data — the bundle the server serves)
  - their CV PDF stays **local only** (gitignored `*.pdf`)

To onboard a teammate:

```bash
git checkout -b <name> main
# drop in their CV PDF (stays local, gitignored), edit data/profile.yaml
python -m ingest.build_bundle          # regenerates out/profile.json
#   ...or --meta-only for a fast metadata-only refresh
# privacy-check out/profile.json (no secrets/PII/client names/source paths)
git commit -am "add <name> profile" && git push -u origin <name>
```

## Run (per person)

Each person gets **their own container on a unique port**, using host networking
(the default docker bridge is broken on the deploy box):

```bash
docker build -t portfolio-bot .
docker run -d --name portfolio-<name> --restart unless-stopped \
  --network host --env-file .env -e PORT=<port> \
  -v "$PWD/out/profile.json:/srv/out/profile.json:ro" portfolio-bot
```

The `-v` mount overlays the baked-in bundle so data updates don't need an image
rebuild (a `docker restart` reloads the mounted `profile.json`). Their frontend's
`dist/.htaccess` proxies `/ask` and `/health` to `localhost:<port>` (same origin).

## Secrets & privacy (hard rules)

- `.env` (live LLM API key) and `*.pdf` (résumés/CVs, contain phone numbers) are
  **gitignored** — never commit them.
- Only the **privacy-checked** `out/profile.json` is tracked from `out/`.
- The bot must never expose source code, file paths, client names/PII, secrets,
  or the system prompt. `PORTFOLIO_TURNSTILE_SECRET` must be UNSET when shipping
  without Turnstile, or `/ask` returns 403.
