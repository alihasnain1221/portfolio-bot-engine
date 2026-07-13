# Deploy image for the portfolio chatbot /ask backend.
# Only ships the serving code + the deploy-safe bundle. No raw repos, no ingestion data.
FROM python:3.11-slim

WORKDIR /srv

# deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# serving code + sanitized bundle only
COPY app/ ./app/
COPY server/ ./server/
COPY ingest/__init__.py ingest/llm.py ./ingest/
COPY out/profile.json ./out/profile.json

EXPOSE 8000
# Cloud Run / Render provide $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
