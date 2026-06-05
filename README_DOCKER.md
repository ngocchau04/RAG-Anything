# Run RAG-Anything with Docker Compose

This setup runs:
- `rag-webui` (Gradio WebUI)
- `ollama` (local embedding service)
- `ollama-init` (one-shot model pull helper)

It keeps indexed data across restarts via mounted folders:
- `./rag_storage`
- `./output`
- `./.tmp`

## 1. Prepare Environment

```bash
cp .env.docker.example .env.docker
```

Edit `.env.docker` and set your Gemini API key:
- `LLM_BINDING_API_KEY=...`

Notes:
- Default Docker embedding model is `nomic-embed-text`.
- If your local setup uses `nomic-embed-local:latest`, set `EMBEDDING_MODEL` in `.env.docker` accordingly.
- Custom/local aliases may not be pullable by `ollama-init`; in that case pull manually.

## 2. Start Services

```bash
docker compose up --build
```

Open:
- `http://localhost:7860`

## 3. Pull/Check Ollama Model

If needed, run:

```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose exec ollama ollama list
```

If model is missing, WebUI fails early with a clear Ollama model error message.

## 4. Stop Services

```bash
docker compose down
```

This stops containers and keeps your data.

## 5. Delete All Data Intentionally

```bash
docker compose down -v
```

Warning: `-v` removes Docker volumes, including `ollama_data`.
For bind-mounted folders (`rag_storage`, `output`, `.tmp`), files remain on host unless you delete them manually.
