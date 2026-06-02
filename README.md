# RAG-Anything Run Guide

> Local CPU-only multimodal RAG chatbot with Gradio WebUI, Ollama embedding, Gemini LLM/Vision, Docker Compose, and PDF export.

## Overview

This repository is a customized fork of **RAG-Anything**, an all-in-one multimodal RAG framework built on top of LightRAG. The original project focuses on processing and querying documents that contain multiple modalities such as text, images, tables, equations, and charts.

This fork focuses on a practical CPU-only setup that can run on a normal local machine or inside Docker Compose without requiring GPU or MinerU.

The current customized version supports:

* Local Gradio WebUI chatbot.
* Multi-file upload, indexing, and querying.
* Persistent indexed files after restarting the server.
* Ollama-based local embedding.
* Gemini-based LLM and Vision model.
* PDF hybrid mode for CPU-friendly PDF processing.
* Image understanding through Vision-based description.
* Basic PDF export.
* PDF report agent that generates structured PDF reports from indexed documents.
* Docker Compose deployment with persistent storage.

## Main Differences from the Original RAG-Anything

The original RAG-Anything project supports a broad multimodal pipeline, including MinerU-based parsing, multimodal processors, and LightRAG-based retrieval.

This fork changes the runtime direction to fit a CPU-only environment:

| Area           | Original Direction              | This Fork                                     |
| -------------- | ------------------------------- | --------------------------------------------- |
| Runtime target | General multimodal RAG          | CPU-only local/WebUI chatbot                  |
| GPU            | May be useful for heavy parsing | Not required                                  |
| MinerU         | Supported in upstream           | Avoided/disabled in this fork                 |
| Embedding      | Configurable                    | Ollama local embedding by default             |
| LLM/Vision     | Configurable                    | Gemini 3.1 Flash Lite by default              |
| UI             | Example scripts                 | Gradio WebUI chatbot                          |
| Deployment     | Source usage                    | Docker Compose setup                          |
| PDF export     | Not the main focus              | Added basic PDF export and report agent       |
| Storage        | Working directory               | Persistent `rag_storage/`, `output/`, `.tmp/` |

## Architecture

Current runtime architecture:

```text
Browser
  ↓
Gradio WebUI
  ↓
RAG-Anything / LightRAG
  ↓
Ollama Embedding
  ↓
Gemini LLM / Vision
  ↓
Persistent storage:
  - rag_storage/
  - output/
  - .tmp/
```

Docker Compose architecture:

```text
docker-compose.yml
├── rag-webui
│   └── runs examples/webui_gradio.py
├── ollama
│   └── serves local embedding model
└── ollama-init
    └── pulls embedding model on startup
```

## Supported File Types

Current practical support:

| File type               | Status                             | Notes                                                            |
| ----------------------- | ---------------------------------- | ---------------------------------------------------------------- |
| `.jpg`, `.jpeg`, `.png` | Supported                          | Uses image path + Vision description                             |
| `.pdf`                  | Partially supported                | Text works; table/equation/figure extraction is being stabilized |
| `.docx`                 | Supported depending on parser path | Uses lightweight/simple DOCX flow when available                 |
| `.pptx`                 | Limited                            | May require additional parser dependency                         |
| `.md`, `.html`          | Supported in fixture-level tests   | Depends on current parser path                                   |

## Requirements

### Local Python Runtime

Recommended:

* Python 3.11
* Windows/Linux/macOS
* Docker Desktop if using Docker Compose
* Ollama if running embedding locally outside Docker
* Gemini API key

### Docker Runtime

Recommended:

* Docker Desktop
* Docker Compose
* Enough disk space for Docker images and Ollama models
* Internet connection for pulling Docker images and Ollama model

## Environment Variables

Create a local `.env` or `.env.docker` depending on how you run the project.

### LLM and Vision

```env
LLM_BINDING_API_KEY=your_gemini_api_key
LLM_BINDING_HOST=https://generativelanguage.googleapis.com/v1beta/openai/

LLM_MODEL=gemini-3.1-flash-lite
VISION_MODEL=gemini-3.1-flash-lite
FALLBACK_VISION_MODEL=
```

`LLM_BINDING_API_KEY` is your API key.

`LLM_BINDING_HOST` is the API endpoint, not the API key.

### Embedding with Ollama

For local machine:

```env
EMBEDDING_PROVIDER=ollama
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=nomic-embed-local:latest
EMBEDDING_DIM=768
OLLAMA_HOST=http://localhost:11434
```

For Docker Compose:

```env
EMBEDDING_PROVIDER=ollama
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIM=768
OLLAMA_HOST=http://ollama:11434
OLLAMA_MODEL_TO_PULL=nomic-embed-text
```

Do not use Gemini embedding by default in this fork, because Gemini embedding quota can be limited.

### WebUI

```env
WEBUI_HOST=0.0.0.0
WEBUI_PORT=7860
GRADIO_SHARE=false

WEBUI_AUTH_USER=admin
WEBUI_AUTH_PASSWORD=change-me
```

If both `WEBUI_AUTH_USER` and `WEBUI_AUTH_PASSWORD` are set, Gradio authentication is enabled.

## Run Locally with Python

### 1. Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 3. Start Ollama locally

Make sure Ollama is running, then check model:

```powershell
ollama list
```

If needed:

```powershell
ollama pull nomic-embed-text
```

or use your existing local model:

```powershell
ollama pull nomic-embed-local
```

### 4. Run WebUI

```powershell
.\.venv\Scripts\python.exe .\examples\webui_gradio.py
```

Open:

```text
http://localhost:7860
```

## Run with Docker Compose

### 1. Prepare Docker environment file

```powershell
copy .env.docker.example .env.docker
```

Edit `.env.docker` and add your Gemini API key:

```env
LLM_BINDING_API_KEY=your_gemini_api_key
LLM_BINDING_HOST=https://generativelanguage.googleapis.com/v1beta/openai/
```

Do not commit `.env.docker`.

### 2. Build and start services

```powershell
docker compose up --build
```

Or run in background:

```powershell
docker compose up --build -d
```

### 3. Open WebUI

```text
http://localhost:7860
```

### 4. Check Ollama model inside Docker

```powershell
docker compose exec ollama ollama list
```

If needed:

```powershell
docker compose exec ollama ollama pull nomic-embed-text
```

### 5. Restart WebUI only

```powershell
docker compose restart rag-webui
```

### 6. Stop services

```powershell
docker compose down
```

### 7. Remove all Docker volumes intentionally

Warning: this removes Docker-managed volumes such as Ollama model data.

```powershell
docker compose down -v
```

## WebUI Usage

### 1. Upload files

Use the file upload component in WebUI.

Supported practical cases:

* Image files: `.jpg`, `.jpeg`, `.png`
* PDF files: `.pdf`
* DOCX files: `.docx`

### 2. Process / Index

Click:

```text
Process / Index
```

The file will be copied into persistent storage and indexed into LightRAG.

### 3. Chat

After indexing, select the file and ask questions.

Examples:

```text
What is this document about?
```

```text
Tóm tắt tài liệu này bằng tiếng Việt.
```

```text
What is this picture about?
```

### 4. Multi-file behavior

If multiple files are indexed:

* Select the target file from the dropdown.
* Or mention the filename in your question.
* If the request is ambiguous, the system should ask you to select a file.

## PDF Processing Modes

The fork adds CPU-friendly PDF processing modes.

### `pdf_fast`

Text-only PDF extraction.

Good for quick testing.

### `pdf_hybrid`

Hybrid PDF processing.

Designed to support:

* text
* tables
* equations
* selected visual targets

Current note: text indexing is stable, while table/equation/figure retrieval is still being stabilized for complex PDFs.

Example CLI:

```powershell
.\.venv\Scripts\python.exe .\examples\raganything_example.py .\inputs\sample.pdf --parser pdf_hybrid --working_dir .\rag_storage\sample --query "What is this article about?"
```

### Visual target analysis

For figures/charts, use selected visual target or page range instead of analyzing the whole PDF.

Example:

```powershell
.\.venv\Scripts\python.exe .\examples\raganything_example.py .\inputs\sample.pdf --parser pdf_hybrid --vision-target "Fig. 2" --max-vision-pages 1 --working_dir .\rag_storage\sample_fig2 --query "What is Fig 2 about?"
```

## PDF Export

This fork adds PDF export in the WebUI.

### Export current answer

Exports the latest question and answer into a PDF file.

### Export chat history

Exports the current conversation history into a PDF file.

### Output location

PDF files are saved in:

```text
output/reports/
```

In Docker:

```text
/app/output/reports/
```

On the host machine:

```text
./output/reports/
```

The WebUI returns a download button using Gradio `gr.File`.

## PDF Report Agent

The PDF Report Agent generates a structured PDF report from an indexed file.

Flow:

```text
User report request
→ route selected file
→ query existing LightRAG index
→ generate structured report
→ export PDF
→ return download button
```

Example report request:

```text
Tạo báo cáo tóm tắt tài liệu này bằng tiếng Việt.
```

Example English request:

```text
Generate a technical summary report for the selected document.
```

Default report structure:

```text
Title
Generated time
Source file
User report request
Executive summary
Key points
Detailed analysis
Important tables/equations/figures note
Limitations
References/source notes
```

## Persistent Storage

The following folders are important:

```text
rag_storage/
output/
.tmp/
```

### `rag_storage/`

Stores uploaded files, registry, LightRAG index, vector DB files, graph files, and text chunks.

### `output/`

Stores generated outputs such as PDF reports.

### `.tmp/`

Stores temporary files used during processing.

In Docker Compose, these are mounted so data can survive container restart.

## Reprocess and Delete

If a document was indexed with an older broken pipeline, use reprocess.

Reprocess should:

* clear the old working directory;
* remove stale LightRAG state;
* rebuild the index.

Delete should:

* remove registry record;
* remove working directory;
* remove stored upload file if it is not shared.

If you see:

```text
Content already exists
```

use Reprocess so the old index is cleared before indexing again.

## Known Limitations

### PDF tables/equations/figures

Text extraction from PDFs is currently the most stable path.

Table, equation, and figure/chart retrieval may need further stabilization for complex academic PDFs.

Known issue examples:

* table retrieval may pick the wrong table if the PDF text layout is noisy;
* equation extraction may fail if the equation is split across multiple visual/text layers;
* figure/chart understanding requires Vision and selected target/page range.

### Vision quota

Vision calls use Gemini API quota.

To save quota:

* avoid processing all PDF pages with Vision;
* use `vision_target` or `vision_page_range`;
* keep `max_vision_pages` small.

### Docker build time

Docker build can be slow because of dependencies such as PaddleOCR/OpenCV.

If only Python source files changed and Docker image already contains dependencies, sometimes restart is enough.

If Docker image does not reflect code changes, rebuild:

```powershell
docker compose build --no-cache rag-webui
docker compose up -d
```

## Troubleshooting

### WebUI opens locally but not from Docker

Check:

```powershell
docker compose ps
docker compose logs rag-webui
```

Make sure WebUI binds to:

```env
WEBUI_HOST=0.0.0.0
```

### Ollama model not found

Check:

```powershell
docker compose exec ollama ollama list
```

Pull model:

```powershell
docker compose exec ollama ollama pull nomic-embed-text
```

### Gemini API connection error

Check `.env.docker`:

```env
LLM_BINDING_API_KEY=...
LLM_BINDING_HOST=https://generativelanguage.googleapis.com/v1beta/openai/
```

### PDF export shows broken Vietnamese font

Make sure Unicode fonts are available.

Docker should include:

```text
fonts-dejavu-core
```

The exporter supports:

```env
PDF_EXPORT_FONT_PATH=
PDF_EXPORT_BOLD_FONT_PATH=
```

### PDF formula is printed as LaTeX

ReportLab does not render LaTeX math automatically.

The exporter should normalize simple LaTeX formulas into readable plain text before rendering PDF.

Example:

```text
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
```

should become:

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Docker still runs old code

Rebuild the image:

```powershell
docker compose down
docker compose build --no-cache rag-webui
docker compose up -d
```

Then reprocess the affected document.

## Tests

Run focused tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_pdf_export.py tests/test_pdf_report_agent.py tests/test_docker_setup.py
```

Run all tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -rs
```

## Recommended Development Workflow

1. Make code changes locally.
2. Run focused tests.
3. Run WebUI locally.
4. If Docker-related, rebuild Docker.
5. Test upload/index/chat.
6. Test export PDF.
7. Reprocess old documents if indexing logic changed.
8. Commit only relevant files.

## Git Notes

Do not commit:

```text
.env
.env.docker
.venv/
rag_storage/
output/
.tmp/
*.log
```

Commit:

```text
Dockerfile
docker-compose.yml
.env.docker.example
README_DOCKER.md
README_RUN.md
examples/
raganything/
tests/
```

## Current Recommended Usage

For local development:

```powershell
.\.venv\Scripts\python.exe .\examples\webui_gradio.py
```

For Docker deployment:

```powershell
docker compose up --build
```

Then open:

```text
http://localhost:7860
```

## Roadmap

Planned next improvements:

* Stabilize PDF table retrieval.
* Stabilize PDF equation extraction.
* Improve figure/chart indexing through selected Vision targets.
* Improve PDF report agent deduplication.
* Improve Docker build speed.
* Prepare VPS/cloud deployment using Docker Compose.
* Add authentication and quota protection for public deployment.
