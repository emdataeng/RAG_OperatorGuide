# RAG Operator Guide

A Retrieval-Augmented Generation (RAG) assistant for industrial machine operators.  
It indexes official manuals, links relevant imagery, and serves step-by-step procedures through a Chainlit chat UI backed by a shared Dockerized environment.

---

## Overview

The project runs two coordinated services from a single Docker image:

- **`rag_mvp`** – a Jupyter/Gradio workspace for developing and monitoring the RAG pipeline, the `rag_backend.py` is deployed here and the notebook `RAG_Operator_Guide.ipynb` is also available. 
- **`rag-assistant`** – a Chainlit front end that calls `rag_backend.answer_query` to serve image-supported instructions in real time.

Both services mount the repository into `/home/jovyan/RAG_OperatorGuide`, read the same `.env` configuration, and share artifacts such as FAISS indexes, image catalogs, and logs.

---

## Features

- Retrieval pipeline that parses manuals, deduplicates imagery, and maintains text and vision FAISS indexes.  
- `answer_query` orchestrates retrieval, prompt rendering, and structured JSON generation via OpenAI-compatible LLMs.  
- Chainlit chat interface delivers numbered instructions with inline component images.  
- Notebook-friendly environment for rapid experimentation and debugging.  
- Optional GPU support by selecting CUDA builds at compose time.

---

## Architecture

| Component | Role | Notes |
|-----------|------|-------|
| `rag_backend.py` | Core RAG engine implementing ingestion, retrieval, cataloging, and prompting. | Driven by environment variables defined in `.env`. |
| `app.py` | Chainlit entry point. | Calls `answer_query`, formats JSON responses, streams inline images. |
| `docker-compose.yml` | Orchestrates the shared image for notebooks and Chainlit. | Select CPU/GPU via the `DEVICE` environment variable. |
| `templates/operator_prompt.j2` | Jinja2 prompt enforcing structured, multimodal output. | Updated JSON contract reflected in the Chainlit UI. |
| `config/system_prompt.txt` | Optional guardrails and domain rules injected into the LLM. | Leave empty for no extra system instructions. |

The backend keeps intermediate assets under `artifacts/` (embeddings, catalogs, extracted images) and reads manuals from `data/`.

---

## Project Structure

```
RAG_OperatorGuide/
├── app.py
├── rag_backend.py
├── docker-compose.yml
├── templates/
│   └── operator_prompt.j2
├── config/
│   └── system_prompt.txt
├── artifacts/               # Generated catalogs, FAISS stores, extracted imagery
├── data/                    # Source manuals and operator documents
├── notebooks/
│   └── RAG_Operator_Guide.ipynb
├── docker/
│   ├── Dockerfile-cpu
│   ├── Dockerfile-cuda11
│   └── Dockerfile-cuda12
└── requirements_openai.txt
```

---

## Prerequisites

- Docker and Docker Compose.  
- Valid OpenAI API key (or compatible endpoint) with network access from the containers.  
- Optional: NVIDIA GPU drivers and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for CUDA builds.

---

## Configure the Environment

All runtime settings are pulled from `.env`. At minimum provide the OpenAI credentials and core directories:

```bash
# .env (example)
OPENAI_API_KEY=sk-...
MACHINE_NAMES=SIF-400,SIF-410
DEFAULT_MACHINE_NAME=SIF-400
PDF_DIR=artifacts/manuals
OUT_DIR=artifacts
IMAGES_DIR_NAME=images
CATALOG_JSON=artifacts/catalog.json
TEMPLATES_DIR=templates
PATH_SYSTEM_PROMPT=config/system_prompt.txt
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.2
TOP_K=6
IMAGE_BOOST=1.4
DIMS=768
DEVICE=cpu        # or cuda11 / cuda12
HF_HOME=/home/jovyan/.cache/huggingface
TRANSFORMERS_CACHE=/home/jovyan/.cache/huggingface/transformers
HUGGINGFACE_HUB_CACHE=/home/jovyan/.cache/huggingface/hub
```

Adjust paths to match where manuals and artifacts should be stored inside the container volume.

---

## Build the Shared Image

```bash
# Clean rebuild for the selected DEVICE (defaults to cpu)
docker compose build --no-cache
```

This installs Python dependencies such as `langchain`, `chainlit`, `openai`, `faiss`, and OpenCLIP.

---

## Run the Services

```bash
docker compose up
```

Access the interfaces:

- JupyterLab / Gradio backend: <http://localhost:8888>  
- Chainlit assistant: <http://localhost:8000>

Both services hot-reload source code because the project directory is bind-mounted into the containers.

---

## Using the Chainlit Assistant

1. Browse to <http://localhost:8000>.  
2. Ask an operator-focused question, e.g. `I see a red warning on the SIF-400 panel. What should I do?`  
3. `answer_query` retrieves relevant text and imagery, renders the `operator_prompt.j2` template, and returns JSON with ordered steps plus `images_used`.  
4. Chainlit formats the steps as Markdown and displays inline thumbnails for each referenced image.

---

## Working with Manuals and Artifacts

- Place PDFs in the directory specified by `PDF_DIR`.  
- On first run the backend extracts deduplicated images, builds FAISS indexes (text + vision), and writes a consolidated catalog to `CATALOG_JSON`.  
- Generated assets land under `artifacts/` and are reused across sessions until the source manuals change.  
- Set `MACHINE_NAMES` in `.env` to control which catalog entries are exposed to the assistant.

---

## Development Tips

- The repository is mounted into the containers, so edits to Python files refresh immediately.  
- Tail the Chainlit logs: `docker compose logs -f rag-assistant`.  
- Inspect backend traces in `logs/debug_log_*.jsonl` (created per session).  
- Use the notebook at `/home/jovyan/RAG_OperatorGuide/notebooks/RAG_Operator_Guide.ipynb` for deeper debugging or ad-hoc queries.

---

## Prompt Design

`templates/operator_prompt.j2` enforces a structured response:

```json
{
  "query": "Operator question",
  "steps": [
    {"title": "Prepare the station", "instruction": "Isolate power..."}
  ],
  "images_used": [
    {"path": "artifacts/images/sif-400_panel.jpg", "name": "Panel overview"}
  ]
}
```

Modify the template or `config/system_prompt.txt` to adjust JSON contracts, tone, or compliance requirements.

---

## Stopping the Stack

```bash
docker compose down
```

Add `-v` to remove named volumes if you purposefully want a clean slate.

---

## Component Summary

| Service | Description | URL |
|---------|-------------|-----|
| `rag_mvp` | Notebook & backend workspace | <http://localhost:8888> |
| `rag-assistant` | Chainlit chat interface | <http://localhost:8000> |

**RAG Operator Guide** brings multimodal, procedure-focused assistance directly from official manuals to the shop floor.
