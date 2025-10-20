# RAG Operator Guide

A Retrieval-Augmented Generation (RAG) system designed to assist industrial machine operators.  
It retrieves information from official technical manuals and generates clear, step-by-step, image-supported procedures to guide operators safely.

---

## Overview

The RAG Operator Guide integrates a Jupyter-based development environment with a Chainlit chat interface for real-time operator assistance.  
Both services run inside Docker containers and share the same environment and dependencies.

---

## Features

- LLM-powered assistant trained on machine manuals and procedures.  
- Image-supported instructions for improved clarity and safety.  
- Chainlit chat interface for natural interaction.  
- Jupyter Notebook backend for development and debugging.  
- Unified Docker Compose deployment with GPU support (optional).

---

## Project Structure

```
RAG_OperatorGuide/
├── app.py                    # Chainlit chat interface
├── rag_backend.py            # Core RAG logic (answer_query)
├── operator_prompt.j2        # Jinja2 prompt template
├── system_prompt.txt         # Domain-specific rules
├── docker-compose.yml
├── requirements_openai.txt
└── notebooks/
    └── RAG_Operator_Guide.ipynb
```

---

## Prerequisites

- Docker and Docker Compose installed  
- Valid OpenAI API key (or compatible LLM endpoint)  
- Optional: NVIDIA GPU drivers if running in GPU mode  

---

## Building the Environment

Build the shared Docker image used by both Jupyter and Chainlit services:

```bash
docker compose build --no-cache
```

This installs all Python dependencies such as `openai`, `langchain`, and `chainlit`.

---

## Running the System

Start both services:

```bash
docker compose up
```

Access the interfaces:

- Jupyter / Gradio backend: [http://localhost:8888](http://localhost:8888)
- Chainlit chat interface: [http://localhost:8000](http://localhost:8000)

---

## Using the Chat Interface

1. Open [http://localhost:8000](http://localhost:8000) in your browser.  
2. Ask an operator question, for example:  
   ```
   I see a red warning on the SIF-400 panel, what should I do?
   ```
3. The assistant retrieves information from the manuals and returns a structured JSON with:
   - Sequential step-by-step instructions  
   - Inline reference images for relevant components

---

## Updating Code or Dependencies

| Change Type | Command |
|--------------|----------|
| Edited `app.py` or `rag_backend.py` | `docker compose up` |
| Changed dependencies or Dockerfile | `docker compose build --no-cache && docker compose up` |
| Full reset (clean build) | `docker compose down -v && docker compose build --no-cache && docker compose up` |

---

## Development Tips

- The project directory is volume-mounted inside the container, so code edits take effect immediately.  
- To view logs for the Chainlit service:
  ```bash
  docker compose logs -f rag-assistant
  ```
- The Jupyter Notebook environment is available at `/home/jovyan/RAG_OperatorGuide` inside the container.

---

## Prompt Design

The prompt template (`operator_prompt.j2`) enforces structured, image-supported, and JSON-formatted outputs.

Expected JSON structure:

```json
{
  "query": "...",
  "steps": [
    "Step 1 ...",
    "Step 2 ..."
  ],
  "images_used": [
    "artifacts/images/...",
    "artifacts/images/..."
  ]
}
```

---

## Stopping Services

To stop all containers:

```bash
docker compose down
```

---

## Summary

| Component | Description | URL |
|------------|-------------|-----|
| `rag_mvp` | Jupyter / Gradio backend | [http://localhost:8888](http://localhost:8888) |
| `rag-assistant` | Chainlit chat interface | [http://localhost:8000](http://localhost:8000) |

---

**RAG Operator Guide** provides a multimodal AI assistant to help machine operators perform accurate, safe, and visualized procedures directly from official manuals.
