# RAG Operator Guide

This repository provides a minimal but complete retrieval-augmented generation (RAG) workflow to help shop-floor operators query complex machine manuals. The pipeline pairs text chunks and visual references from PDF documentation with a language model that returns step-by-step procedures enriched with relevant imagery.

## Features
- Parses a PDF manual, extracting both text and deduplicated RGB images while filtering out boilerplate.
- Builds separate FAISS indexes for text (MiniLM) and images (CLIP) to support mixed-modality retrieval.
- Ranks candidate images for each query and guarantees every instruction step references at least one relevant visual.
- Generates structured JSON instructions through a safety-oriented prompt rendered with Jinja templates.
- Ships with Docker configurations for CPU or GPU inference and an example notebook for experimentation.

## Repository Layout
- `main.py` – end-to-end RAG pipeline orchestrating PDF processing, retrieval, prompting, and display.
- `data/` – sample manuals and the extracted page images created during indexing.
- `config/system_prompt.txt` – optional safety rules injected into the operator prompt.
- `templates/operator_prompt.j2` – Jinja template defining the LLM output contract.
- `docker/` – CPU and CUDA Dockerfiles used by `docker-compose.yml`.
- `requirements_openai.txt` – Python dependencies for running against OpenAI endpoints.
- `MVP_starting_machine_v3 copy.ipynb` – exploratory notebook showcasing the workflow interactively.

## Prerequisites
- Python 3.10+ recommended.
- An OpenAI API key exported as the `OPENAI_API_KEY` environment variable.
- (Optional) NVIDIA drivers and Docker with GPU support if you plan to use the CUDA images.

Install dependencies and prepare the environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements_openai.txt
export OPENAI_API_KEY=sk-...
```

## Running the Pipeline
1. Place the target PDF manual in the `data/` directory and update the `PDF_PATH`, `MACHINE_NAME`, and `TOP_K` constants near the top of `main.py` as needed.
2. (Optional) Adjust `config/system_prompt.txt` to steer the assistant’s tone or ordering of procedures.
3. Execute the script:

```bash
python main.py
```

The first run extracts images to `data/page##_img#.ext`, builds FAISS indexes, and prints the count of indexed text chunks and visuals. The `ask` helper is then called with a sample query (customize it or import `ask` from another script or notebook).

During execution the pipeline:
1. Retrieves top-k text passages and associated images for the query.
2. Re-ranks images with a CLIP similarity search to keep only the most relevant visuals.
3. Builds an image catalog and renders a Jinja prompt that forces JSON output.
4. Calls `ChatOpenAI` (`gpt-4o-mini`) and parses the structured response.
5. Attaches missing images heuristically and prints each step with file paths for quick reference.

## Configuration Notes
- **Prompting:** Edit `templates/operator_prompt.j2` to change the JSON schema or instructions. The system prompt file is optional; an empty file results in no extra injection.
- **Image filtering:** Tweak `min_size`, `ignore_top_pct`, `ignore_bottom_pct`, or the duplicate filtering logic inside `load_pdf` to adapt to new manuals.
- **Retrieval:** Adjust `TOP_K` or the FAISS index type if you need cosine similarity or IVF-based scaling.
- **Post-processing:** The `auto_attach_images` function uses cosine similarity between step text and image embeddings; update `sim_thresh` to be more permissive or strict.

## Docker & Compose
The provided `docker-compose.yml` launches a JupyterLab container and (optionally) a Gradio interface:

```bash
# CPU build (default)
docker compose up --build

# CUDA build (set DEVICE=cuda11 or DEVICE=cuda12 before running compose)
DEVICE=cuda12 docker compose up --build
```

Mounts map the entire project into `/home/jovyan/RAG_OperatorGuide`, so edits on the host are reflected in the container. Add your `.env` file with `OPENAI_API_KEY` (and any other secrets) before starting the stack.

## Working with the Notebook
Launch `MVP_starting_machine_v3 copy.ipynb` through JupyterLab (port `8888`) to experiment interactively. The notebook mirrors the logic in `main.py` but offers cells for inspecting intermediate embeddings, FAISS searches, and prompt outputs.

## Troubleshooting
- Ensure `OPENAI_API_KEY` is available in the environment where the script or container runs; the code raises a `ValueError` otherwise.
- If FAISS fails to load, verify that `faiss-cpu` matches your Python version and reinstall within a clean virtual environment.
- Image extraction saves files next to the PDF; confirm the `data/` directory is writable.
- For GPU runs, confirm that the NVIDIA Container Toolkit is installed and the host exposes GPUs to Docker.

## License
Distributed under the terms of the repository’s `LICENSE`.
