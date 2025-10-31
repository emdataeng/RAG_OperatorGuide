import os, io
from PIL import Image
import fitz  # PyMuPDF
import numpy as np
import faiss
from IPython.display import display

from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
#from langchain.schema import Document


import hashlib
import json, re
from math import inf
from jinja2 import Environment, FileSystemLoader

## 1. Configuration
from dataclasses import dataclass
from pathlib import Path
import os

ENV_PATH = Path(".env")


def load_env_file(path: Path = ENV_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Environment file '{path}' not found.")
    env_vars = {}
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            env_vars[key] = value
    os.environ.update(env_vars)
    return env_vars


_ENV_CACHE = load_env_file()


def env_get(key: str, *, cast, required: bool = True, default=None):
    raw = os.environ.get(key)
    if raw is None or raw == "":
        if required:
            raise ValueError(f"Missing required environment variable '{key}' in {ENV_PATH}")
        return default
    if cast is str:
        return raw
    try:
        return cast(raw)
    except Exception as exc:
        raise ValueError(f"Invalid value for '{key}': {raw}") from exc


def env_get_list(key: str, *, separator: str = ",", required: bool = True):
    raw = env_get(key, cast=str, required=required)
    if raw is None:
        return []
    entries = [part.strip() for part in raw.split(separator)]
    values = [part for part in entries if part]
    if required and not values:
        raise ValueError(f"Environment variable '{key}' must contain at least one entry.")
    return values


@dataclass
class Config:
    # Paths
    pdf_dir: str = env_get("PDF_DIR", cast=str)
    out_dir: str = env_get("OUT_DIR", cast=str)
    images_dir_name: str = env_get("IMAGES_DIR_NAME", cast=str)
    catalog_json: str = env_get("CATALOG_JSON", cast=str)

    # OpenCLIP model settings
    openclip_model: str = env_get("OPENCLIP_MODEL", cast=str)
    openclip_pretrained: str = env_get("OPENCLIP_PRETRAINED", cast=str)
    device: str = env_get("DEVICE", cast=str)

    # Indexing / retrieval
    dims: int = env_get("DIMS", cast=int)
    top_k: int = env_get("TOP_K", cast=int)
    image_boost: float = env_get("IMAGE_BOOST", cast=float)


cfg = Config()

# Prepare paths
Path(cfg.pdf_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
(Path(cfg.out_dir) / cfg.images_dir_name).mkdir(parents=True, exist_ok=True)

#print(cfg)

# --- 1) Settings ---
MACHINE_NAMES = env_get_list("MACHINE_NAMES")
DEFAULT_MACHINE_NAME = env_get("DEFAULT_MACHINE_NAME", cast=str, required=False)
if DEFAULT_MACHINE_NAME:
    ordered_names = [DEFAULT_MACHINE_NAME] + [name for name in MACHINE_NAMES if name != DEFAULT_MACHINE_NAME]
    MACHINE_NAMES = ordered_names
    MACHINE_NAME = DEFAULT_MACHINE_NAME
else:
    MACHINE_NAME = MACHINE_NAMES[0]

OPENAI_API_KEY = env_get("OPENAI_API_KEY", cast=str)
PATH_SYSTEM_PROMPT = env_get("PATH_SYSTEM_PROMPT", cast=str)
TEMPLATES_DIR = env_get("TEMPLATES_DIR", cast=str)
env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

LLM_MODEL = env_get("LLM_MODEL", cast=str)
LLM_TEMPERATURE = env_get("LLM_TEMPERATURE", cast=float)
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)

os.environ["HF_HOME"] = env_get("HF_HOME", cast=str)
os.environ["TRANSFORMERS_CACHE"] = env_get("TRANSFORMERS_CACHE", cast=str)
os.environ["HUGGINGFACE_HUB_CACHE"] = env_get("HUGGINGFACE_HUB_CACHE", cast=str)


## 2. Utilities
import json
import re
from typing import List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

_DEBUG_LOG_PATH = None

def _get_debug_log_path() -> Path:
    global _DEBUG_LOG_PATH
    if _DEBUG_LOG_PATH is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        _DEBUG_LOG_PATH = LOGS_DIR / f"debug_log_{timestamp}.jsonl"
    return _DEBUG_LOG_PATH

def write_log_entry(section: str, function: str, payload):
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "section": section,
        "function": function,
        "data": payload,
    }
    with open(_get_debug_log_path(), "a", encoding="utf-8") as log_file:
        json.dump(record, log_file)
        log_file.write("\n")


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def safe_fname(s: str) -> str:
    s = norm_space(s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:200]

def heading_score(fontsize: float) -> float:
    # Heuristic: larger font size -> higher heading score
    return fontsize

def build_heading_hierarchy(previous_headings: List[Dict[str, Any]], current: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Simplistic heuristic based on font sizes: if current fontsize is smaller, it could be a subheading
    # Otherwise, it might reset the hierarchy
    if not previous_headings:
        return [current]
    last = previous_headings[-1]
    if current["fontsize"] < last["fontsize"]:
        return previous_headings + [current]  # deeper level
    else:
        return [current]  # reset hierarchy



## 3. PDF Parsing (text blocks, font sizes, images, positions)
def region_filter(bbox, page_height, ignore_top_pct=0.05, ignore_bottom_pct=0.05, min_size=40, image=None):
    """
    Filters out unwanted regions/images:
      - skip images too close to top/bottom (logos, footers)
      - skip small icons/logos (width or height < min_size)
      - skip completely black images
    Returns True if the image should be kept.
    """
    x0, y0, x1, y1 = bbox
    width, height = x1 - x0, y1 - y0

    # 1️ Skip top/bottom zones
    if y1 < ignore_top_pct * page_height or y0 > (1 - ignore_bottom_pct) * page_height:
        return False

    # 2️ Skip small icons
    if width < min_size or height < min_size:
        return False

    # 3️ Skip fully black images
    if image is not None:
        arr = np.array(image.convert("L"))  # grayscale
        if np.mean(arr) < 5:  # near-black threshold
            return False

    return True

import fitz  # PyMuPDF
from PIL import Image
import io
from tqdm import tqdm
import os

def extract_pdf(pdf_path: str, out_dir: str, images_dir_name: str,
                MACHINE_NAME: str,
                ignore_top_pct=0.05, ignore_bottom_pct=0.05, min_size=40) -> List[Dict[str, Any]]:
    """
    Extracts text blocks and images from a PDF.
    Includes MACHINE_NAME in all image metadata and filenames.
    Filters images by region_filter() to skip header/footer/small/black ones.
    """
    doc = fitz.open(pdf_path)
    items = []
    img_counter_global = 0

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_height = page.rect.height
        page_dict = page.get_text("dict")  # contains blocks, lines, spans with fonts/sizes
        blocks = page_dict.get("blocks", [])

        # Collect text spans with font sizes and bbox
        text_items = []
        for b in blocks:
            if b["type"] == 0:  # text block
                bbox = b["bbox"]
                lines = b.get("lines", [])
                full_text = ""
                max_font = 0.0
                for ln in lines:
                    for sp in ln.get("spans", []):
                        full_text += sp["text"]
                        if sp.get("size", 0) > max_font:
                            max_font = sp["size"]
                text_items.append({
                    "type": "text",
                    "text": norm_space(full_text.encode("utf-8", "ignore").decode("utf-8")),
                    "fontsize": max_font,
                    "bbox": bbox,
                    "page": page_index
                })

            elif b["type"] == 1:  # image block
                bbox = b["bbox"]
                # extract image by id from page.get_images
                # Multiple images may exist; we will grab the one whose bbox matches approximately
                # Instead, use doc.extract_image for images listing; safer: use get_pixmap on bbox.
                pix = page.get_pixmap(clip=fitz.Rect(*bbox), dpi=150)
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))

                #apply region filter:
                if not region_filter(bbox, page_height,
                                     ignore_top_pct=ignore_top_pct,
                                     ignore_bottom_pct=ignore_bottom_pct,
                                     min_size=min_size,
                                     image=img):
                    continue  # skip unwanted images

                # Save image with machine name prefix
                img_counter_global += 1
                img_name = f"{MACHINE_NAME}_p{page_index+1:03d}_img{img_counter_global:04d}.png"
                img_path = os.path.join(out_dir, images_dir_name, img_name)
                img.save(img_path)

                text_items.append({
                    "type": "image",
                    "image_path": img_path,
                    "bbox": bbox,
                    "page": page_index,
                })

        # Sort items by vertical position (y0)
        text_items.sort(key=lambda x: x["bbox"][1] if "bbox" in x else 0.0)

        # Build heading contexts and attach nearest preceding headings to images
        heading_stack: List[Dict[str, Any]] = []
        image_counters_by_heading: Dict[str, int] = {}

        for it in text_items:
            if it["type"] == "text":
                # Heuristic: treat as heading if font size is among the largest on page or if it matches numbered pattern
                is_numbered = bool(re.match(r"^\\d+(\\.\\d+)*\\s+.+", it["text"]))
                if is_numbered or it["fontsize"] >= (max([t["fontsize"] for t in text_items if t["type"]=="text"]+[0])*0.9):
                    heading_stack = build_heading_hierarchy(heading_stack, it)
                    #Keep only meaningful text
                    heading_stack[-1]["text"] = it["text"]

            elif it["type"] == "image":
                # Build heading chain text
                if heading_stack:
                    chain_texts = [h["text"] for h in heading_stack if h.get("text")]
                    chain_compact = " > ".join(chain_texts)
                    top_heading = heading_stack[-1]["text"]
                else:
                    chain_texts = []
                    chain_compact = ""
                    top_heading = "Unlabeled"

                # Create metadata key for counting images under the top heading
                top_key = safe_fname(top_heading) if top_heading else "Unlabeled"
                image_counters_by_heading.setdefault(top_key, 0)
                image_counters_by_heading[top_key] += 1
                img_idx = image_counters_by_heading[top_key]

                # Build user-style metadata name
                # Prefer the deepest numbered heading if available
                numbered = [t for t in chain_texts if re.match(r"^\\d+(\\.\\d+)*\\s+.+", t)]
                final_heading = numbered[-1] if numbered else top_heading or "Unlabeled"

                image_metadata_name = f"{MACHINE_NAME}_{final_heading}_image_{img_idx}"

                items.append({
                    "modality": "image",
                    "page": it["page"],
                    "bbox": it["bbox"],
                    "image_path": it["image_path"],
                    "heading_chain": chain_texts,
                    "heading_path": chain_compact,
                    "image_metadata_name": image_metadata_name,
                    "source_pdf": pdf_path,
                    "source_machine": MACHINE_NAME
                })

        # Record text blocks for text retrieval
        for t in text_items:
            if t["type"] == "text" and t.get("text"):
                items.append({
                    "modality": "text",
                    "page": t["page"],
                    "bbox": t["bbox"],
                    "text": t["text"],
                    "fontsize": t["fontsize"],
                    "source_pdf": pdf_path,
                    "source_machine": MACHINE_NAME
                })

    doc.close()
    return items


### 3.3 Extract text and images from pdfs to build the catalog
def run_extraction(pdf_dir: str, out_dir: str, images_dir_name: str, catalog_json: str,
                    MACHINE_NAMES: List[str], ignore_bottom_pct: float, ignore_top_pct: float):
    machine_names = []
    for name in MACHINE_NAMES:
        cleaned = norm_space(name)
        if cleaned:
            machine_names.append(cleaned)
    if not machine_names:
        raise ValueError("MACHINE_NAMES must include at least one non-empty name.")

    all_items = []
    items_per_machine: Dict[str, int] = {name: 0 for name in machine_names}
    pdfs = sorted(Path(pdf_dir).glob("**/*.pdf"))
    unmatched_pdfs = []
    progress = tqdm(pdfs, desc="Extracting PDFs")

    for pdf_path in progress:
        pdf_path = Path(pdf_path)
        stem_upper = pdf_path.stem.upper()
        matched_name = None
        for candidate in machine_names:
            if stem_upper.startswith(candidate.upper()):
                matched_name = candidate
                break

        if matched_name is None:
            progress.set_description("Extracting PDFs")
            unmatched_pdfs.append(str(pdf_path))
            continue

        progress.set_description(f"Extracting {matched_name}")
        items = extract_pdf(
            str(pdf_path),
            out_dir,
            images_dir_name,
            matched_name,
            ignore_top_pct=ignore_top_pct,
            ignore_bottom_pct=ignore_bottom_pct,
        )
        all_items.extend(items)
        items_per_machine[matched_name] = items_per_machine.get(matched_name, 0) + len(items)

    progress.set_description("Extracting PDFs")

    if unmatched_pdfs:
        print("Skipped PDFs without matching machine name:")
        for skipped in unmatched_pdfs:
            print(f"  - {skipped}")

    out_path = os.path.join(out_dir, catalog_json)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in all_items:
            line = json.dumps(it, ensure_ascii=False)
            line = line.replace("\n", "\\n")  # escape accidental newlines inside JSON
            f.write(line + "\n")

    print(f"Wrote {len(all_items)} items to {out_path}")
    for name in machine_names:
        count = items_per_machine.get(name, 0)
        if count:
            print(f"  {name}: {count} items")

    return all_items


## 4. OpenCLIP: Model & Embeddings
import os
import numpy as np


# Lazy import to avoid failures if packages aren't installed yet
def load_openclip(model_name: str, pretrained: str, device: str="cpu"):
    import torch
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device,
                cache_dir=os.environ["HF_HOME"]   # ensure it uses the safe cache
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer

def embed_texts_openclip(texts: List[str], model, tokenizer, device="cpu", batch_size=32) -> np.ndarray:
    import torch
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tok = tokenizer(batch)
            tok = {k: v.to(device) for k,v in tok.items()} if isinstance(tok, dict) else tok.to(device)
            feats = model.encode_text(tok)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats.cpu().numpy())
    return np.vstack(embs) if embs else np.zeros((0,))

def embed_images_openclip(img_paths: List[str], model, preprocess, device="cpu", batch_size=16) -> np.ndarray:
    import torch
    from PIL import Image
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(img_paths), batch_size):
            batch = img_paths[i:i+batch_size]
            imgs = []
            for p in batch:
                im = Image.open(p).convert("RGB")
                imgs.append(preprocess(im))
            imgs = torch.stack(imgs).to(device)
            feats = model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats.cpu().numpy())
    return np.vstack(embs) if embs else np.zeros((0,))

##############Captions#############
# Build a textual surrogate (caption) since images in pdfs don't come with captions
import pandas as pd
def build_caption_for_images(df_img: pd.DataFrame) -> List[str]:
    # Build a textual surrogate (caption)
    captions = []
    for rec in df_img.to_dict(orient="records"):
        # Combine semantic hints into a short phrase
        meta_parts = [
            rec.get("image_metadata_name") or "",
            rec.get("heading_path") or "",
        ]
        caption = " | ".join(p for p in meta_parts if p)
        captions.append(caption if caption.strip() else "image from manual")
    return captions
#################################################


## 5. Build a Multimodal FAISS Index
import pandas as pd
def json_parser(catalog_path: str) -> pd.DataFrame:
    # Load catalog
    rows = []
    with open(catalog_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            #Try safe parsing — skip malformed lines
            try:
                # Some lines may accidentally contain multiple JSON objects -> split them
                parts = re.findall(r'\{.*?\}(?=$|\s*\{)', line)
                if len(parts) > 1:
                    for p in parts:
                        rows.append(json.loads(p))
                else:
                    rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] Skipping bad JSON line {i}: {str(e)[:80]}")
                continue
    
    print(f"Loaded {len(rows)} valid JSON objects from {catalog_path}")
    df = pd.DataFrame(rows)
    print(df.head(2))
    return df

import faiss
import pandas as pd

def build_indices(catalog_path: str, cfg: Config):
    # Load catalog    
    df = json_parser(catalog_path)

    # Separate modalities
    df_text = df[df["modality"]=="text"].copy()
    df_img = df[df["modality"]=="image"].copy()

    # Load OpenCLIP
    model, preprocess, tokenizer = load_openclip(cfg.openclip_model, cfg.openclip_pretrained, cfg.device)

    # Prepare contents for embeddings
    text_payloads = df_text["text"].tolist()
    img_paths = df_img["image_path"].tolist()

    print(f"Embedding {len(text_payloads)} text blocks and {len(img_paths)} images...")
    text_embs = embed_texts_openclip(text_payloads, model, tokenizer, cfg.device)
    img_embs = embed_images_openclip(img_paths, model, preprocess, cfg.device)


###########New###########################
    # Create text embeddings for captions using the same text encoder
    captions = build_caption_for_images(df_img)
    text_capt_embs = embed_texts_openclip(captions, model, tokenizer, cfg.device)

    # Normalize both before fusing
    img_embs = img_embs / np.linalg.norm(img_embs, axis=1, keepdims=True)
    text_capt_embs = text_capt_embs / np.linalg.norm(text_capt_embs, axis=1, keepdims=True)

    # Weighted fusion (adjust alpha to taste)
    alpha = 0.5  # try 0.3–0.7 range
    emb_fused = (1 - alpha) * img_embs + alpha * text_capt_embs
    emb_fused = emb_fused / np.linalg.norm(emb_fused, axis=1, keepdims=True)

    # 5. Use emb_fused for indexing
    #index_image.add(emb_fused.astype("float32"))

    print("Image-only mean sim:", np.mean(img_embs @ text_embs.T))
    print("Fused mean sim:", np.mean(emb_fused @ text_embs.T))

    # Replace img_embs with fused embeddings
    img_embs = emb_fused
    ########################################



    # Build FAISS indices (cosine similarity via inner product on normalized vectors)
    dim_text = text_embs.shape[1] if len(text_embs.shape)==2 else cfg.dims
    dim_img  = img_embs.shape[1] if len(img_embs.shape)==2 else cfg.dims

    index_text = faiss.IndexFlatIP(dim_text)
    index_img  = faiss.IndexFlatIP(dim_img)

    if len(text_embs):
        index_text.add(text_embs.astype("float32"))
    if len(img_embs):
        index_img.add(img_embs.astype("float32"))

    # Persist
    faiss.write_index(index_text, os.path.join(cfg.out_dir, "faiss_text.index"))
    faiss.write_index(index_img,  os.path.join(cfg.out_dir, "faiss_image.index"))
    df_text.to_json(os.path.join(cfg.out_dir, "df_text.json"), orient="records", lines=True)
    df_img.to_json(os.path.join(cfg.out_dir, "df_img.json"), orient="records", lines=True)

    print("Indices built and saved.")
    return {
        "index_text_path": os.path.join(cfg.out_dir, "faiss_text.index"),
        "index_image_path": os.path.join(cfg.out_dir, "faiss_image.index"),
        "df_text_path": os.path.join(cfg.out_dir, "df_text.json"),
        "df_image_path": os.path.join(cfg.out_dir, "df_img.json"),
    }


## 6. Retrieval & Composition
################New
# --- Normalize scores per modality
def normalize_scores(recs):
    if not recs: return recs
    vals = np.array([r["score"] for r in recs], dtype=float)
    mu, sigma = vals.mean(), vals.std() + 1e-6
    for r in recs:
        r["score_norm"] = (r["score"] - mu) / sigma
    return recs
###################

def load_indices(cfg: Config):
    import pandas as pd
    index_text = faiss.read_index(os.path.join(cfg.out_dir, "faiss_text.index"))
    index_image = faiss.read_index(os.path.join(cfg.out_dir, "faiss_image.index"))
    df_text = pd.read_json(os.path.join(cfg.out_dir, "df_text.json"), orient="records", lines=True)
    df_img  = pd.read_json(os.path.join(cfg.out_dir, "df_img.json"),  orient="records", lines=True)
    model, preprocess, tokenizer = load_openclip(cfg.openclip_model, cfg.openclip_pretrained, cfg.device)
    return index_text, index_image, df_text, df_img, model, preprocess, tokenizer

def search(query: str, cfg: Config, top_k: int=None) -> Dict[str, Any]:
    import torch
    top_k = top_k or cfg.top_k
    index_text, index_image, df_text, df_img, model, preprocess, tokenizer = load_indices(cfg)

    # Embed query
    q_emb = embed_texts_openclip([query], model, tokenizer, cfg.device)[0].astype("float32").reshape(1, -1)

    # Search text
    D_t, I_t = index_text.search(q_emb, top_k)
    # Search images
    D_i, I_i = index_image.search(q_emb, top_k)

    # Combine with simple late fusion (scores already cosine similarities)
    #results = []
    text_hits = []
    img_hits = []
    
    for score, idx in zip(D_t[0].tolist(), I_t[0].tolist()):
        if idx == -1: continue
        rec = df_text.iloc[idx].to_dict()
        rec.update({"score": float(score), "modality": "text"})
        ##results.append(rec)
        text_hits.append(rec)

    for score, idx in zip(D_i[0].tolist(), I_i[0].tolist()):
        if idx == -1: continue
        rec = df_img.iloc[idx].to_dict()
        rec.update({"score": float(score * cfg.image_boost), "modality": "image"})
        #results.append(rec)
        img_hits.append(rec)

    # Sort by score desc and take top_k overall
    #results.sort(key=lambda x: x["score"], reverse=True)
    #results = results[:top_k]
    
    # --- Normalize scores per modality
    text_hits = normalize_scores(text_hits)
    img_hits = normalize_scores(img_hits)

    # --- Merge and sort by normalized score ---
    merged = text_hits + img_hits
    merged.sort(key=lambda x: x["score_norm"], reverse=True)
    results = merged[:top_k]


    return {
        "query": query,
        "results": results
    }


## 7. JSON Output Composer (Steps + Relevant Images)

import re

def extract_steps_from_text(text: str) -> List[str]:
    # Simple heuristic: split on numbered or bulleted lines; clean
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    steps = []
    for ln in lines:
        if re.match(r"^(?:\d+\.|[-*•])\s+", ln):
            steps.append(re.sub(r"^(?:\d+\.|[-*•])\s+", "", ln).strip())
    # Fallback: if no explicit bullets, chunk sentences
    if not steps:
        sents = re.split(r"(?<=[.!?])\s+", text)
        steps = [s.strip() for s in sents if len(s.strip()) > 0][:6]
    return steps[:8]

def compose_json_plan(query: str, search_out: Dict[str, Any], max_images: int = 4) -> Dict[str, Any]:
    # Gather top text to form step suggestions, and top images to include
    texts = [r["text"] for r in search_out["results"] if r["modality"]=="text" and isinstance(r.get("text"), str)]
    big_context = "\n".join(texts[:5]) if texts else ""
    steps = extract_steps_from_text(big_context) if big_context else []

    img_hits = [r for r in search_out["results"] if r["modality"]=="image"]
    img_hits = img_hits[:max_images]

    images_payload = []
    for r in img_hits:
        images_payload.append({
            "path": r.get("image_path"),
            "page": int(r.get("page", -1)),
            "bbox": r.get("bbox"),
            "heading_path": r.get("heading_path"),
            "image_metadata_name": r.get("image_metadata_name"),
            "source_pdf": r.get("source_pdf")
        })

    return {
        "query": query,
        "steps": steps,
        "images": images_payload,
        "context_used_charlen": len(big_context)
    }

# Example usage (after building the index):
# out = search("Emergency stop procedure", cfg, top_k=10)
# plan = compose_json_plan("Emergency stop procedure", out, max_images=4)
# plan


## 8. Builds the PDF Extraction, parsing and Embeddings Index Creation: 

def build_all(cfg: Config, machines_list: List):
    catalog_path = os.path.join(cfg.out_dir, cfg.catalog_json)
    _ = run_extraction(
        cfg.pdf_dir,
        cfg.out_dir,
        cfg.images_dir_name,
        cfg.catalog_json,
        MACHINE_NAMES=machines_list,
        ignore_bottom_pct=0.1,
        ignore_top_pct=0.1,
    )
    idx_paths = build_indices(catalog_path, cfg)
    return idx_paths


### 8.1 End-to-End Runner TEST

def retrieve_plan(query: str, cfg: Config, top_k: int=10, max_images: int=4):
    search_out = search(query, cfg, top_k=top_k)
    plan = compose_json_plan(query, search_out, max_images=max_images)


    write_log_entry(
        "8. End-to-End Runner",
        "retrieve_plan",
        {
            "instructions": [
                f"Put PDFs in: {cfg.pdf_dir}",
                "Run build_all(cfg) # may take time",
                f"retrieve_plan(query='{query}', cfg)",
            ],
            "query": query,
            "plan": plan,
        },
    )

    return plan


## 9. Build a context that merges text + image info
#It creates a single human-readable context string summarizing everything retrieved for the query.

import os

def make_context_from_plan(plan: dict, base_path: str = ".") -> str:
    """
    Build a readable text context for the LLM that includes:
    - Textual context / steps retrieved
    - References to relevant images with true metadata names and relative paths
    
    Args:
        plan: dict returned by retrieve_plan()
        base_path: root folder (e.g. project Basepath), used to resolve image paths
    
    Returns:
        str: context string for LLM prompt
    """
    ctx = []
    ctx.append(f"Query: {plan.get('query', '')}\n")

    # ---- TEXT CONTEXT ----
    if plan.get("steps"):
        ctx.append("Relevant textual context or steps found:\n")
        for i, step in enumerate(plan["steps"], start=1):
            ctx.append(f"  Step {i}: {step}")
    else:
        ctx.append("No textual steps found in retrieval.\n")

    # ---- IMAGE CONTEXT ----
    images = plan.get("images", [])
    if images:
        ctx.append("\nRelevant images found in manuals:\n")
        for i, img in enumerate(images, start=1):
            heading = img.get("heading_path", "Unlabeled section")
            meta_name = img.get("image_metadata_name", "Unknown_image")
            rel_path = img.get("path", "")
            abs_path = os.path.abspath(os.path.join(base_path, rel_path))
            page = img.get("page", "?")

            ctx.append(
                f"  Image {i}: {heading}\n"
                f"    • metadata_name: {meta_name}\n"
                f"    • page: {page}\n"
                f"    • file: {rel_path}\n"
            )
    else:
        ctx.append("\nNo images retrieved for this query.\n")

    return "\n".join(ctx)


## 10. Build a focused LLM prompt
def load_system_prompt(path) -> str:
    """Load system prompt text from a file."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def make_prompt(
    query: str,
    context: str,
    system_prompt_file: str = PATH_SYSTEM_PROMPT,
    env=env,
) -> str:
    """
    Create a concise, instruction-style prompt for the LLM.
    Renders the operator assistant prompt using Jinja2 template and system prompt file,
    dynamically injecting context-specific focus rules when appropriate."""
    
    # Load system prompt text (extra safety / specialization rules)
    system_prompt = load_system_prompt(system_prompt_file)

    # Dynamically compute min_images based on retrieved context
    #if plan and "images" in plan and len(plan["images"]) > 0:
    #    min_images = len(plan["images"])
    #else:
    #    min_images = 1  # fallback to 1 as a baseline
    min_images = getattr(cfg, "min_images", 3)
    min_steps = getattr(cfg, "min_steps", 3)

    # --- NEW LOGIC: Inject dynamic focus reminder based on query ---
    # Detect whether this query is about a specific machine, station, or operational behavior
    machine_keywords = [
        "station", "machine", "module", "unit", "cell",
        "manual mode", "operation", "configure", "function", "procedure"
    ]

    # Lowercase match to reduce brittleness
    if any(k in query.lower() for k in machine_keywords):
        context_focus_reminder = """
        Before answering, ensure you analyze and understand the following sections
        for the relevant machine or station (if present in the context):

        1. "Station Function" — what the station does and its role in the process.
        2. "Station Operation" — how the operator interacts with it.
        3. "Configuring the Station in Manual Mode" — how to safely prepare or test it.

        Base your reasoning primarily on these sections before considering any
        other parts of the manual. Only then integrate additional or more specific
        steps related to the user’s query.
        """
        system_prompt += "\n\n" + context_focus_reminder
    
    
    template = env.get_template("operator_prompt.j2")
    
    prompt = template.render(
        query=query,
        context=context,
        min_images=min_images,
        min_steps=min_steps,
        system_prompt=system_prompt,
    )

    # Add a final reminder to prevent extra formatting
    prompt += "\n\nIMPORTANT: Respond ONLY with a valid JSON object. Do not use Markdown code fences or explanations."

    return prompt


from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

#query = "Emergency stop procedure for station X"
#query =f"I see in the panel in {MACHINE_NAME} a message: 'Please refill red pellets', what should I do?"
#query =f"I see a round canister in front of the {MACHINE_NAME} and I want to fill it with yello pellets, what should I do next?"

def answer_query(query: str)-> str:
    plan = retrieve_plan(query, cfg)
    
    # Build full multimodal context + prompt
    context = make_context_from_plan(plan, base_path=".")

    #For Debug:
    #print("Plan retrieved:")
    #print(json.dumps(plan, indent=2))
    #print("Context:")
    #print(f"{context}")
    ####

    # Render the prompt dynamically
    prompt = make_prompt(query, context)

    write_log_entry(
        "10. Build a focused LLM prompt",
        "make_prompt",
        {
            "query": query,
            "plan": plan,
            "context": context,
            "prompt": prompt,
        },
    )
    

    # Send to LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # or gpt-4-turbo / gpt-4o depending on your plan
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_answer = response.choices[0].message.content.strip()
    
    write_log_entry(
        "Main wrapper",
        "client.chat.completions.create",
        {"response": raw_answer},
    )

    # Try to extract JSON between braces if the model added text
    if "{" in raw_answer and "}" in raw_answer:
        json_part = raw_answer[raw_answer.find("{") : raw_answer.rfind("}") + 1]
        try:
            json.loads(json_part)
            raw_answer = json_part
        except Exception:
            pass  # leave as-is if still invalid

    return raw_answer

if __name__ == "__main__":
    print(f"Loaded {len(MACHINE_NAMES)} machine names. Primary: {MACHINE_NAME}")
    
    # Confirm effective cache directory
    print("Hugging Face cache directory:", os.environ["HF_HOME"])

    print(f"2) {build_all(cfg, MACHINE_NAMES)} # may take time")

    #For DEBUGGING:
    #print("To run:")
    #print("1) Put PDFs in:", cfg.pdf_dir)
    #print(f"2) {build_all(cfg)} # may take time")
    #print(f"3) {retrieve_plan('Emergency stop procedure for station X', cfg)}")
    #query =f"I see in the panel in {MACHINE_NAME} a message: 'Please refill red pellets', what should I do?"
    #query =f"What can you tell me about machine SIF401?"
    #plan = retrieve_plan(query, cfg)
    #print(f"3) {plan}")

