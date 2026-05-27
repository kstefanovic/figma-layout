import base64
import binascii
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Literal

import requests
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from figma_semantic import (
    FIGMA_CONVERT_PROMPT,
    FIGMA_CONVERT_SYSTEM_PROMPT,
    TOP_LEVEL_SEMANTIC_SYSTEM_PROMPT,
    apply_top_level_semantic_names_to_raw,
    apply_semantic_names,
    build_top_level_semantic_user_text,
    build_naming_user_prompt,
    chunk_list,
    collect_allowed_ids_from_mid,
    extract_first_json_value,
    fill_missing_top_level_names,
    flatten_raw_to_mid,
    mid_node_prompt_slice,
    missing_name_ids,
    normalize_convert_semantic_output,
    postprocess_top_level_semantic_names,
    parse_top_level_names_object,
    parse_names_object,
    raw_fig_tree_to_mid_blocks,
)
from figma_semantic_strict import (
    STRICT_QWEN_SYSTEM_PROMPT,
    assert_rich_metadata_preserved,
    build_qwen_ambiguous_payload,
    build_qwen_ambiguous_user_text,
    build_semantic_json_from_strict_names,
    extract_node_features,
    run_strict_semantic_naming,
)
from json_embedding import (
    MAX_CLASS_NUMBER,
    MIN_CLASS_NUMBER,
    VALID_CLASSES,
    attach_full_json,
    build_all_indexes,
    parse_aspect_ratio,
    parse_resolution,
    rerank_candidates_by_raw_similarity,
    resize_source_json_using_guide,
    search_index,
    select_frame,
)
from layout_engine.convert import convert_banner
from layout_training.model.inference import predict_top_level_layout_json
from layout_transformer_v2.src.predict import LayoutTransformerV2Service
from layout_transformer_v2.src.prototypes import load_prototypes, select_prototype
from layout_transformer_v2.src.rich_utils import load_frames
from layout_transformer_v2.src.schema import ALL_ROLES as LAYOUT_TRANSFORMER_V2_ROLES


load_dotenv()

MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://127.0.0.1:20400")
HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
PORT = int(os.getenv("BACKEND_PORT", "20401"))
REQUEST_TIMEOUT = float(os.getenv("MODEL_REQUEST_TIMEOUT", "300"))
CATEGORY_PROMPT = os.getenv(
    "CATEGORY_PROMPT",
    "You are a vision classifier. Look at the image and output exactly one short category "
    "label (2–6 words, no punctuation, no quotes, no explanation). Examples: outdoor landscape, "
    "product packaging, UI screenshot, document scan, chart diagram, people portrait, animal, food.",
)
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PROMPT_MAX_LEN = 8000
FIGMA_MAX_JSON_BYTES = int(os.getenv("FIGMA_MAX_JSON_BYTES", str(52 * 1024 * 1024)))
FIGMA_SEMANTIC_MAX_CHUNKS = int(os.getenv("FIGMA_SEMANTIC_MAX_CHUNKS", "80"))
FIGMA_CONVERT_TIMEOUT = float(os.getenv("FIGMA_CONVERT_TIMEOUT", str(max(REQUEST_TIMEOUT, 900.0))))
FIGMA_SEMANTIC_RUNS_DIR = Path(
    os.getenv("FIGMA_SEMANTIC_RUNS_DIR", "runs/figma_convert_semantic_json"),
).resolve()
FIGMA_LAYOUT_TRANSFORMER_RUNS_DIR = Path(
    os.getenv("FIGMA_LAYOUT_TRANSFORMER_RUNS_DIR", "runs/layout_transformer"),
).resolve()
FIGMA_BANNER_PIPELINE_RUNS_DIR = Path(
    os.getenv("FIGMA_BANNER_PIPELINE_RUNS_DIR", "runs/banner_pipeline"),
).resolve()
# Max long edge (px) for images sent to Qwen on /figma/convert-semantic-json (0 = disable).
FIGMA_SEMANTIC_BANNER_MAX_EDGE = int(os.getenv("FIGMA_SEMANTIC_BANNER_MAX_EDGE", "1024"))
FIGMA_SEMANTIC_GRID_MAX_EDGE = int(os.getenv("FIGMA_SEMANTIC_GRID_MAX_EDGE", "1024"))
FIGMA_TOP_LEVEL_CHILD_MAX_EDGE = int(os.getenv("FIGMA_TOP_LEVEL_CHILD_MAX_EDGE", "768"))
# When false (0/no/false), skip writing per-request run folders (faster I/O; no artifacts for debugging).
FIGMA_SEMANTIC_STRICT_MODE = os.getenv("FIGMA_SEMANTIC_STRICT_MODE", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
FIGMA_SEMANTIC_PERSIST_RUNS = os.getenv("FIGMA_SEMANTIC_PERSIST_RUNS", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
FIGMA_LAYOUT_TRANSFORMER_PERSIST_RUNS = os.getenv("FIGMA_LAYOUT_TRANSFORMER_PERSIST_RUNS", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
FIGMA_BANNER_PIPELINE_PERSIST_RUNS = os.getenv("FIGMA_BANNER_PIPELINE_PERSIST_RUNS", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
VISUAL_RETRIEVAL_DB = os.getenv(
    "VISUAL_RETRIEVAL_DB",
    "layout_engine/retrieval_db/visual_layout_db.json",
)
VISUAL_RETRIEVAL_TOP_K = int(os.getenv("VISUAL_RETRIEVAL_TOP_K", "15"))
GNN_LAYOUT_CHECKPOINT = os.getenv(
    "GNN_LAYOUT_CHECKPOINT",
    "gnn_layout/data/checkpoints/gnn_brand_headline_legal_smoke.pt",
)
LAYOUT_TRANSFORMER_PARENT_CHECKPOINT = Path(
    os.getenv("LAYOUT_TRANSFORMER_PARENT_CHECKPOINT", "layout_transformer_v2/checkpoints/parent.pt"),
).resolve()
LAYOUT_TRANSFORMER_CHILD_CHECKPOINT = Path(
    os.getenv("LAYOUT_TRANSFORMER_CHILD_CHECKPOINT", "layout_transformer_v2/checkpoints/child.pt"),
).resolve()
LAYOUT_TRANSFORMER_FLOATING_CHECKPOINT = Path(
    os.getenv("LAYOUT_TRANSFORMER_FLOATING_CHECKPOINT", "layout_transformer_v2/checkpoints/floating.pt"),
).resolve()
LAYOUT_TRANSFORMER_PROTOTYPES_PATH = Path(
    os.getenv("LAYOUT_TRANSFORMER_PROTOTYPES_PATH", "layout_transformer_v2/data/prototypes/layout_prototypes.json"),
).resolve()
LAYOUT_TRANSFORMER_RICH_FAMILIES_DIR = Path(
    os.getenv("LAYOUT_TRANSFORMER_RICH_FAMILIES_DIR", "layout_transformer/data/clean_families_rich"),
).resolve()
LAYOUT_TRANSFORMER_DEVICE = os.getenv("LAYOUT_TRANSFORMER_DEVICE", "").strip() or None
TOP_LEVEL_LAYOUT_MODEL_CHECKPOINT = os.getenv(
    "TOP_LEVEL_LAYOUT_MODEL_CHECKPOINT",
    "layout_training/checkpoints/top_level_layout_v1.pt",
)
TOP_LEVEL_LAYOUT_MODEL_DEVICE = os.getenv(
    "TOP_LEVEL_LAYOUT_MODEL_DEVICE",
    "auto",
)
TOP_LEVEL_LAYOUT_RETRIEVAL_ENABLED = os.getenv(
    "TOP_LEVEL_LAYOUT_RETRIEVAL_ENABLED",
    "1",
).lower() not in ("0", "false", "no")
TOP_LEVEL_LAYOUT_RETRIEVAL_RECORDS = os.getenv(
    "TOP_LEVEL_LAYOUT_RETRIEVAL_RECORDS",
    "layout_training/data/layout_records/top_level_records.jsonl",
)
TOP_LEVEL_LAYOUT_RETRIEVAL_K = int(os.getenv(
    "TOP_LEVEL_LAYOUT_RETRIEVAL_K",
    "5",
))
TOP_LEVEL_LAYOUT_RETRIEVAL_BLEND = os.getenv(
    "TOP_LEVEL_LAYOUT_RETRIEVAL_BLEND",
    "1",
).lower() not in ("0", "false", "no")
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
UNSUPPORTED_CLASS_NUMBER = -1
CLASS_TO_PROTOTYPE_FAMILY = {
    1: 2,  # Пряники прямо на ёлку
    2: 3,  # Пряничный ровер
    3: 1,  # Мегапорция оливье для гостей
    4: 4,  # Праздничная вишня в шоколаде
    5: 6,  # Имбирный пряничный латте
    6: 5,  # Еловый лимонад с малиной
}

BANNER_VLM_CATEGORY_PROMPT = """You classify one retail / food banner image into exactly ONE of six campaigns.

The six categories (match by main visual theme — product, hero, colors, headline; OCR need not be exact):

1. Пряники прямо на ёлку
2. Пряничный ровер
3. Мегапорция оливье для гостей
4. Праздничная вишня в шоколаде
5. Имбирный пряничный латте
6. Еловый лимонад с малиной

Output rules:
- If the banner does not belong to any of these six campaigns, reply with ONLY -1.
- Otherwise reply with ONLY the digit 1, 2, 3, 4, 5, or 6.
- Single line. No other words, no JSON, no markdown, no explanation."""


class ContentItem(BaseModel):
    type: Literal["text", "image", "video"]
    text: str | None = None
    image: str | None = None
    video: Any | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentItem]


class ChatRequest(BaseModel):
    prompt: str | None = None
    image: str | None = None
    images: list[str] = Field(default_factory=list)
    video: Any | None = None
    messages: list[ChatMessage] | None = None
    max_new_tokens: int = Field(default=256, ge=1, le=8192)


class ChatResponse(BaseModel):
    response: str


class CategorizeItem(BaseModel):
    filename: str
    category: str | None = None
    error: str | None = None


class CategorizeResponse(BaseModel):
    results: list[CategorizeItem]


class FigmaSemanticMidResponse(BaseModel):
    mid_json: list[dict[str, Any]]
    semantic_mid_json: list[dict[str, Any]]
    node_count: int
    chunks_used: int
    frame_index: int
    used_reference_grid: bool = False
    warnings: list[str] = Field(default_factory=list)


class FigmaConvertSemanticResponse(BaseModel):
    """Single-shot full raw JSON → semantic JSON (model rewrites tree and names)."""

    semantic_json: Any
    warnings: list[str] = Field(default_factory=list)
    semantic_debug: dict[str, Any] | None = Field(
        default=None,
        description="Strict namer debug: prelabel, qwen, conflicts, validator (when FIGMA_SEMANTIC_STRICT_MODE=1)",
    )


class BannerCategoryResponse(BaseModel):
    """VLM banner classification into campaign 1–6, or -1 for unsupported banners."""

    category: int = Field(
        ...,
        ge=UNSUPPORTED_CLASS_NUMBER,
        le=MAX_CLASS_NUMBER,
        description="Campaign index per product brief, or -1 when unsupported",
    )
    raw_model_text: str = Field(default="", description="Trimmed model output used for parsing")


class JsonEmbeddingBuildItem(BaseModel):
    class_number: int
    count: int
    source_file: str


class JsonEmbeddingBuildResponse(BaseModel):
    indexes: list[JsonEmbeddingBuildItem]


class JsonEmbeddingCandidate(BaseModel):
    class_number: int
    source_file: str
    frame_index: int
    id: str | None = None
    name: str | None = None
    type: str | None = None
    bounds: dict[str, Any] | None = None
    aspect_ratio: float | None = None
    node_count: int
    leaf_count: int
    score: float
    embedding_score: float
    aspect_error: float
    resolution_error: float | None = None
    raw_similarity: float | None = None
    selection_score: float | None = None
    full_json: dict[str, Any] | None = None


class JsonEmbeddingSearchResponse(BaseModel):
    class_number: int
    aspect_ratio: float
    top_k: int
    candidates: list[JsonEmbeddingCandidate]


class JsonEmbeddingSearchByRawRequest(BaseModel):
    class_number: int = Field(
        ...,
        ge=MIN_CLASS_NUMBER,
        le=MAX_CLASS_NUMBER,
        description="Select one of the campaign class indexes",
    )
    raw_json: Any = Field(..., description="Root frame dict or array of frames from the Figma plugin")
    target_resolution: str | None = Field(None, description="WIDTHxHEIGHT or aspect ratio")
    target_width: float | None = None
    target_height: float | None = None
    raw_frame_index: int = Field(default=0, ge=0)
    top_k: int = Field(default=3, ge=1, le=20)
    include_full_json: bool = True


class JsonEmbeddingSearchByRawResponse(JsonEmbeddingSearchResponse):
    raw_frame_index: int
    selected_candidate: JsonEmbeddingCandidate


class BannerSearchPipelineResponse(BaseModel):
    category: int = Field(..., ge=UNSUPPORTED_CLASS_NUMBER, le=MAX_CLASS_NUMBER)
    raw_model_text: str = ""
    aspect_ratio: float
    top_k: int
    candidates: list[JsonEmbeddingCandidate]
    supported: bool = True


class BannerRawToTargetJsonResponse(BaseModel):
    category: int = Field(..., ge=UNSUPPORTED_CLASS_NUMBER, le=MAX_CLASS_NUMBER)
    raw_model_text: str = ""
    target_width: float
    target_height: float
    aspect_ratio: float
    top_k: int
    supported: bool = True
    message: str = ""
    run_id: str | None = None
    run_dir: str | None = None
    selected_candidate: JsonEmbeddingCandidate | None = None
    candidates: list[JsonEmbeddingCandidate]
    final_json: dict[str, Any] | None = None


class BannerRawToTargetJsonJsonRequest(BaseModel):
    banner_png_base64: str
    raw_json: Any
    target_resolution: str | None = None
    target_width: float | None = None
    target_height: float | None = None
    raw_frame_index: int = Field(default=0, ge=0)
    top_k: int = Field(default=3, ge=1, le=20)
    max_new_tokens: int = Field(default=64, ge=8, le=512)


class LayoutEngineConvertRequest(BaseModel):
    """Run CP-SAT layout_engine on plugin-serialized banner JSON (no VLM / embedding index)."""

    raw_json: Any = Field(..., description="Root frame dict or single-element list from the Figma plugin")
    target_resolution: str | None = Field(None, description="WIDTHxHEIGHT, e.g. 1536x640")
    target_width: float | None = None
    target_height: float | None = None
    visual_mode: Literal["default", "retrieval"] = "default"
    visual_retrieval_db: str | None = None
    visual_retrieval_top_k: int = Field(default=15, ge=1, le=100)
    gnn_layout_checkpoint: str | None = None


class LayoutEngineConvertResponse(BaseModel):
    final_json: dict[str, Any]
    target_width: int
    target_height: int


class LayoutTransformerRequest(BaseModel):
    source_json: dict[str, Any]
    target_width: int = Field(..., ge=1)
    target_height: int = Field(..., ge=1)


class LayoutTransformerDebug(BaseModel):
    target_width: int
    target_height: int
    model_roles: list[str]
    postprocess_mode: str | None = None
    prototype_id: str | None = None
    prototype_match_score: float | None = None
    corrections_applied: dict[str, bool] | None = None
    postprocess_report: dict[str, Any] | None = None


class LayoutTransformerResponse(BaseModel):
    final_json: dict[str, Any]
    engine: Literal["layout_transformer_v2_multi_model"] = "layout_transformer_v2_multi_model"
    debug: LayoutTransformerDebug


class FigmaTopLevelLayoutPredictRequest(BaseModel):
    semantic_json: dict[str, Any]
    target_width: int = Field(..., ge=1, le=20000)
    target_height: int = Field(..., ge=1, le=20000)
    checkpoint: str | None = None
    retrieval_enabled: bool | None = None
    retrieval_k: int | None = Field(default=None, ge=1, le=100)


class FigmaTopLevelLayoutPredictResponse(BaseModel):
    final_json: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="Public Qwen2.5-VL Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

layout_transformer_service: LayoutTransformerV2Service | None = None


@app.on_event("startup")
def load_layout_transformer_service() -> None:
    global layout_transformer_service
    if layout_transformer_service is not None:
        return
    missing = [
        str(path)
        for path in (
            LAYOUT_TRANSFORMER_PARENT_CHECKPOINT,
            LAYOUT_TRANSFORMER_CHILD_CHECKPOINT,
            LAYOUT_TRANSFORMER_FLOATING_CHECKPOINT,
        )
        if not path.exists()
    ]
    if missing:
        raise RuntimeError(f"Layout Transformer V2 checkpoint(s) not found: {missing}")
    layout_transformer_service = LayoutTransformerV2Service(
        parent_checkpoint=LAYOUT_TRANSFORMER_PARENT_CHECKPOINT,
        child_checkpoint=LAYOUT_TRANSFORMER_CHILD_CHECKPOINT,
        floating_checkpoint=LAYOUT_TRANSFORMER_FLOATING_CHECKPOINT,
        device=LAYOUT_TRANSFORMER_DEVICE,
    )


def _model_url(path: str) -> str:
    return f"{MODEL_SERVICE_URL.rstrip('/')}/{path.lstrip('/')}"


def _resolve_top_level_layout_checkpoint_path(value: str | None) -> Path:
    raw = (value or "").strip() or TOP_LEVEL_LAYOUT_MODEL_CHECKPOINT
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / path).resolve()
    return path


def _resolve_top_level_layout_retrieval_records_path(value: str | None) -> Path:
    raw = (value or "").strip() or TOP_LEVEL_LAYOUT_RETRIEVAL_RECORDS
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / path).resolve()
    return path


def _content_from_request(request: ChatRequest) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []

    for image in [request.image, *request.images]:
        if image:
            content.append({"type": "image", "image": image})

    if request.video is not None:
        content.append({"type": "video", "video": request.video})

    if request.prompt:
        content.append({"type": "text", "text": request.prompt})

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Provide either messages or at least one of prompt, image, images, or video.",
        )

    return content


def _messages_for_model(request: ChatRequest) -> list[dict[str, Any]]:
    if request.messages:
        return [message.model_dump(exclude_none=True) for message in request.messages]

    return [{"role": "user", "content": _content_from_request(request)}]


def _call_model(payload: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
    t = REQUEST_TIMEOUT if timeout is None else timeout
    try:
        response = requests.post(
            _model_url("/generate"),
            json=payload,
            timeout=t,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        detail = str(exc)
        if exc.response is not None:
            detail = exc.response.text
        raise HTTPException(status_code=502, detail=f"Model service error: {detail}") from exc

    return response.json()


def _data_uri(content: bytes, content_type: str | None) -> str:
    mime_type = content_type or "application/octet-stream"
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _resize_raster_max_long_edge(data: bytes, max_edge: int) -> tuple[bytes, dict[str, Any]]:
    """
    Downscale a raster image so max(width, height) <= max_edge.
    Returns PNG bytes when resized; otherwise original bytes. ``info`` describes the transform.
    """
    info: dict[str, Any] = {"max_edge": max_edge, "resized": False}
    if max_edge <= 0 or not data:
        return data, info
    try:
        with Image.open(BytesIO(data)) as im:
            im.load()
            w, h = im.size
            info["original_px"] = {"w": w, "h": h}
            longest = max(w, h)
            if longest <= max_edge:
                info["model_px"] = {"w": w, "h": h}
                return data, info

            if im.mode in ("RGBA", "LA"):
                rgba = im.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.getchannel("A"))
                im = background
            elif im.mode != "RGB":
                im = im.convert("RGB")

            scale = max_edge / longest
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            im = im.resize((nw, nh), Image.Resampling.BILINEAR)
            out = BytesIO()
            im.save(out, format="PNG", optimize=True)
            resized = out.getvalue()
            info["resized"] = True
            info["model_px"] = {"w": nw, "h": nh}
            return resized, info
    except Exception as exc:
        info["resize_error"] = str(exc)
        info["model_px"] = info.get("original_px")
        return data, info


def _new_figma_semantic_run_id() -> str:
    """UTC wall time + random suffix so each call has a unique, sortable folder name."""
    ts = datetime.now(timezone.utc)
    return f"{ts.strftime('%Y%m%dT%H%M%S')}_{ts.microsecond:06d}Z_{uuid.uuid4().hex[:10]}"


def _persist_figma_semantic_run_inputs(
    run_dir: Path,
    *,
    run_id: str,
    max_new_tokens: int,
    banner_body: bytes,
    grid_body: bytes,
    raw_bytes: bytes,
    banner_content_type: str | None,
    grid_content_type: str | None,
    banner_filename: str | None,
    raw_json_filename: str | None,
    grid_filename: str | None,
) -> dict[str, Any]:
    """Write banner, grid, and raw JSON bytes under ``run_dir``; return meta fields (without status)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "input_banner.png").write_bytes(banner_body)
    (run_dir / "input_grid.png").write_bytes(grid_body)
    (run_dir / "input_raw.json").write_bytes(raw_bytes)
    started = datetime.now(timezone.utc).isoformat()
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "endpoint": "/figma/convert-semantic-json",
        "started_at_utc": started,
        "max_new_tokens": max_new_tokens,
        "upload_filenames": {
            "banner": (banner_filename or "").strip() or None,
            "raw_json": (raw_json_filename or "").strip() or None,
            "grid": (grid_filename or "").strip() or None,
        },
        "input_files": {
            "input_banner.png": {"bytes": len(banner_body), "content_type": banner_content_type or ""},
            "input_grid.png": {"bytes": len(grid_body), "content_type": grid_content_type or ""},
            "input_raw.json": {"bytes": len(raw_bytes)},
        },
    }


def _write_figma_semantic_run_meta(run_dir: Path, meta: dict[str, Any]) -> None:
    payload = dict(meta)
    payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "meta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _persist_layout_transformer_run_input(
    run_dir: Path,
    *,
    run_id: str,
    request: LayoutTransformerRequest,
    endpoint: str = "/api/layout-transformer",
) -> dict[str, Any]:
    """Write the Layout Transformer plugin request under ``run_dir`` and return base metadata."""
    run_dir.mkdir(parents=True, exist_ok=True)
    input_payload = request.model_dump(mode="json")
    (run_dir / "input.json").write_text(
        json.dumps(input_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "endpoint": endpoint,
        "engine": "layout_transformer_v2_multi_model",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_width": request.target_width,
        "target_height": request.target_height,
        "checkpoints": {
            "parent": str(LAYOUT_TRANSFORMER_PARENT_CHECKPOINT),
            "child": str(LAYOUT_TRANSFORMER_CHILD_CHECKPOINT),
            "floating": str(LAYOUT_TRANSFORMER_FLOATING_CHECKPOINT),
        },
        "input_files": {
            "input.json": {"bytes": (run_dir / "input.json").stat().st_size},
        },
    }


def _write_layout_transformer_run_meta(run_dir: Path, meta: dict[str, Any]) -> None:
    payload = dict(meta)
    payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "meta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_layout_transformer_run_response(
    run_dir: Path,
    *,
    filename: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    path = run_dir / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {filename: {"bytes": path.stat().st_size}}


def _persist_banner_pipeline_run_input(
    run_dir: Path,
    *,
    run_id: str,
    endpoint: str,
    banner_body: bytes,
    banner_content_type: str | None,
    raw_json_payload: Any,
    target_resolution: str,
    raw_frame_index: int,
    top_k: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "input_banner.png").write_bytes(banner_body)
    input_payload = {
        "raw_json": raw_json_payload,
        "target_resolution": target_resolution,
        "raw_frame_index": raw_frame_index,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
    }
    (run_dir / "input.json").write_text(
        json.dumps(input_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "endpoint": endpoint,
        "engine": "banner_classify_layout_transformer_v2_prototype_retrieval",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_resolution": target_resolution,
        "raw_frame_index": raw_frame_index,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "prototypes_path": str(LAYOUT_TRANSFORMER_PROTOTYPES_PATH),
        "rich_families_dir": str(LAYOUT_TRANSFORMER_RICH_FAMILIES_DIR),
        "input_files": {
            "input_banner.png": {"bytes": len(banner_body), "content_type": banner_content_type or ""},
            "input.json": {"bytes": (run_dir / "input.json").stat().st_size},
        },
    }


def _write_banner_pipeline_run_meta(run_dir: Path, meta: dict[str, Any]) -> None:
    payload = dict(meta)
    payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "meta.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_banner_pipeline_run_response(
    run_dir: Path,
    *,
    response: BannerRawToTargetJsonResponse | dict[str, Any],
) -> dict[str, Any]:
    payload = response.model_dump(mode="json") if isinstance(response, BaseModel) else response
    path = run_dir / "response.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"response.json": {"bytes": path.stat().st_size}}


def _decode_base64_bytes(value: str, field_name: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} is not valid base64") from exc


def _is_png(data: bytes, content_type: str | None) -> bool:
    if content_type == "image/png":
        return True
    return len(data) >= len(PNG_MAGIC) and data[: len(PNG_MAGIC)] == PNG_MAGIC


def _normalize_category(text: str) -> str:
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return line.strip()[:200] if line else ""


def _parse_banner_category(text: str) -> int:
    """Extract campaign index from VLM reply, allowing -1 for unsupported banners."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty model response")

    inner = raw
    fence = re.search(r"```(?:json|text)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fence:
        inner = fence.group(1).strip()

    def _in_range(n: int) -> bool:
        return n == UNSUPPORTED_CLASS_NUMBER or MIN_CLASS_NUMBER <= n <= MAX_CLASS_NUMBER

    if inner.startswith("{") or inner.startswith("["):
        try:
            parsed = extract_first_json_value(inner)
        except (ValueError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            for key in ("category", "class", "label", "id", "campaign"):
                v = parsed.get(key)
                if isinstance(v, bool):
                    continue
                if isinstance(v, int) and _in_range(v):
                    return v
                if isinstance(v, str) and re.fullmatch(r"-?\d+", v.strip()):
                    n = int(v.strip())
                    if _in_range(n):
                        return n
        if isinstance(parsed, list) and len(parsed) == 1:
            only = parsed[0]
            if isinstance(only, int) and _in_range(only):
                return only

    for line in inner.splitlines():
        s = line.strip()
        if re.fullmatch(r"-1", s):
            return UNSUPPORTED_CLASS_NUMBER
        if re.fullmatch(rf"[{MIN_CLASS_NUMBER}-{MAX_CLASS_NUMBER}]", s):
            return int(s)

    m = re.search(r"(?<!\d)(-1)(?!\d)", inner)
    if m:
        return UNSUPPORTED_CLASS_NUMBER

    m = re.search(rf"\b([{MIN_CLASS_NUMBER}-{MAX_CLASS_NUMBER}])\b", inner)
    if m:
        return int(m.group(1))

    raise ValueError(
        f"no class digit -1 or {MIN_CLASS_NUMBER}–{MAX_CLASS_NUMBER} found in: {inner[:300]!r}"
    )


def _classify_banner_bytes(body: bytes, content_type: str | None, max_new_tokens: int = 64) -> tuple[int, str]:
    if not body:
        raise HTTPException(status_code=400, detail="Empty file.")

    image = _data_uri(body, content_type)
    request = ChatRequest(
        prompt=BANNER_VLM_CATEGORY_PROMPT,
        image=image,
        max_new_tokens=max_new_tokens,
    )
    result = _call_model(
        {
            "messages": _messages_for_model(request),
            "max_new_tokens": max_new_tokens,
        }
    )
    raw = (result.get("response") or "").strip()
    try:
        category = _parse_banner_category(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"VLM output could not be parsed as -1 or {MIN_CLASS_NUMBER}–{MAX_CLASS_NUMBER}: {exc}. Raw: {raw[:500]!r}",
        ) from exc
    return category, raw


def _prototype_class_number(proto: dict[str, Any]) -> int | None:
    for value in (proto.get("prototype_id"), proto.get("source_file")):
        text = str(value or "").strip()
        match = re.match(r"^(\d+)(?::|[_-])", text)
        if match:
            number = int(match.group(1))
            if number in VALID_CLASSES:
                return number
    return None


def _load_prototype_guide_frame(proto: dict[str, Any]) -> tuple[dict[str, Any], Path, int]:
    source_file = str(proto.get("source_file") or "").strip()
    if not source_file:
        raise ValueError("selected prototype is missing source_file")
    source_path = Path(source_file)
    if not source_path.is_absolute():
        source_path = LAYOUT_TRANSFORMER_RICH_FAMILIES_DIR / source_path
    if not source_path.exists():
        raise FileNotFoundError(f"prototype source JSON not found: {source_path}")

    frame_index = 0
    proto_id = str(proto.get("prototype_id") or "").strip()
    match = re.match(r"^\d+:(\d+)$", proto_id)
    if match:
        frame_index = int(match.group(1))

    frames = load_frames(source_path)
    if frame_index < 0 or frame_index >= len(frames):
        raise ValueError(f"prototype frame index {frame_index} is out of range for {source_path}")
    return frames[frame_index], source_path, frame_index


def _node_count(node: Any) -> int:
    if not isinstance(node, dict):
        return 0
    return 1 + sum(_node_count(child) for child in node.get("children") or [])


def _leaf_count(node: Any) -> int:
    if not isinstance(node, dict):
        return 0
    children = [child for child in node.get("children") or [] if isinstance(child, dict)]
    if not children:
        return 1
    return sum(_leaf_count(child) for child in children)


def _candidate_from_prototype(
    proto: dict[str, Any],
    guide_frame: dict[str, Any],
    source_path: Path,
    frame_index: int,
    class_number: int,
) -> dict[str, Any]:
    bounds = guide_frame.get("bounds") if isinstance(guide_frame.get("bounds"), dict) else {}
    width = bounds.get("width") if isinstance(bounds, dict) else None
    height = bounds.get("height") if isinstance(bounds, dict) else None
    aspect_ratio = (
        float(width) / float(height)
        if isinstance(width, int | float) and isinstance(height, int | float) and height
        else proto.get("aspect")
    )
    score = float(proto.get("match_score") or 0.0)
    return {
        "class_number": class_number,
        "source_file": str(source_path),
        "frame_index": frame_index,
        "id": guide_frame.get("id"),
        "name": guide_frame.get("name") or proto.get("frame_name"),
        "type": guide_frame.get("type"),
        "bounds": bounds if isinstance(bounds, dict) else None,
        "aspect_ratio": aspect_ratio,
        "node_count": _node_count(guide_frame),
        "leaf_count": _leaf_count(guide_frame),
        "score": score,
        "embedding_score": score,
        "aspect_error": abs(float(proto.get("aspect") or 1.0) - float(aspect_ratio or 1.0)),
        "resolution_error": None,
        "raw_similarity": None,
        "selection_score": score,
        "full_json": guide_frame,
    }


def _select_layout_transformer_prototype_candidate(
    *,
    class_number: int,
    uploaded_frame: dict[str, Any],
    target_width: float,
    target_height: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prototypes = load_prototypes(LAYOUT_TRANSFORMER_PROTOTYPES_PATH)
    prototype_family = CLASS_TO_PROTOTYPE_FAMILY.get(class_number, class_number)
    class_prototypes = [
        proto for proto in prototypes
        if _prototype_class_number(proto) == prototype_family
    ]
    if not class_prototypes:
        raise FileNotFoundError(
            f"No layout transformer prototypes found for class {class_number} "
            f"(prototype family {prototype_family}) in {LAYOUT_TRANSFORMER_PROTOTYPES_PATH}"
        )

    selected_proto = _select_banner_pipeline_prototype(
        class_prototypes,
        source_json=uploaded_frame,
        target_width=target_width,
        target_height=target_height,
    )
    if selected_proto is None:
        raise FileNotFoundError(f"No matching prototype found for class {class_number}")

    guide_frame, source_path, frame_index = _load_prototype_guide_frame(selected_proto)
    candidate = _candidate_from_prototype(
        selected_proto,
        guide_frame,
        source_path,
        frame_index,
        class_number,
    )
    return selected_proto, candidate


def _select_banner_pipeline_prototype(
    prototypes: list[dict[str, Any]],
    *,
    source_json: dict[str, Any],
    target_width: float,
    target_height: float,
) -> dict[str, Any] | None:
    if not prototypes:
        return None

    structural = select_prototype(
        prototypes,
        source_json=source_json,
        target_width=target_width,
        target_height=target_height,
    )
    structural_id = structural.get("prototype_id") if structural else None
    target_aspect = target_width / max(target_height, 1.0)

    def score(proto: dict[str, Any]) -> float:
        width = float(proto.get("width") or 0.0)
        height = float(proto.get("height") or 0.0)
        aspect = float(proto.get("aspect") or (width / height if height else 1.0))
        aspect_error = abs(math.log(max(aspect, 1e-6) / max(target_aspect, 1e-6)))
        if width > 0 and height > 0:
            resolution_error = math.sqrt(
                math.log(width / target_width) ** 2 + math.log(height / target_height) ** 2
            )
        else:
            resolution_error = 10.0
        exact = abs(width - target_width) <= 0.5 and abs(height - target_height) <= 0.5
        orientation_match = (height > width * 1.05) == (target_height > target_width * 1.05)
        structural_bonus = 1.0 if structural_id and proto.get("prototype_id") == structural_id else 0.0
        return (
            (10000.0 if exact else 0.0)
            + (250.0 if orientation_match else 0.0)
            - 500.0 * aspect_error
            - 25.0 * resolution_error
            + structural_bonus
        )

    best = max(prototypes, key=score)
    selected = dict(best)
    selected["match_score"] = float(score(best))
    selected["structural_prototype_id"] = structural_id
    return selected


def _banner_pipeline_layout_transformer_fallback(
    uploaded_frame: dict[str, Any],
    target_width: float,
    target_height: float,
) -> dict[str, Any] | None:
    """When VLM returns -1, still produce a target layout via the loaded V2 models."""
    if layout_transformer_service is None:
        return None
    try:
        return layout_transformer_service.predict(uploaded_frame, target_width, target_height)
    except Exception as exc:
        print(f"[banner_pipeline] layout_transformer_v2 fallback failed: {exc}")
        return None


def _run_banner_raw_to_target_pipeline(
    *,
    banner_body: bytes,
    banner_content_type: str | None,
    uploaded_raw: Any,
    target_resolution: str,
    raw_frame_index: int,
    top_k: int,
    max_new_tokens: int,
    run_id: str | None = None,
    run_dir: Path | None = None,
) -> BannerRawToTargetJsonResponse:
    category, raw_model_text = _classify_banner_bytes(
        banner_body, banner_content_type, max_new_tokens=max_new_tokens
    )
    try:
        uploaded_frame = select_frame(uploaded_raw, raw_frame_index)
        target_width, target_height = parse_resolution(target_resolution)
        parsed_aspect = target_width / target_height
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if category == UNSUPPORTED_CLASS_NUMBER:
        fallback_json = _banner_pipeline_layout_transformer_fallback(
            uploaded_frame,
            target_width,
            target_height,
        )
        if fallback_json is not None:
            return BannerRawToTargetJsonResponse(
                category=category,
                raw_model_text=raw_model_text[:2000],
                target_width=target_width,
                target_height=target_height,
                aspect_ratio=parsed_aspect,
                top_k=top_k,
                supported=True,
                message=(
                    "VLM class -1 (not in 6 campaigns); used layout_transformer_v2 "
                    "direct prediction fallback."
                ),
                run_id=run_id,
                run_dir=str(run_dir) if run_dir is not None else None,
                selected_candidate=None,
                candidates=[],
                final_json=fallback_json,
            )
        return BannerRawToTargetJsonResponse(
            category=category,
            raw_model_text=raw_model_text[:2000],
            target_width=target_width,
            target_height=target_height,
            aspect_ratio=parsed_aspect,
            top_k=top_k,
            supported=False,
            message="This banner is not supported by the current 6-class retrieval pipeline.",
            run_id=run_id,
            run_dir=str(run_dir) if run_dir is not None else None,
            selected_candidate=None,
            candidates=[],
            final_json=None,
        )

    try:
        selected_proto, selected = _select_layout_transformer_prototype_candidate(
            class_number=category,
            uploaded_frame=uploaded_frame,
            target_width=target_width,
            target_height=target_height,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    selected_json = selected.get("full_json")
    if not isinstance(selected_json, dict):
        raise HTTPException(status_code=500, detail="Selected candidate does not include full_json.")

    final_json = resize_source_json_using_guide(
        uploaded_frame,
        selected_json,
        target_width,
        target_height,
    )

    return BannerRawToTargetJsonResponse(
        category=category,
        raw_model_text=raw_model_text[:2000],
        target_width=target_width,
        target_height=target_height,
        aspect_ratio=parsed_aspect,
        top_k=top_k,
        supported=True,
        message=f"Selected layout_transformer_v2 prototype {selected_proto.get('prototype_id')}",
        run_id=run_id,
        run_dir=str(run_dir) if run_dir is not None else None,
        selected_candidate=JsonEmbeddingCandidate(**selected),
        candidates=[JsonEmbeddingCandidate(**selected)],
        final_json=final_json,
    )


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        model_health = requests.get(_model_url("/health"), timeout=5).json()
    except requests.RequestException as exc:
        model_health = {"status": "unreachable", "error": str(exc)}

    return {
        "status": "ok",
        "model_service_url": MODEL_SERVICE_URL,
        "model": model_health,
        "layout_transformer": {
            "engine": "layout_transformer_v2_multi_model",
            "checkpoints": {
                "parent": str(LAYOUT_TRANSFORMER_PARENT_CHECKPOINT),
                "child": str(LAYOUT_TRANSFORMER_CHILD_CHECKPOINT),
                "floating": str(LAYOUT_TRANSFORMER_FLOATING_CHECKPOINT),
            },
            "loaded": layout_transformer_service is not None,
            "roles": (layout_transformer_service.model_roles if layout_transformer_service else LAYOUT_TRANSFORMER_V2_ROLES),
        },
    }


@app.post("/banner/category", response_model=BannerCategoryResponse)
async def banner_category(
    file: UploadFile = File(..., description="Banner PNG (or JPEG/WebP)"),
    max_new_tokens: int = Form(64, ge=8, le=512),
) -> BannerCategoryResponse:
    """VLM picks which Yandex Lavka-style campaign the banner belongs to (output 1–6)."""
    body = await file.read()
    category, raw = _classify_banner_bytes(body, file.content_type, max_new_tokens=max_new_tokens)

    return BannerCategoryResponse(category=category, raw_model_text=raw[:2000])


@app.post("/json-embeddings/build", response_model=JsonEmbeddingBuildResponse)
def build_json_embeddings() -> JsonEmbeddingBuildResponse:
    """Build one local embedding index per class: raw_jsons/{n}.json for each configured class."""
    try:
        indexes = build_all_indexes()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JsonEmbeddingBuildResponse(
        indexes=[
            JsonEmbeddingBuildItem(
                class_number=int(index["class_number"]),
                count=int(index["count"]),
                source_file=str(index["source_file"]),
            )
            for index in indexes
        ]
    )


@app.get("/json-embeddings/search", response_model=JsonEmbeddingSearchResponse)
def search_json_embeddings(
    class_number: int = Query(
        ...,
        ge=MIN_CLASS_NUMBER,
        le=MAX_CLASS_NUMBER,
        description="Select one of the campaign class indexes",
    ),
    aspect_ratio: str = Query(..., description="Aspect ratio: 16:9, 1080x1920, 1.777, etc."),
    top_k: int = Query(3, ge=1, le=20),
    include_full_json: bool = Query(True, description="Attach the full retrieved top-level Figma JSON frame"),
) -> JsonEmbeddingSearchResponse:
    """Select class embedding, then retrieve top candidates by requested aspect ratio."""
    if class_number not in VALID_CLASSES:
        raise HTTPException(status_code=400, detail=f"class_number must be one of {sorted(VALID_CLASSES)}")
    try:
        parsed_aspect = parse_aspect_ratio(aspect_ratio)
        results = search_index(class_number, aspect_ratio, top_k=top_k)
        if include_full_json:
            results = attach_full_json(results)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JsonEmbeddingSearchResponse(
        class_number=class_number,
        aspect_ratio=parsed_aspect,
        top_k=top_k,
        candidates=[JsonEmbeddingCandidate(**row) for row in results],
    )


@app.post("/json-embeddings/search-by-raw-json", response_model=JsonEmbeddingSearchByRawResponse)
def search_json_embeddings_by_raw_json(
    request: JsonEmbeddingSearchByRawRequest,
) -> JsonEmbeddingSearchByRawResponse:
    """Retrieve by class/resolution, then rerank against an uploaded raw Figma frame."""
    if request.class_number not in VALID_CLASSES:
        raise HTTPException(
            status_code=400,
            detail=f"class_number must be one of {sorted(VALID_CLASSES)}",
        )

    target_resolution = request.target_resolution
    if not target_resolution:
        if request.target_width is None or request.target_height is None:
            raise HTTPException(
                status_code=400,
                detail="Provide target_resolution or both target_width and target_height.",
            )
        target_resolution = f"{request.target_width}x{request.target_height}"

    try:
        parsed_aspect = parse_aspect_ratio(target_resolution)
        raw_frame = select_frame(request.raw_json, request.raw_frame_index)
        results = search_index(request.class_number, target_resolution, top_k=request.top_k)
        if request.include_full_json:
            results = attach_full_json(results)
        reranked = rerank_candidates_by_raw_similarity(raw_frame, results)
        ranked_results = reranked or results
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not ranked_results:
        raise HTTPException(status_code=404, detail="No retrievable candidates found.")

    return JsonEmbeddingSearchByRawResponse(
        class_number=request.class_number,
        aspect_ratio=parsed_aspect,
        raw_frame_index=request.raw_frame_index,
        top_k=request.top_k,
        selected_candidate=JsonEmbeddingCandidate(**ranked_results[0]),
        candidates=[JsonEmbeddingCandidate(**row) for row in ranked_results],
    )


@app.post("/pipeline/banner-search", response_model=BannerSearchPipelineResponse)
async def classify_banner_then_search_json(
    file: UploadFile = File(..., description="Banner PNG/JPEG/WebP to classify"),
    target_resolution: str = Form(..., description="Target resolution or aspect ratio: 2280x360, 16:9, 1.777"),
    top_k: int = Form(3, ge=1, le=20),
    max_new_tokens: int = Form(64, ge=8, le=512),
    include_full_json: bool = Form(True),
) -> BannerSearchPipelineResponse:
    """One flow: classify banner into a campaign class, then search that class index by target resolution."""
    body = await file.read()
    category, raw = _classify_banner_bytes(body, file.content_type, max_new_tokens=max_new_tokens)

    try:
        parsed_aspect = parse_aspect_ratio(target_resolution)
        if category == UNSUPPORTED_CLASS_NUMBER:
            return BannerSearchPipelineResponse(
                category=category,
                raw_model_text=raw[:2000],
                aspect_ratio=parsed_aspect,
                top_k=top_k,
                candidates=[],
                supported=False,
            )
        results = search_index(category, target_resolution, top_k=top_k)
        if include_full_json:
            results = attach_full_json(results)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return BannerSearchPipelineResponse(
        category=category,
        raw_model_text=raw[:2000],
        aspect_ratio=parsed_aspect,
        top_k=top_k,
        candidates=[JsonEmbeddingCandidate(**row) for row in results],
        supported=True,
    )


@app.post("/pipeline/banner-raw-to-target-json", response_model=BannerRawToTargetJsonResponse)
async def banner_raw_to_target_json(
    file: UploadFile = File(..., description="Banner PNG/JPEG/WebP to classify"),
    raw_json: UploadFile = File(..., description="Source raw Figma JSON to compare against retrieved candidates"),
    target_resolution: str = Form(..., description="Exact target resolution, for example 2280x360"),
    raw_frame_index: int = Form(0, ge=0),
    top_k: int = Form(3, ge=1, le=20),
    max_new_tokens: int = Form(64, ge=8, le=512),
) -> BannerRawToTargetJsonResponse:
    """
    One flow:
    1. Classify banner into a campaign class (1–6).
    2. Retrieve top candidates from that class by target resolution/aspect.
    3. Rerank those candidates by similarity to uploaded raw JSON.
    4. Use the selected candidate as the layout guide and resize its bboxes to target resolution.
    """
    banner_body = await file.read()
    raw_bytes = await raw_json.read()
    if len(raw_bytes) > FIGMA_MAX_JSON_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"raw_json exceeds limit of {FIGMA_MAX_JSON_BYTES} bytes.",
        )
    try:
        uploaded_raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    run_id = _new_figma_semantic_run_id()
    run_dir = FIGMA_BANNER_PIPELINE_RUNS_DIR / run_id
    meta_base: dict[str, Any] | None = None
    if FIGMA_BANNER_PIPELINE_PERSIST_RUNS:
        try:
            meta_base = _persist_banner_pipeline_run_input(
                run_dir,
                run_id=run_id,
                endpoint="/pipeline/banner-raw-to-target-json",
                banner_body=banner_body,
                banner_content_type=file.content_type,
                raw_json_payload=uploaded_raw,
                target_resolution=target_resolution,
                raw_frame_index=raw_frame_index,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
            )
        except OSError as exc:
            print(f"[banner_pipeline] failed to persist input run_id={run_id}: {exc}")
            meta_base = None

    try:
        response = _run_banner_raw_to_target_pipeline(
            banner_body=banner_body,
            banner_content_type=file.content_type,
            uploaded_raw=uploaded_raw,
            target_resolution=target_resolution,
            raw_frame_index=raw_frame_index,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            run_id=run_id if meta_base else None,
            run_dir=run_dir if meta_base else None,
        )
    except HTTPException as exc:
        if meta_base:
            _write_banner_pipeline_run_response(run_dir, response={"error": exc.detail, "status_code": exc.status_code})
            _write_banner_pipeline_run_meta(run_dir, {**meta_base, "status": "error", "error": exc.detail})
        print(f"[banner_pipeline] error run_id={run_id}: {exc.detail}")
        raise

    if meta_base:
        output_files = _write_banner_pipeline_run_response(run_dir, response=response)
        selected = response.selected_candidate
        _write_banner_pipeline_run_meta(
            run_dir,
            {
                **meta_base,
                "status": "ok",
                "category": response.category,
                "raw_model_text": response.raw_model_text,
                "message": response.message,
                "selected_candidate": selected.model_dump(mode="json") if selected else None,
                "output_files": output_files,
            },
        )
    print(
        f"[banner_pipeline] ok run_id={run_id} category={response.category} "
        f"selected={response.selected_candidate.name if response.selected_candidate else None}"
    )
    return response


@app.post("/pipeline/banner-raw-to-target-json-json", response_model=BannerRawToTargetJsonResponse)
def banner_raw_to_target_json_json(request: BannerRawToTargetJsonJsonRequest) -> BannerRawToTargetJsonResponse:
    target_resolution = request.target_resolution
    if not target_resolution:
        if not request.target_width or not request.target_height:
            raise HTTPException(
                status_code=400,
                detail="Provide target_resolution or both target_width and target_height.",
            )
        target_resolution = f"{request.target_width}x{request.target_height}"
    banner_body = _decode_base64_bytes(request.banner_png_base64, "banner_png_base64")
    run_id = _new_figma_semantic_run_id()
    run_dir = FIGMA_BANNER_PIPELINE_RUNS_DIR / run_id
    meta_base: dict[str, Any] | None = None
    if FIGMA_BANNER_PIPELINE_PERSIST_RUNS:
        try:
            meta_base = _persist_banner_pipeline_run_input(
                run_dir,
                run_id=run_id,
                endpoint="/pipeline/banner-raw-to-target-json-json",
                banner_body=banner_body,
                banner_content_type="image/png",
                raw_json_payload=request.raw_json,
                target_resolution=target_resolution,
                raw_frame_index=request.raw_frame_index,
                top_k=request.top_k,
                max_new_tokens=request.max_new_tokens,
            )
        except OSError as exc:
            print(f"[banner_pipeline] failed to persist input run_id={run_id}: {exc}")
            meta_base = None

    try:
        response = _run_banner_raw_to_target_pipeline(
            banner_body=banner_body,
            banner_content_type="image/png",
            uploaded_raw=request.raw_json,
            target_resolution=target_resolution,
            raw_frame_index=request.raw_frame_index,
            top_k=request.top_k,
            max_new_tokens=request.max_new_tokens,
            run_id=run_id if meta_base else None,
            run_dir=run_dir if meta_base else None,
        )
    except HTTPException as exc:
        if meta_base:
            _write_banner_pipeline_run_response(run_dir, response={"error": exc.detail, "status_code": exc.status_code})
            _write_banner_pipeline_run_meta(run_dir, {**meta_base, "status": "error", "error": exc.detail})
        print(f"[banner_pipeline] error run_id={run_id}: {exc.detail}")
        raise

    if meta_base:
        output_files = _write_banner_pipeline_run_response(run_dir, response=response)
        selected = response.selected_candidate
        _write_banner_pipeline_run_meta(
            run_dir,
            {
                **meta_base,
                "status": "ok",
                "category": response.category,
                "raw_model_text": response.raw_model_text,
                "message": response.message,
                "selected_candidate": selected.model_dump(mode="json") if selected else None,
                "output_files": output_files,
            },
        )
    print(
        f"[banner_pipeline] ok run_id={run_id} category={response.category} "
        f"selected={response.selected_candidate.name if response.selected_candidate else None}"
    )
    return response


@app.post("/layout-engine/convert", response_model=LayoutEngineConvertResponse)
def layout_engine_convert(request: LayoutEngineConvertRequest) -> LayoutEngineConvertResponse:
    """
    Resize / relayout the selected banner using ``layout_engine.convert.convert_banner`` (OR-Tools CP-SAT).
    Expects the same serialized JSON shape the Figma plugin produces (semantic-ish layer names give best results).
    """
    if request.target_resolution:
        try:
            tw, th = parse_resolution(request.target_resolution)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    elif request.target_width is not None and request.target_height is not None:
        tw, th = float(request.target_width), float(request.target_height)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide target_resolution or both target_width and target_height.",
        )

    tw_i = max(1, int(round(tw)))
    th_i = max(1, int(round(th)))

    raw = request.raw_json
    if isinstance(raw, list):
        if not raw:
            raise HTTPException(status_code=400, detail="raw_json list is empty.")
        raw = raw[0]
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="raw_json must be a JSON object or a non-empty list.")

    retrieval_db = request.visual_retrieval_db
    if request.visual_mode == "retrieval" and not retrieval_db:
        retrieval_db = VISUAL_RETRIEVAL_DB
    retrieval_top_k = request.visual_retrieval_top_k or VISUAL_RETRIEVAL_TOP_K
    gnn_checkpoint = request.gnn_layout_checkpoint or GNN_LAYOUT_CHECKPOINT
    if gnn_checkpoint and not Path(gnn_checkpoint).exists():
        gnn_checkpoint = None

    try:
        final_json = convert_banner(
            raw,
            tw_i,
            th_i,
            visual_retrieval_db=retrieval_db,
            visual_retrieval_top_k=retrieval_top_k,
            visual_mode=request.visual_mode,
            gnn_layout_checkpoint=gnn_checkpoint,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"layout_engine.convert failed: {exc}",
        ) from exc

    return LayoutEngineConvertResponse(
        final_json=final_json,
        target_width=tw_i,
        target_height=th_i,
    )


@app.post("/api/layout-transformer", response_model=LayoutTransformerResponse)
@app.post("/api/layout-transformer-v2", response_model=LayoutTransformerResponse)
def layout_transformer_predict(
    request: LayoutTransformerRequest,
    fastapi_request: Request,
) -> LayoutTransformerResponse:
    """
    Predict a target clean semantic JSON from a source clean semantic JSON and target size.
    Uses the V2 multi-model rich Layout Transformer; no family id or template id is accepted or required.
    """
    run_id = _new_figma_semantic_run_id()
    run_dir = FIGMA_LAYOUT_TRANSFORMER_RUNS_DIR / run_id
    endpoint = fastapi_request.url.path
    meta_base: dict[str, Any] | None = None
    if FIGMA_LAYOUT_TRANSFORMER_PERSIST_RUNS:
        try:
            meta_base = _persist_layout_transformer_run_input(
                run_dir,
                run_id=run_id,
                request=request,
                endpoint=endpoint,
            )
        except OSError:
            meta_base = None

    if layout_transformer_service is None:
        if meta_base is not None:
            try:
                error_payload = {"detail": "Layout Transformer service is not loaded."}
                output_files = _write_layout_transformer_run_response(
                    run_dir,
                    filename="error_response.json",
                    payload=error_payload,
                )
                _write_layout_transformer_run_meta(
                    run_dir,
                    {
                        **meta_base,
                        "status": "error",
                        "error": "service_not_loaded",
                        "http_status": 503,
                        "output_files": output_files,
                    },
                )
            except OSError:
                pass
        raise HTTPException(status_code=503, detail="Layout Transformer service is not loaded.")

    try:
        final_json = layout_transformer_service.predict(
            request.source_json,
            request.target_width,
            request.target_height,
        )
    except ValueError as exc:
        if meta_base is not None:
            try:
                error_payload = {"detail": str(exc)}
                output_files = _write_layout_transformer_run_response(
                    run_dir,
                    filename="error_response.json",
                    payload=error_payload,
                )
                _write_layout_transformer_run_meta(
                    run_dir,
                    {
                        **meta_base,
                        "status": "error",
                        "error": "validation_failed",
                        "http_status": 422,
                        "output_files": output_files,
                    },
                )
            except OSError:
                pass
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        detail = f"layout transformer prediction failed: {exc}"
        if meta_base is not None:
            try:
                error_payload = {"detail": detail}
                output_files = _write_layout_transformer_run_response(
                    run_dir,
                    filename="error_response.json",
                    payload=error_payload,
                )
                _write_layout_transformer_run_meta(
                    run_dir,
                    {
                        **meta_base,
                        "status": "error",
                        "error": "prediction_failed",
                        "http_status": 500,
                        "output_files": output_files,
                    },
                )
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=detail) from exc

    response = LayoutTransformerResponse(
        final_json=final_json,
        engine="layout_transformer_v2_multi_model",
        debug=LayoutTransformerDebug(
            target_width=request.target_width,
            target_height=request.target_height,
            model_roles=layout_transformer_service.model_roles,
            postprocess_mode=layout_transformer_service.last_report.get("postprocess_mode"),
            prototype_id=layout_transformer_service.last_report.get("prototype_id"),
            prototype_match_score=layout_transformer_service.last_report.get("prototype_match_score"),
            corrections_applied=layout_transformer_service.last_report.get("corrections_applied"),
            postprocess_report=layout_transformer_service.last_report,
        ),
    )
    if meta_base is not None:
        try:
            output_files = _write_layout_transformer_run_response(
                run_dir,
                filename="response.json",
                payload=response.model_dump(mode="json"),
            )
            _write_layout_transformer_run_meta(
                run_dir,
                {
                    **meta_base,
                    "status": "ok",
                    "output_files": output_files,
                },
            )
        except OSError:
            pass
    return response


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = _call_model(
        {
            "messages": _messages_for_model(request),
            "max_new_tokens": request.max_new_tokens,
        }
    )
    return ChatResponse(response=result["response"])


@app.post("/analyze-image", response_model=ChatResponse)
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = Form(
        """
        You are a strict visual layout parser for Figma banner advertisements.

Your task:
Analyze the provided banner image and output a consistent semantic JSON structure describing the visual layout.

IMPORTANT RULES:
1. Output ONLY valid JSON.
2. Do NOT output markdown.
3. Do NOT explain your reasoning.
4. Do NOT invent elements that are not visible.
5. Use ONLY the allowed semantic roles listed below.
6. Use the same role names every time for the same type of visual element.
7. Group elements by semantic meaning, not by visual style.
8. Prefer fewer, stronger groups over many weak groups.
9. Every visible important element must belong to one semantic group.
10. Coordinates must be approximate but spatially correct.

Coordinate system:
- Use normalized coordinates from 0 to 1000.
- Origin is top-left of the image.
- bbox format is [x1, y1, x2, y2].
- x1 < x2 and y1 < y2.
- bbox should tightly cover the visible element or group.

Allowed group roles:
- banner_root
- background_group
- text_zone
- visual_zone
- brand_group
- headline_group
- delivery_info_group
- legal_group
- age_badge_group
- hero_group
- product_visual_group
- offer_group
- price_group
- discount_badge_group
- decoration_group
- overlay_effect_group

Allowed element roles:
- base_background
- text_panel_background
- visual_panel_background
- brand_name_first
- brand_mark
- brand_name_second
- headline_text
- headline_line
- delivery_time_text
- courier_text
- subheadline_text
- legal_text
- age_badge_text
- person_photo
- face_partial
- hands
- clothing
- body_partial
- main_product
- product_packshot
- product_container
- product_label
- food_item
- drink_item
- current_price
- old_price
- currency_symbol
- discount_badge_background
- discount_badge_text
- sparkle
- christmas_light
- snowflake
- ornament
- confetti
- seasonal_prop
- shine_effect
- glow_effect
- gradient_overlay
- mask_overlay

Semantic grouping rules:

A. banner_root
- Always create exactly one banner_root.
- It contains all top-level semantic groups.

B. background_group
- Always create one background_group.
- Include the base banner background, colored panels, photo backgrounds, gradients, and large background shapes.
- Do not put text, logos, products, or people inside background_group.

C. text_zone
- Create when there is a clear text area, usually a blue panel or left-side region.
- It contains brand_group, headline_group, delivery_info_group, legal_group, offer_group if they are visually inside the text area.
- text_zone is a layout container, not a visible object.

D. visual_zone
- Create when there is a clear image/product/photo area.
- It contains hero_group, product_visual_group, decorations, and image-side effects.
- visual_zone is a layout container, not a visible object.

E. brand_group
- Use for the brand identity area.
- For Yandex Lavka, group together:
  - "Яндекс"
  - heart/check mark logo
  - "Лавка"
- Use child roles:
  - brand_name_yandex
  - brand_mark
  - brand_name_lavka
- If the logo or brand text is split into multiple visual pieces, still keep them inside one brand_group.
- Do not classify brand text as headline.

F. headline_group
- Use for the largest main message or product title.
- Includes all large bold title lines.
- Use headline_text if the headline is one text object.
- Use headline_line for separate visible lines.
- Do not include brand, delivery text, legal text, price, or age rating.

G. delivery_info_group
- Use for delivery promise or service information.
- Examples:
  - "от 15 минут"
  - "с доставкой от 15 минут"
  - "привезёт курьер"
- Use delivery_time_text for time promises.
- Use courier_text for courier/delivery action phrases.
- Use subheadline_text for other secondary text below headline.

H. legal_group
- Use for very small disclaimer text, usually near the bottom.
- Use legal_text as child.
- Do not merge legal text with delivery info.

I. age_badge_group
- Use for age rating such as "0+".
- Usually top-right or bottom-right.
- Use age_badge_text as child.
- Do not classify "0+" as price or discount.

J. hero_group
- Use for people, body parts, Santa, hands, clothing, face, lifestyle photo subject.
- Use child roles:
  - person_photo
  - face_partial
  - hands
  - clothing
  - body_partial
- If a person holds a product, the person/body belongs to hero_group and the product belongs to product_visual_group.

K. product_visual_group
- Use for the actual advertised product or food/drink item.
- Examples:
  - gingerbread
  - bowl of salad
  - drink bottle
  - coffee cup
  - candy box
  - packaged nuggets
  - fish sandwich
- Use child roles:
  - main_product
  - product_packshot
  - product_container
  - product_label
  - food_item
  - drink_item
- Product package and its label should stay inside product_visual_group.

L. offer_group
- Use only when visible price, discount, promo, or old price exists.
- If no price or discount is visible, do not create offer_group.
- offer_group may contain price_group and discount_badge_group.

M. price_group
- Use for current price, old crossed price, and currency.
- Use child roles:
  - current_price
  - old_price
  - currency_symbol

N. discount_badge_group
- Use for visible discount badges like "-37%".
- Use child roles:
  - discount_badge_background
  - discount_badge_text

O. decoration_group
- Use for decorative objects that are not product, not logo, and not text.
- Examples:
  - Christmas lights
  - ornaments
  - snowflakes
  - confetti
  - decorative stars
- Use separate children for major decorations if visible.
- Do not create many tiny decoration children unless they are visually important.

P. overlay_effect_group
- Use for visual effects such as glow, shine, sparkle overlays, gradients, or masks.
- Use shine_effect or glow_effect for large bright star/glow effects.
- If a sparkle is decorative and small, it can go in decoration_group.
- If it strongly overlays the layout, use overlay_effect_group.

Output JSON schema:

{
  "image_type": "figma_banner",
  "layout_type": "left_text_right_visual | right_text_left_visual | top_text_bottom_visual | bottom_text_top_visual | centered | mixed",
  "orientation": "landscape | portrait | square",
  "banner_root": {
    "role": "banner_root",
    "name": "banner_root",
    "bbox": [0, 0, 1000, 1000],
    "children": []
  },
  "groups": [
    {
      "id": "group_001",
      "role": "text_zone",
      "name": "text_zone",
      "bbox": [0, 0, 500, 1000],
      "confidence": 0.95,
      "children": [
        {
          "id": "element_001",
          "role": "headline_text",
          "name": "headline_text",
          "bbox": [50, 250, 420, 520],
          "text": "visible text if readable",
          "confidence": 0.95
        }
      ]
    }
  ],
  "missing_expected_groups": [],
  "quality_checks": {
    "has_brand": true,
    "has_headline": true,
    "has_delivery_info": true,
    "has_legal": true,
    "has_age_badge": true,
    "has_visual_product_or_hero": true
  }
}

Naming rules:
- Group names must exactly match role names.
- Element names must exactly match role names, or role name plus index when repeated.
- Example: headline_line_1, headline_line_2, sparkle_1, sparkle_2.
- Do not use raw OCR text as the name.
- Put readable text in the "text" field only.

Consistency rules:
- Always return brand_group for visible "Яндекс Лавка".
- Always return age_badge_group for "0+".
- Always return legal_group for tiny disclaimer text.
- Always return delivery_info_group for "от 15 минут" or similar delivery text.
- Always separate headline_group from delivery_info_group.
- Always separate hero_group from product_visual_group.
- Always separate offer_group from headline_group.
- Always separate decoration_group from overlay_effect_group.

Now analyze the provided banner image and return only the JSON.
        """
    ),
    max_new_tokens: int = Form(512),
) -> ChatResponse:
    image = _data_uri(await file.read(), file.content_type)
    request = ChatRequest(prompt=prompt, image=image, max_new_tokens=max_new_tokens)
    result = _call_model(
        {
            "messages": _messages_for_model(request),
            "max_new_tokens": request.max_new_tokens,
        }
    )
    return ChatResponse(response=result["response"])


@app.post("/categorize-images", response_model=CategorizeResponse)
async def categorize_images(
    files: Annotated[list[UploadFile], File(description="PNG images only")],
    prompt: str | None = Form(None),
    max_new_tokens: int = Query(128, ge=16, le=512),
) -> CategorizeResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one PNG file.")

    user_prompt = (prompt or "").strip()
    if len(user_prompt) > PROMPT_MAX_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt exceeds maximum length ({PROMPT_MAX_LEN} characters).",
        )
    effective_prompt = user_prompt or CATEGORY_PROMPT

    results: list[CategorizeItem] = []
    for upload in files:
        name = upload.filename or "image.png"
        try:
            raw = await upload.read()
            if not _is_png(raw, upload.content_type):
                results.append(
                    CategorizeItem(filename=name, error="Not a PNG (expected image/png or PNG signature).")
                )
                continue
            image = _data_uri(raw, "image/png")
            request = ChatRequest(prompt=effective_prompt, image=image, max_new_tokens=max_new_tokens)
            out = _call_model(
                {
                    "messages": _messages_for_model(request),
                    "max_new_tokens": max_new_tokens,
                }
            )
            cat = _normalize_category(out.get("response", ""))
            results.append(CategorizeItem(filename=name, category=cat or None))
        except HTTPException as exc:
            results.append(CategorizeItem(filename=name, error=str(exc.detail)))
        except Exception as exc:  # noqa: BLE001
            results.append(CategorizeItem(filename=name, error=str(exc)))

    return CategorizeResponse(results=results)


@app.post("/figma/semantic-mid-json", response_model=FigmaSemanticMidResponse)
async def figma_semantic_mid_json(
    banner: UploadFile = File(..., description="Banner image matching the chosen Figma frame"),
    raw_json: UploadFile = File(..., description="Figma layout export JSON"),
    grid: UploadFile | None = File(
        None,
        description=(
            "Optional reference grid: each cell shows an element thumbnail and its id; "
            "sent as second image after the banner for sharper naming."
        ),
    ),
    frame_index: int = Form(0, ge=0),
    chunk_size: int = Form(40, ge=5, le=80),
    max_new_tokens: int = Form(3072, ge=256, le=8192),
) -> FigmaSemanticMidResponse:
    warnings: list[str] = []

    raw_bytes = await raw_json.read()
    if len(raw_bytes) > FIGMA_MAX_JSON_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"raw_json exceeds limit of {FIGMA_MAX_JSON_BYTES} bytes.",
        )
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    try:
        mid = flatten_raw_to_mid(raw, frame_index=frame_index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not mid:
        return FigmaSemanticMidResponse(
            mid_json=[],
            semantic_mid_json=[],
            node_count=0,
            chunks_used=0,
            frame_index=frame_index,
            warnings=["No leaf nodes found for this frame."],
        )

    chunk_n = max(5, min(chunk_size, 80))
    chunks = chunk_list(mid, chunk_n)
    if len(chunks) > FIGMA_SEMANTIC_MAX_CHUNKS:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Too many leaf nodes ({len(mid)}) for configured chunking "
                f"({chunk_n} per chunk, max {FIGMA_SEMANTIC_MAX_CHUNKS} chunks). "
                "Raise FIGMA_SEMANTIC_MAX_CHUNKS / FIGMA_SEMANTIC_CHUNK_SIZE or split the export."
            ),
        )

    banner_body = await banner.read()
    banner_uri = _data_uri(banner_body, banner.content_type)

    grid_uri: str | None = None
    if grid is not None and (grid.filename or "").strip() != "":
        grid_body = await grid.read()
        if not grid_body:
            raise HTTPException(status_code=400, detail="grid file is empty.")
        grid_uri = _data_uri(grid_body, grid.content_type)

    root_name = ""
    if isinstance(raw, list) and 0 <= frame_index < len(raw) and isinstance(raw[frame_index], dict):
        root_name = str(raw[frame_index].get("name") or "")
    frame_hint = f"frame_index={frame_index}" + (f", root_name={root_name!r}" if root_name else "")

    merged_names: dict[str, str] = {}
    used_grid = bool(grid_uri)
    for idx, chunk in enumerate(chunks):
        minimal = [mid_node_prompt_slice(n) for n in chunk]
        user_text = build_naming_user_prompt(
            minimal, frame_hint, has_reference_grid=used_grid
        )
        request = ChatRequest(
            prompt=user_text,
            image=banner_uri,
            images=[grid_uri] if grid_uri else [],
            max_new_tokens=max_new_tokens,
        )
        try:
            result = _call_model(
                {
                    "messages": _messages_for_model(request),
                    "max_new_tokens": max_new_tokens,
                }
            )
            text = result.get("response", "")
            chunk_names = parse_names_object(text)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=502,
                detail=f"Chunk {idx + 1}/{len(chunks)}: failed to parse model output: {exc}",
            ) from exc

        expected_ids = {str(n.get("id")) for n in chunk if n.get("id") is not None}
        got_ids = set(chunk_names.keys())
        missing_chunk = expected_ids - got_ids
        if missing_chunk:
            warnings.append(
                f"Chunk {idx + 1}/{len(chunks)}: model omitted {len(missing_chunk)} id(s); "
                f"kept old names for those nodes."
            )
        extra = got_ids - expected_ids
        if extra:
            warnings.append(f"Chunk {idx + 1}/{len(chunks)}: ignoring {len(extra)} unexpected id(s) from model.")

        for k, v in chunk_names.items():
            if k in expected_ids:
                merged_names[k] = v

    still_missing = missing_name_ids(mid, merged_names)
    if still_missing:
        warnings.append(f"{len(still_missing)} node(s) still have anonymous names (no model mapping).")

    semantic = apply_semantic_names(mid, merged_names)

    return FigmaSemanticMidResponse(
        mid_json=mid,
        semantic_mid_json=semantic,
        node_count=len(mid),
        chunks_used=len(chunks),
        frame_index=frame_index,
        used_reference_grid=used_grid,
        warnings=warnings,
    )


@app.post("/figma/semantic-top-level-children", response_model=FigmaConvertSemanticResponse)
async def figma_semantic_top_level_children(
    banner: UploadFile = File(..., description="Full Figma banner export image"),
    raw_json: UploadFile = File(..., description="Raw Figma layout JSON"),
    top_children_json: UploadFile = File(..., description="Top-level child metadata JSON"),
    top_child_pngs: list[UploadFile] = File(..., description="One PNG per top-level child"),
    max_new_tokens: int = Form(4096, ge=256, le=8192),
) -> FigmaConvertSemanticResponse:
    warnings: list[str] = []
    run_id = _new_figma_semantic_run_id()
    run_dir = FIGMA_SEMANTIC_RUNS_DIR / run_id
    meta_base: dict[str, Any] | None = None

    raw_bytes = await raw_json.read()
    if len(raw_bytes) > FIGMA_MAX_JSON_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"raw_json exceeds limit of {FIGMA_MAX_JSON_BYTES} bytes.",
        )

    banner_body = await banner.read()
    top_children_bytes = await top_children_json.read()
    uploaded_children: list[dict[str, Any]] = []
    for upload in top_child_pngs:
        body = await upload.read()
        if body:
            uploaded_children.append({"upload": upload, "body": body})

    if not banner_body:
        raise HTTPException(status_code=400, detail="banner must be a non-empty file.")
    if not top_children_bytes:
        raise HTTPException(status_code=400, detail="top_children_json must be a non-empty file.")
    if not uploaded_children:
        raise HTTPException(status_code=400, detail="At least one non-empty top_child_png is required.")

    def _safe_artifact_name(value: str | None, fallback: str) -> str:
        base = os.path.basename((value or "").strip()) or fallback
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
        return safe or fallback

    if FIGMA_SEMANTIC_PERSIST_RUNS:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "input_banner.png").write_bytes(banner_body)
            (run_dir / "input_raw.json").write_bytes(raw_bytes)
            (run_dir / "input_top_children.json").write_bytes(top_children_bytes)
            child_dir = run_dir / "top_child_pngs"
            child_dir.mkdir(parents=True, exist_ok=True)
            child_files: list[dict[str, Any]] = []
            used_names: set[str] = set()
            for idx, item in enumerate(uploaded_children):
                upload = item["upload"]
                filename = _safe_artifact_name(upload.filename, f"top_child_{idx}.png")
                if filename in used_names:
                    stem = filename.rsplit(".", 1)[0]
                    ext = "." + filename.rsplit(".", 1)[1] if "." in filename else ".png"
                    filename = f"{stem}_{idx}{ext}"
                used_names.add(filename)
                rel = f"top_child_pngs/{filename}"
                (run_dir / rel).write_bytes(item["body"])
                child_files.append(
                    {
                        "file": rel,
                        "bytes": len(item["body"]),
                        "content_type": upload.content_type or "",
                        "upload_filename": upload.filename,
                    }
                )
            meta_base = {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "endpoint": "/figma/semantic-top-level-children",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "max_new_tokens": max_new_tokens,
                "upload_filenames": {
                    "banner": (banner.filename or "").strip() or None,
                    "raw_json": (raw_json.filename or "").strip() or None,
                    "top_children_json": (top_children_json.filename or "").strip() or None,
                    "top_child_pngs": [
                        (item["upload"].filename or "").strip() or None for item in uploaded_children
                    ],
                },
                "input_files": {
                    "input_banner.png": {
                        "bytes": len(banner_body),
                        "content_type": banner.content_type or "",
                    },
                    "input_raw.json": {"bytes": len(raw_bytes)},
                    "input_top_children.json": {"bytes": len(top_children_bytes)},
                    "top_child_pngs": child_files,
                },
            }
        except OSError as exc:
            warnings.append(f"Run artifacts not saved under {run_dir}: {exc}")
            meta_base = None
    else:
        warnings.append("FIGMA_SEMANTIC_PERSIST_RUNS disabled: run inputs/output not saved to disk.")

    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        if meta_base is not None:
            try:
                _write_figma_semantic_run_meta(
                    run_dir,
                    {
                        **meta_base,
                        "status": "error",
                        "error": "invalid_raw_json",
                        "detail": str(exc),
                    },
                )
            except OSError:
                pass
        raise HTTPException(status_code=400, detail=f"Invalid raw_json: {exc}") from exc

    if isinstance(raw, list):
        raw_root = raw[0] if raw and isinstance(raw[0], dict) else None
    else:
        raw_root = raw if isinstance(raw, dict) else None
    if raw_root is None or not isinstance(raw_root.get("children"), list):
        raise HTTPException(
            status_code=400,
            detail="raw_json must be a root object, or a list with root object at index 0, containing children.",
        )

    try:
        top_children_payload = json.loads(top_children_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        if meta_base is not None:
            try:
                _write_figma_semantic_run_meta(
                    run_dir,
                    {
                        **meta_base,
                        "status": "error",
                        "error": "invalid_top_children_json",
                        "detail": str(exc),
                    },
                )
            except OSError:
                pass
        raise HTTPException(status_code=400, detail=f"Invalid top_children_json: {exc}") from exc

    if not isinstance(top_children_payload, dict) or not isinstance(
        top_children_payload.get("children"), list
    ):
        raise HTTPException(status_code=400, detail="top_children_json must contain a children list.")

    children = [c for c in top_children_payload["children"] if isinstance(c, dict)]
    if not children:
        raise HTTPException(status_code=400, detail="top_children_json children list must be non-empty.")

    banner_model_bytes, banner_resize_info = _resize_raster_max_long_edge(
        banner_body, FIGMA_SEMANTIC_BANNER_MAX_EDGE
    )
    banner_mime = (
        "image/png"
        if banner_resize_info.get("resized")
        else (banner.content_type or "image/png")
    )
    banner_uri = _data_uri(banner_model_bytes, banner_mime)

    uploads_by_name: dict[str, tuple[int, dict[str, Any]]] = {}
    for idx, item in enumerate(uploaded_children):
        upload_name = os.path.basename((item["upload"].filename or "").strip())
        if upload_name and upload_name not in uploads_by_name:
            uploads_by_name[upload_name] = (idx, item)

    child_content: list[ContentItem] = []
    child_image_debug: list[dict[str, Any]] = []
    used_upload_indexes: set[int] = set()
    for child_index, child in enumerate(children):
        file_name = os.path.basename(str(child.get("file_name") or "").strip())
        match: tuple[int, dict[str, Any]] | None = None
        match_mode = "filename"
        if file_name:
            by_name = uploads_by_name.get(file_name)
            if by_name is not None and by_name[0] not in used_upload_indexes:
                match = by_name
        if match is None:
            match_mode = "index"
            if child_index < len(uploaded_children) and child_index not in used_upload_indexes:
                match = (child_index, uploaded_children[child_index])
                if file_name:
                    warnings.append(f"top_child_png filename match failed for {file_name}; used index {child_index}.")
                else:
                    warnings.append(f"top_child_png file_name missing for child {child_index}; used index fallback.")
        if match is None:
            warnings.append(f"Missing top_child_png for child index {child_index}; skipped image.")
            continue

        upload_index, item = match
        used_upload_indexes.add(upload_index)
        model_bytes, resize_info = _resize_raster_max_long_edge(
            item["body"], FIGMA_TOP_LEVEL_CHILD_MAX_EDGE
        )
        mime = (
            "image/png"
            if resize_info.get("resized")
            else (item["upload"].content_type or "image/png")
        )
        child_content.append(ContentItem(type="image", image=_data_uri(model_bytes, mime)))
        child_image_debug.append(
            {
                "child_index": child_index,
                "child_id": child.get("id"),
                "child_path": child.get("path"),
                "expected_file_name": child.get("file_name"),
                "upload_filename": item["upload"].filename,
                "match_mode": match_mode,
                "resize": resize_info,
            }
        )

    if not child_content:
        raise HTTPException(status_code=400, detail="No top_child_png files matched top_children_json children.")

    if meta_base is not None:
        meta_base["model_image_resize"] = {
            "banner": banner_resize_info,
            "top_child_pngs": child_image_debug,
        }

    user_text = build_top_level_semantic_user_text(top_children_payload)
    user_content: list[ContentItem] = [
        ContentItem(type="image", image=banner_uri),
        *child_content,
        ContentItem(type="text", text=user_text),
    ]
    semantic_messages = [
        ChatMessage(role="system", content=TOP_LEVEL_SEMANTIC_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
    ]
    semantic_payload = {
        "messages": [m.model_dump(exclude_none=True) for m in semantic_messages],
        "max_new_tokens": max_new_tokens,
    }

    response_text = ""
    try:
        result = _call_model(semantic_payload, timeout=FIGMA_CONVERT_TIMEOUT)
        response_text = result.get("response", "")
        top_level_names = parse_top_level_names_object(response_text, top_children_payload, warnings)
        top_level_names = fill_missing_top_level_names(
            top_level_names,
            top_children_payload,
            warnings,
        )
        top_level_names = postprocess_top_level_semantic_names(
            top_level_names,
            top_children_payload,
            warnings,
        )
        semantic_json = apply_top_level_semantic_names_to_raw(raw, top_level_names, warnings)
    except HTTPException as he:
        if meta_base is not None:
            try:
                detail = he.detail
                if not isinstance(detail, str):
                    detail = json.dumps(detail, ensure_ascii=False)
                _write_figma_semantic_run_meta(
                    run_dir,
                    {
                        **meta_base,
                        "status": "error",
                        "error": "model_service_http",
                        "http_status": he.status_code,
                        "detail": detail,
                    },
                )
            except OSError:
                pass
        raise
    except ValueError as exc:
        if meta_base is not None:
            try:
                (run_dir / "model_response_raw.txt").write_text(
                    response_text,
                    encoding="utf-8",
                    errors="replace",
                )
                _write_figma_semantic_run_meta(
                    run_dir,
                    {
                        **meta_base,
                        "status": "error",
                        "error": "semantic_top_level_json_parse_failed",
                        "detail": str(exc),
                        "output_files": ["model_response_raw.txt"],
                    },
                )
            except OSError:
                pass
        snippet = response_text[:1200]
        raise HTTPException(
            status_code=502,
            detail=f"Could not parse top-level semantic model output: {exc}. Output starts with: {snippet!r}",
        ) from exc

    semantic_debug = {
        "mode": "top_level_children_only",
        "run_id": run_id,
        "model_response": response_text[:4000],
        "top_level_names_count": len(top_level_names),
        "top_level_names": top_level_names,
        "named_paths": [str(x.get("path")) for x in top_level_names if isinstance(x, dict)],
        "child_images": child_image_debug,
        "warnings": warnings,
    }

    if meta_base is not None:
        try:
            (run_dir / "output_semantic.json").write_text(
                json.dumps(semantic_json, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _write_figma_semantic_run_meta(
                run_dir,
                {
                    **meta_base,
                    "status": "ok",
                    "output_files": ["output_semantic.json"],
                    "semantic_debug": semantic_debug,
                },
            )
        except OSError as exc:
            warnings.append(f"Run output not written to {run_dir}: {exc}")

    return FigmaConvertSemanticResponse(
        semantic_json=semantic_json,
        warnings=warnings,
        semantic_debug=semantic_debug,
    )


@app.get("/figma/top-level-layout-predict/health")
async def figma_top_level_layout_predict_health() -> dict[str, Any]:
    path = _resolve_top_level_layout_checkpoint_path(None)
    retrieval_path = _resolve_top_level_layout_retrieval_records_path(None)
    return {
        "ok": path.exists(),
        "checkpoint": str(path),
        "device": TOP_LEVEL_LAYOUT_MODEL_DEVICE,
        "retrieval_enabled": TOP_LEVEL_LAYOUT_RETRIEVAL_ENABLED,
        "retrieval_records": str(retrieval_path),
        "retrieval_records_exists": retrieval_path.exists(),
        "retrieval_k": TOP_LEVEL_LAYOUT_RETRIEVAL_K,
    }


@app.post("/figma/top-level-layout-predict", response_model=FigmaTopLevelLayoutPredictResponse)
async def figma_top_level_layout_predict(
    req: FigmaTopLevelLayoutPredictRequest,
) -> FigmaTopLevelLayoutPredictResponse:
    checkpoint_path = _resolve_top_level_layout_checkpoint_path(req.checkpoint)
    retrieval_path = _resolve_top_level_layout_retrieval_records_path(None)
    if not checkpoint_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Top-level layout checkpoint not found: {checkpoint_path}",
        )
    retrieval_enabled = TOP_LEVEL_LAYOUT_RETRIEVAL_ENABLED if req.retrieval_enabled is None else req.retrieval_enabled
    retrieval_k = TOP_LEVEL_LAYOUT_RETRIEVAL_K if req.retrieval_k is None else req.retrieval_k
    try:
        result = predict_top_level_layout_json(
            semantic_json=req.semantic_json,
            target_width=req.target_width,
            target_height=req.target_height,
            checkpoint_path=str(checkpoint_path),
            device=TOP_LEVEL_LAYOUT_MODEL_DEVICE,
            retrieval_enabled=retrieval_enabled,
            retrieval_records_path=str(retrieval_path),
            retrieval_k=retrieval_k,
            retrieval_blend=TOP_LEVEL_LAYOUT_RETRIEVAL_BLEND,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Top-level layout prediction failed: {exc}") from exc
    return FigmaTopLevelLayoutPredictResponse(
        final_json=result["final_json"],
        warnings=result.get("warnings", []),
        debug=result.get("debug", {}),
    )


@app.post("/figma/convert-semantic-json", response_model=FigmaConvertSemanticResponse)
async def figma_convert_semantic_json(
    banner: UploadFile = File(..., description="Full Figma banner export image"),
    raw_json: UploadFile = File(..., description="Raw Figma layout JSON"),
    grid: UploadFile = File(..., description="Grid image: each cell = element + raw JSON id"),
    max_new_tokens: int = Form(2048, ge=256, le=8192),
) -> FigmaConvertSemanticResponse:
    warnings: list[str] = []
    run_id = _new_figma_semantic_run_id()
    run_dir = FIGMA_SEMANTIC_RUNS_DIR / run_id
    meta_base: dict[str, Any] | None = None

    raw_bytes = await raw_json.read()
    if len(raw_bytes) > FIGMA_MAX_JSON_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"raw_json exceeds limit of {FIGMA_MAX_JSON_BYTES} bytes.",
        )

    banner_body = await banner.read()
    grid_body = await grid.read()
    if not banner_body or not grid_body:
        raise HTTPException(status_code=400, detail="banner and grid must be non-empty files.")

    if FIGMA_SEMANTIC_PERSIST_RUNS:
        try:
            meta_base = _persist_figma_semantic_run_inputs(
                run_dir,
                run_id=run_id,
                max_new_tokens=max_new_tokens,
                banner_body=banner_body,
                grid_body=grid_body,
                raw_bytes=raw_bytes,
                banner_content_type=banner.content_type,
                grid_content_type=grid.content_type,
                banner_filename=banner.filename,
                raw_json_filename=raw_json.filename,
                grid_filename=grid.filename,
            )
        except OSError as exc:
            warnings.append(f"Run artifacts not saved under {run_dir}: {exc}")
            meta_base = None
    else:
        meta_base = None
        warnings.append("FIGMA_SEMANTIC_PERSIST_RUNS disabled: run inputs/output not saved to disk.")

    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        if meta_base is not None:
            try:
                _write_figma_semantic_run_meta(
                    run_dir,
                    {
                        **meta_base,
                        "status": "error",
                        "error": "invalid_raw_json",
                        "detail": str(exc),
                    },
                )
            except OSError:
                pass
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    try:
        mid_blocks = raw_fig_tree_to_mid_blocks(raw, frame_index=0)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if meta_base is not None:
        try:
            (run_dir / "input_mid.json").write_text(
                json.dumps(mid_blocks, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            meta_base.setdefault("input_files", {})["input_mid.json"] = {
                "bytes": (run_dir / "input_mid.json").stat().st_size,
            }
        except OSError as exc:
            warnings.append(f"Could not save input_mid.json: {exc}")

    banner_model_bytes, banner_resize_info = _resize_raster_max_long_edge(
        banner_body, FIGMA_SEMANTIC_BANNER_MAX_EDGE
    )
    grid_model_bytes, grid_resize_info = _resize_raster_max_long_edge(
        grid_body, FIGMA_SEMANTIC_GRID_MAX_EDGE
    )
    if meta_base is not None:
        meta_base["model_image_resize"] = {
            "banner": banner_resize_info,
            "grid": grid_resize_info,
        }

    banner_mime = (
        "image/png"
        if banner_resize_info.get("resized")
        else (banner.content_type or "image/png")
    )
    grid_mime = (
        "image/png"
        if grid_resize_info.get("resized")
        else (grid.content_type or "image/png")
    )
    banner_uri = _data_uri(banner_model_bytes, banner_mime)
    grid_uri = _data_uri(grid_model_bytes, grid_mime)

    semantic_debug: dict[str, Any] | None = None
    response_text = ""
    semantic_json: Any = None

    try:
        if FIGMA_SEMANTIC_STRICT_MODE:
            strict_pre = run_strict_semantic_naming(mid_blocks)
            semantic_debug = strict_pre.semantic_debug
            qwen_names: dict[str, str] = {}
            ambiguous_ids = strict_pre.ambiguous_ids

            if ambiguous_ids:
                features = extract_node_features(mid_blocks)
                ambiguous_nodes = build_qwen_ambiguous_payload(
                    mid_blocks,
                    features,
                    strict_pre.semantic_debug["prelabel_roles"],
                    ambiguous_ids,
                )
                user_text = build_qwen_ambiguous_user_text(ambiguous_nodes)
                user_content: list[ContentItem] = [
                    ContentItem(type="image", image=banner_uri),
                    ContentItem(type="image", image=grid_uri),
                    ContentItem(type="text", text=user_text),
                ]
                semantic_messages = [
                    ChatMessage(role="system", content=STRICT_QWEN_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=user_content),
                ]
                semantic_payload = {
                    "messages": [m.model_dump(exclude_none=True) for m in semantic_messages],
                    "max_new_tokens": max_new_tokens,
                }
                result = _call_model(semantic_payload, timeout=FIGMA_CONVERT_TIMEOUT)
                response_text = result.get("response", "")
                qwen_names = parse_names_object(response_text)
                warnings.append(f"strict_qwen_ambiguous:{len(ambiguous_ids)} nodes")

            strict_final = run_strict_semantic_naming(mid_blocks, qwen_names=qwen_names or None)
            semantic_debug = strict_final.semantic_debug
            semantic_json = build_semantic_json_from_strict_names(
                mid_blocks, strict_final.names, warnings
            )
            meta_warnings = assert_rich_metadata_preserved(mid_blocks, semantic_json)
            warnings.extend(meta_warnings)
            if strict_final.validation.errors:
                warnings.append(
                    "strict_validator_errors:" + json.dumps(strict_final.validation.errors, ensure_ascii=False)
                )
        else:
            required_ids = sorted(collect_allowed_ids_from_mid(mid_blocks))
            vlm_layout_payload = {"mid": mid_blocks, "required_node_ids": required_ids}
            layout_text = json.dumps(vlm_layout_payload, ensure_ascii=False, separators=(",", ":"))
            user_text = (
                FIGMA_CONVERT_PROMPT
                + "\n\nLayout JSON (mid + required_node_ids; no full raw tree):\n"
                + layout_text
            )

            user_content = [
                ContentItem(type="image", image=banner_uri),
                ContentItem(type="image", image=grid_uri),
                ContentItem(type="text", text=user_text),
            ]
            semantic_messages = [
                ChatMessage(role="system", content=FIGMA_CONVERT_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_content),
            ]
            semantic_payload = {
                "messages": [m.model_dump(exclude_none=True) for m in semantic_messages],
                "max_new_tokens": max_new_tokens,
            }
            result = _call_model(
                semantic_payload,
                timeout=FIGMA_CONVERT_TIMEOUT,
            )
            response_text = result.get("response", "")
            parsed = extract_first_json_value(response_text)
            semantic_json = normalize_convert_semantic_output(parsed, mid_blocks, warnings)
    except HTTPException as he:
        if meta_base is not None:
            try:
                detail = he.detail
                if not isinstance(detail, str):
                    detail = json.dumps(detail, ensure_ascii=False)
                _write_figma_semantic_run_meta(
                    run_dir,
                    {
                        **meta_base,
                        "status": "error",
                        "error": "model_service_http",
                        "http_status": he.status_code,
                        "detail": detail,
                    },
                )
            except OSError:
                pass
        raise
    except ValueError as exc:
        if meta_base is not None:
            try:
                raw_out = response_text if isinstance(response_text, str) else ""
                (run_dir / "model_response_raw.txt").write_text(raw_out, encoding="utf-8", errors="replace")
                _write_figma_semantic_run_meta(
                    run_dir,
                    {
                        **meta_base,
                        "status": "error",
                        "error": "semantic_json_parse_failed",
                        "detail": str(exc),
                        "output_files": ["model_response_raw.txt"],
                    },
                )
            except OSError:
                pass
        snippet = (response_text if isinstance(response_text, str) else "")[:1200]
        raise HTTPException(
            status_code=502,
            detail=f"Could not parse model output as JSON: {exc}. Output starts with: {snippet!r}",
        ) from exc

    if meta_base is not None:
        try:
            (run_dir / "output_semantic.json").write_text(
                json.dumps(semantic_json, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _write_figma_semantic_run_meta(
                run_dir,
                {
                    **meta_base,
                    "status": "ok",
                    "output_files": ["output_semantic.json"],
                },
            )
        except OSError as exc:
            warnings.append(f"Run output not written to {run_dir}: {exc}")

    return FigmaConvertSemanticResponse(
        semantic_json=semantic_json,
        warnings=warnings,
        semantic_debug=semantic_debug,
    )


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host=HOST, port=PORT)
