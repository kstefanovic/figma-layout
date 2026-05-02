import base64
import binascii
import json
import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from figma_semantic import (
    FIGMA_CONVERT_PROMPT,
    apply_semantic_names,
    build_naming_user_prompt,
    chunk_list,
    extract_first_json_value,
    flatten_raw_to_mid,
    mid_node_prompt_slice,
    missing_name_ids,
    parse_names_object,
)
from json_embedding import (
    VALID_CLASSES,
    attach_full_json,
    build_all_indexes,
    parse_aspect_ratio,
    parse_resolution,
    resize_figma_json_to_resolution,
    search_index,
)


load_dotenv()

MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://127.0.0.1:9297")
HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
PORT = int(os.getenv("BACKEND_PORT", "9298"))
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
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

BANNER_VLM_CATEGORY_PROMPT = """You classify one retail / food banner image into exactly ONE of four campaigns.

The four categories (match by main visual theme — product, hero, colors, headline; OCR need not be exact):

1. Пряники прямо на ёлку
2. Пряничный ровер
3. Мегапорция оливье для гостей
4. Еловый лимонад с малиной

Output rules:
- Reply with ONLY the digit 1, 2, 3, or 4.
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
    max_new_tokens: int = Field(default=256, ge=1, le=4096)


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


class BannerCategoryResponse(BaseModel):
    """VLM banner classification into campaign 1–4."""

    category: int = Field(..., ge=1, le=4, description="Campaign index 1–4 per product brief")
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


class BannerSearchPipelineResponse(BaseModel):
    category: int = Field(..., ge=1, le=4)
    raw_model_text: str = ""
    aspect_ratio: float
    top_k: int
    candidates: list[JsonEmbeddingCandidate]


class BannerRawToTargetJsonResponse(BaseModel):
    category: int = Field(..., ge=1, le=4)
    raw_model_text: str = ""
    target_width: float
    target_height: float
    aspect_ratio: float
    top_k: int
    selected_candidate: JsonEmbeddingCandidate
    candidates: list[JsonEmbeddingCandidate]
    final_json: dict[str, Any]


class BannerRawToTargetJsonJsonRequest(BaseModel):
    banner_png_base64: str
    raw_json: Any
    target_resolution: str | None = None
    target_width: float | None = None
    target_height: float | None = None
    raw_frame_index: int = Field(default=0, ge=0)
    top_k: int = Field(default=3, ge=1, le=20)
    max_new_tokens: int = Field(default=64, ge=8, le=512)


app = FastAPI(title="Public Qwen2.5-VL Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _model_url(path: str) -> str:
    return f"{MODEL_SERVICE_URL.rstrip('/')}/{path.lstrip('/')}"


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


def _parse_banner_category_1_to_4(text: str) -> int:
    """Extract integer 1–4 from VLM reply (digit only, JSON, or first matching token)."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty model response")

    inner = raw
    fence = re.search(r"```(?:json|text)?\s*([\s\S]*?)```", raw, re.IGNORECASE)
    if fence:
        inner = fence.group(1).strip()

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
                if isinstance(v, int) and 1 <= v <= 4:
                    return v
                if isinstance(v, str) and v.strip().isdigit():
                    n = int(v.strip())
                    if 1 <= n <= 4:
                        return n
        if isinstance(parsed, list) and len(parsed) == 1:
            only = parsed[0]
            if isinstance(only, int) and 1 <= only <= 4:
                return only

    for line in inner.splitlines():
        s = line.strip()
        if re.fullmatch(r"[1-4]", s):
            return int(s)

    m = re.search(r"\b([1-4])\b", inner)
    if m:
        return int(m.group(1))

    raise ValueError(f"no digit 1–4 found in: {inner[:300]!r}")


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
        category = _parse_banner_category_1_to_4(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"VLM output could not be parsed as 1–4: {exc}. Raw: {raw[:500]!r}",
        ) from exc
    return category, raw


def _run_banner_raw_to_target_pipeline(
    *,
    banner_body: bytes,
    banner_content_type: str | None,
    uploaded_raw: Any,
    target_resolution: str,
    raw_frame_index: int,
    top_k: int,
    max_new_tokens: int,
) -> BannerRawToTargetJsonResponse:
    category, raw_model_text = _classify_banner_bytes(
        banner_body, banner_content_type, max_new_tokens=max_new_tokens
    )
    try:
        target_width, target_height = parse_resolution(target_resolution)
        parsed_aspect = target_width / target_height
        retrieved = attach_full_json(search_index(category, target_resolution, top_k=top_k))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not retrieved:
        raise HTTPException(status_code=404, detail="No retrievable candidates found.")

    selected = retrieved[0]
    selected_json = selected.get("full_json")
    if not isinstance(selected_json, dict):
        raise HTTPException(status_code=500, detail="Selected candidate does not include full_json.")

    final_json = resize_figma_json_to_resolution(selected_json, target_width, target_height)

    return BannerRawToTargetJsonResponse(
        category=category,
        raw_model_text=raw_model_text[:2000],
        target_width=target_width,
        target_height=target_height,
        aspect_ratio=parsed_aspect,
        top_k=top_k,
        selected_candidate=JsonEmbeddingCandidate(**selected),
        candidates=[JsonEmbeddingCandidate(**row) for row in retrieved],
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
    }


@app.post("/banner/category", response_model=BannerCategoryResponse)
async def banner_category(
    file: UploadFile = File(..., description="Banner PNG (or JPEG/WebP)"),
    max_new_tokens: int = Form(64, ge=8, le=512),
) -> BannerCategoryResponse:
    """VLM picks which of four Yandex Lavka-style campaigns the banner belongs to (output 1–4)."""
    body = await file.read()
    category, raw = _classify_banner_bytes(body, file.content_type, max_new_tokens=max_new_tokens)

    return BannerCategoryResponse(category=category, raw_model_text=raw[:2000])


@app.post("/json-embeddings/build", response_model=JsonEmbeddingBuildResponse)
def build_json_embeddings() -> JsonEmbeddingBuildResponse:
    """Build one local embedding index for each class: raw_jsons/1.json ... raw_jsons/4.json."""
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
    class_number: int = Query(..., ge=1, le=4, description="Select one of the 4 class indexes"),
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


@app.post("/pipeline/banner-search", response_model=BannerSearchPipelineResponse)
async def classify_banner_then_search_json(
    file: UploadFile = File(..., description="Banner PNG/JPEG/WebP to classify"),
    target_resolution: str = Form(..., description="Target resolution or aspect ratio: 2280x360, 16:9, 1.777"),
    top_k: int = Form(3, ge=1, le=20),
    max_new_tokens: int = Form(64, ge=8, le=512),
    include_full_json: bool = Form(True),
) -> BannerSearchPipelineResponse:
    """One flow: classify banner into class 1–4, then search that class index by target resolution."""
    body = await file.read()
    category, raw = _classify_banner_bytes(body, file.content_type, max_new_tokens=max_new_tokens)

    try:
        parsed_aspect = parse_aspect_ratio(target_resolution)
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
    1. Classify banner into class 1-4.
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
    return _run_banner_raw_to_target_pipeline(
        banner_body=banner_body,
        banner_content_type=file.content_type,
        uploaded_raw=uploaded_raw,
        target_resolution=target_resolution,
        raw_frame_index=raw_frame_index,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
    )


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
    return _run_banner_raw_to_target_pipeline(
        banner_body=banner_body,
        banner_content_type="image/png",
        uploaded_raw=request.raw_json,
        target_resolution=target_resolution,
        raw_frame_index=request.raw_frame_index,
        top_k=request.top_k,
        max_new_tokens=request.max_new_tokens,
    )


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
    max_new_tokens: int = Form(3072, ge=256, le=4096),
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


@app.post("/figma/convert-semantic-json", response_model=FigmaConvertSemanticResponse)
async def figma_convert_semantic_json(
    banner: UploadFile = File(..., description="Full Figma banner export image"),
    raw_json: UploadFile = File(..., description="Raw Figma layout JSON"),
    grid: UploadFile = File(..., description="Grid image: each cell = element + raw JSON id"),
    max_new_tokens: int = Form(4096, ge=256, le=4096),
) -> FigmaConvertSemanticResponse:
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

    banner_body = await banner.read()
    grid_body = await grid.read()
    if not banner_body or not grid_body:
        raise HTTPException(status_code=400, detail="banner and grid must be non-empty files.")

    banner_uri = _data_uri(banner_body, banner.content_type)
    grid_uri = _data_uri(grid_body, grid.content_type)

    raw_text = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
    user_text = (
        FIGMA_CONVERT_PROMPT
        + "\n\nRaw Figma JSON (apply the rules above; output only the transformed JSON):\n"
        + raw_text
    )

    user_content: list[ContentItem] = [
        ContentItem(type="image", image=banner_uri),
        ContentItem(type="image", image=grid_uri),
        ContentItem(type="text", text=user_text),
    ]
    request = ChatRequest(
        messages=[ChatMessage(role="user", content=user_content)],
        max_new_tokens=max_new_tokens,
    )

    response_text = ""
    try:
        result = _call_model(
            {
                "messages": _messages_for_model(request),
                "max_new_tokens": max_new_tokens,
            },
            timeout=FIGMA_CONVERT_TIMEOUT,
        )
        response_text = result.get("response", "")
        semantic_json = extract_first_json_value(response_text)
    except HTTPException:
        raise
    except ValueError as exc:
        snippet = (response_text if isinstance(response_text, str) else "")[:1200]
        raise HTTPException(
            status_code=502,
            detail=f"Could not parse model output as JSON: {exc}. Output starts with: {snippet!r}",
        ) from exc

    return FigmaConvertSemanticResponse(semantic_json=semantic_json, warnings=warnings)


if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host=HOST, port=PORT)
