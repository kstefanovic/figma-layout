import base64
import os
from typing import Any, Literal

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


load_dotenv()

MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://127.0.0.1:9297")
HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
PORT = int(os.getenv("BACKEND_PORT", "9298"))
REQUEST_TIMEOUT = float(os.getenv("MODEL_REQUEST_TIMEOUT", "300"))


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


def _call_model(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = requests.post(
            _model_url("/generate"),
            json=payload,
            timeout=REQUEST_TIMEOUT,
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
    prompt: str = Form("Describe this image."),
    max_new_tokens: int = Form(256),
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host=HOST, port=PORT)
