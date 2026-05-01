import os
from typing import Any, Literal

from dotenv import load_dotenv


load_dotenv()

# Keep the model pinned to the configured GPU before torch initializes CUDA.
GPU_DEVICE = os.getenv("QWEN_GPU_DEVICE", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", GPU_DEVICE)

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL_PATH = os.getenv("QWEN_MODEL_PATH", "./Qwen2.5-VL-7B-Instruct")
HOST = os.getenv("QWEN_HOST", "127.0.0.1")
PORT = int(os.getenv("QWEN_PORT", "9297"))
DEVICE = os.getenv("QWEN_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")


class ContentItem(BaseModel):
    type: Literal["text", "image", "video"]
    text: str | None = None
    image: str | None = None
    video: Any | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentItem]


class GenerateRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_new_tokens: int = Field(default=256, ge=1, le=4096)


class GenerateResponse(BaseModel):
    response: str


app = FastAPI(title="Qwen2.5-VL-7B-Instruct Service")
model: Qwen2_5_VLForConditionalGeneration | None = None
processor: AutoProcessor | None = None


def _device() -> str:
    return DEVICE


@app.on_event("startup")
def load_model() -> None:
    global model, processor

    model_kwargs: dict[str, Any] = {"torch_dtype": "auto"}
    if DEVICE.startswith("cuda"):
        model_kwargs["device_map"] = {"": DEVICE}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        **model_kwargs,
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    if DEVICE == "cpu":
        model.to("cpu")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok" if model is not None and processor is not None else "loading",
        "model_path": MODEL_PATH,
        "device": _device(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model is still loading")

    messages = [message.model_dump(exclude_none=True) for message in request.messages]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=request.max_new_tokens)
    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids, strict=True)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return GenerateResponse(response=output_text[0])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("model_server:app", host=HOST, port=PORT)
