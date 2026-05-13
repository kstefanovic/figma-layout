# Qwen2.5-VL FastAPI Project

This project exposes a public FastAPI backend on `0.0.0.0:20401` and keeps the Qwen2.5-VL-7B-Instruct model service private on `127.0.0.1:20400`.

## Run

Start the local Qwen model service:

```bash
source .venv/bin/activate
python model_server.py
```

Pick a physical GPU without changing `.env` (writes `CUDA_VISIBLE_DEVICES` before PyTorch loads):

```bash
python model_server.py --gpu 1
# short form
python model_server.py -g 0
```

Start the public backend in another terminal:

```bash
source .venv/bin/activate
python backend.py
```

The backend is available from other machines on port `20401`. The model service stays bound to localhost on port `20400`. Both processes read settings from `.env` automatically.

## PM2

Start both services:

```bash
pm2 start ecosystem.config.js
```

Check status and logs:

```bash
pm2 status
pm2 logs
pm2 logs qwen-model-server
pm2 logs qwen-backend
```

Turn services off:

```bash
pm2 stop qwen-backend
pm2 stop qwen-model-server
```

Turn services back on:

```bash
pm2 start qwen-model-server
pm2 start qwen-backend
```

Restart after changing `.env` or code:

```bash
pm2 restart qwen-model-server
pm2 restart qwen-backend
```

Remove both services from PM2:

```bash
pm2 delete qwen-backend
pm2 delete qwen-model-server
```

## `.env` Settings

```bash
QWEN_MODEL_PATH=./Qwen2.5-VL-7B-Instruct
QWEN_HOST=127.0.0.1
QWEN_PORT=20400
# Host GPU index; use 1 for the second card, etc. Optional: QWEN_DEVICE (see .env.example).
QWEN_GPU_DEVICE=0

BACKEND_HOST=0.0.0.0
BACKEND_PORT=20401
MODEL_SERVICE_URL=http://127.0.0.1:20400
MODEL_REQUEST_TIMEOUT=300
```

`QWEN_GPU_DEVICE` sets `CUDA_VISIBLE_DEVICES` before `torch` loads, so you select which physical GPU(s) the process may use. Inside the process, the first visible GPU is always `cuda:0`; you only need `QWEN_DEVICE` if you expose multiple GPUs or need a non-default mapping (see `.env.example`).

## API

Health check:

```bash
curl http://localhost:20401/health
```

Text chat:

```bash
curl -X POST http://localhost:20401/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What can you do?","max_new_tokens":128}'
```

Image URL:

```bash
curl -X POST http://localhost:20401/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Describe this image.","image":"https://example.com/image.jpg"}'
```

Image upload:

```bash
curl -X POST http://localhost:20401/analyze-image \
  -F "file=@/path/to/image.jpg" \
  -F "prompt=Describe this image."
```

Advanced chat messages are also supported through `/chat`:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "image", "image": "file:///path/to/image.jpg" },
        { "type": "text", "text": "What is in this image?" }
      ]
    }
  ],
  "max_new_tokens": 256
}
```
