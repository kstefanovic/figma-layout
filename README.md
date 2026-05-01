# Qwen2.5-VL FastAPI Project

This project exposes a public FastAPI backend on `0.0.0.0:9298` and keeps the Qwen2.5-VL-7B-Instruct model service private on `127.0.0.1:9297`.

## Run

Start the local Qwen model service:

```bash
source .venv/bin/activate
python model_server.py
```

Start the public backend in another terminal:

```bash
source .venv/bin/activate
python backend.py
```

The backend is available from other machines on port `9298`. The model service stays bound to localhost on port `9297`. Both processes read settings from `.env` automatically.

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
QWEN_PORT=9297
QWEN_GPU_DEVICE=0
QWEN_DEVICE=cuda:0

BACKEND_HOST=0.0.0.0
BACKEND_PORT=9298
MODEL_SERVICE_URL=http://127.0.0.1:9297
MODEL_REQUEST_TIMEOUT=300
```

`QWEN_GPU_DEVICE` controls `CUDA_VISIBLE_DEVICES` before `torch` initializes CUDA. With `QWEN_GPU_DEVICE=0`, the model runs on GPU 0 as `cuda:0`.

## API

Health check:

```bash
curl http://localhost:9298/health
```

Text chat:

```bash
curl -X POST http://localhost:9298/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What can you do?","max_new_tokens":128}'
```

Image URL:

```bash
curl -X POST http://localhost:9298/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Describe this image.","image":"https://example.com/image.jpg"}'
```

Image upload:

```bash
curl -X POST http://localhost:9298/analyze-image \
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
