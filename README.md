# drawing-calc-app-backend-qwen

Qwen3-VL service for exterior-drawing polygons. It runs on a GPU server. Local `drawing-calc-app-backend-ai` sends the canvas and parses the reply.

## API

- `GET /health` ? process is up. `cuda` is true when the GPU is visible.
- `POST /v1/generate` ? multipart field `image` (PNG or JPEG).

```json
{"text": "<raw model reply>", "width": 1024, "height": 768}
```

One generate runs at a time. Uvicorn uses one worker so the weights stay on one GPU.

## GPU server

The host needs an NVIDIA driver and [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). The image installs PyTorch CUDA 12.4 wheels. The driver must support CUDA 12.4.

```bash
cp .env.example .env
docker compose up --build
```

The first start downloads `Qwen/Qwen3-VL-8B-Instruct` into the `qwen-huggingface` volume and loads it onto the GPU. bfloat16 8B needs about 16GB of VRAM. Port `8090` is published on the host. Allow that port from the machine that runs `backend-ai`.

Check from another machine:

```bash
curl http://<gpu-server-host>:8090/health
```

## Local backend-ai

In `drawing-calc-app-backend-ai/.env`:

```bash
USE_QWEN_VL_BOXES=true
QWEN_VL_API_URL=http://<gpu-server-host>:8090
QWEN_VL_API_TIMEOUT=1800
```

Restart `ai-inference-server` after changing the URL. That container does not load Qwen weights.
