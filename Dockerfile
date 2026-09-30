FROM python:3.11-slim

WORKDIR /drawing-calc-app-backend-qwen

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CUDA wheels ship the runtime libs. The host only needs the NVIDIA driver
# and nvidia-container-toolkit. Override the index for another CUDA major.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
RUN pip install --upgrade pip \
    && pip install torch==2.8.0 torchvision==0.23.0 \
        --index-url ${TORCH_INDEX_URL}

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

ENV HF_HOME=/opt/huggingface
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN mkdir -p /opt/huggingface

EXPOSE 8090

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "1"]
