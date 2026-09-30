"""Qwen3-VL generation. One process keeps the weights in memory."""

import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .prompt import _QWEN_PATCH, image_content, prompt_text, qwen_pixel_limits
from .settings import Settings

logger = logging.getLogger("qwen")

_model = None
_processor = None


def local_snapshot_dir(model_id: str) -> Optional[str]:
    """Hub snapshot under HF_HOME/hub. None when files are missing."""
    hf_home = Path(os.environ.get("HF_HOME") or "/opt/huggingface")
    snapshots = (
        hf_home / "hub" / ("models--" + model_id.replace("/", "--")) / "snapshots"
    )
    if not snapshots.is_dir():
        return None
    for path in sorted(snapshots.iterdir()):
        if path.is_dir() and (path / "config.json").is_file():
            return str(path)
    return None


def ensure_weights() -> Optional[str]:
    """Download the snapshot when it is not already on disk."""
    model_id = str(Settings.QWEN_VL_MODEL)
    found = local_snapshot_dir(model_id)
    if found:
        logger.info("Qwen VL weights present source=%s", found)
        return found
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download

    logger.info("Qwen VL download start model=%s", model_id)
    path = snapshot_download(model_id)
    logger.info("Qwen VL download done path=%s", path)
    return local_snapshot_dir(model_id) or path


def _torch_device():
    """GPU on the server. Refuse a silent CPU fallback when cuda is required."""
    import torch

    requested = str(Settings.QWEN_DEVICE)
    if requested == "cpu":
        return torch.device("cpu")
    if requested != "cuda":
        raise RuntimeError(f"QWEN_DEVICE must be cuda or cpu, got {requested}")
    if not torch.cuda.is_available():
        raise RuntimeError("QWEN_DEVICE=cuda but CUDA is not available")
    return torch.device("cuda")


def _load_model() -> Tuple[object, object]:
    """Lazy-load weights onto the GPU. Reused across images in this process."""
    global _model, _processor
    if _model is not None and _processor is not None:
        logger.info("Qwen VL step: weights already in memory")
        return _model, _processor
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    device = _torch_device()
    model_id = str(Settings.QWEN_VL_MODEL)
    local_dir = local_snapshot_dir(model_id)
    source = local_dir or model_id
    dtype_name = str(Settings.QWEN_VL_DTYPE)
    if dtype_name == "float32":
        torch_dtype = torch.float32
    elif dtype_name == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.bfloat16
    min_px, max_px = qwen_pixel_limits()
    logger.info(
        "Qwen VL step: load processor device=%s dtype=%s source=%s "
        "min_pixels=%s max_pixels=%s",
        device,
        dtype_name,
        source,
        min_px,
        max_px,
    )
    load_kw = {"local_files_only": True} if local_dir else {}
    load_t0 = time.perf_counter()
    _processor = AutoProcessor.from_pretrained(
        source,
        min_pixels=min_px,
        max_pixels=max_px,
        **load_kw,
    )
    logger.info("Qwen VL step: load weights")
    _model = Qwen3VLForConditionalGeneration.from_pretrained(
        source,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        **load_kw,
    )
    _model.to(device)
    _model.eval()
    logger.info(
        "Qwen VL step: weights ready device=%s elapsed_sec=%.1f",
        device,
        time.perf_counter() - load_t0,
    )
    return _model, _processor


def generate_text(image: np.ndarray) -> str:
    """Run Qwen3-VL on a full BGR canvas. Does not crop."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    class TokenProgress(StoppingCriteria):
        """Log generate progress so a long run is visible."""

        def __init__(self, max_new_tokens: int, every: int = 32) -> None:
            super().__init__()
            self.max_new_tokens = max_new_tokens
            self.every = max(1, every)
            self.steps = 0
            self.started = time.perf_counter()

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            self.steps += 1
            if self.steps == 1 or self.steps % self.every == 0:
                elapsed = time.perf_counter() - self.started
                logger.info(
                    "Qwen VL generate tokens=%s/%s elapsed_sec=%.1f",
                    self.steps,
                    self.max_new_tokens,
                    elapsed,
                )
            return False

    if image is None or image.size == 0 or image.ndim < 2:
        raise ValueError("empty image")
    model, processor = _load_model()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    from PIL import Image

    pil = Image.fromarray(rgb)
    min_px, max_px = qwen_pixel_limits()
    visual_tokens = max_px // (_QWEN_PATCH * _QWEN_PATCH)
    logger.info(
        "Qwen VL step: preprocess image=%sx%s min_pixels=%s max_pixels=%s "
        "visual_tokens<=%s",
        pil.width,
        pil.height,
        min_px,
        max_px,
        visual_tokens,
    )
    messages = [
        {
            "role": "user",
            "content": [
                image_content(pil),
                {"type": "text", "text": prompt_text(pil.width, pil.height)},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    try:
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        if image_inputs:
            vision = image_inputs[0]
            if hasattr(vision, "size"):
                vw, vh = vision.size
                logger.info(
                    "Qwen VL step: vision resized=%sx%s tokens~=%s",
                    vw,
                    vh,
                    (vw // _QWEN_PATCH) * (vh // _QWEN_PATCH),
                )
        proc_kw = {
            "text": [text],
            "images": image_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        if video_inputs:
            proc_kw["videos"] = video_inputs
        inputs = processor(**proc_kw)
    except ImportError:
        inputs = processor(
            text=[text], images=[pil], padding=True, return_tensors="pt"
        )
    if hasattr(inputs, "to"):
        inputs = inputs.to(next(model.parameters()).device)
    max_new = int(Settings.QWEN_VL_MAX_NEW_TOKENS)
    prompt_len = int(inputs["input_ids"].shape[1])
    logger.info(
        "Qwen VL step: start generate prompt_tokens=%s max_new_tokens=%s",
        prompt_len,
        max_new,
    )
    started = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new,
            repetition_penalty=1.1,
            stopping_criteria=StoppingCriteriaList([TokenProgress(max_new)]),
        )
    new_tokens = int(generated.shape[1]) - prompt_len
    logger.info(
        "Qwen VL step: generate done new_tokens=%s elapsed_sec=%.1f",
        new_tokens,
        time.perf_counter() - started,
    )
    trimmed = generated[:, prompt_len:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]
