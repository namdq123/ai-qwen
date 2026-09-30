"""Runtime settings for the Qwen VL service."""

import os


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


class Settings:
    """Environment-backed settings. Read at call time via class attributes."""

    QWEN_VL_MODEL: str = os.getenv("QWEN_VL_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    QWEN_VL_MAX_NEW_TOKENS: int = int(os.getenv("QWEN_VL_MAX_NEW_TOKENS", "2048"))
    QWEN_VL_DTYPE: str = os.getenv("QWEN_VL_DTYPE", "bfloat16").strip().lower()
    # cuda on the GPU server. cpu is only for a machine without a GPU.
    QWEN_DEVICE: str = os.getenv("QWEN_DEVICE", "cuda").strip().lower()
    QWEN_VL_MIN_PIXELS: int = int(os.getenv("QWEN_VL_MIN_PIXELS", str(4 * 28 * 28)))
    QWEN_VL_MAX_PIXELS: int = int(os.getenv("QWEN_VL_MAX_PIXELS", str(1280 * 28 * 28)))
    HOST: str = os.getenv("QWEN_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("QWEN_PORT", "8090"))
    DOWNLOAD_ON_START: bool = _flag("QWEN_DOWNLOAD_ON_START", "true")
    # Warm the GPU at process start. Tests leave this unset.
    LOAD_ON_START: bool = _flag("QWEN_LOAD_ON_START", "false")
