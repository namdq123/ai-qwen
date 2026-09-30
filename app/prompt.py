"""Prompt and vision-token budget for Qwen3-VL. No model imports."""

from pathlib import Path
from typing import Tuple

from .settings import Settings

_QWEN_PATCH = 28
_SKILLS_MD = Path(__file__).resolve().parent / "skills.md"


def qwen_pixel_limits() -> Tuple[int, int]:
    """min_pixels, max_pixels for Qwen vision resize. Caps visual tokens."""
    patch_area = _QWEN_PATCH * _QWEN_PATCH
    min_px = max(patch_area, int(Settings.QWEN_VL_MIN_PIXELS))
    max_px = max(min_px, int(Settings.QWEN_VL_MAX_PIXELS))
    return min_px, max_px


def image_content(pil: object) -> dict:
    """Full-canvas image for Qwen. Downscales via max_pixels, does not crop."""
    min_px, max_px = qwen_pixel_limits()
    return {
        "type": "image",
        "image": pil,
        "min_pixels": min_px,
        "max_pixels": max_px,
    }


def skills_prompt_text() -> str:
    """Paint-takeoff prompt (skills.md). Drops sk- lines and the trailing note."""
    if not _SKILLS_MD.is_file():
        return ""
    lines = []
    for line in _SKILLS_MD.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("sk-"):
            continue
        lines.append(line)
    text = "\n".join(lines)
    cut = text.find("Sau khi ho?n t?t JSON")
    if cut >= 0:
        text = text[:cut].rstrip()
    return text.strip()


def prompt_text(width: int, height: int) -> str:
    """User text sent with the full drawing."""
    body = skills_prompt_text()
    return (
        f"{body}\n\n"
        f"This image is width={width} height={height} pixels. "
        "Origin is top-left. JSON only."
    )
