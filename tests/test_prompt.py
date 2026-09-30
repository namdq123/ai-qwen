"""Prompt text. Does not load model weights."""

from app.prompt import image_content, prompt_text, qwen_pixel_limits
from app.settings import Settings


def test_prompt_uses_skills_md() -> None:
    text = prompt_text(1024, 768)
    assert '"s"' in text and '"r"' in text
    assert "width=1024" in text
    assert '["roof",[[x,y],[x,y],[x,y],...]]' in text
    assert '["wall",[[x,y],[x,y],[x,y],...]]' in text
    assert '["window",[[x,y],[x,y],[x,y],...]]' in text
    assert '["door",[[x,y],[x,y],[x,y],...]]' in text
    assert "exterior house painting" in text
    assert "follows the visible outer boundary" in text
    assert "Do not repeat a point." in text
    assert "Wall polygons must not overlap each other." in text
    assert "Merge wall regions into one polygon" in text
    assert "does not split it" in text
    assert "Do not return one wall polygon that contains a roof." in text
    assert "Do not create one polygon for the whole facade." in text
    assert "Never merge a roof with a wall." in text
    assert '["opening",[[x,y],[x,y],[x,y],...]]' in text
    assert '["foundation",[[x,y],[x,y],[x,y],...]]' in text
    assert "Walls are the main paint area." in text
    assert "Use pixel coordinates." in text
    assert '["soffit",[[x,y],[x,y],[x,y],...]]' in text
    assert '["fascia",[[x,y],[x,y],[x,y],...]]' in text
    assert "Group 1 is the paint area." not in text
    assert "same outer frame are one window" in text
    assert "same frame are one door" in text
    assert "Do not draw one polygon that wraps around a roof" in text
    assert "Stop at a window or door." not in text
    assert "?nh n?y" not in text
    assert "sk-" not in text


def test_budget_caps_visual_tokens(monkeypatch) -> None:
    monkeypatch.setattr(Settings, "QWEN_VL_MIN_PIXELS", 4 * 28 * 28)
    monkeypatch.setattr(Settings, "QWEN_VL_MAX_PIXELS", 256 * 28 * 28)
    min_px, max_px = qwen_pixel_limits()
    assert min_px == 4 * 28 * 28
    assert max_px == 256 * 28 * 28
    assert max_px // (28 * 28) == 256


def test_image_content_passes_max_pixels(monkeypatch) -> None:
    monkeypatch.setattr(Settings, "QWEN_VL_MIN_PIXELS", 4 * 28 * 28)
    monkeypatch.setattr(Settings, "QWEN_VL_MAX_PIXELS", 128 * 28 * 28)
    content = image_content("pil-placeholder")
    assert content["type"] == "image"
    assert content["image"] == "pil-placeholder"
    assert content["min_pixels"] == 4 * 28 * 28
    assert content["max_pixels"] == 128 * 28 * 28
