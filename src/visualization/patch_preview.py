"""Preview utilities for patch selection over small images."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class PatchBox:
    """Patch rectangle metadata for visual previews."""

    x: int
    y: int
    width: int
    height: int
    selected: bool


def save_patch_selection_preview(
    rgb_image: Image.Image,
    patches: list[PatchBox],
    output_path: str | Path,
    selected_color: tuple[int, int, int] = (0, 180, 0),
    rejected_color: tuple[int, int, int] = (220, 40, 40),
) -> Path:
    """Draw patch rectangles over an RGB image and save the preview.

    Green rectangles indicate selected patches and red rectangles indicate
    rejected patches. This helper is intended for ordinary images during early
    development, not for direct rendering of gigapixel WSI.
    """
    preview = rgb_image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    line_width = max(1, min(preview.size) // 180)

    for patch in patches:
        color = selected_color if patch.selected else rejected_color
        x0 = patch.x
        y0 = patch.y
        x1 = min(preview.width - 1, patch.x + patch.width - 1)
        y1 = min(preview.height - 1, patch.y + patch.height - 1)
        if x0 >= preview.width or y0 >= preview.height:
            continue
        max_patch_line_width = max(1, min(patch.width, patch.height) // 2)
        for offset in range(min(line_width, max_patch_line_width)):
            if x0 + offset > x1 - offset or y0 + offset > y1 - offset:
                break
            draw.rectangle(
                (x0 + offset, y0 + offset, x1 - offset, y1 - offset),
                outline=color,
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output)
    return output
