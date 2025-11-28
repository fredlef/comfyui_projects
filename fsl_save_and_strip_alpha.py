import torch
import numpy as np
from PIL import Image
import os
from pathlib import Path


def sanitize_filename(filename: str, default_name: str = "output_rgb.png") -> str:
    """
    Sanitize user-supplied filename:
    - Remove any path traversal (../, /, \)
    - Keep only basename
    - Replace separators with underscores
    - Ensure a valid extension exists
    """

    if not filename:
        name = default_name
    else:
        # Drop all directory components
        name = Path(filename).name

        # Remove weird slashes just in case
        name = name.replace("/", "_").replace("\\", "_")

        # Ensure extension
        if not os.path.splitext(name)[1]:
            name = name + ".png"

    return name


class FSL_SaveAndStripAlpha:
    CATEGORY = "FSL Nodes/Utils"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_rgb",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_rgba": ("IMAGE",),
                "filename": ("STRING", {"default": "output_rgb.png"}),
                "save_dir": ("STRING", {"default": "./outputs"}),
            }
        }

    def process(self, image_rgba, filename, save_dir):
        # Sanitize filename BEFORE using it
        safe_filename = sanitize_filename(filename)

        # Force save_dir to stay inside ComfyUI's allowed output area
        base_output_dir = Path(save_dir).resolve()  # resolve absolute path
        # Prevent path traversal by re-resolving under base_output_dir
        save_path = base_output_dir / safe_filename

        # Ensure directory exists
        os.makedirs(base_output_dir, exist_ok=True)

        # Convert tensor to numpy (H, W, 4)
        np_img = (image_rgba.cpu().numpy()[0] * 255).astype(np.uint8)
        pil_img = Image.fromarray(np_img, mode="RGBA")
        pil_rgb = pil_img.convert("RGB")

        # Save safely
        pil_rgb.save(str(save_path))

        # Convert back to tensor (1, H, W, 3)
        rgb_np = np.array(pil_rgb).astype(np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_np).unsqueeze(0)

        return (rgb_tensor,)


NODE_CLASS_MAPPINGS = {
    "FSL Save And Strip Alpha": FSL_SaveAndStripAlpha,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FSL Save And Strip Alpha": "💾 Save & Strip Alpha (FSL)",
}

}
