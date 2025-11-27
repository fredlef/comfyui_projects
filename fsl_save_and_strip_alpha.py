import torch
import numpy as np
from PIL import Image
import os

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
        # Convert tensor to numpy array
        np_img = (image_rgba.cpu().numpy()[0] * 255).astype(np.uint8)  # shape: (H, W, 4)
        pil_img = Image.fromarray(np_img, mode="RGBA")
        pil_rgb = pil_img.convert("RGB")  # Remove alpha channel[5]

        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        pil_rgb.save(save_path)

        # Convert back to tensor for ComfyUI workflow (float32, 0-1, shape [1, H, W, 3])
        rgb_np = np.array(pil_rgb).astype(np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_np).unsqueeze(0)

        return (rgb_tensor,)

NODE_CLASS_MAPPINGS = {
    "FSL Save And Strip Alpha": FSL_SaveAndStripAlpha,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FSL Save And Strip Alpha": "💾 Save & Strip Alpha (FSL)",
}
