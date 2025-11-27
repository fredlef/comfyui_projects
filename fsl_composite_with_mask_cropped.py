import torch
import numpy as np
from PIL import Image
import os

class FSL_CompositeWithMask:
    CATEGORY = "FSL Nodes/Utils"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composited_rgb",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "foreground": ("IMAGE",),
                "mask": ("MASK",),
                "background": ("IMAGE",),
                "filename": ("STRING", {"default": "composited_rgb.png"}),
                "save_dir": ("STRING", {"default": "./outputs"}),
            }
        }

    def process(self, foreground=None, mask=None, background=None, filename="composited_rgb.png", save_dir="./outputs"):
        # Ensure filename has valid extension
        valid_extensions = [".png", ".jpg", ".jpeg", ".webp"]
        if not any(filename.lower().endswith(ext) for ext in valid_extensions):
            filename += ".png"  # Default to PNG

        # Set default size
        DEFAULT_SIZE = (512, 512)

        # Create default background if not provided (black)
        if background is None:
            bg_np = np.zeros((DEFAULT_SIZE[1], DEFAULT_SIZE[0], 3), dtype=np.uint8)
        else:
            bg_np = (background.cpu().numpy()[0] * 255).astype(np.uint8)
            if bg_np.shape[-1] == 4:
                bg_np = bg_np[..., :3]

        h, w = bg_np.shape[:2]

        # Create default mask if not provided (all white)
        if mask is None:
            mask_np = np.ones((h, w), dtype=np.uint8) * 255
        else:
            mask_np = (mask.cpu().numpy()[0] * 255).astype(np.uint8)
            if mask_np.shape != (h, w):
                mask_np = np.array(Image.fromarray(mask_np).resize((w, h), Image.BILINEAR))

        # Prepare mask for blending (0.0-1.0 float, shape HxWx1)
        mask_f = (mask_np.astype(np.float32) / 255.0)[..., None]

        # If foreground is provided, blend; else just use background
        if foreground is not None:
            fg_np = (foreground.cpu().numpy()[0] * 255).astype(np.uint8)
            if fg_np.shape[-1] == 4:
                fg_np = fg_np[..., :3]
            if fg_np.shape[:2] != (h, w):
                fg_np = np.array(Image.fromarray(fg_np).resize((w, h), Image.BILINEAR))
            composited = fg_np.astype(np.float32) * mask_f + bg_np.astype(np.float32) * (1.0 - mask_f)
        else:
            composited = bg_np.astype(np.float32) * (1.0 - mask_f)

        composited = composited.astype(np.uint8)

        # Convert composited image and mask to PIL
        pil_img = Image.fromarray(composited, mode="RGB")
        pil_mask = Image.fromarray(mask_np, mode="L")

        # Calculate bounding box of mask (nonzero areas)
        bbox = pil_mask.getbbox()
        if bbox:
            pil_img = pil_img.crop(bbox)

        # Save cropped RGB image
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        pil_img.save(save_path)

        # Convert back to tensor
        out_np = np.array(pil_img).astype(np.float32) / 255.0
        out_tensor = torch.from_numpy(out_np).unsqueeze(0)
        return (out_tensor,)


NODE_CLASS_MAPPINGS = {
    "FSL Composite With Mask": FSL_CompositeWithMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FSL Composite With Mask": "🖼️ Composite With Mask Cropped (FSL)",
}
