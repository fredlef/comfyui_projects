#You’ll need to create a folder in custom_nodes/ and place this as manual_alpha_mask_painter.py.

"""
CustomManualAlphaMaskPainter for Tshirt Designer Workflow
Version: 1.0.0
Author: Fred LeFevre
Date: 2025-04-27
Description: Custom MaskPainter Node to convert a black mask to alpha.
"""

__version__ = "1.0.0"


import numpy as np
from PIL import Image
import torch

class ManualAlphaMaskPainter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "paint_mask"
    CATEGORY = "image/masking"

    def paint_mask(self, image, mask):
        # Convert tensors to numpy arrays
        image_np = image.cpu().numpy()[0] * 255
        image_np = image_np.astype(np.uint8)
        
        mask_np = mask.cpu().numpy()[0] * 255
        mask_np = mask_np.astype(np.uint8)

        # Create RGBA image
        img_rgba = Image.fromarray(image_np).convert("RGBA")
        rgba_array = np.array(img_rgba)
        
        # Apply mask to alpha channel
        rgba_array[:, :, 3] = mask_np
        
        # Convert back to tensor format
        result_image = torch.from_numpy(rgba_array.astype(np.float32) / 255.0).unsqueeze(0)
        result_mask = torch.from_numpy(mask_np.astype(np.float32) / 255.0).unsqueeze(0)

        return (result_image, result_mask)

NODE_CLASS_MAPPINGS = {
    "ManualAlphaMaskPainter": ManualAlphaMaskPainter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ManualAlphaMaskPainter": "Manual Alpha Mask Painter"
}
