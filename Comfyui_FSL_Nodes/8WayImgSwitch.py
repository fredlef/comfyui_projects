# fsl_nodes.py

# 8-Way Image Switch Node for ComfyUI
class Image_Switch_8Way:
    CATEGORY = "FSL Nodes/Switches"  # This controls sidebar grouping in ComfyUI
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("img_out",)
    FUNCTION = "get_image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 8}),
            },
            "optional": {
                "img_1": ("IMAGE",),
                "img_2": ("IMAGE",),
                "img_3": ("IMAGE",),
                "img_4": ("IMAGE",),
                "img_5": ("IMAGE",),
                "img_6": ("IMAGE",),
                "img_7": ("IMAGE",),
                "img_8": ("IMAGE",),
            }
        }

    def get_image(self, select, img_1=None, img_2=None, img_3=None, img_4=None,
                  img_5=None, img_6=None, img_7=None, img_8=None):
        # Default to img_1 if selection is invalid or missing
        images = [img_1, img_2, img_3, img_4, img_5, img_6, img_7, img_8]
        idx = max(1, min(select, 8)) - 1  # Clamp select to 1-8, convert to 0-based index
        img_out = images[idx]
        return (img_out,)

# Example: Another custom node for demonstration (can be removed or replaced)
class FSL_Image_Passthrough:
    CATEGORY = "FSL Nodes/Switches"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("img_out",)
    FUNCTION = "passthrough"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "img_in": ("IMAGE",),
            }
        }

    def passthrough(self, img_in):
        return (img_in,)

# Node registration dictionaries
NODE_CLASS_MAPPINGS = {
    "FSL Image Switch 8Way": Image_Switch_8Way,
    "FSL Image Passthrough": FSL_Image_Passthrough,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # Optional: Controls how the node appears in the sidebar (can use emojis, etc.)
    "FSL Image Switch 8Way": "🔀 8-Way Image Switch (FSL)",
    "FSL Image Passthrough": "➡️ Image Passthrough (FSL)",
}
