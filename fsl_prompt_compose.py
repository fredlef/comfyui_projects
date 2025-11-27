# FSL / Prompt Compose: merge Positive + Negative + optional scene lock into one STRING prompt.

class FSLPromptCompose:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("STRING", {"multiline": True, "default": "Describe the change you want…"}),
                "negative": ("STRING", {"multiline": True, "default": "studio backdrop, black background, portrait-only, isolation, cropping"}),
                "lock_scene": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "compose"
    CATEGORY = "FSL/Utils"

    def compose(self, positive, negative, lock_scene):
        lock = ""
        if lock_scene:
            lock = (
                "Use the provided image as the base. "
                "Keep composition, subjects, positions, background, lighting, camera angle, clothing and colors unchanged. "
                "Change only what I specify. "
                "Do not crop; do not isolate a single subject; do not change the background.\n"
            )
        out = f"{lock}{positive.strip()}\n\nNEGATIVE: {negative.strip()}"
        return (out,)

NODE_CLASS_MAPPINGS = {"FSLPromptCompose": FSLPromptCompose}
NODE_DISPLAY_NAME_MAPPINGS = {"FSLPromptCompose": "FSL / Prompt Compose"}
