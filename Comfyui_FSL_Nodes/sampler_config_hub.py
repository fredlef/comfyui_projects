# ─────────────────────────────────────────────────────────────────────────────
# Sampler Config Hub Node
# Created by: Fred & Mycroft (ChatGPT)
# Date: 2025-04-21
# Purpose: Outputs a consistent sampler config for KSampler nodes.
# Workflow: Used in Tshirt-SDXL-Gold-*.json workflows.
# ─────────────────────────────────────────────────────────────────────────────

class SamplerConfigHub:
    """
    Custom node that emits a unified sampler config dictionary.
    This allows centralized control over sampler settings when using multiple
    KSampler nodes in a workflow (e.g., for SDXL base and refiner).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": ([
                    "euler", "euler_ancestral", "heun",
                    "dpm_2", "dpm_2_ancestral", "lms",
                    "dpm_fast", "dpm_adaptive",
                    "dpmpp_2s_ancestral", "dpmpp_2m", "dpmpp_sde",
                    "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde",
                    "ddim", "plms"
                ],),
                "scheduler": (["normal", "karras", "exponential", "sgm_uniform"],),
                "steps": (int, {"default": 30, "min": 1, "max": 150}),
                "cfg": (float, {"default": 7.5, "min": 0.0, "max": 20.0}),
                "seed": (int, {"default": 42}),
            }
        }

    RETURN_TYPES = ("SAMPLER_CONFIG",)
    RETURN_NAMES = ("sampler_config",)
    FUNCTION = "build_config"
    CATEGORY = "utils"

    def build_config(self, sampler_name, scheduler, steps, cfg, seed):
        return ({
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
        },)

NODE_CLASS_MAPPINGS = {
    "SamplerConfigHub": SamplerConfigHub
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerConfigHub": "Sampler Config Hub"
}