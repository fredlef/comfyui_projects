class SamplerConfigHub:
    """
    Custom node that emits a unified sampler config dictionary.
    This allows centralized control over sampler settings when using multiple
    KSampler nodes in a workflow (e.g., for SDXL base and refiner).
    """
    CATEGORY = "FSL Nodes/Utils"  # Changed to match your preferred category
    RETURN_TYPES = ("SAMPLER_CONFIG",)
    RETURN_NAMES = ("sampler_config",)
    FUNCTION = "build_config"

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
                "steps": ("INT", {"default": 30, "min": 1, "max": 150}),
                "cfg": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0}),
                "seed": ("INT", {"default": 42}),
            }
        }

    def build_config(self, sampler_name, scheduler, steps, cfg, seed):
        return ({
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
        },)

NODE_CLASS_MAPPINGS = {
    "FSL SamplerConfigHub": SamplerConfigHub,  # Added prefix and comma
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FSL SamplerConfigHub": "🎛️ Sampler Config Hub (FSL)",  # Added prefix
}
