"""
FSL Sampler & Scheduler Config Node

Version: 1.0.0
Author: Fred LeFevre
Date: 2025-04-27

Description: Custom Sampler Node to feed settings to metadata.
"""

__version__ = "1.0.0"

class FSL_SamplerSchedulerConfig:
    CATEGORY = "FSL Nodes/Utils"
    RETURN_TYPES = ("SAMPLER", "SCHEDULER")
    RETURN_NAMES = ("sampler", "scheduler")
    FUNCTION = "bundle"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": ([
                    "euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp",
                    "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast",
                    "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp",
                    "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_cfg_pp",
                    "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde",
                    "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ipndm", "ipndm_v", "deis",
                    "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral",
                    "res_multistep_ancestral_cfg_pp", "gradient_estimation", "er_sde",
                    "ddim", "uni_pc", "uni_pc_bh2"
                ], {"default": "euler"}),
                "scheduler_name": ([
                    "normal", "karras", "exponential", "sgm_uniform",
                    "simple", "ddim_uniform", "beta", "linear_quadratic", "kl_optimal"
                ], {"default": "karras"})
            }
        }

    def bundle(self, sampler_name, scheduler_name):
        return (sampler_name, scheduler_name)

NODE_CLASS_MAPPINGS = {
    "FSL Sampler & Scheduler Config": FSL_SamplerSchedulerConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FSL Sampler & Scheduler Config": "🛠️ Sampler & Scheduler Config (FSL)",
}
