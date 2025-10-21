# FSL / Ensure NHWC Batch (returns 1xH×W×3 float32)
import torch

class FSLEnsureNHWCBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_nhwc",)
    FUNCTION = "ensure_nhwc"
    CATEGORY = "FSL/Utils"

    def _to_nhwc(self, t: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(t)}")
        if t.dtype != torch.float32:
            t = t.float()

        # 3D cases: HWC / CHW / HCW
        if t.ndim == 3:
            if t.shape[-1] in (3, 4):          # HWC already
                a = t
            elif t.shape[0] in (3, 4):         # CHW -> HWC
                a = t.permute(1, 2, 0).contiguous()
            elif t.shape[1] in (3, 4):         # HCW -> HWC
                a = t.permute(0, 2, 1).contiguous()
            else:
                raise ValueError(f"3D tensor has unexpected shape {tuple(t.shape)}")
            return a.unsqueeze(0).clamp(0.0, 1.0)  # -> 1 x H x W x C

        # 4D cases: NHWC / NCHW / NHCW
        if t.ndim == 4:
            if t.shape[-1] in (3, 4):          # NHWC already
                a = t
            elif t.shape[1] in (3, 4):         # NCHW -> NHWC
                a = t.permute(0, 2, 3, 1).contiguous()
            elif t.shape[2] in (3, 4):         # NHCW -> NHWC
                a = t.permute(0, 1, 3, 2).contiguous()
            else:
                raise ValueError(f"4D tensor has unexpected shape {tuple(t.shape)}")
            return a.clamp(0.0, 1.0)

        raise ValueError(f"Expected 3D/4D tensor, got {t.ndim}D")

    def ensure_nhwc(self, image):
        t = image[0] if isinstance(image, list) else image
        t = self._to_nhwc(t)
        # Drop alpha if present so esrgan-like models stay happy
        if t.shape[-1] == 4:
            t = t[..., :3]
        return (t,)

NODE_CLASS_MAPPINGS = {"FSLEnsureNHWCBatch": FSLEnsureNHWCBatch}
NODE_DISPLAY_NAME_MAPPINGS = {"FSLEnsureNHWCBatch": "FSL / Ensure NHWC Batch"}
