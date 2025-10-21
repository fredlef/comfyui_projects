
import os
import time
import base64
import random
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

try:
    import torch
except Exception:
    torch = None

try:
    from PIL import Image
except Exception:
    Image = None


# ---------------- Helpers ----------------

def _tensor_to_png_b64(t) -> str:
    """Convert torch.Tensor or np.ndarray image to base64 PNG."""
    if torch is not None and isinstance(t, torch.Tensor):
        x = t.detach().cpu().float()
        if x.dim() == 4 and x.shape[0] > 0:
            x = x[0]
        if x.dim() == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (3, 4):
            x = x.permute(1, 2, 0)
        arr = x.numpy()
    elif isinstance(t, np.ndarray):
        arr = t
    else:
        raise TypeError(f"Expected image as torch.Tensor or np.ndarray, got {type(t)}")

    if arr.max() <= 1.5:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    if Image is None:
        raise RuntimeError("Pillow is required to encode PNG images.")
    im = Image.fromarray(arr)
    buf = BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_base64_image(resp_json: Dict[str, Any]) -> Optional[str]:
    """Return first inline image payload from Gemini JSON response."""
    def _walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k, v
                yield from _walk(v)
        elif isinstance(node, list):
            for item in node:
                yield from _walk(item)

    for k, v in _walk(resp_json):
        if k in ("inline_data", "inlineData") and isinstance(v, dict):
            data = v.get("data")
            mime = v.get("mime_type") or v.get("mimeType") or ""
            if data and ("image/" in mime or mime == "" or mime is None):
                return data
    return None


def _post_with_backoff(url: str, headers: Dict[str, str], params: Dict[str, Any], payload: Dict[str, Any]):
    delay = 1.0
    last = None
    for _ in range(4):
        r = requests.post(url, headers=headers, params=params, json=payload, timeout=60)
        last = r
        if r.status_code != 429:
            return r
        time.sleep(delay)
        delay *= 2
    return last


MODEL_CHOICES = (
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-image-preview",
    "gemini-2.5-pro",
    "gemini-2.0-flash-exp",
    "gemini-2.0-pro-exp",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.0-pro-vision",
    "gemini-2.5-flash-audio-preview",
    "gemini-2.5-vision-preview",
)


class FSLGeminiGenerateImageV6:
    """Gemini image generator with iterative + external image toggles."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (MODEL_CHOICES, {"default": "gemini-2.5-flash-image-preview"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7FFFFFFF}),
                "use_init_image": ("BOOLEAN", {"default": True}),
                "use_images": ("BOOLEAN", {"default": False}),
                "refresh_models": ("BOOLEAN", {"default": False}),
                "output_prompt": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "init_image": ("IMAGE",),
                "images": ("IMAGE",),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "Leave blank to use GEMINI_API_KEY env var"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT", "FLOAT", "INT", "STRING", "DICT")
    RETURN_NAMES = (
        "image",
        "prompt",
        "model",
        "width",
        "height",
        "temperature",
        "seed",
        "seed_mode",
        "metadata",
    )
    FUNCTION = "generate"
    CATEGORY = "FSL"

    # ---------- internals ----------

    def _resolve_api_key(self, key_in: str) -> str:
        k = (key_in or "").strip()
        if not k:
            k = os.environ.get("GEMINI_API_KEY", "").strip()
        if not k:
            raise RuntimeError("Gemini API key missing. Provide it in the node or set GEMINI_API_KEY env var.")
        return k

    def _append_tensor_as_part(self, img_like, out: List[Dict[str, Any]]):
        b64 = _tensor_to_png_b64(img_like)
        out.append({"inline_data": {"mime_type": "image/png", "data": b64}})

    def _collect_reference_parts(
        self,
        use_init_image: bool,
        init_image,
        use_images: bool,
        images,
    ) -> List[Dict[str, Any]]:
        parts: List[Dict[str, Any]] = []

        # init first (if enabled)
        if use_init_image and init_image is not None:
            if isinstance(init_image, (list, tuple)):
                if len(init_image) > 0:
                    self._append_tensor_as_part(init_image[0], parts)
            elif torch is not None and isinstance(init_image, torch.Tensor):
                if init_image.dim() == 4 and init_image.shape[0] > 0:
                    self._append_tensor_as_part(init_image[0], parts)
                else:
                    self._append_tensor_as_part(init_image, parts)
            elif isinstance(init_image, np.ndarray):
                self._append_tensor_as_part(init_image, parts)

        # then all external images (if enabled)
        if use_images and images is not None:
            if isinstance(images, (list, tuple)):
                for t in images:
                    self._append_tensor_as_part(t, parts)
            elif torch is not None and isinstance(images, torch.Tensor):
                if images.dim() == 4:
                    for i in range(images.shape[0]):
                        self._append_tensor_as_part(images[i], parts)
                else:
                    self._append_tensor_as_part(images, parts)
            elif isinstance(images, np.ndarray):
                self._append_tensor_as_part(images, parts)

        return parts

    # ---------- main ----------

    def generate(
        self,
        prompt: str,
        model: str,
        width: int,
        height: int,
        temperature: float,
        seed: int,
        use_init_image: bool,
        refresh_models: bool,
        output_prompt: bool,
        use_images: bool = False,
        init_image: Optional["torch.Tensor"] = None,
        images: Optional["torch.Tensor"] = None,
        api_key: str = "",
    ) -> Tuple["torch.Tensor", str, str, int, int, float, int, str, Dict[str, Any]]:

        seed_mode = "fixed"
        if seed == -1:
            seed = random.randint(0, 0x7FFFFFFF)
            seed_mode = "randomize"

        api_key = self._resolve_api_key(api_key)
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"Content-Type": "application/json"}

        parts: List[Dict[str, Any]] = [{"text": prompt}]
        ref_parts = self._collect_reference_parts(use_init_image, init_image, use_images, images)
        parts.extend(ref_parts)

        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": float(temperature),
                "seed": int(seed),
            },
        }

        _ = refresh_models  # UI no-op

        resp = _post_with_backoff(endpoint, headers=headers, params={"key": api_key}, payload=payload)

        # Parse response
        try:
            resp_json = resp.json()
        except Exception as e:
            raise RuntimeError(f"Gemini response was not JSON: {e}")

        if isinstance(resp_json, dict) and "error" in resp_json:
            err = resp_json["error"]
            raise RuntimeError(f"Gemini error: {err.get('status')} - {err.get('message')}")

        b64_img = _extract_base64_image(resp_json)
        if not b64_img:
            raise RuntimeError(f"Failed to find inline image in Gemini response. First 1k of JSON: {str(resp_json)[:1000]}")

        try:
            img_bytes = base64.b64decode(b64_img)
            if Image is None:
                raise RuntimeError("Pillow is required to decode returned image.")
            pil = Image.open(BytesIO(img_bytes)).convert("RGBA")
        except Exception as e:
            raise RuntimeError(f"Failed to decode inline image: {e}")

        np_img = np.array(pil).astype(np.float32) / 255.0
        if torch is None:
            raise RuntimeError("torch is required for IMAGE tensor output.")
        image_tensor = torch.from_numpy(np_img)[None, ...]

        out_prompt = prompt if output_prompt else ""

        metadata_output = {
            "prompt": prompt,
            "model": model,
            "width": width,
            "height": height,
            "temperature": temperature,
            "seed": seed,
            "seed_mode": seed_mode,
            "use_init_image": bool(use_init_image and init_image is not None),
            "use_images": bool(use_images and images is not None),
            "init_image_connected": bool(init_image is not None),
            "batch_images": (
                int(images.shape[0]) if (torch is not None and isinstance(images, torch.Tensor) and images.dim() == 4)
                else (len(images) if isinstance(images, (list, tuple)) else (1 if images is not None else 0))
            ),
        }

        return (
            image_tensor,
            out_prompt,
            model,
            width,
            height,
            temperature,
            seed,
            seed_mode,
            metadata_output,
        )


NODE_CLASS_MAPPINGS = {"FSLGeminiGenerateImageV6": FSLGeminiGenerateImageV6}
NODE_DISPLAY_NAME_MAPPINGS = {"FSLGeminiGenerateImageV6": "FSL Gemini Generate Image V6"}
