# FSLGeminiGenerateImageV8.py
import os, time, base64, random
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

try:
    import torch
except Exception:
    torch = None

try:
    from PIL import Image, ImageOps, ImageFilter
except Exception:
    Image = None


# -------- helpers --------
def _tensor_like_to_numpy(img_like):
    """Accept torch or np tensor/array and return uint8 ndarray in HxWxC or HxW."""
    if torch is not None and isinstance(img_like, torch.Tensor):
        t = img_like.detach().cpu().float()
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 3 and t.shape[0] in (1, 3, 4) and t.shape[-1] not in (1, 3, 4):
            t = t.permute(1, 2, 0)  # CHW -> HWC
        arr = t.numpy()
    elif isinstance(img_like, np.ndarray):
        arr = img_like
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(img_like)}")

    # scale to 0..255
    if arr.dtype != np.uint8 or arr.max() <= 1.5:
        arr = np.clip(arr * (255.0 if arr.max() <= 1.5 else 1.0), 0, 255).astype(np.uint8)
    return arr


def _to_pil(img_like) -> Image.Image:
    arr = _tensor_like_to_numpy(img_like)
    if Image is None:
        raise RuntimeError("Pillow is required.")
    if arr.ndim == 2:
        return Image.fromarray(arr, "L")
    c = arr.shape[2] if arr.ndim == 3 else 1
    if c == 1:
        return Image.fromarray(arr[:, :, 0], "L")
    if c == 3:
        return Image.fromarray(arr, "RGB")
    if c == 4:
        return Image.fromarray(arr, "RGBA")
    return Image.fromarray(arr[:, :, :3], "RGB")


def _tensor_to_png_b64(img_like) -> str:
    pil = _to_pil(img_like)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _prepare_mask_png_b64(
    mask_like=None,
    mask_image_like=None,
    target_size=None,
    invert: bool = False,
    feather_radius: int = 0,
    strength: float = 1.0,
) -> Optional[str]:
    """
    Build an 8-bit 'L' mask PNG b64.
      - White (255) = editable
      - Black (0)   = preserved
    Optional transforms: invert, feather (GaussianBlur), strength (0..1 linear scale).
    """
    if mask_like is None and mask_image_like is None:
        return None
    src = mask_like if mask_like is not None else mask_image_like
    m = _to_pil(src).convert("L")
    if target_size is not None and m.size != target_size:
        m = m.resize(target_size, resample=Image.NEAREST)
    if invert:
        m = ImageOps.invert(m)
    if feather_radius and feather_radius > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=int(feather_radius)))
    if strength is not None:
        s = max(0.0, min(1.0, float(strength)))
        if s < 1.0:
            # linear scale towards 0 (stronger protection)
            m = m.point(lambda px: int(px * s))
    buf = BytesIO()
    m.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_base64_image(resp_json: Dict[str, Any]) -> Optional[str]:
    def _walk(n):
        if isinstance(n, dict):
            for k, v in n.items():
                yield k, v
                yield from _walk(v)
        elif isinstance(n, list):
            for i in n:
                yield from _walk(i)
    for k, v in _walk(resp_json):
        if k in ("inline_data", "inlineData") and isinstance(v, dict):
            data = v.get("data")
            mime = v.get("mime_type") or v.get("mimeType") or ""
            if data and ("image/" in mime or mime in ("", None)):
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
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-image-preview",
    "gemini-2.0-flash-exp",
)


class FSLGeminiGenerateImageV8:
    """Gemini image generator with inpainting support.
       Base selection: auto / init_image / images.
       Mask inputs: MASK or IMAGE, with invert/feather/strength options.
    """

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
                "use_mask": ("BOOLEAN", {"default": False}),
                "base_for_inpaint": (("auto", "init_image", "images"), {"default": "auto"}),
                "refresh_models": ("BOOLEAN", {"default": False}),
                "output_prompt": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "init_image": ("IMAGE",),
                "images": ("IMAGE",),
                "mask": ("MASK",),        # true mask
                "mask_image": ("IMAGE",), # image as mask
                "invert_mask": ("BOOLEAN", {"default": False}),
                "feather_radius": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "Leave blank to use GEMINI_API_KEY env var"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT", "FLOAT", "INT", "STRING", "DICT")
    RETURN_NAMES  = ("image", "prompt", "model", "width", "height", "temperature", "seed", "seed_mode", "metadata")
    FUNCTION = "generate"
    CATEGORY = "FSL"

    # ---- internals ----
    def _resolve_api_key(self, k: str) -> str:
        k = (k or "").strip() or os.environ.get("GEMINI_API_KEY", "").strip()
        if not k:
            raise RuntimeError("Gemini API key missing. Provide it in the node or set GEMINI_API_KEY.")
        return k

    def _append_tensor_as_part(self, img_like, out: List[Dict[str, Any]]):
        out.append({"inline_data": {"mime_type": "image/png", "data": _tensor_to_png_b64(img_like)}})

    def _iter_imgs(self, imgs):
        if imgs is None:
            return []
        if isinstance(imgs, (list, tuple)):
            return list(imgs)
        if torch is not None and isinstance(imgs, torch.Tensor):
            if imgs.dim() == 4:
                return [imgs[i] for i in range(imgs.shape[0])]
            return [imgs]
        if isinstance(imgs, np.ndarray):
            return [imgs]
        return []

    def _collect_parts_with_base_choice(
        self,
        base_choice: str,
        use_init_image: bool, init_image,
        use_images: bool, images,
        use_mask: bool, mask, mask_image,
        invert_mask: bool, feather_radius: int, mask_strength: float,
    ) -> Tuple[List[Dict[str, Any]], Optional[Tuple[int, int]], str]:
        """
        Returns (parts, base_size, base_source)
        base_source in {"init_image","images","none"}
        """
        parts: List[Dict[str, Any]] = []
        base_size = None
        base_source = "none"

        imgs_list = self._iter_imgs(images)

        if base_choice == "init_image" and use_init_image and init_image is not None:
            # base = init_image
            if isinstance(init_image, (list, tuple)) and len(init_image) > 0:
                pil = _to_pil(init_image[0]); base_size = pil.size; self._append_tensor_as_part(init_image[0], parts)
            elif torch is not None and isinstance(init_image, torch.Tensor):
                t = init_image[0] if (init_image.dim() == 4 and init_image.shape[0] > 0) else init_image
                pil = _to_pil(t); base_size = pil.size; self._append_tensor_as_part(t, parts)
            elif isinstance(init_image, np.ndarray):
                pil = _to_pil(init_image); base_size = pil.size; self._append_tensor_as_part(init_image, parts)
            base_source = "init_image"

            # mask after base
            if use_mask and (mask is not None or mask_image is not None):
                b64 = _prepare_mask_png_b64(mask, mask_image, base_size, invert_mask, feather_radius, mask_strength)
                if b64:
                    parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})

            # remaining refs
            if use_images and imgs_list:
                for t in imgs_list:
                    self._append_tensor_as_part(t, parts)
            return parts, base_size, base_source

        if base_choice == "images" and use_images and imgs_list:
            # promote first image as base
            first = imgs_list[0]
            pil = _to_pil(first); base_size = pil.size; self._append_tensor_as_part(first, parts)
            base_source = "images"

            # mask after base
            if use_mask and (mask is not None or mask_image is not None):
                b64 = _prepare_mask_png_b64(mask, mask_image, base_size, invert_mask, feather_radius, mask_strength)
                if b64:
                    parts.append({"mime_type": "image/png", "inline_data": {"mime_type": "image/png", "data": b64}})  # keep schema consistent
                    # Note: either {"inline_data":{...}} or the same dict; we use the same structure as other parts
                    parts[-1] = {"inline_data": {"mime_type": "image/png", "data": b64}}

            # rest as refs
            for t in imgs_list[1:]:
                self._append_tensor_as_part(t, parts)
            return parts, base_size, base_source

        # no valid base selected — still push refs if present
        if use_images and imgs_list:
            for t in imgs_list:
                if base_size is None:
                    base_size = _to_pil(t).size
                self._append_tensor_as_part(t, parts)

        # If mask without a base, we can still send it (model may ignore)
        if use_mask and (mask is not None or mask_image is not None):
            b64 = _prepare_mask_png_b64(mask, mask_image, base_size, invert_mask, feather_radius, mask_strength)
            if b64:
                parts.append({"inline_data": {"mime_type": "image/png", "data": b64}})

        return parts, base_size, base_source

    # ---- main ----
    def generate(
        self,
        prompt: str, model: str, width: int, height: int, temperature: float, seed: int,
        use_init_image: bool, use_images: bool, use_mask: bool, base_for_inpaint: str,
        refresh_models: bool, output_prompt: bool,
        init_image: Optional["torch.Tensor"] = None, images: Optional["torch.Tensor"] = None,
        mask: Optional["torch.Tensor"] = None, mask_image: Optional["torch.Tensor"] = None,
        invert_mask: bool = False, feather_radius: int = 0, mask_strength: float = 1.0,
        api_key: str = "",
    ):
        seed_mode = "fixed"
        if seed == -1:
            seed = random.randint(0, 0x7FFFFFFF)
            seed_mode = "randomize"

        api_key = self._resolve_api_key(api_key)
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"Content-Type": "application/json"}

        # Resolve effective base with AUTO fallback
        requested = base_for_inpaint
        have_init   = bool(use_init_image and (init_image is not None))
        have_images = bool(use_images and (images is not None))

        if requested == "auto":
            base_choice = "init_image" if have_init else ("images" if have_images else "none")
        else:
            if requested == "init_image" and not have_init:
                base_choice = "images" if have_images else "none"
            elif requested == "images" and not have_images:
                base_choice = "init_image" if have_init else "none"
            else:
                base_choice = requested

        # Build parts
        parts: List[Dict[str, Any]] = []
        if use_mask and (mask is not None or mask_image is not None):
            parts.append({"text":
                "INPAINT MODE: You are given a base image and a mask image. "
                "Modify ONLY pixels where the mask is white; keep black (unmasked) pixels exactly the same as the base image. "
                "Respect edges and composition."
            })
        parts.append({"text": prompt})

        ref_parts, base_size, resolved_base = self._collect_parts_with_base_choice(
            base_choice=base_choice,
            use_init_image=use_init_image, init_image=init_image,
            use_images=use_images, images=images,
            use_mask=use_mask, mask=mask, mask_image=mask_image,
            invert_mask=invert_mask, feather_radius=feather_radius, mask_strength=mask_strength,
        )
        parts.extend(ref_parts)

        if use_mask and (mask is not None or mask_image is not None) and resolved_base == "none":
            raise RuntimeError("Inpaint requires a base image. Enable and connect either init_image or images.")

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": float(temperature), "seed": int(seed)},
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
        metadata = {
            "prompt": prompt, "model": model, "width": width, "height": height,
            "temperature": temperature, "seed": seed, "seed_mode": seed_mode,
            "use_init_image": bool(use_init_image and init_image is not None),
            "use_images": bool(use_images and images is not None),
            "use_mask": bool(use_mask and (mask is not None or mask_image is not None)),
            "mask_source": "MASK" if mask is not None else ("IMAGE" if mask_image is not None else "none"),
            "requested_base_for_inpaint": base_for_inpaint,
            "effective_base_for_inpaint": base_choice,
            "resolved_base_source": resolved_base,
            "invert_mask": bool(invert_mask),
            "feather_radius": int(feather_radius),
            "mask_strength": float(mask_strength),
        }
        return (image_tensor, out_prompt, model, width, height, temperature, seed, seed_mode, metadata)


NODE_CLASS_MAPPINGS = {"FSLGeminiGenerateImageV8": FSLGeminiGenerateImageV8}
NODE_DISPLAY_NAME_MAPPINGS = {"FSLGeminiGenerateImageV8": "FSL Gemini Generate Image V8"}
