import os
import json
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image, PngImagePlugin
import folder_paths
from datetime import datetime


class FSLImageSaverWithMetadata:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE",),
                "filename": ("STRING", {"default": "gemini_out"}),
                "save_to_subfolder": ("STRING", {"default": ""}),
            },
            "optional": {
                "append_datetime": ("BOOLEAN", {"default": False}),
                "model_string": ("STRING", {"default": ""}),
                "modelname": ("STRING", {"default": ""}),
                "seed_string": ("STRING", {"default": ""}),
                "width_string": ("STRING", {"default": ""}),
                "height_string": ("STRING", {"default": ""}),
                "temperature_string": ("STRING", {"default": ""}),
                "positive_string": ("STRING", {"default": ""}),
                "negative_string": ("STRING", {"default": ""}),
                "cfg_string": ("STRING", {"default": ""}),
                "steps_string": ("STRING", {"default": ""}),
                "sampler_string": ("STRING", {"default": ""}),
                "scheduler_string": ("STRING", {"default": ""}),
                "extra_info": ("STRING", {"default": ""}),
                "external_metadata": ("DICT", {}),
            },
        }

    RETURN_TYPES = ("DICT", "IMAGE",)
    RETURN_NAMES = ("METADATA", "IMAGE",)
    FUNCTION = "save_images"
    CATEGORY = "FSL/IO"

    def _to_numpy_image(self, img: torch.Tensor) -> np.ndarray:
        """
        Convert a single image tensor to a uint8 numpy array [H,W,C].
        Expects img in [H,W,C] or [C,H,W] with float 0..1 or uint8.
        """
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
        else:
            arr = np.asarray(img)

        # Handle [C,H,W] -> [H,W,C]
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] != arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))

        # Normalize float images to 0..255
        if arr.dtype in (np.float32, np.float64, np.float16):
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = arr.astype(np.uint8)

        # Ensure we have at least 3 channels
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)

        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        return arr

    @staticmethod
    def _human_readable_size(num_bytes: int) -> str:
        """
        Return a human-readable file size, e.g. '1.23 MB'.
        """
        if num_bytes is None:
            return ""
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        size = float(num_bytes)
        for unit in units:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} PB"

    def save_images(
        self,
        images: torch.Tensor,
        filename: str,
        save_to_subfolder: str = "",
        append_datetime: bool = False,
        model_string: str = "",
        modelname: str = "",
        seed_string: str = "",
        width_string: str = "",
        height_string: str = "",
        temperature_string: str = "",
        positive_string: str = "",
        negative_string: str = "",
        cfg_string: str = "",
        steps_string: str = "",
        sampler_string: str = "",
        scheduler_string: str = "",
        extra_info: str = "",
        external_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor]:

        # Base Comfy output directory
        base = folder_paths.get_output_directory()

        # User subfolder, normalized and forced to be relative
        sub = (save_to_subfolder or "").strip()
        sub = os.path.normpath(sub).replace("\\", "/").lstrip("/")

        # Absolute directory we will actually save into
        outdir = os.path.join(base, sub) if sub else base
        os.makedirs(outdir, exist_ok=True)

        # Normalize batch dimension
        if images.dim() == 3:
            images = images.unsqueeze(0)

        # One canonical, timezone-aware timestamp used everywhere
        now_local = datetime.now().astimezone()
        ts_file = now_local.strftime("%Y%m%d_%H%M%S")        # e.g. 20251014_163733
        iso_human = now_local.isoformat(timespec="seconds")  # e.g. 2025-10-14T16:37:33-04:00

        if append_datetime:
            filename = f"{filename}_{ts_file}"

        model_value = (model_string or modelname).strip()

        # Build a "parameters" text similar to standard Comfy / A1111
        lines: List[str] = []
        if model_value:
            lines.append("model: " + model_value)
        if seed_string:
            lines.append("seed: " + seed_string)
        if width_string or height_string:
            lines.append(f"size: {width_string}x{height_string}")
        if temperature_string:
            lines.append("temperature: " + temperature_string)
        if cfg_string:
            lines.append("cfg: " + cfg_string)
        if steps_string:
            lines.append("steps: " + steps_string)
        if sampler_string:
            lines.append("sampler: " + sampler_string)
        if scheduler_string:
            lines.append("scheduler: " + scheduler_string)
        if positive_string:
            lines.append("positive: " + positive_string)
        if negative_string:
            lines.append("negative: " + negative_string)
        if extra_info:
            lines.append("extra: " + extra_info)

        parameters_text = "\n".join(lines)

        pnginfo = PngImagePlugin.PngInfo()
        if parameters_text:
            pnginfo.add_text("parameters", parameters_text)
        if external_metadata:
            try:
                pnginfo.add_text("fsl_metadata", json.dumps(external_metadata, ensure_ascii=False))
            except Exception:
                pnginfo.add_text("fsl_metadata", str(external_metadata))

        metadata_raw_out: Dict[str, Any] = {
            "fileinfo": {
                "filename": "",
                "resolution": "",
                "date": "",
                "size_bytes": 0,
                "size_human": "",
                "size": "",
                "format": "PNG",
            }
        }

        preview_tensor: Optional[torch.Tensor] = None

        for i in range(images.shape[0]):
            np_img = self._to_numpy_image(images[i])
            h, w = np_img.shape[0], np_img.shape[1]

            # Explicit mode preserves alpha if present
            mode = "RGBA" if np_img.shape[-1] == 4 else ("RGB" if np_img.shape[-1] == 3 else None)
            pil = Image.fromarray(np_img, mode=mode) if mode else Image.fromarray(np_img)

            # Use Comfy's helper with a RELATIVE subfolder, not an absolute path
            try:
                # 'sub' is something like "metadata/gem3" or ""
                ret = folder_paths.get_save_image_path(filename, sub, i)
            except Exception:
                ret = None

            # Known return: (full_output_folder, filename_prefix, counter, subfolder, filename_with_subfolder)
            if isinstance(ret, (list, tuple)) and len(ret) >= 3:
                full_output_folder, filename_prefix, counter = ret[0], ret[1], ret[2]
                path = os.path.join(full_output_folder, f"{filename_prefix}_{counter:05}.png")
            else:
                # Fallback: ensure our own dir exists and build the path manually
                os.makedirs(outdir, exist_ok=True)
                path = os.path.join(outdir, f"{filename}_{i:05}.png")

            pil.save(path, format="PNG", pnginfo=pnginfo)

            # Fill metadata_raw_out only from the first image (representative)
            if i == 0:
                try:
                    stat = os.stat(path)
                    size_bytes = stat.st_size
                except Exception:
                    size_bytes = 0

                metadata_raw_out["fileinfo"] = {
                    "filename": os.path.basename(path),
                    "resolution": f"{w}x{h}",
                    "date": iso_human,
                    "size_bytes": size_bytes,
                    "size_human": self._human_readable_size(size_bytes),
                    "size": self._human_readable_size(size_bytes),
                    "format": "PNG",
                }

            # Keep first image as preview
            if preview_tensor is None:
                # Return normalized [1,H,W,C] float tensor
                preview_tensor = torch.from_numpy(np_img.astype(np.float32) / 255.0)[None, ...]

        # Safety: if nothing was saved (empty batch), provide a blank preview
        if preview_tensor is None:
            preview_tensor = torch.zeros((1, 64, 64, 4), dtype=torch.float32)

        # Return METADATA first (compat), IMAGE second (for optional preview wiring)
        return (metadata_raw_out, preview_tensor,)


NODE_CLASS_MAPPINGS = {"FSLImageSaverWithMetadata": FSLImageSaverWithMetadata}
NODE_DISPLAY_NAME_MAPPINGS = {"FSLImageSaverWithMetadata": "FSL Image Saver w/ Metadata"}
