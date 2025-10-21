
import os
import json
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image, PngImagePlugin
import folder_paths
from datetime import datetime

class FSLImageSaverWithMetadataV5:
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
                "guidance_scale_string": ("STRING", {"default": ""}),
                "watermark_string": ("STRING", {"default": ""}),
                "external_metadata": ("DICT", {"default": {}}),
            },
        }

    # Keep METADATA_RAW as output #1 for compatibility.
    # Add IMAGE as output #2 so you can optionally preview.
    RETURN_TYPES = ("METADATA_RAW", "IMAGE",)
    RETURN_NAMES = ("metadata_raw", "image",)
    FUNCTION = "save_files"
    OUTPUT_NODE = True
    CATEGORY = "FSL/Save"

    def _to_numpy_image(self, t: torch.Tensor) -> np.ndarray:
        """Convert a tensor to HWC uint8, preserving alpha if present."""
        if t.device.type != "cpu":
            t = t.cpu()
        t = t.detach().float()
        # If CHW, convert to HWC
        if t.dim() == 3 and t.shape[0] in (1, 3, 4) and t.shape[-1] not in (3, 4):
            t = t.permute(1, 2, 0)
        arr = (t.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return arr

    def save_files(
        self,
        images: torch.Tensor,
        filename: str,
        save_to_subfolder: str,
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
        guidance_scale_string: str = "",
        watermark_string: str = "",
        external_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], torch.Tensor]:

        base = folder_paths.get_output_directory()
        sub = (save_to_subfolder or "").strip()
        sub = os.path.normpath(sub).replace("\\", "/").lstrip("/")
        outdir = os.path.join(base, sub) if sub else base
        os.makedirs(outdir, exist_ok=True)

        # Normalize batch
        if images.dim() == 3:
            images = images.unsqueeze(0)

        # One canonical, timezone-aware timestamp used everywhere
        now_local = datetime.now().astimezone()
        ts_file = now_local.strftime("%Y%m%d_%H%M%S")        # e.g. 20251014_163733
        iso_human = now_local.isoformat(timespec="seconds")  # e.g. 2025-10-14T16:37:33-04:00

        if append_datetime:
            filename = f"{filename}_{ts_file}"

        model_value = (model_string or modelname).strip()

        lines: List[str] = []
        if model_value:
            lines.append("model: " + model_value)
        if seed_string:
            lines.append("seed: " + seed_string)
        if width_string or height_string:
            lines.append(f"size: {width_string}x{height_string}")
        if temperature_string:
            lines.append("temperature: " + temperature_string)
        if positive_string:
            lines.append("positive: " + positive_string)
        if negative_string:
            lines.append("negative: " + negative_string)
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

            ret = folder_paths.get_save_image_path(filename, outdir, i)
            # Known return: (full_output_folder, filename_prefix, counter, subfolder, filename_with_subfolder)
            if isinstance(ret, (list, tuple)) and len(ret) >= 3:
                full_output_folder, filename_prefix, counter = ret[0], ret[1], ret[2]
                path = os.path.join(full_output_folder, f"{filename_prefix}_{counter:05}.png")
            else:
                path = os.path.join(outdir, f"{filename}_{i:05}.png")

            pil.save(path, format="PNG", pnginfo=pnginfo)

            # Align filesystem mtime with our canonical timestamp so all tools agree
            epoch = now_local.timestamp()
            os.utime(path, (epoch, epoch))

            filesize = os.path.getsize(path)
            size_human = f"{filesize / (1024*1024):.2f} MB"

            metadata_raw_out["fileinfo"] = {
                "filename": path,
                "resolution": f"{w}x{h}",
                "date": iso_human,       # unified local, tz-aware
                "size_bytes": filesize,
                "size_human": size_human,
                "size": size_human,
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

NODE_CLASS_MAPPINGS = {"FSLImageSaverWithMetadataV5": FSLImageSaverWithMetadataV5}
NODE_DISPLAY_NAME_MAPPINGS = {"FSLImageSaverWithMetadataV5": "FSL Image Saver w/ Metadata V5"}
