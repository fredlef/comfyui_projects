# ComfyUI/custom_nodes/comfyui_fsl_nodes/fsl_image_memory.py
# FSL / Image Memory: store & recall IMAGE tensors between runs.

import torch, time

# One in-process cache for IMAGEs: key -> list[HWC float tensors]
_MEM = {}

def _ensure_list_of_hwctensors(x):
    """Return a flat list of HWC float32 tensors in [0,1]."""
    imgs = x if isinstance(x, list) else [x]
    # unwrap single nested list: [[t]] -> [t]
    if len(imgs) == 1 and isinstance(imgs[0], list):
        imgs = imgs[0]

    out = []
    for t in imgs:
        # normalize to torch tensor
        if not torch.is_tensor(t):
            t = torch.tensor(t)

        # common incoming shapes:
        # (H,W,C) OK; (C,H,W) -> (H,W,C); (1,C,H,W) -> (C,H,W)
        if t.ndim == 4 and t.shape[0] == 1:
            t = t.squeeze(0)  # (1,C,H,W) -> (C,H,W)
        if t.ndim == 3 and t.shape[0] in (1, 3) and t.shape[-1] not in (1, 3):
            # likely CHW -> HWC
            t = t.movedim(0, -1)
        if t.ndim == 2:
            # grayscale H,W -> H,W,1
            t = t.unsqueeze(-1)

        # dtype/range hygiene
        if t.dtype != torch.float32:
            t = t.float()
        t = t.clamp(0, 1).contiguous()
        out.append(t)
    return out

def _first_shape(lst):
    try:
        t = lst[0]
        return tuple(t.shape)
    except Exception:
        return "n/a"

class FSLImageMemoryStore:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "key": ("STRING", {"default": "last"})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "store"
    CATEGORY = "FSL/Memory"

    def store(self, image, key):
        data = _ensure_list_of_hwctensors(image)
        _MEM[key] = data
        print(f"[FSLImageMemoryStore] stored key='{key}' ({len(data)} img) first={_first_shape(data)} mem_id={id(_MEM)}")
        return (data,)  # pass-through as IMAGE

class FSLImageMemoryRecallSafe:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"key": ("STRING", {"default": "last"})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "recall"
    CATEGORY = "FSL/Memory"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    def recall(self, key):
        imgs = _MEM.get(key, None)
        if imgs:
            imgs = _ensure_list_of_hwctensors(imgs)
            print(f"[FSLImageMemoryRecallSafe] key='{key}' ({len(imgs)} img) first={_first_shape(imgs)} mem_id={id(_MEM)}")
            return (imgs,)
        print(f"[FSLImageMemoryRecallSafe] WARNING: key='{key}' missing -> returning 1x1 black mem_id={id(_MEM)}")
        t = torch.zeros((1, 1, 3), dtype=torch.float32)  # HWC in [0,1]
        return ([t],)

class FSLImageMemoryClear:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"key": ("STRING", {"default": "last"})}}
    
    RETURN_TYPES = ()
    FUNCTION = "clear"
    OUTPUT_NODE = True
    CATEGORY = "FSL/Memory"

    def clear(self, key):
        _MEM.pop(key, None)
        print(f"[FSLImageMemoryClear] cleared key='{key}' mem_id={id(_MEM)}")
        return ()

class FSLImageMemoryClearAll:
    @classmethod
    def INPUT_TYPES(cls):
        return {}
    
    RETURN_TYPES = ()
    FUNCTION = "clear_all"
    OUTPUT_NODE = True
    CATEGORY = "FSL/Memory"

    def clear_all(self):
        global _MEM
        _MEM.clear()
        print(f"[FSLImageMemoryClearAll] cleared all memory, mem_id={id(_MEM)}")
        return ()

NODE_CLASS_MAPPINGS = {
    "FSLImageMemoryStore": FSLImageMemoryStore,
    "FSLImageMemoryRecallSafe": FSLImageMemoryRecallSafe,
    "FSLImageMemoryClear": FSLImageMemoryClear,
    "FSLImageMemoryClearAll": FSLImageMemoryClearAll,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FSLImageMemoryStore": "FSL / Image Memory → Store",
    "FSLImageMemoryRecallSafe": "FSL / Image Memory → Recall (Safe)",
    "FSLImageMemoryClear": "FSL / Image Memory → Clear",
    "FSLImageMemoryClearAll": "FSL / Image Memory → Clear All",
}