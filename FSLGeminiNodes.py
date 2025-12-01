import os
import time
import json
import base64
import urllib.request
import numpy as np
from PIL import Image
from io import BytesIO
import folder_paths
import torch
from pathlib import Path   # <-- required for path sanitization

# --- NEW UNIFIED SDK IMPORT ---
try:
    from google import genai
    from google.genai import types
except ImportError:
    print("FSL Nodes Error: 'google-genai' library not found.")
    print("Please run: pip install google-genai")
    genai = None

NODE_CATEGORY = "FSL/Gemini"
CHAT_SESSIONS = {}

###############################################################
# FSLGeminiChat
###############################################################

class FSLGeminiChat:
    """
    Interactive Chat Node with Multimodal Support & Agent Logic.
    V7: Split Outputs (Image vs Video) to prevent double-generation.
    """
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}), 
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "Hello, Gemini."}),
                "session_id": ("STRING", {"default": "chat_01"}),
                "model_name": ([
                    "gemini-1.5-flash", 
                    "gemini-1.5-pro", 
                    "gemini-2.0-flash-exp", 
                    "gemini-3-pro-preview"
                ], {"default": "gemini-1.5-flash"}),
                "reset_chat": ("BOOLEAN", {"default": False, "label_on": "Reset Memory", "label_off": "Keep Memory"}),
                "enhance_hook": ("BOOLEAN", {"default": True, "label_on": "Agent Mode (Enhance Prompts)", "label_off": "Raw Mode"}),
                "debug_models": ("BOOLEAN", {"default": False, "label_on": "Debug On", "label_off": "Debug Off"}),
            },
            "optional": { "image_input": ("IMAGE",) }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("LATEST_REPLY", "IMAGE_PROMPT", "VIDEO_PROMPT", "CONVERSATION_HISTORY")
    FUNCTION = "chat_session"
    CATEGORY = NODE_CATEGORY

    def tensor2pil(self, image):
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    def chat_session(self, api_key, prompt, session_id, model_name, reset_chat, enhance_hook, debug_models, image_input=None):
        global CHAT_SESSIONS
        if genai is None:
            return ("Error: google-genai library missing.", "", "", "")

        final_api_key = api_key.strip() or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not final_api_key:
            return ("Error: API Key missing.", "", "", "")

        client = genai.Client(api_key=final_api_key)
        history = []
        if reset_chat:
            if session_id in CHAT_SESSIONS:
                del CHAT_SESSIONS[session_id]
        elif session_id in CHAT_SESSIONS:
            history = CHAT_SESSIONS[session_id]

        base_instruction = "You are a helpful AI assistant in ComfyUI."
        
        if enhance_hook:
            system_instruction = f"""
            {base_instruction}
            You have a dual role: Chatbot and Creative Director.
            
            OUTPUT RULES:
            1. Respond naturally to the user.
            2. Analyze Intent: Does the user want a STATIC IMAGE or a VIDEO/ANIMATION?
            
            IF STATIC IMAGE REQUESTED:
               - Output a hidden block starting with 'IMG_HOOK:'.
               - Content: "8k, masterpiece, photorealistic, [Description]..."
            
            IF VIDEO/ANIMATION REQUESTED:
               - Output a hidden block starting with 'VID_HOOK:'.
               - Content: "Cinematic, [Camera Move], [Physics], [Action], 4k..."
               - Only use this if user asks for video, motion, or animation.
            
            3. IF JUST CHATTING: Output NO hooks.
            """
        else:
            system_instruction = f"{base_instruction} If generating, output 'IMG_HOOK:' for images or 'VID_HOOK:' for video."

        try:
            chat = client.chats.create(
                model=model_name,
                history=history,
                config=types.GenerateContentConfig(system_instruction=system_instruction)
            )
            
            if image_input is not None:
                pil_img = self.tensor2pil(image_input)
                response = chat.send_message([pil_img, prompt])
            else:
                response = chat.send_message(prompt)

            text_output = response.text
            
            history.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
            history.append(types.Content(role="model", parts=[types.Part(text=text_output)]))
            CHAT_SESSIONS[session_id] = history

            image_prompt = ""
            video_prompt = ""
            clean_reply = text_output
            
            if "VID_HOOK:" in text_output:
                parts = text_output.split("VID_HOOK:")
                clean_reply = parts[0].strip()
                video_prompt = parts[1].strip()
                if debug_models: print(f"FSL: Detected VIDEO Hook.")
            elif "IMG_HOOK:" in text_output:
                parts = text_output.split("IMG_HOOK:")
                clean_reply = parts[0].strip()
                image_prompt = parts[1].strip()
                if debug_models: print(f"FSL: Detected IMAGE Hook.")

            formatted_history = f"--- SESSION: {session_id} ---\n"
            for msg in history:
                role = msg.role.upper()
                txt = ""
                for p in msg.parts:
                    if p.text:
                        txt += p.text + " "
                formatted_history += f"[{role}]: {txt.strip()}\n"

            return (clean_reply, image_prompt, video_prompt, formatted_history)

        except Exception as e:
            # FIX: Ensure chat errors are captured but don't flow downstream as a path
            error_message = f"Gemini API Error: {str(e)}"
            print(error_message)
            return (error_message, "", "", "")


###############################################################
# FSLGeminiImageGenerator
###############################################################

class FSLGeminiImageGenerator:
    """
    Google Gemini Image Generator.
    Uses 'generate_content' (Chat API) to create images via Gemini models.
    """
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}), 
                "model_name": ([
                    "gemini-3-pro-image-preview",
                    "gemini-2.0-flash", 
                    "gemini-2.0-flash-exp",
                    "gemini-1.5-pro",
                ], {"default": "gemini-3-pro-image-preview"}),
                "aspect_ratio": (["16:9", "1:1", "9:16", "4:3", "3:4", "3:2", "2:3"], {"default": "16:9"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = NODE_CATEGORY

    def generate_image(self, api_key, prompt, model_name, aspect_ratio):
        if not prompt or prompt.strip() == "":
            print("FSL Gemini Image: No prompt (Chat Mode). Skipping.")
            return (self.create_blank_image(),)

        final_api_key = api_key.strip() or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not final_api_key:
            return (self.create_blank_image(),)

        print(f"FSL Gemini Image: Generating with {model_name}...")
        
        try:
            client = genai.Client(api_key=final_api_key)
            enhanced_prompt = f"{prompt}\n\nGenerate a photorealistic image based on this description. Aspect Ratio: {aspect_ratio}."

            response = client.models.generate_content(
                model=model_name,
                contents=enhanced_prompt
            )

            img_bytes = None
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        img_bytes = part.inline_data.data
                        break
            
            if img_bytes:
                i = Image.open(BytesIO(img_bytes)).convert("RGB")
                image = np.array(i).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                return (image,)
            else:
                print(f"FSL Gemini Image: Model returned text: {response.text[:100]}...")
                return (self.create_blank_image(),)

        except Exception as e:
            print(f"FSL Gemini Image Failed: {e}")
            return (self.create_blank_image(),)

    def create_blank_image(self):
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)



###############################################################
# SAFE PATH SANITIZATION FOR VIDEO OUTPUT
###############################################################

def sanitize_video_save_path(save_location: str, base_output_dir: Path, default_filename: str) -> Path:
    """
    Sanitize a user-supplied save_location for video output.

    - Always keep the result inside base_output_dir.
    - Treat save_location as either:
        * a relative directory, or
        * a relative .mp4 file path under base_output_dir.
    - Strip any absolute roots and '..' segments.
    """
    base_output_dir = base_output_dir.resolve()

    if not save_location or not save_location.strip():
        return (base_output_dir / default_filename).resolve()

    raw = save_location.strip()
    p = Path(raw)

    # Convert absolute → relative
    if p.is_absolute():
        p = Path(*p.parts[1:])

    # Strip unsafe segments
    safe_parts = [part for part in p.parts if part not in ("..", ".", "", "/", "\\")]

    if not safe_parts:
        return (base_output_dir / default_filename).resolve()

    # Case 1: save_location ends with .mp4
    if safe_parts[-1].lower().endswith(".mp4"):
        filename = safe_parts[-1].replace("/", "_").replace("\\", "_")
        dir_parts = safe_parts[:-1]
        safe_dir = Path(*dir_parts) if dir_parts else Path()
        candidate = (base_output_dir / safe_dir / filename).resolve()
    else:
        # Case 2: treat as directory only
        safe_dir = Path(*safe_parts)
        candidate = (base_output_dir / safe_dir / default_filename).resolve()

    # Enforce inside base_output_dir
    try:
        if os.path.commonpath([str(base_output_dir), str(candidate)]) != str(base_output_dir):
            return (base_output_dir / default_filename).resolve()
    except Exception:
        return (base_output_dir / default_filename).resolve()

    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate



###############################################################
# FSLVeoGenerator (HARDENED and RETRY-ENABLED)
###############################################################

class FSLVeoGenerator:
    """
    Google Veo Generator - STRICT SAFE MODE
    Uses Default Configuration ONLY (No custom duration/aspect) to prevent API validation 400 errors.
    """
    def __init__(self):
        self.default_output_dir = folder_paths.get_output_directory()
        self.prefix = "Veo_Vid"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}), 
                "model_name": ([
                    "veo-3.1-generate-preview",
                    "veo-3.0-generate-preview",
                    "veo-2.0-generate-001"
                ], {"default": "veo-3.1-generate-preview"}),
                # Widgets retained for UI but ignored internally
                "duration_seconds": ("INT", {"default": 5, "min": 5, "max": 8, "step": 1}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4"], {"default": "16:9"}),
            },
            "optional": {
                "image_input": ("IMAGE",),
                "save_location": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "placeholder": "Relative path under ComfyUI/output"
                    }
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("VIDEO_PATH",)
    FUNCTION = "generate_video"
    CATEGORY = NODE_CATEGORY
    OUTPUT_NODE = True 

    def tensor2pil(self, image):
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    def generate_video(self, api_key, prompt, model_name, duration_seconds, aspect_ratio, image_input=None, save_location=""):
        if not prompt or prompt.strip() == "":
            print("FSL Veo: No prompt. Skipping.")
            return (None,) # <-- FIX 1: Return None to signal 'no data' to ComfyUI

        final_api_key = api_key.strip() or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not final_api_key:
            print("FSL Veo Error: API Key missing.")
            return (None,) # <-- FIX 2: Return None on API Key error

        print(f"FSL Veo: Requesting {model_name} (Forced Safe Defaults)...")
        
        try:
            # ... (rest of the API request setup) ...

            # ... (inside the polling loop, where errors occur) ...
            
                if retry_count == max_retry_net:
                    print("FSL Veo Error: Failed to connect/poll after multiple network retries.")
                    return (None,) # <-- FIX 3: Return None if network fails persistently

                # --- Process result data ---
                if "done" in data and data["done"]:
                    # ... (download/processing logic) ...

                        if not payload:
                            print("FSL Veo Error: No result payload.")
                            return (None,) # <-- FIX 4: Return None

                        # ... (check for samples/safety block) ...
                        if not samples: 
                            # ... (log error) ...
                            return (None,) # <-- FIX 5: Return None on no samples/safety block

                        # ... (download success/failure) ...
                        except Exception as download_e:
                            print(f"FSL Veo Error downloading video bytes: {download_e}")
                            return (None,) # <-- FIX 6: Return None on download failure

                    break
                else:
                    print(".", end="", flush=True)

            if video_bytes:
                # ... (save path logic) ...
                return (str(final_path),) # Success
            else:
                print("FSL Veo Error: Generation failed/empty bytes (post-polling).")
                return (None,) # <-- FIX 7: Return None on final failure

        except Exception as e:
            print(f"FSL Veo General Error: {e}")
            return (None,) # <-- FIX 8: Return None on general exception



###############################################################
# NODE REGISTRATION
###############################################################

NODE_CLASS_MAPPINGS = {
    "FSLGeminiChat": FSLGeminiChat,
    "FSLVeoGenerator": FSLVeoGenerator,
    "FSLGeminiImageGenerator": FSLGeminiImageGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FSLGeminiChat": "FSL Gemini Chat (Unified SDK)",
    "FSLVeoGenerator": "FSL Google Veo Generator",
    "FSLGeminiImageGenerator": "FSL Gemini Image Generator"
}