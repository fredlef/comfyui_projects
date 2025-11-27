import os
import time
import json
import base64
import urllib.request
import numpy as np
from PIL import Image
from io import BytesIO
import torch

# ComfyUI Imports
import folder_paths

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

class FSLGeminiChat:
    """
    Interactive Chat Node with Multimodal Support & Agent Logic.
    """
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}), 
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": "Hello, Gemini."}),
                "session_id": ("STRING", {"default": "chat_01"}),
                "model_name": (["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-3-pro-preview"], {"default": "gemini-1.5-flash"}),
                "reset_chat": ("BOOLEAN", {"default": False, "label_on": "Reset Memory", "label_off": "Keep Memory"}),
                "enhance_hook": ("BOOLEAN", {"default": True, "label_on": "Agent Mode (Enhance Prompts)", "label_off": "Raw Mode"}),
                "debug_models": ("BOOLEAN", {"default": False, "label_on": "Debug On", "label_off": "Debug Off"}),
            },
            "optional": { "image_input": ("IMAGE",) }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("LATEST_REPLY", "MEDIA_PROMPT_HOOK", "CONVERSATION_HISTORY")
    FUNCTION = "chat_session"
    CATEGORY = NODE_CATEGORY

    def tensor2pil(self, image):
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

    def chat_session(self, api_key, prompt, session_id, model_name, reset_chat, enhance_hook, debug_models, image_input=None):
        global CHAT_SESSIONS
        if genai is None: return ("Error: google-genai library missing.", "", "")

        final_api_key = api_key.strip() or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not final_api_key: return ("Error: API Key missing.", "", "")

        client = genai.Client(api_key=final_api_key)
        history = []
        if reset_chat:
            if session_id in CHAT_SESSIONS: del CHAT_SESSIONS[session_id]
        elif session_id in CHAT_SESSIONS:
            history = CHAT_SESSIONS[session_id]

        base_instruction = "You are a helpful AI assistant in ComfyUI."
        if enhance_hook:
            # IDENTITY PRESERVATION LOGIC
            has_image = (image_input is not None)
            if has_image:
                system_instruction = f"""
                {base_instruction}
                You have a dual role: Chatbot and Creative Director.
                CRITICAL: The user provided an Input Image. 
                If asked to animate/generate based on it:
                1. Output a hidden block starting with 'HOOK:'.
                2. Do NOT describe physical features (eyes, hair) in detail.
                3. Instead write: "HOOK: Animate the exact subject in the provided image, high fidelity, preserve identity, [Action], [Camera], 4k."
                """
            else:
                system_instruction = f"""
                {base_instruction}
                Role: Creative Director.
                RULES:
                1. Respond normally to text.
                2. If user asks for Visual Generation, output 'HOOK:'.
                3. HOOK Content: Raw, optimized prompt.
                4. If just chatting, DO NOT output a HOOK.
                """
        else:
            system_instruction = f"{base_instruction} If user asks for generation, output block starting 'HOOK:'."

        try:
            chat = client.chats.create(model=model_name, history=history, config=types.GenerateContentConfig(system_instruction=system_instruction))
            
            if image_input is not None:
                pil_img = self.tensor2pil(image_input)
                response = chat.send_message([pil_img, prompt])
            else:
                response = chat.send_message(prompt)

            text_output = response.text
            history.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
            history.append(types.Content(role="model", parts=[types.Part(text=text_output)]))
            CHAT_SESSIONS[session_id] = history

            media_prompt_hook = ""
            clean_reply = text_output
            
            if "HOOK:" in text_output:
                parts = text_output.split("HOOK:")
                clean_reply = parts[0].strip()
                media_prompt_hook = parts[1].strip()
                if debug_models: print(f"FSL Hook: {media_prompt_hook}")

            formatted_history = f"--- SESSION: {session_id} ---\n"
            for msg in history:
                role = msg.role.upper()
                txt = ""
                for p in msg.parts:
                    if p.text: txt += p.text + " "
                formatted_history += f"[{role}]: {txt.strip()}\n"

            return (clean_reply, media_prompt_hook, formatted_history)

        except Exception as e: return (f"Gemini API Error: {str(e)}", "", "")


class FSLImagenGenerator:
    """
    Google Imagen 3 Generator (Pro Version).
    Includes: Aspect Ratio, Safety Toggles, Seed Control, and Smart Gatekeeper.
    """
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}), # Connect Hook Here
                "model_name": (["imagen-3.0-generate-001", "imagen-3.0-fast-generate-001"], {"default": "imagen-3.0-generate-001"}),
                "aspect_ratio": (["16:9", "1:1", "9:16", "4:3", "3:4", "3:2", "2:3"], {"default": "16:9"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "disable_safety": ("BOOLEAN", {"default": True, "label_on": "Disable Safety Filters", "label_off": "Use Default Safety"}),
                "allow_people": ("BOOLEAN", {"default": True, "label_on": "Allow People Generation", "label_off": "Block People"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = NODE_CATEGORY

    def generate_image(self, api_key, prompt, model_name, aspect_ratio, seed, disable_safety, allow_people):
        # 1. SMART GATEKEEPER
        if not prompt or prompt.strip() == "":
            print("FSL Imagen: No prompt (Chat Mode). Skipping generation.")
            return (self.create_blank_image(),)

        final_api_key = api_key.strip() or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not final_api_key: return (self.create_blank_image(),)

        print(f"FSL Imagen: Generating '{prompt[:20]}...' (Seed: {seed})")
        
        try:
            client = genai.Client(api_key=final_api_key)
            
            # Configure Safety
            safety_config = []
            if disable_safety:
                # Imagen 3 specific safety bypass attempts (Note: Google overrides this for extreme content)
                safety_config = [
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                ]

            # Generate
            response = client.models.generate_images(
                model=model_name,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio=aspect_ratio,
                    include_rai_reason=True,
                    person_generation="allow_adult" if allow_people else "block_adult",
                    safety_settings=safety_config if disable_safety else None,
                    seed=seed
                )
            )

            if response.generated_images:
                img_bytes = response.generated_images[0].image.image_bytes
                
                # Convert to ComfyUI Tensor
                i = Image.open(BytesIO(img_bytes))
                i = i.convert("RGB")
                image = np.array(i).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                return (image,)
            else:
                print("FSL Imagen: No image returned. Check Safety/Quota.")
                return (self.create_blank_image(),)

        except Exception as e:
            print(f"FSL Imagen Failed: {e}")
            return (self.create_blank_image(),)

    def create_blank_image(self):
        # Return tiny black square
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


class FSLVeoGenerator:
    """
    Google Veo Generator - "No-Config" Fix for Img2Vid
    Removes ALL config parameters for Image-to-Video to bypass API validation bugs.
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
                "model_name": (["veo-3.1-generate-preview", "veo-3.0-generate-preview", "veo-2.0-generate-001"], {"default": "veo-3.1-generate-preview"}),
                "duration_seconds": ("INT", {"default": 5, "min": 5, "max": 8, "step": 1}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4"], {"default": "16:9"}),
            },
            "optional": {
                "image_input": ("IMAGE",),
                "save_location": ("STRING", {"multiline": False, "default": "", "placeholder": "Leave empty for default output..."}),
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
            return ("",)

        final_api_key = api_key.strip() or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not final_api_key: return ("Error: API Key missing.",)

        # Force safe duration logic
        safe_duration = int(max(5, min(8, duration_seconds)))
        
        print(f"FSL Veo: Requesting {model_name}...")
        
        try:
            client = genai.Client(api_key=final_api_key)
            text_prompt = prompt 
            
            # --- 1. HANDLE IMAGE INPUT ---
            img_arg = None
            if image_input is not None:
                print("FSL Veo: Image Input Detected (Image-to-Video Mode).")
                pil_img = self.tensor2pil(image_input)
                
                buffered = BytesIO()
                pil_img.save(buffered, format="PNG")
                img_bytes = buffered.getvalue()
                
                img_arg = types.Image(
                    image_bytes=img_bytes, 
                    mime_type='image/png'
                )

            # --- 2. SUBMIT JOB ---
            if img_arg:
                 # IMAGE-TO-VIDEO MODE
                 # CRITICAL FIX: We remove the 'config' parameter entirely.
                 # Current API seems to reject manual duration/aspect_ratio when an image is present.
                 print(f"FSL Veo: Submitting Img2Vid (Using Model Defaults to avoid errors)")
                 operation = client.models.generate_videos(
                    model=model_name,
                    prompt=text_prompt, 
                    image=img_arg
                    # NO config=... passed here.
                )
            else:
                 # TEXT-TO-VIDEO MODE
                 # We use full config
                 print(f"FSL Veo: Submitting Text2Vid ({safe_duration}s | {aspect_ratio})")
                 operation = client.models.generate_videos(
                    model=model_name,
                    prompt=text_prompt, 
                    config=types.GenerateVideosConfig(
                        number_of_videos=1,
                        duration_seconds=safe_duration,
                        aspect_ratio=aspect_ratio
                    )
                )
            
            op_name = operation.name if hasattr(operation, "name") else str(operation)
            print(f"FSL Veo: Job Submitted. ID: {op_name}")
            print("FSL Veo: Polling via HTTP...")

            # --- 3. POLL STATUS ---
            base_url = "https://generativelanguage.googleapis.com/v1beta/"
            poll_url = f"{base_url}{op_name}?key={final_api_key}"
            video_bytes = None
            
            while True:
                time.sleep(10)
                req = urllib.request.Request(poll_url)
                with urllib.request.urlopen(req) as response:
                    data = json.loads(response.read().decode())
                
                if "done" in data and data["done"]:
                    print("\nFSL Veo: Cloud processing finished. Checking result...")
                    try:
                        payload = data.get("response") or data.get("result")
                        
                        if not payload:
                            if "error" in data:
                                return (f"Error: {data['error'].get('message', 'Unknown Error')}",)
                            return ("Error: Generation blocked. No result payload.",)

                        samples = payload.get("generateVideoResponse", {}).get("generatedSamples", []) or payload.get("generatedVideos", []) or payload.get("generatedSamples", [])
                        
                        if not samples:
                            rai = payload.get("generateVideoResponse", {})
                            if rai.get("raiMediaFilteredCount", 0) > 0:
                                return ("Error: Video BLOCKED by Safety Filters.",)
                            return (f"Error: No samples found. Data: {str(payload)[:200]}",)

                        video_obj = samples[0]["video"]

                        if "uri" in video_obj:
                            dl_uri = video_obj["uri"] + (f"&key={final_api_key}" if "?alt=media" in video_obj["uri"] else f"?key={final_api_key}")
                            with urllib.request.urlopen(urllib.request.Request(dl_uri, headers={'User-Agent': 'Mozilla/5.0'})) as v_resp:
                                video_bytes = v_resp.read()
                        elif "videoBytes" in video_obj:
                            video_bytes = base64.b64decode(video_obj["videoBytes"])
                        break
                    except Exception as e: return (f"Error downloading: {e}",)
                else: print(".", end="", flush=True)

            # --- 4. SAVE FILE ---
            if video_bytes:
                filename = f"{self.prefix}_{int(time.time())}.mp4"
                save_dir = self.default_output_dir
                final_path = os.path.join(save_dir, filename)

                if save_location and save_location.strip():
                    custom_path = save_location.strip()
                    if custom_path.lower().endswith(".mp4"):
                        final_path = custom_path
                        if not os.path.exists(os.path.dirname(final_path)): os.makedirs(os.path.dirname(final_path), exist_ok=True)
                    else:
                        if not os.path.exists(custom_path): os.makedirs(custom_path, exist_ok=True)
                        final_path = os.path.join(custom_path, filename)

                with open(final_path, "wb") as f: f.write(video_bytes)
                print(f"FSL Veo SAVED: {final_path}")
                return (final_path,)
            else:
                return ("Error: Generation failed.",)

        except Exception as e:
            print(f"FSL Veo Failed: {e}")
            return (f"Error: {e}",)

NODE_CLASS_MAPPINGS = {
    "FSLGeminiChat": FSLGeminiChat,
    "FSLVeoGenerator": FSLVeoGenerator,
    "FSLImagenGenerator": FSLImagenGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FSLGeminiChat": "FSL Gemini Chat (Unified SDK)",
    "FSLVeoGenerator": "FSL Google Veo Generator",
    "FSLImagenGenerator": "FSL Google Imagen 3 Generator (Pro)"
}