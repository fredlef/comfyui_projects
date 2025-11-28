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
        if genai is None: return ("Error: google-genai library missing.", "", "", "")

        final_api_key = api_key.strip() or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not final_api_key: return ("Error: API Key missing.", "", "", "")

        client = genai.Client(api_key=final_api_key)
        history = []
        if reset_chat:
            if session_id in CHAT_SESSIONS: del CHAT_SESSIONS[session_id]
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
                    if p.text: txt += p.text + " "
                formatted_history += f"[{role}]: {txt.strip()}\n"

            return (clean_reply, image_prompt, video_prompt, formatted_history)

        except Exception as e: return (f"Gemini API Error: {str(e)}", "", "", "")


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
                # Gemini 3 Preview set as default per request
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
        if not final_api_key: return (self.create_blank_image(),)

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
                "model_name": (["veo-3.1-generate-preview", "veo-3.0-generate-preview", "veo-2.0-generate-001"], {"default": "veo-3.1-generate-preview"}),
                # Widgets retained for UI, but IGNORED internally for stability
                "duration_seconds": ("INT", {"default": 5, "min": 5, "max": 8, "step": 1}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4"], {"default": "16:9"}),
            },
            "optional": {
                "image_input": ("IMAGE",),
                "save_location": ("STRING", {"multiline": False, "default": "", "placeholder": "Default: ComfyUI/output folder"}),
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

        print(f"FSL Veo: Requesting {model_name} (Forced Safe Defaults)...")
        
        try:
            client = genai.Client(api_key=final_api_key)
            text_prompt = prompt 
            
            img_arg = None
            if image_input is not None:
                print("FSL Veo: Image Input Detected.")
                pil_img = self.tensor2pil(image_input)
                buffered = BytesIO()
                pil_img.save(buffered, format="PNG")
                img_bytes = buffered.getvalue()
                img_arg = types.Image(image_bytes=img_bytes, mime_type='image/png')

            # --- SUBMISSION (SAFE MODE) ---
            # We strip ALL config parameters except 'number_of_videos'. 
            # This forces the API to use its internal defaults, bypassing the validation 400 errors.
            
            if img_arg:
                 print(f"FSL Veo: Submitting Img2Vid (Defaults)")
                 operation = client.models.generate_videos(
                    model=model_name, prompt=text_prompt, image=img_arg, 
                    config=types.GenerateVideosConfig(number_of_videos=1)
                )
            else:
                 print(f"FSL Veo: Submitting Text2Vid (Defaults)")
                 operation = client.models.generate_videos(
                    model=model_name, prompt=text_prompt, 
                    config=types.GenerateVideosConfig(number_of_videos=1)
                )
            
            op_name = operation.name if hasattr(operation, "name") else str(operation)
            print(f"FSL Veo: Job {op_name}. Polling...")

            base_url = "https://generativelanguage.googleapis.com/v1beta/"
            poll_url = f"{base_url}{op_name}?key={final_api_key}"
            video_bytes = None
            
            while True:
                time.sleep(10)
                req = urllib.request.Request(poll_url)
                with urllib.request.urlopen(req) as response:
                    data = json.loads(response.read().decode())
                
                if "done" in data and data["done"]:
                    print("\nFSL Veo: Finished. Downloading...")
                    try:
                        payload = data.get("response") or data.get("result")
                        if not payload: return ("Error: No result payload.",)

                        samples = payload.get("generateVideoResponse", {}).get("generatedSamples", []) or payload.get("generatedVideos", []) or payload.get("generatedSamples", [])
                        if not samples: 
                            rai = payload.get("generateVideoResponse", {})
                            if rai.get("raiMediaFilteredCount", 0) > 0: return ("Error: Video BLOCKED by Safety Filters.",)
                            return (f"Error: No samples found. {str(payload)[:200]}",)

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

            if video_bytes:
                filename = f"{self.prefix}_{int(time.time())}.mp4"
                save_dir = self.default_output_dir
                
                # Default path
                final_path = os.path.join(save_dir, filename)

                if save_location and save_location.strip():
                    custom_path = save_location.strip()
                    # If path is relative, make it relative to output/
                    if not os.path.isabs(custom_path):
                        custom_path = os.path.join(save_dir, custom_path)

                    if custom_path.lower().endswith(".mp4"):
                        final_path = custom_path
                        if not os.path.exists(os.path.dirname(final_path)): 
                            try: os.makedirs(os.path.dirname(final_path), exist_ok=True)
                            except: final_path = os.path.join(save_dir, filename)
                    else:
                        if not os.path.exists(custom_path): 
                            try: os.makedirs(custom_path, exist_ok=True)
                            except: custom_path = save_dir
                        final_path = os.path.join(custom_path, filename)

                with open(final_path, "wb") as f: f.write(video_bytes)
                print(f"FSL Veo SAVED: {final_path}")
                return (final_path,)
            else: return ("Error: Generation failed.",)

        except Exception as e: return (f"Error: {e}",)

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