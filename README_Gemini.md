# FSL Gemini & Veo Workflow for ComfyUI

**Version:** 6.5 (The "Production" Edition)
**Powered By:** Google DeepMind (Gemini 1.5/2.0/3.0, Veo 3.1, Imagen 3)

This node suite creates a full-stack, cloud-powered AI media studio inside ComfyUI. It offloads heavy processing to Google Cloud, allowing for high-fidelity **Text-to-Video**, **Image-to-Video**, and **Agent-Assisted Creation** without requiring massive local VRAM.

---

## ✨ Core Features

1.  **The "Creative Director" Agent:** A memory-enabled Chat Node that acts as a middleman. It intelligently detects if you want to chat or create. If you want to create, it writes professional "Hidden Hooks" to control the generators.
2.  **True Image-to-Video (Identity Preservation):** Direct integration with **Veo 3.1** allows you to send local pixels to the cloud. Veo animates *your* specific image rather than hallucinating a new one from text.
3.  **Smart Gatekeeping ($0 Cost Mode):** The Image and Video nodes feature "Gatekeeper Logic." If you are just chatting with the Agent (e.g., "How are you?"), the generators receive empty prompts and "sleep" instantly, incurring **zero cost**.  If you tell it to create an image the video output will be ignored (no need to bypass).  If you tell it to generate a video the FSLGeminiGenerateImage node will still create an image but the FSL Google Imagen 3 Generator Pro will not.  Only one of these nodes is necssary for the workflow.  Both are provided by way of example.
4.  **Robust Cloud Polling:** Uses a custom HTTP polling engine to bypass Python SDK timeouts, ensuring large video files download successfully even if generation takes minutes.
5.  **Hybrid Logic:** Automatically switches Veo configuration based on input type (Text vs. Image) to prevent API validation errors.

---

## 🛠️ Installation

### 1. Requirements
These nodes rely on Google's modern Unified SDK (`google-genai`).

**For ComfyUI Portable (Windows):**
Open a terminal in your `ComfyUI_windows_portable` folder and run:
```bash
.\python_embeded\python.exe -m pip install --upgrade google-genai requests
```

**For Standard Python:**
```bash
pip install --upgrade google-genai requests
```

### 2. API Key & Billing
*   **Gemini/Imagen:** Works with standard Google AI Studio keys.
*   **Veo (Video):** Requires a Google Cloud Project with **Billing Enabled**. Free-tier keys usually return 403/404 errors for video.

**Setup:**
*   **Option A (Best):** Set an Environment Variable `GEMINI_API_KEY` or `GOOGLE_API_KEY` in your startup batch file.
*   **Option B:** Paste the key directly into the `api_key` widget on every node.

---

## 📦 The Node Suite

### 1. FSL Gemini Chat (Unified SDK)
*   **Role:** The Director / Logic Engine.
*   **Inputs:** Text Prompt + Optional Image (Vision).
*   **Agent Mode:**
    *   **Text-to-Media:** If you ask for a video, it writes a detailed prompt (Lighting, Camera Move, Physics) into the `MEDIA_PROMPT_HOOK`.
    *   **Image-to-Media:** If an image is connected, it writes a "Fidelity Prompt" instructing the generator to *preserve the identity* of the input image.
*   **Outputs:** `LATEST_REPLY` (Chat), `MEDIA_PROMPT_HOOK` (Technical Prompt), `CONVERSATION_HISTORY` (Log).

### 2. FSL Google Veo Generator
*   **Role:** The Animator.
*   **Inputs:** Text Prompt (from Chat Hook) + Optional Image.
*   **Feature:** **Save Location**.
    *   *Note:* Paths are relative to the **ComfyUI Root Folder**, not the Output folder.
    *   Example: `my_videos` saves to `ComfyUI/my_videos/`.
    *   Example: `output/my_videos` saves to `ComfyUI/output/my_videos/`.
    *   Example: `C:/Videos/` saves to an absolute path.

### 3. FSL Imagen Generator (Pro)
*   **Role:** The Artist (Fast).
*   **Engine:** Google Imagen 3.
*   **Gatekeeper:** Returns a tiny black square instantly if prompt is empty.

### 4. FSL Gemini Generate Image (Legacy)
*   **Role:** The Editor (Advanced).
*   **Usage:** Keep this node if you need **Inpainting**, **Masking**, or **Image-to-Image** editing features not present in the lightweight Pro node.
*   *Tip:* If you encounter timeouts with this node, edit the `.py` file and increase `timeout=60` to `timeout=180`.

---

## 🚦 Veo Logic & Modes (Critical)

The **Veo Generator** node has two distinct modes of operation depending on what you connect to it.

### Mode A: Image-to-Video (Identity Preservation)
*   **Trigger:** A wire is connected to `image_input`.
*   **Behavior:** Veo animates the pixels provided.
*   **Constraint:** **Duration and Aspect Ratio sliders are IGNORED.**
    *   *Why?* The Veo 3.1 API currently crashes if you try to force a custom duration or aspect ratio onto an existing image. The node automatically strips these settings to prevent errors.
    *   *Result:* You will get the default Veo output (usually ~5 seconds, Native Aspect Ratio).

### Mode B: Text-to-Video (Generation)
*   **Trigger:** Nothing is connected to `image_input`.
*   **Behavior:** Veo generates a video from scratch based on words.
*   **Capabilities:** Full control over `duration_seconds` (5s - 8s) and `aspect_ratio`.

### 🔄 How to Switch Modes
If you want to create a video from scratch (Text-to-Video), you **must disconnect** the image wire.
*   **Method:** Drag the wire off the `image_input` pin.
*   If you leave the image connected but type "Create a video of a spaceship", Veo will get confused and try to turn your input image *into* a spaceship (or crash).

---

## 🔌 Wiring Strategies

### Strategy A: The "Creative Director" (Text-to-Video)
1.  Connect `FSL Gemini Chat` (Hook) -> `FSL Veo Generator` (Prompt).
2.  Ensure `image_input` on Veo is **Disconnected**.
3.  **Prompt:** "Generate a cinematic video of a cyberpunk city."

### Strategy B: "Identity Preservation" (Image-to-Video)
1.  **The Y-Split:** Connect your **Load Image** node to **TWO** inputs:
    *   `FSL Gemini Chat` -> `image_input` (So the Agent sees context).
    *   `FSL Veo Generator` -> `image_input` (So Veo gets the raw pixels).
2.  **Prompt:** "Animate this person turning their head."
3.  **Result:** The Agent tells Veo to "preserve fidelity," and Veo uses the direct pixel connection to animate your exact image.

---

## ❓ Troubleshooting

**Q: Veo Error: "The number value for durationSeconds is out of bound"**
*   **A:** This happens if you are in **Image-to-Video** mode but older code was trying to force a duration. Ensure you are using **v6.5+** of the nodes, which automatically disables config parameters when an image is detected.

**Q: Load Video Path Error: "video is not a valid path..."**
*   **A:** This means the Veo node failed to generate the MP4, so it passed an Error Message string instead of a file path. Check the ComfyUI console/terminal to see the real error (usually API Key or Safety Filter related).

**Q: "RAI Media Filtered" / Blocked by Safety**
*   **A:** Google blocked the generation. Veo is strict about photorealistic people. Try describing the *action* rather than the *person*, or try the "Minimize Safety" toggle (though Veo 3.1 is very strict regardless).

---

## 📝 Credits
*   **Developer:** FSL (Fred's Super Logic)
*   **Engine:** Google DeepMind (Gemini / Veo / Imagen)