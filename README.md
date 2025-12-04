# ComfyUI Projects

A collection of custom ComfyUI workflows and nodes created by **Fred LeFevre**.

Clone this repo into your `ComfyUI/custom_nodes` folder.

## 🚀 Update Notes

- **📌 Version 1.1.1 – Update Notes

This release includes a maintenance update to keep the FSLGeminiGenerateImage node compatible with current Google Gemini model availability.
Changes in 1.1.1
Removed two deprecated Google image-generation models:

gemini-2.5-flash-image-preview
gemini-2.0-flash-exp

Google has scheduled these models for discontinuation, so they have been removed from the selectable model list to prevent API errors and improve workflow stability.
No other functionality has changed in this version.

- **New: FSL Gemini & Veo Suite**
  Added a full-stack cloud media suite (`FSLGeminiNodes.py`) powered by the new Google Unified SDK. Features include:
  - **Gemini Chat Agent:** A "Creative Director" that handles conversation and writes technical prompts.
  - **Veo Video Generator:** Supports Text-to-Video and True Image-to-Video (Identity Preservation).
  - **Gemini Image Generator:** A lightweight, high-speed image generator using `generate_content`.
  - **Gatekeeper Logic:** Nodes automatically "sleep" (costing $0) when you are just chatting with the agent.

- **Node Updates**
  - **Gemini Generate Image Node:** Consolidated versions. The standalone `FSLGeminiGenerateImage.py` is the maintained advanced version.
  - **Workflow Updates:** All Nano Banana workflows have been updated to use the latest nodes.

---

## 🧩 Custom Nodes: Comfyui_FSL_Nodes

Located in the [`Comfyui_FSL_Nodes/`](https://github.com/fredlef/comfyui_projects/tree/main/custom_nodes/Comfyui_FSL_Nodes) directory.

### 🤖 Gemini & Veo Cloud Suite (File: `FSLGeminiNodes.py`)
*   **FSLGeminiChat (Unified SDK):** The "Brain." Handles interactive chat, vision analysis, and prompt engineering. It intelligently routes requests to `IMAGE_PROMPT` or `VIDEO_PROMPT` to prevent double generation.
*   **FSLVeoGenerator:** The "Animator." Generates high-fidelity video using Google Veo (v3.1). Supports **Image-to-Video** (animating specific pixels) and **Text-to-Video**, with robust HTTP polling to handle large file downloads.
*   **FSLGeminiImageGenerator:** The "Artist" (Pro). A lightweight, native node using Gemini models (e.g., `gemini-2.0-flash`) for image generation via the Chat endpoint. Features "Smart Gatekeeping" to return a blank image instantly if the prompt is empty.

### 🛠️ Utility & Legacy Nodes
*   **FSLGeminiGenerateImage.py:** This is the current standalone advanced image generator. Supports Inpainting, Masking, and Init Images for complex editing workflows.
*   **FSLGeminiGenerateImageV8.py:** *[Legacy]* Previous version.
*   **manual_alpha_mask_painter.py:** Manually convert black mask areas into alpha transparency. Essential for GRIPTAPE inpainting (which requires alpha channel masks).
*   **8WayImgSwitch:** An image switcher with eight inputs.
*   **fsl_image_memory.py:** A set of 4 nodes (Store, Recall, Clear, Clear All) to save images into memory using a specific 'key' for complex workflow routing.
*   **fsl_prompt_compose.py:** Handles Positive/Negative prompts and "Scene Lock." When `lock_scene` is True, it injects instructions to freeze composition, lighting, and subjects while only changing specific details.
*   **fsl_ensure_nhwc_batch.py:** Guarantees incoming image tensors are converted to `[N, H, W, C]` layout, ensuring compatibility across different node packs.
*   **FSLImageSaverWithMetadata.py:** The current metadata saver. Saves images with embedded metadata, readable via the `LoadImage-w-Metadata` workflow. Supports custom subfolders.
*   **FSLImageSaverWithMetadataV5.py:** *[Legacy]* De-versioned in the newest update.
*   **fsl_composite_with_mask_cropped.py:** *[Legacy]* Removes alpha channel but keeps transparent parts as background color.
*   **fsl_save_and_strip_alpha:** *[Legacy]* Strips alpha channel from RGBA images.

---

## 🎥 Workflow: Gemini Chat & Veo Video
*   **Gemini_Chat.json**
    *   **Concept:** A "Creative Director" workflow. You chat with Gemini, and it intelligently decides whether to reply with text or generate media.
    *   **Text-to-Video:** Ask for a video, and Gemini writes the prompt for Veo.
    *   **Image-to-Video:** Upload an image and connect it to both Chat and Veo nodes. Gemini instructs Veo to "Preserve Identity," allowing you to animate specific characters or photos.
    *   **Requirements:** Google API Key (Billing enabled for Veo).
    *   **Feature:** Note the field **"enhance_hook"**. When enabled, this automatically uses Gemini to enhance the raw prompt before sending it to the generator.

---

## 🚦 Veo Logic & Modes (Critical)

The **Veo Generator** node operates in two distinct modes to satisfy API requirements:

### Mode A: Image-to-Video (Identity Preservation)
*   **Trigger:** A wire is connected to `image_input`.
*   **Behavior:** Veo animates the pixels provided.
*   **Constraint:** **Duration and Aspect Ratio sliders are IGNORED.** The Veo 3.1 API currently rejects custom configs when an image is present. The node automatically uses the model's defaults to prevent crashes.

### Mode B: Text-to-Video (Generation)
*   **Trigger:** Nothing is connected to `image_input`.
*   **Behavior:** Veo generates a video from scratch based on words.
*   **Capabilities:** Full control over `duration_seconds` (5s - 8s) and `aspect_ratio`.

---

## 🍌 Workflow: Nano Banana (API Required)
*   **Nano Banana Iterative-Base.json**
    Create images using `gemini-2.5-flash-image-preview`. Supports iterative generation where each succeeding image is based on the previous one.
*   **Nano Banana Iterative-Load 5 Images.json**
    Uses up to 5 input images to create a new generation. Includes instructions in Note nodes.
*   **Nano Banana w_Griptape w_Upscaler.json**
    Uses Griptape Nodes as a front-end prompt loader. Includes an excellent upscaler by Alex (ComfyUiStudio). *Requires an OpenAI API Key for Griptape.*
*   **Nano Banana img2img w_Griptape.json**
    Griptape-assisted Image-to-Image workflow.
*   **Nano Banana img2img Base.json**
    Standard Image-to-Image workflow using Nano Banana.

**Clarification: Init Image vs Image (Legacy Node)**
*   **Iterative Workflow:** Set `Init-Image` socket to **True** (and `Images` to False) after the first generation to loop the result back in.
*   **Standard Workflow:** Always set `Images` socket to **True** and `Init-Image` socket to **False**.

---

## 🎨 Workflow: T-shirt Designer
*   **Tshirt-Designer-ver_2.0.json**
    *   *Note:* A less complex v3.0 replacement is pending.
    *   Generates masked transparent images optimized for T-shirt and object printing.
    *   Full workflow details available in the [`tshirt_designer/`](https://github.com/fredlef/comfyui_projects/tree/main/workflows/tshirt_designer) directory.

---

## 📂 Miscellaneous Workflows
*   **LoadImage-w-Metadata.json**
    A simple utility workflow to read and display the metadata stored by the `FSLImageSaverWithMetadata` node.

---

## 📜 Changelog
- ##1.1.1 — Maintenance Update

Removed deprecated Gemini models
(gemini-2.5-flash-image-preview, gemini-2.0-flash-exp)
from FSLGeminiGenerateImage to maintain compatibility with the latest Google API.

## Acknowledgements
- **Alex (ComfyUiStudio):** For the excellent upscaler included in the Nano Banana workflows.
- **Google DeepMind:** For the powerful Gemini, Veo, and Imagen models driving these nodes.
- **Mycroft Holmes (ChatGPT) and Gemini 3:** For assistance and guidance in the creation of Custom Nodes.

---

## Author

Created and maintained by **Fred LeFevre**.