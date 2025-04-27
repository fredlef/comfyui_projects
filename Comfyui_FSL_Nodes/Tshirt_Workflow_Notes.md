---
title: T-Shirt and Object Masking Workflow Notes
author: Fred LeFevre
created: 2025-04-27
description: Documentation for the T-Shirt and Object Masking Workflow, including custom node usage, image source control, and metadata settings.
---

# T-Shirt and Object Masking Workflow — Testing & Direction Notes

This workflow was primarily designed to create masked images for T-shirts and other objects. It has grown beyond my initial expectations.

---

## 1. Pending Work
- **Add an upscaler.**
  - I currently use **Topaz Photo AI** for upscaling and it does a great job.
  
---

## 2. Custom Nodes
- There are **three custom nodes** designed for this workflow.
- The **Sampler** and **Scheduler** nodes are included primarily to **feed settings to metadata**.
  - Available fields (except for **Denoise** and **Control After Generate**) should be entered in the **custom nodes**, **not** the **KSampler**.
  - **Denoise** and **Control After Generate** should still be entered directly inside the **KSampler**.

---

## 3. Image Input Options
- There are **four Griptape image nodes** plus a standard **Load Image** node that can feed images into the workflow.
- The image that is processed depends on the **5-Way Toggle Image Switch** inside the **Toggle Image Source Group**.
- **Griptape nodes** require **both OpenAI** and **Black Forest Labs API keys**.
  - These keys **used to** be entered in a `.env` file.
  - They are now entered in the **Griptape Settings window** inside ComfyUI.
  - Instructions for obtaining the necessary API keys can easily be found **online** or in the **Griptape documentation**.

---

## 4. SDXL or Direct Path Group
- The **SDXL or Direct Path Group switch** determines:
  - Whether the image is processed through the full **SDXL workflow**, or
  - Whether it **goes directly** to the **Transparent Background Node**, bypassing SDXL processing.

---

## 5. Prompt vs Image Source Priority
- The **Toggle Prompt or Image Src Group** controls **whether the prompt or the image source takes precedence**.
- This works in conjunction with setting **Denoise** inside the **KSampler**:
  - **Denoise = 0.1** → **Prompt** takes precedence.
  - **Denoise = 0.5** → **Mix** of prompt and image.
  - **Denoise = 0.9** → **Image** takes precedence.

### Choices Table:

| Toggle Prompt or Image Src | SDXL or Direct Path Group | Denoise Setting | Precedence | Notes |
|:---------------------------|:--------------------------|:----------------|:-----------|:------|
| 0 and 0 | (Doesn't matter) | 0.9 | Manual Prompt (Enter Positive Prompt Here) | |
| 0 and 0 | (Doesn't matter) | 0.1 | No Image | If SDXL or Direct Path Group switch is set to 1, image skips SDXL workflow |
| 1 and 0 | (Doesn't matter) | 0.9 | Griptape Prompt | |
| 1 and 0 | (Doesn't matter) | 0.1 | No Image | |
| 0 and 1 | (Doesn't matter) | 0.9 | Manual Prompt | |
| 0 and 1 | (Doesn't matter) | 0.5 | Mix of Manual Prompt and Image Source | |
| 0 and 1 | (Doesn't matter) | 0.1 | Image Source | |
| 1 and 1 | (Doesn't matter) | 0.9 | Manual Prompt | |
| 1 and 1 | (Doesn't matter) | 0.5 | Mix of Manual Prompt and Image Source | |
| 1 and 1 | (Doesn't matter) | 0.1 | Mix of Manual Prompt and Image Source | |

---

# End of Notes
