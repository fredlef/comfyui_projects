# Tshirt Designer Workflow v1.0 for ComfyUI

## Overview
This project provides a specialized **ComfyUI** workflow designed to create **masked images** for T-shirts, apparel, and other printable products.  
It features custom nodes, flexible image/prompt control, Griptape API integration, and metadata optimization.

Originally developed for internal use, the workflow has grown into a versatile system for high-quality, transparent-background image creation.

---

## Features
- **Custom Sampler and Scheduler nodes** to properly embed settings into image metadata.
- **5-way image toggle** system: Choose between uploaded images, Griptape-generated images, or manual prompts.
- **Denoise-controlled priority**: Flexibly blend or prioritize prompt vs. image input.
- **Direct or SDXL processing path**: Choose between full SDXL workflow or direct transparent background processing.
- **Supports Griptape API** (OpenAI + Black Forest Labs) for automated image generation.
- **Designed for upscaling** (e.g., with Topaz Photo AI).
- **Ready-to-print transparent PNG output**.

---

## Requirements
- **ComfyUI** installed and operational.
- **Custom nodes included** in this repository:
  - `CustomSamplerNode`
  - `CustomSchedulerNode`
  - (List any others you included)

- **API keys** for:
  - **OpenAI**
  - **Black Forest Labs**
  
  > Note: Keys are now entered directly inside the **Griptape settings window** in ComfyUI.  
  > (Instructions for obtaining these keys are available online or in Griptape documentation.)

---

## Installation
1. Clone or download this repository.
2. Copy the `custom_nodes` into your ComfyUI `custom_nodes` directory.
3. Copy the workflow `.json` file into your `users/default/workflows` directory.
4. Configure your Griptape API keys inside ComfyUI Settings -> Griptape -> API Keys.
5. Launch ComfyUI and load the workflow!

---

## Usage
- Choose your image source with the **5-Way Toggle Image Switch**.
- Set the **Prompt vs Image Source toggle** depending on your needs.
- Adjust the **Denoise setting** in the KSampler:
  - `0.9` → Image takes precedence
  - `0.5` → Mixed image and prompt
  - `0.1` → Prompt takes precedence
- Set the **SDXL/Direct Path** switch if you want to bypass SDXL for faster processing.

> Detailed instructions are available in the file:  
> **[`Tshirt_Workflow_Notes.md`](./Tshirt_Workflow_Notes.md)**

---

## License
[MIT License](LICENSE)  
You are free to use, modify, and distribute this workflow with attribution.

---

## Credits
- Workflow and custom node design by **Fred LeFevre**.
- Powered by **ComfyUI** and **Griptape AI**.

---
