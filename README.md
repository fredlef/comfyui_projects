# ComfyUI Projects

A collection of custom ComfyUI workflows and nodes created by **Fred LeFevre**.

Clone this repo into your comfyui folder.

## Projects Included

### 🧩 Custom Nodes: Comfyui_FSL_Nodes
- **sampler_config_hub.py**  
  One of two custom nodes to configure sampler and scheduler settings for workflows.  Used to feed data to the meta node
- **sampler_scheduler_config.py**
  Two of two custom nodes to configure sampler and scheduler settings for workflows.  Used to feed data to the meta node
- **manual_alpha_mask_painter.py**  
  Custom node to manually convert black mask areas into alpha transparency.  Useful with GRIPTAPE when inpainting and creating the mask by right clicking on the loaded image and select load mask editor.  Griptape requires an alpha channel mask.
- **8WayImgSwitch**
  Image switch with eight inputs
- **fsl_composite_with_mask_cropped**
  Designed to remove the Alpha Channel But Keep Transparent Parts as Background Color (not currently used)
- **fsl_save_and_strip_alpha**
  Strips alpha channel from RGBA image.  (not currently used)
- **How to use**
  place this node folder into the custom_nodes folder - no additional action is necessary.  Nodes can be found in the comfyui onscreen nodes folder by searching for FSL

Located in the [`Comfyui_FSL_Nodes/`](./Comfyui_FSL_Nodes) directory.

---

### 🎨 Workflow: T-shirt Designer
- **Tshirt-Designer-ver_2.0.json**  
  Workflow designed for generating masked transparent images for T-shirt and object printing.
- Many corrections and updates
- Full workflow details and notes are available in [`tshirt_designer/`](./tshirt_designer).

---

## License

This project and all sub-projects are licensed under the terms of the [LICENSE](./LICENSE) provided at the repository root.

---

## Author

Created and maintained by **Fred LeFevre**.

