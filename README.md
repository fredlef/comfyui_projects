# ComfyUI Projects

A collection of custom ComfyUI workflows and nodes created by **Fred LeFevre**.

Clone this repo into your comfyui folder.

## Projects Included

### Update notes

- **Node Updates**
   Gemini Generate Image Node v6 was replaced with Gemini Generate Image Node v8.  The v8 version has removed models that do not return an image.  The v8 version also has added the Gemini-3-Pro-Image_Preview model. 
- **Workflow Updates**
	All workflows have been updated to have the Gemini Generate Image Node v8.

### 🧩 Custom Nodes: Comfyui_FSL_Nodes
- **manual_alpha_mask_painter.py**  
  Custom node to manually convert black mask areas into alpha transparency.  Useful with GRIPTAPE when inpainting and creating the mask by right clicking on the loaded image and select load mask editor.  Griptape requires an alpha channel mask.
- **8WayImgSwitch**
  Image switch with eight inputs
- **fsl_composite_with_mask_cropped.py**
  Designed to remove the Alpha Channel But Keep Transparent Parts as Background Color (not currently used)
- **fsl_save_and_strip_alpha**
  Strips alpha channel from RGBA image.  (not currently used)
- **FSLGeminiGenerateImageV8.py**
  Allows for the use of Gemini API's to create image.  This node also provide for whether or not the preceding image is used for iterative generation.  The dimension fields are just to support Metadata.  They do not effect the image.
- **fsl_image_memory.py**
  Provides 4 nodes.  Image Memory Store and Image Memory Recall stores an image to the key entered in the 'key' field.  Image memory recall recalls the image for subsequent use.  Image Memory Clear is used to delete the stored image based on the entered key.  Image Memory Clear All clears all keys that have been entered.
- **fsl_prompt_compose.py**
  Node for Positive and Negative prompt as well as 'scene-lock'.  When scene-lock is true -
  When lock_scene: True, the node prepends a short instruction block before your positive text that tells the model to treat the provided image as fixed “scene layout” and only change what you explicitly ask. The included language (summarized) is:
    - use the provided image as base;
    - keep composition, subjects, positions, background, lighting, camera angle, clothing, colors unchanged;
    - change only what you specify;
    - do not crop, do not isolate a single subject, do not change the background.
- **fsl_ensure_nhwc_batch.py**
  Node guarantees that any incoming image tensor is converted to the [N, H, W, C] (batch–height–width–channels) layout expected by downstream nodes, automatically re-ordering dimensions and normalizing types so all image data conform to a consistent format for safe processing.
- **FSLImageSaverWithMetadataV5.py**
  Node stores metadata with image.  Data can be easily read using the LoadImage-w-Metadata.json workflow

Located in the [`Comfyui_FSL_Nodes/`](https://github.com/fredlef/comfyui_projects/tree/main/custom_nodes/Comfyui_FSL_Nodes) directory.

---

### 🎨 Workflow: T-shirt Designer
- **Tshirt-Designer-ver_2.0.json**  
  It is recommended not to use this workflow.  A replacement workflow that is less complex will be uploaded as 3.0
  Workflow designed for generating masked transparent images for T-shirt and object printing.
- Many corrections and updates
- Full workflow details and notes are available in [`tshirt_designer/`](https://github.com/fredlef/comfyui_projects/tree/main/workflows/tshirt_designer).

---

### 🎨 Workflow: Nano Banana (An API is required)
- **Nano Banana Iterative-Base.json**
  Allows the user to create an image using Gemini-2.5 Flash-image-preview or Gemini-2.5 Flash.  Once the initial image is created additional iterations of each succeeding image can be created just be entering a new prompt.  Setup is included in a Note node in the workflow.
- **Nano Banana Iterative-Load 5 Images.json**
  Provides for using up to 5 nodes to create an image with Nano Banana.  The images can they be modified as in the Iterative-Base workflow. Basic instructions are provided in Notes nodes in the workflow.
- **Nano Banana w_Griptape w_Upscaler.json**
  Uses Griptape Nodes as a front end prompt loader to Nano Banana.  An excellent upscaler by Alex at ComfyUiStudio is included.  An OpenAPI Key is required.
- **Nano Banana img2img w_Griptape.json**
  Uses Griptape Nodes as a front end prompt loader to Nano Banana in support of image to image workflow.
- **Nano Banana img2img Base.json**
  Base workflow for image to image with Nano Banana.
  
---

### Miscellaneous Workflows
- **LoadImage-w-Metadata.json**
  Simple workflow to read the metadata created by the FSLImageSaverWithMetadataV5 node
  
---

### Clarification - Init Image vs Image
- **Gemini Generate Image Node v8**
This node has input sockets for both Init-Image and Images.  This was primarily designed to support the iterative workflow.  The Init-Image socket is specifically for the iterative workflow and should be set to "True" after the first image is created.  For the first image both the Init-Image and Images socket should be set to "False".  

If the iterative workflow has a front loader for the first image then for the first run the Images socket should be set to "True" and the Init-Image should be set to "False".  After the first run the Images socket should be changed to "False" and the Init-images socket should be set to True.

I am using this node generally for all my nano banana workflows so if this is for a workflow other than the iterative workflow the Images socket should always be set to "True" and the Init-Image socket should always be set to false.

## Acknowledgements
- **Thanks to Alex and ComfyuiStudio for allowing me to use his excellent upscaler**
- **Mycroft Holmes - AKA ChatGPT for assistance and guidance in the creation of Custom Nodes**

---

## Author

Created and maintained by **Fred LeFevre**.

