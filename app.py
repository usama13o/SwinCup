import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import traceback

# Add project root to sys.path to allow importing project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models import get_model
# ModelOpts is not directly used here as get_model handles it via json_opts.model
from dataio.transformation import get_dataset_transformation # For preprocessing
from utils.util import json_file_to_pyobj # To load config

# --- Configuration ---
DEFAULT_CONFIG_PATH = "configs/config_SwinT_v2_decoderCup.json"
DEFAULT_CHECKPOINT_PATH = "checkpoints/SwinCup_Example/fold_0/099_net_S.pth" 
# IMPORTANT: Users MUST replace DEFAULT_CHECKPOINT_PATH with an actual checkpoint.

# Global variables
loaded_model = None
model_device = None
model_config_for_transform = None # Stores the part of config needed for transforms (e.g., augmentation opts, dataset type)
model_opts_for_postprocessing = None # Stores the ModelOpts object for things like output_nc
transform_fn = None

def load_model_and_config(config_path, checkpoint_path, device_str="cpu"):
    global loaded_model, model_device, model_config_for_transform, model_opts_for_postprocessing, transform_fn

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    json_opts_raw = json_file_to_pyobj(config_path, args=None)
    
    # This is the equivalent of json_opts.model in pywich_test.py
    model_construction_opts = json_opts_raw.model 
    model_construction_opts.isTrain = False 
    model_construction_opts.device = torch.device(device_str)
    model_construction_opts.save_dir = "./gradio_cache" 
    if not os.path.exists(model_construction_opts.save_dir):
        os.makedirs(model_construction_opts.save_dir, exist_ok=True)

    model_device = model_construction_opts.device
    
    print(f"Loading model with config: {config_path} on device: {device_str}")
    # get_model creates ModelOpts internally and calls model.initialize(ModelOpts_instance)
    model = get_model(model_construction_opts, model_device) 
    model_opts_for_postprocessing = model.opts # This is the ModelOpts instance, has .output_nc, .img_size etc.
    
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint file not found at {checkpoint_path}. Model is using initial (random) weights.")
        # Proceeding with an un-trained model for UI testing, actual use needs a checkpoint.
    else:
        print(f"Loading checkpoint: {checkpoint_path}")
        model.load_network_from_path(model.net, checkpoint_path, strict=False)
        print("Checkpoint loaded successfully.")
    
    loaded_model = model
    
    # Store parts of config needed for transformations
    model_config_for_transform = {
        'dataset_type': getattr(json_opts_raw.dataset, 'type', 'epi'), # e.g., 'epi', 'glas'
        'augmentation_opts': getattr(json_opts_raw, 'augmentation', None)
    }
    # If dataset type from config is the model name, use a default transform name
    if model_config_for_transform['dataset_type'] == json_opts_raw.model.name:
        model_config_for_transform['dataset_type'] = 'epi'


    # Get transformations
    transform_name = model_config_for_transform['dataset_type']
    aug_opts_for_transform = model_config_for_transform['augmentation_opts']
    
    transforms_dict = get_dataset_transformation(transform_name, opts=aug_opts_for_transform)
    
    priority_keys = ['test', 'valid', 'validation']
    selected_transform_key = None
    for key in priority_keys:
        if key in transforms_dict:
            selected_transform_key = key
            break
    
    if selected_transform_key:
        transform_fn = transforms_dict[selected_transform_key]
        print(f"Using '{selected_transform_key}' transformations for '{transform_name}'.")
    else:
        print(f"Warning: No 'test' or 'valid' transform found for '{transform_name}'. Using basic ToTensor fallback.")
        from torchvision import transforms as T
        img_h, img_w = 224, 224 # Default
        if aug_opts_for_transform and hasattr(aug_opts_for_transform, transform_name):
            specific_aug_opts = getattr(aug_opts_for_transform, transform_name)
            if hasattr(specific_aug_opts, 'scale_size'):
                scale_sz = specific_aug_opts.scale_size
                img_h, img_w = scale_sz[0], scale_sz[1]
        elif hasattr(model_opts_for_postprocessing, 'img_size'):
             size_from_model = model_opts_for_postprocessing.img_size
             if isinstance(size_from_model, (list,tuple)): img_h, img_w = size_from_model[0], size_from_model[1]
             else: img_h = img_w = size_from_model
        transform_fn = T.Compose([T.Resize((img_h, img_w)), T.ToTensor()])
    
    print("Model and transformations ready.")


def preprocess_image(image_pil, current_transform_fn):
    if image_pil.mode != 'RGB':
        image_pil = image_pil.convert('RGB')

    if current_transform_fn:
        try:
            dummy_label_pil = Image.new('L', image_pil.size, 0) 
            transformed_data = current_transform_fn({'image': image_pil, 'label': dummy_label_pil})
            image_tensor = transformed_data['image']
            
            # Ensure tensor is C, H, W and then add batch dim
            if image_tensor.ndim == 3 and image_tensor.shape[-1] in [1, 3]: # H, W, C
                image_tensor = image_tensor.permute(2, 0, 1)
            elif image_tensor.ndim == 4 and image_tensor.shape[0] == 1 and image_tensor.shape[-1] in [1,3]: # 1, H, W, C
                image_tensor = image_tensor.squeeze(0).permute(2,0,1)
            # Add other potential input formats if necessary
            
            if image_tensor.ndim == 2: # H, W, convert to 1, H, W
                image_tensor = image_tensor.unsqueeze(0)

        except Exception as e:
            print(f"Error applying configured transform: {e}. Traceback: {traceback.format_exc()}")
            from torchvision import transforms as T
            img_h, img_w = 224, 224
            if model_opts_for_postprocessing and hasattr(model_opts_for_postprocessing, 'img_size'):
                 size_from_model = model_opts_for_postprocessing.img_size
                 if isinstance(size_from_model, (list,tuple)): img_h, img_w = size_from_model[0], size_from_model[1]
                 else: img_h = img_w = size_from_model
            simple_transform = T.Compose([T.Resize((img_h, img_w)), T.ToTensor()])
            image_tensor = simple_transform(image_pil)
    else: # Fallback, should ideally not happen if load_model_and_config works
        from torchvision import transforms as T
        print("Warning: transform_fn was None. Using basic ToTensor and resize.")
        img_h, img_w = 224,224
        if model_opts_for_postprocessing and hasattr(model_opts_for_postprocessing, 'img_size'):
             size_from_model = model_opts_for_postprocessing.img_size
             if isinstance(size_from_model, (list,tuple)): img_h, img_w = size_from_model[0], size_from_model[1]
             else: img_h = img_w = size_from_model
        simple_transform = T.Compose([T.Resize((img_h, img_w)), T.ToTensor()])
        image_tensor = simple_transform(image_pil)

    if image_tensor.ndim == 3: # C, H, W
        image_tensor = image_tensor.unsqueeze(0) # Add batch dimension -> 1, C, H, W
    elif image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
         raise ValueError(f"Preprocessed tensor has unexpected shape: {image_tensor.shape}. Expected (1,C,H,W) or (C,H,W).")
    return image_tensor


def generate_error_image(size=(224,224), text="Error"):
    img = Image.new("RGB", size, color="red")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    draw.text((10, size[1]//2 -10), text, fill=(255,255,255), font=font)
    return img

def postprocess_output(raw_output_tensor, num_classes):
    if raw_output_tensor is None: return generate_error_image(text="No output from model")

    if raw_output_tensor.ndim == 4 and raw_output_tensor.shape[0] == 1:
        processed_tensor = raw_output_tensor.squeeze(0)
    elif raw_output_tensor.ndim == 3:
        processed_tensor = raw_output_tensor
    else:
        print(f"Warning: Unexpected output tensor shape: {raw_output_tensor.shape}.")
        return generate_error_image(text=f"Bad output shape\n{raw_output_tensor.shape}")

    if processed_tensor.ndim == 3 and processed_tensor.shape[0] == num_classes and num_classes > 1:
        seg_map = torch.argmax(processed_tensor, dim=0)
    elif processed_tensor.ndim == 3 and processed_tensor.shape[0] == 1: # Single class output (already a map)
        seg_map = processed_tensor.squeeze(0)
    elif processed_tensor.ndim == 2: # Already a H,W map
        seg_map = processed_tensor
    elif num_classes == 1 and processed_tensor.ndim == 3 and processed_tensor.shape[0] == 1 : # Binary case, single channel output
         seg_map = (processed_tensor.squeeze(0) > 0.5).byte() # Threshold if it's logits/probabilities
    else:
        print(f"Warning: Unhandled processed tensor shape: {processed_tensor.shape} for {num_classes} classes.")
        return generate_error_image(text=f"Bad processed shape\n{processed_tensor.shape}")

    seg_map_np = seg_map.cpu().numpy().astype(np.uint8)
    
    palette = np.array([
        [0,0,0],[128,0,0],[0,128,0],[128,128,0],[0,0,128],[128,0,128],[0,128,128],
        [192,192,192],[128,128,128],[255,0,0],[0,255,0],[255,255,0],[0,0,255],
        [255,0,255],[0,255,255],[255,228,196] # Adding Bisque for class 15
    ])
    if num_classes > len(palette):
        extra_colors = np.random.randint(0, 255, size=(num_classes - len(palette), 3))
        palette = np.vstack((palette, extra_colors))

    colored_seg_map = np.zeros((seg_map_np.shape[0], seg_map_np.shape[1], 3), dtype=np.uint8)
    for class_idx in range(num_classes):
        colored_seg_map[seg_map_np == class_idx] = palette[class_idx % len(palette)]
        
    return Image.fromarray(colored_seg_map, 'RGB')

def predict(image_pil, device_choice_ui="cpu"):
    global loaded_model, model_device, model_opts_for_postprocessing, transform_fn
    
    status_message = "Starting..."
    if image_pil is None: return None, generate_error_image(text="No image uploaded."), "Please upload an image."

    try:
        if loaded_model is None or model_device.type != device_choice_ui:
            status_message = f"Loading model for {device_choice_ui}..."
            print(status_message)
            load_model_and_config(DEFAULT_CONFIG_PATH, DEFAULT_CHECKPOINT_PATH, device_choice_ui)
            status_message = f"Model loaded on {model_device.type}."
            print(status_message)
        
        status_message = "Preprocessing image..."
        print(status_message)
        input_tensor = preprocess_image(image_pil, transform_fn)
        input_tensor = input_tensor.to(model_device)
        status_message = "Image preprocessed."
        print(status_message)

        status_message = "Running inference..."
        print(status_message)
        loaded_model.net.eval()
        with torch.no_grad():
            loaded_model.set_input(input_tensor, None)
            raw_output = None
            if hasattr(loaded_model, 'test') and callable(getattr(loaded_model, 'test')):
                loaded_model.test() # Populates loaded_model.pred_seg or similar
            elif hasattr(loaded_model, 'validate') and callable(getattr(loaded_model, 'validate')):
                loaded_model.validate() # May also populate standard output attributes
            else: # Fallback to direct net call
                loaded_model.prediction = loaded_model.net(input_tensor)
            
            # Determine output tensor
            if hasattr(loaded_model, 'pred_seg'): raw_output = loaded_model.pred_seg
            elif hasattr(loaded_model, 'prediction'): raw_output = loaded_model.prediction
            else: raise RuntimeError("Model did not produce recognizable output attribute ('pred_seg' or 'prediction').")
        
        status_message = "Inference complete."
        print(status_message)

        status_message = "Postprocessing output..."
        print(status_message)
        num_classes = getattr(model_opts_for_postprocessing, 'output_nc', 2)
        output_image_pil = postprocess_output(raw_output, num_classes)
        status_message = "Output processed successfully."
        print(status_message)
    
        return image_pil, output_image_pil, status_message

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return image_pil, generate_error_image(text=error_msg[:100]), error_msg


title = "SwinCup: Cascaded Swin Transformer for Histopathological Structures Segmentation"
description = """
This is a demo for SwinCup, a segmentation model for histopathological images.
Upload an image to see the model segment relevant structures.
**Note:** This demo uses a pre-configured model (`config_SwinT_v2_decoderCup.json`). 
You **MUST** provide a valid pre-trained checkpoint file at the path defined by `DEFAULT_CHECKPOINT_PATH` in `app.py` 
(currently: `{DEFAULT_CHECKPOINT_PATH}`). If no valid checkpoint is found, the model will use random weights, and results will be meaningless.
"""
article = f"""
Check out the research paper: <a href="https://doi.org/10.1016/j.eswa.2022.119452" target="_blank">SwinCup: Cascaded swin transformer for histopathological structures segmentation in colorectal cancer</a>.<br>
Default Config: {DEFAULT_CONFIG_PATH}<br>
**Required Checkpoint Path (edit in app.py): {DEFAULT_CHECKPOINT_PATH}**
"""

example_image_paths = [] 
if os.path.exists("figures/fig 1.jpg"):
    example_image_paths.append("figures/fig 1.jpg")

# Initial load attempt on CPU to avoid blocking GPU for long if app just started
# This is optional, predict function will load if model is None
try:
    print("Attempting to preload model on app startup (CPU)...")
    load_model_and_config(DEFAULT_CONFIG_PATH, DEFAULT_CHECKPOINT_PATH, "cpu")
except Exception as e:
    print(f"Preloading failed (this is okay if checkpoint/config is not set yet): {e}")

ui = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload Histopathological Image"),
        gr.Radio(choices=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"], 
                 label="Select Device", 
                 value="cuda" if torch.cuda.is_available() else "cpu")
    ],
    outputs=[
        gr.Image(type="pil", label="Original Image"),
        gr.Image(type="pil", label="Segmented Output"),
        gr.Textbox(label="Status")
    ],
    title=title,
    description=description,
    article=article,
    examples=example_image_paths if example_image_paths else None,
    allow_flagging="never",
    live=False # Avoids multiple rapid calls if inputs change fast
)

if __name__ == "__main__":
    print("Starting Gradio app... Ensure your checkpoint path is correctly set in app.py.")
    ui.launch(share=False)

# To run: python app.py
# Ensure Gradio is installed: pip install gradio
# Ensure a valid checkpoint is at DEFAULT_CHECKPOINT_PATH or modify the path.
# Ensure a valid config is at DEFAULT_CONFIG_PATH.
