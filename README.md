# SwinCup <br /> 

Pytorch implementation of SwinCup: Cascaded Swin Transformer for Histopathological Structures Segmentation in Colorectal Cancer
## Abstract
Transformer models have recently become the dominant architecture in many computer vision tasks, including image classification, object detection, and image segmentation. The main reason behind their success is the ability to incorporate global context information into the learning process. By utilising self-attention, recent advancements in the Transformer architecture design enable models to consider long-range dependencies. In this paper, we propose a novel transformer, named Swin Transformer with Cascaded UPsampling (SwinCup) model for the segmentation of histopathology images. We use a hierarchical Swin Transformer with shifted windows as an encoder to extract global context features. The multi-scale feature extraction in a Swin transformer enables the model to attend to different areas in the image at different scales. A cascaded up-sampling decoder is used with an encoder to improve its feature aggregation. Experiments on GLAS and CRAG histopathology colorectal cancer datasets were used to validate the model.

# Model Architecture
<img src="figures/fig 1.jpg"> <br />

### References:
If you find this helpful, please cite our paper:

1) "Usama Zidan, Mohamed Medhat Gaber, Mohammed M. Abdelsamea,
SwinCup: Cascaded swin transformer for histopathological structures segmentation in colorectal cancer, Expert Systems with Applications, Volume 216, 2023." <br />
[ Paper](https://doi.org/10.1016/j.eswa.2022.119452) <br />


### Installation
pip install --process-dependency-links -e .

## Environment Setup

It is recommended to use a virtual environment to manage dependencies.

Key dependencies for this project include:
- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-image
- OpenCV
- And other packages listed in `requriments.txt` and `setup.py`.

To install the dependencies, navigate to the project's root directory and run:
```bash
pip install -r requriments.txt
pip install --process-dependency-links -e .
```

## Testing the Model

To train and validate the model, you can use the `pywich_test.py` script. This script utilizes configuration files to define the training and validation parameters.

Available configuration files can be found in the `configs/` directory. Some examples include:
- `configs/config_SwinT_v2_decoderCup.json`
- `configs/config_TransUnet.json`
- `configs/config_deeplab.json`

To run the script, use a command similar to the following:
```bash
python pywich_test.py --config_path configs/your_config_file.json --device cuda:0
```
Replace `your_config_file.json` with the desired configuration file. You can also specify the device using the `--device` argument (e.g., `--device cpu`, `--device cuda`, `--device cuda:1`). See the "CPU/GPU Execution" section for more details.

## Example Usage

To run inference on a sample image, you would typically load your trained model and pass the image through it. Below is a conceptual example:

```python
# Import necessary libraries
import torch
from torchvision import transforms
from PIL import Image
# Assuming your model is defined in models.networks.swin_unet
from models.networks.swin_unet import SwinUnet # Adjust if your model class is different

# Load your trained model checkpoint
# model_path = 'path/to/your/trained_model.pth' # Replace with your model path
# model = SwinUnet(num_classes=NUM_CLASSES) # Replace NUM_CLASSES with the actual number of classes
# model.load_state_dict(torch.load(model_path))
# model.eval()

# Define image transformations (adjust as per your model's requirements)
# preprocess = transforms.Compose([
#     transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)), # Replace with your image dimensions
#     transforms.ToTensor(),
#     # Add any other necessary normalization or transformations
# ])

# Load and preprocess the image
# image_path = 'path/to/your/sample_image.png' # Replace with your image path
# input_image = Image.open(image_path).convert("RGB")
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0) # Create a mini-batch as expected by the model

# Perform inference
# with torch.no_grad():
#     output = model(input_batch)

# Process the output (e.g., apply softmax, get class predictions)
# probabilities = torch.softmax(output[0], dim=0)
# predicted_class = torch.argmax(probabilities, dim=0)

# print(f"Predicted segmentation map: {predicted_class}")
```
**Note:**
- Replace placeholders like `path/to/your/trained_model.pth`, `NUM_CLASSES`, `IMAGE_HEIGHT`, `IMAGE_WIDTH`, and `path/to/your/sample_image.png` with actual values relevant to your setup.
- The input image should be a histopathological image suitable for the task (e.g., colorectal cancer tissue).
- The exact model loading and inference steps might vary based on the final model implementation and how checkpoints are saved.

## CPU/GPU Execution

The model training and inference can be run on either CPU or GPU. The `pywich_test.py` script has been updated to support device selection via a command-line argument.

**Device Selection:**

Use the `--device` argument when running `pywich_test.py` to specify the desired execution device.
-   To use a specific GPU: `--device cuda:0` (for the first GPU), `--device cuda:1` (for the second GPU), etc.
-   To use any available GPU (if multiple exist, typically defaults to `cuda:0`): `--device cuda`
-   To use CPU: `--device cpu`

**Example:**
```bash
# Run on the first GPU
python pywich_test.py --config_path configs/your_config.json --device cuda:0

# Run on CPU
python pywich_test.py --config_path configs/your_config.json --device cpu
```

**Default Behavior:**

If the `--device` argument is not provided, the script will automatically:
1.  Check if a CUDA-enabled GPU is available.
2.  If a GPU is available, it will default to using `cuda`.
3.  If no GPU is available, it will default to using `cpu`.

The selected device will be printed to theconsole when the script starts. All model parameters and tensors will be moved to this selected device.

## Gradio User Interface

A Gradio interface is available to interactively test the segmentation model.

**Installation for Gradio App:**

Before running the app, ensure you have Gradio and other necessary libraries installed:
```bash
pip install gradio torch torchvision Pillow numpy
# Ensure other project dependencies from requriments.txt are also installed
```

**Running the Gradio App:**

1.  **IMPORTANT: Configure Checkpoint Path:**
    Open the `app.py` file and locate the `DEFAULT_CHECKPOINT_PATH` variable.
    You **MUST** update this path to point to a valid pre-trained model checkpoint file (`.pth`) that you have generated (e.g., by running `pywich_test.py`).
    Example: `DEFAULT_CHECKPOINT_PATH = "checkpoints/MyExperiment/fold_0/050_net_S.pth"`

2.  **Run the application:**
    ```bash
    python app.py
    ```

3.  Open your web browser and navigate to the local URL provided by Gradio (usually `http://127.0.0.1:7860`).

**Features:**
-   Upload your own histopathological images for segmentation.
-   Select the execution device (CPU or GPU if available).
-   View the original and segmented images side-by-side.
-   Get status updates during model loading and processing.

If a valid checkpoint is not provided or found, the model will load with random weights, and the segmentation results will not be meaningful.

## Unit Tests

This project includes unit tests to verify the functionality of key components, including data loading, model forward passes, and loss functions.

**Running Tests:**

The tests are written using Python's built-in `unittest` module. To discover and run all tests, navigate to the root directory of the project and execute:

```bash
python -m unittest discover tests
```

You can also run individual test files:

```bash
python -m unittest tests.test_dataio
python -m unittest tests.test_models
python -m unittest tests.test_loss
```

Refer to `tests/README.md` for more details on the test structure and coverage.

## Detailed Model Architectures

This section provides a brief overview of the key model architectures implemented in this project.

### 1. SwinCup (from `models/networks/swin_unet.py`)

The SwinCup model is a U-Net like architecture that leverages a Swin Transformer for its encoder and a custom upsampling decoder.

*   **Overall Architecture:** Encoder-decoder structure with skip connections, designed for segmentation tasks.
*   **Encoder (`SwinTransformerSys` -> `forward_features` method):**
    *   `PatchEmbed`: The input image is first divided into non-overlapping patches using a convolutional layer (`nn.Conv2d`), and each patch is treated as a token.
    *   `BasicLayer` (Hierarchical Stages): The encoder consists of multiple stages, each composed of:
        *   `SwinTransformerBlock` (repeated `depth` times per stage): These are the core blocks of the Swin Transformer. Each block contains:
            *   Layer Normalization (`nn.LayerNorm`).
            *   Windowed Multi-Head Self-Attention (W-MSA or SW-MSA): Self-attention is computed within local windows. For SW-MSA (Shifted Window Multi-Head Self-Attention), windows are shifted across blocks to enable cross-window connections. Relative position bias is added to the attention mechanism.
            *   MLP: A Multi-Layer Perceptron with two linear layers and GELU activation.
        *   `PatchMerging`: At the end of each stage (except the last), this layer downsamples the feature map by 2x. It achieves this by concatenating features from 2x2 neighboring patches and applying a linear layer to reduce dimensionality.
*   **Decoder (`SwinTransformerSys` -> `forward_up_features` method):**
    *   The decoder mirrors the encoder's hierarchical structure, progressively upsampling features.
    *   Initial `PatchExpand`: The features from the encoder bottleneck are first upsampled.
    *   `BasicLayer_up` (Hierarchical Stages): Each decoder stage includes:
        *   `SwinTransformerBlock`: Similar to the encoder, these blocks process features at the current resolution.
        *   `PatchExpand`: This layer upsamples the features (typically by 2x) to increase resolution, effectively reversing the `PatchMerging` operation. It reshapes and expands the features, often followed by a normalization layer.
    *   **Skip Connections:** Features from the corresponding encoder stages are concatenated with the upsampled features at each decoder stage. A linear layer (`nn.Linear` in `concat_back_dim`) is applied to the concatenated features to merge them.
*   **Output Head (`SwinTransformerSys` -> `up_x4` method):**
    *   `FinalPatchExpand_X4`: A final patch expansion layer that upsamples the feature map resolution by 4x.
    *   `nn.Conv2d`: A 1x1 convolution then maps the features to the desired number of output classes for segmentation.

### 2. Standard U-Net (from `models/networks/unet_2D.py`)

This is a classical U-Net architecture, well-known for biomedical image segmentation.

*   **Overall Architecture:** Symmetric encoder-decoder structure with prominent skip connections.
*   **Encoder (Downsampling Path):**
    *   Consists of repeated blocks, each containing:
        *   `unetConv2`: Two consecutive 3x3 convolutional layers (`nn.Conv2d`), each typically followed by Batch Normalization (`nn.BatchNorm2d` - optional via `is_batchnorm`) and ReLU activation.
        *   `nn.MaxPool2d`: A 2x2 max pooling operation for downsampling, reducing spatial dimensions by half.
*   **Bottleneck:**
    *   The deepest part of the network, connecting the encoder and decoder. It consists of a `unetConv2` block.
*   **Decoder (Upsampling Path):**
    *   Consists of repeated blocks, each performing:
        *   Upsampling (`unetUp`): Feature maps are upsampled, either using transposed convolution (`nn.ConvTranspose2d` if `is_deconv=True`) or bilinear upsampling (`nn.Upsample`).
        *   Skip Connection: The upsampled features are concatenated with the corresponding feature maps from the encoder path.
        *   `unetConv2`: Two consecutive 3x3 convolutional layers (with optional BatchNorm and ReLU) to refine the combined features.
*   **Output Head:**
    *   `nn.Conv2d`: A final 1x1 convolution maps the features from the last decoder stage to the specified number of output classes (`n_classes`).

### 3. Vision Transformer (ViT) based Segmentation Model (e.g., TransUNet - from `models/networks/vit_seg_modeling.py`)

This model, often referred to as TransUNet, combines a Vision Transformer (ViT) based encoder with a CNN-style decoder for segmentation.

*   **Overall Architecture:** A ViT encoder processes image patches as sequences, and a convolutional decoder upsamples these features to produce a segmentation map.
*   **Encoder (`VisionTransformer` -> `Transformer` class):**
    *   `Embeddings`:
        *   Hybrid Backbone (Optional): Can use a ResNetV2 (`ResNetV2` class) as an initial feature extractor. The output feature maps from ResNet are then used for patch embedding.
        *   `patch_embeddings` (`nn.Conv2d`): If not using a hybrid backbone, a convolutional layer directly converts input image patches into linear embeddings (tokens). If using ResNet, this Conv2D processes the ResNet features.
        *   `position_embeddings` (`nn.Parameter`): Learnable parameters are added to the patch embeddings to incorporate positional information for each token.
    *   `Encoder` (Stack of `Block` modules): This is the core Transformer encoder.
        *   `Block`: Each Transformer block consists of:
            *   Layer Normalization (`nn.LayerNorm`).
            *   Multi-Head Self-Attention (`Attention` class): Standard self-attention mechanism.
            *   Residual Connection.
            *   Another Layer Normalization.
            *   MLP (`Mlp` class): A feed-forward network (two linear layers with GELU activation).
            *   Residual Connection.
*   **Decoder (`VisionTransformer` -> `DecoderCup` class):**
    *   The ViT output (sequence of context-aware token embeddings) is reshaped into a 2D feature map.
    *   `conv_more` (`Conv2dReLU`): An initial 3x3 convolutional block (Conv2D -> BatchNorm -> ReLU) processes the reshaped features.
    *   `DecoderBlock` (repeated): The decoder path consists of several blocks, each performing:
        *   Upsampling (`nn.UpsamplingBilinear2d`): Increases spatial resolution. CARAFE upsampler is also an option commented out.
        *   Skip Connection: Concatenates with features from the ResNetV2 backbone if used (`features` from `Embeddings`). The number of skip connections is configurable (`config.n_skip`).
        *   Two `Conv2dReLU` blocks: Each block is a 3x3 Conv2D followed by BatchNorm and ReLU, to refine features.
*   **Output Head (`VisionTransformer` -> `SegmentationHead` class):**
    *   A final convolutional layer (`nn.Conv2d`, typically 3x3 or 1x1) maps the decoder output to the number of classes.
    *   An optional `nn.UpsamplingBilinear2d` layer can further upscale the output to match the input image size.

### 4. Generic Swin Transformer Backbone (from `models/networks/swin_transformer.py`)

This file defines a `SwinTransformer` class that can serve as a powerful hierarchical vision backbone, suitable for various downstream tasks beyond direct segmentation with a fixed head.

*   **Encoder Structure:** The encoder part of this `SwinTransformer` is very similar to that described for SwinCup:
    *   `PatchEmbed`: Converts the input image into patch embeddings.
    *   `BasicLayer` (repeated for stages): Each stage contains `SwinTransformerBlock` modules (with W-MSA/SW-MSA and MLP layers) for feature learning.
    *   `PatchMerging`: Used for downsampling between stages.
*   **Multi-Stage Feature Output:** A key characteristic is its ability to output feature maps from multiple stages of the hierarchy, as specified by `out_indices`. This makes it flexible for U-Net like architectures or Feature Pyramid Network (FPN) style heads. Each output feature map is normalized using `nn.LayerNorm`.
*   **Versatility with Decoder Heads:** While it can be used as a standalone feature extractor, the file also demonstrates its integration with various decoder/segmentation heads:
    *   `DecoderCup`: The same decoder structure used in `vit_seg_modeling.py` (TransUNet).
    *   `UPerHead`: A popular segmentation head (Unified Perceptual Parsing for Scene Understanding).
    *   `SegmentationHead`: A simple convolutional head.
    *   `MaskTransformer`: A more experimental head involving Transformer layers for mask prediction.
    This modularity allows the Swin Transformer backbone to be adapted to different segmentation approaches.

