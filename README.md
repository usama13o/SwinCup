# SwinCup <br /> 

Pytorch implementation of SwinCup: Cascaded Swin Transformer for Histopathological Structures Segmentation in Colorectal Cancer
## Abstract
Transformer models have recently become the dominant architecture in many computer vision tasks, including image classification, object detection, and image segmentation. The main reason behind their success is the ability to incorporate global context information into the learning process. By utilising self-attention, recent advancements in the Transformer architecture design enable models to consider long-range dependencies. In this paper, we propose a novel transformer, named Swin Transformer with Cascaded UPsampling (SwinCup) model for the segmentation of histopathology images. We use a hierarchical Swin Transformer with shifted windows as an encoder to extract global context features. The multi-scale feature extraction in a Swin transformer enables the model to attend to different areas in the image at different scales. A cascaded up-sampling decoder is used with an encoder to improve its feature aggregation. Experiments on GLAS and CRAG histopathology colorectal cancer datasets were used to validate the model.

# Model Architecture
<img src="figures/fig 1.jpg"> <br />
### Results

### References:
If you find this helpful, please cite our paper:

1) "SwinCup: Cascaded Swin Transformer for Histopathological Structures Segmentation in Colorectal Cancer", <br />
[ Paper](https://doi.org/10.1016/j.eswa.2022.119452) <br />


### Installation
pip install --process-dependency-links -e .

