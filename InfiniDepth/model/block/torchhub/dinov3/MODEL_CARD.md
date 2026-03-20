# Model Card for DINOv3

DINOv3 is a family of versatile vision foundation models that outperforms the specialized state of the art across a broad range of settings, without fine-tuning. DINOv3 produces high-quality dense features that achieve outstanding performance on various vision tasks, significantly surpassing previous self- and weakly-supervised foundation models.

## Model Details

These are Vision Transformer and ConvNeXt models trained following the method described in the DINOv3 paper. 12 models are provided:

- 10 models pretrained on web data (LVD-1689M dataset)
  - 1 ViT-7B trained from scratch,
  - 5 ViT-S/S+/B/L/H+ models distilled from the ViT-7B,
  - 4 ConvNeXt-{T/S/B/L} models distilled from the ViT-7B,
- 2 models pretrained on satellite data (SAT-493M dataset)
  - 1 ViT-7B trained from scratch
  - 1 ViT-L distilled from the ViT-7B


Each Transformer-based model takes an image as input and returns a class token, patch tokens (and register tokens). These models follow a ViT architecture, with a patch size of 16. For a 224x224 image, this results in 1 class token + 4 register tokens + 196 patch tokens = 201 tokens (for DINOv2 with registers this resulted in 1 + 4 + 256 = 261 tokens).

The models can accept larger images provided the image shapes are multiples of the patch size (16). If this condition is not verified, the model will crop to the closest smaller multiple of the patch size.

### Model Description

- **Developed by:** Meta AI
- **Model type:** Vision Transformer, ConvNeXt
- **License:** [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)

### Model Sources

- **Repository:** [https://github.com/facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- **Paper:** [https://arxiv.org/abs/2508.10104](https://arxiv.org/abs/2508.10104)

## Uses

The models are vision backbones providing multi-purpose features for downstream tasks.

### Direct Use

The models can be used without fine-tuning, with downstream classifiers as simple as linear layers, to obtain competitive results:

- on image classification, using k-NN classifiers on the class token
- on image classification, with logistic regression classifiers applied on the class token
- on image classification, with a linear layer applied on the class token and the average of the patch tokens
- on image retrieval using nearest neighbors
- on geometric and semantic 3D keypoint correspondances
- on depth estimation, semantic segmentation, using linear layers
- on unsupervised object discovery
- on video segmentation tracking
- on video classification, using a small 4-layer attentive probe

### Downstream Use

While fine-tuning the models can yield some gains, it is recommended to keep this option as a last resort: the frozen features are expected to provide good performance out-of-the-box.

## Bias, Risks, and Limitations

Compared to DINOv2 and SEERv2, DINOv3 delivers somewhat consistent performance across income categories on geographical fairness and diversity, although with a notable performance drop in the low-income bucket compared to the highest-income bucket.

DINOv3 also achieves relatively good scores across different regions, improving over its predecessor DINOv2. However, a relative difference is still observed between Europe and Africa.

### Recommendations

Fine-tuning is expected to increase the biases in the features produced by the model as they will be tuned to the fine-tuning labels.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
import torch

model = torch.hub.load(
    repo_or_dir='facebookresearch/dinov3',
    model='<MODEL_NAME>',
    weights='<PATH/OR/URL/TO/CHECKPOINT>',
)

# where MODEL_NAME can be one of:
# - dinov3_vits16
# - dinov3_vits16plus
# - dinov3_vitb16
# - dinov3_vitl16
# - dinov3_vith16plus
# - dinov3_vit7b16
# - dinov3_convnext_tiny
# - dinov3_convnext_small
# - dinov3_convnext_base
# - dinov3_convnext_large

# For instance
dinov3_vits16 = torch.hub.load(
    repo_or_dir='facebookresearch/dinov3',
    model='dinov3_vits16',
    weights='<PATH/OR/URL/TO/DINOV3/VITS16/LVD1689M/CHECKPOINT>',
)
```

## Training Details

### Training Data

- Web dataset (LVD-1689M): a curated dataset of 1,689 millions of images extracted from a large data
pool of 17 billions web images collected from public posts on Instagram

- Satellite dataset (SAT-493M): a dataset of 493 millions of 512x512 images sampled randomly from Maxar RGB ortho-rectified imagery at 0.6 meter resolution

### Training Procedure

**Training objective:**

- DINO self-distillation loss with multi-crop
- iBOT masked-image modeling loss
- KoLeo regularization on [CLS] tokens
- Gram anchoring

- **Training regime:** PyTorch FSDP2 (with bf16 and fp8 matrix multiplications)

**Distillation:**

- Distillation follows the standard DINOv3 pretraining procedure, except the teacher is a frozen pretrained ViT-7B.

## Evaluation

**Results**

The reader is referred to the associated paper for details on the evaluation protocols

*Results for ViT backbones pretrained (or distilled) on web (LVD-1689M)*

<table>
  <tr>
    <th></th>
    <!-- <th></th> -->
    <th colspan="4">Global Tasks</th>
    <th colspan="5">Dense Tasks</th>
  </tr>
  <tr>
    <th>Model</th>
    <!-- <th>Dataset</th> -->
    <th>IN-ReaL</th>
    <th>IN-R</th>
    <th>Obj.Net</th>
    <th>Ox.-H</th>
    <th>ADE20k</th>
    <th>NYU↓</th>
    <th>DAVIS</th>
    <th>NAVI</th>
    <th>SPair</th>
  </tr>
  <tr>
    <td>DINOv3 ViT-S/16</td>
    <!-- <td>LVD-1689M</td> -->
    <td align="right">87.0</td>
    <td align="right">60.4</td>
    <td align="right">50.9</td>
    <td align="right">49.5</td>
    <td align="right">47.0</td>
    <td align="right">0.403</td>
    <td align="right">72.7</td>
    <td align="right">56.3</td>
    <td align="right">50.4</td>
  </tr>
  <tr>
    <td>DINOv3 ViT-S+/16</td>
    <!-- <td>LVD-1689M</td> -->
    <td align="right">88.0</td>
    <td align="right">68.8</td>
    <td align="right">54.6</td>
    <td align="right">50.0</td>
    <td align="right">48.8</td>
    <td align="right">0.399</td>
    <td align="right">75.5</td>
    <td align="right">57.1</td>
    <td align="right">55.2</td>
  </tr>
  <tr>
    <td>DINOv3 ViT-B/16</td>
    <!-- <td>LVD-1689M</td> -->
    <td align="right">89.3</td>
    <td align="right">76.7</td>
    <td align="right">64.1</td>
    <td align="right">58.5</td>
    <td align="right">51.8</td>
    <td align="right">0.373</td>
    <td align="right">77.2</td>
    <td align="right">58.8</td>
    <td align="right">57.2</td>
  </tr>
  <tr>
    <td>DINOv3 ViT-L/16</td>
    <!-- <td>LVD-1689M</td> -->
    <td align="right">90.2</td>
    <td align="right">88.1</td>
    <td align="right">74.8</td>
    <td align="right">63.1</td>
    <td align="right">54.9</td>
    <td align="right">0.352</td>
    <td align="right">79.9</td>
    <td align="right">62.3</td>
    <td align="right">61.3</td>
  </tr>
  <tr>
    <td>DINOv3 ViT-H+/16</td>
    <!-- <td>LVD-1689M</td> -->
    <td align="right">90.3</td>
    <td align="right">90.0</td>
    <td align="right">78.6</td>
    <td align="right">64.5</td>
    <td align="right">54.8</td>
    <td align="right">0.352</td>
    <td align="right">79.3</td>
    <td align="right">63.3</td>
    <td align="right">56.3</td>
  </tr>
  <tr>
    <td>DINOv3 ViT-7B/16</td>
    <!-- <td>LVD-1689M</td> -->
    <td align="right">90.4</td>
    <td align="right">91.1</td>
    <td align="right">91.1</td>
    <td align="right">72.8</td>
    <td align="right">55.9</td>
    <td align="right">0.309</td>
    <td align="right">79.7</td>
    <td align="right">64.4</td>
    <td align="right">58.7</td>
  </tr>
</table>

*Results for ConvNeXt backbones distilled on web (LVD-1689M)*

<table>
  <tr>
    <th></th>
    <th colspan="6">Global Tasks</th>
    <th colspan="2">Dense Tasks</th>
  </tr>
  <tr>
    <th>Model</th>
    <th colspan="2">IN-ReaL</th>
    <th colspan="2">IN-R</th>
    <th colspan="2">Obj.Net</th>
    <th>ADE20k</th>
    <th>NYU↓</th>
  </tr>
  <tr>
    <td></th>
    <td>@256px</td>
    <td>@512px</td>
    <td>@256px</td>
    <td>@512px</td>
    <td>@256px</td>
    <td>@512px</td>
    <td colspan="2"></td>
  </tr>
  <tr>
    <td>DINOv3 ConvNeXt Tiny</td>
    <td align="right">86.6</td>
    <td align="right">87.7</td>
    <td align="right">73.7</td>
    <td align="right">74.1</td>
    <td align="right">52.6</td>
    <td align="right">58.7</td>
    <td align="right">42.7</td>
    <td align="right">0.448</td>
  </tr>
  <tr>
    <td>DINOv3 ConvNeXt Small</td>
    <td align="right">87.9</td>
    <td align="right">88.7</td>
    <td align="right">73.7</td>
    <td align="right">74.1</td>
    <td align="right">52.6</td>
    <td align="right">58.7</td>
    <td align="right">44.8</td>
    <td align="right">0.432</td>
  </tr>
  <tr>
    <td>DINOv3 ConvNeXt Base</td>
    <td align="right">88.5</td>
    <td align="right">89.2</td>
    <td align="right">77.2</td>
    <td align="right">78.2</td>
    <td align="right">56.2</td>
    <td align="right">61.3</td>
    <td align="right">46.3</td>
    <td align="right">0.420</td>
  </tr>
  <tr>
    <td>DINOv3 ConvNeXt Large</td>
    <td align="right">88.9</td>
    <td align="right">89.4</td>
    <td align="right">81.3</td>
    <td align="right">82.4</td>
    <td align="right">59.3</td>
    <td align="right">65.2</td>
    <td align="right">47.8</td>
    <td align="right">0.403</td>
  </tr>
</table>

*Results for ViT backbones pretrained (or distilled) on satellite (SAT-493M)*

<table>
  <tr>
    <th></th>
    <th colspan="7">(GEO-Bench) Classification</th>
  </tr>
  <tr>
    <th>Model</ht>
    <th>m-BEnet</th>
    <th>m-brick-kiln
    <th>m-eurosat</th>
    <th>m-forestnet</th>
    <th>m-pv4ger</th>
    <th>m-so2sat</th>
    <th>mean</th>
  </tr>
  <tr>
    <td>DINOv3 ViT-L/16</td>
    <td>73.0</td>
    <td>96.5</td>
    <td>94.1</td>
    <td>60.6</td>
    <td>96.0</td>
    <td>57.4</td>
    <td>79.6</td>
  </tr>
  <tr>
    <td>DINOv3 ViT-7B/16</td>
    <td>74.0</td>
    <td>97.2</td>
    <td>94.8</td>
    <td>62.3</td>
    <td>96.1</td>
    <td>62.1</td>
    <td>81.1</td>
  </tr>
  <tr>
    <th></th>
    <th colspan="7">(GEO-Bench) Segmentation</th>
  </tr>
  <tr>
    <th>Model</th>
    <th>m-cashew</th>
    <th>m-chesapeake</th>
    <th>m-NeonTree</th>
    <th>m-nz-cattle</th>
    <th>m-pv4ger-seg</th>
    <th>m-SA-crop</th>
    <th>mean</th>
  </tr>
  <tr>
    <td>DINOv3 ViT-L/16</td>
    <td>94.2</td>
    <td>75.6</td>
    <td>61.8</td>
    <td>83.7</td>
    <td>95.2</td>
    <td>36.8</td>
    <td>74.5</td>
  </tr>
  <tr>
    <td>DINOv3 ViT-7B/16</td>
    <td>94.1</td>
    <td>76.6</td>
    <td>62.6</td>
    <td>83.4</td>
    <td>95.5</td>
    <td>37.6</td>
    <td>75.0</td>
  </tr>
</table>


## Environmental Impact

- **Hardware Type:** Nvidia H100
- **Hours used:** 61,440 hours for ViT-7B model training
- **Cloud Provider:** Private infrastructure
- **Compute Region:** USA
- **Carbon Emitted:** 18t CO2eq

## Technical Specifications

### Model Architecture and Objective

Vision Transformer models:

- ViT-S (21M parameters): patch size 16, embedding dimension 384, 4 register tokens, 6 heads, MLP FFN, RoPE
- ViT-S+ (29M parameters): patch size 16, embedding dimension 384, 4 register tokens, 6 heads, SwiGLU FFN, RoPE
- ViT-B (86M parameters): patch size 16, embedding dimension 768, 4 register tokens, 12 heads, MLP FFN, RoPE
- ViT-L (300M parameters): patch size 16, embedding dimension 1024, 4 register tokens, 16 heads, MLP FFN, RoPE
- ViT-H+ (840M parameters): patch size 16, embedding dimension 1280, 4 register tokens, 20 heads, SwiGLU FFN, RoPE
- ViT-7B (6716M parameters): patch size 16, embedding dimension 4096, 4 register tokens, 32 heads, SwiGLU FFN, RoPE

ConvNeXt models:

- ConvNeXt Tiny (29M parameters)
- ConvNeXt Small (50M parameters)
- ConvNeXt Base (89M parameters)
- ConvNeXt Large (198M parameters)

### Compute Infrastructure

#### Hardware

Nvidia H100 GPUs

#### Software

PyTorch 2.7

## More Information

See the [blog post](https://ai.meta.com/blog/dinov3-self-supervised-vision-model/) and the associated [website](https://ai.meta.com/dinov3/).

## Citation

**BibTeX**

```
@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie, Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut, Patrick and Bojanowski, Piotr},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.10104},
}
```
