# Generating Images with 3D Annotations Using Diffusion Models

[![LICENSE](https://img.shields.io/badge/license-MIT-green?style=flat-square)](https://github.com/yaoyao-liu/wufeim/DST3D/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg?style=flat-square&logo=python&color=3776AB)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0.1-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/)

\[[PDF](https://openreview.net/pdf?id=XlkN11Xj6J)\] \[[Project Page](https://ccvl.jhu.edu/3D-DST/)\] 

This repository contains the PyTorch implementation for the **ICLR 2024 Spotlight Paper** "Generating Images with 3D Annotations Using Diffusion Models" by the following authors.
<br>
[Wufei Ma\*](https://wufeim.github.io), [Qihao Liu\*](https://qihao067.github.io), [Jiahao Wang\*](https://jiahaoplus.github.io/), [Angtian Wang](https://github.com/Angtian), [Xiaoding Yuan](https://www.xiaodingyuan.com), [Yi Zhang](https://edz-o.github.io/), [Zihao Xiao](https://scholar.google.com/citations?user=ucb6UssAAAAJ&hl), [Beijia Lu](https://github.com/Beijia11), [Ruxiao Duan](https://scholar.google.com/citations?user=aG-fi1cAAAAJ&hl=en), [Yongrui Qi](https://github.com/Auroraaa-Qi), [Adam Kortylewski](https://gvrl.mpi-inf.mpg.de/), [Yaoyao Liu](https://www.cs.jhu.edu/~yyliu/)<sup>✉</sup>, [Alan Yuille](https://www.cs.jhu.edu/~ayuille/)


## Overview

We present 3D Diffusion Style Transfer (3D-DST), a simple and effective approach to generate images with 3D annotations using diffusion models. Our method exploits ControlNet, which extends diffusion models by using visual prompts in addition to text prompts. We render 3D CAD models from a variety of poses and viewing directions, compute the edge maps of the rendered images, and use these edge maps as visual prompts to generate realistic images. With explicit 3D geometry control, we can easily change the 3D structures of the objects in the generated images and obtain ground-truth 3D annotations automatically. Experiments on image classification, 3D pose estimation, and 3D object detection show that with 3D-DST data we can effectively improve the models' performance in both in-distribution and out-of-ditribution settings.

Besides code to reproduce our data generation pipeline, we also release the following data to support other research projects in the community:

1. **Aligned CAD models for all 1000 classes in ImageNet-1k.** See [`ccvl/3D-DST-models`](https://huggingface.co/datasets/ccvl/3D-DST-models).
2. **LLM-generated captions for all 1000 classes in ImageNet-1k.** See [`ccvl/3D-DST-captions`](https://huggingface.co/datasets/ccvl/3D-DST-captions).
3. **3D-DST data for all 1000 classes in ImageNet-1k.** See [`ccvl/3D-DST-data`](https://huggingface.co/datasets/ccvl/3D-DST-data).

## Installation

Please check [INSTALL.md](INSTALL.md) for installation instructions.

## Quick Start

1. Rendering images with Blender.

    ```sh
    python3 scripts/render_synthetic_data.py \
        --data_path DST3D/train \
        --model_path /path/to/all_dst_models \
        --shapenet_path /path/to/ShapeNetCore.v2 \
        --objaverse_path /path/to/objaverse_models \
        --omniobject3d_path /path/to/OpenXD-OmniObject3D-New \
        --synsets n02690373 \
        --workers 48 \
        --num_samples 2500 \
        --disable_random_distance
    ```

2. DST image generation with visual prompts and LLM prompts.

    ```sh
    CUDA_VISIBLE_DEVICES=0 python3 scripts/controllable_generation.py \
        --model_name control_v11p_sd15_canny \
        --data_path DST3D \
        --data_name image_dst \
        --synsets n02690373
    ```

3. Run K-fold Consistency Filter (KCF) on the generated images. The KCF code trains a ResNet50 pose estimation model and produces a validation loss for each sample. The results are saved in a `JSON` file in `--output_dir`.

    ```sh
    CUDA_VISIBLE_DEVICES=0 python3 scripts/run_kcf_filter.py \
        --data_path DST3D/train \
        --category n02690373 \
        --output_dir exp/kcf_n02690373
    ```

## Released 3D-DST Data

We release our generated 3D-DST data for all 1000 classes in ImageNet-1k [here](https://huggingface.co/datasets/ccvl/3D-DST-data). We also provide the [DeiT-small](https://github.com/facebookresearch/deit/blob/main/README_deit.md) models trained on our 3D-DST data.

**Image Classification on ImageNet-200.**

| model | data | acc@1 | url |
| --- | --- | --- | --- |
| DeiT-small | baseline | 81.5 | [checkpoint & log](https://drive.google.com/file/d/12RQpkuWunUeCoI4nzoWuhgxvekn2eNl0/view?usp=sharing) |
| DeiT-small | with 3D-DST | 84.8 | [checkpoint & log](https://drive.google.com/file/d/1bFPgPXOssT7SVAce31tNESMpMImkA0MR/view?usp=sharing) |

**Image Classification on ImageNet-1k.** We provide baseline results on ImageNet-1k with 3D-DST pretraining.

| model | data | acc@1 |
| --- | --- | --- |
| DeiT-small | baseline | 80.1 |
| DeiT-small | with 3D-DST | 81.1 |

## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation

If you find this repository helpful, please consider citing:

```
@inproceedings{ma2024generating,
title={Generating Images with 3D Annotations Using Diffusion Models},
author={Wufei Ma and Qihao Liu and Jiahao Wang and Angtian Wang and Xiaoding Yuan and Yi Zhang and Zihao Xiao and Guofeng Zhang and Beijia Lu and Ruxiao Duan and Yongrui Qi and Adam Kortylewski and Yaoyao Liu and Alan Yuille},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=XlkN11Xj6J}
}
```
