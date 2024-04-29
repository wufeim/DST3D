# Installation

## Setup for Rendering

1. Install Blender v3.2.2

    ```
    wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz
    tar -xf blender-3.2.2-linux-x64.tar.xz
    rm blender-3.2.2-linux-x64.tar.xz
    ```

2. Install `NumPy` and `SciPy` for Blender Python

    ```
    cd blender-3.2.2-linux-x64/3.2/python/bin
    ./python3.10 -m ensurepip
    ./python3.10 -m pip install numpy==1.23.1 scipy
    ```

## Python Environment

```
conda create -n dst3d python=3.10
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install transformers diffusers open_clip_torch pytorch-lightning==1.5.0
pip install objaverse wget einops omegaconf opencv-python scipy
pip install -e .
```

## Download ControlNet Checkpoints

```sh
cd models
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt
wget https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth
```

## Dataset Preparation

1. Download the [ShapeNet Dataset (v2)](https://shapenet.org). Move/link the data to the default location (`./data/ShapeNetCore.v2`), or specify the path during rendering.

2. The [Objaverse Dataset](https://objaverse.allenai.org) is loaded on the fly. Keep the default location at `./data/Objaverse` or specify the path during rendering.
