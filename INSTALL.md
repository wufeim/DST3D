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

## Dependency Setup

## Dataset Preparation

1. Download the [ShapeNet Dataset (v2)](https://shapenet.org). Move/link the data to the default location (`./data/ShapeNetCore.v2`), or specify the path during rendering.

2. The [Objaverse Dataset](https://objaverse.allenai.org) is loaded on the fly. Keep the default location at `./data/Objaverse` or specify the path during rendering.
