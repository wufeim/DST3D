# DST3D

Official implementation of **DST3D**, from the following paper:

[Adding 3D Geometry Control to Diffusion Models](https://arxiv.org/abs/2306.08103). ICLR 2024 (Spotlight).\
[Wufei Ma\*](https://wufeim.github.io), [Qihao Liu\*](https://qihao067.github.io), [Jiahao Wang\*](https://jiahaoplus.github.io/), [Angtian Wang](https://github.com/Angtian), [Xiaoding Yuan](https://www.xiaodingyuan.com), [Yi Zhang](https://edz-o.github.io/), [Zihao Xiao](https://scholar.google.com/citations?user=ucb6UssAAAAJ&hl), [Beijia Lu](https://github.com/Beijia11), [Ruxiao Duan](https://scholar.google.com/citations?user=aG-fi1cAAAAJ&hl=en), [Yongrui Qi](https://github.com/Auroraaa-Qi), [Adam Kortylewski](https://gvrl.mpi-inf.mpg.de/), [Yaoyao Liu](https://www.cs.jhu.edu/~yyliu/), [Alan Yuille](https://www.cs.jhu.edu/~ayuille/)\
Johns Hopkins University, University of Freiburg, Max Planck Institute for Informatics\
[[`arXiv`](https://arxiv.org/abs/2306.08103)] [[`iclr.cc`](https://iclr.cc/virtual/2024/poster/18443)]

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

## TODO

- [ ] 3D-DST Dataset for image classification.
- [ ] 3D-DST Dataset for 3D pose estimation.
- [ ] Diverse prompt generation.

## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation

If you find this repository helpful, please consider citing:

```
@article{ma2023adding,
  title={Adding 3D Geometry Control to Diffusion Models},
  author={Ma, Wufei and Liu, Qihao and Wang, Jiahao and Wang, Angtian and Liu, Yaoyao and Kortylewski, Adam and Yuille, Alan},
  journal={arXiv preprint arXiv:2306.08103},
  year={2023}
}
```
