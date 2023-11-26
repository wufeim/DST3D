# Documentation

## Rendering with Blender

```
python3 scripts/render_synthetic_data.py \
    --data_path DST_pose/train \
    --model_path /path/to/all_dst_models \
    --shapenet_path /path/to/ShapeNetCore.v2 \
    --objaverse_path /path/to/objaverse_models \
    --omniobject3d_path /path/to/OpenXD-OmniObject3D-New \
    --synsets n02690373 \
    --workers 48 \
    --num_samples 2500 \
    --disable_random_distance
```

## Diffusion Models

### ControlNet on Canny edges from SDXL v1.0

* Model name: `controlnet-canny-sdxl-1.0`
* Source: [`diffusers/controlnet-canny-sdxl-1.0`](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0)
