import argparse
import os
import random
import shutil

import cv2
import einops
import numpy as np
from PIL import Image
from pytorch_lightning import seed_everything
import torch

from annotator.canny import CannyDetector
from annotator.util import resize_image, HWC3
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
import config
from share import *

colors = [
    'red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'purple', 'white', 'black', 'gray',
    'silver', 'pink', 'maroon', 'brown', 'beige', 'tan', 'peach', 'lime', 'olive', 'turquoise',
    'teal', 'navy blue', 'indigo', 'violet'
]


def parse_args():
    parser = argparse.ArgumentParser(description="Style transfer with ControlNet")
    parser.add_argument('--src_data_path', type=str, default='../data/diffusion_synthetic_dataset')
    parser.add_argument('--output_path', type=str, default='../data/diffusion_synthetic_dataset_transfer')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--img_resolution', type=int, default=512)
    parser.add_argument('--low_thr', type=int, default=100)
    parser.add_argument('--high_thr', type=int, default=200)
    parser.add_argument('--aperture_size', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--ddim_steps', type=int, default=20)
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--scale', type=float, default=9.0)
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--guess_mode', action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--prompt_color', action='store_false')
    return parser.parse_args()


class RandomPrompt():
    def __init__(self, args):
        self.prompts = []
        if args.prompt_color:
            for c in colors:
                self.prompts.append(f'{c} CATE')

    def __call__(self, cate):
        return np.random.choice(self.prompts).replace('CATE', cate)


@torch.no_grad()
def style_transfer(model, apply_canny, random_prompt, ddim_sampler, cate, img, args):
    prompt = random_prompt(cate)
    a_prompt = 'best quality, extremely detailed'
    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    img = resize_image(HWC3(img), args.img_resolution)
    H, W, C = img.shape

    detected_map = apply_canny(img, args.low_thr, args.high_thr, apertureSize=args.aperture_size)
    detected_map = HWC3(detected_map)

    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(args.num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * args.num_samples)]}
    un_cond = {"c_concat": None if args.guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * args.num_samples)]}
    shape = (4, H // 8, W // 8)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)

    model.control_scales = [args.strength * (0.825 ** float(12 - i)) for i in range(13)] if args.guess_mode else ([args.strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(args.ddim_steps, args.num_samples,
                                                 shape, cond, verbose=False, eta=args.eta,
                                                 unconditional_guidance_scale=args.scale,
                                                 unconditional_conditioning=un_cond)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    results = [x_samples[i] for i in range(args.num_samples)]
    return prompt, [255 - detected_map] + results


def main():
    args = parse_args()
    print(args)

    # Build model
    model = create_model('models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('models/control_sd15_canny.pth', location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    if args.seed == -1:
        args.seed = random.randint(0, 65535)
    seed_everything(args.seed)

    apply_canny = CannyDetector()
    random_prompt = RandomPrompt(args)

    for cate in os.listdir(args.src_data_path):
        all_model_names = os.listdir(os.path.join(args.src_data_path, cate))
        if args.debug:
            all_model_names = all_model_names[:5]
        for model_name in all_model_names:
            os.makedirs(os.path.join(args.output_path, cate, model_name), exist_ok=True)
            img_names = [
                x for x in os.listdir(os.path.join(args.src_data_path, cate, model_name))
                if x.endswith('.png')]
            if args.debug:
                img_names = img_names[:5]
            for name in img_names:
                src_img_path = os.path.join(args.src_data_path, cate, model_name, name)
                dst_img_dir = os.path.join(args.output_path, cate, model_name)
                img = np.array(Image.open(src_img_path))

                prompt, results = style_transfer(
                    model, apply_canny, random_prompt, ddim_sampler, cate, img, args)

                prefix = '.'.join(name.split('.')[:-1])
                for i in range(args.num_samples):
                    if args.debug:
                        img = np.concatenate([results[0], results[i+1]], axis=1)
                        img = cv2.putText(img, prompt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        img = results[i+1]
                    Image.fromarray(img).save(os.path.join(dst_img_dir, f'{prefix}_{i:02d}.png'))


if __name__ == '__main__':
    main()
