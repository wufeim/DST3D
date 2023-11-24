import os
import sys

import einops
import numpy as np
import torch

from .base_model import BaseModel

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
this_dir = os.path.dirname(__file__)
lib_path = os.path.join(this_dir, 'controlnet_v11_src')
add_path(lib_path)

from annotator.canny import CannyDetector
from annotator.util import resize_image, HWC3
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from share import *


class ControlNetv11Canny(BaseModel):
    def __init__(self, model_name='control_v11p_sd15_canny', model_path='models', save_memory=False, img_resolution=512, canny_lower=100, canny_upper=200, aperture_size=3, eta=1.0, scale=9.0, strength=1.0, guess_mode=False, device='cpu', ddim_steps=20, **kwargs):
        super().__init__(model_name, device)
        self.save_memory = save_memory
        self.img_resolution = img_resolution
        self.canny_lower = canny_lower
        self.canny_upper = canny_upper
        self.aperture_size = aperture_size
        self.strength = strength
        self.ddim_steps = ddim_steps
        self.eta = eta
        self.scale = scale
        self.guess_mode = guess_mode

        # Build model
        self.model = create_model(os.path.join(model_path, f'{self.model_name}.yaml')).cpu()
        self.model.load_state_dict(load_state_dict(os.path.join(model_path, 'v1-5-pruned.ckpt'), location='cuda'), strict=False)
        self.model.load_state_dict(load_state_dict(os.path.join(model_path, f'{self.model_name}.pth'), location='cuda'), strict=False)
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)

        self.apply_canny = CannyDetector()

    def get_condition(self, image):
        image = resize_image(HWC3(image), self.img_resolution)
        detected_map = self.apply_canny(image, self.canny_lower, self.canny_upper, self.aperture_size)
        detected_map = HWC3(detected_map)
        return detected_map

    def forward(self, image=None, visual_prompt=None, prompt=None, negative_prompt=None, strength=None):
        assert prompt is not None and negative_prompt is not None
        assert image is not None or visual_prompt is not None

        if strength is None:
            strength = self.strength

        if visual_prompt is None:
            visual_prompt = [self.get_condition(im) for im in image]

        control = torch.cat([
            torch.from_numpy(vp.copy()).float().unsqueeze(0).cuda() / 255.0
            for vp in visual_prompt], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w')

        B, _, H, W = control.shape

        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        cond = {'c_concat': [control], 'c_crossattn': [self.model.get_learned_conditioning(prompt)]}
        un_cond = {'c_concat': None if self.guess_mode else [control], 'c_crossattn': [self.model.get_learned_conditioning(negative_prompt)]}
        shape = (4, H // 8, W // 8)

        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=True)

        self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if self.guess_mode else ([strength] * 13)
        samples, intermediates = self.ddim_sampler.sample(
            self.ddim_steps, B,
            shape, cond, verbose=False, eta=self.eta,
            unconditional_guidance_scale=self.scale,
            unconditional_conditioning=un_cond)

        if self.save_memory:
            self.model.low_vram_shift(is_diffusing=False)

        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(B)]
        return results
