import cv2
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
import numpy as np
from PIL import Image
import torch

from .base_model import BaseModel


class ControlNetCannySDXL(BaseModel):
    def __init__(self, model_name='diffusers/controlnet-canny-sdxl-1.0', canny_lower=100, canny_upper=200, controlnet_conditioning_scale=1.0, device='cpu', **kwargs):
        super().__init__(model_name, device)
        self.canny_lower = canny_lower
        self.canny_upper = canny_upper
        self.controlnet_conditioning_scale = controlnet_conditioning_scale

        # Build ControlNet pipeline
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16).to(device)
        vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(device)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16).to(device)
        self.pipe.enable_model_cpu_offload()

    def get_condition(self, image):
        edges = cv2.Canny(image, self.canny_lower, self.canny_upper)
        edges = edges[:, :, np.newaxis]
        edges = np.concatenate([edges, edges, edges], axis=2)
        return Image.fromarray(edges)

    @torch.no_grad()
    def forward(self, image=None, visual_prompt=None, prompt=None, negative_prompt=None, strength=None):
        assert prompt is not None and negative_prompt is not None
        assert image is not None or visual_prompt is not None

        if strength is None:
            strength = self.controlnet_conditioning_scale

        if visual_prompt is None:
            visual_prompt = [self.get_condition(im) for im in image]

        return self.pipe(prompt, negative_prompt=negative_prompt, image=visual_prompt, controlnet_conditioning_scale=strength).images
