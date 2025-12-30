import cv2
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel
import numpy as np
from PIL import Image
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation

from .base_model import BaseModel


class ControlNetDepthSD3(BaseModel):
    def __init__(self, model_name='InstantX/SD3-Controlnet-Depth', controlnet_conditioning_scale=1.0, device='cpu', **kwargs):
        super().__init__(model_name, device)
        self.controlnet_conditioning_scale = controlnet_conditioning_scale

        # Build Midas pipeline
        self.midas_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)
        self.midas_image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        self.device = device

        # Build ControlNet pipeline
        controlnet = SD3ControlNetModel.from_pretrained(
            "InstantX/SD3-Controlnet-Depth",
            torch_dtype=torch.float16).to(device)
        self.pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", 
            controlnet=controlnet,
            torch_dtype=torch.float16).to(device)
        self.pipe.enable_model_cpu_offload()

    
    def get_depth(self, image):
        image = Image.fromarray(image)
        inputs = self.midas_image_processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.midas_model(**inputs)
            depth = outputs.predicted_depth

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        depth = depth.squeeze().cpu().numpy()
        depth = (depth * 255 / np.max(depth)).astype("uint8")
        depth = Image.fromarray(depth).convert('RGB')
        return depth


    @torch.no_grad()
    def forward(self, image=None, visual_prompt=None, prompt=None, negative_prompt=None, strength=None):
        assert prompt is not None and negative_prompt is not None
        assert image is not None or visual_prompt is not None

        if strength is None:
            strength = self.controlnet_conditioning_scale

        print(len(image))
        if visual_prompt is None:
            visual_prompt = [self.get_depth(im).resize((1024, 1024)) for im in image]

        return self.pipe(prompt, negative_prompt=negative_prompt, control_image=visual_prompt, controlnet_conditioning_scale=strength).images
