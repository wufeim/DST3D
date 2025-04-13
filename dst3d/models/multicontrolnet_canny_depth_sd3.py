import cv2
from diffusers import StableDiffusion3ControlNetPipeline, UniPCMultistepScheduler
from diffusers.models import SD3ControlNetModel
import numpy as np
from PIL import Image
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation

from .base_model import BaseModel

# Must use batch_size == 1
class MultiControlNetCannyDepthSD3(BaseModel):
    def __init__(self, model_name='multicontrolnet-canny-depth-sd3', canny_lower=100, canny_upper=200, controlnet_conditioning_scale=1.0, device='cpu', **kwargs):
        super().__init__(model_name, device)
        self.canny_lower = canny_lower
        self.canny_upper = canny_upper
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.device = device

        # Build Midas pipeline
        self.midas_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas", low_cpu_mem_usage=True)
        self.midas_image_processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        
        # Build ControlNet pipeline
        controlnets = [
            SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny", torch_dtype=torch.float16),
            SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Depth", torch_dtype=torch.float16),
        ]
        self.pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnets, torch_dtype=torch.float16
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()


    def get_condition(self, image):
        edges = cv2.Canny(image, self.canny_lower, self.canny_upper)
        edges = edges[:, :, np.newaxis]
        edges = np.concatenate([edges, edges, edges], axis=2)
        return Image.fromarray(edges)
    

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
        assert len(image) == 1, "Batch size must be 1 for multicontrolnet"

        if strength is None:
            strength = self.controlnet_conditioning_scale

        if visual_prompt is None:
            canny_conditions = [self.get_condition(im).reisze((1024, 1024)) for im in image]
            depth_conditions = [self.get_depth(im).resize((1024, 1024)) for im in image]
            visual_prompt = [canny_conditions[0], depth_conditions[0]]
        
        if isinstance(strength, (int, float)):
            strength = [strength, strength]

        return self.pipe(prompt, control_image=visual_prompt, negative_prompt=negative_prompt, controlnet_conditioning_scale=strength).images
