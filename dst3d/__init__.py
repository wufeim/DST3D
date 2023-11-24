import os
import torch

from .models.controlnet_v11 import ControlNetv11Canny
from .models.controlnet_canny_sdxl import ControlNetCannySDXL
from .prompt_gen.simple_prompt import generate_simple_prompt, generate_simple_synset_prompt


def build_gen_model(name, device='cpu', **kwargs):
    if name == 'control_v11p_sd15_canny':
        return ControlNetv11Canny(**kwargs)
    elif name == 'controlnet-canny-sdxl-1.0':
        return ControlNetCannySDXL(**kwargs)
    else:
        raise NotImplementedError(f'Model {name} is not implemented')


def generate_prompts(name, num, class_name, **kwargs):
    if name == 'simple_prompt':
        return generate_simple_prompt(num=num, class_name=class_name, **kwargs)
    elif name == 'simple_synset_prompt':
        return generate_simple_synset_prompt(num=num, class_name=class_name, **kwargs)
    else:
        raise NotImplementedError(f'Prompt generator {name} is not implemented')
