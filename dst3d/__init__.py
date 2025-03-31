import os
import torch

from .models.controlnet_v11 import ControlNetv11Canny
from .models.controlnet_canny_sdxl import ControlNetCannySDXL
from .models.controlnet_canny_sd3 import ControlNetCannySD3
from .models.multicontrolnet_canny_depth_sdxl import MultiControlNetCannyDepthSDXL
from .models.multicontrolnet_canny_depth_sd3 import MultiControlNetCannyDepthSD3
from .prompt_gen.simple_prompt import generate_simple_prompt, generate_simple_synset_prompt


def build_gen_model(name, device='cpu', **kwargs):
    if name == 'control_v11p_sd15_canny':
        return ControlNetv11Canny(**kwargs)
    elif name == 'controlnet-canny-sdxl-1.0':
        return ControlNetCannySDXL(**kwargs)
    elif name == 'SD3-Controlnet-Canny':
        return ControlNetCannySD3(**kwargs)
    elif name == 'multicontrolnet-canny-depth-sdxl-1.0':
        return MultiControlNetCannyDepthSDXL(**kwargs)
    elif name == 'multicontrolnet-canny-depth-sd3':
        return MultiControlNetCannyDepthSD3(**kwargs)
    else:
        raise NotImplementedError(f'Model {name} is not implemented')


def generate_prompts(name, num, class_name, **kwargs):
    if name == 'simple_prompt':
        return generate_simple_prompt(num=num, class_name=class_name, **kwargs)
    elif name == 'simple_synset_prompt':
        return generate_simple_synset_prompt(num=num, class_name=class_name, **kwargs)
    else:
        raise NotImplementedError(f'Prompt generator {name} is not implemented')
