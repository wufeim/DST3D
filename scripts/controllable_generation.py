import argparse
import json
import math
import os
import random
import string

import cv2
import numpy as np
from PIL import Image
from pytorch_lightning import seed_everything
import torch

from dst3d import build_gen_model, generate_prompts


def parse_args():
    parser = argparse.ArgumentParser(description='Generate DST dataset')
    parser.add_argument('--model_name', type=str, default='controlnet-canny-sdxl-1.0')
    parser.add_argument('--data_path', type=str, default='DST')
    parser.add_argument('--data_name', type=str, default='image_llama')
    parser.add_argument('--synsets', type=str, nargs='+', default=None)
    parser.add_argument('--splits', type=str, nargs='+', default=['train'])
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--prompt_name', type=str, default='simple_synset_prompt')
    parser.add_argument('--a_prompt', type=str, default='best quality, extremely detailed')
    parser.add_argument('--n_prompt', type=str, default='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
    parser.add_argument('--num_imgs_per_synset', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--img_resolution', type=int, default=512)
    parser.add_argument('--canny_lower', type=int, default=100)
    parser.add_argument('--canny_upper', type=int, default=200)
    parser.add_argument('--aperture_size', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--ddim_steps', type=int, default=20)
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--scale', type=float, default=9.0)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--guess_mode', type=bool, default=False)
    parser.add_argument('--regenerate', action='store_true')
    return parser.parse_args()


class ImagePromptIterator:
    def __init__(
        self,
        src_path,
        dst_path,
        synset,
        model_id,
        args,
        num_imgs_limit=-1,
        sample_idx=0,
    ):
        self.src_path = src_path
        self.dst_path = dst_path
        self.synset = synset
        self.model_id = model_id
        self.args = args
        self.num_imgs_limit = num_imgs_limit
        self.sample_idx = sample_idx
        self.regenerate = self.args.regenerate

        self.ptr = 0
        self.all_img_names = [x for x in os.listdir(self.src_path) if x.endswith('.png')]
        if self.num_imgs_limit != -1 and len(self.all_img_names) > self.num_imgs_limit:
            np.random.shuffle(self.all_img_names)
            self.all_img_names = self.all_img_names[:self.num_imgs_limit]

        self.prompts, self.negative_prompts = generate_prompts(
            name=args.prompt_name, num=len(self.all_img_names), class_name=synset, model_id=model_id, **vars(args))

    def __iter__(self):
        return self

    def hasnext(self):
        return self.ptr < len(self.all_img_names)

    def get_n_next(self, n):
        if self.ptr == len(self.all_img_names):
            raise StopIteration
        img_name_list, image_list, prompt_list, negative_prompt_list = [], [], [], []
        while len(img_name_list) < n:
            img_name = self.all_img_names[self.ptr]
            img = np.array(Image.open(os.path.join(self.src_path, img_name)).convert('RGB'))
            pos_p = self.prompts[self.ptr]
            neg_p = self.negative_prompts[self.ptr]
            self.ptr += 1

            if not self.regenerate and os.path.isfile(os.path.join(self.dst_path, f'{img_name[:-4]}_{self.sample_idx:02d}.png')):
                continue

            img_name_list.append(img_name)
            image_list.append(img)
            prompt_list.append(pos_p)
            negative_prompt_list.append(neg_p)

            if self.ptr == len(self.all_img_names):
                break
        return img_name_list, image_list, prompt_list, negative_prompt_list


def main():
    args = parse_args()
    print(args)

    # Build model
    model = build_gen_model(name=args.model_name, **vars(args))

    if args.seed == -1:
        args.seed = random.randint(0, 65535)
    seed_everything(args.seed)

    for split in os.listdir(args.data_path):
        if split not in args.splits:
            continue
        all_synsets = os.listdir(os.path.join(args.data_path, split))
        if args.synsets is not None:
            all_synsets = args.synsets
        for synset in all_synsets:
            all_model_ids = [x for x in os.listdir(os.path.join(args.data_path, split, synset)) if os.path.isdir(os.path.join(args.data_path, split, synset, x))]
            if args.num_imgs_per_synset != 0:
                num_imgs_limit = math.ceil(args.num_imgs_per_synset / len(all_model_ids))
            else:
                num_imgs_limit = -1
            prompt_anno = {}
            num_finished_imgs = 0
            for model_id in all_model_ids:
                src_path = os.path.join(args.data_path, split, synset, model_id, 'image_render')
                dst_path = os.path.join(args.data_path, split, synset, model_id, args.data_name)
                os.makedirs(dst_path, exist_ok=True)
                prompt_anno[model_id] = {}

                for sample_idx in range(args.num_samples):
                    img_prompt_itr = ImagePromptIterator(
                        src_path, dst_path, synset, model_id, args, num_imgs_limit, sample_idx)

                    while img_prompt_itr.hasnext():
                        img_name_list, image_list, prompt_list, negative_prompt_list = \
                            img_prompt_itr.get_n_next(args.batch_size)

                        results = model.forward(
                            image=image_list, prompt=prompt_list, negative_prompt=negative_prompt_list)
                        for batch_idx in range(len(results)):
                            prefix = img_name_list[batch_idx][:-4]
                            img = results[batch_idx]
                            if isinstance(img, np.ndarray):
                                img = Image.fromarray(img)
                            img.save(os.path.join(dst_path, f'{prefix}_{sample_idx:02d}.png'))
                            prompt_anno[model_id][f'{prefix}_{sample_idx:02d}'] = prompt_list[batch_idx]

                        num_finished_imgs += len(results)
                        print(f'[{split}-{synset}] {num_finished_imgs} images generated')

            with open(os.path.join(args.data_path, split, synset, 'prompts.json'), 'w') as fp:
                json.dump(prompt_anno, fp, indent=4)


if __name__ == '__main__':
    main()
