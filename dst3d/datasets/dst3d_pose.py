import glob
import logging
import os

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision


class DST3DPose(Dataset):
    def __init__(self, data_path, kfold=None, splits=None, transform=None, is_file=None, img_exts=['.png'], resolution=256):
        self.data_path = data_path
        self.kfold = kfold
        self.splits = splits if isinstance(splits, list) else [splits]
        self.transform = transform
        self.is_file = is_file if is_file is not None else lambda x: True
        self.img_exts = img_exts
        self.resolution = resolution

        self.img_files = sorted([x for x in os.listdir(self.data_path) if x.endswith('.png') and is_file(x)])
        self.img_files = sum([
            [x for x in glob.glob(os.path.join(self.data_path, f'**/*{_ext}'), recursive=True) if self.is_file(x)]
            for _ext in self.img_exts
        ], [])

        if kfold is not None:
            np.random.seed(42)
            np.random.shuffle(self.img_files)
            l = len(self.img_files) // kfold
            img_files = []
            for i in range(kfold):
                if i in splits:
                    if i != kfold - 1:
                        img_files += self.img_files[i*l:i*l+l]
                    else:
                        img_files += self.img_files[i*l:]
            self.img_files = img_files

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, item):
        img_file = os.path.join(self.data_path, self.img_files[item])

        img = Image.open(img_file)
        if img.mode != "RGB":
            img = img.convert("RGB")

        img = img.resize((self.resolution, self.resolution), Image.BILINEAR)

        annotation_file = img_file.replace('_00.png', '.npy')
        foldername = os.path.basename(os.path.dirname(annotation_file))
        annotation_file = annotation_file.replace(foldername, 'annotation')
        annot = np.load(annotation_file, allow_pickle=True)[()]

        img = self.transform({'img': img})['img']

        sample = {'img': img}
        sample['azimuth'] = annot['theta']
        sample['elevation'] = np.pi/2 - annot['phi']
        sample['theta'] = 0.0

        return sample


class ToTensor:
    def __init__(self):
        self.trans = torchvision.transforms.ToTensor()

    def __call__(self, sample):
        sample["img"] = self.trans(sample["img"])
        if "kpvis" in sample and not isinstance(sample["kpvis"], torch.Tensor):
            sample["kpvis"] = torch.Tensor(sample["kpvis"])
        if "kp" in sample and not isinstance(sample["kp"], torch.Tensor):
            sample["kp"] = torch.Tensor(sample["kp"])
        return sample


class Normalize:
    def __init__(self):
        self.trans = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __call__(self, sample):
        sample["img"] = self.trans(sample["img"])
        return sample
