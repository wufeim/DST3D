import argparse
import json
import multiprocessing
import os
import subprocess

import numpy as np
import pandas as pd
from tqdm import tqdm
import wget

from dst3d.utils import ModelLoader


def get_args_parser():
    parser = argparse.ArgumentParser(description='Render images with Blender', add_help=False)
    parser.add_argument('--model_path', type=str, default='/path/to/3d-dst-models.csv')
    parser.add_argument('--data_path', type=str, default='DST/train')
    parser.add_argument('--synsets', type=str, default=[], nargs='+')
    parser.add_argument('--workers', type=int, default=8)

    parser.add_argument('--shapenet_path', type=str, default=None)
    parser.add_argument('--objaverse_path', type=str, default=None, help='If empty, will download automatically.')
    parser.add_argument('--omniobject3d_path', type=str, default=None)
    parser.add_argument('--toys4k_path', type=str, default=None)
    parser.add_argument('--bg_hdr_dir', type=str, default=None, help='If given, add random world environment')

    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--disable_random', action='store_true')
    parser.add_argument('--disable_random_distance', action='store_true')
    return parser


def worker(params):
    model_path, out_path, anno_path, depth_path, distance, azimuth, elevation, sampling, strength, n_sample_model, args = params

    command = (
        f"blender-3.2.2-linux-x64/blender -b -P scripts/blender_render.py -- "
        f"--object_path {model_path} --engine CYCLES --output_dir {out_path} --depth_dir {depth_path} "
        f"--distance {distance} --azimuth {azimuth} --elevation {elevation} --strength {strength} "
        f"--num_images {n_sample_model} --anno_dir {anno_path} --pose_sampling {sampling} ")
    if args.bg_hdr_dir is not None:
        command += f"--bg_hdr_dir {args.bg_hdr_dir} "
    if args.disable_random:
        command += "--disable_random "
    if args.disable_random_distance:
        command += "--distance_min 0.0 --distance_max 0.0 "
    subprocess.run(command, shell=True)


def main(args):
    model_loader = ModelLoader(
        shapenet_path=args.shapenet_path,
        objaverse_path=args.objaverse_path,
        omniobject3d_path=args.omniobject3d_path,
        toys4k_path=args.toys4k_path)

    assert args.model_path.endswith('.csv'), '--model_path should specify a CSV file'
    raw_model_data = pd.read_csv(args.model_path).values.tolist()
    model_data = {}
    for row in raw_model_data:
        if row[0] not in model_data:
            model_data[row[0]] = {}
        model_data[row[0]].append(row[1:])

    if args.synsets is None or len(args.synsets) == 0:
        args.synsets = sorted(list(model_data.keys()))

    jobs = []
    for synset in args.synsets:
        models = model_data[synset]

        n_cad_models = len(models)
        n_sample = [args.num_samples // n_cad_models] * n_cad_models
        for i in range(args.num_samples % n_cad_models):
            n_sample[i] += 1
        np.random.shuffle(n_sample)

        for (m, n_sample_model) in zip(models, n_sample):
            model_id, distance, azimuth, elevation, strength, sampling = m

            out_path = os.path.join(
                args.data_path, synset, model_id, 'image_render')
            anno_path = os.path.join(
                args.data_path, synset, model_id, 'annotation')
            depth_path = os.path.join(
                args.data_path, synset, model_id, 'depth')

            model_path = model_loader.load_model(model_id)
            jobs.append((model_path, out_path, anno_path, depth_path, distance, azimuth, elevation, sampling, strength, n_sample_model, args))

    print(f'{len(jobs)} jobs created')
    with multiprocessing.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(worker, jobs), total=len(jobs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Render images with Blender', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
