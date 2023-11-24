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
    parser.add_argument('--model_path', type=str, default='DST/Models')
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

    assert os.path.isdir(args.model_path) or args.model_path.endswith('.csv'), \
        '--model_path should be a CSV file or a directory containing CSV files'
    if os.path.isdir(args.model_path):
        if len(args.synsets) == 0:
            all_model_csv_fnames = [
                os.path.join(args.model_path, x) for x in os.listdir(args.model_path)
                if x.endswith('.csv')]
        else:
            all_model_csv_fnames = [
                os.path.join(args.model_path, x+'.csv') for x in args.synsets
                if os.path.isfile(os.path.join(args.model_path, x+'.csv'))]
    else:
        all_model_csv_fnames = [args.model_path]

    jobs = []
    for model_csv_fname in all_model_csv_fnames:
        csv_data = pd.read_csv(model_csv_fname)
        models = csv_data.values.tolist()

        n_cad_models = len(models)
        n_sample = [args.num_samples // n_cad_models] * n_cad_models
        for i in range(args.num_samples % n_cad_models):
            n_sample[i] += 1
        np.random.shuffle(n_sample)

        for (m, n_sample_model) in zip(models, n_sample):
            model_id, distance, azimuth, elevation, strength, sampling = m

            out_path = os.path.join(
                args.data_path, os.path.basename(model_csv_fname)[:-4], model_id, 'image_render')
            anno_path = os.path.join(
                args.data_path, os.path.basename(model_csv_fname)[:-4], model_id, 'annotation')
            depth_path = os.path.join(
                args.data_path, os.path.basename(model_csv_fname)[:-4], model_id, 'depth')

            model_path = model_loader.load_model(model_id)
            jobs.append((model_path, out_path, anno_path, depth_path, distance, azimuth, elevation, sampling, strength, n_sample_model, args))

    print(f'{len(jobs)} jobs created')
    with multiprocessing.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(worker, jobs), total=len(jobs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Render images with Blender', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
