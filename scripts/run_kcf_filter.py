import argparse
import json
import logging
import os

import numpy as np
import torch
import torchvision

from dst3d.datasets.dst3d_pose import DST3DPose, ToTensor, Normalize
from dst3d.models import ResNetPose
from dst3d.utils import setup_logging, set_seed, str2bool


def parse_args():
    parser = argparse.ArgumentParser(description='Run K-fold consistency filter (KCF)')
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--category', type=str, default='n02690373')
    parser.add_argument('--output_dir', type=str, default='exp/kcf_n02690373')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')

    # Model params
    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--num_bins', type=int, default=41)
    parser.add_argument('--output_dim', type=int, default=123)

    # Data params
    parser.add_argument('--data_path', type=str, default='/path/to/DST_pose')
    parser.add_argument('--data_name', type=str, default='image_dst')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataloader_workers', type=int, default=8)

    # Training params
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=50)
    parser.add_argument('--ckpt_interval', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.1)
    return parser.parse_args()


def build_dataset(is_train, splits, args):
    transform = [ToTensor()]
    transform = torchvision.transforms.Compose(transform)

    dataset = DST3DPose(args.data_path, kfold=args.kfold, splits=splits, transform=transform, is_file=lambda x: args.data_name in x)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True if is_train else False,
        num_workers=args.dataloader_workers)
    return dataloader


def build_model(args):
    if args.model == 'resnet':
        transforms = [Normalize()]
        transforms = torchvision.transforms.Compose(transforms)
        model = ResNetPose(
            backbone=args.backbone, num_bins=args.num_bins, output_dim=args.output_dim,
            checkpoint=None, transforms=transforms, device=args.device, lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, step_size=args.step_size,
            gamma=args.gamma)
    else:
        raise NotImplementedError(f'Model {args.model} not implemented')
    return model


def main():
    args = parse_args()
    args.data_path = os.path.join(args.data_path, args.category)

    setup_logging(args.output_dir)
    logging.info(args)
    set_seed(args.seed)

    all_splits = list(range(args.kfold))
    loss_mapping = {}
    for fold_idx in range(args.kfold):
        logging.info(f'Running fold {fold_idx+1}/{args.kfold}...')
        train_splits = [x for x in all_splits if x != fold_idx]
        train_dataloader = build_dataset(True, train_splits, args)
        val_splits = [fold_idx]
        val_dataloader = build_dataset(False, val_splits, args)

        logging.info(f'Building {args.model} model...')
        model = build_model(args)

        logging.info(f'Building dataset...')
        logging.info(f'Train dataset: {len(train_dataloader.dataset)}')
        logging.info(f'Val dataset: {len(val_dataloader.dataset)}')

        logging.info(f'Start training...')
        for epo in range(args.epochs):
            loss_list = []
            for i, sample in enumerate(train_dataloader):
                loss_dict = model.train(sample)
                loss_list.append(loss_dict['loss'])

            if (epo + 1) % args.log_interval == 0:
                logging.info(
                    f"[Epoch {epo+1}/{args.epochs}] lr={model.optimizer.param_groups[0]['lr']:.5f} loss={np.mean(loss_list)}"
                )

            if (epo + 1) % args.eval_interval == 0:
                pose_errors = []
                for i, sample in enumerate(val_dataloader):
                    pose_errors += model.evaluate(sample)['pose_error']
                pose_errors = np.array(pose_errors)
                logging.info(
                    f"[Val Epoch {epo+1}] pi/18={np.mean(pose_errors<np.pi/18)*100:.2f}% "
                    f"pi/6={np.mean(pose_errors<np.pi/6)*100:.2f}% "
                    f"pi/4={np.mean(pose_errors<np.pi/4)*100:.2f}% "
                    f"pi/2={np.mean(pose_errors<np.pi/2)*100:.2f}% "
                    f"MeanErr={np.mean(pose_errors)/np.pi*180.0:.2f} "
                    f"MedianErr={np.median(pose_errors)/np.pi*180.0:.2f}"
                )

            if (epo + 1) % args.ckpt_interval == 0:
                ckpt_path = os.path.join(args.output_dir, 'ckpts', f'model_fold{fold_idx+1}_{epo+1}.pth')
                torch.save(model.get_ckpt(epoch=epo+1, args=vars(args)), ckpt_path)
                logging.info(f"[Ckpt Epoch {epo+1}] saved to {ckpt_path}")

            if (epo + 1) == args.epochs:
                ckpt_path = os.path.join(args.output_dir, 'ckpts', f'model_fold{fold_idx+1}_last.pth')
                torch.save(model.get_ckpt(epoch=epo+1, args=vars(args)), ckpt_path)
                logging.info(f"[Ckpt Epoch {epo+1}] saved to {ckpt_path}")

            model.step_scheduler()

        logging.info('Training finished')

        logging.info('Evaluation on training data')
        pose_errors = []
        for i, sample in enumerate(train_dataloader):
            pose_errors += model.evaluate(sample)['pose_error']
        pose_errors = np.array(pose_errors)
        logging.info(
            f"[Train] pi/18={np.mean(pose_errors<np.pi/18)*100:.2f}% "
            f"pi/6={np.mean(pose_errors<np.pi/6)*100:.2f}% "
            f"pi/4={np.mean(pose_errors<np.pi/4)*100:.2f}% "
            f"pi/2={np.mean(pose_errors<np.pi/2)*100:.2f}% "
            f"MeanErr={np.mean(pose_errors)/np.pi*180.0:.2f} "
            f"MedianErr={np.median(pose_errors)/np.pi*180.0:.2f}")

        logging.info('Evaluation on validation data')
        pose_errors = []
        for i, sample in enumerate(val_dataloader):
            pose_errors += model.evaluate(sample)['pose_error']
        pose_errors = np.array(pose_errors)
        logging.info(
            f"[Val] pi/18={np.mean(pose_errors<np.pi/18)*100:.2f}% "
            f"pi/6={np.mean(pose_errors<np.pi/6)*100:.2f}% "
            f"pi/4={np.mean(pose_errors<np.pi/4)*100:.2f}% "
            f"pi/2={np.mean(pose_errors<np.pi/2)*100:.2f}% "
            f"MeanErr={np.mean(pose_errors)/np.pi*180.0:.2f} "
            f"MedianErr={np.median(pose_errors)/np.pi*180.0:.2f}")

        logging.info(f'Calculating losses for validation samples')
        loss_list = []
        for i, sample in enumerate(val_dataloader):
            scores = model.get_loss(sample)
            loss_list += scores.tolist()
        for f, l in zip(val_dataloader.dataset.img_files, loss_list):
            loss_mapping[f.replace(args.data_path+'/', '')] = l

        del model
        del train_dataloader, val_dataloader

        logging.info(f'Fold {fold_idx+1}/{args.kfold} completed')

    with open(os.path.join(args.output_dir, 'kcf_losses.json'), 'w') as fp:
        json.dump(loss_mapping, fp, indent=4)
    logging.info(f'Score file saved to: {os.path.join(args.output_dir, "kcf_losses.json")}')


if __name__ == '__main__':
    main()
