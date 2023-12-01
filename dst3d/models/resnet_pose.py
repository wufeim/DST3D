import numpy as np
import torch
import torch.nn as nn
import torchvision

from ..utils import pose_error


class ResNetPose:
    def __init__(
            self,
            backbone,
            num_bins=41,
            output_dim=123,
            checkpoint=None,
            transforms=[],
            device='cpu',
            lr=0.01,
            momentum=0.9,
            weight_decay=1e-4,
            step_size=50,
            gamma=0.1
        ):
        self.backbone = backbone
        self.num_bins = num_bins
        self.output_dim = output_dim
        self.checkpoint = checkpoint
        self.transforms = transforms
        self.device = device
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma

        assert self.backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], \
            f"Unsupported backbone {self.backbone} for ResNetPose"
        self.model = torchvision.models.__dict__[self.backbone](pretrained=True)
        self.model.avgpool = nn.AvgPool2d(8, stride=1)
        if self.backbone == 'resnet18':
            self.model.fc = nn.Linear(512 * 1, self.output_dim)
        else:
            self.model.fc = nn.Linear(512 * 4, self.output_dim)

        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=step_size, gamma=gamma)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.num_bins, reduction='none').to(self.device)

        if self.checkpoint is not None:
            ckpt = torch.load(self.checkpoint, map_location='cpu')
            self.model.load_state_dict(ckpt['state'])
            if 'optimizer_state' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer_state'])

        self.model = self.model.to(self.device)

    def train(self, sample):
        self.model.train()
        sample = self.transforms(sample)

        img = sample['img'].to(self.device)
        targets = self._get_targets(sample).long().view(-1).to(self.device)
        output = self.model(img)

        loss = self.criterion(output.view(-1, self.num_bins), targets).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def step_scheduler(self):
        self.scheduler.step()

    def evaluate_pose(self, pred_pose, sample):
        azimuth = pred_pose['azimuth']
        elevation = pred_pose['elevation']
        theta = pred_pose['theta']
        pose_errors = []
        for i in range(len(sample['img'])):
            pose_errors.append(pose_error(
                {'azimuth': azimuth[i], 'elevation': elevation[i], 'theta': theta[i]},
                {'azimuth': sample['azimuth'][i].item(), 'elevation': sample['elevation'][i].item(), 'theta': sample['theta'][i].item()}
            ))
        return pose_errors

    def evaluate(self, sample):
        self.model.eval()
        sample = self.transforms(sample)

        img = sample['img'].to(self.device)
        output = self.model(img).detach().cpu().numpy()

        img_flip = torch.flip(img, dims=[3])
        output_flip = self.model(img_flip).detach().cpu().numpy()

        azimuth = output_flip[:, :self.num_bins]
        elevation = output_flip[:, self.num_bins:2*self.num_bins]
        theta = output_flip[:, 2*self.num_bins:3*self.num_bins]
        output_flip = np.concatenate([azimuth[:, ::-1], elevation, theta[:, ::-1]], axis=1).reshape(-1, self.num_bins * 3)

        output = (output + output_flip) / 2.0
        pose_pred = self._prob_to_pose(output)

        pred = {}
        pred['logits'] = output
        pred['pose'] = {'azimuth': pose_pred[:, 0], 'elevation': pose_pred[:, 1], 'theta': pose_pred[:, 2]}

        if 'azimuth' in sample and 'elevation' in sample and 'theta' in sample:
            pred['pose_error'] = self.evaluate_pose(pred['pose'], sample)

        return pred

    def get_ckpt(self, **kwargs):
        ckpt = kwargs
        ckpt['state'] = self.model.state_dict()
        ckpt['optimizer_state'] = self.optimizer.state_dict()
        ckpt['lr'] = self.optimizer.param_groups[0]['lr']
        return ckpt

    def _get_targets(self, sample):
        azimuth = sample['azimuth'].numpy() / np.pi
        elevation = sample['elevation'].numpy() / np.pi
        theta = sample['theta'].numpy() / np.pi
        theta[theta < -1.0] += 2.0
        theta[theta > 1.0] -= 2.0

        targets = np.zeros((len(azimuth), 3), dtype=np.int32)
        targets[azimuth < 0.0, 0] = self.num_bins - 1 - np.floor(-azimuth[azimuth < 0.0] * self.num_bins / 2.0)
        targets[azimuth >= 0.0, 0] = np.floor(azimuth[azimuth >= 0.0] * self.num_bins / 2.0)
        targets[:, 1] = np.ceil(elevation * self.num_bins / 2.0 + self.num_bins / 2.0 - 1)
        targets[:, 2] = np.ceil(theta * self.num_bins / 2.0 + self.num_bins / 2.0 - 1)

        return torch.from_numpy(targets)

    def _prob_to_pose(self, prob):
        pose_pred = np.argmax(prob.reshape(-1, 3, self.num_bins), axis=2).astype(np.float32)
        pose_pred[:, 0] = (pose_pred[:, 0] + 0.5) * np.pi / (self.num_bins / 2.0)
        pose_pred[:, 1] = (pose_pred[:, 1] - self.num_bins / 2.0) * np.pi / (self.num_bins / 2.0)
        pose_pred[:, 2] = (pose_pred[:, 2] - self.num_bins / 2.0) * np.pi / (self.num_bins / 2.0)
        return pose_pred

    def get_loss(self, sample):
        self.model.train()
        sample = self.transforms(sample)

        img = sample['img'].to(self.device)
        targets = self._get_targets(sample).long().view(-1).to(self.device)
        output = self.model(img)

        loss = self.criterion(output.view(-1, self.num_bins), targets)
        loss = loss.view(-1, 3).sum(dim=1)

        return loss.detach().cpu().numpy()
