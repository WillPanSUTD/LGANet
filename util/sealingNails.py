import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import os


class SealingNailDataset(Dataset):
    def __init__(self, root='../data/normal', npoints=2048, split='train'):
        self.root = root
        self.split = split
        self.sealingNails = []
        if self.split == 'train':
            self.root = os.path.join(self.root, 'train')
            sealingNails = os.listdir(self.root)
            weight = np.zeros(8)
            for index in tqdm(range(len(sealingNails)), desc='Load training set  ', total=len(sealingNails)):
                # self.sealingNails.extend([sealingNails[index]] * 9)
                point_root = os.path.join(self.root, sealingNails[index])
                points = np.loadtxt(point_root, dtype=np.float32)
                loop = points.shape[0] // npoints
                choice = np.random.choice(points.shape[0], npoints * loop, replace=False)
                points_choice = points[choice]
                for idx in range(loop):
                    points = points_choice[idx * npoints:(idx + 1) * npoints]
                    self.sealingNails.append(points)
                targets = points[:, -1]
                tmp, _ = np.histogram(targets, range(9))
                weight += tmp

            l_weight = np.power(np.amax(weight) / weight, 1 / 2)
            l_weight = l_weight / np.sum(l_weight)
            self.weight = weight / np.amax(weight)
            self.l_weight = l_weight

        else:
            self.root = os.path.join(self.root, 'test')
            sealingNails = os.listdir(self.root)
            for index in tqdm(range(len(sealingNails)), desc='Load test set      ', total=len(sealingNails)):
                # self.sealingNails.extend([sealingNails[index]] * 9)
                point_root = os.path.join(self.root, sealingNails[index])
                points = np.loadtxt(point_root, dtype=np.float32)
                choice = np.random.choice(points.shape[0], npoints, replace=False)
                points_choice = points[choice]
                self.sealingNails.append(points_choice)

    def __getitem__(self, index):
        points = self.sealingNails[index]
        points_coords = points[:, 0:3]
        points_feats = points[:, 3:6]
        points_labels = points[:, 6]
        points_coords = torch.tensor(points_coords, dtype=torch.float)
        points_feats = torch.tensor(points_feats, dtype=torch.float)
        points_labels = torch.tensor(points_labels, dtype=torch.float)
        return points_coords, points_feats, points_labels

    def __len__(self):
        return len(self.sealingNails)
