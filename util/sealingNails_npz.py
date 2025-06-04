import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import os

class SealingNailDatasetNPZ(Dataset):
    def __init__(self, root='../data/sealingNail_npz', npoints=2048, split='train', use_cache=True):
        self.root = root
        self.split = split
        self.use_cache = use_cache
        self.sealingNails = []
        self.cache = {}
        
        if self.split == 'train':
            self.root = os.path.join(self.root, 'train')
            sealingNails = os.listdir(self.root)
            weight = np.zeros(8)
            
            for index in tqdm(range(len(sealingNails)), desc='Load training set  ', total=len(sealingNails)):
                point_root = os.path.join(self.root, sealingNails[index])
                data = np.load(point_root, allow_pickle=True)
                points = data['points']
                
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
                point_root = os.path.join(self.root, sealingNails[index])
                data = np.load(point_root, allow_pickle=True)
                points = data['points']
                
                choice = np.random.choice(points.shape[0], npoints, replace=False)
                points_choice = points[choice]
                self.sealingNails.append(points_choice)

    def __getitem__(self, index):
        if self.use_cache and index in self.cache:
            return self.cache[index]
            
        points = self.sealingNails[index]
        points_coords = points[:, 0:3]
        points_feats = points[:, 3:6]
        points_labels = points[:, 6]
        
        # Convert to tensors and move to CPU explicitly
        device = torch.device('cpu')
        points_coords = torch.tensor(points_coords, dtype=torch.float, device=device)
        points_feats = torch.tensor(points_feats, dtype=torch.float, device=device)
        points_labels = torch.tensor(points_labels, dtype=torch.long, device=device)  # Changed to long for labels
        
        if self.use_cache:
            self.cache[index] = (points_coords, points_feats, points_labels)
            
        return points_coords, points_feats, points_labels

    def __len__(self):
        return len(self.sealingNails)