import os
import cv2
import glob
import torch
import numpy as np
import tifffile
from torch.utils.data import Dataset

def clean_and_check_data(image_paths, mask_paths):
    image_paths = [p for p in image_paths if os.path.getsize(p) > 0 and not p.endswith('.DS_Store')]
    mask_paths = [p for p in mask_paths if os.path.getsize(p) > 0 and not p.endswith('.DS_Store')]
    
    if len(image_paths) != len(mask_paths):
        print("Warning: Number of images and masks do not match!")
    return image_paths, mask_paths

class NucleiDataset(Dataset):
    def __init__(self, imgs, masks, size=1024):
        self.imgs = imgs
        self.masks = masks
        self.size = size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = torch.tensor(img / 255.0).permute(2, 0, 1).float()

        mask = tifffile.imread(self.masks[idx])
        mask = cv2.resize(mask, (self.size, self.size))
        mask = torch.tensor((mask > 0).astype(np.float32)).unsqueeze(0)

        return img, mask
