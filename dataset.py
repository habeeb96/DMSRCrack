import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logger = logging.getLogger(__name__)

def get_transforms(stage='train'):
    if stage == 'train':
        return A.Compose([
            A.PadIfNeeded(min_height=448, min_width=448, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Resize(448, 448, interpolation=cv2.INTER_NEAREST),
            # --- Geometric ---
            A.Rotate(limit=45, p=0.5),                                      
            A.Affine(translate_percent=(-0.05, 0.05), p=0.3),            
            A.HorizontalFlip(p=0.5),                                        
            A.VerticalFlip(p=0.5),                                          
            # --- Photometric ---
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),    
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
    else:
        return A.Compose([
            A.PadIfNeeded(min_height=448, min_width=448, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Resize(448, 448, interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})

class CrackDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_classes=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = []
        self.masks = []
        
        all_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')) + glob.glob(os.path.join(image_dir, '*.png')))
        for img_path in all_files:
            img_name = os.path.basename(img_path)
            mask_path = os.path.join(self.mask_dir, img_name)
            
            if not os.path.exists(mask_path): continue
            
            # Filter by Class
            if target_classes:
                is_target = False
                for cls in target_classes:
                    if img_name.startswith(cls):
                        is_target = True; break
                if not is_target: continue
            
            self.image_files.append(img_path)
            self.masks.append(mask_path)
        
        logger.info(f"Loaded {len(self.image_files)} samples. Filters: {target_classes}")

    def __len__(self): return len(self.image_files)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_files[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask'].unsqueeze(0).float()
            
        return {'image': img, 'mask': mask}
