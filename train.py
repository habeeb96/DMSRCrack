import os
import csv
import torch
import numpy as np
import logging
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import F1Score, JaccardIndex, Precision, Recall

# IMPORT FROM OTHER FILES
import config
from dataset import CrackDataset, get_transforms
from model import DMSRCrack
from losses import PaperCompositeLoss

# SETUP LOGGING
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# SYSTEM SETUP
torch.manual_seed(config.SEED)
torch.cuda.manual_seed(config.SEED)
np.random.seed(config.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train():
    logger.info(f"Starting Training [Target: {config.TARGET_CLASSES}] on {config.DEVICE}")
    
    # 1. Dataset & Loaders
    train_ds = CrackDataset(config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR, transform=get_transforms('train'), target_classes=config.TARGET_CLASSES)
    val_ds = CrackDataset(config.VAL_IMG_DIR, config.VAL_MASK_DIR, transform=get_transforms('val'), target_classes=config.TARGET_CLASSES)
    
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    # 2. Model & Optimization
    model = DMSRCrack(in_channels=3, num_classes=2).to(config.DEVICE)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.7, verbose=True)
    
    # 3. Loss & Metrics
    criterion = PaperCompositeLoss().to(config.DEVICE)
    
    metric_dice = F1Score(task='binary').to(config.DEVICE)
    metric_iou = JaccardIndex(task='binary').to(config.DEVICE)
    metric_prec = Precision(task='binary').to(config.DEVICE)
    metric_rec = Recall(task='binary').to(config.DEVICE)
    
    best_dice = 0.0
    
    # Initialize CSV Log
    with open(config.CSV_LOG_NAME, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Dice', 'Val_IoU', 'Val_Prec', 'Val_Rec', 'LR'])
    
    for epoch in range(config.EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]"):
            img, mask = batch['image'].to(config.DEVICE), batch['mask'].to(config.DEVICE)
            optimizer.zero_grad()
            out, _ = model(img)
            loss = criterion(out, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- VALIDATE ---
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                img, mask = batch['image'].to(config.DEVICE), batch['mask'].to(config.DEVICE)
                out, _ = model(img)
                pred = (torch.sigmoid(out) > 0.5).long()
                
                metric_dice.update(pred, mask.long())
                metric_iou.update(pred, mask.long())
                metric_prec.update(pred, mask.long())
                metric_rec.update(pred, mask.long())
        
        # Compute final metrics for epoch
        val_dice = metric_dice.compute().item()
        val_iou = metric_iou.compute().item()
        val_prec = metric_prec.compute().item()
        val_rec = metric_rec.compute().item()
        
        # Reset for next epoch
        metric_dice.reset(); metric_iou.reset(); metric_prec.reset(); metric_rec.reset()
        
        # --- LOGGING ---
        avg_train_loss = train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1}: Loss={avg_train_loss:.4f} | LR={current_lr:.6f}")
        logger.info(f"    Val Dice: {val_dice:.4f} | mIoU: {val_iou:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")
        
        # Write to CSV
        with open(config.CSV_LOG_NAME, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, val_dice, val_iou, val_prec, val_rec, current_lr])
        
        # --- SAVE BEST ---
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), config.SAVE_NAME)
            logger.info(f"    >>> Saved Best Model ({best_dice:.4f})")
            
        scheduler.step(val_dice)

if __name__ == '__main__':
    train()
