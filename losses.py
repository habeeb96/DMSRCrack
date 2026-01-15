import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__(); self.smooth = smooth
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        return 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__(); self.a, self.b, self.g, self.s = alpha, beta, gamma, smooth
    def forward(self, logits, targets):
        p = torch.sigmoid(logits).view(-1); t = targets.view(-1).float()
        TP = (p * t).sum(); FP = ((1 - t) * p).sum(); FN = (t * (1 - p)).sum()
        tversky = (TP + self.s) / (TP + self.a * FP + self.b * FN + self.s)
        return (1 - tversky) ** self.g

class BoundaryIoULoss(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size, device=pred.device)
        pred_b = F.max_pool2d(pred, self.kernel_size, stride=1, padding=self.kernel_size//2) - pred
        target_b = F.max_pool2d(target, self.kernel_size, stride=1, padding=self.kernel_size//2) - target
        inter = (pred_b * target_b).sum()
        union = pred_b.sum() + target_b.sum() - inter
        return 1 - (inter + 1e-6) / (union + 1e-6)

class SoftSkeletonize(nn.Module):
    def __init__(self, num_iter=5):
        super().__init__()
        self.num_iter = num_iter
    def forward(self, img):
        for _ in range(self.num_iter):
            img = F.max_pool2d(img, kernel_size=3, stride=1, padding=1)
            img = -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)
        return img

class clDiceLoss(nn.Module):
    def __init__(self, iter=3, smooth=1e-6):
        super().__init__()
        self.soft_skel = SoftSkeletonize(num_iter=iter)
        self.smooth = smooth
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        skel_pred = self.soft_skel(pred)
        skel_true = self.soft_skel(target)
        tprec = (skel_pred * target).sum() / (skel_pred.sum() + self.smooth)
        tsens = (skel_true * pred).sum() / (skel_true.sum() + self.smooth)
        cl_dice = 2 * tprec * tsens / (tprec + tsens + self.smooth)
        return 1 - cl_dice

def offset_regularization_loss(offset):
    l2_loss = torch.mean(offset ** 2)
    tv_x = torch.mean(torch.abs(offset[:, :, :, :-1] - offset[:, :, :, 1:]))
    tv_y = torch.mean(torch.abs(offset[:, :, :-1, :] - offset[:, :, 1:, :]))
    return l2_loss + (tv_x + tv_y)

class PaperCompositeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.ft = FocalTverskyLoss()
        self.bound = BoundaryIoULoss()
        self.cldice = clDiceLoss()
        self.lambda_seg = 1.0; self.lambda_ft = 0.6; self.lambda_b = 0.5
        self.lambda_cl = 0.3; self.lambda_off = 0.1

    def forward(self, logits, mask, offset=None):
        l_dice = self.dice(logits, mask)
        l_bce = self.bce(logits, mask)
        l_seg = 0.5 * l_dice + 0.5 * l_bce
        l_ft = self.ft(logits, mask)
        l_bound = self.bound(logits, mask)
        l_cldice = self.cldice(logits, mask)
        l_off = 0.0
        if offset is not None:
            l_off = offset_regularization_loss(offset)
        return l_seg + (self.lambda_ft * l_ft) + (self.lambda_b * l_bound) + (self.lambda_cl * l_cldice) + (self.lambda_off * l_off)
