"""Segmentation losses: boundary, focal and Tversky."""
import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


class BoundaryLoss(torch.nn.Module):
    def __init__(self, num_classes):
        super(BoundaryLoss, self).__init__()
        self.num_classes = num_classes

    def compute_sdf(self, img_gt, out_shape):
        img_gt = img_gt.astype(np.uint8)
        sdf = np.zeros(out_shape)
        for b in range(out_shape[0]):
            for c in range(1, out_shape[1]):
                posmask = img_gt[b] == c
                if not posmask.any(): continue
                negmask = ~posmask
                posdis = distance_transform_edt(posmask)
                negdis = distance_transform_edt(negmask)
                sdf[b, c] = negdis - posdis
        return sdf

    def forward(self, probs, gt):
        with torch.no_grad():
            gt_numpy = gt.cpu().numpy()
            sdf_numpy = self.compute_sdf(gt_numpy, probs.shape)
            sdf = torch.from_numpy(sdf_numpy).float().to(probs.device)
        loss = probs * torch.clamp(sdf, min=0)
        return loss.mean()

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, weight=None, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss
        
        if self.alpha is not None:
            at = self.alpha[targets]
            focal_loss = focal_loss * at

        if mask is not None:
            focal_loss = focal_loss * mask
            if self.reduction == 'mean': 
                return focal_loss.sum() / (mask.sum() + 1e-8)
        if self.reduction == 'mean': 
            return focal_loss.mean()
        else: 
            return focal_loss.sum()

class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes, include_background=False):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.include_background = include_background

    def forward(self, pred, target, mask=None):
        pred = torch.softmax(pred, dim=1)
        target_long = target.long()
        target_one_hot = torch.nn.functional.one_hot(target_long, self.num_classes)
        nd = target_one_hot.ndim
        target_one_hot = target_one_hot.permute(0, nd - 1, *range(1, nd - 1)).float()
        if mask is not None:
            mask = mask.unsqueeze(1)
            pred = pred * mask
            target_one_hot = target_one_hot * mask
            
        dims = (0,) + tuple(range(2, pred.ndim))
        tp = torch.sum(pred * target_one_hot, dims)
        fp = torch.sum(pred * (1 - target_one_hot), dims)
        fn = torch.sum((1 - pred) * target_one_hot, dims)
        
        tversky = (tp + 1e-6) / (tp + self.alpha * fp + self.beta * fn + 1e-6)
        
        if not self.include_background:
            return 1 - tversky[1:].mean()
        return 1 - tversky.mean()
