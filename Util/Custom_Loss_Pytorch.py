# pytorch深度學習模組套件
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import _WeightedLoss


'''
https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/173733
'''
class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


'''
https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
'''
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


'''
https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
'''
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, smoothing=1):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + self.smoothing)/(inputs.sum() + targets.sum() + self.smoothing)  
        
        return 1 - dice


'''
https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
'''
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True,  smoothing=1):
        super(DiceBCELoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection +  self.smoothing)/(inputs.sum() + targets.sum() +  self.smoothing)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE


'''
https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
'''
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True,  smoothing=1):
        super(IoULoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + self.smoothing)/(union + self.smoothing)
                
        return 1 - IoU