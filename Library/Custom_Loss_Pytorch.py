# pytorch深度學習模組套件
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import _WeightedLoss


'''
https://zhuanlan.zhihu.com/p/75542467
https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/173733
'''
class CrossEntropyLoss(nn.Module):
    def __init__(self, class_num, alpha=None, use_alpha=False, size_average=True):
        super(CrossEntropyLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):
        prob = self.softmax(pred.view(-1,self.class_num))
        prob = prob.clamp(min=0.0001,max=1.0)

        target_ = torch.zeros(target.size(0),self.class_num).cuda()
        target_.scatter_(1, target.view(-1, 1).long(), 1.)

        if self.use_alpha:
            batch_loss = - self.alpha.double() * prob.log().double() * target_.double()
        else:
            batch_loss = - prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        # print(prob[0],target[0],target_[0],batch_loss[0])
        # print('--')

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

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
https://zhuanlan.zhihu.com/p/75542467
'''
# 針對二分類任務的 Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        # 如果模型最後沒有 nn.Sigmoid()，那麼這裡就需要對預測結果計算一次 Sigmoid 操作
        # pred = nn.Sigmoid()(pred)

        # 展開 pred 和 target,此時 pred.size = target.size = (BatchSize,1)
        pred = pred.view(-1,1)
        target = target.view(-1,1)

        # 此處將預測樣本為正負的概率都計算出來，此時 pred.size = (BatchSize,2)
        pred = torch.cat((1-pred,pred),dim=1)

        # 根據 target 生成 mask，即根據 ground truth 選擇所需概率
        # 用大白話講就是：
        # 當標籤為 1 時，我們就將模型預測該樣本為正類的概率代入公式中進行計算
        # 當標籤為 0 時，我們就將模型預測該樣本為負類的概率代入公式中進行計算
        class_mask = torch.zeros(pred.shape[0],pred.shape[1]).cuda()
        # 這裡的 scatter_ 操作不常用，其函數原型為:
        # scatter_(dim,index,src)->Tensor
        # Wr​​ites all values from the tensor src into self at the indices specified in the index tensor.
        # For each value in src, its output index is specified by its index in src for dimension != dim and by the corresponding value in index for dimension = dim.
        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        # 利用 mask 將所需概率值挑選出來
        probs = (pred * class_mask).sum(dim=1).view(-1,1)
        probs = probs.clamp(min=0.0001,max=1.0)

        # 計算概率的 log 值
        log_p = probs.log()

        # 根據論文中所述，對 alpha　進行設置（該參數用於調整正負樣本數量不均衡帶來的問題）
        alpha = torch.ones(pred.shape[0],pred.shape[1]).cuda()
        alpha[:,0] = alpha[:,0] * (1-self.alpha)
        alpha[:,1] = alpha[:,1] * self.alpha
        alpha = (alpha * class_mask).sum(dim=1).view(-1,1)

        # 根據 Focal Loss 的公式計算 Loss
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

         # Loss Function的常規操作，mean 與 sum 的區別不大，相當於學習率設置不一樣而已
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

# 針對多分類任務的Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, use_alpha=False, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):
        prob = self.softmax(pred.view(-1,self.class_num))
        prob = prob.clamp(min=0.0001,max=1.0)

        target_ = torch.zeros(target.size(0),self.class_num).cuda()
        target_.scatter_(1, target.view(-1, 1).long(), 1.)

        if self.use_alpha:
            batch_loss = - self.alpha.double() * torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

# 針對多標籤任務的 Focal Loss
class FocalLoss_MultiLabel(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss_MultiLabel, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        criterion = FocalLoss(self.alpha,self.gamma,self.size_average)
        loss = torch.zeros(1,target.shape[1]).cuda()

        # 對每個 Label 計算一次 Focal Loss
        for label in range(target.shape[1]):
            batch_loss = criterion(pred[:,label],target[:,label])
            loss[0,label] = batch_loss.mean()

        # Loss Function的常規操作
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss


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