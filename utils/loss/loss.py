import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from utils.losses.lovasz_losses import lovasz_softmax
from torch.nn.modules.loss import _Loss, _WeightedLoss
from torch.nn import NLLLoss2d


__all__ = ["CrossEntropyLoss2d", "CrossEntropyLoss2dLabelSmooth",
           "FocalLoss2d", "LDAMLoss", "ProbOhemCrossEntropy2d",
           "LovaszSoftmax"]

class CrossEntropyLoss2d(_WeightedLoss):
    """
    Standard pytorch weighted nn.CrossEntropyLoss
    """
    def __init__(self, weight=None, ignore_label=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()

        self.nll_loss = nn.CrossEntropyLoss(weight, ignore_index=ignore_label, reduction=reduction)

    def forward(self, output, target):
        """
        Forward pass, average over pixels then average over batches
        :param output: torch.tensor (NxCxWxH)
        :param target: torch.tensor (NxWxH)
        :return: scalar
        """
        return self.nll_loss(output, target)

class CrossEntropyLoss2dLabelSmooth(_WeightedLoss):
    """
    Refer from https://arxiv.org/pdf/1512.00567.pdf
    :param target: N,
    :param n_classes: int
    :param eta: float
    :return:
        N x C onehot smoothed vector
    """

    def __init__(self, weight=None, ignore_label=255, epsilon=0.1, reduction='mean'):
        super(CrossEntropyLoss2dLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.nll_loss = nn.CrossEntropyLoss(weight, ignore_index=ignore_label, reduction=reduction)

    def forward(self, output, target):
        """
        Forward pass
        :param output: torch.tensor (NxC)
        :param target: torch.tensor (N)
        :return: scalar
        """
        n_classes = output.size(1)
        # batchsize, num_class = input.size()
        # log_probs = F.log_softmax(inputs, dim=1)
        targets = torch.zeros_like(output).scatter_(1, target.unsqueeze(1), 1) # NxCxWxH Nx1xWxH (dim, index, value) -> NxCxWxH (one hot)
        targets = (1 - self.epsilon) * targets + self.epsilon / n_classes

        return self.nll_loss(output, targets)

"""
https://arxiv.org/abs/1708.02002
# Credit to https://github.com/clcarwin/focal_loss_pytorch
"""
class FocalLoss2d(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255, size_average=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.size_average = size_average
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, output, target):

        if output.dim()>2:
            output = output.contiguous().view(output.size(0), output.size(1), -1) #NxCx(W*H)
            output = output.transpose(1,2) # channel last
            output = output.contiguous().view(-1, output.size(2)).squeeze()

        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1) # 
        else:
            target = target.view(-1, 1)
        
        logpt = self.ce_fn(output, target) # -log(1-p) y =0 or -log(p) y=1
        pt = torch.exp(-logpt) # 1-p or p
        loss = ((1-pt) ** self.gamma) * self.alpha * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
# ==========================================================================================================================
# ==========================================================================================================================
# class-balanced loss
class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255, use_weight=True):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.use_weight   = use_weight
        # if self.use_weight:
        #     self.weight = torch.FloatTensor(
        #         [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
        #          0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
        #          1.0865, 1.1529, 1.0507])
        #     print('CrossEntropy2d weights : {}'.format(self.weight))
        # else:
        #     self.weight = None


    def forward(self, predict, target, weight=None):

        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        # Variable(torch.randn(2,10)
        if self.use_weight:
            print('target size {}'.format(target.shape))
            freq = np.zeros(19)
            for k in range(19):
                mask = (target[:, :, :] == k)
                freq[k] = torch.sum(mask)
                print('{}th frequency {}'.format(k, freq[k]))
            weight = freq / np.sum(freq)
            print(weight)
            self.weight = torch.FloatTensor(weight)
            print('Online class weight: {}'.format(self.weight))
        else:
            self.weight = None


        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_label)
        # torch.FloatTensor([2.87, 13.19, 5.11, 37.98, 35.14, 30.9, 26.23, 40.24, 6.66, 32.07, 21.08, 28.14, 46.01, 10.35, 44.25, 44.9, 44.25, 47.87, 40.39])
        #weight = Variable(torch.FloatTensor([1, 1.49, 1.28, 1.62, 1.62, 1.62, 1.64, 1.62, 1.49, 1.62, 1.43, 1.62, 1.64, 1.43, 1.64, 1.64, 1.64, 1.64, 1.62]), requires_grad=False).cuda()
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous() # channel last
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = criterion(predict, target)
        return loss
# ==========================================================================================================================
# ==========================================================================================================================