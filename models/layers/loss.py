import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable

def cross_entropy_2D(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss

def ce_loss(logits, true, weights=None, ignore=0):
    """Computes the weighted multi-class cross-entropy loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        weight: a tensor of shape [C,]. The weights attributed
            to each class.
        ignore: the class index to ignore.
    Returns:
        ce_loss: the weighted multi-class cross-entropy loss.
    """

    weights = torch.FloatTensor(weights).to(logits.device) if weights is not None else None
    true = true.squeeze(1)
    try:
        ce_loss = F.cross_entropy(
            logits.float(),
            true.long(),
            ignore_index=ignore,
            weight=weights,
        )
    except:
        print(true.shape, logits.shape)
        raise ValueError('Error in cross entropy loss')
    return ce_loss

class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)

        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)

        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score + ce_loss(input,target)


class CustomSoftDiceLoss(nn.Module):
    def __init__(self, n_classes, class_ids):
        super(CustomSoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.class_ids = class_ids

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)

        input = F.softmax(input[:,self.class_ids], dim=1).view(batch_size, len(self.class_ids), -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        target = target[:, self.class_ids, :]

        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score

class IoU_loss(nn.Module):
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Notes: [Batch size,Num classes,Height,Width]
    Args:
        targs: a tensor of shape [B, H, W] or [B, 1, H, W].
        preds: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model. (prediction)
        eps: added to the denominator for numerical stability.
    Returns:
        iou: the average class intersection over union value 
             for multi-class image segmentation
    """
    def __init__(self,n_classes):
        super(IoU_loss, self).__init__()
        self.num_classes = n_classes
    def forward(self,preds, targs, eps=1e-8):
    # Single class segmentation?
        num_classes = self.num_classes
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[targs.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(preds)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
            
        # Multi-class segmentation
        else:
            # Convert target to one-hot encoding
            # true_1_hot = torch.eye(num_classes)[torch.squeeze(targs,1)]
            true_1_hot = torch.eye(num_classes)[targs.squeeze(1)]
            
            # Permute [B,H,W,C] to [B,C,H,W]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            
            # Take softmax along class dimension; all class probs add to 1 (per pixel)
            probas = F.softmax(preds, dim=1)
            
        true_1_hot = true_1_hot.type(preds.type()).to(preds.device)
        
        # Sum probabilities by class and across batch images
        dims = (0,) + tuple(range(2, targs.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims) # [class0,class1,class2,...]
        cardinality = torch.sum(probas + true_1_hot, dims)  # [class0,class1,class2,...]
        union = cardinality - intersection
        iou = (intersection / (union + eps)).mean()   # find mean of class IoU values
        #manage cases where num of calsses is less than 2  
        if self.num_classes ==3:
            weight =[0.1,0.59,0.9]
        if self.num_classes ==4:
            weight = [0.5,0.5,0.5,0.5]
        else:
            weight = [0.5,0.5]
        return (1-iou) + ce_loss(preds,targs, weights=weight)
class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth).cuda()

    def forward(self, X_in):
        n_dim = X_in.dim()
        output_size = X_in.size() + torch.Size([self.depth])
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element)
        out = Variable(self.ones.index_select(0, X_in)).view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


if __name__ == '__main__':
    from torch.autograd import Variable
    depth=3
    batch_size=2
    encoder = One_Hot(depth=depth).forward
    y = Variable(torch.LongTensor(batch_size, 1, 1, 2 ,2).random_() % depth) # 4 classes,1x3x3 img
    y_onehot = encoder(y)
    x = Variable(torch.randn(y_onehot.size()).float())
    dicemetric = SoftDiceLoss(n_classes=depth)
    dicemetric(x,y)