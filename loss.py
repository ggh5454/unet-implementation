import torch
import torch.nn as nn
import torch.nn.functional as F

class EnergyLoss(nn.Module):
    def __init__(self):
        super(EnergyLoss, self).__init__()

    def forward(self, logits, target):
        softmax2d = torch.nn.Softmax2d()
        # logits is of shape [batch_size, feature_channels, height, width]
        logits = softmax2d(logits)
        m = nn.CrossEntropyLoss()
        
        # At this point:
        # logits.size() == [batch_size, feature_channels, height, width]
        # target.size() == [batch_size, feature_channels, height, width]
        loss = m(logits, target)
        return loss
    
    
def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()
    
    
class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()