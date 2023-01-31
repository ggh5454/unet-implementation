import numpy as np
import torch


def IoU(y_pred, y_true):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    return intersection / union


def pixel_accuracy(y_pred, y_true):
    with torch.no_grad():
        correct = torch.eq(y_pred, y_true).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

# def mIoU(pred_mask, mask, smooth=1e-10, n_classes=23):
#     with torch.no_grad():
#         pred_mask = F.softmax(pred_mask, dim=1)
#         pred_mask = torch.argmax(pred_mask, dim=1)
#         pred_mask = pred_mask.contiguous().view(-1)
#         mask = mask.contiguous().view(-1)

#         iou_per_class = []
#         for clas in range(0, n_classes): #loop per pixel class
#             true_class = pred_mask == clas
#             true_label = mask == clas

#             if true_label.long().sum().item() == 0: #no exist label in this loop
#                 iou_per_class.append(np.nan)
#             else:
#                 intersect = torch.logical_and(true_class, true_label).sum().float().item()
#                 union = torch.logical_or(true_class, true_label).sum().float().item()

#                 iou = (intersect + smooth) / (union +smooth)
#                 iou_per_class.append(iou)
#         return np.nanmean(iou_per_class)




# mask에서 1과 0 말고 여러개 있어서 아래 코드 사용 중지
# def iou(pred, target, n_classes = 2):
#     # n_classes : 하나의 클래스의 경우 2
#     iou = []
#     pred = pred.view(-1)
#     target = target.view(-1)

#     # Ignore IoU for background class ("0")
#     for cls in range(1, n_classes):
#         pred_inds = pred == cls
#         target_inds = target == cls
#         intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()
#         print(intersection)
#         union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection

#         if union == 0:
#             iou.append(float('nan'))  # If there is no ground truth, do not include in evaluation
#         else:
#             iou.append(float(intersection) / float(max(union, 1)))

#     return sum(iou)