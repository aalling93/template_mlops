"""

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BBoxRegressionLoss(nn.Module):
    """
    This class implements the Smooth L1 Loss for bounding box regression in object detection models.
    The loss is calculated on the predicted and target bounding boxes. Each box is expected
    to be in the format of (x_center, y_center, width, height).

    The Smooth L1 Loss is less sensitive to outliers than the squared error loss and is defined as:
    - L(x, y) = 0.5 * (x - y)^2 for |x - y| < 1
    - L(x, y) = |x - y| - 0.5 for |x - y| >= 1

    Args:
        reduction (str): Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.
                         The default is 'mean'.

    Returns:
        torch.Tensor: The computed loss.

    Examples:
        >>> loss_fn = BBoxRegressionLoss(reduction='mean')
        >>> # Assuming target and input are [batch_size, 4, feature_map_height, feature_map_width]
        >>> input = torch.randn(3, 4, 32, 32, requires_grad=True)
        >>> target = torch.randn(3, 4, 32, 32)
        >>> loss = loss_fn(input, target)
        >>> loss.backward()
        >>> print(f"Computed Loss: {loss.item()}")
    """

    def __init__(self, reduction="mean"):
        super(BBoxRegressionLoss, self).__init__()
        self.loss_fn = nn.SmoothL1Loss(reduction=reduction)

    def forward(self, input, target):
        # Ensure that the input and target tensors are of the same shape
        assert input.shape == target.shape, "Input and target must have the same shape"

        # Compute the loss
        loss = self.loss_fn(input, target)
        return loss


class ClassificationLoss(nn.Module):
    """
    A classification loss module that can compute either Binary Cross-Entropy Loss with logits
    or Focal Loss for each class separately in a batch of class probability maps. This module
    is designed to handle outputs where each class's importance can be weighted differently,
    especially useful in scenarios with class imbalance.
    """

    def __init__(self, use_focal_loss=False, gamma=2.0, alpha=None, reduction="none"):
        super(ClassificationLoss, self).__init__()
        self.use_focal_loss = use_focal_loss
        self.gamma = gamma
        self.alpha = alpha if alpha is not None else [1.0 for _ in range(2)]  # Assume two classes if not specified
        self.reduction = reduction
        self.eps = 1e-7  # Small epsilon to avoid division by zero

    def forward(self, prediction, target):
        if not self.use_focal_loss:
            # Use Binary Cross-Entropy Loss
            return self.binary_cross_entropy_loss(prediction, target)
        else:
            # Use Focal Loss
            return self.focal_loss(prediction, target)

    def binary_cross_entropy_loss(self, prediction, target):
        losses = []
        for i in range(prediction.shape[1]):
            class_input = prediction[:, i, :, :]
            class_target = target[:, i, :, :]
            loss = F.binary_cross_entropy_with_logits(class_input, class_target, reduction=self.reduction)
            losses.append(loss)
        
        stacked_losses = torch.stack(losses)
        if self.reduction == "mean":
            return stacked_losses.mean()
        elif self.reduction == "sum":
            return stacked_losses.sum()
        return stacked_losses.mean()  # Defaults to mean

    def focal_loss(self, prediction, target):
        losses = []
        for i in range(prediction.shape[1]):
            class_input = prediction[:, i, :, :]
            class_target = target[:, i, :, :]
            p = torch.sigmoid(class_input)
            p_t = torch.where(class_target == 1, p, 1 - p)
            p_t = torch.clamp(p_t, min=self.eps, max=1.0 - self.eps)
            alpha_weight = self.alpha[i]
            class_loss = -alpha_weight * ((1 - p_t) ** self.gamma) * torch.log(p_t)
            
            if self.reduction == "none":
                class_loss = class_loss.sum().sum() / (class_target.sum() + self.eps)
            losses.append(class_loss)
        
        stacked_losses = torch.stack(losses)
        if self.reduction == "mean":
            return stacked_losses.mean()
        elif self.reduction == "sum":
            return stacked_losses.sum()
        return stacked_losses.mean()  # Defaults to mean





class IoULoss(nn.Module):
    r"""
    can handle various types of simple iou losses..

    However, it does not calculate the full advanced ones, for instance, compare the CIoU in this one with the one in CIoULoss.
    inputs are preds and targets of same shape
    M x 4 where M is number of boxes...


    """

    def __init__(self, xywh: bool = True, GIoU=False, DIoU=False, CIoU=True, eps=1e-7):

        super(IoULoss, self).__init__()
        self.xywh = xywh
        self.GIoU = GIoU
        self.DIoU = DIoU
        self.CIoU = CIoU
        self.eps = eps

    def forward(self, preds, targets):
        # Ensure that the input and target tensors are of the same shape
        # target shape: Mx4, so number of targets is M
        num_targets = targets.shape[0]
        assert preds.shape == targets.shape, "Input and target must have the same shape"

        iou_loss = []
        for i in range(preds.shape[0]):

            iou_loss.append(
                1
                - bbox_ious(
                    preds[i].unsqueeze(0), targets[i].unsqueeze(0), xywh=self.xywh, CIoU=self.CIoU, GIoU=self.GIoU, DIoU=self.DIoU, eps=self.eps
                )
            )

        if iou_loss:
            iou_loss = torch.sum(torch.stack(iou_loss)) / num_targets
        else:
            print("iou_loss is empty")
            # Handle the case when iou_loss is empty
            # For instance, you can return a zero loss or any base value that makes sense in your context
            iou_loss = torch.tensor(0.0)

        return iou_loss

