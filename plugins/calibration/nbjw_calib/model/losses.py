import torch
import torch.nn as nn
import torch.nn.functional as F

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, output, target, mask=None):
        loss = self.criterion(output, target)
        if mask is not None:
            loss = (loss * mask).mean()
        else:
            loss = (loss).mean()
        return loss

class KLDivLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, output, target, mask=None):
        if mask is not None:
            output_masked = output * mask
            target_masked = target * mask
            loss = self.criterion(F.log_softmax(output_masked), target_masked)
        else:
            loss = self.criterion(F.log_softmax(output), target)
        return loss

class HeatmapWeightingMSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, output, target, mask=None):
        """Forward function."""
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0.

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(1)
            heatmap_gt = heatmaps_gt[idx].squeeze(1)
            """
            Set different weight generation functions.
            weight = heatmap_gt + 1
            weight = heatmap_gt * 2 + 1
            weight = heatmap_gt * heatmap_gt + 1
            weight = torch.exp(heatmap_gt + 1)
            """

            if mask is not None:
                #weight = heatmap_gt * mask[:, idx] + 1
                weight = torch.exp(heatmap_gt * mask[:, idx] + 1)
                loss += torch.mean(self.criterion(heatmap_pred * mask[:, idx],
                                                    heatmap_gt * mask[:, idx]) * weight)
            else:
                weight = heatmap_gt + 1
                loss += torch.mean(self.criterion(heatmap_pred, heatmap_gt) * weight)
        return loss / (num_joints+1)


class CombMSEAW(nn.Module):
    def __init__(self, lambda1=1, lambda2=1, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        # Adaptive wing loss
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.criterion1 = nn.MSELoss(reduction='none')
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta


    def forward(self, pred, target, mask=None):
        loss = 0
        if mask is not None:
            pred_masked, target_masked = pred * mask, target * mask
            loss += self.lambda1 * self.criterion1(pred_masked, target_masked)
            loss += self.lambda2 * self.adaptive_wing(pred_masked, target_masked)
        else:
            loss += self.lambda1 * self.criterion1(pred, target)
            loss += self.lambda2 * self.adaptive_wing(pred, target)
        return torch.mean(loss)

    def adaptive_wing(self, pred, target):
        delta = (target - pred).abs()
        alpha_t = self.alpha - target
        A = self.omega * (
                1 / (1 + torch.pow(self.theta / self.epsilon,
                                   alpha_t))) * alpha_t \
            * (torch.pow(self.theta / self.epsilon,
                         self.alpha - target - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, alpha_t))

        losses = torch.where(delta < self.theta,
                             self.omega * torch.log(
                                 1 + torch.pow(delta / self.epsilon, alpha_t)),
                             A * delta - C)
        return losses



class AdaptiveWingLoss(nn.Module):
    def __init__(self, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        # Adaptive wing loss
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta

    def forward(self, pred, target, mask=None):
        if mask is not None:
            pred_masked, target_masked = pred * mask, target * mask
            loss = self.adaptive_wing(pred_masked, target_masked)
        else:
            loss = self.adaptive_wing(pred, target)
        return loss

    def adaptive_wing(self, pred, target):
        delta = (target - pred).abs()
        alpha_t = self.alpha - target
        A = self.omega * (
                1 / (1 + torch.pow(self.theta / self.epsilon,
                                   alpha_t))) * alpha_t \
            * (torch.pow(self.theta / self.epsilon,
                         self.alpha - target - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, alpha_t))

        losses = torch.where(delta < self.theta,
                             self.omega * torch.log(
                                 1 + torch.pow(delta / self.epsilon, alpha_t)),
                             A * delta - C)
        return torch.mean(losses)

class GaussianFocalLoss(nn.Module):
    """GaussianFocalLoss is a variant of focal loss.
    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_
    Code is modified from `kp_utils.py
    <https://github.com/princeton-vl/CornerNet/blob/master/models/py_utils/kp_utils.py#L152>`_  # noqa: E501
    Please notice that the target in GaussianFocalLoss is a gaussian heatmap,
    not 0/1 binary target.
    Args:
        alpha (float): Power of prediction.
        gamma (float): Power of target for negative samples.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                mask=None,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        if mask is not None:
            pred_masked, target_masked = pred * mask, target * mask
            loss_reg = self.loss_weight * self.gaussian_focal_loss(pred_masked, target_masked, alpha=self.alpha,
                                                                   gamma=self.gamma)
        else:
            loss_reg = self.loss_weight * self.gaussian_focal_loss(pred, target, alpha=self.alpha, gamma=self.gamma)
        return loss_reg.mean()

    def gaussian_focal_loss(self, pred, gaussian_target, alpha=2.0, gamma=4.0):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
        distribution.
        Args:
            pred (torch.Tensor): The prediction.
            gaussian_target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 2.0.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 4.0.
        """
        eps = 1e-12
        pos_weights = gaussian_target.eq(1)
        neg_weights = (1 - gaussian_target).pow(gamma)
        pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
        neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
        return pos_loss + neg_loss