import torch
import torch.nn as nn


class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, weight_map, target, mask):
        y = target * mask.bool().unsqueeze(2).unsqueeze(3)
        y_hat = pred * mask.bool().unsqueeze(2).unsqueeze(3)

        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]

        loss1 = self.omega * torch.log(1 + torch.pow(
            delta_y1 / self.epsilon, self.alpha - y1)) * weight_map[delta_y < self.theta]

        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * \
            torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))

        loss2 = (A * delta_y2 - C) * weight_map[delta_y >= self.theta]

        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))



class AdaptiveWingLoss_l(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss_l, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, weight_map, target):
        y = target
        y_hat = pred

        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]

        loss1 = self.omega * torch.log(1 + torch.pow(
            delta_y1 / self.epsilon, self.alpha - y1)) * weight_map[delta_y < self.theta]

        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * \
            torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))

        loss2 = (A * delta_y2 - C) * weight_map[delta_y >= self.theta]

        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

