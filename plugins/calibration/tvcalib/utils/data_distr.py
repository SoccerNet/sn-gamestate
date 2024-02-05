import torch


def mean_std_with_confidence_interval(
    vmin, vmax, sigma_scale: float, _steps=1000, round_decimals=4
):
    """Computes mean and std given min,max values with respect a confidence interval (sigma_scale).

    sigma_scale = 1.65 -> 90% of samples are in range [vmin, vmax]
    sigma_scale = 1.96 -> 95% of samples are in range [vmin, vmax]
    sigma_scale = 2.58 -> 99% of samples are in range [vmin, vmax]
    """

    # sample from uniform distribution
    x = torch.linspace(vmin, vmax, _steps)
    mu = x.mean(dim=-1)
    sigma = x.std(dim=-1)
    return (round(mu.item(), round_decimals), round((sigma * sigma_scale).item(), round_decimals))


class FeatureScalerZScore(torch.nn.Module):
    def __init__(self, loc: float, scale: float) -> None:
        # Transforms data from distribution parameterized by loc (mean) and scale (=sigma*scaling factor).
        super(FeatureScalerZScore, self).__init__()

        self.loc = loc
        self.scale = scale

    def forward(self, z):
        """
        Args:
            z (Tensor): tensor of size (B, *) to be denormalized.
        Returns:
            x: tensor.
        """
        return self.denormalize(z)

    def denormalize(self, z):
        x = z * self.scale + self.loc
        return x

    def normalize(self, x):
        z = (x - self.loc) / self.scale
        return z
