import torch
import math
import numbers
import torch.nn.functional as F
import torchvision.transforms.functional as f
import torchvision.transforms as T
import torchvision.transforms.v2 as v2

from torchvision import transforms as _transforms
from typing import List, Optional, Tuple, Union
from scipy import ndimage
from torch import Tensor

from sn_calibration_baseline.evaluate_extremities import mirror_labels

class ToTensor(torch.nn.Module):
    def __call__(self, sample):
        image, target, mask = sample['image'], sample['target'], sample['mask']

        return {'image': f.to_tensor(image).float(),
                'target': torch.from_numpy(target).float(),
                'mask': torch.from_numpy(mask).float()}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.swap_dict = {1:3, 2:2, 3:1, 4:7, 5:6, 6:5, 7:4, 8:11, 9:10, 10:9, 11:8, 12:15, 13:14, 14:13, 15:12,
                          16:19, 17:18, 18:17, 19:16, 20:23, 21:22, 22:21, 23:20, 24:27, 25:26, 26:25, 27:24, 28:30,
                          29:29, 30:28, 31:33, 32:32, 33:31, 34:36, 35:35, 36:34, 37:40, 38:39, 39:38, 40:37, 41:44,
                          42:43, 43:42, 44:41, 45:57, 46:56, 47:55, 48:49, 49:48, 50:52, 51:51, 52:50, 53:54, 54:53,
                          55:47, 56:46, 57:45, 58:58}

    def forward(self, sample):
        if torch.rand(1) < self.p:
            image, target, mask = sample['image'], sample['target'], sample['mask']
            image = f.hflip(image)
            target = f.hflip(target)

            target_swap, mask_swap = self.swap_layers(target, mask)

            return {'image': image,
                    'target': target_swap,
                    'mask': mask_swap}
        else:
            return {'image': sample['image'],
                    'target': sample['target'],
                    'mask': sample['mask']}


    def swap_layers(self, target, mask):
        target_swap = torch.zeros_like(target)
        mask_swap = torch.zeros_like(mask)
        for kp in self.swap_dict.keys():
            kp_swap = self.swap_dict[kp]
            target_swap[kp_swap-1, :, :] = target[kp-1, :, :].clone()
            mask_swap[kp_swap-1] = mask[kp-1].clone()

        return target_swap, mask_swap


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=2.):
        self.std = std
        self.mean = mean

    def __call__(self, sample):
        image = sample['image']
        image += torch.randn(image.size()) * self.std + self.mean
        image = torch.clip(image, 0, 1)

        return {'image': image,
                'target': sample['target'],
                'mask': sample['mask']}

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ColorJitter(torch.nn.Module):

    def __init__(
            self,
            brightness: Union[float, Tuple[float, float]] = 0,
            contrast: Union[float, Tuple[float, float]] = 0,
            saturation: Union[float, Tuple[float, float]] = 0,
            hue: Union[float, Tuple[float, float]] = 0,
    ) -> None:
        super().__init__()
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            value = [float(value[0]), float(value[1])]
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(f"{name} values should be between {bound}, but got {value}.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            return None
        else:
            return tuple(value)

    @staticmethod
    def get_params(
            brightness: Optional[List[float]],
            contrast: Optional[List[float]],
            saturation: Optional[List[float]],
            hue: Optional[List[float]],
    ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h


    def forward(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        image = sample['image']

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                image = f.adjust_brightness(image, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                image = f.adjust_contrast(image, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                image = f.adjust_saturation(image, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                image = f.adjust_hue(image, hue_factor)

        return {'image': image,
                'target': sample['target'],
                'mask': sample['mask']}


    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}"
            f", contrast={self.contrast}"
            f", saturation={self.saturation}"
            f", hue={self.hue})"
        )
        return s



transforms = v2.Compose([
    ToTensor(),
    RandomHorizontalFlip(p=.5),
    ColorJitter(brightness=(0.05), contrast=(0.05), saturation=(0.05), hue=(0.05)),
    #Normalize(mean=[0.37853125, 0.47014403, 0.2679843], std=[0.14201856, 0.16033882, 0.15999168]),
    AddGaussianNoise(0, .1)
])


no_transforms = v2.Compose([
    ToTensor(),
])


