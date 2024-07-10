import math
import torch
import numbers
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as f
import torchvision.transforms.v2 as v2

from torchvision.transforms.functional import _interpolation_modes_from_int, InterpolationMode
from torchvision import transforms as _transforms
from typing import List, Optional, Tuple, Union
from scipy import ndimage
from torch import Tensor

from sn_calibration_baseline.evaluate_extremities import mirror_labels



class ToTensor(torch.nn.Module):
    def __call__(self, sample):
        image, target = sample['image'], sample['data']


        return {'image': f.to_tensor(image).float(),
                'data': sample['data']}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, sample):
        image = sample['image']
        image = f.normalize(image, self.mean, self.std)

        return {'image': image,
                'data': sample['data']}


    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


FLIP_POSTS = {
    'Goal left post right': 'Goal left post left ',
    'Goal left post left ': 'Goal left post right',
    'Goal right post right': 'Goal right post left',
    'Goal right post left': 'Goal right post right'
}

h_lines = ['Goal left crossbar', 'Side line left', 'Small rect. left main', 'Big rect. left main', 'Middle line',
                   'Big rect. right main', 'Small rect. right main', 'Side line right', 'Goal right crossbar']
v_lines = ['Side line top', 'Big rect. left top', 'Small rect. left top', 'Small rect. left bottom',
                   'Big rect. left bottom', 'Big rect. right top', 'Small rect. right top', 'Small rect. right bottom',
                              'Big rect. right bottom', 'Side line bottom']

def swap_top_bottom_names(line_name: str) -> str:
    x: str = 'top'
    y: str = 'bottom'
    if x in line_name or y in line_name:
        return y.join(part.replace(y, x) for part in line_name.split(x))
    return line_name


def swap_posts_names(line_name: str) -> str:
    if line_name in FLIP_POSTS:
        return FLIP_POSTS[line_name]
    return line_name


def flip_annot_names(annot, swap_top_bottom: bool = True, swap_posts: bool = True):
    annot = mirror_labels(annot)
    if swap_top_bottom:
        annot = {swap_top_bottom_names(k): v for k, v in annot.items()}
    if swap_posts:
        annot = {swap_posts_names(k): v for k, v in annot.items()}
    return annot


class LRAmbiguityFix(torch.nn.Module):
    def __init__(self, v_th, h_th):
        super().__init__()
        self.v_th = v_th
        self.h_th = h_th

    def forward(self, sample):
        data = sample['data']

        if len(data) == 0:
            return {'image': sample['image'],
                    'data': sample['data']}

        n_left, n_right = self.compute_n_sides(data)

        angles_v, angles_h = [], []
        for line in data.keys():
            line_points = []
            for point in data[line]:
                line_points.append((point['x'], point['y']))

            sorted_points = sorted(line_points, key=lambda point: (point[0], point[1]))
            pi, pf = sorted_points[0], sorted_points[-1]
            if line in h_lines:
                angle_h = self.calculate_angle_h(pi[0], pi[1], pf[0], pf[1])
                if angle_h:
                    angles_h.append(abs(angle_h))
            if line in v_lines:
                angle_v = self.calculate_angle_v(pi[0], pi[1], pf[0], pf[1])
                if angle_v:
                    angles_v.append(abs(angle_v))


        if len(angles_h) > 0 and len(angles_v) > 0:
            if np.mean(angles_h) < self.h_th and np.mean(angles_v) < self.v_th:
                if n_right > n_left:
                    data = flip_annot_names(data, swap_top_bottom=False, swap_posts=False)

        return {'image': sample['image'],
                'data': data}

    def calculate_angle_h(self, x1, y1, x2, y2):
        if not x2 - x1 == 0:
            slope = (y2 - y1) / (x2 - x1)
            angle = math.atan(slope)
            angle_degrees = math.degrees(angle)
            return angle_degrees
        else:
            return None
    def calculate_angle_v(self, x1, y1, x2, y2):
        if not x2 - x1 == 0:
            slope = (y2 - y1) / (x2 - x1)
            angle = math.atan(1 / slope) if slope != 0 else math.pi / 2  # Avoid division by zero
            angle_degrees = math.degrees(angle)
            return angle_degrees
        else:
            return None

    def compute_n_sides(self, data):
        n_left, n_right = 0, 0
        for line in data:
            line_words = line.split()[:3]
            if 'left' in line_words:
                n_left += 1
            elif 'right' in line_words:
                n_right += 1
        return n_left, n_right

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(v_th={self.v_th}, h_th={self.h_th})"


class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, sample):
        if torch.rand(1) < self.p:
            image, data = sample['image'], sample['data']
            image = f.hflip(image)
            data = flip_annot_names(data)
            for line in data:
                for point in data[line]:
                    point['x'] = 1.0 - point['x']

            return {'image': image,
                    'data': data}
        else:
            return {'image': sample['image'],
                    'data': sample['data']}

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
                'data': sample['data']}

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
                'data': sample['data']}


    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}"
            f", contrast={self.contrast}"
            f", saturation={self.saturation}"
            f", hue={self.hue})"
        )
        return s


class Resize(torch.nn.Module):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        self.size = size

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation

    def forward(self, sample):
        image = sample["image"]
        image = f.resize(image, self.size, self.interpolation)

        return {'image': image,
                'data': sample['data']}


    def __repr__(self):
        interpolate_str = self.interpolation.value
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)




transforms = v2.Compose([
    ToTensor(),
    RandomHorizontalFlip(p=.5),
    ColorJitter(brightness=(0.05), contrast=(0.05), saturation=(0.05), hue=(0.05)),
    #Normalize(mean=[0.37853125, 0.47014403, 0.2679843], std=[0.14201856, 0.16033882, 0.15999168]),
    AddGaussianNoise(0, .1)
])

transforms_w_LR = v2.Compose([
    ToTensor(),
    RandomHorizontalFlip(p=.5),
    LRAmbiguityFix(v_th=70, h_th=20),
    ColorJitter(brightness=(0.05), contrast=(0.05), saturation=(0.05), hue=(0.05)),
    AddGaussianNoise(0, .1)
])

no_transforms = v2.Compose([
    ToTensor(),
])

no_transforms_w_LR = v2.Compose([
    ToTensor(),
    LRAmbiguityFix(v_th=70, h_th=20)
])
