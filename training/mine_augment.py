import kornia as K
from kornia.augmentation import *
from kornia.geometry.transform import translate, scale, shear, rotate
import torch
import random



class Translation(object):
    def __init__(self, ratio=0.25, choice='both'):
        self.ratio = ratio
        self.choice = choice

    def __call__(self, tensor_in: torch.tensor):
        assert tensor_in.ndim == 3  # [C, H, W]
        assert tensor_in.dtype == torch.float32
        _, h, w = tensor_in.shape
        _, h, w = tensor_in.shape
        max_distance_x = self.ratio * w
        max_distance_y = self.ratio * h
        x = random.uniform(-max_distance_x, max_distance_x)
        y = random.uniform(-max_distance_y, max_distance_y)
        translate_dict = {
            'x': torch.tensor([[x, 0]]),
            'y': torch.tensor([[0, y]]),
            'both': torch.tensor([[x, y]]),
        }
        translate_factor = translate_dict[self.choice]
        tensor_out = translate(tensor_in.unsqueeze(0), translate_factor, padding_mode='reflection').squeeze(0)

        return tensor_out


class Zoom(object):
    def __init__(self, min_in_ratio=0.8, max_out_ratio=1.2, choice='in'):
        self.min_in_ratio = min_in_ratio
        self.max_out_ratio = max_out_ratio
        self.choice = choice

    def __call__(self, tensor_in: torch.tensor):
        assert tensor_in.ndim == 3  # [C, H, W]
        assert tensor_in.dtype == torch.float32

        in_ratio_x = random.uniform(1, self.max_out_ratio)
        in_ratio_y = random.uniform(1, self.max_out_ratio)
        out_ratio_x = random.uniform(self.min_in_ratio, 1)
        out_ratio_y = random.uniform(self.min_in_ratio, 1)

        scale_dict = {
            'in': torch.tensor([[in_ratio_x, in_ratio_y], ]),
            'out': torch.tensor([[out_ratio_x, out_ratio_y]]),
        }

        scale_factor = scale_dict[self.choice]
        tensor_out = scale(tensor_in.unsqueeze(0), scale_factor, padding_mode='reflection').squeeze(0)

        return tensor_out


class Shear(object):
    def __init__(self, min_angle=0.15, max_angle=0.2):
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, tensor_in: torch.tensor):
        assert tensor_in.ndim == 3  # [C, H, W]
        assert tensor_in.dtype == torch.float32

        shear_x = random.uniform(self.min_angle, self.max_angle)
        shear_y = random.uniform(self.min_angle, self.max_angle)

        shear_factor = torch.tensor([[shear_x, shear_y]])

        tensor_out = shear(tensor_in.unsqueeze(0), shear_factor, padding_mode='reflection').squeeze(0)

        return tensor_out



class Rotate(object):
    def __init__(self, angle = None, min_angle = -15, max_angle =15):
        self.angle = angle
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.i = 0
    
    def __call__(self, tensor_in : torch.tensor):
        assert tensor_in.ndim == 3 # [C, H, W]
        assert tensor_in.dtype == torch.float32
        
        self.angle = random.uniform(self.min_angle, self.max_angle)

        self.angle = torch.tensor(self.angle)

        tensor_out = rotate(tensor_in.unsqueeze(0), self.angle, padding_mode='reflection').squeeze(0)

        return tensor_out
