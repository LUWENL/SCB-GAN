# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import random

import kornia as K
from kornia.augmentation import *
from torchvision import transforms
from .mine_augment import *

p = 1
size = [512, 512]
grayscale = False

aug_dict = {
    'Translation_x': Translation(choice='x'),
    'Translation_y': Translation(choice='y'),
    'Translation_both': Translation(choice='both'),
    'Zoom_in': Zoom(choice='in'),
    'Zoom_out': Zoom(choice='out'),
    'Shear': Shear(),
    'Rotate': Rotate(),
    'FlipRot_Horizontal': [RandomHorizontalFlip(p=p, keepdim=True), Rotate(angle=90.0)],
    'FlipRot_Vertical': [RandomVerticalFlip(p=p, keepdim=True), Rotate(angle=90.0)],
    'RandomPlanckianJitter': RandomPlanckianJitter(mode='CIED', p=p, keepdim=True),
    'RandomPlasmaShadow': RandomPlasmaShadow(roughness=(0.1, 0.7), p=p, keepdim=True),
    'RandomPlasmaBrightness': RandomPlasmaBrightness(roughness=(0.1, 0.7), p=p, keepdim=True),
    'RandomPlasmaContrast': RandomPlasmaContrast(roughness=(0.1, 0.7), p=p, keepdim=True),
    'ColorJiggle': ColorJiggle(0.1, 0.1, 0.1, 0.1, p=p, keepdim=True),
    'RandomBoxBlur': RandomBoxBlur((7, 7), keepdim=True),
    'RandomChannelShuffle': RandomChannelShuffle(p=p, keepdim=True),
    'RandomGaussianBlur': RandomGaussianBlur((3, 3), (0.1, 2.0), p=p, keepdim=True),
    'RandomGaussianNoise': RandomGaussianNoise(mean=0., std=0.015, p=p, keepdim=True),
    'RandomMotionBlur': RandomMotionBlur(3, 35., 0.5, p=p, keepdim=True),
    'RandomPosterize': RandomPosterize(3, p=p, keepdim=True),
    'RandomRGBShift': RandomRGBShift(p=p, keepdim=True),
    'RandomSharpness': RandomSharpness(1, p=p, keepdim=True),
    'RandomSolarize': RandomSolarize(0.1, 0.1, p=p, keepdim=True),
    'RandomAffine': RandomAffine((-15, 15), padding_mode='reflection', p=p, keepdim=True),
    'RandomElasticTransform': RandomElasticTransform(p=p, keepdim=True),
    'HorizontalFlip': RandomHorizontalFlip(p=p, keepdim=True),
    'VerticalFlip': RandomVerticalFlip(p=p, keepdim=True),
    'RandomInvert': RandomInvert(p=p, keepdim=True),
    'RandomResizedCrop': RandomResizedCrop(size=size, p=p, keepdim=True),
    'RandomThinPlateSpline': RandomThinPlateSpline(p=p, keepdim=True),
}


def tensor_transform_reverse(image_tensor):
    assert len(image_tensor.shape) == 3

    tensor = torch.zeros(image_tensor.size()).type_as(image_tensor)
    tensor = image_tensor

    image_np = (tensor * 255).numpy().astype(np.uint8)
    return image_np


def aug_image(img_in: np.ndarray, aug):
    assert img_in.ndim == 3
    assert img_in.dtype == np.uint8
    tensor_in = torch.tensor(img_in / 255).to(torch.float32)  # [0,255] -> [0,1]
    assert tensor_in.shape[0] == 1 or tensor_in.shape[0] == 3  # tensor_in [C, H, W]

    if isinstance(aug, list):
        for aug_ in aug:
            tensor_out = aug_(tensor_in)
    else:
        tensor_out = aug(tensor_in)

    img_out = tensor_transform_reverse(tensor_out)  # ndarray [C, H, W] range[0, 255]
    return img_out


try:
    import pyspng
except ImportError:
    pyspng = None


# ----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 name,  # Name of the dataset.
                 raw_shape,  # Shape of the raw image data (NCHW).
                 max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
                 use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
                 xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
                 random_seed=0,  # Random seed to use when applying max_size.
                 mine_aug=True, # use Mineaug to make training robust default = True
                 aug_list=['HorizontalFlip', 'RandomAffine','Rotate'], # Data augmentations selected by MineAug
                 ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
        self.xflip = xflip
        self.mine_aug = mine_aug
        self.aug_list = aug_list

        if self.mine_aug:
            self.probs = [1 for i in range(len(self.aug_list))]

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

        # Apply MineAug
        self._mine_aug = np.zeros(self._raw_idx.size, dtype=np.uint8)
        self.tmp_raw_idx = self._raw_idx

        if self.mine_aug:
            for i in range(len(self.aug_list)):
                len_ = int(len(self.tmp_raw_idx) * self.probs[i])

                self._mine_aug = np.concatenate([self._mine_aug, np.zeros(len_, dtype=np.uint8) + (i + 1)])

                random.shuffle(self.tmp_raw_idx)
                self._raw_idx = np.append(self._raw_idx, self.tmp_raw_idx[:len_])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self.xflip:
            if self._xflip[idx]:
                assert image.ndim == 3  # CHW
                image = image[:, :, ::-1]
        if self.mine_aug:
            if self._mine_aug[idx] != 0:
                aug_choice = self._mine_aug[idx] - 1
                aug_key = self.aug_list[aug_choice]
                aug_value = aug_dict[aug_key]
                image = aug_image(image, aug_value)

        # return "image", self.get_label(idx)
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------
