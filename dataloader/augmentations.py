import numpy as np
from torchvision.transforms import transforms


class Flip(object):
    def __init__(self):
        self.h_flip = transforms.RandomHorizontalFlip(p=1)
        self.v_flip = transforms.RandomVerticalFlip(p=1)

    def __call__(self, sample):
        s2_bands = sample['s2_bands']
        if np.random.rand(1) < 0.5:
            sample['s2_bands'] = self.h_flip(s2_bands)
        else:
            sample['s2_bands'] = self.v_flip(s2_bands)
        return sample


class Rotate(object):
    def __init__(self, cfg):
        self.aff_pad = cfg['augmentations']['aff_pad']
        output_size = cfg['transforms']['s2_band_size'][0] + self.aff_pad
        self.scale = transforms.Resize((output_size, output_size))
        self.rand_rot = transforms.RandomRotation(degrees=cfg['augmentations']['degrees'])

    def crop(self, sample):
        crop_size = self.aff_pad // 2
        return sample[:, crop_size:-crop_size, crop_size:-crop_size]

    def __call__(self, sample):
        s2_bands = sample['s2_bands']
        s2_bands = self.scale(s2_bands)
        s2_bands = self.rand_rot(s2_bands)
        s2_bands = self.crop(s2_bands)
        sample['s2_bands'] = s2_bands
        return sample


class Crop(object):
    def __init__(self, cfg):
        pad = np.random.randint(5, 32)
        output_size = cfg['transforms']['s2_band_size'][0] + pad
        crop_size = cfg['transforms']['s2_band_size']
        self.scale = transforms.Resize((output_size, output_size))
        self.rand_crop = transforms.RandomCrop(crop_size, padding=False)

    def __call__(self, sample):
        s2_bands = sample['s2_bands']
        s2_bands = self.scale(s2_bands)
        s2_bands = self.rand_crop(s2_bands)
        sample['s2_bands'] = s2_bands
        return sample
