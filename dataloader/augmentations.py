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
        self.pad = cfg['augmentations']['rotate']['pad']
        output_size = cfg['transforms']['s2_band_size'][0] + self.pad
        self.scale = transforms.Resize((output_size, output_size))
        self.rand_rot = transforms.RandomRotation(degrees=cfg['augmentations']['rotate']['degrees'])

    def crop(self, sample):
        crop_size = self.pad // 2
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
        pad_max = cfg['augmentations']['crop']['pad']
        pad = np.random.randint(1, pad_max)
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


class CutOut(object):
    def __init__(self, cfg):
        self.n_cuts_max = cfg['augmentations']['cutout']['n_cuts_max']
        self.cut_size = cfg['augmentations']['cutout']['cut_size']

    def __call__(self, sample):
        s2_bands = sample['s2_bands']
        _, w, h = s2_bands.shape
        n_cuts = np.random.randint(1, self.n_cuts_max)
        crop_coords = np.random.randint(0, max(w, h), size=(n_cuts, 2))
        for cx, cy in crop_coords:
            s2_bands[:, cx: cx + self.cut_size, cy: cy + self.cut_size] = 0
        sample['s2_bands'] = s2_bands
        return sample


class MixUp(object):
    def __init__(self, cfg):
        self.alpha = cfg['augmentations']['mixup']['alpha']

    def __call__(self, sample):
        s2_bands = sample['s2_bands']
        yields = sample['yield']

        alpha = np.random.beta(self.alpha, self.alpha)
        shuffled_idxs = np.random.permutation(np.arange(s2_bands.shape[0]))

        s2_mixup = alpha * s2_bands + (1 - alpha) * s2_bands[shuffled_idxs]
        yields_mixup = alpha * yields + (1 - alpha) * yields[shuffled_idxs]

        sample['s2_bands'] = s2_mixup
        sample['yields'] = yields_mixup
        return sample
