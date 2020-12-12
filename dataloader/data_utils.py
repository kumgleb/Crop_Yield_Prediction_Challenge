import toml
import numpy as np

from torchvision.transforms import transforms
from .transforms import Resize, Normalize
from .augmentations import Rotate, Crop, Flip


def get_band_to_idx(bands_txt_path):
    band_names = [l.strip() for l in open(bands_txt_path, 'r').readlines()]
    band_to_idx = {band: idx for idx, band in enumerate(band_names)}
    return band_to_idx


def get_s2_mean_std(s2_bands, cfg):
    s2_stats = toml.load('s2_bands_stat.toml')
    n_groups = 12 // cfg['data_loader']['s2_avg_by']
    mean = np.array([[s2_stats['mean'][band]] * n_groups for band in s2_bands]).reshape(-1)
    std = np.array([[s2_stats['std'][band]] * n_groups for band in s2_bands]).reshape(-1)
    return mean, std


def compose_transforms(s2_bands, cfg):
    mean, std = get_s2_mean_std(s2_bands, cfg)
    normalize = Normalize(mean, std)
    resize = Resize(cfg['transforms']['s2_band_size'])
    transforms_dict = {'normalize': normalize,
                       'resize': resize}
    dataloader_transforms = transforms.Compose(
        [transforms_dict[transform] for transform in cfg['transforms']['s2_transforms']])
    return dataloader_transforms


def create_augmentations(cfg):
    assert (cfg['augmentations']['p_aug'] + cfg['augmentations']['p_flip'] +
            + cfg['augmentations']['p_rotate'] + cfg['augmentations'][
                'p_crop']) == 1, 'Probabilities should sum up to 1.'

    augmentations = {'flip': Flip(),
                     'rotate': Rotate(cfg),
                     'crop': Crop(cfg)}
    return augmentations
