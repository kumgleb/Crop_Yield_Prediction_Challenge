import os
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt


def get_band_to_idx(bands_txt_path: str):
    band_names = [l.strip() for l in open(bands_txt_path, 'r').readlines()]
    band_to_idx = {band: idx for idx, band in enumerate(band_names)}
    return band_to_idx


def load_np_data(file_id: str,
                 images_path: str):
    path_to_file = os.path.join(images_path, f'{file_id}.npy')
    images = np.load(path_to_file)
    return images


def plot_band(images: np.array,
              band: str,
              band_to_idx: Dict):
    band_months = [f'{m}_{band}' for m in range(12)]
    idxs = [band_to_idx[band] for band in band_months]
    fig, ax = plt.subplots(4, 3, figsize=(12, 8))
    fig.subplots_adjust(hspace = .5, wspace=.001)
    ax = ax.ravel()
    for i in range(12):
        img = images[idxs[i]] / 4000
        img = img.clip(0, 1)
        ax[i].imshow(img, vmin=0, vmax=1, cmap='cividis')
        ax[i].set_title(f'{band}, month: {i}')
        ax[i].set_axis_off()
    fig.tight_layout()

  
def plot_sample(sample, cfg_data):
    groups = cfg_data['data_loader']['s2_avg_by']
    n = 12 // groups
    bands = sample['s2_bands'][:12, :, :]
    fig, ax = plt.subplots(3, 4, figsize=(12, 8))
    ax = ax.ravel()
    for i in range(n):
      ax[i].imshow(bands[i, :, :], vmin=0, vmax=1, cmap='bone')
      ax[i].set_axis_off()