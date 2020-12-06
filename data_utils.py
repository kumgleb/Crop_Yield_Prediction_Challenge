import os
import pandas as pd
import numpy as np
from typing import Dict

import torch
from torch.utils.data import Dataset


class BandsYieldDataset(Dataset):
    """
    Load specific bands for field with scaling to [0, 1] range.
    Supports averaging of bands by months and cloud filtration.
    """
    def __init__(self,
                 csv_file_path: str,
                 data_path: str,
                 band_to_idx: Dict,
                 s2_bands: list,
                 clim_bands: list,
                 transforms,
                 cfg: Dict):

        self.base_df = pd.read_csv(csv_file_path)
        self.mode = 'train' if 'Quality' in self.base_df.columns else 'eval'
        if self.mode == 'train':
            qualities = cfg['data_loader']['qualities']
            self.base_df = self.base_df.query(f'Quality in {qualities}')

        self.data_path = data_path
        self.band_to_idx = band_to_idx
        self.s2_bands = s2_bands
        self.clim_bands = clim_bands
        self.bands_range = cfg['bands_min_max']
        self.filter_clouds = cfg['data_loader']['filter_clouds']
        self.transforms = transforms

        self.m_groups_s2 = self.create_s2_groups('S2', cfg)
        self.s2_out_dim = len(s2_bands) * len(self.m_groups_s2)

    def create_s2_groups(self, band_type, cfg):
        n_groups = 12 // cfg['data_loader']['s2_avg_by']
        group_size = 12 // n_groups
        groups = [list(range(i*group_size, (i + 1)*group_size)) for i in range(n_groups)]
        return groups

    def drop_bands_with_clouds(self, bands, idxs_to_filter):
        band = 'S2_QA60'
        band_months = [f'{m}_{band}' for m in range(12)]
        idxs = [self.band_to_idx[band] for band in band_months]
        cloud_mask = np.sum(bands[idxs], axis=(1, 2))
        idx_to_drop = np.arange(0, 12)[cloud_mask > 1]
        filtred_idxs = [idx for idx in idxs_to_filter if idx not in idx_to_drop]
        if len(filtred_idxs) == 0:
          filtred_idxs = idxs
        return filtred_idxs

    def fill_s2_bands(self, bands, bands_to_fill):
        i = 0
        for band in self.s2_bands:
            band_min, band_max = self.bands_range[band]
            for group in self.m_groups_s2 :
                band_months = [f'{m}_{band}' for m in group]
                idxs = [self.band_to_idx[band] for band in band_months]
                # clouds filtration
                if self.filter_clouds:
                    idxs = self.drop_bands_with_clouds(bands, idxs)
                # mean over months
                mean_bands = bands[idxs, :40, :40].mean(axis=0)
                # scale to [0, 1]
                mean_bands = (mean_bands - band_min) / (band_max - band_min)
                bands_to_fill[i, :, :] = mean_bands.clip(0, 1)
                i += 1
        return bands_to_fill

    def fill_clim_bands(self, bands, bands_to_fill):
        for k, band in enumerate(self.clim_bands):
          band_min, band_max = self.bands_range[band]
          band_months = [f'{m}_{band}' for m in range(12)]
          idxs = [self.band_to_idx[band] for band in band_months]
          # mean over spatial dimension
          retrieved_bands = np.mean(bands[idxs], axis=(1, 2))
          # scale to [0, 1]
          retrieved_bands = (retrieved_bands - band_min) / (band_max - band_min)
          bands_to_fill[k, :] = retrieved_bands.clip(0, 1)
        return bands_to_fill

    def __len__(self):
        return len(self.base_df)

    def __getitem__(self, idx):

        field_id = self.base_df.iloc[idx]['Field_ID']
        path_to_file = os.path.join(self.data_path, f'{field_id}.npy')
        bands = np.load(path_to_file)

        s2_bands_grouped = np.empty((self.s2_out_dim, 40, 40))
        clim_bands_grouped = np.empty((len(self.clim_bands), 12))

        s2_bands_grouped = self.fill_s2_bands(bands, s2_bands_grouped)
        clim_bands_grouped = self.fill_clim_bands(bands, clim_bands_grouped)

        yields = self.base_df.iloc[idx]['Yield'] if self.mode == 'train' else 0

        # to tensors
        s2_bands_grouped = torch.tensor(s2_bands_grouped, dtype=torch.float32)
        clim_bands_grouped = torch.tensor(clim_bands_grouped, dtype=torch.float32)
        yields = torch.tensor(yields, dtype=torch.float32)

        sample = {
            's2_bands': s2_bands_grouped,
            'clim_bands': clim_bands_grouped,
            'yield': yields
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample


def get_band_to_idx(bands_txt_path):
    band_names = [l.strip() for l in open(bands_txt_path, 'r').readlines()]
    band_to_idx = {band: idx for idx, band in enumerate(band_names)}
    return band_to_idx
