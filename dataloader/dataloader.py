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
                 augmentations,
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
        self.augmentations = augmentations
        self.p_aug = cfg['augmentations']['p_aug']
        self.p_flp = cfg['augmentations']['p_flip']
        self.p_crp = cfg['augmentations']['p_crop']
        self.p_rot = cfg['augmentations']['p_rotate']
        self.p_cto = cfg['augmentations']['p_cutout']
        self.p_mix = cfg['augmentations']['p_mixup']
        self.indexes = cfg['data_loader']['indexes']

        self.m_groups_s2 = self.create_s2_groups(cfg)
        self.s2_out_dim = len(s2_bands) * len(self.m_groups_s2)
        if self.indexes:
            n_grps = len(self.m_groups_s2)
            self.band_to_idx_range = {band: (k * n_grps, (k + 1) * n_grps) for k, band in enumerate(s2_bands)}

    def create_s2_groups(self, cfg):
        n_groups = 12 // cfg['data_loader']['s2_avg_by']
        group_size = 12 // n_groups
        groups = [list(range(i * group_size, (i + 1) * group_size)) for i in range(n_groups)]
        return groups

    def drop_bands_with_clouds(self, bands, idxs_to_filter):
        band = 'S2_QA60'
        band_months = [f'{m}_{band}' for m in range(12)]
        idxs = [self.band_to_idx[band] for band in band_months]
        cloud_mask = np.sum(bands[idxs], axis=(1, 2))
        idx_to_drop = np.arange(0, 12)[cloud_mask > 1]
        filtered_idxs = [idx for idx in idxs_to_filter if idx not in idx_to_drop]
        if len(filtered_idxs) == 0:
            filtered_idxs = idxs
        return filtered_idxs

    def fill_s2_bands(self, bands, bands_to_fill):
        i = 0
        for band in self.s2_bands:
            band_min, band_max = self.bands_range[band]
            for group in self.m_groups_s2:
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

    def apply_augmentations(self, sample):
        aug = np.random.choice(['none', 'flip', 'crop', 'rotate', 'cutout', 'mixup'],
                               p=[self.p_aug, self.p_flp, self.p_crp, self.p_rot, self.p_cto, self.p_mix])
        if aug == 'none':
            return sample
        else:
            sample = self.augmentations[aug](sample)
            return sample

    def add_indexes(self, s2_bands):
        idx_dim = len(self.m_groups_s2)
        out_dim = len(self.indexes) * idx_dim
        indexes_out = np.empty((out_dim, 40, 40))
        for k, index in enumerate(self.indexes):
            if index == 'NDVI':
                band_8 = s2_bands[self.band_to_idx_range['S2_B8'][0]: self.band_to_idx_range['S2_B8'][1]]
                band_4 = s2_bands[self.band_to_idx_range['S2_B4'][0]: self.band_to_idx_range['S2_B4'][1]]
                NDVI = (band_8 - band_4) / (band_8 + band_4)
                indexes_out[k * idx_dim: (k + 1) * idx_dim] = NDVI
            elif index == 'NDVIa':
                band_8a = s2_bands[self.band_to_idx_range['S2_B8A'][0]: self.band_to_idx_range['S2_B8A'][1]]
                band_4 = s2_bands[self.band_to_idx_range['S2_B4'][0]: self.band_to_idx_range['S2_B4'][1]]
                NDVIa = (band_8a - band_4) / (band_8a + band_4)
                indexes_out[k * idx_dim: (k + 1) * idx_dim] = NDVIa
            elif index == 'NDWI':
                band_8a = s2_bands[self.band_to_idx_range['S2_B8A'][0]: self.band_to_idx_range['S2_B8A'][1]]
                band_11 = s2_bands[self.band_to_idx_range['S2_B11'][0]: self.band_to_idx_range['S2_B11'][1]]
                NDWI = (band_8a - band_11) / (band_8a + band_11)
                indexes_out[k * idx_dim: (k + 1) * idx_dim] = NDWI
            else:
                raise NotImplementedError('Such index not supported, valid indexes are NDVI, NDVIa, NDWI.')
        s2_bands = np.concatenate([s2_bands, indexes_out], axis=0)
        return s2_bands

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

        if self.indexes:
            s2_bands_grouped = self.add_indexes(s2_bands_grouped)

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
            if self.augmentations and self.mode == 'train':
                sample = self.apply_augmentations(sample)

        return sample
