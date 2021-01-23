import os
import pandas as pd
import numpy as np
from typing import Dict

import torch
from torch.utils.data import Dataset

from sklearn.linear_model import LinearRegression


class BandsYieldDataset(Dataset):
    """
    Load specific bands for field with scaling to [0, 1] range.
    Supports averaging of bands by months and cloud filtration.
    """

    def __init__(self,
                 csv_file_path: str,
                 data_path: str,
                 transforms,
                 augmentations,
                 bands_cfg: Dict,
                 cfg: Dict):

        self.base_df = pd.read_csv(csv_file_path)
        self.mode = 'train' if 'Quality' in self.base_df.columns else 'eval'
        if self.mode == 'train':
            qualities = cfg['data_loader']['qualities']
            self.base_df = self.base_df.query(f'Quality in {qualities}')

        self.data_path = data_path
        self.transforms = transforms
        self.augmentations = augmentations
        self.s2_bands = cfg['data_loader']['s2_bands']
        self.clim_bands = cfg['data_loader']['clim_bands']
        self.all_bands_names = list(bands_cfg['bands_min_max'].keys())
        self.bands_range = bands_cfg['bands_min_max']
        self.filter_clouds = cfg['data_loader']['filter_clouds']
        self.filter_by_QA60 = cfg['data_loader']['filter_by_QA60']
        self.filter_by_threshold = cfg['data_loader']['filter_by_threshold']
        self.p_aug = cfg['augmentations']['p_aug']
        self.p_flp = cfg['augmentations']['p_flip']
        self.p_crp = cfg['augmentations']['p_crop']
        self.p_rot = cfg['augmentations']['p_rotate']
        self.p_cut = cfg['augmentations']['p_cutout']
        self.p_mup = cfg['augmentations']['p_mixup']
        self.indexes = cfg['data_loader']['indexes']
        self.interpolate = cfg['data_loader']['interpolate']

        self.m_groups_s2 = self.create_s2_groups(cfg)
        self.s2_out_dim = (len(self.s2_bands) + len(self.indexes)) * len(self.m_groups_s2)
        if self.indexes:
            n_grps = len(self.m_groups_s2)
            self.band_to_idx_range = {band: (k * n_grps, (k + 1) * n_grps) for k, band in enumerate(self.s2_bands)}

    def create_s2_groups(self, cfg):
        n_groups = 12 // cfg['data_loader']['s2_avg_by']
        group_size = 12 // n_groups
        groups = [list(range(i * group_size, (i + 1) * group_size)) for i in range(n_groups)]
        return groups

    def interpolate_(self, band_data):
        n_months = len(band_data)
        x_gt_idx = np.array(range(n_months))[band_data.sum(axis=(1, 2)) != 0]
        x_prd_idx = np.array([i for i in range(n_months) if i not in x_gt_idx])
        for i in range(40):
            for j in range(40):
                model = LinearRegression()
                x = x_gt_idx.reshape(-1, 1)
                y = band_data[x_gt_idx, i, j].reshape(-1, 1)
                model.fit(x, y)
                y_prd = model.predict(x_prd_idx.reshape(-1, 1)).reshape(1, -1)
                band_data[x_prd_idx, i, j] = y_prd
        return band_data

    def filter_clouds_(self, band_data, bands_dict):
        if self.filter_by_QA60:
            no_clouds = bands_dict['S2_QA60'].max(axis=(1, 2)) == 0.
        if self.filter_by_threshold:
            s2_b2 = (bands_dict['S2_B2'] / 4000).clip(0, 1)
            mask = (s2_b2.sum(axis=(1, 2)) / 1600) < self.filter_by_threshold
            no_clouds = no_clouds * mask
            no_clouds = no_clouds.reshape(12, 1, 1) * np.ones((12, 40, 40), dtype=np.float32)
        band_data = band_data * no_clouds
        if self.interpolate and no_clouds.min() == 0:
            band_data = self.interpolate_(band_data)
        return band_data

    def fill_s2_bands(self, bands_dict):
        i = 0
        bands_to_fill = np.empty((self.s2_out_dim, 40, 40))
        all_bands = self.s2_bands + self.indexes
        for band in all_bands:
            band_data = bands_dict[band]
            if self.filter_clouds:
                band_data = self.filter_clouds_(band_data, bands_dict)
            for group in self.m_groups_s2:
                n_clean = (band_data[group].max(axis=(1, 2)) != 0.).sum()
                n_clean = len(group) if n_clean == 0 else n_clean
                mean_bands = band_data[group].sum(axis=0) / n_clean
                # scale to [0, 1]
                if band not in self.indexes:
                    band_min, band_max = self.bands_range[band]
                    mean_bands = (mean_bands - band_min) / (band_max - band_min)
                bands_to_fill[i, :, :] = mean_bands.clip(0, 1)
                i += 1
        return bands_to_fill

    def fill_clim_bands(self, bands_dict):
        bands_to_fill = np.empty((12, len(self.clim_bands)))
        for k, band in enumerate(self.clim_bands):
            band_min, band_max = self.bands_range[band]
            # mean over spatial dimension
            retrieved_bands = np.mean(bands_dict[band], axis=(1, 2))
            # scale to [0, 1]
            retrieved_bands = (retrieved_bands - band_min) / (band_max - band_min)
            bands_to_fill[:, k] = retrieved_bands.clip(0, 1)
        return bands_to_fill

    def apply_augmentations(self, sample):
        aug = np.random.choice(['none', 'flip', 'crop', 'rotate', 'cutout'],
                               p=[self.p_aug, self.p_flp, self.p_crp, self.p_rot, self.p_cut])
        if aug == 'none':
            return sample
        else:
            sample = self.augmentations[aug](sample)
            return sample

    def add_indexes(self, bands_dict):
        for index_name in self.indexes:
            if index_name == 'NDVI':
                band_8 = (bands_dict['S2_B8'] / 4095).clip(0, 1)
                band_4 = (bands_dict['S2_B4'] / 4095).clip(0, 1)
                index_val = (band_8 - band_4) / (band_8 + band_4)
            elif index_name == 'NDWI':
                band_8a = (bands_dict['S2_B8A'] / 4095).clip(0, 1)
                band_11 = (bands_dict['S2_B11'] / 4095).clip(0, 1)
                index_val = (band_8a - band_11) / (band_8a + band_11)
            elif index_name == 'EVI2':
                band_8 = (bands_dict['S2_B8'] / 4095).clip(0, 1)
                band_4 = (bands_dict['S2_B4'] / 4095).clip(0, 1)
                index_val = 2.5*(band_8 - band_4) / (band_8 + 2.4*band_4 + 1)
            else:
                raise NotImplementedError('Such index not supported, valid indexes are NDVI, NDWI, EVI2.')

            bands_dict[index_name] = index_val

        return bands_dict

    def __len__(self):
        return len(self.base_df)

    def __getitem__(self, idx):

        field_id = self.base_df.iloc[idx]['Field_ID']
        path_to_file = os.path.join(self.data_path, f'{field_id}.npy')
        bands = np.load(path_to_file)

        bands_dict = {}
        for k, band_name in enumerate(self.all_bands_names):
            band_idx = list(range(k, 360, 30))
            bands_dict[band_name] = bands[band_idx, :40, :40]

        if self.indexes:
            bands_dict = self.add_indexes(bands_dict)

        s2_bands_grouped = self.fill_s2_bands(bands_dict)
        clim_bands = self.fill_clim_bands(bands_dict)

        yields = self.base_df.iloc[idx]['Yield'] if self.mode == 'train' else 0

        # to tensors
        s2_bands_grouped = torch.tensor(s2_bands_grouped, dtype=torch.float32)
        clim_bands = torch.tensor(clim_bands, dtype=torch.float32)
        yields = torch.tensor(yields, dtype=torch.float32)

        sample = {
            's2_bands': s2_bands_grouped,
            'clim_bands': clim_bands,
            'yield': yields
        }

        if self.transforms:
            sample = self.transforms(sample)
            if self.augmentations and self.mode == 'train':
                sample = self.apply_augmentations(sample)

        return sample
