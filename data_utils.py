import os
import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class BandsYieldDataset(Dataset):
    """
    Load specific bands for field with scaling to [0, 1] range.
    Supports averaging of bands by months and cloud filtration.
    """
    def __init__(self,
                 csv_file_path,
                 data_path,
                 band_to_idx,
                 s2_bands,
                 clim_bands,
                 cfg):

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
        self.transform = cfg['data_loader']['transform']
        self.filter_clouds = cfg['data_loader']['filter_clouds']

        self.m_groups_s2 = self.create_groups('S2', cfg)
        self.m_groups_clim = self.create_groups('CLIM', cfg)

        self.s2_out_dim = len(s2_bands) * len(self.m_groups_s2)
        self.clim_out_dim = len(clim_bands) * len(self.m_groups_clim)

    def create_groups(self, band_type, cfg):
        if band_type == 'S2':
            n_groups = 12 // cfg['data_loader']['s2_avg_by']
        elif band_type == 'CLIM':
            n_groups = 12 // cfg['data_loader']['clim_avg_by']
        group_size = 12 // n_groups
        groups = [list(range(i*group_size, (i + 1)*group_size)) for i in range(n_groups)]
        return groups

    def filter_clouds(self, bands, idxs_to_filter):
        band = 'S2_QA60'
        band_months = [f'{m}_{band}' for m in range(12)]
        idxs = [self.band_to_idx[band] for band in band_months]
        cloud_mask = np.sum(bands[idxs], axis=(1, 2))
        idx_to_drop = np.arange(0, 12)[cloud_mask > 1]
        filtred_idxs = [idx if idx not in idx_to_drop for idx in idxs_to_filter]
        return filtred_idxs


    def fill_bands(self, bands, bands_to_fill, bands_list, band_type):
        if band_type == 'S2':
            groups = self.m_groups_s2
        elif band_type == 'CLIM':
            groups = self.m_groups_clim

        i = 0
        for band in bands_list:
            band_min, band_max = self.bands_range[band]
            for group in groups:
                band_months = [f'{m}_{band}' for m in group]
                idxs = [self.band_to_idx[band] for band in band_months]
                # clouds filtration
                if band_type == 'S2' and self.filter_clouds:
                    idxs = self.filter_clouds(bands, idxs)
                # mean over months
                mean_bands = bands[idxs, :40, :40].mean(axis=0)
                # scale to [0, 1]
                mean_bands = (mean_bands - band_min) / (band_max - band_min)
                bands_to_fill[i, :, :] = mean_bands.clip(0, 1)
                i += 1
        return bands_to_fill

    def __len__(self):
        return len(self.base_df)

    def __getitem__(self, idx):

        field_id = self.base_df.iloc[idx]['Field_ID']
        path_to_file = os.path.join(self.data_path, f'{field_id}.npy')
        bands = np.load(path_to_file)

        s2_bands_grouped = np.empty((self.s2_out_dim, 40, 40))
        clim_bands_grouped = np.empty((self.clim_out_dim, 40, 40))

        s2_bands_grouped = self.fill_bands(bands, s2_bands_grouped, self.s2_bands, 'S2')
        clim_bands_grouped = self.fill_bands(bands, clim_bands_grouped, self.clim_bands, 'CLIM')

        yields = self.base_df.iloc[idx]['Yield'] if self.mode == 'train' else 0


        # to tensors

        sample = {
            's2_bands': s2_bands_grouped,
            'clim_bands': clim_bands_grouped,
            'yield': yields
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_band_to_idx(bands_txt_path):
    band_names = [l.strip() for l in open(bands_txt_path, 'r').readlines()]
    band_to_idx = {band: idx for idx, band in enumerate(band_names)}
    return band_to_idx