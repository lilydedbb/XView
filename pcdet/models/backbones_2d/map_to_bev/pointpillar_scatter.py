import torch
import torch.nn as nn
import numpy as np


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg, grid_size, pview_grid_sizes, **kwargs):
        super().__init__()

        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.nx, self.ny, self.nz = grid_size
        assert self.nz == 1

        if pview_grid_sizes is not None:
            self.pview_nxs = np.array(pview_grid_sizes)[:, 0]
            self.pview_nys = np.array(pview_grid_sizes)[:, 1]
            self.pview_nzs = np.array(pview_grid_sizes)[:, 2]
            for pview_nz in self.pview_nzs:
                assert pview_nz == 1

    def get_spatial_features_from_pillar_features(self, pillar_features, coords, grid_size):
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        nx, ny, nz = grid_size
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                nz * nx * ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features * nz, ny, nx)
        return batch_spatial_features

    def forward(self, batch_dict, **kwargs):
        batch_dict['spatial_features'] = \
            self.get_spatial_features_from_pillar_features(
                batch_dict['pillar_features'], batch_dict['voxel_coords'], [self.nx, self.ny, self.nz])

        num_pviews = len([key for key in batch_dict.keys()
                                 if key.startswith('pview_pillar_features_')])
        for i in range(num_pviews):
            batch_dict[f'pview_spatial_features_{i}'] = \
                self.get_spatial_features_from_pillar_features(
                    batch_dict[f'pview_pillar_features_{i}'], batch_dict[f'pview_voxel_coords_{i}'],
                    [self.pview_nxs[i], self.pview_nys[i], self.pview_nzs[i]])

        return batch_dict
