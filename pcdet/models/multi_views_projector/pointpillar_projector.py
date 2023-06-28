import torch
import torch.nn as nn
import numpy as np

class PointPillarMultiViewsProjector(nn.Module):

    def __init__(self, model_cfg, voxel_size, point_cloud_range,
                 pview_centers, pview_ranges, pview_voxel_sizes):
        super().__init__()
        self.cfg = model_cfg
        self.fuse_method = model_cfg.FUSE_METHOD
        self.interpolation = getattr(model_cfg, 'INTERPOLATION', None)
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.pview_centers = pview_centers
        self.pview_ranges = pview_ranges
        self.pview_voxel_sizes = pview_voxel_sizes

    def xy_to_rp(self, x, y):
        r = torch.sqrt(x ** 2 + y ** 2)
        psi = torch.zeros_like(r)
        psi[x > 0] = torch.atan(y[x > 0] / x[x > 0])
        psi[torch.logical_and(x == 0, y >= 0)] = np.pi / 2.
        psi[torch.logical_and(x == 0, y < 0)] = -np.pi / 2.
        mask = torch.logical_and(x < 0, y >= 0)
        psi[mask] = torch.atan(y[mask] / x[mask]) + np.pi
        mask = torch.logical_and(x < 0, y < 0)
        psi[mask] = torch.atan(y[mask] / x[mask]) - np.pi
        return r, psi

    def cart_to_cylindrical(self, coords, cylindrical_view_centers,
                            cylindrical_point_cloud_range, cylindrical_voxel_size):
        z_, y_, x_ = coords[:, 1], coords[:, 2], coords[:, 3]
        y = y_.float() * self.voxel_size[1] + self.point_cloud_range[1] - cylindrical_view_centers[1]
        x = x_.float() * self.voxel_size[0] + self.point_cloud_range[0] - cylindrical_view_centers[0]
        r, psi= self.xy_to_rp(x, y)
        r = (r - cylindrical_point_cloud_range[0]) / cylindrical_voxel_size[0]
        psi = (psi - cylindrical_point_cloud_range[1]) / cylindrical_voxel_size[1]
        brpz_ = torch.stack((coords[:, 0], r, psi, z_), dim=-1)
        return brpz_

    def forward(self, batch_dict):
        for i in range(len(self.pview_centers)):
            bzyx_ = batch_dict['voxel_coords']
            brpt_ = getattr(self, f'cart_to_{batch_dict["pview_coordinates"][0][i]}')(bzyx_, self.pview_centers[i],
                                                                             self.pview_ranges[i], self.pview_voxel_sizes[i])
            pview_spatial_features = batch_dict[f'pview_spatial_features_{i}']

            if self.interpolation is None:
                brpt_ = brpt_.long()
                features_pview = pview_spatial_features[brpt_[:, 0], :, brpt_[:, 2], brpt_[:, 1]]
            elif self.interpolation == 'linear':
                brpt_floor = brpt_.long().unsqueeze(dim=0)
                brpt_ceil = brpt_floor + 1
                brpt_ceil[:, :, 1:] = torch.min(brpt_ceil[:, :, 1:], torch.tensor(pview_spatial_features.size()[-1:1:-1]).cuda() - 1)[0]
                brpt_tmp = torch.cat((brpt_floor, brpt_ceil), dim=0)
                brpt_float_part = brpt_.unsqueeze(dim=0) - brpt_floor
                brpt_one_minus_float_part = 1 - brpt_float_part
                weight_tmp = torch.cat((brpt_one_minus_float_part, brpt_float_part), dim=0)
                for _j, j in enumerate([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
                          [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1]]):
                    b, r, phi, theta = brpt_tmp[j[0], :, 0], brpt_tmp[j[1], :, 1], brpt_tmp[j[2], :, 2], brpt_tmp[j[3], :, 3]
                    r_weight, phi_weight, theta_weight = weight_tmp[j[1], :, 1:2], weight_tmp[j[2], :, 2:3], weight_tmp[j[3], :, 3:4]
                    if _j == 0 :
                        features_pview = r_weight * phi_weight * theta_weight * pview_spatial_features[b, theta, phi, r]
                    else:
                        features_pview += r_weight * phi_weight * theta_weight * pview_spatial_features[b, theta, phi, r]
            else:
                raise NotImplementedError

            if self.fuse_method == 'add':
                bzyx_ = bzyx_.long()
                batch_dict['spatial_features'][bzyx_[:, 0], :, bzyx_[:, 2], bzyx_[:, 1]] = \
                    batch_dict['spatial_features'][bzyx_[:, 0], :, bzyx_[:, 2], bzyx_[:, 1]] + features_pview
            elif self.fuse_method == 'cat':
                # batch_dict['spatial_features'] = torch.cat((batch_dict['spatial_features'], features_pview), dim=1)
                raise NotImplementedError
            else:
                raise NotImplementedError

        return batch_dict
