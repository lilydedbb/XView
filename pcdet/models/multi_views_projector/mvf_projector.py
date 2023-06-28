import torch
import torch.nn as nn
import numpy as np


class MVFProjector(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range,
                 pview_centers, pview_ranges, pview_voxel_sizes):
        super().__init__()
        self.cfg = model_cfg
        self.fuse_method = model_cfg.FUSE_METHOD
        self.model_for = getattr(model_cfg, 'MODEL_FOR', None)
        self.interpolation = getattr(model_cfg, 'INTERPOLATION', None)
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.pview_centers = pview_centers
        self.pview_ranges = pview_ranges
        self.pview_voxel_sizes = pview_voxel_sizes

    def xyz_to_rpt(self, x, y, z):
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = torch.acos(z / torch.sqrt(x ** 2 + y ** 2 + z ** 2))
        psi = torch.zeros_like(r).cuda()
        # print(psi.size(), x.size(), y.size())
        psi[x > 0] = torch.atan(y[x > 0] / x[x > 0])
        psi[torch.logical_and(x == 0, y >= 0)] = np.pi / 2.
        psi[torch.logical_and(x == 0, y < 0)] = -np.pi / 2.
        mask = torch.logical_and(x < 0, y >= 0)
        psi[mask] = torch.atan(y[mask] / x[mask]) + np.pi
        mask = torch.logical_and(x < 0, y < 0)
        psi[mask] = torch.atan(y[mask] / x[mask]) - np.pi
        return r, psi, theta

    def xyz_to_rpz(self, x, y, z):
        r = torch.sqrt(x ** 2 + y ** 2)
        psi = torch.zeros_like(r)
        psi[x > 0] = torch.atan(y[x > 0] / x[x > 0])
        psi[torch.logical_and(x == 0, y >= 0)] = np.pi / 2.
        psi[torch.logical_and(x == 0, y < 0)] = -np.pi / 2.
        mask = torch.logical_and(x < 0, y >= 0)
        psi[mask] = torch.atan(y[mask] / x[mask]) + np.pi
        mask = torch.logical_and(x < 0, y < 0)
        psi[mask] = torch.atan(y[mask] / x[mask]) - np.pi
        return r, psi, z

    def cart_to_spherical(self, coords, spherical_view_centers,
                                     spherical_point_cloud_range, spherical_voxel_size):
        z_, y_, x_ = coords[:, 1], coords[:, 2], coords[:, 3]
        z = ((z_ * 20).float() + 10) * self.voxel_size[2] + self.point_cloud_range[2] - spherical_view_centers[2]
        y = ((y_ * 8).float() + 4) * self.voxel_size[1] + self.point_cloud_range[1] - spherical_view_centers[1]
        x = ((x_ * 8).float() + 4) * self.voxel_size[0] + self.point_cloud_range[0] - spherical_view_centers[0]
        r, psi, theta = self.xyz_to_rpt(x, y, z)
        r = (r - spherical_point_cloud_range[0]) / spherical_voxel_size[0]
        psi = (psi - spherical_point_cloud_range[1]) / spherical_voxel_size[1]
        theta = (theta - spherical_point_cloud_range[2]) / spherical_voxel_size[2]
        r_ = r / 8
        theta_ = theta / 16
        psi_ = psi / 8
        brpt_ = torch.stack((coords[:, 0], r_, psi_, theta_), dim=-1)
        return brpt_

    def cart_to_cylindrical(self, coords, cylindrical_view_centers,
                                     cylindrical_point_cloud_range, cylindrical_voxel_size):
        z_, y_, x_ = coords[:, 1], coords[:, 2], coords[:, 3]
        z = ((z_ * 20).float() + 10) * self.voxel_size[2] + self.point_cloud_range[2] - cylindrical_view_centers[2]
        y = ((y_ * 8).float() + 4) * self.voxel_size[1] + self.point_cloud_range[1] - cylindrical_view_centers[1]
        x = ((x_ * 8).float() + 4) * self.voxel_size[0] + self.point_cloud_range[0] - cylindrical_view_centers[0]
        r, psi, z = self.xyz_to_rpz(x, y, z)
        r = (r - cylindrical_point_cloud_range[0]) / cylindrical_voxel_size[0]
        psi = (psi - cylindrical_point_cloud_range[1]) / cylindrical_voxel_size[1]
        z_ = (z - cylindrical_point_cloud_range[2]) / cylindrical_voxel_size[2]
        r_ = r / 8
        psi_ = psi / 8
        z_ = z_ / 16
        brpz_ = torch.stack((coords[:, 0], r_, psi_, z_), dim=-1)
        return brpz_

    def cart_to_spherical_for_second(self, coords, spherical_view_centers,
                                     spherical_point_cloud_range, spherical_voxel_size):
        return self.cart_to_spherical(coords, spherical_view_centers,
                                     spherical_point_cloud_range, spherical_voxel_size)

    def cart_to_spherical_for_pvrcnn(self, coords, spherical_view_centers,
                                     spherical_point_cloud_range, spherical_voxel_size):
        return self.cart_to_spherical(coords, spherical_view_centers,
                                     spherical_point_cloud_range, spherical_voxel_size)

    def cart_to_spherical_for_pointpillar(self, coords, spherical_view_centers,
                                     spherical_point_cloud_range, spherical_voxel_size):
        return self.cart_to_spherical(coords, spherical_view_centers,
                                     spherical_point_cloud_range, spherical_voxel_size)

    def cart_to_cylindrical_for_second(self, coords, cylindrical_view_centers,
                                     cylindrical_point_cloud_range, cylindrical_voxel_size):
        return self.cart_to_cylindrical(coords, cylindrical_view_centers,
                                     cylindrical_point_cloud_range, cylindrical_voxel_size)

    def cart_to_cylindrical_for_pvrcnn(self, coords, cylindrical_view_centers,
                                     cylindrical_point_cloud_range, cylindrical_voxel_size):
        return self.cart_to_cylindrical(coords, cylindrical_view_centers,
                                     cylindrical_point_cloud_range, cylindrical_voxel_size)

    def cart_to_cylindrical_for_pointpillar(self, coords, cylindrical_view_centers,
                                     cylindrical_point_cloud_range, cylindrical_voxel_size):
        return self.cart_to_cylindrical(coords, cylindrical_view_centers,
                                     cylindrical_point_cloud_range, cylindrical_voxel_size)

    def cart_to_spherical_for_parta2(self, coords, spherical_view_centers, spherical_point_cloud_range, spherical_voxel_size):
        z_, y_, x_ = coords[:, 1], coords[:, 2], coords[:, 3]
        z = ((z_ * 20).float() + 10) * self.voxel_size[2] + self.point_cloud_range[2] - spherical_view_centers[2]
        y = ((y_ * 8).float() + 4) * self.voxel_size[1] + self.point_cloud_range[1] - spherical_view_centers[1]
        x = ((x_ * 8).float() + 4) * self.voxel_size[0] + self.point_cloud_range[0] - spherical_view_centers[0]
        r, psi, theta = self.xyz_to_rpt(x, y, z)
        r = (r - spherical_point_cloud_range[0]) / spherical_voxel_size[0]
        psi = (psi - spherical_point_cloud_range[1]) / spherical_voxel_size[1]
        theta = (theta - spherical_point_cloud_range[2]) / spherical_voxel_size[2]
        # theta = torch.clamp((theta - spherical_point_cloud_range[2]),
        #                     min=spherical_point_cloud_range[2],
        #                     max=spherical_point_cloud_range[5]) / spherical_voxel_size[2]
        r_ = r / 8
        theta_ = theta / 16
        psi_ = psi / 8
        brpt_ = torch.stack((coords[:, 0], r_, psi_, theta_), dim=-1)
        return brpt_

    def forward(self, batch_dict):
        print(batch_dict.keys())
        for key in batch_dict.keys():
            try:
                print(batch_dict[key].size())
            except:
                pass
        print(a)
        sparse_cart_coord_features = batch_dict['encoded_spconv_tensor']
        bzyx_ = sparse_cart_coord_features.indices

        pview_keys = [key for key in batch_dict.keys()
                                 if key.startswith('pview_encoded_spconv_tensor_') and key.find('stride') < 0]

        for i in range(len(pview_keys)):
            sparse_pview_coord_features = batch_dict[f'pview_encoded_spconv_tensor_{i}']
            dense_pview_coord_features = sparse_pview_coord_features.dense()
            brpt_ = getattr(self, f'cart_to_{batch_dict["pview_coordinates"][0][i]}_for_{self.model_for}')(bzyx_, self.pview_centers[i],
                                                                             self.pview_ranges[i], self.pview_voxel_sizes[i])
            # batch_dict[f'cart_to_spherical_map_{i}'] = brpt_
            if self.interpolation is None:
                brpt_ = brpt_.long()
                features_pview = dense_pview_coord_features[brpt_[:, 0], :, brpt_[:, 3], brpt_[:, 2], brpt_[:, 1]]
            elif self.interpolation == 'linear':
                brpt_floor = brpt_.long().unsqueeze(dim=0)
                brpt_ceil = brpt_floor + 1
                brpt_ceil[:, :, 1:] = torch.min(brpt_ceil[:, :, 1:], torch.tensor(dense_pview_coord_features.size()[-1:1:-1]).cuda() - 1)[0]
                brpt_tmp = torch.cat((brpt_floor, brpt_ceil), dim=0)
                brpt_float_part = brpt_.unsqueeze(dim=0) - brpt_floor
                brpt_one_minus_float_part = 1 - brpt_float_part
                weight_tmp = torch.cat((brpt_one_minus_float_part, brpt_float_part), dim=0)
                for _j, j in enumerate([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
                          [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1]]):
                    b, r, phi, theta = brpt_tmp[j[0], :, 0], brpt_tmp[j[1], :, 1], brpt_tmp[j[2], :, 2], brpt_tmp[j[3], :, 3]
                    r_weight, phi_weight, theta_weight = weight_tmp[j[1], :, 1:2], weight_tmp[j[2], :, 2:3], weight_tmp[j[3], :, 3:4]
                    if _j == 0 :
                        features_pview = r_weight * phi_weight * theta_weight * dense_pview_coord_features[b, :, theta, phi, r]
                    else:
                        features_pview += r_weight * phi_weight * theta_weight * dense_pview_coord_features[b, :, theta, phi, r]
            else:
                raise NotImplementedError

            if self.fuse_method == 'add':
                batch_dict['encoded_spconv_tensor'].features = batch_dict['encoded_spconv_tensor'].features + features_pview
            elif self.fuse_method == 'cat':
                batch_dict['encoded_spconv_tensor'].features = torch.cat((batch_dict['encoded_spconv_tensor'].features, features_pview), dim=1)
            else:
                raise NotImplementedError

        return batch_dict

