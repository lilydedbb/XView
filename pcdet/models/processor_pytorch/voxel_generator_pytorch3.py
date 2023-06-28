import torch

from mmdet3d.ops.voxel import Voxelization

import numpy as np


class VoxelGeneratorV2:
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000,
                 full_mean=False,
                 block_filtering=False,
                 block_factor=8,
                 block_size=3,
                 height_threshold=0.1,
                 height_high_threshold=2.0):
        self._max_voxels = max_voxels
        self._max_num_points = max_num_points
        self._voxelization = Voxelization(voxel_size,
                                         point_cloud_range,
                                         max_num_points,
                                         max_voxels=max_voxels)

    def generate(self, points, batch_size):
        num_points_per_voxel = torch.zeros((batch_size*self._max_voxels,), dtype=torch.int32, device=torch.device('cuda'))
        voxels = torch.zeros((batch_size*self._max_voxels, self._max_num_points, points.size(-1) - 1), dtype=points.dtype, device=torch.device('cuda'))
        # voxel_point_mask = torch.zeros((batch_size*self._max_voxels, self._max_num_points), dtype=points.dtype, device=torch.device('cuda'))
        coors = torch.zeros((batch_size*self._max_voxels, 4), dtype=torch.int32, device=torch.device('cuda'))
        # voxel_num = torch.zeros((batch_size, ), dtype=torch.int32, device=torch.device('cuda'))

        assert points.dim() == 2
        for b in range(batch_size):
            batch_points = points[points[:, 0] == b, 1:]
            _voxels, _coors, _num_points_per_voxel = self._voxelization(batch_points)

            voxels[b*self._max_voxels:b*self._max_voxels+_voxels.size(0)] = _voxels
            coors[b*self._max_voxels:b*self._max_voxels+_voxels.size(0), 0] = b
            coors[b*self._max_voxels:b*self._max_voxels+_voxels.size(0), 1:4] = _coors
            num_points_per_voxel[b * self._max_voxels:b*self._max_voxels+_voxels.size(0)] = _num_points_per_voxel

        res = {
            "voxels": voxels,
            "coordinates": coors,
            "num_points_per_voxel": num_points_per_voxel,
            # "voxel_point_mask": voxel_point_mask,
        }
        # res["voxel_num"] = voxel_num
        # res["voxel_point_mask"] = res["voxel_point_mask"].view(-1, self._max_num_points, 1)

        # for k, v in res.items():
        #     if k != "voxel_num":
        #         res[k] = v[:res["voxel_num"]]
        return res

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size
