import torch

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
        assert full_mean is False, "don't use this."
        point_cloud_range = torch.tensor(point_cloud_range, dtype=torch.float32).cuda()
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32).cuda()
        grid_size = (point_cloud_range[3:] -
                     point_cloud_range[:3]) / voxel_size
        grid_size = torch.round(grid_size).long().cuda()
        if block_filtering:
            assert block_size > 0
            assert grid_size[0] % block_factor == 0
            assert grid_size[1] % block_factor == 0

        voxelmap_shape = grid_size.cpu().numpy().tolist()[::-1]
        self._coor_to_voxelidx = torch.full(voxelmap_shape, -1, dtype=torch.int32).cuda()
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size
        self._full_mean = full_mean
        self._block_filtering = block_filtering
        self._block_factor = block_factor
        self._height_threshold = height_threshold
        self._block_size = block_size
        self._height_high_threshold = height_high_threshold

    def generate(self, points, batch_size):
        num_points_per_voxel = torch.zeros((batch_size*self._max_voxels,), dtype=torch.int32, device=torch.device('cuda'))
        voxels = torch.zeros((batch_size*self._max_voxels, self._max_num_points, points.size(-1)), dtype=points.dtype, device=torch.device('cuda'))
        voxel_point_mask = torch.zeros((batch_size*self._max_voxels, self._max_num_points), dtype=points.dtype, device=torch.device('cuda'))
        coors = torch.zeros((batch_size*self._max_voxels, 3), dtype=torch.int32, device=torch.device('cuda'))
        voxel_num = torch.zeros((batch_size, ), dtype=torch.int32, device=torch.device('cuda'))

        assert points.dim() == 2
        for b in range(batch_size):
            batch_points = points[points[:, 0] == b]
            voxel_idx = torch.floor((batch_points[:, 1:4] - self._point_cloud_range[0:3].unsqueeze(0)) / self._voxel_size).long()
            _tmp = voxel_idx[:, 0] * 1000000 + voxel_idx[:, 1] * 1000 + voxel_idx[:, 2]
            _tmp, sorted_idx = torch.sort(_tmp)
            # sorted_voxel_idx = voxel_idx[sorted_idx]
            sorted_points = batch_points[sorted_idx]
            unique_voxel_idx, counts = torch.unique(voxel_idx, dim=0, return_counts=True, sorted=True)
            cumsum_counts = torch.cumsum(counts, dim=0)
            for i in range(counts.size(0)):
                if i >= self._max_voxels: break
                if counts[i] > self._max_num_points:
                    voxels[b*self._max_voxels+i] = sorted_points[torch.arange(cumsum_counts[i-1] if i > 0 else 0, cumsum_counts[i])[:self._max_num_points]]
                    num_points_per_voxel[b*self._max_voxels+i] = self._max_num_points
                    voxel_point_mask[b*self._max_voxels+i] = 1
                else:
                    voxels[b*self._max_voxels+i, :counts[i]] = sorted_points[torch.arange(cumsum_counts[i-1] if i > 0 else 0, cumsum_counts[i])]
                    num_points_per_voxel[b*self._max_voxels+i] = counts[i]
                    voxel_point_mask[b*self._max_voxels+i, :counts[i]] = 1
                coors[b*self._max_voxels+i] = unique_voxel_idx[i]
            voxel_num[b] = counts.size(0)

        res = {
            "voxels": voxels,
            "coordinates": coors,
            "num_points_per_voxel": num_points_per_voxel,
            "voxel_point_mask": voxel_point_mask,
        }
        # res["voxel_num"] = voxel_num
        res["voxel_point_mask"] = res["voxel_point_mask"].view(-1, self._max_num_points, 1)

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
