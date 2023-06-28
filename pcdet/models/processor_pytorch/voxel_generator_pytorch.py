# import torch
# import torch.nn as nn
#
# from mmdet3d.ops.voxel import DynamicScatter
#
# import numpy as np
#
#
# class VoxelGeneratorV2(nn.Module):
#     def __init__(self,
#                  voxel_size,
#                  point_cloud_range,
#                  max_num_points,
#                  max_voxels=20000):
#         super().__init__()
#         self.max_voxels = max_voxels
#         self.max_num_points = max_num_points
#         self.voxel_size = voxel_size
#         self.point_cloud_range = point_cloud_range
#         self._dynamic_scatter = DynamicScatter(voxel_size,
#                                                point_cloud_range,
#                                                max_voxels,
#                                                max_num_points)
#
#     def forward(self, points):
#         voxels, coors, num_points_per_voxel = self._dynamic_scatter(points)
#         print('e', voxels.requires_grad)
#
#         res = {
#             "voxels": voxels,
#             "coordinates": coors,
#             "num_points_per_voxel": num_points_per_voxel,
#             # "voxel_point_mask": voxel_point_mask,
#         }
#         # res["voxel_num"] = voxel_num
#         # res["voxel_point_mask"] = res["voxel_point_mask"].view(-1, self.max_num_points, 1)
#
#         # for k, v in res.items():
#         #     if k != "voxel_num":
#         #         res[k] = v[:res["voxel_num"]]
#         return res
#
#     # @property
#     # def grid_size(self):
#     #     return self._grid_size
