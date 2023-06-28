from functools import partial

import torch
import torch.nn as nn
import numpy as np

from mmdet3d.ops.voxel import DynamicScatter


class DataProcessorPytorch(nn.Module):
    def __init__(self, processor_configs, point_cloud_range, training,
                 pview_ranges, num_pviews):
        super().__init__()
        self.point_cloud_range = point_cloud_range
        self.pview_ranges = pview_ranges
        if not isinstance(self.pview_ranges, np.ndarray):
            self.pview_ranges = np.array(pview_ranges)
        self.training = training
        self.num_pviews = num_pviews
        assert self.num_pviews == len(self.pview_ranges)
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.pview_grid_sizes = self.pview_voxel_sizes = None
        self.data_processor_queue = []
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    # def mask_points_and_boxes_outside_range(self, batch_dict=None, config=None):
    #     if batch_dict is None:
    #         return partial(self.mask_points_and_boxes_outside_range, config=config)
    #     mask = common_utils.mask_points_by_range(batch_dict['points'], self.point_cloud_range)
    #     batch_dict['points'] = batch_dict['points'][mask]
    #     if self.num_pviews > 0:
    #         for i in range(self.num_pviews):
    #             batch_dict[f'points_pviews_{i}'] = batch_dict[f'points_pviews_{i}'][mask]
    #     if batch_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
    #         mask = box_utils.mask_boxes_outside_range_numpy(
    #             batch_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1)
    #         )
    #         batch_dict['gt_boxes'] = batch_dict['gt_boxes'][mask]
    #     return batch_dict
    #
    # def shuffle_points(self, data_dict=None, config=None):
    #     if data_dict is None:
    #         return partial(self.shuffle_points, config=config)
    #
    #     if config.SHUFFLE_ENABLED[self.mode]:
    #         points = data_dict['points']
    #         shuffle_idx = torch.randperm(points.size(0))
    #         points = points[shuffle_idx]
    #         data_dict['points'] = points
    #         if self.num_pviews > 0:
    #             for i in range(self.num_pviews):
    #                 data_dict[f'points_pviews_{i}'] = data_dict[f'points_pviews_{i}'][shuffle_idx]
    #
    #     return batch_dict

    def transform_points_in_pviews_to_voxels(self, batch_dict=None, config=None, voxel_generators=None):
        if batch_dict is None:
            voxel_generators = []
            self.pview_grid_sizes = []
            self.pview_voxel_sizes = []
            voxel_sizes = config.VOXEL_SIZES
            for i in range(self.num_pviews):
                pview_range = self.pview_ranges[i]
                voxel_size = voxel_sizes[i]
                voxel_generator = DynamicScatter(
                    voxel_size=voxel_size,
                    point_cloud_range=pview_range,
                    max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
                    max_num_points=config.MAX_POINTS_PER_VOXEL,
                )
                voxel_generators.append(voxel_generator)
                grid_size = (pview_range[3:6] - pview_range[0:3]) / np.array(voxel_size)
                self.pview_grid_sizes.append(np.round(grid_size).astype(np.int32))
                self.pview_voxel_sizes.append(voxel_size)
            return partial(self.transform_points_in_pviews_to_voxels, voxel_generators=voxel_generators)

        for i in range(len(voxel_generators)):
            voxel_generator = voxel_generators[i]
            points = batch_dict[f'points_pviews_{i}']
            voxels, coordinates, num_points = voxel_generator(points)

            if not batch_dict['use_lead_xyz']:
                voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

            batch_dict[f'pview_voxels_{i}'] = voxels
            batch_dict[f'pview_voxel_coords_{i}'] = coordinates
            batch_dict[f'pview_voxel_num_points_{i}'] = num_points
        return batch_dict

    def transform_points_to_voxels(self, batch_dict=None, config=None, voxel_generator=None):
        if batch_dict is None:

            voxel_generator = DynamicScatter(
                voxel_size=config.VOXEL_SIZE,
                point_cloud_range=self.point_cloud_range,
                max_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
                max_num_points=config.MAX_POINTS_PER_VOXEL,
            )
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(dtype=np.int32)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels, voxel_generator=voxel_generator)

        points = batch_dict['points']
        voxels, coordinates, num_points = voxel_generator(points)

        if not batch_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        batch_dict['voxels'] = voxels
        batch_dict['voxel_coords'] = coordinates
        batch_dict['voxel_num_points'] = num_points
        return batch_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        import time
        start_time = time.time()
        for cur_processor in self.data_processor_queue:
            batch_dict = cur_processor(batch_dict=batch_dict)

        return batch_dict
