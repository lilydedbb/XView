import torch

from .vfe_template import VFETemplate


class MeanVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, num_pviews, num_point_features_pviews, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
        self.num_pviews = num_pviews
        self.num_point_features_pviews = num_point_features_pviews

    def get_output_feature_dim(self):
        return self.num_point_features

    def get_output_feature_dim_pviews(self):
        return self.num_point_features_pviews

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        points_mean = points_mean / normalizer
        batch_dict['voxel_features'] = points_mean.contiguous()

        if self.num_pviews > 0:
            assert self.num_pviews == len(self.num_point_features_pviews)
            for i in range(self.num_pviews):
                voxel_features, voxel_num_points = batch_dict[f'pview_voxels_{i}'], batch_dict[f'pview_voxel_num_points_{i}']
                points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
                normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
                points_mean = points_mean / normalizer
                batch_dict[f'pview_voxel_features_{i}'] = points_mean.contiguous()

        return batch_dict
