import torch
import torch.nn as nn


class PointFeatureEncoderPytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.point_encoding_config = config
        self.main_view_used_feature_list = getattr(self.point_encoding_config, 'MAIN_VIEW_USED_FEATURE_LIST',
                                                   getattr(self.point_encoding_config, 'USED_FEATURE_LIST', None))
        self.main_view_src_feature_list = getattr(self.point_encoding_config, 'MAIN_VIEW_SRC_FEATURE_LIST',
                                                  getattr(self.point_encoding_config, 'SRC_FEATURE_LIST', None))
        assert list(self.main_view_src_feature_list[0:4]) == ['b', 'x', 'y', 'z']
        self.pview_used_feature_lists = getattr(self.point_encoding_config, 'PVIEW_USED_FEATURE_LIST', [])
        self.pview_src_feature_lists = getattr(self.point_encoding_config, 'PVIEW_SRC_FEATURE_LIST', [])

    @property
    def num_point_features(self):
        return getattr(self, self.point_encoding_config.ENCODING_TYPE)(batch_dict=None)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            batch_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """
        batch_dict, use_lead_xyz = getattr(self, self.point_encoding_config.ENCODING_TYPE)(
            batch_dict,
        )
        batch_dict['use_lead_xyz'] = use_lead_xyz
        return batch_dict

    def absolute_coordinates_encoding(self, batch_dict=None):
        if batch_dict is None:
            main_view_num_output_features = len(self.main_view_used_feature_list) - 1
            if len(self.pview_used_feature_lists) > 0:
                pview_num_output_features = [len(x) - 1 for x in self.pview_used_feature_lists]
            else:
                pview_num_output_features = None
            return main_view_num_output_features, pview_num_output_features

        main_view_points_list = []
        for x in self.main_view_used_feature_list:
            idx = self.main_view_src_feature_list.index(x)
            main_view_points_list.append(batch_dict['points'][:, idx:idx + 1])
        batch_dict['points_copy'] = batch_dict['points'].clone()
        batch_dict['points'] = torch.cat(main_view_points_list, dim=1)

        if len(self.pview_used_feature_lists) > 0:
            assert len([x for x in batch_dict.keys() if x.startswith('points_pviews_')]) == len(self.pview_used_feature_lists)
            for i in range(len(self.pview_used_feature_lists)):
                pview_points_list = []
                for x in self.pview_used_feature_lists[i]:
                    if x == '1':
                        pview_points_list.append(torch.ones((batch_dict[f'points_pviews_{i}'].size(0), 1),
                                                         dtype=batch_dict[f'points_pviews_{i}'].dtype, requires_grad=False))
                        continue
                    idx = self.pview_src_feature_lists[i].index(x)
                    pview_points_list.append(batch_dict[f'points_pviews_{i}'][:, idx:idx + 1])
                batch_dict[f'points_pviews_{i}_copy'] = batch_dict[f'points_pviews_{i}'].clone()
                batch_dict[f'points_pviews_{i}'] = torch.cat(pview_points_list, dim=1)

        return batch_dict, True
