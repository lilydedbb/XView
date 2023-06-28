import numpy as np


class PointFeatureEncoder(object):
    def __init__(self, config, point_cloud_range=None):
        super().__init__()
        self.point_encoding_config = config
        self.main_view_used_feature_list = getattr(self.point_encoding_config, 'main_view_used_feature_list',
                                                   getattr(self.point_encoding_config, 'used_feature_list', None))
        self.main_view_src_feature_list = getattr(self.point_encoding_config, 'main_view_src_feature_list',
                                                  getattr(self.point_encoding_config, 'src_feature_list', None))
        assert list(self.main_view_src_feature_list[0:3]) == ['x', 'y', 'z']
        self.pview_used_feature_lists = getattr(self.point_encoding_config, 'pview_used_feature_lists', [])
        self.pview_src_feature_lists = getattr(self.point_encoding_config, 'pview_src_feature_lists', [])
        self.point_cloud_range = point_cloud_range

    @property
    def num_point_features(self):
        return getattr(self, self.point_encoding_config.encoding_type)(data_dict=None)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """
        data_dict, use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
            data_dict,
        )
        data_dict['use_lead_xyz'] = use_lead_xyz
        return data_dict

    def absolute_coordinates_encoding(self, data_dict=None):
        if data_dict is None:
            main_view_num_output_features = len(self.main_view_used_feature_list)
            if len(self.pview_used_feature_lists) > 0:
                pview_num_output_features = [len(x) for x in self.pview_used_feature_lists]
            else:
                pview_num_output_features = None
            return main_view_num_output_features, pview_num_output_features

        main_view_points_list = []
        for x in self.main_view_used_feature_list:
            idx = self.main_view_src_feature_list.index(x)
            main_view_points_list.append(data_dict['points'][:, idx:idx + 1])
        data_dict['points_copy'] = data_dict['points'].copy()
        data_dict['points'] = np.concatenate(main_view_points_list, axis=1)

        if len(self.pview_used_feature_lists) > 0:
            assert len(data_dict['points_pviews']) == len(self.pview_used_feature_lists)
            data_dict['points_pviews_copy'] = []
            for i in range(len(self.pview_used_feature_lists)):
                pview_points_list = []
                for x in self.pview_used_feature_lists[i]:
                    if x == '1':
                        pview_points_list.append(np.ones((data_dict['points_pviews'][i].shape[0], 1),
                                                         dtype=data_dict['points_pviews'][i].dtype))
                        continue
                    idx = self.pview_src_feature_lists[i].index(x)
                    pview_points_list.append(data_dict['points_pviews'][i][:, idx:idx + 1])
                data_dict['points_pviews_copy'].append(data_dict['points_pviews'][i].copy())
                data_dict['points_pviews'][i] = np.concatenate(pview_points_list, axis=1)

        return data_dict, True
