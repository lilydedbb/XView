import numpy as np


class PerspectiveViewGenerator(object):
    def __init__(self, view_centers=None, coordinates=None):
        super().__init__()
        self.view_centers = view_centers
        self.coordinates = coordinates

    def cart_to_spherical(self, points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))
        x_neg_mask = x < 0
        x_pos_mask = x > 0
        x_zero_mask = x == 0
        psi = np.zeros_like(r)
        psi[x_pos_mask] = np.arctan(y[x_pos_mask] / x[x_pos_mask])
        psi[np.logical_and(x_zero_mask, y >= 0)] = np.pi / 2.
        psi[np.logical_and(x_zero_mask, y < 0)] = -np.pi / 2.
        mask = np.logical_and(x_neg_mask, y >= 0)
        psi[mask] = np.arctan(y[mask] / x[mask]) + np.pi
        mask = np.logical_and(x_neg_mask, y < 0)
        psi[mask] = np.arctan(y[mask] / x[mask]) - np.pi
        return np.stack((r, psi, theta), axis=1)

    def cart_to_cylindrical(self, points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.sqrt(x ** 2 + y ** 2)
        x_neg_mask = x < 0
        x_pos_mask = x > 0
        x_zero_mask = x == 0
        psi = np.zeros_like(r)
        psi[x_pos_mask] = np.arctan(y[x_pos_mask] / x[x_pos_mask])
        psi[np.logical_and(x_zero_mask, y >= 0)] = np.pi / 2.
        psi[np.logical_and(x_zero_mask, y < 0)] = -np.pi / 2.
        mask = np.logical_and(x_neg_mask, y >= 0)
        psi[mask] = np.arctan(y[mask] / x[mask]) + np.pi
        mask = np.logical_and(x_neg_mask, y < 0)
        psi[mask] = np.arctan(y[mask] / x[mask]) - np.pi
        return np.stack((r, psi, z), axis=1)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_in),
                points_pview: [(N, 3 + C_in), ...]
                ...
        """

        points = data_dict['points']
        data_dict['points_pviews'] = []
        data_dict['pview_coordinates'] = []
        for i in range(len(self.view_centers)):
            view_center = self.view_centers[i]
            coordinate = self.coordinates[i]
            shift_points = points[:, 0:3] - np.expand_dims(view_center, axis=0)
            points_pview = getattr(self, f'cart_to_{coordinate}')(shift_points)
            points_pview = np.concatenate((points_pview, points[:, 3:]), axis=1)
            data_dict['points_pviews'].append(points_pview)
            data_dict['pview_coordinates'].append(coordinate)

        return data_dict
