import torch
import torch.nn as nn

PI = torch.acos(torch.Tensor([-1])).cuda()

class TrainablePerspectiveViewGenerator(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_pviews = self.model_cfg.NUM_PVIEWS
        self.view_centers = torch.nn.Parameter(torch.tensor(self.model_cfg.INIT_PVIEW_CENTERS, dtype=torch.float32, requires_grad=True).cuda())
        self.coordinates = self.model_cfg.COORDINATES

        self.debug_cnt = 0

    def cart_to_spherical(self, points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = torch.acos(z / torch.sqrt(x ** 2 + y ** 2 + z ** 2))
        x_neg_mask = x < 0
        x_pos_mask = x > 0
        x_zero_mask = x == 0
        psi = torch.zeros_like(r)
        psi[x_pos_mask] = torch.atan(y[x_pos_mask] / x[x_pos_mask])
        psi[torch.logical_and(x_zero_mask, y >= 0)] = PI / 2.
        psi[torch.logical_and(x_zero_mask, y < 0)] = -PI / 2.
        mask = torch.logical_and(x_neg_mask, y >= 0)
        psi[mask] = torch.atan(y[mask] / x[mask]) + PI
        mask = torch.logical_and(x_neg_mask, y < 0)
        psi[mask] = torch.atan(y[mask] / x[mask]) - PI
        return torch.stack((r, psi, theta), dim=1)

    def cart_to_cylindrical(self, points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = torch.sqrt(x ** 2 + y ** 2)
        x_neg_mask = x < 0
        x_pos_mask = x > 0
        x_zero_mask = x == 0
        psi = torch.zeros_like(r)
        psi[x_pos_mask] = torch.atan(y[x_pos_mask] / x[x_pos_mask])
        psi[torch.logical_and(x_zero_mask, y >= 0)] = PI / 2.
        psi[torch.logical_and(x_zero_mask, y < 0)] = -PI / 2.
        mask = torch.logical_and(x_neg_mask, y >= 0)
        psi[mask] = torch.atan(y[mask] / x[mask]) + PI
        mask = torch.logical_and(x_neg_mask, y < 0)
        psi[mask] = torch.atan(y[mask] / x[mask]) - PI
        return torch.stack((r, psi, z), dim=1)

    def forward(self, batch_dict):
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
        # self.view_centers[0, :] = 0.
        batch_dict['points'].requires_grad = True
        points = batch_dict['points']

        self.debug_cnt += 1
        if self.debug_cnt % 100 == 0:
            print(self.view_centers)
        for i in range(len(self.view_centers)):
            view_center = self.view_centers[i]
            coordinate = self.coordinates[i]
            shift_points = points[:, 1:4] - torch.unsqueeze(view_center, 0)
            points_pview = getattr(self, f'cart_to_{coordinate}')(shift_points)
            points_pview = torch.cat((points[:, 0:1], points_pview, points[:, 4:]), dim=1)
            batch_dict[f'points_pviews_{i}'] = points_pview
            batch_dict[f'pview_coordinates_{i}'] = [coordinate] * batch_dict['batch_size']

        return batch_dict
