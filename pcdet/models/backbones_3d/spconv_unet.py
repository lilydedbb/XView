from functools import partial

import spconv
import torch
import torch.nn as nn

from ...utils import common_utils
from .spconv_backbone import post_act_block


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity
        out.features = self.relu(out.features)

        return out


class UNetV2(nn.Module):
    """
    Sparse Convolution based UNet for point-wise feature learning.
    Reference Paper: https://arxiv.org/abs/1907.03670 (Shaoshuai Shi, et. al)
    From Points to Parts: 3D Object Detection from Point Cloud with Part-aware and Part-aggregation Network
    """
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        if 'sperical' in kwargs and kwargs['sperical']:
            self.conv4 = spconv.SparseSequential(
                # [400, 352, 11] <- [200, 176, 5]
                block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
                block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
                block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            )
        else:
            self.conv4 = spconv.SparseSequential(
                # [400, 352, 11] <- [200, 176, 5]
                block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
                block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
                block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            )

        if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):

            if 'sperical' in kwargs and kwargs['sperical']:
                self.conv_out = spconv.SparseSequential(
                    # [205, 40, 40] -> [205, 40, 20]
                    spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0),
                                        bias=False, indice_key='spconv_down2'),
                    norm_fn(128),
                    nn.ReLU(),
                )
            else:
                last_pad = self.model_cfg.get('last_pad', 0)
                self.conv_out = spconv.SparseSequential(
                    # [200, 150, 5] -> [200, 150, 2]
                    spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                        bias=False, indice_key='spconv_down2'),
                    norm_fn(128),
                    nn.ReLU(),
                )
        else:
            self.conv_out = None

        # decoder
        # [400, 352, 11] <- [200, 176, 5]
        self.conv_up_t4 = SparseBasicBlock(64, 64, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.inv_conv4 = block(64, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')

        # [800, 704, 21] <- [400, 352, 11]
        self.conv_up_t3 = SparseBasicBlock(64, 64, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(64, 32, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')

        # [1600, 1408, 41] <- [800, 704, 21]
        self.conv_up_t2 = SparseBasicBlock(32, 32, indice_key='subm2', norm_fn=norm_fn)
        self.conv_up_m2 = block(64, 32, 3, norm_fn=norm_fn, indice_key='subm2')
        self.inv_conv2 = block(32, 16, 3, norm_fn=norm_fn, indice_key='spconv2', conv_type='inverseconv')

        # [1600, 1408, 41] <- [1600, 1408, 41]
        self.conv_up_t1 = SparseBasicBlock(16, 16, indice_key='subm1', norm_fn=norm_fn)
        self.conv_up_m1 = block(32, 16, 3, norm_fn=norm_fn, indice_key='subm1')

        self.conv5 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1')
        )
        self.num_point_features = 16

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x.features = torch.cat((x_bottom.features, x_trans.features), dim=1)
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x.features = x_m.features + x.features
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x.features = features.view(n, out_channels, -1).sum(dim=2)
        return x

    def forward(self, batch_dict, inkeys=None, outkeys=None):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        if inkeys is None:
            voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        else:
            voxel_features_key, voxel_coords_key = inkeys
            voxel_features, voxel_coords = batch_dict[voxel_features_key], batch_dict[voxel_coords_key]

        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        if self.conv_out is not None:
            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)

            if outkeys is None:
                batch_dict['encoded_spconv_tensor'] = out  # [B, 128, 2, 200, 176])
                batch_dict['encoded_spconv_tensor_stride'] = 8
            else:
                batch_dict[outkeys[0]] = out  # [B, 128, 20, 40, 205]
                batch_dict[outkeys[1]] = 8

        # for segmentation head
        # [400, 352, 11] <- [200, 176, 5]
        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        # [800, 704, 21] <- [400, 352, 11]
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        # [1600, 1408, 41] <- [800, 704, 21]
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2, self.conv_up_m2, self.inv_conv2)
        # [1600, 1408, 41] <- [1600, 1408, 41]
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1, self.conv_up_m1, self.conv5)

        if outkeys is None:
            batch_dict['point_features'] = x_up1.features
        else:
            batch_dict[outkeys[2]] = x_up1
        point_coords = common_utils.get_voxel_centers(
            x_up1.indices[:, 1:], downsample_times=1, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        if outkeys is None:
            batch_dict['point_coords'] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
            # print(input_sp_tensor.dense().size())
            # print(x.dense().size())
            # print(x_conv1.dense().size())
            # print(x_conv2.dense().size())
            # print(x_conv3.dense().size())
            # print(x_conv4.dense().size())
            # print(x_up4.dense().size())
            # print(x_up3.dense().size())
            # print(x_up2.dense().size())
            # print(x_up1.dense().size())
            # # torch.Size([1, 4, 41, 1600, 1408])                                                                        | 0/3712 [00:00<?, ?it/s]torch.Size([1, 16, 41, 1600, 1408])
            # # torch.Size([1, 16, 41, 1600, 1408])                                                                     | 0/3712 [00:00<?, ?it/s]torch.Size([1, 16, 41, 1600, 1408])
            # # torch.Size([1, 16, 41, 1600, 1408])
            # # torch.Size([1, 32, 21, 800, 704])
            # # torch.Size([1, 64, 11, 400, 352])
            # # torch.Size([1, 64,  5, 200, 176])
            # # torch.Size([1, 64, 11, 400, 352])
            # # torch.Size([1, 32, 21, 800, 704])
            # # torch.Size([1, 16, 41, 1600, 1408])
            # # torch.Size([1, 16, 41, 1600, 1408])
        else:
            batch_dict[outkeys[3]] = torch.cat((x_up1.indices[:, 0:1].float(), point_coords), dim=1)
            # print(input_sp_tensor.dense().size())
            # print(x.dense().size())
            # print(x_conv1.dense().size())
            # print(x_conv2.dense().size())
            # print(x_conv3.dense().size())
            # print(x_conv4.dense().size())
            # print(x_up4.dense().size())
            # print(x_up3.dense().size())
            # print(x_up2.dense().size())
            # print(x_up1.dense().size())
            # # torch.Size([1, 4, 320, 320, 1640])
            # # torch.Size([1, 16, 320, 320, 1640])
            # # torch.Size([1, 16, 320, 320, 1640])
            # # torch.Size([1, 32, 160, 160, 820])                                                                        | 0/3712 [00:00<?, ?it/s]
            # # torch.Size([1, 64, 80, 80, 410])
            # # torch.Size([1, 64, 40, 40, 205])
            # # torch.Size([1, 64, 80, 80, 410])
            # # torch.Size([1, 32, 160, 160, 820])
            # # torch.Size([1, 16, 320, 320, 1640])
            # # torch.Size([1, 16, 320, 320, 1640])
        return batch_dict


class MUNetV2(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_sizes, voxel_sizes, point_cloud_ranges, **kwargs):
        super().__init__()
        self.unet_v2_list = []
        for i in range(len(grid_sizes)):
            kwargs['pview'] = True
            unet_v2 = UNetV2(model_cfg, input_channels[i], grid_sizes[i], voxel_sizes[i], point_cloud_ranges[i], **kwargs)
            self.unet_v2_list.append(unet_v2)
            self.add_module(f'unet_v2_{i}', unet_v2)

    def forward(self, batch_dict):
        num_views = len([key for key in batch_dict if key.startswith('pview_voxel_features_')])
        for i in range(num_views):
            inkeys = [f'pview_voxel_features_{i}', f'pview_voxel_coords_{i}']
            outkeys = [f'pview_encoded_spconv_tensor_{i}', f'pview_encoded_spconv_tensor_stride_{i}',
                        f'pview_point_features_{i}', f'pview_point_coords_{i}']
            batch_dict = self.unet_v2_list[i](batch_dict, inkeys=inkeys, outkeys=outkeys)
        return batch_dict