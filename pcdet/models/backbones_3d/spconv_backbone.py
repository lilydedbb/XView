from functools import partial

import torch
import spconv
import torch.nn as nn


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        return out


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

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


        if 'sperical' in kwargs and kwargs['sperical']:
            self.conv_out = spconv.SparseSequential(
                # [200, 150, 5] -> [200, 150, 2]
                spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0),
                                    bias=False, indice_key='spconv_down2'),
                norm_fn(128),
                nn.ReLU(),
            )
        else:
            last_pad = 0
            last_pad = self.model_cfg.get('last_pad', last_pad)
            self.conv_out = spconv.SparseSequential(
                # [200, 150, 5] -> [200, 150, 2]
                spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                    bias=False, indice_key='spconv_down2'),
                norm_fn(128),
                nn.ReLU(),
            )
        self.num_point_features = 128

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

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        if outkeys is None:
            batch_dict.update({
                'encoded_spconv_tensor': out,
                'encoded_spconv_tensor_stride': 8
            })
            batch_dict.update({
                'multi_scale_3d_features': {
                    'x_conv1': x_conv1,
                    'x_conv2': x_conv2,
                    'x_conv3': x_conv3,
                    'x_conv4': x_conv4,
                }
            })
            # print(input_sp_tensor.dense().size())
            # print(x.dense().size())
            # print(x_conv1.dense().size())
            # print(x_conv2.dense().size())
            # print(x_conv3.dense().size())
            # print(x_conv4.dense().size())
            # # torch.Size([1, 4, 41, 1600, 1408])
            # # torch.Size([1, 16, 41, 1600, 1408])
            # # torch.Size([1, 16, 41, 1600, 1408])
            # # torch.Size([1, 32, 21, 800, 704])
            # # torch.Size([1, 64, 11, 400, 352])
            # # torch.Size([1, 64, 5, 200, 176])
            # # torch.Size([1, 128, 2, 200, 176])
        else:
            batch_dict.update({
                outkeys[0]: out,
                outkeys[1]: 8
            })
            batch_dict.update({
                outkeys[2]: {
                    'x_conv1': x_conv1,
                    'x_conv2': x_conv2,
                    'x_conv3': x_conv3,
                    'x_conv4': x_conv4,
                }
            })
            # # torch.Size([1, 4, 320, 320, 1640])
            # # torch.Size([1, 16, 320, 320, 1640])
            # # torch.Size([1, 32, 160, 160, 820])
            # # torch.Size([1, 64, 80, 80, 410])
            # # torch.Size([1, 64, 40, 40, 205])
            # # torch.Size([1, 128, 20, 40, 205])

        return batch_dict


class MVoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_sizes, **kwargs):
        super().__init__()
        self.voxel_backbone8x_list = []
        for i in range(len(grid_sizes)):
            grid_size = grid_sizes[i]
            input_c = input_channels[i]
            kwargs['pview'] = True
            voxel_backbone8x = VoxelBackBone8x(model_cfg, input_c, grid_size, **kwargs)
            self.voxel_backbone8x_list.append(voxel_backbone8x)
            self.add_module(f'voxel_backbone8x_{i}', voxel_backbone8x)

    def forward(self, batch_dict):
        num_views = len([key for key in batch_dict if key.startswith('pview_voxel_features_')])
        for i in range(num_views):
            inkeys = [f'pview_voxel_features_{i}', f'pview_voxel_coords_{i}']
            outkeys = [f'pview_encoded_spconv_tensor_{i}', f'pview_encoded_spconv_tensor_stride_{i}',
                        f'pview_multi_scale_3d_features_{i}']
            batch_dict = self.voxel_backbone8x_list[i](batch_dict, inkeys=inkeys, outkeys=outkeys)
        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
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

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        return batch_dict
