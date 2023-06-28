#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

python -m torch.distributed.launch --nproc_per_node=${NGPUS} test.py --launcher pytorch ${PY_ARGS}

python -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --cfg_file cfgs/kitti_models/PartA2_mv_v2.yaml

python -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --cfg_file cfgs/kitti_models/PartA2_mv_v2.yaml

python -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --cfg_file cfgs/kitti_models/PartA2_mv_v2.yaml

python -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --cfg_file cfgs/kitti_models/PartA2_mv_v2.yaml

python -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --cfg_file cfgs/kitti_models/PartA2_mv_v2.yaml
