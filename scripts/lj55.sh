# Copyright (c) Meta Platforms, Inc. and affiliates.

# ASBS
python train.py \
    experiment=lj55_asbs \
    adj_num_epochs_per_stage=200,300 \
    seed=0,1,2 \
    -m &

# AS
python train.py \
    experiment=lj155_as \
    sigma_max=2 \
    seed=0,1,2 -m &
