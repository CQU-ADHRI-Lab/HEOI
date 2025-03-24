# !/usr/bin/env bash

# V1 box-level-hoi

# CUDA_VISIBLE_DEVICES=0,1 python train.py --name CAD_DenseFusion_V2_1_3_1_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
CUDA_VISIBLE_DEVICES=0,1 python test_CAD_netpart.py --name CAD_DenseFusion_V2_1_3_1_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
