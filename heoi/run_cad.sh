# !/usr/bin/env bash

# V1 box-level-hoi

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_BoxHOI_B_E_LR --generator_name Dense_V1 --batch_size 24 --lr_G 0.0001 --nepochs_decay 0 --nepochs_no_decay 5 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final.py --name CAD_DenseInter_V1_BoxHOI_B_E_LR --generator_name Dense_V1 --batch_size 300 --lr_G 0.0001 --nepochs_decay 0 --nepochs_no_decay 5 

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_BoxHOI_B150_E5_LR0001 --network_name Dense_V1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 0 --nepochs_no_decay 5 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final.py --name CAD_DenseInter_V1_BoxHOI_B150_E5_LR0001 --network_name Dense_V1 --batch_size 300 --lr_G 0.0001 --nepochs_decay 0 --nepochs_no_decay 5 

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_1_BoxHOI_B150_E5_LR0001 --network_name Dense_V1_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 0 --nepochs_no_decay 5 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final.py --name CAD_DenseInter_V1_1_BoxHOI_B150_E5_LR0001 --network_name Dense_V1_1 --batch_size 300 --lr_G 0.0001 --nepochs_decay 0 --nepochs_no_decay 5 

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_1_BoxHOI_B150_E5_LR00001 --network_name Dense_V1_1 --batch_size 150 --lr_G 0.00001 --nepochs_decay 0 --nepochs_no_decay 5 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final.py --name CAD_DenseInter_V1_1_BoxHOI_B150_E5_LR00001 --network_name Dense_V1_1 --batch_size 300 --lr_G 0.00001 --nepochs_decay 0 --nepochs_no_decay 5 

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_1_BoxHOI_B150_E5_LR001 --network_name Dense_V1_1 --batch_size 150 --lr_G 0.001 --nepochs_decay 0 --nepochs_no_decay 5 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final.py --name CAD_DenseInter_V1_1_BoxHOI_B150_E5_LR001 --network_name Dense_V1_1 --batch_size 300 --lr_G 0.001 --nepochs_decay 0 --nepochs_no_decay 5 

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_1_BoxHOI_B150_E5_LR000001 --network_name Dense_V1_1 --batch_size 150 --lr_G 0.000001 --nepochs_decay 0 --nepochs_no_decay 5 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final.py --name CAD_DenseInter_V1_1_BoxHOI_B150_E5_LR000001 --network_name Dense_V1_1 --batch_size 300 --lr_G 0.000001 --nepochs_decay 0 --nepochs_no_decay 5 

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_1_BoxHOI_B_E15_LR000001 --network_name Dense_V1_1 --batch_size 350 --lr_G 0.000001 --nepochs_decay 5 --nepochs_no_decay 10
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final.py --name CAD_DenseInter_V1_1_BoxHOI_B_E15_LR000001 --network_name Dense_V1_1 --batch_size 400 --lr_G 0.000001 --nepochs_decay 5 --nepochs_no_decay 10

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_2_BoxHOI_B_E10_LR0001 --network_name Dense_V1_2 --batch_size 350 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final.py --name CAD_DenseInter_V1_2_BoxHOI_B_E10_LR0001 --network_name Dense_V1_2 --batch_size 400 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_2_BoxHOI_B_E7_LR0001 --network_name Dense_V1_2 --batch_size 350 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 2
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final.py --name CAD_DenseInter_V1_2_BoxHOI_B_E7_LR0001 --network_name Dense_V1_2 --batch_size 400 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 2

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_2_BoxHOI_B_E7_LR00001 --network_name Dense_V1_2 --batch_size 350 --lr_G 0.00001 --nepochs_decay 5 --nepochs_no_decay 2
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final.py --name CAD_DenseInter_V1_2_BoxHOI_B_E7_LR00001 --network_name Dense_V1_2 --batch_size 400 --lr_G 0.00001 --nepochs_decay 5 --nepochs_no_decay 2

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_2_BoxHOI_B_E7_LR000001 --network_name Dense_V1_2 --batch_size 350 --lr_G 0.000001 --nepochs_decay 5 --nepochs_no_decay 2
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final.py --name CAD_DenseInter_V1_2_BoxHOI_B_E7_LR000001 --network_name Dense_V1_2 --batch_size 400 --lr_G 0.000001 --nepochs_decay 5 --nepochs_no_decay 2

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_2_BoxHOI_B_E10_LR000001 --model_name Dense_V1_1 --network_name Dense_V1_2 --batch_size 350 --lr_G 0.000001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final.py --name CAD_DenseInter_V1_2_BoxHOI_B_E10_LR000001 --model_name Dense_V1_1 --network_name Dense_V1_2 --batch_size 400 --lr_G 0.000001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V1_2_BoxHOI_B_E10_LR000001 --model_name Dense_V1_1 --network_name Dense_V1_2 --batch_size 400 --lr_G 0.000001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V2_0_BoxHOI_B_E10_LR000001 --model_name Dense_V1 --network_name Dense_V2_0 --batch_size 350 --lr_G 0.000001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final.py --name CAD_DenseInter_V2_0_BoxHOI_B_E10_LR000001 --model_name Dense_V1 --network_name Dense_V2_0 --batch_size 400 --lr_G 0.000001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V2_0_BoxHOI_B100_E10_LR000001 --model_name Dense_V1 --network_name Dense_V2_0 --batch_size 100 --lr_G 0.000001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final.py --name CAD_DenseInter_V2_0_BoxHOI_B100_E10_LR000001 --model_name Dense_V1 --network_name Dense_V2_0 --batch_size 100 --lr_G 0.000001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V2_0_BoxHOI_Norm_B100_E10_LR0001 --model_name Dense_V1 --network_name Dense_V2_0 --batch_size 100 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final.py --name CAD_DenseInter_V2_0_BoxHOI_Norm_B100_E10_LR0001 --model_name Dense_V1 --network_name Dense_V2_0 --batch_size 100 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V2_1_test_LossRegular_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V2_0 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final.py --name CAD_DenseInter_V2_1_test_LossRegular_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V2_0 --batch_size 400 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# test
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V2_1_test_LossRegular_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V2_0 --batch_size 400 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_2_BoxHOI_B_E10_LR0001 --model_name Dense_V1_1 --network_name Dense_V1_2 --batch_size 350 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V1_2_BoxHOI_B_E10_LR0001 --model_name Dense_V1_1 --network_name Dense_V1_2 --batch_size 400 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_2_BoxHOI_CE_B_E10_LR0001 --model_name Dense_V1_1 --network_name Dense_V1_2 --batch_size 350 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V1_2_BoxHOI_CE_B_E10_LR0001 --model_name Dense_V1_1 --network_name Dense_V1_2 --batch_size 400 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_2_BoxHOI_softCE_B_E10_LR0001 --model_name Dense_V1_1 --network_name Dense_V1_2 --batch_size 350 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V1_2_BoxHOI_softCE_B_E10_LR0001 --model_name Dense_V1_1 --network_name Dense_V1_2 --batch_size 400 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V3_0_BoxSkeletonHOI_B_E10_LR0001 --model_name Dense_V1_1 --network_name Dense_V3_0 --batch_size 350 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V3_0_BoxSkeletonHOI_B_E10_LR0001 --model_name Dense_V1_1 --network_name Dense_V3_0 --batch_size 400 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V4_0_BoxSkeletonPart_B_E10_LR0001 --model_name Dense_V1_1 --network_name Dense_V4_0 --batch_size 350 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V4_0_BoxSkeletonPart_B_E10_LR0001 --model_name Dense_V1_1 --network_name Dense_V4_0 --batch_size 400 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_3_B_E10_LR0001 --model_name Dense_V1_1 --network_name Dense_V1_3 --batch_size 350 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V1_3_B_E10_LR0001 --model_name Dense_V1_1 --network_name Dense_V1_3 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_3_Sigmoid_B_E10_LR0001 --model_name Dense_V1_1 --network_name Dense_V1_3 --batch_size 350 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V1_3_Sigmoid_B_E10_LR0001 --model_name Dense_V1_1 --network_name Dense_V1_3 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_3_1_Sigmoid_B_E10_LR0001 --model_name Dense_V1_1 --network_name Dense_V1_3_1 --batch_size 350 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V1_3_1_Sigmoid_B_E10_LR0001 --model_name Dense_V1_1 --network_name Dense_V1_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_3_1_OneSigmoid_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V1_3_1_one --batch_size 350 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V1_3_1_OneSigmoid_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V1_3_1_one --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_3_1_OneSigmoid_reviseobjmask_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V1_3_1_one --batch_size 350 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V1_3_1_OneSigmoid_reviseobjmask_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V1_3_1_one --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_3_1_OneSigmoid_removeMaskpool_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V1_3_1_one --batch_size 350 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V1_3_1_OneSigmoid_removeMaskpool_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V1_3_1_one --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_3_1_OneSigmoid_removeMaskpool_RemoveSpatial_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V1_3_1_one --batch_size 350 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V1_3_1_OneSigmoid_removeMaskpool_RemoveSpatial_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V1_3_1_one --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V5_0_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V5_0_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V5_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V5_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V5_3_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V5_3_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V5_2_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_2 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V5_2_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_2 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V5_4_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V5_4_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V5_5_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V5_5_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V5_6_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V5_6_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V5_6_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V5_6_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V5_6_2_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V5_6_2_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V5_4_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V5_4_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V5_4_2_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V5_4_2_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V5_4_3_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V5_4_3_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V5_4_4_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V5_4_4_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V5_4_5_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V5_4_5_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V5_4_5_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V5_4_5_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V5_4_5_2_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 120 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V5_4_5_2_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V5_0 --batch_size 120 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_B100_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6 --batch_size 100 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_B100_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6 --batch_size 100 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_1_revise_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_1_revise_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_0_revise_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_0_revise_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_0_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_0_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_2_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_2_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_2_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_2_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_3_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_4_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_4_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_5_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_5_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_5_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_5_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_5_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_5_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V1_0_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v1_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseFusion_V1_0_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v1_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V1_1_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v1_0 --batch_size 120 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseFusion_V1_1_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v1_0 --batch_size 120 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V1_2_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v1_0 --batch_size 120 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseFusion_V1_2_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v1_0 --batch_size 120 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V1_3_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v1_0 --batch_size 120 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseFusion_V1_3_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v1_0 --batch_size 120 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V1_3_1_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v1_0 --batch_size 120 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseFusion_V1_3_1_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v1_0 --batch_size 120 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V1_3_2_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v1_0 --batch_size 120 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseFusion_V1_3_2_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v1_0 --batch_size 120 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V1_4_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v1_0 --batch_size 120 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseFusion_V1_4_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v1_0 --batch_size 120 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# After relu

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_0_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_0_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_0_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_0_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_0_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_0_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5


# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_3_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_3_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_bodypart_03_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_3_bodypart_03_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_1_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_3_1_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_1_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_3_1_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_1_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_3_1_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5


# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_4_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_4_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_V6_7_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_4_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_4_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_4_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_4_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_5_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_5_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_5_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_5_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_5_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_5_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_6_bodypart_03_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_6_bodypart_03_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_6_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_6_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_6_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_6_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V1_1_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v1_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseFusion_V1_1_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v1_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V1_2_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v1_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseFusion_V1_2_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v1_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V1_3_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v1_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseFusion_V1_3_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v1_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_1_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V2_1_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_2_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseFusion_V2_2_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_3_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseFusion_V2_3_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_4_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseFusion_V2_4_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_5_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseFusion_V2_5_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_5_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseFusion_V2_5_B_E10_LR0001 --model_name Dense_V1 --network_name DenseFusion_v2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5


# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_3_1_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V2_3_1_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5


# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_2_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_3_2_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_2_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_3_2_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_2_body_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_3_2_body_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_3_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_3_3_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_3_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_3_3_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_3_body_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseInter_V6_7_3_3_body_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_0_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_0_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_0_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_0 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_6_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_6 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_6_bodypart_03_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart_6 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_6_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_6 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_4_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_4 --batch_size 8 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_4_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart_4 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_4_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_4 --batch_size 8 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_5_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_5 --batch_size 8 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_5_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart_5 --batch_size 8 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_5_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_5 --batch_size 8 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_3_1_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_3_1 --batch_size 8 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_3_1_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart_3_1 --batch_size 8 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_3_1_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_3_1 --batch_size 8 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_3_3_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_3_3 --batch_size 8 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_3_3_body_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart_3_3 --batch_size 8 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_3_3_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_3_3 --batch_size 8 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V1_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_1_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V1_2_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_1_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V2_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V2_2_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V2_3_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_3_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V2_4_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_4 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V1_1_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_1_1_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_final_quality.py --name CAD_DenseFusion_V1_1_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_1_1_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V1_1_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_1_1_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_3_1_1_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_3_1_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5



# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_6_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_6 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V2_6_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_6 --batch_size 8 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_7_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_7 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V2_7_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_7 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_8_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_8 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V2_8_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_8 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V1_3_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_1_3 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V1_1_2_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_1_1_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V1_1_2_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_1_1_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V2_8_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_8 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V1_1_6_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_1_1_6 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V1_1_7_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_1_1_7 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V1_1_6_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_1_1_6 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V1_1_1_B_32_LR0001 --model_name Dense_V1 --network_name Dense_fusion_1_1_1 --batch_size 32 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5         # 改变batch_size
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V1_1_1_B_64_LR00001 --model_name Dense_V1 --network_name Dense_fusion_1_1_1 --batch_size 64 --lr_G 0.00001 --nepochs_decay 5 --nepochs_no_decay 5       # 改变batch_size和学习率
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V1_1_1_B_150_LR00001 --model_name Dense_V1 --network_name Dense_fusion_1_1_1 --batch_size 150 --lr_G 0.00001 --nepochs_decay 5 --nepochs_no_decay 5     # 学习率不同
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V1_1_1_B_300_LR0001 --model_name Dense_V1 --network_name Dense_fusion_1_1_1 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5       # 改变batch_size
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V1_1_1_B_300_LR001 --model_name Dense_V1 --network_name Dense_fusion_1_1_1 --batch_size 300 --lr_G 0.001 --nepochs_decay 5 --nepochs_no_decay 5       # 改变batch_size和学习率
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V1_1_1_B_300_LR001 --model_name Dense_V1 --network_name Dense_fusion_1_1_1 --batch_size 300 --lr_G 0.001 --nepochs_decay 5 --nepochs_no_decay 5       # 改变batch_size和学习率
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V1_1_1_B_64_LR0001 --model_name Dense_V1 --network_name Dense_fusion_1_1_1 --batch_size 64 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5         # 改变batch_size
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V1_1_1_B_400_LR0001 --model_name Dense_V1 --network_name Dense_fusion_1_1_1 --batch_size 400 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5       # 改变batch_size和学习率
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V1_1_1_B_400_LR0001 --model_name Dense_V1 --network_name Dense_fusion_1_1_1 --batch_size 400 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5       # 改变batch_size和学习率
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V1_1_1_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_1_1_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5       # 改变batch_size和学习率
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V1_1_1_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_1_1_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5       # 改变batch_size和学习率
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_1_1__box_B_300_LR0001 --model_name Dense_V1 --network_name Dense_box_1_1 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5       # 改变batch_size和学习率
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V1_1_1__box_B_300_LR0001 --model_name Dense_V1 --network_name Dense_box_1_1 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5       # 改变batch_size和学习率
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_1_1_skeleton_B_300_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_1_1 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5       # 改变batch_size和学习率
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V1_1_1_skeleton_B_300_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_1_1 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_1_1_bodypart_B_300_LR0001 --model_name Dense_V1 --network_name Dense_bodypart_1_1 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5       # 改变batch_size和学习率
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V1_1_1_bodypart_B_300_LR0001 --model_name Dense_V1 --network_name Dense_bodypart_1_1 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5  
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_1_1_B_300_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_1 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V2_1_1_B_300_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_1 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_1_2_B_300_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_2 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V2_1_2_B_300_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_2 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_1_3_B_150_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V2_1_3_B_150_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_1_3_B_250_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_1_3_B_300_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_1_4_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_4 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V2_1_4_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_4 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_1_5_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_5 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_1_3_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V2_1_3_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python train.py --name CAD_DenseFusion_V2_1_3_B_300_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python test_CAD_netpart.py --name CAD_DenseFusion_V2_1_3_B_300_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python train.py --name CAD_DenseFusion_V2_1_6_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_6 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python test_CAD_netpart.py --name CAD_DenseFusion_V2_1_6_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_6 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python train.py --name CAD_DenseFusion_V2_1_3_1_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python test_CAD_netpart.py --name CAD_DenseFusion_V2_1_3_1_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python train.py --name CAD_DenseFusion_V2_1_3_1_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python test_CAD_netpart.py --name CAD_DenseFusion_V2_1_3_1_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python train.py --name CAD_DenseFusion_V2_9_B_200_LR001 --model_name Dense_V1 --network_name Dense_fusion_2_9 --batch_size 200 --lr_G 0.001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python test_CAD_netpart.py --name CAD_DenseFusion_V2_9_B_200_LR001 --model_name Dense_V1 --network_name Dense_fusion_2_9 --batch_size 200 --lr_G 0.001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python train.py --name CAD_DenseFusion_V2_1_3_2_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_2 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python test_CAD_netpart.py --name CAD_DenseFusion_V2_1_3_2_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_2 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python train.py --name CAD_DenseFusion_V2_1_3_3_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_3 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python test_CAD_netpart.py --name CAD_DenseFusion_V2_1_3_3_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_3 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python train.py --name CAD_DenseFusion_V2_1_3_4_B_150_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_4 --batch_size 64 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python test_CAD_netpart.py --name CAD_DenseFusion_V2_1_3_4_B_150_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_4 --batch_size 64 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python train.py --name CAD_DenseFusion_V2_1_3_5_B_100_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_5 --batch_size 100 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python test_CAD_netpart.py --name CAD_DenseFusion_V2_1_3_5_B_100_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_5 --batch_size 100 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay
# CUDA_VISIBLE_DEVICES=0,1 python train.py --name CAD_DenseFusion_V2_10_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_10 --batch_size 100 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python test_CAD_netpart.py --name CAD_DenseFusion_V2_10_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_10 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python train.py --name CAD_DenseFusion_V2_11_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_11 --batch_size 64 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python test_CAD_netpart.py --name CAD_DenseFusion_V2_11_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_11 --batch_size 100 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5


# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_0_skeleton_B_200_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_0 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_0_skeleton_B_200_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_0 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_6_box_B_200_LR0001 --model_name Dense_V1 --network_name Dense_box_6 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_6_box_B_200_LR0001 --model_name Dense_V1 --network_name Dense_box_6 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_6_bodypart_B_200_LR0001 --model_name Dense_V1 --network_name Dense_bodypart_6 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_6_bodypart_B_200_LR0001 --model_name Dense_V1 --network_name Dense_bodypart_6 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_6_skeleton_B_200_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_6 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_6_skeleton_B_200_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_6 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_4_box_B_200_LR0001 --model_name Dense_V1 --network_name Dense_box_4 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_4_box_B_200_LR0001 --model_name Dense_V1 --network_name Dense_box_4 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_4_skeleton_B_200_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_4 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_4_skeleton_B_200_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_4 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_5_skeleton_B_200_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_5 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_5_skeleton_B_200_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_5 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_5_box_B_200_LR0001 --model_name Dense_V1 --network_name Dense_box_5 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_5_box_B_200_LR0001 --model_name Dense_V1 --network_name Dense_box_5 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_5_bodypart_B_200_LR0001 --model_name Dense_V1 --network_name Dense_bodypart_5 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_5_bodypart_B_200_LR0001 --model_name Dense_V1 --network_name Dense_bodypart_5 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_1_box_B_200_LR0001 --model_name Dense_V1 --network_name Dense_box_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_3_1_box_B_200_LR0001 --model_name Dense_V1 --network_name Dense_box_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_1_skeleton_B_200_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_3_1_skeleton_B_200_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_2_box_B_200_LR0001 --model_name Dense_V1 --network_name Dense_box_3_2 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_3_2_box_B_200_LR0001 --model_name Dense_V1 --network_name Dense_box_3_2 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_2_skeleton_B_200_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_3_2 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_3_2_skeleton_B_200_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_3_2 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_2_bodypart_B_200_LR0001 --model_name Dense_V1 --network_name Dense_bodypart_3_2 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_3_2_bodypart_B_200_LR0001 --model_name Dense_V1 --network_name Dense_bodypart_3_2 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_1_bodypart_B_200_LR0001 --model_name Dense_V1 --network_name Dense_bodypart_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_3_1_bodypart_B_200_LR0001 --model_name Dense_V1 --network_name Dense_bodypart_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_3_skeleton_B_200_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_3_3 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_3_3_skeleton_B_200_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_3_3 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_3_box_B_200_LR0001 --model_name Dense_V1 --network_name Dense_box_3_3 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_3_3_box_B_200_LR0001 --model_name Dense_V1 --network_name Dense_box_3_3 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5


# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_2_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_3_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_3_2_box_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_3_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_2_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart_3_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_3_2_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_bodypart_3_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V6_7_3_2_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_3_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V6_7_3_2_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_3_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_2_box_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_bodypart_1_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V1_2_box_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_bodypart_1_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V2_1_box_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_bodypart_2_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V2_1_box_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_bodypart_2_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_2_box_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_skeleton_1_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V1_2_box_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_skeleton_1_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V2_1_box_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_skeleton_2_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V2_1_box_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_skeleton_2_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5


# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V2_2_box_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_skeleton_2_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V2_3_1_box_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_skeleton_2_3_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V2_4_box_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_skeleton_2_4 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_1_1_box_skeleton_B_300_LR0001 --model_name Dense_V1 --network_name Dense_box_skeleton_1_1_1 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V2_1_3_1_box_skeleton_B_200_LR0001 --model_name Dense_V1 --network_name Dense_box_skeleton_2_1_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V2_2_box_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_skeleton_2_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V2_3_1_box_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_skeleton_2_3_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V2_4_box_skeleton_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_skeleton_2_4 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V1_1_1_box_skeleton_B_300_LR0001 --model_name Dense_V1 --network_name Dense_box_skeleton_1_1_1 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V2_1_3_1_box_skeleton_B_200_LR0001 --model_name Dense_V1 --network_name Dense_box_skeleton_2_1_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V2_2_box_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_bodypart_2_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V2_3_1_box_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_bodypart_2_3_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V2_4_box_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_bodypart_2_4 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V1_1_1_box_bodypart_B_300_LR0001 --model_name Dense_V1 --network_name Dense_box_bodypart_1_1_1 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V2_1_3_1_box_bodypart_B_200_LR0001 --model_name Dense_V1 --network_name Dense_box_bodypart_2_1_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5


# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_2_skeleton_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_bodypart_1_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V2_1_skeleton_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_bodypart_2_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V2_2_skeleton_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_bodypart_2_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V2_3_1_skeleton_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_bodypart_2_3_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V2_4_skeleton_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_bodypart_2_4 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V1_1_1_skeleton_bodypart_B_300_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_bodypart_1_1_1 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_V2_1_3_1_skeleton_bodypart_B_200_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_bodypart_2_1_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V1_2_skeleton_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_bodypart_1_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V2_1_skeleton_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_bodypart_2_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V2_2_skeleton_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_bodypart_2_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V2_3_1_skeleton_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_bodypart_2_3_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V2_4_skeleton_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_bodypart_2_4 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V1_1_1_skeleton_bodypart_B_300_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_bodypart_1_1_1 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V2_1_3_1_skeleton_bodypart_B_200_LR0001 --model_name Dense_V1 --network_name Dense_skeleton_bodypart_2_1_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V2_2_box_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_bodypart_2_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python trtest_CAD_netpartain.py --name CAD_DenseInter_V2_3_1_box_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_bodypart_2_3_1 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V2_4_box_bodypart_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_box_bodypart_2_4 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V1_1_1_box_bodypart_B_300_LR0001 --model_name Dense_V1 --network_name Dense_box_bodypart_1_1_1 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_V2_1_3_1_box_bodypart_B_200_LR0001 --model_name Dense_V1 --network_name Dense_box_bodypart_2_1_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_scene_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_scene --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_scene_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_scene --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_pathfusion_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_pathfusion_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_scene_B_200_LR0001 --model_name Dense_V1 --network_name Dense_scene --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_scene_B_200_LR0001 --model_name Dense_V1 --network_name Dense_scene --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_v2_1_3_2_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_2 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_pathfusion_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_2 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_v2_1_3_3_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_3 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_human_B_200_LR0001 --model_name Dense_V1 --network_name Dense_human --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_human_B_200_LR0001 --model_name Dense_V1 --network_name Dense_human --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_object_B_200_LR0001 --model_name Dense_V1 --network_name Dense_object --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_object_B_200_LR0001 --model_name Dense_V1 --network_name Dense_object --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_scene_object_B_200_LR0001 --model_name Dense_V1 --network_name Dense_scene_object --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_scene_object_B_200_LR0001 --model_name Dense_V1 --network_name Dense_scene_object --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_human_object_B_150_LR0001 --model_name Dense_V1 --network_name Dense_human_object --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_human_object_B_150_LR0001 --model_name Dense_V1 --network_name Dense_human_object --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_human_object_B_200_LR0001 --model_name Dense_V1 --network_name Dense_human_object --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_human_object_B_200_LR0001 --model_name Dense_V1 --network_name Dense_human_object --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseInter_human_scene_B_200_LR0001 --model_name Dense_V1 --network_name Dense_human_scene --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseInter_human_scene_B_200_LR0001 --model_name Dense_V1 --network_name Dense_human_scene --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_v2_1_3_2_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_2 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_v2_1_3_2_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_2 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_v2_1_3_3_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_3 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_v2_1_3_3_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_3 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_v2_1_3_4_B_16_LR00001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_4 --batch_size 16 --lr_G 0.00001 --nepochs_decay 35 --nepochs_no_decay 35
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_v2_1_3_4_B_16_LR00001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_4 --batch_size 16 --lr_G 0.00001 --nepochs_decay 18 --nepochs_no_decay 18

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_v2_3_1_1_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_3_1_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1 python train.py --name CAD_DenseFusion_V1_1_1_B_300_LR0001 --model_name Dense_V1 --network_name Dense_fusion_1_1_1 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5       # 改变batch_size和学习率
# CUDA_VISIBLE_DEVICES=0,1 python test_CAD_netpart.py --name CAD_DenseFusion_V1_1_1_B_300_LR0001 --model_name Dense_V1 --network_name Dense_fusion_1_1_1 --batch_size 300 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5       # 改变batch_size和学习率

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_10_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_10 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V2_10_B_E10_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_10 --batch_size 150 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_V2_1_3_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_V2_1_3_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_v2_1_3_5_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_5 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_v2_1_3_5_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_5 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --name CAD_DenseFusion_v2_1_3_6_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_6 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1,2,3 python test_CAD_netpart.py --name CAD_DenseFusion_v2_1_3_6_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_6 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python train.py --name CAD_DenseFusion_V2_1_3_7_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_7 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python test_CAD_netpart.py --name CAD_DenseFusion_V2_1_3_7_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_7 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python train.py --name CAD_DenseFusion_V2_1_3_2_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_2 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0,1 python test_CAD_netpart.py --name CAD_DenseFusion_V2_1_3_2_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_2 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0,1 python train.py --name CAD_DenseFusion_V2_1_3_8_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_8 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
CUDA_VISIBLE_DEVICES=0,1 python test_CAD_netpart.py --name CAD_DenseFusion_V2_1_3_1_B_200_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_1 --batch_size 200 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5

# CUDA_VISIBLE_DEVICES=0 python train.py --name CAD_1_B_100_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_1 --batch_size 100 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5
# CUDA_VISIBLE_DEVICES=0 python test_CAD_netpart.py --name CAD_1_B_100_LR0001 --model_name Dense_V1 --network_name Dense_fusion_2_1_3_1 --batch_size 100 --lr_G 0.0001 --nepochs_decay 5 --nepochs_no_decay 5