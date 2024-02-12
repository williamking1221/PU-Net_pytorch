gpu=0
model=punet
epoch=9
extra_tag=punet_tda_beta_0_1_voxelnum_216_downsample_32_dim_0_wq_2_test

mkdir -p /usr/xtmp/wgk4/PU-Net_pytorch_outputs/${extra_tag}

python3 -u test.py \
    --gpu ${gpu} \
    --model ${model} \
    --save_dir /usr/xtmp/wgk4/PU-Net_pytorch_outputs/${extra_tag} \
    --resume /usr/xtmp/wgk4/PU-Net_pytorch_logs/${extra_tag}/punet_epoch_${epoch}.pth