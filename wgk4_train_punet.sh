#!/bin/bash
#SBATCH --mem=30G
#SBATCH -G 1 -p compsci-gpu

export PATH=/bin:/usr/bin:/sbin:/usr/sbin:/usr/X11R6/bin:/usr/local/X11R6/bin:/usr/local/bin:/usr/prop/bin:/usr/etc:/usr/local/etc:/usr/hosts:/home/users/wgk4/bin:/home/users/wgk4/.local/bin:/home/users/wgk4/ninja/build-cmake

module load gcc/9.5 cuda/cuda-11.4

gpu=0
model=punet
extra_tag=punet_baseline

rm -rf /usr/xtmp/wgk4/PU-Net_pytorch_logs/${extra_tag}
mkdir /usr/xtmp/wgk4/PU-Net_pytorch_logs/${extra_tag}

# nohup python3 -u train.py \
#     --gpu ${gpu} \
#     --model ${model} \
#     --log_dir /usr/xtmp/wgk4/PU-Net_pytorch_logs/${extra_tag} \
#     --batch_size 32 \
#     --alpha 1.0 \
#     --workers 1 \
#     >> /usr/xtmp/wgk4/PU-Net_pytorch_logs/${extra_tag}/nohup.log 2>&1 &
python3 -u train.py \
    --gpu ${gpu} \
    --model ${model} \
    --log_dir /usr/xtmp/wgk4/PU-Net_pytorch_logs/${extra_tag} \
    --batch_size 32 \
    --alpha 1.0 \
    --workers 1 \