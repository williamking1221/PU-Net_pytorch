#!/bin/bash
#SBATCH --mem=100G
#SBATCH -G 1 -p compsci-gpu

export PATH=/bin:/usr/bin:/sbin:/usr/sbin:/usr/X11R6/bin:/usr/local/X11R6/bin:/usr/local/bin:/usr/prop/bin:/usr/etc:/usr/local/etc:/usr/hosts:/home/users/wgk4/bin:/home/users/wgk4/.local/bin:/home/users/wgk4/ninja/build-cmake
module load gcc/9.5 cuda/cuda-11.4

gpu=0
model=punet
extra_tag=punet_tda_v_nt

rm -rf /usr/xtmp/wgk4/PU-Net_pytorch_logs/${extra_tag}
mkdir -p /usr/xtmp/wgk4/PU-Net_pytorch_logs/${extra_tag}

python3 -u train_tda.py \
    --gpu ${gpu} \
    --model ${model} \
    --log_dir /usr/xtmp/wgk4/PU-Net_pytorch_logs/${extra_tag} \
    --batch_size 32 \
    --alpha 1.0 \
    --beta 0.1 \
    --workers 1