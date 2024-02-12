gpu=0
model=punet
epoch=99
extra_tag=punet_baseline

mkdir -p /usr/xtmp/wgk4/PU-Net_pytorch_outputs/${extra_tag}

python3 -u test.py \
    --gpu ${gpu} \
    --model ${model} \
    --save_dir /usr/xtmp/wgk4/PU-Net_pytorch_outputs/${extra_tag} \
    --resume /usr/xtmp/wgk4/PU-Net_pytorch_logs/${extra_tag}/punet_epoch_${epoch}.pth