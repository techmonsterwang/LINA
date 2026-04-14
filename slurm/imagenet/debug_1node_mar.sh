export TRITON_PTXAS_PATH=/share/project/panting/packages/cu118/ptxas

rank=$1
zero2_bf16='/share/project/wangjiahao/NOVA/deepspeed/zero2_bf16.json'

# multi-gpu debug, success
python -u scripts/train_torchrun.py \
--deepspeed ${zero2_bf16} --tensorboard \
--rank ${rank} --world-size 1 --num-gpus 8 \
--exp-dir "exp_dir" \
--master-ip job-01315d01-46ae-46c4-8967-0d41b6bac6e8-master-0 \
--cfg /share/project/wangjiahao/LAR/diffnext/config/imagenet/mar_large_1k_nova_d48w768_256px_debug.yml \
2>&1 | tee /share/project/wangjiahao/LAR/logs/train/20250314_debug_1node_${rank}_mar_imagenet.log