rank=$1
zero2_bf16='/share/project/wangjiahao/LAR/configs/deepspeed/zero2_bf16.json'

# multi-gpu debug, success
python -u scripts/train_torchrun.py \
--deepspeed ${zero2_bf16} --tensorboard \
--rank ${rank} --world-size 4 --num-gpus 8 \
--exp-dir "exp_dir/linear1" \
--master-ip job-01315d01-46ae-46c4-8967-0d41b6bac6e8-master-0 \
--cfg /share/project/wangjiahao/LAR/diffnext/config/imagenet/linear/linear1_mar1k_nova_d48w768_256px.yml \
2>&1 | tee /share/project/wangjiahao/LAR/logs/train/20250319_run_4node_${rank}_mar_imagenet_linear1.log