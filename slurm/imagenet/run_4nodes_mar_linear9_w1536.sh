rank=$1
zero2_bf16='/share/project/wangjiahao/LAR/configs/deepspeed/zero2_bf16.json'

# multi-gpu debug, success
python -u scripts/train_torchrun.py \
--deepspeed ${zero2_bf16} --tensorboard \
--rank ${rank} --world-size 4 --num-gpus 8 \
--exp-dir "exp_dir/linear9_w1536" \
--master-ip job-70ab8b73-3323-4283-981d-3eca451df53e-master-0 \
--cfg /share/project/wangjiahao/LAR/diffnext/config/imagenet/linear/linear9_mar1k_nova_d48w1536_256px.yml \
2>&1 | tee /share/project/wangjiahao/LAR/logs/train/20250511_run_4node_${rank}_mar_imagenet_linear9_w1536.log