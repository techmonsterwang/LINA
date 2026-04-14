rank=$1
zero2_bf16='/share/project/wangjiahao/LAR/configs/deepspeed/zero2_bf16.json'

# multi-gpu debug, success
python -u scripts/train_torchrun.py \
--deepspeed ${zero2_bf16} --tensorboard \
--rank ${rank} --world-size 4 --num-gpus 8 \
--exp-dir "exp_dir/linear_decouple1_kvscale1_w1024" \
--master-ip job-884532f3-b984-46d4-a5ba-b81c057676b8-master-0 \
--cfg /share/project/wangjiahao/LAR/diffnext/config/imagenet/linear/linear_decouple1_kvscale1_mar1k_nova_d48w1024_256px.yml \
2>&1 | tee /share/project/wangjiahao/LAR/logs/train/20250626_run_4node_${rank}_mar_imagenet_linear_decouple1_kvscale1_w1024.log