rank=$1
zero2_bf16='/share/project/wangjiahao/NOVA/deepspeed/zero2_bf16.json'

# multi-gpu debug, success
python -u scripts/train_torchrun.py \
--deepspeed ${zero2_bf16} --tensorboard \
--rank ${rank} --world-size 1 --num-gpus 8 \
--exp-dir "exp_dir" \
--master-ip job-f585a39e-abc8-4c55-b7ef-b7012a677ae4-master-0 \
--cfg /share/project/wangjiahao/NOVA/diffnext/config/d48_w1024/sdxl16m_nova_d48w1024_1024px.yml \
2>&1 | tee /share/project/wangjiahao/NOVA/logs/train/20250224_debug_multinode_${rank}_first_exp.log


