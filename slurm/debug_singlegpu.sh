rank=$1
zero2_bf16='/share/project/wangjiahao/NOVA/deepspeed/zero2_bf16.json'


# 1 GPU debug, success
python -u scripts/train.py \
--deepspeed ${zero2_bf16} --tensorboard \
--cfg /share/project/wangjiahao/NOVA/diffnext/config/d48_w1024/sdxl16m_nova_d48w1024_1024px.yml > /share/project/wangjiahao/NOVA/logs/train/20250224_debug_${rank}_first_exp.log





