export TRITON_PTXAS_PATH=/share/project/panting/packages/cu118/ptxas

zero2_bf16='/share/project/wangjiahao/NOVA/deepspeed/zero2_bf16.json'


python -u scripts/train.py \
--deepspeed ${zero2_bf16} --tensorboard \
--exp-dir "exp_dir/debug_dev" \
--cfg /share/project/wangjiahao/LAR/diffnext/config/imagenet/mar_large_1k_nova_d48w768_256px_debug.yml \
2>&1 | tee /share/project/wangjiahao/LAR/logs/train/20250321_debug_dev_1node_mar_imagenet.log

