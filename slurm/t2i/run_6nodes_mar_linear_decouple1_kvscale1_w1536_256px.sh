rank=$1
zero2_bf16='/share/project/wangjiahao/LAR/configs/deepspeed/zero2_bf16.json'

# multi-gpu debug, success
python -u scripts/train_torchrun.py \
--deepspeed ${zero2_bf16} --tensorboard \
--rank ${rank} --world-size 6 --num-gpus 8 \
--exp-dir "exp_dir/t2i/linear_decouple1_kvscale1_w1536_6node" \
--master-ip job-ead216a2-e2ba-46da-8e87-bae181933456-master-0 \
--cfg /share/project/wangjiahao/LAR/diffnext/config/t2i/sdxl28m_nova_d48w1536_256px.yml \
2>&1 | tee /share/project/wangjiahao/LAR/logs/train/20250818_run_6node_${rank}_mar_t2i_linear_decouple1_kvscale1_w1536_256px.log