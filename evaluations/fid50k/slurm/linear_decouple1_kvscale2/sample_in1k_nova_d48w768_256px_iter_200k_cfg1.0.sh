export PYTHONPATH=/share/project/wangjiahao/LAR
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4
export PET_NNODES=1
export TRITON_PTXAS_PATH=$(pip show torch | grep -oP 'Location:\s*\K.*' | xargs)/torch/bin/ptxas
BASELINE_ROOT=/share/project/wangjiahao/visual_evaluation/fid50k


torchrun --nproc_per_node 4 sample_bf16.py \
  --cfg /share/project/wangjiahao/LAR/diffnext/config/imagenet/linear/linear_decouple1_kvscale2_mar1k_nova_d48w768_256px.yml \
  --ckpt /share/project/wangjiahao/LAR/exp_dir/linear_decouple1_kvscale2/ema_checkpoints/in1k_nova_d48w768_256px_iter_200000 \
  --num_pred_steps 64 --guidance_scale 1.0 --prompt_size 25 --distributed \
  --outdir ${BASELINE_ROOT}/sample_image/linear_decouple1_kvscale2/vae_mar_1k_w768_256px/200k_256px_ar64_cfg1.0 \
  2>&1 | tee /share/project/wangjiahao/visual_evaluation/fid50k/logs/linear_decouple1_kvscale2/sample/200k_256px_ar64_cfg1.0.log


