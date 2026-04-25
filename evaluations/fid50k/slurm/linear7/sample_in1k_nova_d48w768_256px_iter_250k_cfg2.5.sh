export PYTHONPATH=/share/project/wangjiahao/LAR
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4
export PET_NNODES=1
export TRITON_PTXAS_PATH=$(pip show torch | grep -oP 'Location:\s*\K.*' | xargs)/torch/bin/ptxas
BASELINE_ROOT=/share/project/wangjiahao/visual_evaluation/fid50k


torchrun --nproc_per_node 4 sample.py \
  --cfg /share/project/wangjiahao/LAR/diffnext/config/imagenet/linear/linear7_mar1k_nova_d48w768_256px.yml \
  --ckpt /share/project/wangjiahao/LAR/exp_dir/linear7/ema_checkpoints/in1k_nova_d48w768_256px_iter_250000 \
  --num_pred_steps 64 --guidance_scale 2.5 --prompt_size 25 --distributed \
  --outdir ${BASELINE_ROOT}/sample_image/linear7/vae_mar_1k_w768_256px/250k_256px_ar64_cfg2.5 \
  2>&1 | tee /share/project/wangjiahao/visual_evaluation/fid50k/logs/linear7/sample/250k_256px_ar64_cfg2.5.log


