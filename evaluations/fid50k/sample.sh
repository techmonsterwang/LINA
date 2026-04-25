export PYTHONPATH=/share/project/panting/diffnext-main
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4
export PET_NNODES=1
export TRITON_PTXAS_PATH=$(pip show torch | grep -oP 'Location:\s*\K.*' | xargs)/torch/bin/ptxas
BASELINE_ROOT=/share/project/panting/remotes/Baseline/fid50k
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# ps aux | grep "sample.py" | awk '{print $2}' | xargs kill -9

i=0
EXP=$(((30 + $i) * 5))k
CKPT=iter_$(((30 + $i) * 5000))
nohup torchrun --nproc_per_node 4 sample.py \
--cfg /share/project/panting/diffnext-main/configs/nova_d48w768/vae1k_nova_d48w768_512px.yml \
--ckpt /share/project/panting/diffnext-main/experiments/20250118_082602/ema_checkpoints/in1k_nova_d48w768_512px_${CKPT} \
--num_pred_steps 64 --guidance_scale 3.3 --prompt_size 25 --distributed \
--outdir ${BASELINE_ROOT}/vae_v2_1k_w768_512px/${EXP}_512px_ar64_cfg3.3 \
> ${SCRIPT_DIR}/${EXP}_512px_ar64_cfg3.3.log 2>&1 &
