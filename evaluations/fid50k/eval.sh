export TF_CPP_MIN_LOG_LEVEL=2
REF=/share/project/denghaoge/Autoregressive/Dataset/VIRTUAL_imagenet256_labeled.npz

SAMPLE=/share/project/panting/remotes/Baseline/fid50k/vae_v2_1k_w768_512px/150k_512px_ar64_cfg3.3.npz 
SAMPLE=/share/project/denghaoge/ar_visual/ARPG-main/samples/ARPG-XXL-arpg_1b-size-256-size-256-VQ-16-topk-0-topp-1.0-temperature-1.0-cfg-7.5-schedule-linear-step-64-seed-0.npz

python -u eval.py --ref_batch ${REF} --sample_batch ${SAMPLE} \
> $(basename ${SAMPLE%.*}).log 2>&1 &

