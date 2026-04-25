export TF_CPP_MIN_LOG_LEVEL=2
REF=/share/project/denghaoge/Autoregressive/Dataset/VIRTUAL_imagenet256_labeled.npz
SAMPLE=/share/project/wangjiahao/visual_evaluation/fid50k/sample_image/vae_mar_1k_w768_256px/400k_256px_ar64_cfg3.0.npz

python -u eval.py --ref_batch ${REF} --sample_batch ${SAMPLE} \
2>&1 | tee /share/project/wangjiahao/visual_evaluation/fid50k/logs/eval/$(basename ${SAMPLE%.*}).log


