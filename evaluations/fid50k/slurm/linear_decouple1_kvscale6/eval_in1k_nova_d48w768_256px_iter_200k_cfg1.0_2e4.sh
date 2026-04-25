export TF_CPP_MIN_LOG_LEVEL=2
REF=/share/project/denghaoge/Autoregressive/Dataset/VIRTUAL_imagenet256_labeled.npz
SAMPLE=/share/project/wangjiahao/visual_evaluation/fid50k/sample_image/linear_decouple1_kvscale6_2e4/vae_mar_1k_w768_256px/200k_256px_ar64_cfg1.0.npz

python -u eval.py --ref_batch ${REF} --sample_batch ${SAMPLE} \
2>&1 | tee /share/project/wangjiahao/visual_evaluation/fid50k/logs/linear_decouple1_kvscale6_2e4/eval/$(basename ${SAMPLE%.*}).log


