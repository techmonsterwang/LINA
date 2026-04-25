# 1. Sample
```bash
# For text-to-image demo
python evaluations/geneval/pick.py \
--metadata ./evaluations/geneval/metadata.jsonl \
--prompt ./evaluations/geneval/prompts/pick.pth \
--cfg ./diffnext/config/t2i/sdxl28m_nova_d48w1536_1024px_finetune_nova_v3.yml \
--ckpt /path/to/lina-t2i-d48w1536-sdxl1024 \
--num_pred_steps 128 --guidance_scale 7 --prompt_size 8 --sample_size 4 \
--outdir ./evaluations/geneval/lina-t2i-d48w1536-sdxl1024-cfg7-pick
```