python evaluations/geneval/pick.py \
--metadata /share/project/wangjiahao/LAR/evaluations/geneval/metadata.jsonl \
--prompt /share/project/panting/diffnext-main/prompts/pick.pth \
--cfg /share/project/wangjiahao/LAR/diffnext/config/t2i/sdxl28m_nova_d48w1536_1024px_finetune_nova_v3.yml \
--ckpt /share/project/wangjiahao/LAR/checkpoints_public3/1024 \
--num_pred_steps 128 --guidance_scale 7 --prompt_size 8 --sample_size 4 \
--outdir /share/project/wangjiahao/LAR/evaluations/geneval/image_folder/1024/linearnova-d48w1536-sdxl28m-cfg7-finetune-nova-v3-50000-pick