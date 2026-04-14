python evaluations/geneval/sample.py \
--metadata /share/project/wangjiahao/LAR/evaluations/geneval/metadata.jsonl \
--prompt evaluations/geneval/prompts_rewrite.pth \
--cfg /share/project/wangjiahao/LAR/diffnext/config/t2i/sdxl28m_nova_d48w1536_1024px_finetune_nova_v3.yml \
--ckpt /share/project/wangjiahao/LAR/exp_dir/t2i/linear_decouple1_kvscale1_w1536_6node_finetune_nova_1024px_v3/ema_checkpoints/sdxl28m_nova_d48w1536_256px_iter_100000 \
--num_pred_steps 128 --guidance_scale 7 --prompt_size 8 --sample_size 4 \
--outdir /share/project/wangjiahao/LAR/evaluations/geneval/image_folder/1024/linearnova-d48w1536-sdxl28m-cfg7-finetune-nova-v3-100000-rewrite