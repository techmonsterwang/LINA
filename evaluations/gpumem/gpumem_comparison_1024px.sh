python /share/project/wangjiahao/LAR/evaluations/gpumem/gpumem.py \
--prompt /share/project/wangjiahao/LAR/evaluations/geneval/prompts.pth \
--cfg /share/project/wangjiahao/LAR/diffnext/config/t2i/sdxl28m_nova_d48w1536_1024px_finetune_nova_v3.yml,/share/project/wangjiahao/LAR/diffnext/config/t2i/sdxl28m_nova_d48w1536_1024px_full_attention.yml \
--ckpt /share/project/wangjiahao/LAR/exp_dir/t2i/linear_decouple1_kvscale1_w1536_6node_finetune_nova_1024px_v3/ema_checkpoints/sdxl28m_nova_d48w1536_256px_iter_200000,/share/project/wangjiahao/LAR/pretrained/models.t2i/scratch_full_attention_nova_d48w1536_1024px \
--num_pred_steps 128 --num_diff_steps 25 --guidance_scale 7 \
--prompt_size 1 --sample_size 1 --vae_batch_size 16 \
--num_runs 3 \
--output_filename gpumem_comparison_1024px.txt


