python evaluations/latency/latency.py \
--metadata /share/project/wangjiahao/LAR/evaluations/geneval/metadata.jsonl \
--prompt evaluations/geneval/prompts.pth \
--cfg /share/project/wangjiahao/LAR/diffnext/config/t2i/sdxl28m_nova_d48w1536_1024px_full_attention.yml \
--ckpt /share/project/wangjiahao/LAR/pretrained/models.t2i/scratch_full_attention_nova_d48w1536_1024px \
--num_pred_steps 128 --guidance_scale 7 --prompt_size 8 --sample_size 4 \
--batch_size 1 --num_test_batches 10 --num_runs 3 \
--output_filename latency_test_full_attention_1024.txt