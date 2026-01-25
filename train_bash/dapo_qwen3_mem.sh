

CUDA_VISIBLE_DEVICES=4,5 \
NPROC_PER_NODE=2 \
swift rlhf \
    --rlhf_type grpo \
    --advantage_estimator grpo \
    --model HaluMem_Qwen3-8B \
    --model_type qwen3_nothinking \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_num_seqs 4 \
    --vllm_max_model_len 11264 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset PersonaMem_RL_500.jsonl \
    --dataset_shuffle false \
    --train_dataloader_shuffle false \
    --truncation_strategy left \
    --load_from_cache_file true \
    --split_dataset_ratio 0.05 \
    --max_completion_length 512 \
    --max_length 10240 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --num_generations 8 \
    --dynamic_sample true \
    --max_resample_times 3 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 8 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --loss_type dapo \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --temperature 1 \
    --top_p 0.9 \
    --top_k 50 \
    --deepspeed zero3_offload \
    --num_iterations 1 \
    --beta 0.001 \
    --num_train_epochs 1 \
    --external_plugins train_bash/reward-mem.py \
    --reward_funcs tree_op_reward \
    --eval_strategy steps \
    --eval_steps 20 \
    --logging_steps 1 \
    --log_completions true \
    --log_entropy true \
    --save_steps 20 \
    --save_total_limit 3 \
    --save_only_model true \
    --output_dir result/dapo_qwen3_mem

