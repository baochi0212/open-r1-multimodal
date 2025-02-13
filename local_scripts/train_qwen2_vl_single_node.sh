export WANDB_MODE=offline
export WANDB_PROJECT=open_r1
export WANDB_RUN_NAME=qwen2_2b_2048img_4096prompt_fft
if [[ -f ./wandb/$WANDB_RUN_NAME ]]; then
  echo "Run time found"
  rm -rf ./wandb/$WANDB_RUN_NAME
else
  echo "No run time found"
fi
WANDB_PROJECT=$WANDB_PROJECT WANDB_MODE=$WANDB_MODE torchrun --nproc_per_node=8 ./src/open_r1/grpo.py \
    --deepspeed ./local_scripts/zero3.json \
    --output_dir checkpoints/${WANDB_RUN_NAME} \
    --model_name_or_path "./checkpoint_spp/Qwen2-VL-2B-Instruct" \
    --dataset_name "./dataset_spp/multimodal-open-r1-8k-verified"  \
    --max_prompt_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --max_pixels 1605632 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --save_total_limit 3 \
    --num_train_epochs 1\
    --run_name $WANDB_RUN_NAME \
    --save_strategy "steps" \
    --save_steps 100 &
sleep 120
python ./track_wandb_local.py --filename $WANDB_RUN_NAME
