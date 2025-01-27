# Multimodal Open R1

We conducted a speed-run on to investigate R1's paradigm in multimodal models after observing growing interest in R1 and studying the elegant implementation of the GRPO algorithm in `open-r1` and `trl`. They paused their ongoing projects to explore R1 in the multimodal domain.

<!-- [Dataset](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified) | [2B Model](https://huggingface.co/lmms-lab/Qwen2-VL-2B-GRPO-8k) | [7B Model](https://huggingface.co/lmms-lab/Qwen2-VL-7B-GRPO-8k) -->

> [!NOTE] 
> While we are familiar with multimodal model implementation, we are new to the RL domain. Although our insights may not be guaranteed to be correct, we commit to sharing them truthfully and honestly. We welcome community feedback and discussions to improve our understanding on multimodal reasoning models. We will PR to `open-r1` later to better support community study on multimodal RL.

## What We Did and Insights

![alt text](assets/lmm_r1.png)

**What We Did**
- Implemented Multimodal R1 based on [huggingface/open-r1](https://github.com/huggingface/open-r1) and [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1). 
  - Integrated Qwen2-VL series, Aria-MoE, and other VLMs available in `transformers`.
- Open-sourced the first batch of `8k` multimodal RL training examples focused on Math reasoning. The data is created by GPT4o with reasoning paths and verifiable answers, based on `Math360K` and `Geo170K`. We provide a [script](local_scripts/create_vision_cot_data.py) for users to inspect and create their own data.
  - The dataset is available in [lmms-lab/multimodal-open-r1-8k-verified](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified).
- Open-sourced models trained with GRPO on `8k` examples.
  - [lmms-lab/Qwen2-VL-2B-GRPO-8k](https://huggingface.co/lmms-lab/Qwen2-VL-2B-GRPO-8k) | [lmms-lab/Qwen2-VL-7B-GRPO-8k](https://huggingface.co/lmms-lab/Qwen2-VL-7B-GRPO-8k).

**Insights and Future Plans**
- Multiple-choice option verification is necessary since many math multimodal problems are MCQs. Discussed in [issue#56](https://github.com/huggingface/open-r1/issues/56) and we customize the verification logic in [src/open_r1/grpo.py](src/open_r1/grpo.py).
- Need to curate RL data to be verifiable, requiring further exploration on effectively converting existing data into RL data and validating GPT4o's curation reliability.
- Current framework is not efficient for large-scale training. Qwen2-VL-2B model takes `10 hours` to train `1 epoch` on `8 H100 GPUs` for `8k samples`. So it's necessary to investigate how to efficiently scale up the training.
- Our init model (Qwen2-VL-2/7B-Instruct) do not show good reasoning ability in our experiments, and during training, the model quickly gather rewards from `format` but not `accuracy`, which is not a good sign for whole RL training. We release our [wandb logs](https://api.wandb.ai/links/libo0013/lz60ml8h) for reference.
- The community may need to curate better multimodal dataset for RL training. Current dataset is limited to math scenarios since it has verifiable answers. It's unclear how to expand the RL dataset to other general domains with open-ended answer. We welcome community feedback on our current strategy and plan to release a larger dataset if we get clear scaling insights through community discussions.


## Training Models

> [!NOTE]
> The training commands below are configured for a node of 8 x H100s (80GB). For different hardware and topologies, you may need to tune the batch size and number of gradient accumulation steps.

### GRPO on Qwen2-VL-2/7B

To run GRPO on Qwen2-VL-2B, run:

```
cd /home/tiger/multimodal-open-r1
# pip3 install vllm==0.6.6.post1
pip3 install -e ".[dev]"

pip3 install wandb==0.18.3

torchrun --nproc_per_node="${ARNOLD_WORKER_GPU}" \ # 8
    --nnodes="${ARNOLD_WORKER_NUM}" \ # 1
    --node_rank="${ARNOLD_ID}" \ # 0
    --master_addr="${METIS_WORKER_0_HOST}" \ # 127.0.0.1
    --master_port="${port_in_cmd}" \ # 12345
    src/open_r1/grpo.py \
    --deepspeed scripts/zero3.json \
    --output_dir checkpoints/Qwen2-VL-2B-GRPO-8k \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dataset_name lmms-lab/multimodal-open-r1-8k-verified \
    --max_prompt_length 8192 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 2359296 \
    --save_total_limit 8 \
    --num_train_epochs 1 \
    --run_name Qwen2-VL-2B-GRPO-8k
```

Please refer to [local_scripts/train_qwen2_vl.sh](local_scripts/train_qwen2_vl.sh) for more details.

Above scripts are naively for `multi-gpu/multi-node` training.

### Evaluating models

We use [lmms-eval]([https://github.com/LMMs-Lab/lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)) to evaluate models, please run:

```shell
bash local_scripts/lmms_eval_qwen2vl.sh
```

Visual reasoning task evaluation currently are limited in direct answer format and simple parsing logic. Tasks like `mmmu_val`, `mathvista_testmini`, and `mmmu_pro` expect direct answers rather than reasoning traces, and the current parsing logic cannot process step-by-step reasoning. We are actively working on improving this limitation and welcome community contributions to develop a more comprehensive evaluation framework for visual reasoning models.

### RL Data Generation

We provide the first batch of `8k` multimodal RL training examples focused on Math reasoning. The data is generated by GPT4o. We provide the [script](local_scripts/create_vision_cot_data.py) to users to inspect and create their own data.

Users can view data in [lmms-lab/multimodal-open-r1-8k-verified](https://huggingface.co/datasets/lmms-lab/multimodal-open-r1-8k-verified). The problem/solution are generated by GPT4o with reasoning path and verifiable answer. The `original question`/`original answer` are from the original dataset.
