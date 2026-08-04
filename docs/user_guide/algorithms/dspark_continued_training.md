# DSpark 增训指南：从已有 Checkpoint 继续训练

## 概述

要加载已有的 DSpark 草稿模型继续训练（增训），只需在 `scripts/train.py` 调用中添加 `--from-pretrained` 参数，指向你的预训练 checkpoint 路径。

## 关键参数

| 参数 | 作用 |
|---|---|
| `--from-pretrained <路径>` | 加载已有 speculator checkpoint 的权重，从该权重继续训练 |
| `--no-resume-from-checkpoint` | 禁止从 `--save-path` 自动恢复（加载预训练模型时可选） |

这两个参数定义在 `src/speculators/train/config/schema.py`（`DraftArgs.from_pretrained` 和 `TrainerArgs.no_resume_from_checkpoint`）。

`build_draft_model()` 函数（`scripts/train.py:403`）负责解析 draft model 的加载来源：当 `--from-pretrained` 设置后，会调用 `model_class.from_pretrained()` 加载完整的 speculator checkpoint 权重，然后进入训练循环继续优化。

## 修改后的完整脚本

以下是从 `examples/train/dspark_qwen3_0_6b_sharegpt_online.sh` 修改而来的增训版本，核心变化仅在第 100 行新增 `--from-pretrained`：

```bash
#!/bin/bash
# Online DSpark Continued Training Script
# 从已有的 DSpark checkpoint 出发进行增训
#
# Usage: 修改下方配置变量后运行:
#   bash dspark_qwen3_0_6b_sharegpt_online_continue.sh

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-0.6B"
DATASET="sharegpt"
OUTPUT_DIR="./output/dspark_qwen3_0_6b_sharegpt_continue"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=4096
EPOCHS=5
LR=3e-4                           # 增训可适当降低，如 1e-4

# ★ 预训练 DSpark checkpoint 路径
PRETRAINED_DSPARK="./output/dspark_qwen3_0_6b_sharegpt/checkpoints"  # 👈 改成你的路径

# DSpark-specific parameters (必须与原模型一致)
SPECULATOR_TYPE="dspark"
BLOCK_SIZE=8
MAX_ANCHORS=3072
NUM_LAYERS=3
DRAFT_VOCAB_SIZE=32000
TARGET_LAYER_IDS="2 14 25"

# Markov + confidence head settings
MARKOV_RANK=256
MARKOV_HEAD_TYPE="vanilla"
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

# GPU assignments
VLLM_GPUS="0"
TRAIN_GPUS="1"
NUM_TRAIN_GPUS=1
# =======================================

# Step 1: Prepare data
echo "=== Step 1: Preparing data ==="
python scripts/prepare_data.py \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH"

# Step 2: Launch vLLM server
echo "=== Step 2: Launching vLLM server ==="
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --port "$VLLM_PORT" &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

# Step 3: Train from pretrained checkpoint
echo "=== Step 3: Continued Training ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --vllm-endpoint "http://localhost:${VLLM_PORT}/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --markov-rank "$MARKOV_RANK" \
    --markov-head-type "$MARKOV_HEAD_TYPE" \
    --enable-confidence-head \
    --confidence-head-with-markov \
    --loss-fn "$LOSS_FN" \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --from-pretrained "$PRETRAINED_DSPARK" \
    --on-missing generate \
    --on-generate delete

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
```

## 关键点说明

### 1. `--from-pretrained` 是核心

`--from-pretrained "$PRETRAINED_DSPARK"` 指向已训练的 DSpark checkpoint 目录（包含 `model.safetensors` 和 `config.json` 的那个目录）。这是唯一必要的改动。

### 2. `--save-path` 换一个路径

避免覆盖原 checkpoint。增训的 checkpoint 写入新目录。

### 3. 架构参数必须一致

以下参数必须和预训练模型保持一致，因为 `--from-pretrained` 会加载已有权重，shape 必须匹配：

- `--block-size`
- `--num-layers`
- `--markov-rank`
- `--markov-head-type`
- `--target-layer-ids`
- `--draft-vocab-size`

### 4. LR 可适当降低

增训场景通常用比从头训练更小的学习率（如 `1e-4` 或 `3e-5`），避免破坏已学到的特征。

### 5. `--no-resume-from-checkpoint` 的使用场景

如果 `--save-path` 目录为空（第一次增训），不加也没关系。如果 `--save-path` 和 `--from-pretrained` 指向同一目录，需要加上此标志以避免冲突：

```bash
--no-resume-from-checkpoint
```

### 6. 两种加载路径的内部逻辑

`build_draft_model()` 对 `--from-pretrained` 有两种处理：

- **config-only 目录**（只有 `config.json`）：从 speculator config 初始化全新权重（随机初始化 decoder），仅复用配置结构
- **完整 checkpoint**（包含权重文件如 `model.safetensors`）：调用 `model_class.from_pretrained()` 完整加载所有权重，然后从该状态继续训练

## Dry-run 验证

如果不确定预训练 checkpoint 是否可用，可以先 dry-run 验证：

```bash
python scripts/train.py \
    --speculator-type dspark \
    --verifier-name-or-path "Qwen/Qwen3-0.6B" \
    --from-pretrained "./output/dspark_qwen3_0_6b_sharegpt/checkpoints" \
    --dry-run
```

这会加载 checkpoint、验证结构然后退出，不会进入训练循环。验证通过后去掉 `--dry-run` 即可开始正式增训。

## 离线训练变体

以上示例是 online 训练（通过 vLLM 在线生成 hidden states）。如果要改成离线增训，只需：

1. 提前生成好 hidden states 数据
2. 去掉 vLLM 相关配置（`--vllm-endpoint`、`--on-missing`、`--on-generate`）
3. `--from-pretrained` 的用法完全相同

## 相关文件

- `examples/train/dspark_qwen3_0_6b_sharegpt_online.sh` — 从头训练参考脚本
- `scripts/train.py` — 训练入口，`build_draft_model()` 在 L403
- `src/speculators/train/config/schema.py` — 所有 CLI 参数定义（`DraftArgs` L71, `TrainerArgs` L372）
- `docs/user_guide/algorithms/dspark.md` — DSpark 算法文档
