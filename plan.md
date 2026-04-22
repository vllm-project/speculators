# 为 Qwen3-Omni-Thinking 适配 DFlash Draft Model 的改造计划（speculators 侧）

> 本文档仅覆盖 **speculators 仓库**的代码修改，不包含 vLLM 侧改造（后者另起单独文档）。
>
> 设计原则：**Draft Model 仍然是 Qwen3-like 的稠密（Dense MLP）Transformer**（1 层 decoder），复用现有 `DFlashDraftModel` / `Qwen3DFlashDecoderLayer`，通过适配 verifier config 的嵌套层级、MRoPE、MoE intermediate_size 等差异，把 **`Qwen3/Qwen3-Omni-30B-A3B-Thinking` 作为 verifier** 驱动训练与蒸馏。

---

## 1. 背景与目标

### 1.1 verifier 关键参数

| 项 | 值 | 说明 |
|---|---|---|
| `model_type`（顶层） | `qwen3_omni_moe` | 架构 `Qwen3OmniMoeForConditionalGeneration` |
| 文本塔路径 | `config.thinker_config.text_config` | **两层嵌套**（Qwen3VL 只有一层） |
| 文本塔 `model_type` | `qwen3_omni_moe_text` | transformers 中为 MoE 实现 |
| `hidden_size` | `2048` | |
| `num_hidden_layers` | `48` | |
| `num_attention_heads` / `num_key_value_heads` | `32 / 4` | GQA |
| `head_dim` | `128` | |
| `vocab_size` | `152064` | |
| `max_position_embeddings` | `65536` | |
| `intermediate_size` / `moe_intermediate_size` | `768 / 768` | **MoE 语义下 768 过小**，Dense draft 必须替换 |
| `num_experts / num_experts_per_tok` | `128 / 8` | Dense draft 不使用，但影响 FFN 等效容量 |
| `rope_theta` | `1_000_000` | |
| `rope_scaling` | `{"mrope_section":[24,20,20], "mrope_interleaved":true, "rope_type":"default"}` | **MRoPE，当前 DFlash 的 `apply_rotary_pos_emb` 不支持** |
| `use_qk_norm` | `true` | Qwen3 已内置，draft 保持一致即可 |
| `rms_norm_eps` | `1e-06` | |
| DeepStack 层 | `[8, 16, 24]` | 视觉注入层，**aux hidden state 选层需规避** |
| `tts_pad_token_id` | `151671` | Thinking 版 TTS 关闭，可复用作 MASK token |

### 1.2 draft 目标

- 结构：**1 层 Qwen3 Dense Decoder**（复用 `Qwen3DFlashDecoderLayer`，`Qwen3MLP` 稠密）
- `hidden_size = 2048`（对齐 verifier 便于 hidden_state concat）
- `intermediate_size = 6144`（等价稠密参数量，**见 Step 1 的换算**）
- `num_hidden_layers = 1`，`num_attention_heads = 32`，`num_key_value_heads = 4`，`head_dim = 128`
- `rope_theta = 1_000_000`，`rope_scaling = {mrope_section, mrope_interleaved}` 透传（Step 5 处理 MRoPE）
- `vocab_size = 152064`，`draft_vocab_size` 可取 `32000` ~ `65536`（与主词表做 t2d/d2t 映射）

---

## 2. 范围界定

### 2.1 In-scope（本 plan 要改）

- `scripts/train.py` — verifier→draft config 派生逻辑
- `scripts/launch_vllm.py` — vLLM 启动前的 verifier config 解析
- `src/speculators/models/dflash/core.py` — 运行时 verifier config 解析 + RoPE 初始化
- `src/speculators/models/dflash/config.py` — `validate_transformer_config` 对 `qwen3_omni_moe_text` 的降级处理
- `src/speculators/models/dflash/model_definitions.py` — `apply_rotary_pos_emb` 的 MRoPE 支持
- `src/speculators/data_generation/custom_worker.py` — hidden state 捕获点的 `thinker` 前缀分支
- `examples/data_generation_and_training/` — 新增 Qwen3-Omni-Thinking 示例脚本

### 2.2 Out-of-scope（本 plan 不改）

- `src/speculators/utils/loading.py` — `load_model_layers` 第 47 行已用 `endswith(name)` 做后缀匹配，`"embed_tokens.weight"` 可直接命中 `thinker.model.embed_tokens.weight`，**无需扩展**。
- `src/speculators/train/utils.py` — `resolve_mask_token_id` 四级回退已足够，Qwen3-Omni 会走到"注入新 `<|MASK|>`"路径，或由用户通过 `--mask-token-id 151671` 指定复用 `tts_pad_token_id`。
- `src/speculators/train/data.py`、`metrics.py`、`utils/anchors.py` — 与 verifier 架构无关。
- vLLM 侧 `vllm/model_executor/models/qwen3_dflash.py` / `vllm/v1/spec_decode/dflash.py` / `eagle.py` — 单独文档处理（须解决 non-causal attention backend、MRoPE 在 draft 推理路径上的适配等）。

---

## 3. 分步改造清单

### Step 1：`scripts/train.py` — verifier→draft config 派生（核心）

**文件**：`scripts/train.py`
**函数**：`create_transformer_layer_config`（当前 104-141 行）

**现状问题**：

```python
# scripts/train.py L122-126（现状）
verifier_config = AutoConfig.from_pretrained(verifier_name_or_path)
# For multimodal models (Qwen3VL, etc.), extract text_config
if hasattr(verifier_config, "text_config"):
    verifier_config = verifier_config.text_config
```

- **只剥一层 `text_config`**：Qwen3-Omni 顶层没有 `text_config`，有的是 `thinker_config`，因此这里会拿到 `Qwen3OmniMoeConfig` 而不是 text 塔。
- `verifier_config.intermediate_size = 768` 会被直接透传为 draft 的 `intermediate_size`，Dense 语义下 FFN 太小。
- **未透传** `rope_theta` / `rope_scaling`，draft `Qwen3Config` 会回退到默认 theta=10000 + 无 mrope_section，与 verifier 位置编码不一致。

**修改方案**：

```python
# scripts/train.py   create_transformer_layer_config 修改后
config_class = DRAFT_ARCH_CONFIGS[draft_arch]
verifier_config = AutoConfig.from_pretrained(verifier_name_or_path)

# 嵌套剥离：Qwen3-Omni 走 thinker_config → text_config（两层）
#          Qwen3VL / Llama-VL 走 text_config（一层）
if hasattr(verifier_config, "thinker_config"):
    verifier_config = verifier_config.thinker_config
if hasattr(verifier_config, "text_config"):
    verifier_config = verifier_config.text_config

# MoE → Dense 的 intermediate_size 换算
# 默认策略：稠密 FFN 参数量 ≈ moe_intermediate_size * num_experts_per_tok
is_moe_text = getattr(verifier_config, "model_type", "").endswith("_moe_text") \
              or hasattr(verifier_config, "moe_intermediate_size")
if is_moe_text:
    moe_ffn = getattr(verifier_config, "moe_intermediate_size",
                      verifier_config.intermediate_size)
    experts_per_tok = getattr(verifier_config, "num_experts_per_tok", 1)
    default_draft_intermediate = moe_ffn * experts_per_tok  # 768 * 8 = 6144
else:
    default_draft_intermediate = verifier_config.intermediate_size

# 允许 CLI 显式覆盖
draft_intermediate_size = getattr(args_ns, "draft_intermediate_size", None) \
                          or default_draft_intermediate

# rope_scaling / rope_theta 透传（MRoPE 关键）
rope_kwargs = {}
if getattr(verifier_config, "rope_theta", None) is not None:
    rope_kwargs["rope_theta"] = verifier_config.rope_theta
if getattr(verifier_config, "rope_scaling", None) is not None:
    rope_kwargs["rope_scaling"] = verifier_config.rope_scaling

return config_class(
    vocab_size=verifier_config.vocab_size,
    hidden_size=verifier_config.hidden_size,
    intermediate_size=draft_intermediate_size,
    num_hidden_layers=num_layers,
    num_attention_heads=verifier_config.num_attention_heads,
    num_key_value_heads=verifier_config.num_key_value_heads,
    hidden_act=verifier_config.hidden_act,
    max_position_embeddings=verifier_config.max_position_embeddings,
    initializer_range=verifier_config.initializer_range,
    rms_norm_eps=verifier_config.rms_norm_eps,
    head_dim=getattr(verifier_config, "head_dim", None),
    tie_word_embeddings=False,
    **rope_kwargs,
)
```

**CLI 新增参数**（`build_arg_parser` / `add_train_args` 附近）：

```python
parser.add_argument(
    "--draft-intermediate-size",
    type=int,
    default=None,
    help=(
        "Override draft FFN intermediate size. For MoE verifier (e.g. "
        "Qwen3-Omni-Thinking), the default is moe_intermediate_size * "
        "num_experts_per_tok (6144 for Qwen3-Omni)."
    ),
)
```

并把 `args_ns` 透传进 `create_transformer_layer_config`（调用点加参数即可）。

**验收标准**：
- 用 `Qwen/Qwen3-Omni-30B-A3B-Thinking` 作为 `--verifier` 跑 `create_transformer_layer_config`，得到 `Qwen3Config` 满足：
  - `hidden_size == 2048`
  - `intermediate_size == 6144`（默认）
  - `num_attention_heads == 32`，`num_key_value_heads == 4`，`head_dim == 128`
  - `rope_theta == 1_000_000`
  - `rope_scaling["mrope_section"] == [24, 20, 20]`
  - `vocab_size == 152064`

---

### Step 2：`scripts/launch_vllm.py` — verifier config 嵌套 + aux layer 默认值

**文件**：`scripts/launch_vllm.py`
**位置**：第 60-62 行（verifier config 剥 text_config 的逻辑）

**修改**：

```python
# 同步 Step 1 的嵌套剥离
verifier_config = AutoConfig.from_pretrained(verifier_name_or_path)
if hasattr(verifier_config, "thinker_config"):
    verifier_config = verifier_config.thinker_config
if hasattr(verifier_config, "text_config"):
    verifier_config = verifier_config.text_config

num_layers = verifier_config.num_hidden_layers
```

**aux hidden state 层选择**（`target_layer_ids` 默认值，构造 `speculative_config` 附近）：

```python
# DeepStack 视觉注入层，避开（Qwen3-Omni）
DEEPSTACK_LAYERS = set(getattr(verifier_config, "deepstack_visual_indexes", []))
# 默认三层：浅 / 中 / 深
candidate = [2, num_layers // 2, num_layers - 3]
# 若撞到 DeepStack 层则向前挪 1 层
target_layer_ids = [
    (l - 1 if l in DEEPSTACK_LAYERS else l) for l in candidate
]
```

**验收标准**：
- 对 Qwen3-Omni-Thinking（48 层，DeepStack=[8,16,24]）：
  - 默认候选 `[2, 24, 45]` → 24 撞 DeepStack → 最终 `[2, 23, 45]`
- CLI 显式 `--target-layer-ids 2 23 45` 可覆盖。

---

### Step 3：`src/speculators/models/dflash/core.py` — 运行时 verifier config 嵌套

**文件**：`src/speculators/models/dflash/core.py`
**位置**：第 69-72 行

**现状**：

```python
verifier_config = AutoConfig.from_pretrained(verifier_name_or_path)
if hasattr(verifier_config, "text_config"):
    verifier_config = verifier_config.text_config
```

**修改**：

```python
verifier_config = AutoConfig.from_pretrained(verifier_name_or_path)
# Unwrap nested config: Qwen3-Omni (thinker_config -> text_config) /
# Qwen3VL (text_config)
if hasattr(verifier_config, "thinker_config"):
    verifier_config = verifier_config.thinker_config
if hasattr(verifier_config, "text_config"):
    verifier_config = verifier_config.text_config
```

**影响确认**：
- 第 78-82 行 `target_layer_ids = [2, num_verifier_layers // 2, num_verifier_layers - 3]` 使用 `verifier_config.num_hidden_layers`，剥离后值为 48（正确）。
- `hidden_size`、`vocab_size` 等也都正确对应文本塔。

**验收标准**：
- `DFlashDraftModel.from_training_args(verifier="Qwen/Qwen3-Omni-30B-A3B-Thinking", ...)` 不再抛 AttributeError，且 `self.config.target_hidden_size == 2048`。

---

### Step 4：`src/speculators/models/dflash/config.py` — `qwen3_omni_moe_text` 降级

**文件**：`src/speculators/models/dflash/config.py`
**位置**：`validate_transformer_config`（第 79-90 行）

**现状问题**：

```python
if "model_type" in value:
    config_class = AutoConfig.for_model(
        model_type=value["model_type"]
    ).__class__
return config_class(**value)
```

- 当 `value["model_type"] == "qwen3_omni_moe_text"`，`AutoConfig.for_model` 会返回 `Qwen3OmniMoeTextConfig`（MoE），但 draft **希望走稠密 Qwen3Config**（因为 `Qwen3DFlashDecoderLayer` 继承 `Qwen3MLP`）。
- 此时传 `intermediate_size=768` 给 Dense `Qwen3MLP`，FFN 严重欠拟合。

**修改方案**：白名单降级 + 字段裁剪。

```python
_MOE_TO_DENSE_MAP = {
    # MoE text_config -> 稠密 Qwen3Config
    "qwen3_omni_moe_text": Qwen3Config,
    # 未来可扩展：qwen3_moe 等
}

# 稠密 Qwen3Config 可接受的字段（避免 MoE 专用字段污染）
_QWEN3_DENSE_WHITELIST = {
    "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim", "hidden_act",
    "max_position_embeddings", "initializer_range", "rms_norm_eps",
    "rope_theta", "rope_scaling", "tie_word_embeddings", "attention_bias",
    "attention_dropout", "use_qk_norm", "torch_dtype", "model_type",
}

@field_validator("transformer_layer_config", mode="before")
@classmethod
def validate_transformer_config(cls, value: Any) -> PretrainedConfig:
    if isinstance(value, dict):
        model_type = value.get("model_type")
        if model_type in _MOE_TO_DENSE_MAP:
            # MoE text config -> 降级到稠密等价 draft config
            config_class: type[PretrainedConfig] = _MOE_TO_DENSE_MAP[model_type]
            filtered = {k: v for k, v in value.items() if k in _QWEN3_DENSE_WHITELIST}
            # model_type 覆盖为 qwen3（让下游识别为稠密）
            filtered["model_type"] = "qwen3"
            return config_class(**filtered)

        config_class = Qwen3Config
        if model_type is not None:
            config_class = AutoConfig.for_model(model_type=model_type).__class__
        return config_class(**value)
    return value
```

**验收标准**：
- 给 `validate_transformer_config` 喂一个 `{"model_type":"qwen3_omni_moe_text","moe_intermediate_size":768,"num_experts":128,...}` 返回的 `Qwen3Config`：
  - 不含 `num_experts`、`moe_intermediate_size` 字段
  - `rope_scaling` 透传
  - `model_type == "qwen3"`

> 注：Step 1 已把 MoE intermediate_size 换算成 Dense 的 6144 再写入 `Qwen3Config`，从运行 path 进入时此白名单主要保护"序列化→反序列化 checkpoint" 的回路。

---

### Step 5：`src/speculators/models/dflash/model_definitions.py` — MRoPE 适配

**文件**：`src/speculators/models/dflash/model_definitions.py`
**位置**：第 22-43 行（`_rotate_half` / `apply_rotary_pos_emb`），以及第 130 行调用处。

**现状问题**：

```python
def _rotate_half(x):
    ...
def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos[..., -q_len:, :]) + (_rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (_rotate_half(k) * sin)
```

- 是标准 1D RoPE 实现，假设 `cos/sin` shape = `[..., seq, head_dim]`。
- Qwen3-Omni 的 MRoPE 会让上游 `Qwen3RotaryEmbedding` 产出 shape = `[3, ..., seq, head_dim]`（T/H/W 三段 position），`apply_rotary_pos_emb` 需要按 `mrope_section` 在 head_dim 维度分段再拼回。

**修改方案**：

优先级 1（推荐）：**直接复用 transformers 的 `apply_multimodal_rotary_pos_emb`**，保证和 verifier 行为一致。

```python
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    apply_multimodal_rotary_pos_emb,
)

def apply_rotary_pos_emb(q, k, cos, sin, mrope_section=None):
    """
    Unified RoPE dispatcher.
    - mrope_section is None  -> standard 1D RoPE (legacy behavior)
    - mrope_section is list  -> multimodal RoPE (Qwen3-Omni / Qwen2-VL)
    """
    if mrope_section is None:
        # 原 1D RoPE 实现保留
        q_len = q.shape[-2]
        q_embed = (q * cos[..., -q_len:, :]) + (_rotate_half(q) * sin[..., -q_len:, :])
        k_embed = (k * cos) + (_rotate_half(k) * sin)
        return q_embed, k_embed

    return apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section=mrope_section)
```

**调用点**（第 130 行附近）：

```python
# Qwen3DFlashAttention.forward 内
mrope_section = None
rope_scaling = getattr(self.config, "rope_scaling", None) or {}
if rope_scaling.get("rope_type") in (None, "default") and "mrope_section" in rope_scaling:
    mrope_section = rope_scaling["mrope_section"]

q, k = apply_rotary_pos_emb(q, k, cos, sin, mrope_section=mrope_section)
```

**`Qwen3RotaryEmbedding` 初始化**（`core.py` 第 88 行）：

- `transformers.Qwen3RotaryEmbedding` 本身读 `config.rope_scaling`/`config.rope_theta`，Step 1/Step 4 把这俩字段透传进 `Qwen3Config` 后，**无需修改**该构造行，RoPE 参数会自动生效。

**验收标准**：
- 用真实 Qwen3-Omni input_ids 与 3D position_ids 跑一次 forward：不报 shape 错误，且 `q_embed` shape 与原 verifier 路径一致。
- 关掉 MRoPE（`rope_scaling=None`）时退回 1D RoPE，单元测试保持绿色。

---

### Step 6：`src/speculators/data_generation/custom_worker.py` — hidden state 捕获点

**文件**：`src/speculators/data_generation/custom_worker.py`
**位置**：`_setup_hidden_states_capture` 第 121-132 行

**现状**：

```python
base_model = model.model  # default
if hasattr(model, "get_language_model"):
    # VLM
    base_model = model.get_language_model().model
```

**问题**：
- Qwen3-Omni 的 `Qwen3OmniMoeForConditionalGeneration` 没有 `get_language_model()`，其结构是 `model.thinker.model.layers[i]`（`thinker` 是 `Qwen3OmniMoeThinkerForConditionalGeneration`）。
- 需要新增 thinker 分支。

**修改**：

```python
base_model = model.model  # default (Llama/Qwen dense)
if hasattr(model, "thinker"):
    # Qwen3-Omni family
    thinker = model.thinker
    if hasattr(thinker, "get_language_model"):
        base_model = thinker.get_language_model().model
    elif hasattr(thinker, "model"):
        base_model = thinker.model.model if hasattr(thinker.model, "model") \
                     else thinker.model
elif hasattr(model, "get_language_model"):
    # Qwen3VL / Llama-VL
    base_model = model.get_language_model().model
```

**验收标准**：
- 对 Qwen3-Omni-Thinking 实例化后：`base_model.layers[0]` 是 `Qwen3OmniMoeTextDecoderLayer`（48 层）。
- `target_layer_ids=[2, 23, 45]` 上挂的 forward hook 能正确捕获 `(hidden_states + residual).clone()`。

---

### Step 7：新增 Qwen3-Omni-Thinking 示例脚本

**文件**：`examples/data_generation_and_training/qwen3_omni_thinking_sharegpt.py`（新建）

**模板**：参考已有 `qwen3_8b_sharegpt_ultrachat.py`，差异点：

```python
# 关键配置
VERIFIER = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
DRAFT_ARCH = "qwen3"         # 稠密
NUM_DRAFT_LAYERS = 1
DRAFT_INTERMEDIATE_SIZE = 6144  # moe_intermediate_size(768) * num_experts_per_tok(8)
TARGET_LAYER_IDS = [2, 23, 45]  # 避开 DeepStack [8, 16, 24]
MASK_TOKEN_ID = 151671          # 复用 tts_pad_token_id（Thinking 关闭 TTS）
BLOCK_SIZE = 8
MAX_ANCHORS = 256
DRAFT_VOCAB_SIZE = 32000

# 数据生成：用文本-only 的 sharegpt / ultrachat，不必引入音视频多模态
```

并更新 `examples/data_generation_and_training/README.md` 增加一行链接。

**验收标准**：
- 脚本可一键跑通：hidden states 生成 → draft 训练 → checkpoint 保存。

---

## 4. 端到端验证流程

```bash
# 1. launch vLLM verifier（对 vLLM 侧 DFlash 尚未适配，先只做 hidden state 抽取，不做 spec decode）
python scripts/launch_vllm.py \
    --verifier Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --target-layer-ids 2 23 45 \
    --serve-only

# 2. 数据生成（hidden states + prompt + target logits）
python examples/data_generation_and_training/qwen3_omni_thinking_sharegpt.py generate

# 3. 训练 DFlash draft
python scripts/train.py \
    --verifier Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --data-path ./data/qwen3_omni_thinking_sharegpt \
    --draft-arch qwen3 \
    --num-layers 1 \
    --draft-intermediate-size 6144 \
    --target-layer-ids 2 23 45 \
    --mask-token-id 151671 \
    --block-size 8 \
    --max-anchors 256 \
    --draft-vocab-size 32000 \
    --output-dir ./checkpoints/dflash_qwen3_omni_thinking

# 4. 加载回测（smoke test，跳过 vLLM 侧推理）
python -c "
from speculators.models.dflash import DFlashDraftModel
m = DFlashDraftModel.from_pretrained('./checkpoints/dflash_qwen3_omni_thinking')
print(m.config)
"
```

**必过检查点**：
1. Step 3 生效：`m.config.target_hidden_size == 2048`
2. Step 1 生效：`m.config.transformer_layer_config.intermediate_size == 6144`
3. Step 4 生效：`m.config.transformer_layer_config.model_type == 'qwen3'` 且无 MoE 字段
4. Step 5 生效：训练 forward 无 RoPE shape 错误；loss 在前 200 step 下降
5. Step 6 生效：数据生成阶段捕获的 hidden states 三层形状都是 `[*, 2048]`

---

## 5. 风险与回退

### 5.1 MRoPE 行为一致性

- **风险**：若 `apply_multimodal_rotary_pos_emb` 在不同 transformers 版本间有 API 变动，draft 与 verifier 可能产生位置编码偏差。
- **回退**：Step 5 中保留 1D RoPE 分支（`mrope_section is None`），允许通过 `--disable-mrope` CLI 强制走 1D RoPE 训练一版基线作为下界。
- **兜底**：在 `model_definitions.py` 中复制一份 transformers 当前版本的 `apply_multimodal_rotary_pos_emb` 源码，避免远端升级造成训练漂移。

### 5.2 `mask_token_id` 选择

- **首选**：`--mask-token-id 151671`（tts_pad_token_id，Thinking 版 TTS 关闭）
- **备选 1**：扩展 tokenizer 新增 `<|MASK|>`，同时 verifier embedding / lm_head 需用 `resize_token_embeddings`——**不推荐**（会污染 verifier checkpoint）
- **备选 2**：复用 `pad_token_id` / `eos_token_id`——可能与训练样本冲突

### 5.3 MoE → Dense 的容量差

- **风险**：`moe_intermediate_size * num_experts_per_tok = 6144` 只是参数量等价近似，verifier 实际激活量随路由稀疏而变；6144 的稠密 FFN 可能仍欠表达。
- **回退**：允许 `--draft-intermediate-size 8192` 或 `12288` 观察 loss / acceptance rate 曲线后定档；也可把 draft 改成 2 层 `hidden_size=1024` 的 "窄深" 变体。

### 5.4 DeepStack 层污染 hidden state

- **风险**：层 [8,16,24] 的 hidden state 中注入了视觉侧融合信号。纯文本场景下问题不大，但若训练集含图像/音频，draft 将学到 verifier 文本塔中混入的视觉特征。
- **回退**：Step 2 已在默认值上避让，并强烈建议训练数据用纯文本（Thinking 场景主场）。

### 5.5 vLLM 侧尚未适配

- 本 plan 完成后，**只能保证 speculators 端能产出合法的 DFlash checkpoint**，vLLM 线上推理仍需单独改造（`qwen3_omni_thinking_dflash.py`、`DFlashProposer._raise_if_multimodal`、attention backend non-causal 支持等）。
- 本 plan 的 checkpoint 格式已与现有 `DFlashQwen3ForCausalLM.load_weights` 的 `midlayer.*` / `d2t` key 兼容（因为 draft 本体仍是 `Qwen3DFlashDecoderLayer`），可最大限度减少 vLLM 侧后续适配成本。

---

## 6. 改动文件总览（checklist）

- [ ] `scripts/train.py` — Step 1
- [ ] `scripts/launch_vllm.py` — Step 2
- [ ] `src/speculators/models/dflash/core.py` — Step 3
- [ ] `src/speculators/models/dflash/config.py` — Step 4
- [ ] `src/speculators/models/dflash/model_definitions.py` — Step 5
- [ ] `src/speculators/data_generation/custom_worker.py` — Step 6
- [ ] `examples/data_generation_and_training/qwen3_omni_thinking_sharegpt.py` — Step 7（新增）
- [ ] `examples/data_generation_and_training/README.md` — Step 7（更新链接）

**明确不改**：

- `src/speculators/utils/loading.py`（已支持 suffix 匹配）
- `src/speculators/train/utils.py`（`resolve_mask_token_id` 四级回退已足够）
- `src/speculators/train/data.py` / `src/speculators/models/dflash/metrics.py` / `src/speculators/models/dflash/utils.py`（架构无关）

---

## 7. 下一步

完成上述 7 个 Step 后：

1. 跑一个 **小规模 smoke test**（ShareGPT 500 条 + 50 步训练）验证 pipeline 通畅。
2. 扩展到 **完整训练**（5k ~ 50k 样本），观察 acceptance rate。
3. 启动 **vLLM 侧适配文档**：`plan_vllm.md`（目标：`Qwen3OmniMoeThinkerForConditionalGeneration` + `DFlashQwen3ForCausalLM` 在 vLLM V1 上能端到端 spec decode）。
