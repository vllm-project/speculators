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
- `src/speculators/models/dflash/core.py` — 运行时 verifier config 解析 + RoPE 初始化（MRoPE 时选用 `Qwen3OmniMoeThinkerTextRotaryEmbedding`）
- `src/speculators/models/dflash/config.py` — `validate_transformer_config` 对 `qwen3_omni_moe_text` 的降级处理
- `src/speculators/models/dflash/model_definitions.py` — `apply_rotary_pos_emb` 保持 1D RoPE 单一路径（保留 q 端尾切片以支持 `q_len != k_len` 的非对称 attention）
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

### Step 5：`src/speculators/models/dflash/core.py` + `model_definitions.py` — MRoPE 适配

**文件**：
- `src/speculators/models/dflash/core.py`（RotaryEmbedding 分派）
- `src/speculators/models/dflash/model_definitions.py`（`apply_rotary_pos_emb`）

**关键认知**（直接决定设计方向）：

Qwen3-Omni 的 MRoPE **不是**在 `apply_rotary_pos_emb` 里做分段的。所有多模态分段（T/H/W 三路 freq 的 interleave）**在 `Qwen3OmniMoeThinkerTextRotaryEmbedding.forward` 内部就已经通过 `apply_interleaved_mrope` 完成**，它返回的 `cos/sin` 形状是标准的 `(bsz, seq_len, head_dim)`，**和 1D RoPE 完全一致**。验证方法 —— 查看 transformers 自带的 `Qwen3OmniMoeThinkerTextAttention.forward`，它对 `cos/sin` 的使用就是一句 `apply_rotary_pos_emb(q, k, cos, sin)`，没有任何 `mrope_section` 逻辑。

因此 MRoPE 支持的**唯一改动点**是：**把 `self.rotary_emb` 从 `Qwen3RotaryEmbedding` 换成 `Qwen3OmniMoeThinkerTextRotaryEmbedding`**。`apply_rotary_pos_emb` 一端**不需要任何分支**，只需要维持 DFlash 原本就有的 `q_len` 尾切片以适应 `q_len != k_len` 的非对称 attention。

#### 5.1 `core.py` 的 RotaryEmbedding 分派

```python
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextRotaryEmbedding,
)

# DFlashDraftModel.__init__ 内
rope_scaling = getattr(config.transformer_layer_config, "rope_scaling", None)
if isinstance(rope_scaling, dict) and "mrope_section" in rope_scaling:
    self.rotary_emb = Qwen3OmniMoeThinkerTextRotaryEmbedding(
        config.transformer_layer_config
    )
else:
    self.rotary_emb = Qwen3RotaryEmbedding(config.transformer_layer_config)
```

`Qwen3OmniMoeThinkerTextRotaryEmbedding` 的 `__init__` 只读 `rope_scaling["mrope_section"]`（缺省会 fallback `[24,20,20]`）、`rope_theta`、`max_position_embeddings`，Step 1 已经把这些字段透传到 draft `Qwen3Config` 里，**不需要再传 `Qwen3OmniMoeTextConfig`**，稠密的 `Qwen3Config` 足够构造它。

#### 5.2 `model_definitions.py` 的 `apply_rotary_pos_emb`

**保持单一代码路径，唯一重点是 q 端的尾切片**：

```python
def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary position embeddings for DFlash's asymmetric attention.

    NOTE on MRoPE (Qwen3-Omni):
        Multimodal section interleaving is fully handled inside
        ``Qwen3OmniMoeThinkerTextRotaryEmbedding.forward`` (via
        ``apply_interleaved_mrope``). The returned ``cos/sin`` already have
        the standard 1D-RoPE shape ``(bsz, seq_len, head_dim)``, so a single
        RoPE code path suffices - matching Qwen3-Omni's own attention.

    NOTE on the q-side slicing:
        DFlash attention is asymmetric - ``k`` has length ``ctx_len + q_len``
        while ``q`` has length ``q_len``. ``cos/sin`` are built from the
        concatenated ``position_ids`` and therefore align with ``k``. We must
        slice the last ``q_len`` positions for ``q`` so that the RoPE applied
        on ``q`` corresponds to the noise-block positions at the tail of
        ``position_ids``.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (_rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed
```

**调用点**（`Qwen3DFlashAttention.forward`）保持最简：

```python
cos, sin = position_embeddings
q, k = apply_rotary_pos_emb(q, k, cos, sin)
```

不要再从 `self.config.rope_scaling` 读 `mrope_section` 做分支判断 —— MRoPE 的适配已经在 RotaryEmbedding 侧闭环。

#### 5.3 常见误区与反例（避坑）

**误区**：以为 `transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe.apply_rotary_pos_emb` 是 MRoPE 专用实现，于是在 DFlash 里基于 `mrope_section is not None` 切到它。

**事实**：该函数只是标准 1D RoPE（`(q*cos) + (rotate_half(q)*sin)`），**不做任何 mrope_section 相关的处理**，并且**不做 q 端尾切片**。如果切过去，会在 `q_len != k_len` 时因为 broadcast 失败直接 crash：

```
RuntimeError: The size of tensor a (q_len) must match the size of tensor b
(ctx_len + q_len) at non-singleton dimension 2
```

所以**既不要 import 它，也不要添加 `mrope_section` 分支**，维持单一 RoPE 路径即可。

#### 5.4 验收标准

- 用 Qwen3-Omni-Thinking 的 `rope_scaling={"mrope_section":[24,20,20], "mrope_interleaved":true, "rope_type":"default"}` 构造 draft `Qwen3Config`，`core.py` 能正确选中 `Qwen3OmniMoeThinkerTextRotaryEmbedding`。
- 用 `(q_len=16, k_len=64, head_dim=128)` 的 shape 调用 `apply_rotary_pos_emb`，输出 shape 分别为 `q=(1,32,16,128)`、`k=(1,32,64,128)`，不抛异常。
- 关掉 MRoPE（`rope_scaling=None` 或无 `mrope_section`）时自动回落到 `Qwen3RotaryEmbedding`，原单元测试保持绿色。

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
4. Step 5 生效：
   - `DFlashDraftModel.rotary_emb` 实际类型为 `Qwen3OmniMoeThinkerTextRotaryEmbedding`（当 `rope_scaling` 含 `mrope_section` 时）
   - 训练 forward 无 RoPE shape 错误，即使 `q_len != k_len` 也正常工作
   - loss 在前 200 step 下降
5. Step 6 生效：数据生成阶段捕获的 hidden states 三层形状都是 `[*, 2048]`

---

## 5. 风险与回退

### 5.1 MRoPE 行为一致性

- **风险**：`Qwen3OmniMoeThinkerTextRotaryEmbedding` 跨 transformers 版本的 `apply_interleaved_mrope` 实现若发生变化（例如 mrope_section 默认值、interleave 顺序调整），会导致 draft 的 cos/sin 与 verifier 不再对齐。
- **回退**：`core.py` 的 dispatch 是纯运行时判断（`"mrope_section" in rope_scaling`），可通过 CLI 强制传 `rope_scaling=None` 让 draft 退回 `Qwen3RotaryEmbedding` 的标准 1D RoPE，训练一版基线作为下界。
- **兜底**：必要时在 `speculators/models/dflash/` 下镜像一份当前 transformers 版本的 `Qwen3OmniMoeThinkerTextRotaryEmbedding` 源码，避免远端升级造成训练漂移。
- **反模式提示**：不要通过"在 `apply_rotary_pos_emb` 里按 `mrope_section` 切分支"的方式适配 MRoPE —— transformers 的 `qwen3_omni_moe.apply_rotary_pos_emb` 并不是 MRoPE 专用函数，且不做 DFlash 所需的 q 端尾切片，误用会在 `q_len != k_len` 时直接 crash。MRoPE 的分段完全在 RotaryEmbedding 侧完成。

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
- [ ] `src/speculators/models/dflash/core.py` — Step 3 + Step 5.1（RotaryEmbedding 分派）
- [ ] `src/speculators/models/dflash/config.py` — Step 4
- [ ] `src/speculators/models/dflash/model_definitions.py` — Step 5.2（保持单一 1D RoPE 路径 + q 端尾切片）
- [ ] `src/speculators/data_generation/custom_worker.py` — Step 6
- [ ] `examples/data_generation_and_training/qwen3_omni_thinking_sharegpt.py` — Step 7（新增）
- [ ] `examples/data_generation_and_training/README.md` — Step 7（更新链接）

**明确不改**：

- `src/speculators/utils/loading.py`（已支持 suffix 匹配）
- `src/speculators/train/utils.py`（`resolve_mask_token_id` 四级回退已足够）
- `src/speculators/train/data.py` / `src/speculators/models/dflash/metrics.py` / `src/speculators/models/dflash/utils.py`（架构无关）

---

## 7. Phase 2 — 多模态训练数据支持

> 前 6 个 Step（Phase 1）覆盖的是**文本-only**的 Qwen3-Omni-Thinking DFlash 训练闭环。Phase 2 的目标：**让同一套 DFlash draft 能直接消费含图像 / 视频 / 音频的 ShareGPT-Vision / LLaVA / video-ultrachat 等多模态训练样本**，保证 draft 学到的不仅是文本 token 的 next-token 分布，也包括**视觉 / 音频 placeholder token → 后续文本 token** 的条件分布。

### 7.0 为什么 Phase 1 的 pipeline 不能直接处理多模态数据

当前 pipeline 的 4 个关键假设在多模态场景下全部失效：

| 组件 | Phase 1 假设 | 多模态下的现实 |
|---|---|---|
| `preprocessing.py::_preprocess_batch` | 每条样本只有 `conversations: list[{role, content: str}]` | content 可能是 `list[{"type":"image/video/audio/text", ...}]` |
| vLLM 请求 (`vllm_client.generate_hidden_states` / `VllmHiddenStatesGenerator.generate`) | 只传 `prompt_token_ids: list[int]` | 需要 `pixel_values / image_grid_thw / input_features / feature_attention_mask / video_grid_thw / second_per_grids / use_audio_in_video` 等一整套 multimodal kwargs 以及和 placeholder token 一一对应的 `input_ids` |
| `custom_worker._patched_forward` | `self.embed_input_ids(input_ids)` 后直接进 decoder | 必须先通过 `thinker.get_image_features / get_video_features / get_audio_features` 把视觉/音频特征 scatter 进 `inputs_embeds`，再进 decoder；且需要 `deepstack_visual_embeds` 注入到 [8, 16, 24] 层 |
| `data.py::BaseDataset.__getitem__` | `position_ids = torch.arange(seq_len)` | 必须改成 `(3, seq_len)` 的 MRoPE 3D position_ids（来自 `get_rope_index`），否则 Step 5 装上的 `Qwen3OmniMoeThinkerTextRotaryEmbedding` 会把所有 modality token 当纯文本编码 |

下文所有 Step 都围绕**修补上表中的 4 个断点**展开。

---

### Step 8：多模态数据预处理 —— `preprocessing.py` 支持 `Qwen3OmniMoeProcessor`

**文件**：`src/speculators/data_generation/preprocessing.py`
**目标**：走 `processor.apply_chat_template(..., tokenize=True, return_dict=True)` 替代当前 `tokenizer.apply_chat_template` 分支，输出同时包含 `input_ids` / `loss_mask` / **多模态字段**的样本。

#### 8.1 新增 processor 分支

```python
# preprocessing.py 顶部
from transformers import AutoProcessor

def _is_multimodal_batch(examples: dict) -> bool:
    """Detect content-list style samples (image/video/audio segments)."""
    convs = examples.get("conversations", [])
    if not convs:
        return False
    for conv in convs:
        for turn in conv or []:
            content = turn.get("content") or turn.get("value")
            if isinstance(content, list) and any(
                isinstance(seg, dict) and seg.get("type") in {"image", "video", "audio"}
                for seg in content
            ):
                return True
    return False
```

`_preprocess_batch` 的新分支（伪代码）：

```python
if processor is not None and _is_multimodal_batch(examples):
    encoded = processor.apply_chat_template(
        normalized_conv,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        # 关键：让 processor 根据 content 列表自动 load image/video/audio
        load_audio=True,
        load_image=True,
        load_video=True,
    )
    # encoded 里会同时出现以下字段中的子集：
    #   input_ids, attention_mask, pixel_values, image_grid_thw,
    #   pixel_values_videos, video_grid_thw, second_per_grids,
    #   input_features, feature_attention_mask
    input_ids = encoded["input_ids"][0]
    # loss_mask：沿用 regex 或 HF assistant mask，但要在 input_ids 上做
    # （placeholder token 的 loss_mask 必须为 0，只训练 assistant 文本 token）
    loss_mask = _build_multimodal_loss_mask(
        input_ids,
        tokenizer=processor.tokenizer,
        assistant_pattern=assistant_pattern,
        image_token_id=processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        video_token_id=processor.tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        audio_token_id=processor.tokenizer.convert_tokens_to_ids("<|audio_pad|>"),
    )
    results["input_ids"].append(input_ids)
    results["loss_mask"].append(loss_mask)
    results["seq_len"].append(len(input_ids))
    # 多模态附加字段（保存为 per-sample list of tensors，后续写入 arrow）
    for k in ("pixel_values", "image_grid_thw",
              "pixel_values_videos", "video_grid_thw", "second_per_grids",
              "input_features", "feature_attention_mask"):
        if k in encoded:
            results.setdefault(k, []).append(encoded[k])
```

`_build_multimodal_loss_mask` 在原有 assistant span 检测基础上**额外把所有 `image_token_id / video_token_id / audio_token_id` 置 0**，避免 draft 去学重建 placeholder。

#### 8.2 `build_eagle3_dataset` 接受 `processor`

```python
def build_eagle3_dataset(
    dataset, tokenizer, *,
    processor: "ProcessorMixin | None" = None,
    ...
):
    ...
```

`load_and_preprocess_dataset` 若 `target_model_path` 的 config 存在 `image_token_id` / `audio_token_id`，自动 `AutoProcessor.from_pretrained(..., trust_remote_code=True)` 并传入。

#### 8.3 `DATASET_CONFIGS` 扩展多模态数据源

在 `configs.py` 增加（示例）：

```python
"llava-instruct": DatasetConfig(
    name="llava-instruct",
    hf_path="liuhaotian/LLaVA-Instruct-150K",
    split="train",
    normalize_fn=_normalize_llava_instruct,
    # normalize_fn 把 {image, conversations} 转成统一的 {conversations: [{role, content:[{type:image,image:PIL}, {type:text,text:...}]}]}
),
"video-chatgpt": DatasetConfig(...),
"audio-caps": DatasetConfig(...),
```

**验收标准**：
- 一条含 1 张图 + 3 轮对话的样本经 `build_eagle3_dataset` 后产生：
  - `input_ids` 长度 = prompt text tokens + **展开后的视觉 placeholder tokens**（`image_grid_thw[0,0]*image_grid_thw[0,1]*image_grid_thw[0,2] / (spatial_merge_size ** 2)`）
  - `loss_mask[image_pad_positions] == 0`
  - 样本 dict 同时带 `pixel_values (N_patches, C*P*P)` + `image_grid_thw (1, 3)`

---

### Step 9：Arrow 数据落盘 & DataLoader 还原多模态字段

**文件**：
- `src/speculators/data_generation/preprocessing.py`（保存逻辑）
- `src/speculators/train/data.py`（读取逻辑）

#### 9.1 落盘

HuggingFace `datasets` 对可变长 2D/3D 张量支持不佳。采用**双轨制**：

- 可变长的 `pixel_values` / `input_features` / `pixel_values_videos` → 单独以 `safetensors` 文件保存到 `{datapath}/multimodal/sample_{idx}.safetensors`，Arrow 列只存 `mm_file: str` 指针。
- 定长的 `image_grid_thw / video_grid_thw / second_per_grids / feature_attention_mask_len` → 直接存 Arrow 列。

#### 9.2 `ArrowDataset._get_raw_data` 扩展

```python
def _get_raw_data(self, index):
    base = ...  # 现有逻辑得到 input_ids / loss_mask / hidden_states / verifier_last_hidden_states
    mm_path = self.data[index].get("mm_file")
    if mm_path:
        mm = load_file(Path(self.datapath) / mm_path)
        base.update({
            "pixel_values": mm.get("pixel_values"),
            "image_grid_thw": mm.get("image_grid_thw"),
            "pixel_values_videos": mm.get("pixel_values_videos"),
            "video_grid_thw": mm.get("video_grid_thw"),
            "second_per_grids": mm.get("second_per_grids"),
            "input_features": mm.get("input_features"),
            "feature_attention_mask": mm.get("feature_attention_mask"),
        })
    return base
```

#### 9.3 `BaseDataset.__getitem__` 生成 3D MRoPE position_ids

这是多模态能跑的**最关键**一处修改：

```python
# 替换原来的 torch.arange(seq_len)
if any(k in data for k in ("image_grid_thw", "video_grid_thw")):
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )
    # 复用 verifier 的 get_rope_index（静态方法化或缓存一份 dummy 实例）
    position_ids, _ = self._rope_index_fn(
        input_ids=data["input_ids"].unsqueeze(0),
        image_grid_thw=data.get("image_grid_thw"),
        video_grid_thw=data.get("video_grid_thw"),
        second_per_grids=data.get("second_per_grids"),
        use_audio_in_video=bool(data.get("input_features") is not None),
        audio_seqlens=(
            data["feature_attention_mask"].sum(-1)
            if "feature_attention_mask" in data else None
        ),
    )
    data["position_ids"] = position_ids.squeeze(1)  # (3, seq_len)
else:
    data["position_ids"] = torch.arange(seq_len, dtype=torch.long)
    # 1D 情况 Qwen3OmniMoeThinkerTextRotaryEmbedding 会自动 expand 到 (3, bs, seq)
```

`self._rope_index_fn` 在 `BaseDataset.__init__` 中一次性构造：

```python
if verifier_name_or_path is not None:
    cfg = AutoConfig.from_pretrained(verifier_name_or_path)
    if hasattr(cfg, "thinker_config"):  # Qwen3-Omni
        # 仅取 get_rope_index 方法 + 所需常量，不实例化整个 thinker
        self._rope_index_fn = _make_rope_index_fn(cfg.thinker_config)
```

`_make_rope_index_fn` 把 `image_token_id / video_token_id / audio_token_id / vision_start_token_id / audio_start_token_id / position_id_per_seconds / spatial_merge_size` 从 config 读出来，包装成纯函数（无需权重）。

#### 9.4 `create_collate_fn` 处理可变长多模态张量

多模态张量**不能**按 `input_ids` 的"cat + pad 到 max_len"方式处理。规则：

| 字段 | 处理策略 |
|---|---|
| `input_ids / loss_mask / hidden_states / verifier_last_hidden_states` | 原逻辑（cat + pad 到 `max_len`） |
| `position_ids` | 若形状 `(3, L)`：按 `dim=1` cat，再 pad 到 `max_len`，结果 `(3, max_len)`；否则沿用 1D 分支 |
| `pixel_values / pixel_values_videos / input_features` | **跨样本 concat** 成 `(sum_N_patches, ...)`，同时维护 `image_sample_offsets`，在后续 scatter 时用 |
| `image_grid_thw / video_grid_thw / second_per_grids` | 按 `dim=0` cat |
| `feature_attention_mask` | 按 `dim=0` cat，保存原始 shape |

---

### Step 10：`custom_worker._patched_forward` 支持多模态 `inputs_embeds`

**文件**：`src/speculators/data_generation/custom_worker.py`
**位置**：`_patched_forward`（现在拿 `input_ids` 直接 `self.embed_input_ids`）

**修改思路**：patch 的不再是 `base_model`（即 `thinker.model`）的 forward，而是**在 `thinker` 这一层做拦截**，让 vision / audio tower 先行、hidden state capture 还在 text backbone 层生效。

```python
def _patched_thinker_forward(self, *args, **kwargs):
    # self == thinker (Qwen3OmniMoeThinkerForConditionalGeneration)
    # 1) 调用原始 forward 的前半段：计算 inputs_embeds 并 scatter 视觉/音频特征
    pixel_values = kwargs.pop("pixel_values", None)
    image_grid_thw = kwargs.pop("image_grid_thw", None)
    pixel_values_videos = kwargs.pop("pixel_values_videos", None)
    video_grid_thw = kwargs.pop("video_grid_thw", None)
    input_features = kwargs.pop("input_features", None)
    feature_attention_mask = kwargs.pop("feature_attention_mask", None)
    input_ids = kwargs["input_ids"]

    inputs_embeds = self.model.get_input_embeddings()(input_ids)
    deepstack_visual_embeds = None

    if pixel_values is not None:
        image_embeds, ds_embeds = self._process_image_input({
            "pixel_values": pixel_values, "image_grid_thw": image_grid_thw,
        })
        mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(mask, image_embeds)
        deepstack_visual_embeds = ds_embeds  # list[Tensor]，对应 layers [8,16,24]

    if pixel_values_videos is not None:
        ...  # 类似 image 分支
    if input_features is not None:
        audio_embeds = self.get_audio_features(input_features, feature_attention_mask)
        mask = (input_ids == self.config.audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(mask, audio_embeds)

    # 2) 交给 text model，forward 会走到我们打了 hook 的 base_model.forward
    kwargs["inputs_embeds"] = inputs_embeds
    kwargs["input_ids"] = None
    if deepstack_visual_embeds is not None:
        kwargs["deepstack_visual_embeds"] = deepstack_visual_embeds
    return self._orig_forward(*args, **kwargs)
```

并调整 `_setup_hidden_states_capture`：

```python
if hasattr(model, "thinker"):
    thinker = model.thinker
    # 保存原 forward，挂壳
    thinker._orig_forward = thinker.forward
    thinker.forward = types.MethodType(_patched_thinker_forward, thinker)
    base_model = thinker.model  # Qwen3OmniMoeThinkerTextModel
    # 继续 patch base_model.forward 做 hidden state capture（原逻辑）
```

另需把 `_patched_forward`（text backbone 侧）签名扩展以接受 `deepstack_visual_embeds`，并在 `target_layer_ids ∈ [8,16,24]` 命中时按 DeepStack 规则融合（若 `target_layer_ids` 已规避 DeepStack 则直接丢给原 decoder）。

**验收标准**：
- 一条"1 图 + 1 段文字"样本经 patched thinker forward 后，text backbone 收到的 `inputs_embeds` 在 image placeholder 位置上数值等于 `visual(pixel_values)` 的输出。
- 三层 `target_layer_ids=[2, 23, 45]` 捕获的 hidden state 形状 `[seq_len, hidden_size]`，seq_len 与 Step 8 里计算的 placeholder-展开后 input_ids 长度一致。

---

### Step 11：vLLM client 侧携带多模态数据

**文件**：`src/speculators/data_generation/vllm_client.py`
**现状**：`generate_hidden_states(client, model, token_ids, ...)` 只传 `prompt=token_ids`。

**修改**：走 vLLM OpenAI-compatible 的 `chat.completions.create(..., messages=...)`，由 vLLM server 端的 `Qwen3OmniMoeProcessor` 解析 multimodal content：

```python
def generate_hidden_states_multimodal(
    client,
    model: str,
    messages: list[dict],   # 含 image_url / video_url / audio_url 的 chat messages
    timeout=None,
) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1,
        extra_body={"return_token_ids": True},
        timeout=timeout,
    )
    return extract_output_no_token_check(completion)  # placeholder 展开后 token_ids 与原始不完全等长，改为 hidden_states_path 校验
```

相应地 `ArrowDataset._maybe_generate_hs` 分支：若样本带 `messages`（多模态 chat）走新函数；否则走旧的 token_ids completion API。

> 注：这一步要求 vLLM 侧 verifier 是用 `Qwen3-Omni-30B-A3B-Thinking` 原生模型**而非** draft 模型启动——本 plan Phase 1 已经是这个前提，无需改。但 `launch_vllm.py` 需要把 `--limit-mm-per-prompt image=N,video=N,audio=N` 之类的 vLLM 多模态参数透传出去。

---

### Step 12：`DFlashDraftModel.forward` 的 position_ids 广播修正

**文件**：`src/speculators/models/dflash/core.py`
**位置**：第 299-307 行（position_ids 拼接 + rotary_emb 调用）

**现状**（Phase 1 假设 1D position_ids）：

```python
mask_position_ids = get_base_indices_for_anchored_blocks(
    position_ids[:, anchor_positions], self.block_size, input_ids.numel()
)
position_ids = torch.cat([position_ids, mask_position_ids.unsqueeze(0)], dim=1)
position_embeddings = self.rotary_emb(hidden_states, position_ids)
```

**问题**：多模态场景下 `position_ids.shape == (3, 1, total_seq_len)`（collate 已处理），当前 `position_ids[:, anchor_positions]` 的索引语义错误，`torch.cat(..., dim=1)` 会沿 batch 维而不是 seq 维拼接。

**修改**：

```python
is_mrope = position_ids.dim() == 3  # (3, 1, L)
if is_mrope:
    # anchor_positions 形状 [num_anchors]，在 seq (dim=-1) 上索引
    anchor_pos_ids = position_ids[..., anchor_positions]  # (3, 1, num_anchors)
    # get_base_indices_for_anchored_blocks 返回 [num_anchors * block_size]
    mask_position_ids_1d = get_base_indices_for_anchored_blocks(
        anchor_pos_ids[0],  # 仅用 T 维算 anchor-block index offset 足够
        self.block_size, input_ids.numel(),
    )
    # 广播到三维：T/H/W 在 mask-block 内共享相同步进
    mask_position_ids = mask_position_ids_1d.view(1, 1, -1).expand(3, 1, -1)
    position_ids = torch.cat([position_ids, mask_position_ids], dim=-1)
else:
    # 原 1D 路径
    mask_position_ids = get_base_indices_for_anchored_blocks(
        position_ids[:, anchor_positions], self.block_size, input_ids.numel()
    )
    position_ids = torch.cat([position_ids, mask_position_ids.unsqueeze(0)], dim=1)

position_embeddings = self.rotary_emb(hidden_states, position_ids)
```

`Qwen3OmniMoeThinkerTextRotaryEmbedding.forward` 对 `position_ids.ndim == 3` 原生支持（transformers L1263-1264）。`ndim == 2` 时它会自动 `position_ids[None, ...].expand(3, ...)`，**因此 Phase 1 的文本样本走这条路径仍然正确**。

**验收标准**：
- 混合 batch（一半 text-only、一半含图）通过 `DFlashDraftModel.forward` 无 shape 错误。
- 对同一条纯文本样本，是否启用 MRoPE 两种模式下 RoPE 后的 `q/k` 数值等价（误差 < 1e-5）。

---

### Step 13：`scripts/train.py` / `gen_and_train.py` 的多模态开关

**文件**：
- `scripts/train.py`
- `scripts/gen_and_train.py`（`DataGenArgs` / `TrainArgs`）

**新增 CLI / args**：

```python
# scripts/train.py
parser.add_argument("--multimodal", action="store_true",
    help="Enable multimodal preprocessing via AutoProcessor. Required when "
         "training with image/video/audio datasets.")
parser.add_argument("--verifier-name-or-path", type=str, required=False,
    help="Used to construct get_rope_index in dataset; defaults to the value "
         "already present in speculators_config.verifier.name_or_path")
```

```python
# scripts/gen_and_train.py::DataGenArgs
multimodal: bool = False
# DATASET_CONFIGS key 里增加多模态数据集后，train_data_path 可以直接填 "llava-instruct"
```

`create_dataset(...)` 内部：当 `args.multimodal` 为 True 时把 `processor=AutoProcessor.from_pretrained(verifier)` 传进 `build_eagle3_dataset`，并把 `verifier_name_or_path` 传进 `ArrowDataset` 以让其初始化 `_rope_index_fn`。

---

### Step 14：新增多模态训练示例

**文件**：`examples/data_generation_and_training/qwen3_omni_thinking_llava.py`（新建）

```python
from gen_and_train import DataGenArgs, TrainArgs, VocabMappingArgs, run_e2e

if __name__ == "__main__":
    VERIFIER = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
    OUT = "./output/qwen3_omni_thinking_llava"
    TOTAL_SEQ_LEN = 16384  # 视觉 placeholder 会显著拉长，预算要放大

    AUX_TARGET_LAYER_IDS = [2, 23, 45]      # 继续避开 DeepStack [8, 16, 24]
    CAPTURE_LAYER_IDS = [*AUX_TARGET_LAYER_IDS, 48]
    DRAFT_VOCAB_SIZE = 32000

    data_gen_args = DataGenArgs(
        train_data_path="llava-instruct",   # Step 8.3 新增
        seq_length=TOTAL_SEQ_LEN,
        turn_dropout=True,
        layer_ids=CAPTURE_LAYER_IDS,
        multimodal=True,                    # Step 13 新开关
    )

    train_args = TrainArgs(
        speculator_type="dflash",
        draft_arch="qwen3",
        num_layers=1,
        draft_intermediate_size=6144,
        draft_vocab_size=DRAFT_VOCAB_SIZE,
        target_layer_ids=AUX_TARGET_LAYER_IDS,
        mask_token_id=151671,
        block_size=8,
        max_anchors=256,
        total_seq_len=TOTAL_SEQ_LEN,
        run_name="qwen3_omni_thinking_llava",
        lr=3e-5,
        epochs=3,
        logger="trackio",
    )

    run_e2e(
        verifier_name_or_path=VERIFIER,
        output_path=OUT,
        data_gen_args=data_gen_args,
        vocab_mapping_args=VocabMappingArgs(
            draft_vocab_size=DRAFT_VOCAB_SIZE, target_vocab_size=152064,
        ),
        train_args=train_args,
    )
```

---

### Phase 2 验收（端到端）

```bash
# 只有第 2 步需要能够处理 multimodal payload，其它和 Phase 1 相同
python examples/data_generation_and_training/qwen3_omni_thinking_llava.py
```

**必过检查点**：

1. Step 8：`build_eagle3_dataset` 对一条含 1 张图 + 3 段文本的样本产出 `input_ids` 长度严格等于 `text_tokens + (H/merge * W/merge * T)`。
2. Step 9：`ArrowDataset[i]["position_ids"].shape == (3, L)`，且 image placeholder 段的 T/H/W 维度互不相等（验证 3D RoPE 生效）。
3. Step 10：数据生成阶段 hidden state shape `[L, 2048]`，L 与 Step 8 产出一致，且 `[8, 16, 24]` 层有 DeepStack 注入（与 verifier 原生 forward 对比，L2 误差 < 1e-4）。
4. Step 12：`DFlashDraftModel.forward` 在 3D `position_ids` 下的 loss 曲线前 200 步下降；同时用纯文本样本做 A/B，验证 1D 路径未回退。
5. 整体：mixed-modality batch（2 条 text + 2 条 image）训练 1 个 epoch，acceptance rate 指标稳定（不出现 NaN / inf）。

---

### Phase 2 风险与回退

1. **placeholder 展开导致 OOM**：一张 448×448 的图 ≈ 1024 个 patch token，10 张图即可把 seq_len 推到 > 10K。
   - 回退：`TOTAL_SEQ_LEN` 预算拉大到 16K，并在 `preprocessing` 里 drop 掉展开后超出 `max_length` 的样本。
2. **`get_rope_index` 纯函数化失败**：若 transformers 版本的 `get_rope_index` 强依赖 `self.spatial_merge_size` 等 instance attr，改为**实例化一个不加载权重的 `Qwen3OmniMoeThinkerForConditionalGeneration.from_config(meta_init=True)`** 仅用于调用 `get_rope_index`；只有一次性 CPU meta init 开销。
3. **vLLM 侧的多模态 hidden state 抽取**：`launch_vllm.py` 当前走 `--speculative_config method=extract_hidden_states`，需要确认 vLLM 在多模态请求上也能触发这条 extract path。若存在不兼容，降级到"本地 HF forward 抽 hidden states"路径（绕过 vLLM）。
4. **DeepStack 必须处理**：纯文本时 Phase 1 可以规避 [8,16,24]；但多模态下视觉 token 必须经过这三层的 DeepStack 注入才能得到和 verifier 一致的 hidden state。**Phase 2 不再建议规避**——建议直接选 `[10, 25, 45]` 这类与 DeepStack 不冲突的层，或者显式在 patched forward 中复刻 DeepStack 融合。
5. **loss_mask 覆盖 placeholder**：必须确认 `_build_multimodal_loss_mask` 把所有 `image_token_id / video_token_id / audio_token_id` 位置清零，否则 draft 会去学预测图像 patch token，训练无意义且严重拖累收敛。

---

## 8. 改动文件总览 v2（Phase 1 + Phase 2）

Phase 1（文本-only，可独立落地）：

- [ ] `scripts/train.py` — Step 1
- [ ] `scripts/launch_vllm.py` — Step 2
- [ ] `src/speculators/models/dflash/core.py` — Step 3 + Step 5.1
- [ ] `src/speculators/models/dflash/config.py` — Step 4
- [ ] `src/speculators/models/dflash/model_definitions.py` — Step 5.2
- [ ] `src/speculators/data_generation/custom_worker.py` — Step 6
- [ ] `examples/data_generation_and_training/qwen3_omni_thinking_sharegpt.py` — Step 7

Phase 2（多模态，依赖 Phase 1 完成）：

- [ ] `src/speculators/data_generation/preprocessing.py` — Step 8
- [ ] `src/speculators/data_generation/configs.py` — Step 8.3（新增多模态 `DATASET_CONFIGS`）
- [ ] `src/speculators/train/data.py` — Step 9（Arrow 读 multimodal + 3D position_ids + collate）
- [ ] `src/speculators/data_generation/custom_worker.py` — Step 10（`_patched_thinker_forward`）
- [ ] `src/speculators/data_generation/vllm_client.py` — Step 11
- [ ] `src/speculators/models/dflash/core.py` — Step 12（`forward` 里 3D position_ids 分支）
- [ ] `scripts/train.py` + `scripts/gen_and_train.py` — Step 13（`--multimodal` 开关）
- [ ] `examples/data_generation_and_training/qwen3_omni_thinking_llava.py` — Step 14

---

## 9. 下一步

Phase 1 完成后：

1. **小规模 smoke test**（ShareGPT 500 条 + 50 步训练）验证纯文本 pipeline。
2. **完整文本训练**（5k ~ 50k 样本），观察 acceptance rate 与 Qwen3-8B / Qwen3-32B 基线的相对关系。

Phase 2 完成后：

3. **多模态 smoke test**（LLaVA-Instruct 500 条 + 50 步）验证多模态 pipeline，重点看 `position_ids==3D` 和 text-only 两条 RoPE 路径数值等价。
4. **混合 modality 训练**（文本 + 图像 + 视频 + 音频 各 5k），评估跨 modality 的 acceptance rate 是否不低于纯文本基线。

最后：

5. 启动 **vLLM 侧适配文档**：`plan_vllm.md`（目标：`Qwen3OmniMoeThinkerForConditionalGeneration` + `DFlashQwen3ForCausalLM` 在 vLLM V1 上能端到端多模态 spec decode；这里会需要对 `DFlashProposer._raise_if_multimodal` 的 pass-through、attention backend non-causal、MRoPE 在 draft 推理路径上的复刻做联合改造）。
