# Qwen3-Omni-Thinking 多模态 DFlash 训练 —— 修复指导

> 范围：针对当前 `patchy/qwen3-omni` 分支上已写入但未提交的多模态改动（对应 `plan.md` Phase 2 Step 8–14）的 code review 结论与逐项修复细节。
>
> 基线 commit：`27b66c5 qwen3 omni v0.1`（Phase 1 文本-only 已通过）
> 目标：让 `examples/data_generation_and_training/qwen3_omni_thinking_llava.py` 这条端到端多模态 pipeline 真正能跑通，且训练数值语义正确。

---

## 0. 修复优先级总览

| 优先级 | 文件 | 问题 | 影响 |
|---|---|---|---|
| P0 | `scripts/prepare_data.py` | 缺 `--multimodal` CLI | `gen_and_train.run_e2e` 多模态分支直接 argparse 报错 |
| P0 | `scripts/data_generation_offline2.py` | 缺 `import json` | 多模态样本第一条 `NameError` |
| P0 | `src/speculators/data_generation/custom_worker.py` | `_patched_forward` 使用未定义变量 `deepstack_input_embeds` | 任何样本进 for-loop 即 `NameError` |
| P0 | `src/speculators/train/data.py` | 引用不存在的 transformers 类 `Qwen3OmniMoePreTrainedModelForConditionalGeneration` | `ArrowDataset(verifier_name_or_path=...)` 构造时 `ImportError` |
| P0 | `scripts/train.py` | `train_dataset` 漏传 `verifier_name_or_path`；`--multimodal` CLI 是 no-op | train / val 两个 dataset 的 RoPE 行为不一致 |
| P1 | `src/speculators/data_generation/custom_worker.py` | 未实现 `_patched_thinker_forward`（plan §Step 10 遗漏） | vLLM 侧多模态 HS 抽取链路残缺 |
| P1 | `src/speculators/models/dflash/core.py` | anchor 落在视觉/音频段时 `mask_position_ids` T/H/W 被错误地强同步 | 多模态 loss 静默退化，acceptance rate 下降 |
| P1 | `src/speculators/data_generation/preprocessing.py` | `load_audio/load_image/load_video` + `return_assistant_tokens_mask` 对 processor 版本假设过强 | 旧版 transformers 全量样本被归入失败分支 |
| P2 | `src/speculators/data_generation/configs.py` | `_split_llava_user_content` 对 `<image>` 相邻文本 `strip()` 丢换行 | tokenize 结果边界异常 |
| P2 | `scripts/gen_and_train.py` | 多模态分支未显式传 `--hidden-states-path` | 靠默认回退，改默认行为时易踩坑 |

---

## 1. P0 级修复（未修则脚本无法启动）

### 1.1 `scripts/prepare_data.py`：新增 `--multimodal` CLI

**现状**

`main()` 里直接使用 `args.multimodal`，但 `parse_args()`（L42-134）从未注册该参数：

```python
# scripts/prepare_data.py  main() L166-179
dataset, _ = load_and_preprocess_dataset(
    ...
    multimodal=args.multimodal,
    multimodal_output_dir=output if args.multimodal else None,
)
```

`scripts/gen_and_train.py::run_e2e` 多模态分支又会组装 `"multimodal": True` 透传，argparse 会以 `unrecognized arguments: --multimodal` 直接退出。

**修复**

在 `parse_args()` 的 Processing 块（紧邻 `--minimum-valid-tokens` 之后、`return parser.parse_args()` 之前）加：

```python
parser.add_argument(
    "--multimodal",
    action="store_true",
    help=(
        "Use AutoProcessor.apply_chat_template and emit multimodal sidecars. "
        "Required for image/video/audio datasets such as llava-instruct."
    ),
)
```

**验收**

```bash
python scripts/prepare_data.py --help | grep multimodal
# 能看到 --multimodal 帮助文字
```

---

### 1.2 `scripts/data_generation_offline2.py`：补 `import json`

**现状**

L294-310、L353-357 使用 `json.loads(messages_json)`，但文件头部 import（L15-33）无 `import json`。

**修复**

顶部 import 块新增一行：

```python
import json
```

建议位置：与 `import asyncio`、`import logging` 放在同一块（按字母序紧跟 `import asyncio` 之后）。

**验收**

```bash
python -c "import ast; ast.parse(open('scripts/data_generation_offline2.py').read())"
# 无语法错误；再在多模态样本上 dry-run 至少一条记录
```

---

### 1.3 `src/speculators/data_generation/custom_worker.py`：`_patched_forward` 签名补 `deepstack_input_embeds`

**现状**

```python
def _patched_forward(
    self,
    input_ids,
    positions,
    intermediate_tensors: dict[str, torch.Tensor] | None = None,
    inputs_embeds=None,
    **_kwargs,
):
    ...
    for idx, layer in enumerate(...):
        ...
        if deepstack_input_embeds is not None:        # ← NameError
            deepstack_key = f"deepstack_input_embeds_{absolute_layer_idx}"
            if deepstack_key in deepstack_input_embeds:
                hidden_states = hidden_states + deepstack_input_embeds[deepstack_key]
```

签名里既没有 `deepstack_input_embeds` 形参，又没把它从 `**_kwargs` 里显式拉出来，任何样本进入 for-loop 都会 `NameError`。

**修复**

```python
def _patched_forward(
    self,
    input_ids,
    positions,
    intermediate_tensors: dict[str, torch.Tensor] | None = None,
    inputs_embeds=None,
    deepstack_input_embeds: dict[str, torch.Tensor] | None = None,
    **_kwargs,
):
    ...
    for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
        hidden_states, residual = layer(
            hidden_states=hidden_states, positions=positions, residual=residual
        )
        absolute_layer_idx = self.start_layer + idx

        if deepstack_input_embeds:
            deepstack_key = f"deepstack_input_embeds_{absolute_layer_idx}"
            ds_embed = deepstack_input_embeds.get(deepstack_key)
            if ds_embed is not None:
                hidden_states = hidden_states + ds_embed
        ...
```

并同步更新顶部 docstring，把 `deepstack_input_embeds` 的形参说明从 kwargs 挪到正文：

```python
"""...
Args:
    input_ids:                   token id 序列（prefill 阶段）
    positions:                   position ids
    intermediate_tensors:        pipeline parallel 中间态
    inputs_embeds:               若上游已嵌入完成（VLM / thinker），直接使用
    deepstack_input_embeds:      DeepStack 视觉注入 dict，key 形如
                                 ``deepstack_input_embeds_{layer_idx}``
                                 （Qwen3VL / Qwen3-Omni）
"""
```

**验收**

纯文本 batch + 多模态 batch 分别过一次 vLLM prefill，日志里 `Hidden states capture setup complete` 后 forward 不再 `NameError`。

---

### 1.4 `src/speculators/train/data.py::_make_rope_index_fn`：修正 transformers 类名

**现状**

```python
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoePreTrainedModelForConditionalGeneration,   # 不存在
)
```

transformers 在 `modeling_qwen3_omni_moe` 中导出的是：
- `Qwen3OmniMoePreTrainedModel`（基类，无 `get_rope_index`）
- `Qwen3OmniMoeThinkerForConditionalGeneration`（**含 `get_rope_index` / `get_llm_pos_ids_for_vision`**）
- `Qwen3OmniMoeForConditionalGeneration`（对外入口，转发到 thinker）

**修复**

```python
def _make_rope_index_fn(verifier_name_or_path: str):
    try:
        verifier_root_config = AutoConfig.from_pretrained(
            verifier_name_or_path, trust_remote_code=True
        )
    except Exception:
        return None

    thinker_config = getattr(verifier_root_config, "thinker_config", None)
    if thinker_config is None:
        return None

    try:
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeThinkerForConditionalGeneration,
        )
    except ImportError:
        return None  # 退化到 1D arange；让纯文本训练不受牵连

    vision_config = getattr(thinker_config, "vision_config", None)
    spatial_merge_size = getattr(vision_config, "spatial_merge_size", 1)
    dummy = SimpleNamespace(
        config=SimpleNamespace(
            image_token_id=getattr(thinker_config, "image_token_id", None),
            video_token_id=getattr(thinker_config, "video_token_id", None),
            audio_token_id=getattr(thinker_config, "audio_token_id", None),
            vision_start_token_id=getattr(
                thinker_config, "vision_start_token_id", None
            ),
            audio_start_token_id=getattr(thinker_config, "audio_start_token_id", None),
            position_id_per_seconds=getattr(
                thinker_config, "position_id_per_seconds", 25
            ),
        ),
        spatial_merge_size=spatial_merge_size,
    )
    dummy.get_llm_pos_ids_for_vision = MethodType(
        Qwen3OmniMoeThinkerForConditionalGeneration.get_llm_pos_ids_for_vision,
        dummy,
    )
    return MethodType(
        Qwen3OmniMoeThinkerForConditionalGeneration.get_rope_index,
        dummy,
    )
```

**说明**

- 两个 `try/except` 让这个函数变成"尽力而为"——失败 silently 返回 `None`，`BaseDataset._build_position_ids` 里 `rope_index_fn is None` 分支会走 `torch.arange`，不会污染纯文本路径。
- `trust_remote_code=True` 对 Qwen3-Omni 必须；不加会在某些镜像上失败。

**验收**

```python
from speculators.train.data import _make_rope_index_fn
fn_omni  = _make_rope_index_fn("Qwen/Qwen3-Omni-30B-A3B-Thinking")
fn_qwen3 = _make_rope_index_fn("Qwen/Qwen3-8B")
fn_bad   = _make_rope_index_fn("/not/exist")
assert callable(fn_omni)
assert fn_qwen3 is None     # 无 thinker_config
assert fn_bad is None       # AutoConfig 失败兜底
```

---

### 1.5 `scripts/train.py`：`train_dataset` 对称传 `verifier_name_or_path`；让 `--multimodal` 真正生效

**现状**

```python
# scripts/train.py L330-358
train_dataset = ArrowDataset(
    datapath=args.data_path,
    ...
    model=args.verifier_name_or_path,
    # ← 缺 verifier_name_or_path=args.verifier_name_or_path
)
val_dataset = ArrowDataset(
    datapath=args.data_path,
    ...
    model=args.verifier_name_or_path,
    verifier_name_or_path=args.verifier_name_or_path,   # ← 只有这一边传了
)
```

后果：
- `train_dataset._rope_index_fn` = None → train 样本全走 1D arange
- `val_dataset._rope_index_fn`   = 真函数 → val 样本走 3D MRoPE
- batch 内一旦混合两种 ndim 的 `position_ids`，collate 就 shape 不齐。
- 同时 `--multimodal` 这个 CLI 完全是 no-op（没有任何分支条件读它）。

**修复**

```python
# 统一入口：只有 --multimodal 时才构造 rope_index_fn，避免纯文本训练也去做
# AutoConfig.from_pretrained(verifier) 的额外 IO
rope_verifier = args.verifier_name_or_path if args.multimodal else None

train_dataset = ArrowDataset(
    datapath=args.data_path,
    max_len=args.total_seq_len,
    hidden_states_path=args.hidden_states_path,
    vllm_endpoint=args.vllm_endpoint,
    on_missing=args.on_missing,
    on_generate=args.on_generate,
    transform=noise_transform,
    split_ratio=0.9,
    model=args.verifier_name_or_path,
    hidden_states_dtype=hidden_states_dtype,
    request_timeout=args.request_timeout,
    max_retries=args.max_retries,
    verifier_name_or_path=rope_verifier,       # ← 新增
)
val_dataset = ArrowDataset(
    datapath=args.data_path,
    ...
    model=args.verifier_name_or_path,
    ...
    verifier_name_or_path=rope_verifier,       # ← 改成同一个门控
)
```

**验收**

```bash
torchrun --standalone --nproc_per_node=1 scripts/train.py \
    --verifier-name-or-path Qwen/Qwen3-Omni-30B-A3B-Thinking \
    --data-path ./output/qwen3_omni_thinking_llava/gen/llava-instruct \
    --multimodal --draft-arch qwen3 --num-layers 1 --draft-intermediate-size 6144 \
    --target-layer-ids 2 23 45 --mask-token-id 151671 \
    --block-size 8 --max-anchors 256 --epochs 1 --total-seq-len 16384
# 第一个 iter 应有 collated position_ids shape == (3, 1, max_len)；
# 无 shape mismatch / NameError；loss 前 50 step 下降。
```

---

## 2. P1 级修复（不 crash，但语义错误）

### 2.1 `custom_worker.py`：实现 `_patched_thinker_forward`（plan §Step 10）

**现状**

`_setup_hidden_states_capture` 里选出了 `base_model = thinker.model[.model]`，但没有在 `thinker` 这一层做拦截。

对于 Qwen3-Omni，多模态请求在 vLLM 内部会走：

```
Qwen3OmniMoeForConditionalGeneration.forward
  -> self.thinker(input_ids=..., pixel_values=..., image_grid_thw=..., ...)
     -> thinker.model(inputs_embeds=...)  ← hook 挂在这里
```

Phase 1 纯文本只走 `thinker.model(input_ids=...)`，我们 patch 它就行；Phase 2 需要先让 thinker 把 vision/audio 特征 scatter 进 `inputs_embeds`，再调用 text backbone。

**修复**（最小改动）

在 `custom_worker.py` 里新增：

```python
def _patched_thinker_forward(self, *args, **kwargs):
    """Patched thinker forward that scatters vision/audio embeds, then delegates
    to the (already patched) text backbone, so hidden-state capture on text
    decoder layers still fires.
    """
    pixel_values = kwargs.pop("pixel_values", None)
    image_grid_thw = kwargs.pop("image_grid_thw", None)
    pixel_values_videos = kwargs.pop("pixel_values_videos", None)
    video_grid_thw = kwargs.pop("video_grid_thw", None)
    second_per_grids = kwargs.pop("second_per_grids", None)
    input_features = kwargs.pop("input_features", None)
    feature_attention_mask = kwargs.pop("feature_attention_mask", None)

    input_ids = kwargs.get("input_ids")
    if input_ids is None:
        # 下游已给好 inputs_embeds
        return self._orig_forward(*args, **kwargs)

    inputs_embeds = self.model.get_input_embeddings()(input_ids)
    deepstack_visual_embeds = None

    # vision image
    if pixel_values is not None and image_grid_thw is not None:
        image_embeds, deepstack_visual_embeds = self._process_image_input({
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        })
        mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
        inputs_embeds = inputs_embeds.masked_scatter(
            mask.expand_as(inputs_embeds), image_embeds
        )

    # vision video （省略，对齐 image 分支）

    # audio
    if input_features is not None:
        audio_embeds = self.get_audio_features(input_features, feature_attention_mask)
        mask = (input_ids == self.config.audio_token_id).unsqueeze(-1)
        inputs_embeds = inputs_embeds.masked_scatter(
            mask.expand_as(inputs_embeds), audio_embeds
        )

    kwargs["inputs_embeds"] = inputs_embeds
    kwargs["input_ids"] = None
    if deepstack_visual_embeds is not None:
        kwargs["deepstack_input_embeds"] = {
            f"deepstack_input_embeds_{layer}": emb
            for layer, emb in zip(
                self.config.vision_config.deepstack_visual_indexes,
                deepstack_visual_embeds,
                strict=True,
            )
        }
    return self._orig_forward(*args, **kwargs)
```

并在 `_setup_hidden_states_capture` 的 thinker 分支下装壳：

```python
if hasattr(model, "thinker"):
    thinker = model.thinker
    thinker._orig_forward = thinker.forward
    thinker.forward = types.MethodType(_patched_thinker_forward, thinker)

    if hasattr(thinker, "get_language_model"):
        base_model = thinker.get_language_model().model
    elif hasattr(thinker, "model"):
        base_model = (
            thinker.model.model
            if hasattr(thinker.model, "model")
            else thinker.model
        )
    else:
        ...
```

**备注**：`_process_image_input` / `get_audio_features` 的具体签名依 transformers 版本而定；若版本不一致，降级方案是**不 patch thinker，只在数据侧用本地 HF `Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(..., device_map="auto")` 跑 forward 抓 hidden states**（慢但确定性高），并在 plan 侧把 Step 10 重新标为 "partial / fallback"。

**验收**

一张 448×448 图 + 一句文本的样本，patched thinker forward 后，text backbone 收到的 `inputs_embeds[image_placeholder_positions]` 的 L2 误差相对原生 `thinker.forward` 的输出 < 1e-4。

---

### 2.2 `src/speculators/models/dflash/core.py`：anchor 落在视觉/音频段时 `mask_position_ids` 处理

**现状**

```python
if position_ids.dim() == 3:
    anchor_position_ids = position_ids[..., anchor_positions]    # (3, 1, num_anchors)
    mask_position_ids_1d = get_base_indices_for_anchored_blocks(
        anchor_position_ids[0], self.block_size, input_ids.numel()
    )
    mask_position_ids = mask_position_ids_1d.view(1, 1, -1).expand(
        position_ids.shape[0], position_ids.shape[1], -1
    )
    position_ids = torch.cat([position_ids, mask_position_ids], dim=-1)
```

问题：
1. **只用 T 通道 (`[0]`) 的 anchor pos 去算 base offset**，然后把结果复制到 H、W 通道。
   - 若 anchor 落在纯文本段：T==H==W，OK。
   - 若 anchor 落在视觉段（image/video placeholder）：T, H, W 互不相等，mask block 的 H/W 被错误强制与 T 相等，3D RoPE 语义破坏。
2. `anchor_position_ids[0]` 形状是 `(1, num_anchors)`，进 `get_base_indices_for_anchored_blocks` 的第一个参数 —— 1D 路径里那里传的也是 `(1, num_anchors)`，形状兼容 ✅。
3. bsz>1 目前不会发生（`DFlashDraftModel.forward` 默认 bsz=1），但 `expand` 用的是 `position_ids.shape[1]` 做 batch 维、形式上支持，只是 anchor 选取逻辑还是只看 T 通道。

**推荐修复（保守方案）**

在 **数据侧** 把 `anchor` 候选限制为"T/H/W 相等的 token"（即所有非视觉/非音频 placeholder token），`core.py` 保持现在的 expand 逻辑即可。

具体做法：`BaseDataset.__getitem__` 里算完 `position_ids` 后，派生一个 `anchor_mask`：

```python
# data.py BaseDataset.__getitem__
pos = data["position_ids"]
if pos.ndim == 2:   # (3, seq_len) MRoPE
    data["anchor_mask"] = (pos[0] == pos[1]) & (pos[1] == pos[2])
else:
    data["anchor_mask"] = torch.ones(seq_len, dtype=torch.bool)
```

然后 `DFlashDraftModel.forward` 在选 anchor 时与 `anchor_mask` 做 AND：

```python
anchor_candidates = anchor_candidates & anchor_mask_batch
```

这样 3D 分支里的 expand 就总是安全的。

**激进修复（若未来希望 anchor 也能落到视觉段）**

三通道各自算 offset：

```python
if position_ids.dim() == 3:
    anchor_position_ids = position_ids[..., anchor_positions]   # (3, 1, num_anchors)
    mask_position_ids_channels = []
    for c in range(position_ids.shape[0]):
        mp = get_base_indices_for_anchored_blocks(
            anchor_position_ids[c], self.block_size, input_ids.numel()
        )
        mask_position_ids_channels.append(mp.view(1, -1))
    mask_position_ids = torch.stack(mask_position_ids_channels, dim=0).unsqueeze(1)
    position_ids = torch.cat([position_ids, mask_position_ids], dim=-1)
```

但要注意 `get_base_indices_for_anchored_blocks` 的单调递增假设在 H/W 上是否仍成立（H/W 在同一图像内是 2D plateau），需要单独验证。**Phase 2 初期建议走保守方案。**

**验收**

- 构造 `position_ids = (3, 1, L)`，其中 anchor 全部落在 T==H==W 段 → forward 正常。
- 构造同上但 anchor 有一个落在视觉段 → 保守方案下该 anchor 被 `anchor_mask` 过滤，总 anchor 数 -1；激进方案下 mask_position_ids 三通道数值互不相等。
- 纯文本样本 1D 路径数值与当前主分支完全一致。

---

### 2.3 `preprocessing.py`：processor kwarg 版本兼容

**现状**

```python
encoded_any = processor.apply_chat_template(
    normalized_conv,
    tokenize=True,
    add_generation_prompt=False,
    return_assistant_tokens_mask=True,
    return_dict=True,
    return_tensors="pt",
    max_length=max_length,
    truncation=True,
    load_audio=True,
    load_image=True,
    load_video=True,
)
...
if mask_key not in encoded:
    raise ValueError(
        "Processor output missing assistant token mask for multimodal sample"
    )
```

风险：
1. `load_audio / load_image / load_video`：较老 transformers 的 `Qwen3OmniMoeProcessor` 会把这些 kwarg 吞掉但不 auto-load 媒体资源；部分版本期待的是 `load_audios`（复数）。
2. `return_assistant_tokens_mask=True`：在 processor 层（不是 tokenizer）支持性依赖版本。多数 transformers 版本的 `Qwen3OmniMoeProcessor.apply_chat_template` **不返回** `assistant_masks`，直接走到 `raise ValueError` 分支，全量样本失败。

**修复**

```python
import inspect

_PROCESSOR_ACT_SIG_CACHE: dict[int, set[str]] = {}

def _processor_kwargs(processor) -> set[str]:
    key = id(processor)
    if key in _PROCESSOR_ACT_SIG_CACHE:
        return _PROCESSOR_ACT_SIG_CACHE[key]
    try:
        sig = inspect.signature(processor.apply_chat_template)
        names = set(sig.parameters.keys())
    except (TypeError, ValueError):
        names = set()
    _PROCESSOR_ACT_SIG_CACHE[key] = names
    return names


# in _preprocess_batch  multimodal branch
allowed = _processor_kwargs(processor)
call_kwargs = dict(
    tokenize=True,
    add_generation_prompt=False,
    return_dict=True,
    return_tensors="pt",
    max_length=max_length,
    truncation=True,
)
for k in ("load_audio", "load_image", "load_video", "load_audios", "load_images", "load_videos"):
    if k in allowed:
        call_kwargs[k] = True
supports_mask = "return_assistant_tokens_mask" in allowed
if supports_mask:
    call_kwargs["return_assistant_tokens_mask"] = True

encoded_any = processor.apply_chat_template(normalized_conv, **call_kwargs)
encoded = cast("dict[str, Any]", encoded_any)
input_ids = _maybe_strip_batch_dim(encoded["input_ids"]).to(torch.long)

if supports_mask and (
    "assistant_masks" in encoded or "assistant_mask" in encoded
):
    mask_key = "assistant_masks" if "assistant_masks" in encoded else "assistant_mask"
    base_loss_mask = _maybe_strip_batch_dim(encoded[mask_key]).to(torch.long)
else:
    # fallback: 用 tokenizer 做 text-only 的 assistant regex mask，
    # 再把 placeholder token 位置置 0
    assert assistant_pattern is not None, (
        "Processor did not return assistant_masks; need assistant_pattern fallback"
    )
    formatted_raw = processor.tokenizer.apply_chat_template(
        normalized_conv, tokenize=False, add_generation_prompt=False,
    )
    base_loss_mask = _loss_mask_from_ids(
        input_ids, formatted_raw, processor.tokenizer, assistant_pattern
    )

loss_mask = _build_multimodal_loss_mask(
    input_ids, base_loss_mask, placeholder_token_ids
)
```

其中 `_loss_mask_from_ids` 是一个辅助函数：走 `tokenizer(formatted_raw, return_offsets_mapping=True)` 得到 offsets，再喂 `_create_loss_mask_from_offsets` 得到 mask；若长度与 processor 的 `input_ids` 不一致，截断/对齐后再返回。

**验收**

- 用较老版本 transformers（假设不支持 `return_assistant_tokens_mask`）构造一个 llava 样本，preprocessing 不再抛 `Processor output missing assistant token mask`。
- 用新版本 transformers，走新路径，loss_mask 全部由 processor 返回，与 fallback 路径 L2 误差 == 0（因为 fallback 只是后备）。

---

## 3. P2 级修复（质量增强）

### 3.1 `configs.py::_split_llava_user_content`：保留 `<image>` 周围空白/换行

**现状**

```python
elif part:
    content.append({"type": "text", "text": part.strip()})
...
return [segment for segment in content if segment.get("text", "") or segment["type"] != "text"]
```

`"<image>\nWhat is in this picture?"` 会被拆成 `[image, "What is in this picture?"]`，丢失换行；有些 chat template 在此处依赖换行分隔，丢了就可能让 tokenizer 把 `<|image_pad|>What` 做成奇怪的 BPE。

**修复**

```python
elif part:
    content.append({"type": "text", "text": part})
...
return [
    seg for seg in content
    if seg["type"] != "text" or (seg.get("text", "") and seg["text"].strip())
]
```

即：保留原始 `part`（包括空白/换行），只在过滤时用 `.strip()` 判断是否是"纯空白"。

---

### 3.2 `scripts/gen_and_train.py::run_e2e`：显式传 `--hidden-states-path`

**现状**

多模态分支里：

```python
offline2_args = {
    ...
    "output": str(dataset_output_dir / "hidden_states"),
}
...
# train 侧
ta_dict = {..., "data-path": str(train_data_path), ...}
# 没有 ta_dict["hidden-states-path"]
```

`ArrowDataset` 默认把 `self.datapath / "hidden_states"` 当作 hidden_states 路径，刚好与 offline2 的 `--output` 对齐 ✅，但**该对齐依赖默认值**。如果将来改默认布局，会静默错 path。

**修复**

```python
ta_dict = {
    **train_args._asdict(),
    "verifier-name-or-path": verifier_name_or_path,
    "data-path": str(train_data_path),
    "save-path": str(output_path / "checkpoints"),
    "log-dir": str(output_path / "logs"),
}
if uses_multimodal:
    ta_dict["multimodal"] = True
    ta_dict["hidden-states-path"] = str(train_data_path / "hidden_states")
```

---

## 4. 建议的修复与验证流水

### 4.1 修复顺序

按文件合并，减少反复编辑：

1. **批次 A（P0，~10 行）**
   - `scripts/prepare_data.py` 加 `--multimodal`
   - `scripts/data_generation_offline2.py` 加 `import json`
   - `scripts/train.py` 加 `rope_verifier = args.verifier_name_or_path if args.multimodal else None` + `train_dataset` / `val_dataset` 对称传参
   - `src/speculators/train/data.py` 修正 transformers 类名 + try/except 兜底

2. **批次 B（P0，`custom_worker.py` 单文件）**
   - `_patched_forward` 签名补 `deepstack_input_embeds=None`

3. **批次 C（P1，跨 2 文件）**
   - `custom_worker.py` 实现 `_patched_thinker_forward` + `_setup_hidden_states_capture` 装壳
   - `core.py` + `data.py` 引入 `anchor_mask`

4. **批次 D（P1，`preprocessing.py` 单文件）**
   - processor kwarg 版本兼容 + loss_mask fallback

5. **批次 E（P2）**
   - `configs.py` 保留空白
   - `gen_and_train.py` 显式 `--hidden-states-path`

### 4.2 最小验证脚本

```bash
# smoke-1: 纯文本路径不受影响
torchrun --standalone --nproc_per_node=1 scripts/train.py \
    --verifier-name-or-path Qwen/Qwen3-8B \
    --data-path ./output/qwen3_8b_sharegpt/gen/sharegpt \
    --draft-arch qwen3 --num-layers 1 --epochs 1 --total-seq-len 4096
# 期望：收敛曲线与修复前完全一致（两次 loss 列差 < 1e-5）

# smoke-2: 多模态 LLaVA 路径第一次真正跑通
python examples/data_generation_and_training/qwen3_omni_thinking_llava.py
# 期望：
#   - prepare_data.py 打印 "Using HF assistant token mask"（或 fallback 信息）
#   - data_generation_offline2.py 成功保存 hs_0.safetensors
#   - train.py 首个 batch 的 collated position_ids.shape == (3, 1, max_len)
#   - 前 50 step loss 曲线单调下降；无 NaN / shape error
```

### 4.3 Regression 关卡

在 CI / 本地 regression 中加两条断言：

```python
# tests/test_multimodal_position_ids.py
def test_3d_position_ids_shape():
    from speculators.train.data import create_collate_fn
    # mock 1 multimodal sample + 1 text sample
    ...
    collate = create_collate_fn(max_len=128, hidden_size=2048)
    out = collate(batch)
    assert out["position_ids"].shape == (3, 1, 128)

def test_text_only_position_ids_shape():
    ...
    assert out["position_ids"].shape == (1, 128)

def test_anchor_mask_excludes_placeholders():
    # anchor_mask 在视觉 placeholder 位置必须为 0
    ...
```

---

## 5. 附：不在本轮修复范围内的 TODO（留给下一轮）

这些不阻塞当前多模态 smoke test，但在正式落 PR 前仍需要处理：

1. `DFlash.pdf`（4225 行）仍在 repo，建议 LFS 或移出仓库。
2. `custom_worker._patched_thinker_forward` 的 video 分支（plan §Step 10）在本轮修复中略写；等 image 分支跑通后再补。
3. `vLLM` 侧 `qwen3_omni_thinking_dflash.py`（推理 path）未改，本 plan 仍在 speculators 仓内闭环。
4. `gen_and_train.run_e2e` 多模态只支持单数据集，未来要支持混合 modality 训练时需要扩展 `combine_token_frequency_distributions` 兼容多模态的 placeholder token 计数。

---

> **完成本文档所列 P0 + P1 修复后**，即可认为 `plan.md` Phase 2 达到"smoke test 可跑通 + 语义正确"的水平；P2 为质量增强，可作为后续小 PR 分批合入。
