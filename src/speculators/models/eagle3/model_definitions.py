# ruff: noqa: ERA001
from typing import Any

import torch
from transformers import Cache, LlamaConfig, PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs

from speculators.models import base_components


class Eagle3FirstLayerMixin:
    """Shared Eagle3 first-layer modifications for any decoder layer.

    Patches q/k/v projections to accept 2x hidden_size input (cat([embeds, hidden]))
    and overrides forward to split, normalize, and recombine before attention.
    """

    # Provided by the decoder layer base class
    self_attn: Any
    input_layernorm: Any
    post_attention_layernorm: Any
    mlp: Any
    norm_before_residual: bool
    hidden_norm: Any

    def _patch_eagle3_projections(
        self,
        config: PretrainedConfig,
        norm_class: type[torch.nn.Module],
        norm_before_residual: bool,
    ):
        """Replace q/k/v projections with 2x hidden_size input and add hidden_norm."""
        self.norm_before_residual = norm_before_residual
        self.hidden_norm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn.q_proj = torch.nn.Linear(
            2 * config.hidden_size,  # previous: config.hidden_size
            config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.self_attn.k_proj = torch.nn.Linear(
            2 * config.hidden_size,  # previous: config.hidden_size
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.self_attn.v_proj = torch.nn.Linear(
            2 * config.hidden_size,  # previous: config.hidden_size
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],  # type: ignore[valid-type]
    ) -> torch.Tensor:
        # Previously in the parent DecoderLayer:
        #   residual = hidden_states
        #   hidden_states = self.input_layernorm(hidden_states)

        # ##### Start of Eagle3 modifications #####

        # hidden_states are cat([embeds, hidden], dim=-1)
        # so residual should be hidden part only, and embeds should be normalized
        mid = hidden_states.shape[2] // 2
        embeds, hidden = hidden_states.split(mid, dim=-1)
        residual = hidden

        # Apply norms
        embeds = self.input_layernorm(embeds)
        hidden = self.hidden_norm(hidden)
        if self.norm_before_residual:
            residual = hidden  # set residual to normalized hidden
        hidden_states = torch.cat([embeds, hidden], dim=-1)
        if torch.__version__ >= "2.10":
            # As of `torch==2.10`, compile attempts to fuse together too many
            # ops, resulting in a fused kernel that exceeds shared memory limits
            # For now, we force a graph break to prevent this
            # https://github.com/pytorch/pytorch/issues/175250
            torch._dynamo.graph_break()  # noqa: SLF001

        # ##### End of Eagle3 modifications #####

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states  # noqa: RET504


class LlamaDecoderEagle3FirstLayer(Eagle3FirstLayerMixin, LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        norm_before_residual: bool = False,
    ):
        super().__init__(config, layer_idx)
        self._patch_eagle3_projections(config, LlamaRMSNorm, norm_before_residual)


class Qwen3DecoderEagle3FirstLayer(Eagle3FirstLayerMixin, Qwen3DecoderLayer):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        norm_before_residual: bool = False,
    ):
        super().__init__(config, layer_idx)
        self._patch_eagle3_projections(config, Qwen3RMSNorm, norm_before_residual)


#   [新增] VwnEagle3PreVwnLayer           — pre_vwn_layer_v0，对应设计文档主流程
#   [新增] VwnEagle3PreVwnLayerV1         — pre_vwn_layer_v1，用于消融实验
#   [新增] VwnEagle3FirstLayerMixin       — 替换 Eagle3FirstLayerMixin
#          核心差异：
#            * 不再 patch q/k/v 为 2x input（Attn 在 D 空间运行）
#            * _patch_vwn_eagle3_projections 新增 pre_vwn_layer、A2/B1/A3/B2/A4 等矩阵
#            * _patch_vwn_eagle3_projections 接收 norm_class 参数，同时适配 Llama/Qwen3
#            * forward 按照 pre_vwn → A2 → Attn → B1 → A3 → FFN → B2 → A4 的顺序执行
#   [新增] LlamaDecoderVwnEagle3FirstLayer — 对应 LlamaDecoderEagle3FirstLayer
#   [新增] QwenDecoderVwnEagle3FirstLayer  — 对应 Qwen3DecoderEagle3FirstLayer
#   [修改] model_classes 补充 llama，与 eagle3/model_definitions.py 保持一致


# ──────────────────────────────────────────────────────────────────────────────
# pre_vwn_layer_v0（主流程）
# ──────────────────────────────────────────────────────────────────────────────

class VwnEagle3PreVwnLayer(nn.Module):
    """pre_vwn_layer_v0：特征融合 + 宽度扩展模块。

    数据流（对应设计文档细节流程图左侧部分）：

        embeds [B,T,D]  ──upward(A0)──► e_up [B,T,D']  ──E-norm──► e_normed
        hidden [B,T,D]  ──upward(A0)──► h_up [B,T,D']
                                            │
                                        extend(A1)
                                            │
                               h_res [B,T,D'] ← split → h_norm_in [B,T,D']
                                                              │
                                                          H-norm
                                                              │
                                                          h_normed
                            cat([e_normed, h_normed]) [B,T,2D']
                                            │
                                          fc (FC Layer2)
                                            │
                                         fused [B,T,D']
                                            │
                                    full_connected (B0)
                                            │  + h_res (残差)
                                            ▼
                                        out [B,T,D']
    """

    def __init__(
        self,
        hidden_size: int,
        vwn: VwnConfig,
        norm_eps: float,
        norm_class: type[nn.Module] = LlamaRMSNorm,  # 支持 LlamaRMSNorm / Qwen3RMSNorm
    ):
        super().__init__()
        d_prime = vwn.expanded_hidden_size(hidden_size)

        # A0: D → D'（embeds 和 hidden 共享同一个 upward 矩阵）
        self.upward = nn.Linear(hidden_size, d_prime, bias=False)

        # A1: D' → 2D'（split 后分别作为残差分支和 norm 输入分支）
        self.extend = nn.Linear(d_prime, 2 * d_prime, bias=False)

        # E-norm / H-norm 工作在 D' 空间，norm_class 由外部传入以适配不同架构
        self.input_layernorm = norm_class(d_prime, eps=norm_eps)  # E-norm
        self.hidden_norm = norm_class(d_prime, eps=norm_eps)       # H-norm

        # FC Layer2: cat([e_normed, h_normed]) 2D' → D'
        self.fc = nn.Linear(2 * d_prime, d_prime, bias=False)

        # B0: D' → D'（与残差相加）
        self.full_connected = nn.Linear(d_prime, d_prime, bias=False)

    def forward(
        self,
        embeds: torch.Tensor,   # [B, T, D]
        hidden: torch.Tensor,   # [B, T, D]
    ) -> torch.Tensor:          # [B, T, D']
        # 扩维
        e_up = self.upward(embeds)                           # [B, T, D']
        h_up = self.upward(hidden)                           # [B, T, D']（共享 A0）

        # extend: 产生残差分支 + norm 输入分支
        h_extended = self.extend(h_up)                       # [B, T, 2D']
        h_norm_in, h_res = h_extended.chunk(2, dim=-1)       # 各 [B, T, D']

        # 归一化
        e_normed = self.input_layernorm(e_up)                # E-norm
        h_normed = self.hidden_norm(h_norm_in)               # H-norm

        # 融合
        fused = torch.cat([e_normed, h_normed], dim=-1)      # [B, T, 2D']
        fused = self.fc(fused)                               # [B, T, D']

        # 残差
        out = self.full_connected(fused) + h_res             # [B, T, D']
        return out


# ──────────────────────────────────────────────────────────────────────────────
# pre_vwn_layer_v1（消融实验用，替换 v0）
# ──────────────────────────────────────────────────────────────────────────────

class VwnEagle3PreVwnLayerV1(nn.Module):
    """pre_vwn_layer_v1：简化版特征融合，用于消融实验。

    数据流：
        embeds [B,T,D] ──E-norm1──┐
        hidden [B,T,D] ──H-norm1──┤ cat [B,T,2D] → FC Layer3 → D → upward → D'
    """

    def __init__(
        self,
        hidden_size: int,
        vwn: VwnConfig,
        norm_eps: float,
        norm_class: type[nn.Module] = LlamaRMSNorm,  # 支持 LlamaRMSNorm / Qwen3RMSNorm
    ):
        super().__init__()
        self.vwn=vwn
        self.hidden_size = hidden_size
        self.upward_hidden_size = vwn.expanded_hidden_size(self.hidden_size)
        self.input_layernorm = norm_class(self.hidden_size, eps=norm_eps)  # E-norm1
        self.hidden_norm = norm_class(self.hidden_size, eps=norm_eps)       # H-norm1
        self.fc = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)         # FC Layer3: 2D → D
        if self.vwn.expand is True:
            self.upward = nn.Linear(self.hidden_size//self.vwn.m, self.upward_hidden_size//self.vwn.m, bias=False)         # upward B: D → D'

    def forward(
        self,
        embeds: torch.Tensor,   # [B, T, D]
        hidden: torch.Tensor,   # [B, T, D]
    ) -> torch.Tensor:          # [B, T, D']
        e_normed = self.input_layernorm(embeds)
        h_normed = self.hidden_norm(hidden)
        fused = torch.cat([e_normed, h_normed], dim=-1)   # [B, T, 2D]
        fused = self.fc(fused)                             # [B, T, D]
        if self.vwn.expand is False:
            return fused
        
        fused_grouped = fused.view(-1, self.hidden_size//self.vwn.m) # [B*T*m, D/m]
        upward_fused_grouped = self.upward(fused_grouped) # [B*T*m, D'/m]
        upward_fused = upward_fused_grouped.view(fused.shape[0], fused.shape[1], self.upward_hidden_size)  # [B, T, D']
        return upward_fused


# ──────────────────────────────────────────────────────────────────────────────
# VwnEagle3FirstLayerMixin
# ──────────────────────────────────────────────────────────────────────────────

class VwnEagle3FirstLayerMixin:
    """VWN+Eagle3 第一层 Mixin。

    与 Eagle3FirstLayerMixin 的关键区别：
      * Eagle3：q/k/v 接受 2×D 输入，Attn 在 2D 空间运行
      * VWN   ：q/k/v 保持原始 D 输入，用 A2/A3 降维后再送入 Attn/FFN，
                用 B1/B2 将结果升回 D' 空间做残差，最后用 A4 降回 D 传给下一层

    新增子模块（通过 _patch_vwn_eagle3_projections 注入）：
        pre_vwn_layer                  VwnEagle3PreVwnLayer  [D → D']
        downward_and_forgot      A2    Linear(D' → D + D')
        pre_attention_layernorm  norm0 RMSNorm(D)
        upward_after_attn        B1    Linear(D  → D')
        downward_and_forgot_after_attn A3 Linear(D' → D + D')
        pre_mlp_layernorm        norm1 RMSNorm(D)
        upward_after_mlp         B2    Linear(D  → D')
        downward                 A4    Linear(D' → D) not exist for if self.vwn.expand False:

    self_attn / post_attention_layernorm / mlp 直接复用基类，无需修改。
    """

    # 由基类（LlamaDecoderLayer 或 Qwen3DecoderLayer）提供
    self_attn: Any
    # ── norm1: Attn 输出后归一化（工作在 D 空间），架构对齐 ────────────
    post_attention_layernorm: Any
    mlp: Any   
    count = 0

    def save_tensor_to_dir(self, tensor_data, sub_dir_name):
        import os
        save_path = "/mnt/share/t00886357/eagle3/qwen3_30b_gsm8k_fix_pattern/logits/vwn_eagle3/hidden"
        file_name_new = f"{sub_dir_name}_{self.count}.pth"
        target_dir = os.path.join(save_path, sub_dir_name)
        os.makedirs(target_dir, exist_ok=True)
        save_path = os.path.join(target_dir, file_name_new)
        torch.save(tensor_data, save_path)
        print(f"\n\n layer save tensors {sub_dir_name}, shape is {tensor_data.shape}, file_name is {file_name_new}")
        self.count += 1

    def _patch_vwn_eagle3_projections(
        self,
        config: PretrainedConfig,                    # 接受 LlamaConfig / Qwen3Config
        norm_class: type[nn.Module],                 # LlamaRMSNorm 或 Qwen3RMSNorm
        vwn: VwnConfig,
        pre_vwn_layer_class: type = VwnEagle3PreVwnLayer,
    ):
        """注入所有 VWN 专属子模块。
        Args:
            config:              模型配置（提供 hidden_size / rms_norm_eps）
            norm_class:          归一化层类型，LlamaRMSNorm 或 Qwen3RMSNorm
            vwn:                 VwnConfig 实例（提供 m / r / expanded_hidden_size）
            pre_vwn_layer_class: 支持替换为 VwnEagle3PreVwnLayerV1（消融用）
        """
        self.hidden_size = config.hidden_size
        self.d_prime = vwn.expanded_hidden_size(config.hidden_size)
        self.vwn = vwn

        norm_eps = config.rms_norm_eps

        # ── pre_vwn_layer（v0 或 v1），传入 norm_class 适配不同架构 ─────────
        self.pre_vwn_layer = pre_vwn_layer_class(self.hidden_size, vwn, norm_eps, norm_class)

        # ── A2: D' → D + D'（split 后：前 D 送 Attn，后 D' 作为新残差） ──
        self.downward_and_forgot = nn.Linear(self.d_prime//self.vwn.m, (self.hidden_size + self.d_prime)//self.vwn.m, bias=False)

        # ── norm0: Attn 输入前归一化（工作在 D 空间），架构对齐 ────────────
        self.pre_attention_layernorm = norm_class(self.hidden_size, eps=norm_eps)

        # ── B1: D → D'（将 Attn 输出扩维并加到残差上） ────────────────────
        self.upward_after_attn = nn.Linear(self.hidden_size//self.vwn.m, self.d_prime//self.vwn.m, bias=False)

        # ── A3: 同 A2，但权重独立 ──────────────────────────────────────────
        self.downward_and_forgot_after_attn = nn.Linear(
            self.d_prime//self.vwn.m, (self.hidden_size + self.d_prime)//self.vwn.m, bias=False)
        
        # ── B2: D → D'（将 FFN 输出扩维并加到残差上） ─────────────────────
        self.upward_after_mlp = nn.Linear(self.hidden_size//self.vwn.m, self.d_prime//self.vwn.m, bias=False)

        # ── A4: D' → D（层间数据连接） ────────────────────────────────────
        if self.vwn.expand is True:
            self.downward = nn.Linear(self.d_prime//self.vwn.m, self.hidden_size//self.vwn.m, bias=False)

        del self.input_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, T, 2*D]  cat([embeds, hidden])
        attention_mask=None,
        position_ids=None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs: Unpack[TransformersKwargs],  # type: ignore[valid-type]
    ) -> torch.Tensor:                        # [B, T, D]
        # ── 1. 拆分 cat([embeds, hidden]) ────────────────────────────────
        # 与 Eagle3FirstLayerMixin 相同：输入是 2×D 的拼接
        mid = hidden_states.shape[-1] // 2
        embeds, hidden = hidden_states.split(mid, dim=-1)
        # embeds: [B, T, D],  hidden: [B, T, D]

        ## TODO: save pre-vwn layer output
        # ── 2. pre_vwn_layer：特征融合 + 扩维到 D' ───────────────────────
        vwn_states = self.pre_vwn_layer(embeds, hidden)
        # vwn_states: [B, T, D']

        # ── 3. A2：D' → D（Attn 输入）+ D'（VWN 残差） ───────────────────
        vwn_states_grouped = vwn_states.view(-1, self.d_prime//self.vwn.m) # [B*T*m, D'//m]
        daf_out_grounped = self.downward_and_forgot(vwn_states_grouped)          # [B*T*m, T, (D + D')//m]
        daf_out = daf_out_grounped.view(vwn_states.shape[0], vwn_states.shape[1], self.hidden_size + self.d_prime) #[# [B, T, D + D']
        attn_in, vwn_residual = daf_out.split([mid, daf_out.shape[-1] - mid], dim=-1)
        # attn_in:      [B, T, D]
        # vwn_residual: [B, T, D']

        # ── 4. Attention（在 D 空间，q/k/v 保持原始尺寸） ─────────────────
        attn_in = self.pre_attention_layernorm(attn_in)
        attn_out, _ = self.self_attn(
            hidden_states=attn_in,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # attn_out: [B, T, D]

        # ── 5. B1：将 Attn 输出扩维并更新 VWN 残差 ───────────────────────
        # TODO: print hidden_states after attention
        attn_out_grouped = attn_out.view(-1, self.hidden_size//self.vwn.m) # [B*T*m, D/m]
        attn_out_upward_grouped = self.upward_after_attn(attn_out_grouped) # [B*T*m, D'/m]
        attn_out_upward = attn_out_upward_grouped.view(attn_out.shape[0], attn_out.shape[1], self.d_prime) # [B, T, D']
        vwn_states = vwn_residual + attn_out_upward # [B, T, D']
        # vwn_states: [B, T, D']

        # ── 6. A3：D' → D（FFN 输入）+ D'（VWN 残差） ────────────────────
        vwn_states_grouped =  vwn_states.view(-1, self.d_prime//self.vwn.m) #[B*T*m, D'//m]
        daf2_out_grouped = self.downward_and_forgot_after_attn(vwn_states_grouped) #[B*T*m, (D'+D)//m]
        daf2_out = daf2_out_grouped.view(vwn_states.shape[0], vwn_states.shape[1], self.d_prime + self.hidden_size) #[B, T, D'+D]

        ffn_in, vwn_residual2 = daf2_out.split([mid, daf2_out.shape[-1] - mid], dim=-1)
        # ffn_in:        [B, T, D]
        # vwn_residual2: [B, T, D']
        
        # ── 7. FFN（在 D 空间，mlp 直接复用基类） ────────────────────────
        ffn_in = self.post_attention_layernorm(ffn_in)
        mlp_out = self.mlp(ffn_in)
        # mlp_out: [B, T, D]

        # ── 8. B2：将 FFN 输出扩维并更新 VWN 残差 ────────────────────────
        # TODO: print hidden_states after MLP
        mlp_out_grouped = mlp_out.view(-1, self.hidden_size//self.vwn.m) # [B*T*m, D/m]
        mlp_out_upward_grouped = self.upward_after_mlp(mlp_out_grouped) # [B*T*m, D'/m]
        mlp_out_upward = mlp_out_upward_grouped.view(mlp_out.shape[0], mlp_out.shape[1], self.d_prime) # [B, T, D']
        vwn_states = vwn_residual2 + mlp_out_upward # [B, T, D']
        # vwn_states: [B, T, D']

        if self.vwn.expand is False:
            return vwn_states

        # ── 9. A4：降回 D，传给下一层 ─────────────────────────────────────
        # TODO: print hidden_states after MLP
        vwn_states_grouped = vwn_states.view(-1, self.d_prime//self.vwn.m) # [B*T*m, D'/m]
        vwn_states_upward_grouped = self.downward(vwn_states_grouped) # [B*T*m, D/m]
        vwn_states_upward = vwn_states_upward_grouped.view(vwn_states.shape[0], vwn_states.shape[1], self.hidden_size)

        return vwn_states_upward
        # output: [B, T, D]


# ──────────────────────────────────────────────────────────────────────────────
# 具体 Layer 类
# ──────────────────────────────────────────────────────────────────────────────

class LlamaDecoderVwnEagle3FirstLayer(VwnEagle3FirstLayerMixin, LlamaDecoderLayer):
    """LLaMA 解码层 + VWN+Eagle3 第一层改造。

    对应 eagle3/model_definitions.py 中的 LlamaDecoderEagle3FirstLayer，
    写法完全对称：仅基类和 norm_class 与 Qwen3 版本不同。

    Args:
        config:               LlamaConfig
        layer_idx:            层编号（传给 LlamaDecoderLayer）
        vwn:                  VwnConfig 实例
        norm_before_residual: 保留参数签名以与训练脚本兼容，VWN 内部不依赖该标志
        pre_vwn_layer_class:  支持替换为 VwnEagle3PreVwnLayerV1（消融用）
    """

    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        vwn: VwnConfig,
        norm_before_residual: bool = True,
        pre_vwn_layer_class: type = VwnEagle3PreVwnLayer,
    ):
        super().__init__(config, layer_idx)
        self._patch_vwn_eagle3_projections(config, LlamaRMSNorm, vwn, pre_vwn_layer_class)


# ──────────────────────────────────────────────────────────────────────────────
# model_classes 注册表（与 eagle3/model_definitions.py 格式相同）
# ──────────────────────────────────────────────────────────────────────────────

model_classes: dict[str, base_components.ModelComponents] = {
    "llama": base_components.override_components(
        "llama", first_layer_class=LlamaDecoderEagle3FirstLayer
    ),
    "qwen3": base_components.override_components(
        "qwen3", first_layer_class=Qwen3DecoderEagle3FirstLayer
    ),
}
