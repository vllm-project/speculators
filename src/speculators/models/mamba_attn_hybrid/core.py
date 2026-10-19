import copy
import logging
from typing import ClassVar

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.dspark.model_definitions import ConfidenceHead, MarkovHead
from speculators.models.mamba_attn_hybrid.attention import (
    create_anchor_block_mask_mod,
    create_block_only_mask_mod,
)
from speculators.models.mamba_attn_hybrid.config import (
    MambaAttnHybridSpeculatorConfig,
    parse_block_pattern,
)
from speculators.models.mamba_attn_hybrid.loss import (
    LossConfig,
    resolve_loss_config,
)
from speculators.models.mamba_attn_hybrid.metrics import compute_metrics
from speculators.models.mamba_attn_hybrid.model_definitions import (
    HybridAttnSubLayer,
    HybridMambaSubLayer,
    HybridMLPSubLayer,
)
from speculators.models.mamba_attn_hybrid.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)

logger = logging.getLogger(__name__)


def _flat_rope_from_scalars(rope_theta, rope_scaling):
    rope_dict = {"rope_theta": rope_theta}
    if isinstance(rope_scaling, dict) and rope_scaling:
        rope_dict.update(rope_scaling)
    rope_dict.setdefault("rope_type", "default")
    return rope_dict


def resolve_verifier_rope(real_verifier_config, layer_id):
    model_type = getattr(real_verifier_config, "model_type", "") or ""
    rope_params_attr = getattr(real_verifier_config, "rope_parameters", None)
    if model_type.startswith("qwen") or model_type == "llama":
        if isinstance(rope_params_attr, dict) and rope_params_attr:
            rope_dict = dict(rope_params_attr)
        else:
            rope_dict = _flat_rope_from_scalars(
                real_verifier_config.rope_theta,
                getattr(real_verifier_config, "rope_scaling", None),
            )
        return rope_dict, None
    if "gemma" in model_type:
        layer_types = getattr(real_verifier_config, "layer_types", None)
        if layer_types is not None:
            layer_type = layer_types[layer_id]
        else:
            pattern = real_verifier_config.sliding_window_pattern
            layer_type = (
                "full_attention"
                if (layer_id + 1) % pattern == 0
                else "sliding_attention"
            )
        sliding = layer_type == "sliding_attention"
        if isinstance(rope_params_attr, dict) and layer_type in rope_params_attr:
            rope_dict = dict(rope_params_attr[layer_type])
            rope_dict.setdefault("rope_type", "default")
        elif sliding:
            rope_dict = {
                "rope_type": "default",
                "rope_theta": real_verifier_config.rope_local_base_freq,
            }
        else:
            rope_dict = _flat_rope_from_scalars(
                real_verifier_config.rope_theta,
                getattr(real_verifier_config, "rope_scaling", None),
            )
        sliding_window = real_verifier_config.sliding_window if sliding else None
        return rope_dict, sliding_window
    raise ValueError(
        f"unsupported verifier model_type {model_type!r}; only qwen3, llama, and gemma supported"
    )


def verifier_has_qk_norm(real_verifier_config):
    model_type = getattr(real_verifier_config, "model_type", "") or ""
    return model_type != "llama"


@SpeculatorModel.register("mamba_attn_hybrid")
class MambaAttnHybridDraftModel(DraftVocabMixin, SpeculatorModel):
    config_class: ClassVar[type[MambaAttnHybridSpeculatorConfig]] = (
        MambaAttnHybridSpeculatorConfig  # type: ignore[misc]
    )
    _no_split_modules = [
        "HybridMambaSubLayer",
        "HybridAttnSubLayer",
        "HybridMLPSubLayer",
    ]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "verifier_norm.weight",
        "t2d",
        "d2t",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def __init__(
        self,
        config: MambaAttnHybridSpeculatorConfig,
    ) -> None:
        if config.transformer_layer_config._attn_implementation is None:  # noqa: SLF001
            config.transformer_layer_config._attn_implementation = (  # noqa: SLF001
                "simple_flex_attention"
            )
        super().__init__(config=config)
        self._init_vocab(config)

        tl_config = config.transformer_layer_config
        tl_config.use_qk_norm = config.use_qk_norm

        if config.layer_mapping is not None:
            raise NotImplementedError(
                "layer_mapping not supported for mamba_attn_hybrid"
            )

        if config.aux_hidden_state_layer_ids is None:
            raise ValueError(
                "aux_hidden_state_layer_ids must be set in MambaAttnHybridSpeculatorConfig. "
                "Use MambaAttnHybridDraftModel.from_training_args() to resolve defaults."
            )
        self.target_layer_ids = config.aux_hidden_state_layer_ids

        if config.latent_fusion_layer_ids is None:
            raise ValueError(
                "latent_fusion_layer_ids must be set in MambaAttnHybridSpeculatorConfig. "
                "Use MambaAttnHybridDraftModel.from_training_args() to resolve defaults."
            )
        self.latent_fusion_layer_ids = config.latent_fusion_layer_ids
        F = len(self.latent_fusion_layer_ids)

        if config.block_pattern is None:
            raise ValueError(
                "block_pattern must be set in MambaAttnHybridSpeculatorConfig. "
                "Use MambaAttnHybridDraftModel.from_training_args() with --block-pattern."
            )
        block_pattern = parse_block_pattern(config.block_pattern)
        self.block_pattern = block_pattern
        num_attn = block_pattern.count("attention")
        num_mamba = block_pattern.count("mamba")

        if config.attn_kv_layer_ids is None:
            raise ValueError(
                "attn_kv_layer_ids must be set in MambaAttnHybridSpeculatorConfig "
                "(one verifier KV layer per attention sub-layer). Use "
                "MambaAttnHybridDraftModel.from_training_args() with --attn-kv-layer-ids."
            )
        if len(config.attn_kv_layer_ids) != num_attn:
            raise ValueError(
                f"attn_kv_layer_ids has {len(config.attn_kv_layer_ids)} entries but "
                f"block_pattern has {num_attn} attention sub-layer(s); they must match."
            )
        for lid in config.attn_kv_layer_ids:
            if lid not in self.target_layer_ids:
                raise ValueError(
                    f"attn_kv_layer_ids entry {lid} is not in aux_hidden_state_layer_ids "
                    f"{self.target_layer_ids}; the data only carries KV for those layers "
                    "(aux_hidden_state_layer_ids). from_training_args derives this set "
                    "from --attn-kv-layer-ids, so this only arises from direct config "
                    "construction with an inconsistent aux_hidden_state_layer_ids."
                )
        self.attn_kv_layer_ids = list(config.attn_kv_layer_ids)
        attn_kv_index = [
            self.target_layer_ids.index(lid) for lid in self.attn_kv_layer_ids
        ]

        num_attn_layers = num_attn
        if tl_config.num_hidden_layers != num_attn_layers:
            raise ValueError(
                f"transformer_layer_config.num_hidden_layers ({tl_config.num_hidden_layers}) "
                f"must equal the number of attention sub-layers ({num_attn_layers}). "
                "train.py derives --num-layers from --block-pattern automatically; if "
                "constructing the config directly, set num_hidden_layers to the attention count."
            )

        self.sliding_window = tl_config.sliding_window
        self.sliding_window_prefix = config.sliding_window_prefix
        self.sliding_window_non_causal = config.sliding_window_non_causal
        draft_sw_indices = [
            i
            for i in range(num_attn_layers)
            if tl_config.layer_types[i] == "sliding_attention"
        ]
        per_layer_sw = getattr(tl_config, "per_layer_sliding_window", None)
        if per_layer_sw is not None:
            self._layer_to_sw = {int(k): v for k, v in per_layer_sw.items()}
        else:
            self._layer_to_sw = dict.fromkeys(draft_sw_indices, self.sliding_window)
        if config.attn_rope is not None:
            for i in range(num_attn_layers):
                vw = config.attn_rope[i]["sliding_window"]
                if i not in self._layer_to_sw and vw is not None:
                    self._layer_to_sw[i] = vw
        self.sliding_window_indices = sorted(self._layer_to_sw.keys())
        self.uses_sliding_window_attn = bool(self.sliding_window_indices)
        self.uses_full_attn = bool(num_attn_layers - len(self.sliding_window_indices))

        self.mamba_d_state = config.mamba_d_state
        self.mamba_num_heads = config.mamba_num_heads
        self.mamba_head_dim = config.mamba_head_dim

        seed_mode = config.mamba_seed_mode
        self.disable_mamba_init_state_injection = bool(
            getattr(config, "disable_mamba_init_state_injection", False)
        )
        self.disable_target_kv_reuse = bool(
            getattr(config, "disable_target_kv_reuse", False)
        )
        layers: list[nn.Module] = []
        self._sub_type: list[str] = []
        self._sub_seed_idx: list[int] = []
        self._sub_kv_idx: list[int] = []
        self._sub_attn_idx: list[int] = []
        attn_i = 0
        mamba_i = 0
        for tok in block_pattern:
            if tok == "mamba":
                layers.append(
                    HybridMambaSubLayer(
                        tl_config,
                        d_state=config.mamba_d_state,
                        num_heads=config.mamba_num_heads,
                        head_dim=config.mamba_head_dim,
                        n_groups=config.mamba_n_groups,
                        conv_kernel=config.mamba_conv_kernel,
                        expand=config.mamba_expand,
                        num_anchors=config.max_anchors,
                    )
                )
                self._sub_type.append("mamba")
                if self.disable_mamba_init_state_injection:
                    self._sub_seed_idx.append(-1)
                elif seed_mode == "per_layer":
                    self._sub_seed_idx.append(mamba_i)
                elif seed_mode == "shared":
                    self._sub_seed_idx.append(0)
                else:
                    self._sub_seed_idx.append(0 if mamba_i == 0 else -1)
                self._sub_kv_idx.append(-1)
                self._sub_attn_idx.append(-1)
                mamba_i += 1
            elif tok == "attention":
                layers.append(
                    HybridAttnSubLayer(
                        tl_config,
                        layer_idx=attn_i,
                    )
                )
                self._sub_type.append("attention")
                self._sub_seed_idx.append(-1)
                self._sub_kv_idx.append(attn_kv_index[attn_i])
                self._sub_attn_idx.append(attn_i)
                attn_i += 1
            else:
                layers.append(
                    HybridMLPSubLayer(
                        tl_config,
                    )
                )
                self._sub_type.append("mlp")
                self._sub_seed_idx.append(-1)
                self._sub_kv_idx.append(-1)
                self._sub_attn_idx.append(-1)
        self.layers = nn.ModuleList(layers)

        if config.attn_rope is not None:
            _attn_sublayers = [
                layer for layer in self.layers if isinstance(layer, HybridAttnSubLayer)
            ]
            for _k, _sublayer in enumerate(_attn_sublayers):
                _sublayer.self_attn.sliding_window = self._layer_to_sw.get(_k, None)

        self.norm = Qwen3RMSNorm(
            tl_config.hidden_size,
            eps=tl_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        if config.attn_rope is not None:
            self.rotary_emb = None
            self._unique_rotaries = nn.ModuleList()
            _rope_key_to_idx: dict[tuple, int] = {}
            self.attn_rotary_idx: list[int] | None = []
            for _entry in config.attn_rope:
                _rope_params = _entry["rope_parameters"]
                _rope_type = _rope_params.get("rope_type", "default")
                _rope_theta = _rope_params["rope_theta"]
                _key = tuple(sorted((k, str(v)) for k, v in _rope_params.items()))
                if _key not in _rope_key_to_idx:
                    _rope_cfg = copy.deepcopy(tl_config)
                    _rope_cfg.rope_theta = _rope_theta
                    _rope_cfg.rope_scaling = (
                        None
                        if _rope_type in (None, "default")
                        else {
                            k: v for k, v in _rope_params.items() if k != "rope_theta"
                        }
                    )
                    _rope_cfg.partial_rotary_factor = _rope_params.get(
                        "partial_rotary_factor", 1.0
                    )
                    _rope_cfg.rope_parameters = dict(_rope_params)
                    _rope_cfg.rope_parameters.setdefault("partial_rotary_factor", 1.0)
                    _rope_key_to_idx[_key] = len(self._unique_rotaries)
                    self._unique_rotaries.append(
                        Qwen3RotaryEmbedding(_rope_cfg)  # type: ignore[arg-type]
                    )
                self.attn_rotary_idx.append(_rope_key_to_idx[_key])
        else:
            _rope_params = getattr(tl_config, "rope_parameters", None)
            if (
                isinstance(_rope_params, dict)
                and "partial_rotary_factor" in _rope_params
            ):
                tl_config.partial_rotary_factor = _rope_params["partial_rotary_factor"]
            self.rotary_emb = Qwen3RotaryEmbedding(tl_config)  # type: ignore[arg-type]
            self._unique_rotaries = None
            self.attn_rotary_idx = None

        self.verifier_norm = Qwen3RMSNorm(
            tl_config.hidden_size,
            eps=tl_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm.weight.requires_grad = False

        if self.disable_mamba_init_state_injection:
            self.fc = None
            self.hidden_norm = None
            self.fc_norm = None
            self.seed_projs = nn.ModuleList([])
        else:
            self.fc = nn.Linear(
                F * tl_config.hidden_size, tl_config.hidden_size, bias=False
            )
            self.hidden_norm = Qwen3RMSNorm(
                tl_config.hidden_size,
                eps=tl_config.rms_norm_eps,  # type: ignore[arg-type]
            )

            self.fc_norm = (
                nn.ModuleList(
                    [
                        Qwen3RMSNorm(tl_config.hidden_size, eps=tl_config.rms_norm_eps)
                        for _ in range(F)
                    ]
                )
                if config.fc_norm
                else None
            )

            seed_out = (
                config.mamba_num_heads * config.mamba_head_dim * config.mamba_d_state
            )
            num_seed_projs = num_mamba if seed_mode == "per_layer" else 1
            self.seed_projs = nn.ModuleList(
                [
                    nn.Linear(tl_config.hidden_size, seed_out, bias=False)
                    for _ in range(num_seed_projs)
                ]
            )

        self.block_size = config.block_size

        hidden_size = tl_config.hidden_size
        self.markov_head: MarkovHead | None = None
        if config.markov_rank > 0:
            self.markov_head = MarkovHead(
                verifier_vocab_size=self.verifier_vocab_size,
                draft_vocab_size=self.draft_vocab_size,
                markov_rank=config.markov_rank,
                hidden_size=hidden_size,
                head_type=config.markov_head_type,
            )
        self.confidence_head: ConfidenceHead | None = None
        if config.enable_confidence_head:
            if config.confidence_head_with_markov and self.markov_head is None:
                raise ValueError(
                    "confidence_head_with_markov=True requires markov_rank > 0."
                )
            conf_in_dim = hidden_size + (
                config.markov_rank if config.confidence_head_with_markov else 0
            )
            self.confidence_head = ConfidenceHead(conf_in_dim)

        logger.info("mamba_attn_hybrid recipe: %s", " ".join(block_pattern))
        for k, lid in enumerate(self.attn_kv_layer_ids):
            logger.info("Attention block %d <- verifier KV layer %d (direct)", k, lid)

        self.post_init()

    def load_verifier_weights(self):
        from speculators.utils.loading import load_model_layers  # noqa: PLC0415

        super().load_verifier_weights()

        mask_token_id = getattr(self.config, "mask_token_id", None)
        if (
            mask_token_id is not None
            and 0 <= mask_token_id < self.embed_tokens.weight.shape[0]
            and self.embed_tokens.weight[mask_token_id].float().norm().item() < 1e-4
        ):
            table = self.embed_tokens.weight
            dead_l2 = table[mask_token_id].float().norm().item()
            std = table.float().std().item()
            generator = torch.Generator().manual_seed(0)
            new_row = torch.randn(table.shape[1], generator=generator) * std
            with torch.no_grad():
                table[mask_token_id] = new_row.to(
                    dtype=table.dtype, device=table.device
                )
            logger.warning(
                "mask_token_id=%d has a numerically dead verifier embedding row "
                "(L2=%.3e); re-initialized from a fixed seed with std=%.5f "
                "(new L2=%.4f) and kept frozen.",
                mask_token_id,
                dead_l2,
                std,
                table[mask_token_id].float().norm().item(),
            )

        verifier_path = self.config.speculators_config.verifier.name_or_path
        if verifier_path is None:
            return

        cfg = self.config

        if cfg.init_q_proj == "random" and not cfg.trainable_q_proj:
            raise ValueError("init_q_proj='random' requires trainable_q_proj=True")
        if cfg.init_kv_proj == "random" and not cfg.trainable_kv_proj:
            raise ValueError("init_kv_proj='random' requires trainable_kv_proj=True")
        use_qk_norm = getattr(cfg, "use_qk_norm", True)
        if use_qk_norm and cfg.init_q_norm == "random" and not cfg.trainable_q_norm:
            raise ValueError("init_q_norm='random' requires trainable_q_norm=True")
        if use_qk_norm and cfg.init_k_norm == "random" and not cfg.trainable_k_norm:
            raise ValueError("init_k_norm='random' requires trainable_k_norm=True")

        layer_prefix = getattr(self.config, "verifier_layer_prefix", "model.layers")
        load_layer_ids = sorted(set(self.attn_kv_layer_ids))
        weight_names = []
        for layer_id in load_layer_ids:
            prefix = f"{layer_prefix}.{layer_id}.self_attn"
            names = [
                f"{prefix}.q_proj.weight",
                f"{prefix}.k_proj.weight",
                f"{prefix}.v_proj.weight",
            ]
            if use_qk_norm:
                names.append(f"{prefix}.q_norm.weight")
                names.append(f"{prefix}.k_norm.weight")
            weight_names.extend(names)
        weight_names = list(dict.fromkeys(weight_names))
        verifier_weights = (
            load_model_layers(weight_names, verifier_path) if weight_names else {}
        )

        attn_sublayers = [
            layer for layer in self.layers if isinstance(layer, HybridAttnSubLayer)
        ]
        for k, sublayer in enumerate(attn_sublayers):
            attn = sublayer.self_attn
            lid = self.attn_kv_layer_ids[k]
            prefix = f"{layer_prefix}.{lid}.self_attn"

            if cfg.init_q_proj == "verifier":
                attn.q_proj.load_state_dict(
                    {"weight": verifier_weights[f"{prefix}.q_proj.weight"]}
                )
            attn.q_proj.weight.requires_grad = cfg.trainable_q_proj

            if cfg.init_kv_proj == "verifier":
                attn.k_proj.load_state_dict(
                    {"weight": verifier_weights[f"{prefix}.k_proj.weight"]}
                )
                v_key = f"{prefix}.v_proj.weight"
                has_v_proj = v_key in verifier_weights
                if not has_v_proj:
                    logger.warning(
                        "Verifier layer %s has no v_proj (k_eq_v); initializing draft v_proj from k_proj.",
                        prefix,
                    )
                v_weight = verifier_weights.get(
                    f"{prefix}.v_proj.weight",
                    verifier_weights[f"{prefix}.k_proj.weight"],
                )
                attn.v_proj.load_state_dict({"weight": v_weight})
            attn.k_proj.weight.requires_grad = cfg.trainable_kv_proj
            attn.v_proj.weight.requires_grad = cfg.trainable_kv_proj

            if use_qk_norm:
                w_q = verifier_weights.get(f"{prefix}.q_norm.weight")
                if cfg.init_q_norm == "verifier" and w_q is not None:
                    attn.q_norm.load_state_dict({"weight": w_q})
                attn.q_norm.weight.requires_grad = cfg.trainable_q_norm or w_q is None

                w_k = verifier_weights.get(f"{prefix}.k_norm.weight")
                if cfg.init_k_norm == "verifier" and w_k is not None:
                    attn.k_norm.load_state_dict({"weight": w_k})
                attn.k_norm.weight.requires_grad = cfg.trainable_k_norm or w_k is None

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "MambaAttnHybridDraftModel":
        from speculators.config import (  # noqa: PLC0415
            SpeculatorsConfig,
            VerifierConfig,
        )
        from speculators.proposals.greedy import (  # noqa: PLC0415
            GreedyTokenProposalConfig,
        )

        if kwargs.get("layer_mapping") is not None:
            raise NotImplementedError(
                "layer_mapping not supported for mamba_attn_hybrid"
            )

        if kwargs.get("block_pattern") is None:
            raise ValueError(
                "--block-pattern is required for --speculator-type mamba_attn_hybrid, e.g. "
                "mamba mlp mamba mlp mamba attention mlp."
            )
        block_pattern = parse_block_pattern(kwargs["block_pattern"])
        num_attn = block_pattern.count("attention")

        if kwargs.get("target_layer_ids") is not None:
            raise ValueError(
                "--target-layer-ids is not used by mamba_attn_hybrid; the verifier-KV "
                "layers come from --attn-kv-layer-ids (pass the same ids to "
                "launch_vllm_nkv.py --attn-kv-layer-ids at data-gen)."
            )
        attn_kv_layer_ids = kwargs.get("attn_kv_layer_ids")
        if attn_kv_layer_ids is None:
            raise ValueError(
                "--attn-kv-layer-ids is required for mamba_attn_hybrid: it lists, in "
                "attention-block order, the verifier KV layer each attention sub-layer "
                "reads, and defines the data's KV set (= sorted(set(...)))."
            )
        if len(attn_kv_layer_ids) != num_attn:
            raise ValueError(
                f"--attn-kv-layer-ids has {len(attn_kv_layer_ids)} entries but "
                f"--block-pattern has {num_attn} attention sub-layer(s); they must match."
            )
        target_layer_ids = sorted(set(attn_kv_layer_ids))

        latent_fusion_layer_ids = kwargs.get("latent_fusion_layer_ids")
        if latent_fusion_layer_ids is None:
            latent_fusion_layer_ids = list(target_layer_ids)

        real_verifier_config = AutoConfig.from_pretrained(
            kwargs["verifier_name_or_path"], trust_remote_code=True
        )
        verifier_is_multimodal = hasattr(real_verifier_config, "text_config")
        if verifier_is_multimodal:
            real_verifier_config = real_verifier_config.text_config
        verifier_layer_prefix = (
            "model.language_model.layers" if verifier_is_multimodal else "model.layers"
        )
        n_verifier_layers = real_verifier_config.num_hidden_layers
        bad_ids = [lid for lid in target_layer_ids if not 0 <= lid < n_verifier_layers]
        if bad_ids:
            raise ValueError(
                f"--attn-kv-layer-ids {bad_ids} out of range for a verifier with "
                f"{n_verifier_layers} layers (valid 0..{n_verifier_layers - 1})."
            )

        def _reused_layer_attn_geom(lid):
            base_hd = getattr(real_verifier_config, "head_dim", None)
            base_kv = getattr(real_verifier_config, "num_key_value_heads", None)
            global_hd = getattr(real_verifier_config, "global_head_dim", None)
            global_kv = getattr(
                real_verifier_config, "num_global_key_value_heads", None
            )
            _, _sw = resolve_verifier_rope(real_verifier_config, lid)
            is_full = _sw is None
            k_eq_v = getattr(real_verifier_config, "attention_k_eq_v", False)
            hd = global_hd if (is_full and global_hd is not None) else base_hd
            kv = (
                global_kv if (is_full and k_eq_v and global_kv is not None) else base_kv
            )
            return (hd, kv)

        _reused_geoms = {_reused_layer_attn_geom(lid) for lid in target_layer_ids}
        if len(_reused_geoms) > 1:
            raise NotImplementedError(
                "reused verifier layers have non-uniform attention geometry "
                f"(head_dim, num_kv_heads) {sorted(_reused_geoms)}; per-sub-layer attention "
                "geometry is not supported — pick reused layers of one attention type."
            )
        if getattr(real_verifier_config, "attn_logit_softcapping", None):
            raise ValueError(
                "verifier attn_logit_softcapping is set but the hybrid draft attention does "
                "not support attention logit soft-capping."
            )

        _reused_hd, _reused_kv = next(iter(_reused_geoms))
        if _reused_hd is not None:
            verifier_config.head_dim = _reused_hd
        if _reused_kv is not None:
            verifier_config.num_key_value_heads = _reused_kv

        attn_rope = []
        for i in range(num_attn):
            rope_dict, sliding_window = resolve_verifier_rope(
                real_verifier_config, attn_kv_layer_ids[i]
            )
            layer_type = (
                "sliding_attention" if sliding_window is not None else "full_attention"
            )
            attn_rope.append(
                {
                    "rope_parameters": rope_dict,
                    "sliding_window": sliding_window,
                    "layer_type": layer_type,
                }
            )

        if attn_rope:
            verifier_config.rope_parameters = dict(attn_rope[0]["rope_parameters"])
            verifier_config.partial_rotary_factor = 1.0

        use_qk_norm = bool(kwargs.get("use_qk_norm", True))
        if use_qk_norm and not verifier_has_qk_norm(real_verifier_config):
            logger.warning(
                "use_qk_norm=True with a normless verifier (model_type=%r): draft "
                "q_norm/k_norm start at ones and are forced trainable (no verifier "
                "weights to copy).",
                getattr(real_verifier_config, "model_type", ""),
            )
        if not use_qk_norm and (
            kwargs.get("trainable_q_norm", False)
            or kwargs.get("trainable_k_norm", False)
        ):
            logger.warning(
                "--no-use-qk-norm: ignoring --trainable-q-norm/--trainable-k-norm "
                "(norms disabled).",
            )

        config = MambaAttnHybridSpeculatorConfig(
            transformer_layer_config=verifier_config,
            draft_vocab_size=kwargs["draft_vocab_size"],
            block_size=kwargs.get("block_size", 8),
            max_anchors=kwargs.get("max_anchors", 3072),
            aux_hidden_state_layer_ids=target_layer_ids,
            mask_token_id=kwargs.get("mask_token_id"),
            sliding_window_non_causal=kwargs.get("sliding_window_non_causal", False),
            sliding_window_prefix=kwargs.get("sliding_window_prefix", 0),
            trainable_q_proj=kwargs.get("trainable_q_proj", False),
            trainable_kv_proj=kwargs.get("trainable_kv_proj", False),
            trainable_q_norm=kwargs.get("trainable_q_norm", False),
            trainable_k_norm=kwargs.get("trainable_k_norm", False),
            init_q_proj=kwargs.get("init_q_proj", "verifier"),
            init_kv_proj=kwargs.get("init_kv_proj", "verifier"),
            init_q_norm=kwargs.get("init_q_norm", "verifier"),
            init_k_norm=kwargs.get("init_k_norm", "verifier"),
            use_qk_norm=use_qk_norm,
            verifier_layer_prefix=verifier_layer_prefix,
            layer_mapping=None,
            mamba_d_state=kwargs.get("mamba_d_state", 16),
            mamba_num_heads=kwargs.get("mamba_num_heads", 64),
            mamba_head_dim=kwargs.get("mamba_head_dim", 64),
            mamba_n_groups=kwargs.get("mamba_n_groups", 1),
            mamba_conv_kernel=kwargs.get("mamba_conv_kernel", 4),
            mamba_expand=kwargs.get("mamba_expand", 1),
            latent_fusion_layer_ids=latent_fusion_layer_ids,
            mamba_seed_mode=kwargs.get("mamba_seed_mode", "per_layer"),
            disable_mamba_init_state_injection=kwargs.get(
                "disable_mamba_init_state_injection", False
            ),
            disable_target_kv_reuse=kwargs.get("disable_target_kv_reuse", False),
            block_pattern=block_pattern,
            attn_kv_layer_ids=attn_kv_layer_ids,
            attn_rope=attn_rope,
            markov_rank=kwargs.get("markov_rank", 0),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_confidence_head=kwargs.get("enable_confidence_head", False),
            confidence_head_with_markov=kwargs.get(
                "confidence_head_with_markov", False
            ),
            fc_norm=kwargs.get("fc_norm", False),
            speculators_config=SpeculatorsConfig(
                algorithm="mamba_attn_hybrid",
                proposal_methods=[
                    GreedyTokenProposalConfig(
                        speculative_tokens=kwargs.get("block_size", 8) - 1,
                    )
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_config, name_or_path=kwargs["verifier_name_or_path"]
                ),
            ),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        loss_config = resolve_loss_config(kwargs.get("loss_fn", "ce"))
        shared = {
            "parallel_drafting_loss_weight": kwargs.get(
                "parallel_drafting_loss_weight", "dflash"
            ),
            "dpace_alpha": kwargs.get("dpace_alpha", 0.5),
            "confidence_head_alpha": kwargs.get("confidence_head_alpha", 1.0),
            "loss_config": loss_config,
        }
        return dict(shared), dict(shared)

    @property
    def mask_token_id(self) -> int:
        if self.config.mask_token_id is None:
            raise ValueError(
                "mask_token_id is not set on the config. "
                "Pass --mask-token-id during training or ensure the config "
                "was saved with mask_token_id set."
            )
        return self.config.mask_token_id

    @torch.compiler.disable
    def _build_attention_mask(self, loss_mask, document_ids, device):
        total_seq_len = loss_mask.shape[1]

        anchor_positions, anchor_valid = select_anchors(
            loss_mask, self.config.max_anchors, self.block_size
        )

        full_attn_mask = None
        if self.uses_full_attn:
            mask_mod, q_len, kv_len = (
                create_block_only_mask_mod(
                    anchor_positions=anchor_positions,
                    block_size=self.block_size,
                    sliding_window=None,
                )
                if self.disable_target_kv_reuse
                else create_anchor_block_mask_mod(
                    document_ids=document_ids.squeeze(0).to(device),
                    total_seq_len=total_seq_len,
                    anchor_positions=anchor_positions,
                    block_size=self.block_size,
                    sliding_window=None,
                )
            )
            full_attn_mask = create_block_mask(
                mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device
            )

        sw_masks = {}
        if self.uses_sliding_window_attn:
            unique_sws = sorted(set(self._layer_to_sw.values()))
            for sw in unique_sws:
                mask_mod, q_len, kv_len = (
                    create_block_only_mask_mod(
                        anchor_positions=anchor_positions,
                        block_size=self.block_size,
                        sliding_window=sw,
                        sliding_window_non_causal=self.sliding_window_non_causal,
                    )
                    if self.disable_target_kv_reuse
                    else create_anchor_block_mask_mod(
                        document_ids=document_ids.squeeze(0).to(device),
                        total_seq_len=total_seq_len,
                        anchor_positions=anchor_positions,
                        block_size=self.block_size,
                        sliding_window=sw,
                        sliding_window_non_causal=self.sliding_window_non_causal,
                        sliding_window_prefix=self.sliding_window_prefix,
                    )
                )
                sw_masks[sw] = create_block_mask(
                    mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device
                )

        return full_attn_mask, sw_masks, anchor_positions, anchor_valid

    @torch.compile
    def forward(
        self,
        kv_caches: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_hidden_states: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        parallel_drafting_loss_weight: str = "dflash",
        dpace_alpha: float = 0.5,
        confidence_head_alpha: float = 1.0,
        loss_config: LossConfig | None = None,
        **kwargs,
    ):
        kv_caches = kv_caches.squeeze(0).permute(1, 2, 0, 3, 4).contiguous()

        device = kv_caches.device
        total_seq_len = input_ids.shape[1]
        num_anchors = self.config.max_anchors

        if (
            not self.disable_mamba_init_state_injection
            and verifier_hidden_states.shape[-1] != self.fc.in_features
        ):
            raise ValueError(
                f"verifier_hidden_states width {verifier_hidden_states.shape[-1]} != "
                f"fc.in_features {self.fc.in_features} "
                f"(F_train={len(self.latent_fusion_layer_ids)}); "
                "--latent-fusion-layer-ids must match between launch_vllm_nkv.py "
                "(data-gen) and scripts/train.py (training)."
            )

        if kv_caches.shape[0] != len(self.target_layer_ids):
            raise ValueError(
                f"injected verifier KV layer count {kv_caches.shape[0]} != "
                f"len(target_layer_ids) {len(self.target_layer_ids)} "
                f"({self.target_layer_ids}); the data's KV set must equal "
                "sorted(set(--attn-kv-layer-ids)) — keep --attn-kv-layer-ids consistent "
                "between launch_vllm_nkv.py (data-gen) and scripts/train.py (training)."
            )

        if document_ids is None:
            document_ids = torch.zeros(
                (1, total_seq_len), dtype=torch.long, device=device
            )
        if position_ids is None:
            position_ids = 1 + torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        full_attn_mask, sw_masks, anchor_positions, anchor_valid = (
            self._build_attention_mask(loss_mask, document_ids, device)
        )

        anchor_valid = anchor_valid & (anchor_positions > 0)

        mask_tokens_size = num_anchors * self.block_size
        mask_token_ids = torch.full(
            (1, mask_tokens_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        noise_embedding = self.embed_tokens(mask_token_ids)

        per_layer_kv = []
        for i in range(kv_caches.shape[0]):
            k = kv_caches[i, 0].unsqueeze(0).transpose(1, 2)
            v = kv_caches[i, 1].unsqueeze(0).transpose(1, 2)
            if self.disable_target_kv_reuse:
                k, v = k[:, :, :0], v[:, :, :0]
            per_layer_kv.append((k, v))

        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[:, anchor_positions], self.block_size, input_ids.numel()
        )
        position_ids = torch.cat([position_ids, mask_position_ids.unsqueeze(0)], dim=1)

        if self._unique_rotaries is not None:
            unique_position_embeddings = [
                rotary(noise_embedding, position_ids)
                for rotary in self._unique_rotaries
            ]
            position_embeddings = None
        else:
            position_embeddings = self.rotary_emb(noise_embedding, position_ids)
            unique_position_embeddings = None

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size, input_ids.numel()
        )

        if self.disable_mamba_init_state_injection:
            z = None
        else:
            seed_positions = (anchor_positions - 1).clamp(min=0)
            fusion_part = verifier_hidden_states[:, seed_positions, :]
            if self.fc_norm is not None:
                chunks = fusion_part.chunk(len(self.fc_norm), dim=-1)
                fusion_part = torch.cat(
                    [norm(c) for norm, c in zip(self.fc_norm, chunks, strict=True)],
                    dim=-1,
                )
            z = self.hidden_norm(self.fc(fusion_part)).squeeze(0)

        with torch.no_grad():
            verifier_logits = self.verifier_lm_head(
                self.verifier_norm(verifier_last_hidden_states)
            )
            verifier_logits = torch.roll(verifier_logits, 1, dims=1)
            targets = verifier_logits[:, anchored_block_indices]

        for i, layer in enumerate(self.layers):
            t = self._sub_type[i]
            if t == "mamba":
                seed_idx = self._sub_seed_idx[i]
                if seed_idx >= 0:
                    ssm_state0 = self.seed_projs[seed_idx](z).view(
                        num_anchors,
                        self.mamba_num_heads,
                        self.mamba_head_dim,
                        self.mamba_d_state,
                    )
                else:
                    ssm_state0 = None
                noise_embedding = layer(noise_embedding, ssm_state0)
            elif t == "attention":
                kv_idx = self._sub_kv_idx[i]
                attn_idx = self._sub_attn_idx[i]
                if attn_idx in self._layer_to_sw:
                    attn_mask = sw_masks[self._layer_to_sw[attn_idx]]
                else:
                    attn_mask = full_attn_mask
                if unique_position_embeddings is not None:
                    sub_position_embeddings = unique_position_embeddings[
                        self.attn_rotary_idx[attn_idx]
                    ]
                else:
                    sub_position_embeddings = position_embeddings
                noise_embedding = layer(
                    noise_embedding,
                    context_kv=per_layer_kv[kv_idx],
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    position_embeddings=sub_position_embeddings,
                    use_cache=False,
                    **kwargs,
                )
            else:
                noise_embedding = layer(noise_embedding)

        hidden = self.norm(noise_embedding)
        logits = self.lm_head(hidden)

        confidence_logits = None
        if self.markov_head is not None or self.confidence_head is not None:
            block = self.block_size
            block_tokens = input_ids[0, anchored_block_indices].view(num_anchors, block)
            prev_token_ids = torch.cat(
                [block_tokens[:, :1], block_tokens[:, :-1]], dim=1
            )
            hidden_blocks = hidden.view(num_anchors, block, -1)

            prev_emb = None
            if self.markov_head is not None:
                prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
                markov_bias = self.markov_head.block_bias(
                    prev_token_ids=prev_token_ids,
                    hidden_states=hidden_blocks,
                    prev_emb=prev_emb,
                )
                logits = (logits.view(num_anchors, block, -1) + markov_bias).view(
                    1, num_anchors * block, -1
                )

            if self.confidence_head is not None:
                if self.config.confidence_head_with_markov and prev_emb is not None:
                    conf_features = torch.cat(
                        [
                            hidden_blocks.detach(),
                            prev_emb.detach().to(hidden_blocks.dtype),
                        ],
                        dim=-1,
                    )
                else:
                    conf_features = hidden_blocks.detach()
                confidence_logits = self.confidence_head(conf_features).reshape(
                    1, num_anchors * block
                )

        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]

        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )

        aligned_loss_mask[:, :: self.block_size] = 0
        loss, metrics = compute_metrics(
            logits,
            targets,
            aligned_loss_mask,
            self.block_size,
            parallel_drafting_loss_weight=parallel_drafting_loss_weight,
            dpace_alpha=dpace_alpha,
            confidence_logits=confidence_logits,
            confidence_head_alpha=confidence_head_alpha,
            loss_config=loss_config,
        )
        draft_tokens = torch.argmax(logits, dim=-1)

        return draft_tokens, loss, metrics
