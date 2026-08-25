"""The DSpark method over a DeepSeek-V4-Flash decoder backbone.

Subclasses :class:`~speculators.models.dspark.core.DSparkDraftModel` and overrides ONLY
the decoder stack inside ``_backbone_forward``: anchor sampling, the target
distribution, the Markov and confidence heads, the compound loss, registration,
``from_training_args`` and the data contract are all inherited unchanged. In place of
the Qwen3 DFlash decoder layers it runs the DSV4 stack -- multi-head latent attention
with per-head sinks, a 256-expert MoE, and hyper-connections -- with DSV4's interleaved
RoPE.

The block-attention contract is identical to DFlash's: the draft block queries
(``noise_embedding [1, TB, H]``) attend to ``[target-hidden context | block]`` under the
additive ``attention_mask`` (block-diagonal + sliding window). ``MhcDecoderBlock``
already takes exactly that shape (``block_x``, ``context_x``, per-position freqs,
``attn_bias``), so this module only wires up the RoPE positions and the mask, and
manages the hyper-connection streams across the stack (expand once, collapse with the
HyperHead at the end).
"""

from __future__ import annotations

import logging
import os
from typing import ClassVar, Literal, cast

import torch
from torch import nn
from transformers import PretrainedConfig

from speculators import SpeculatorModelConfig
from speculators.model import SpeculatorModel
from speculators.models.dflash.utils import get_base_indices_for_anchored_blocks
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel

# Absolute (not relative) imports: transformers' custom_object_save parses the config
# module's RELATIVE imports to bundle them for trust_remote_code, and mishandles
# two-level ones (``from .backbone.block`` -> looks for the file ``backbone.block.py``).
# Absolute imports are not parsed, so save_pretrained works.
from speculators.models.dsv4_dspark import checkpoint_mapping
from speculators.models.dsv4_dspark.backbone.block import MhcDecoderBlock
from speculators.models.dsv4_dspark.backbone.hyper import HyperHead
from speculators.models.dsv4_dspark.backbone.rotary import precompute_freqs_cis
from speculators.models.dsv4_dspark.config import DSparkDraftConfig

__all__ = ["DSV4DSparkConfig", "DSV4DSparkDraftModel", "resolve_init_parts"]

# A parameter is "matrix-like" (and worth a NaN/Inf scan) from rank 2 up.
_MATRIX_RANK = 2
# Parts of a draft layer that can be warm-started from the verifier.
_INIT_PARTS = frozenset({"attn", "moe", "hc", "norm"})


def resolve_init_parts(requested: list[str] | None) -> frozenset[str]:
    """Validate ``--init-from-target`` and expand ``all``."""
    parts = frozenset(requested or ())
    unknown = parts - _INIT_PARTS - {"all"}
    if unknown:
        raise ValueError(
            f"--init-from-target: unknown part(s) {sorted(unknown)}; expected any of "
            f"{sorted(_INIT_PARTS)} or 'all'"
        )
    return _INIT_PARTS if "all" in parts else parts


# The additive attention bias is [1, TB, Sk]; anything wider carries a head axis.
_BIAS_RANK = 3

logger = logging.getLogger(__name__)


@SpeculatorModelConfig.register("dsv4_dspark")
class DSV4DSparkConfig(DSparkSpeculatorConfig):
    """Dense-line DSpark config + the DSV4 sparse-backbone hyperparameters.

    ``transformer_layer_config`` still carries the shared shape the SpeculatorModel
    machinery needs (hidden_size, vocab_size, num_hidden_layers = draft depth,
    rms_norm_eps); the fields below configure the MLA + MoE + mHC backbone.
    """

    speculators_model_type: Literal["dsv4_dspark"] = "dsv4_dspark"  # type: ignore[assignment]

    # multi-head latent attention
    num_heads: int = 64
    head_dim: int = 512
    rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    window_size: int = 128
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    original_seq_len: int = 65536
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    # mixture of experts
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    moe_inter_dim: int = 2048
    score_func: str = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    # hyper-connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    def backbone_config(self) -> DSparkDraftConfig:
        """Build the plain dataclass the backbone modules consume."""
        tl = self.transformer_layer_config
        return DSparkDraftConfig(
            vocab_size=tl.vocab_size,
            hidden_size=tl.hidden_size,
            rms_norm_eps=tl.rms_norm_eps,
            n_draft_layers=tl.num_hidden_layers,
            block_size=self.block_size,
            noise_token_id=self.mask_token_id or 0,
            target_layer_ids=tuple(self.aux_hidden_state_layer_ids or (0, 1, 2)),
            markov_rank=self.markov_rank,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            q_lora_rank=self.q_lora_rank,
            o_lora_rank=self.o_lora_rank,
            o_groups=self.o_groups,
            window_size=self.window_size,
            rope_theta=self.rope_theta,
            rope_factor=self.rope_factor,
            original_seq_len=self.original_seq_len,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
            n_routed_experts=self.n_routed_experts,
            n_shared_experts=self.n_shared_experts,
            n_activated_experts=self.n_activated_experts,
            moe_inter_dim=self.moe_inter_dim,
            score_func=self.score_func,
            route_scale=self.route_scale,
            swiglu_limit=self.swiglu_limit,
            hc_mult=self.hc_mult,
            hc_sinkhorn_iters=self.hc_sinkhorn_iters,
            hc_eps=self.hc_eps,
        )


@SpeculatorModel.register("dsv4_dspark")
class DSV4DSparkDraftModel(DSparkDraftModel):
    """DSpark method (inherited) over the DSV4-native sparse backbone."""

    config_class: ClassVar[type[DSV4DSparkConfig]] = DSV4DSparkConfig  # type: ignore[assignment,misc]
    _no_split_modules: ClassVar[list[str]] = ["MhcDecoderBlock"]  # type: ignore[assignment,misc]

    freqs_cis: torch.Tensor  # non-persistent buffer, see __init__

    def __init__(self, config: DSV4DSparkConfig) -> None:
        # Force the additive (eager) float mask BEFORE super().__init__ reads it to pick
        # the mask builder: sink attention consumes it as an additive bias.
        config.transformer_layer_config._attn_implementation = "eager"  # noqa: SLF001
        # DFlash/DSpark __init__ builds fc(=main_proj role)/hidden_norm/norm/embed/
        # lm_head/verifier + markov/confidence + a Qwen3 layer stack, discarded below.
        super().__init__(config=config)
        bb = config.backbone_config()
        self.backbone_cfg = bb

        # DSPARK_RECOMPUTE=1 recomputes each draft layer in backward instead of
        # keeping its activations, trading step time for the memory to run larger
        # batches. Applied in _backbone_forward.
        self.grad_checkpoint = os.environ.get(
            "DSPARK_RECOMPUTE", ""
        ).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        # The released DSV4 layout has no slot for a bias on the confidence head, and
        # upstream's ConfidenceHead always builds one. Replace the projection rather
        # than changing how that head is built for every other DSpark draft.
        if self.confidence_head is not None and not getattr(
            config, "confidence_head_bias", False
        ):
            proj = self.confidence_head.proj
            self.confidence_head.proj = nn.Linear(proj.in_features, 1, bias=False)

        # Swap the decoder stack for the DSV4 blocks and the hyper-connection head.
        self.layers = nn.ModuleList(
            MhcDecoderBlock(bb) for _ in range(bb.n_draft_layers)
        )
        self.hc_head = HyperHead(bb)

        # Interleaved DSV4 RoPE cache, indexed by absolute position (YaRN is off on
        # the sliding path -> original_seq_len=0). Held as a REAL [seqlen, rope//2, 2]
        # tensor rather than a complex one: a complex buffer cast to bf16 under AMP
        # loses its imaginary part, which turns the rotation into a scale with nothing
        # raised. The real cos/sin form survives that cast and indexes on any backend.
        # See apply_rotary_emb.
        self._rope_dim = bb.rope_head_dim
        self.register_buffer(
            "freqs_cis",
            torch.view_as_real(
                precompute_freqs_cis(
                    bb.rope_head_dim,
                    bb.original_seq_len or 1,
                    0,
                    bb.rope_theta,
                    bb.rope_factor,
                    bb.beta_fast,
                    bb.beta_slow,
                )
            ).contiguous(),
            persistent=False,
        )
        self._init_backbone_params()

        # The released layout puts the conditioning projection on the first stage and
        # the heads on the last, which a rule can only express with a concrete index.
        # The module-level registration below covers the released depth; a draft of
        # another depth re-registers for its own.
        if bb.n_draft_layers != checkpoint_mapping.RELEASED_N_LAYERS:
            checkpoint_mapping.register(n_layers=bb.n_draft_layers)

    def freeze_routed_experts(self) -> int:
        """Hold the routed experts read-only; returns how many tensors were frozen.

        The routed experts are ~99% of this model's parameters, and training the rest
        (attention, hyper-connections, the projections and the two heads) is a coherent
        recipe on its own: with the experts read-only every rank holds the same weights,
        nothing has to be sharded across ranks, and ordinary FSDP over the remainder is
        enough. The optimizer picks this up on its own -- it already skips parameters
        with ``requires_grad=False``.

        The memory does not go away, it only stops being sharded: the frozen stack is
        still resident on every rank.
        """
        frozen = 0
        for block in self.blocks:
            for param in block.ffn.experts.parameters():
                param.requires_grad_(False)
                frozen += 1
        return frozen

    @property
    def blocks(self) -> list[MhcDecoderBlock]:
        """``self.layers`` with its element type preserved.

        ``nn.ModuleList`` indexing returns a bare ``Module``, which loses every
        submodule this code reaches for (``ffn.router``, ``ffn.experts``, ``attn``).
        """
        return cast("list[MhcDecoderBlock]", list(self.layers))

    def state_dict_from_checkpoint(self, state_dict: dict) -> dict:
        """Translate a released-layout checkpoint into this model's parameter names.

        ``save_pretrained`` writes the released ``mtp.*`` layout, but training does not
        resume through ``from_pretrained``: it reads the safetensors directly, against
        parameters named ``layers.*``. This runs before ``set_model_state_dict``, not
        inside ``load_state_dict``, because that call has already sharded the tensors
        it prepared into DTensors, and a full tensor handed over afterwards would be a
        plain-versus-DTensor copy.

        The guard is load-bearing: resume passes ``strict=False``, which it needs
        because the verifier weights are absent by design, and that also turns a layout
        mismatch into a silent no-op -- nothing loads, nothing raises, and training
        continues from the initial weights.

        Checkpoints already in module layout pass through: their keys match no rule.
        """
        if not checkpoint_mapping.is_released_layout(state_dict):
            return state_dict
        translated = checkpoint_mapping.to_module_layout(
            state_dict, n_layers=self.backbone_cfg.n_draft_layers
        )
        uncovered = sorted(
            name
            for name, _ in self.named_parameters()
            if name not in translated and not name.startswith("verifier_")
        )
        if uncovered:
            raise RuntimeError(
                f"a released-layout checkpoint left {len(uncovered)} parameters with "
                f"no source, e.g. {uncovered[:3]}. Loading it would leave them at "
                f"their initial values."
            )
        return translated

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> DSV4DSparkDraftModel:
        """Build the model from training arguments.

        The DSV4 backbone shape comes from the config defaults, which are the released
        draft's; only the DSpark method fields are taken from the CLI.
        """
        config = DSV4DSparkConfig(
            **cls._build_base_config_kwargs("dsv4_dspark", verifier_config, **kwargs),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_confidence_head=kwargs.get("enable_confidence_head", True),
            confidence_head_with_markov=kwargs.get("confidence_head_with_markov", True),
        )
        model = cls(config=config)
        if kwargs.get("freeze_experts"):
            n = model.freeze_routed_experts()
            logger.info("--freeze-experts: %d routed-expert tensors held read-only", n)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        parts = resolve_init_parts(kwargs.get("init_from_target"))
        if parts:
            model.load_verifier_layer(parts)
        return model

    def load_verifier_weights(self) -> None:  # noqa: C901
        """Load the shared frozen embed + lm_head + verifier norm from the DSV4
        verifier, which uses the OFFICIAL DeepSeek key names (``embed.weight`` /
        ``head.weight`` / ``norm.weight``) rather than HF's (``embed_tokens`` /
        ``lm_head`` / ``model.norm``). Same masking/freezing as the base."""
        import warnings  # noqa: PLC0415

        from speculators.utils.loading import load_model_layers  # noqa: PLC0415

        # On a meta build the parameters carry no data, so there is nothing to load;
        # the real weights arrive by broadcast from rank 0. Freezing still has to
        # happen here: every rank must agree on which parameters are trainable, or
        # FSDP2's gradient reduce-scatter hangs.
        if self.embed_tokens.weight.is_meta:
            self.embed_tokens.weight.requires_grad_(False)
            self.lm_head.weight.requires_grad_(False)
            self.verifier_lm_head.weight.requires_grad_(False)
            if hasattr(self, "verifier_norm"):
                self.verifier_norm.weight.requires_grad_(False)
            return

        sc = getattr(getattr(self, "config", None), "speculators_config", None)
        if sc is None or sc.verifier.name_or_path is None:
            return
        w = load_model_layers(
            ["embed.weight", "head.weight", "norm.weight"], sc.verifier.name_or_path
        )
        embed_w = w["embed.weight"]
        lm_head_w = w.get("head.weight", embed_w)

        if self.embed_tokens.weight.isnan().any():
            self.embed_tokens.load_state_dict({"weight": embed_w})
        if self.use_draft_vocab:
            if self.t2d is None or not torch.any(self.t2d).item():
                raise ValueError("t2d not set; call load_vocab_mappings first.")
            lm_head_w = lm_head_w[
                self.t2d.to(device=lm_head_w.device, dtype=torch.bool), :
            ]
        if self.lm_head.weight.isnan().any():
            self.lm_head.load_state_dict(
                {"weight": lm_head_w.detach().clone()}, strict=False
            )
        self.verifier_lm_head.load_state_dict(
            {"weight": lm_head_w.detach().clone()}, strict=False
        )
        if hasattr(self, "verifier_norm"):
            if "norm.weight" not in w:
                warnings.warn(
                    f"no norm.weight in {sc.verifier.name_or_path}; using default.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                self.verifier_norm.load_state_dict({"weight": w["norm.weight"]})
        self.embed_tokens.weight.requires_grad_(False)
        self.lm_head.weight.requires_grad_(False)
        self.verifier_lm_head.weight.requires_grad_(False)
        if hasattr(self, "verifier_norm"):
            self.verifier_norm.weight.requires_grad_(False)

    def load_verifier_moe(self) -> None:
        """Warm-start only the MoE. See :meth:`load_verifier_layer`."""
        self.load_verifier_layer({"moe"})

    def load_verifier_layer(self, parts) -> None:  # noqa: C901
        """Warm-start the requested PARTS of each draft layer from the matching verifier
        layer: draft layer ``n`` <- verifier layer ``target_layer_ids[n]`` (the layers
        ``main_proj`` conditions on). A draft layer is architecturally a DSV4 target
        layer, so this is a strong TRAINABLE init (not frozen), unlike the shared
        embed/lm_head.

        ``parts`` is any subset of:
          * ``"attn"`` — the MLA **core** projections
          (wq_a/q_norm/wq_b/wkv/kv_norm/wo_a/
            wo_b/attn_sink). The verifier's sparse-attn ``compressor``/``indexer``
            submodules
            are skipped — the draft does dense sliding-window sink attention (no such
            parts).
          * ``"hc"``   — the two hyper-connections (``attn_hc`` <- ``hc_attn_*``,
            ``ffn_hc`` <- ``hc_ffn_*``; the sole flat->dotted rename).
          * ``"norm"`` — the two RMSNorms (``attn_norm``, ``ffn_norm``).
          * ``"moe"``  — routed experts (EP-local slice) + router (<- ``ffn.gate``) +
          shared.
            The router is never warm-started: neither its weight nor its balance
            bias transfers to a different hidden distribution.

        EP-aware for experts (the stacked ``GroupedExperts`` weights are still plain
        per-rank tensors at build time — before the ``Shard(0)`` wrap — so each rank
        copies only its
        ``[ep_expert_offset : +n_local]`` slice); the rest are replicated (full copy per
        rank). No-op on meta params (non-rank0 under ``--init-on-meta``; broadcast
        fills). Requires the draft dims to match the verifier's (the faithful config);
        raises on any missing
        key / shape mismatch. The verifier uses DeepSeek names, so copies are 1:1."""
        import logging  # noqa: PLC0415

        from speculators.utils.loading import load_model_layers  # noqa: PLC0415

        parts = set(parts)
        if not parts:
            return
        # meta build (non-rank0 under --init-on-meta): weights arrive via broadcast.
        if self.blocks[0].ffn.router.weight.is_meta:
            return
        sc = getattr(getattr(self, "config", None), "speculators_config", None)
        if sc is None or sc.verifier.name_or_path is None:
            return
        path = sc.verifier.name_or_path
        bb = self.backbone_cfg
        tids = bb.target_layer_ids
        if len(tids) < bb.n_draft_layers:
            raise ValueError(
                f"init-from-target needs >= n_draft_layers ({bb.n_draft_layers}) "
                f"target_layer_ids, got {tids}."
            )

        # replicated (non-EP) parts: (verifier-key suffix, draft attribute path). For
        # attn and norm the two are identical; hc renames the flat hc_{attn,ffn}_* ->
        # attn_hc/ffn_hc.
        part_map = {
            "attn": [
                ("attn.wq_a.weight", "attn.wq_a.weight"),
                ("attn.q_norm.weight", "attn.q_norm.weight"),
                ("attn.wq_b.weight", "attn.wq_b.weight"),
                ("attn.wkv.weight", "attn.wkv.weight"),
                ("attn.kv_norm.weight", "attn.kv_norm.weight"),
                ("attn.wo_a.weight", "attn.wo_a.weight"),
                ("attn.wo_b.weight", "attn.wo_b.weight"),
                ("attn.attn_sink", "attn.attn_sink"),
            ],
            "hc": [
                ("hc_attn_fn", "attn_hc.fn"),
                ("hc_attn_base", "attn_hc.base"),
                ("hc_attn_scale", "attn_hc.scale"),
                ("hc_ffn_fn", "ffn_hc.fn"),
                ("hc_ffn_base", "ffn_hc.base"),
                ("hc_ffn_scale", "ffn_hc.scale"),
            ],
            "norm": [
                ("attn_norm.weight", "attn_norm.weight"),
                ("ffn_norm.weight", "ffn_norm.weight"),
            ],
        }

        def _need(dct: dict, k: str):
            if k not in dct:
                raise KeyError(f"init-from-target: '{k}' not found in verifier {path}")
            return dct[k]

        def _copy(param, src) -> None:
            src = src.to(device=param.device, dtype=param.dtype)
            if tuple(src.shape) != tuple(param.shape):
                raise ValueError(
                    f"init-from-target: source shape {tuple(src.shape)} != draft "
                    f"{tuple(param.shape)} -- needs a config matching the verifier."
                )
            param.data.copy_(src)

        def _resolve(module, dotted: str):
            obj = module
            for a in dotted.split("."):
                obj = getattr(obj, a)
            return obj

        for n in range(bb.n_draft_layers):
            blk = self.blocks[n]
            pre = f"layers.{tids[n]}."

            simple = [
                (pre + vk, da)
                for part in ("attn", "hc", "norm")
                if part in parts
                for vk, da in part_map[part]
            ]

            moe_keys: list[str] = []
            eids: list[int] = []
            if "moe" in parts:
                ffn = blk.ffn
                off = int(getattr(ffn, "ep_expert_offset", 0))
                eids = [off + i for i in range(ffn.experts.w1.shape[0])]
                moe_keys = [
                    pre + "ffn.gate.weight",
                    pre + "ffn.gate.bias",
                    pre + "ffn.shared_experts.w1.weight",
                    pre + "ffn.shared_experts.w2.weight",
                    pre + "ffn.shared_experts.w3.weight",
                ]
                for e in eids:
                    moe_keys += [
                        f"{pre}ffn.experts.{e}.w1.weight",
                        f"{pre}ffn.experts.{e}.w2.weight",
                        f"{pre}ffn.experts.{e}.w3.weight",
                    ]

            w = load_model_layers([vk for vk, _ in simple] + moe_keys, path)

            for vk, da in simple:
                _copy(_resolve(blk, da), _need(w, vk))

            if "moe" in parts:
                # The experts are warm-started; the router is not. The verifier's
                # gate.weight was fitted to spread its own much wider hidden
                # distribution, and on the draft's narrower one it concentrates routing
                # on a few experts; its balance bias solves for the verifier's routing,
                # not the draft's. A small normal init routes close to uniformly at
                # step 0 and lets training find the draft's own routing. Set here
                # explicitly, since the build-time torch.empty is not usable.
                nn.init.normal_(ffn.router.weight, std=0.02)
                for wn in ("w1", "w2", "w3"):
                    _copy(
                        getattr(ffn.shared_experts, wn).weight,
                        _need(w, f"{pre}ffn.shared_experts.{wn}.weight"),
                    )
                # routed experts: stacked [n_local, out, in]; local slot i <- global
                # expert off+i
                for wn in ("w1", "w2", "w3"):
                    stacked = getattr(ffn.experts, wn)
                    slot_shape = tuple(stacked.shape[1:])
                    for i, e in enumerate(eids):
                        src = _need(w, f"{pre}ffn.experts.{e}.{wn}.weight").to(
                            device=stacked.device, dtype=stacked.dtype
                        )
                        if tuple(src.shape) != slot_shape:
                            raise ValueError(
                                f"init-from-target: expert {e} {wn} "
                                f"{tuple(src.shape)} != "
                                f"draft slot {slot_shape} (needs faithful config)."
                            )
                        stacked.data[i].copy_(src)

        router_note = " [router: freshly initialized]" if "moe" in parts else ""
        logging.getLogger(__name__).info(
            "init-from-target: warm-started %s of %d draft layers from verifier %s.%s",
            sorted(parts),
            bb.n_draft_layers,
            list(tids[: bb.n_draft_layers]),
            router_note,
        )

    def _init_backbone_params(self) -> None:
        """Initialize the freshly-built backbone params (post_init ran on the old
        Qwen3 layers). Uninitialized ``torch.empty`` params (mHC fn) would NaN."""
        # Meta build (non-rank0 under --init-on-meta): no data to init; the random init
        # done on rank 0 is broadcast to every rank in the FSDP setup.
        if any(p.is_meta for p in self.parameters()):
            return
        std = 0.02
        for m in [*self.layers, self.hc_head]:
            for name, p in m.named_parameters():
                matrix_like = p.dim() >= _MATRIX_RANK and (
                    ".fn" in name or "weight" in name or "hc_fn" in name
                )
                degenerate = matrix_like and (
                    torch.isnan(p).any()
                    or not p.abs().sum().isfinite()
                    or p.abs().sum() == 0
                )
                if degenerate:
                    nn.init.normal_(p, std=std)

    def _rebuild_freqs(self, seqlen: int) -> None:
        self.freqs_cis = (
            torch.view_as_real(
                precompute_freqs_cis(
                    self._rope_dim,
                    seqlen,
                    0,
                    self.backbone_cfg.rope_theta,
                    self.backbone_cfg.rope_factor,
                    self.backbone_cfg.beta_fast,
                    self.backbone_cfg.beta_slow,
                )
            )
            .contiguous()
            .to(self.freqs_cis.device)
        )

    def _rope_at(self, positions: torch.Tensor) -> torch.Tensor:
        """Rotary cache at the given absolute positions.

        Returns a REAL ``[len, rope_dim//2, 2]`` view (cos in ``[...,0]``, sin in
        ``[...,1]``); the buffer holds ``view_as_real(freqs)``. See
        :func:`apply_rotary_emb` for why the real form rather than a complex one.
        """
        # from_pretrained's meta-init zeroes persistent=False FLOATING buffers via
        # to_empty() (it doesn't re-run __init__ and freqs_cis isn't in the state_dict)
        # -> rebuild once. (Training builds via from_training_args, no meta, so this is
        # a one-shot no-op there.)
        if not getattr(self, "_freqs_ok", False):
            self._freqs_ok = True
            if not bool(self.freqs_cis.any()):
                self._rebuild_freqs(self.freqs_cis.shape[0])
        if positions.numel() and int(positions.max()) >= self.freqs_cis.shape[0]:
            self._rebuild_freqs(int(positions.max()) + 1)
        return self.freqs_cis.to(positions.device)[positions]

    def _backbone_forward(  # noqa: C901
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        verifier_last_hidden_states: torch.Tensor,
        document_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        """DFlash scaffolding (copied) with the decoder stack swapped for our
        DSV4 sparse stack + DSV4 RoPE. Returns the same 5-tuple DSpark consumes."""
        device = hidden_states.device
        total_seq_len = hidden_states.shape[1]
        num_anchors = kwargs.pop("max_anchors", 3072)

        if position_ids is None:
            position_ids = torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid = (
            self._build_attention_mask(loss_mask, num_anchors, document_ids, device)
        )

        mask_tokens_size = num_anchors * self.block_size
        mask_token_ids = torch.full(
            (1, mask_tokens_size), self.mask_token_id, dtype=torch.long, device=device
        )
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        noise_embedding = self.embed_tokens(mask_token_ids)  # [1, TB, H]

        fc_output = self.fc(hidden_states)
        fc_output = self.hidden_norm(fc_output)  # [1, T, H]  (main_x context)

        block_positions = get_base_indices_for_anchored_blocks(
            position_ids[0, anchor_positions], self.block_size
        )  # [TB]
        ctx_positions = position_ids[0]  # [T]

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )

        with torch.no_grad():
            # verifier_last_hidden_states is already the verifier's final
            # post-norm hidden state, so its next-token logits are one lm_head
            # away; normalising again would not be the distribution the verifier
            # accepts against.
            verifier_logits = self.verifier_lm_head(verifier_last_hidden_states)
            if not self.config.sample_from_anchor:
                # False: shift right by one, so slot j predicts the token at
                # position j and slot 0 is the given anchor. True (the DSpark
                # convention): no shift, slot k predicts position k+1.
                verifier_logits = torch.roll(verifier_logits, 1, dims=1)
            targets = verifier_logits[:, anchored_block_indices]

        # DSV4 RoPE freqs at the ctx and block absolute positions.
        ctx_freqs = self._rope_at(ctx_positions)
        block_freqs = self._rope_at(block_positions)

        # mHC streams across the stack; each block attends noise -> [ctx | block].
        hc = self.backbone_cfg.hc_mult
        streams = noise_embedding.unsqueeze(2).repeat(1, 1, hc, 1)  # [1, TB, hc, H]

        # noaux_tc load balancing: nudge the selection bias from the PREVIOUS step's
        # load, before this step's layers run. The bias must stay constant across a
        # forward and its activation-checkpoint recompute, or the recompute routes
        # differently and the saved tensors no longer match. The one-step lag is
        # immaterial, and the all-reduce inside makes the update identical on every
        # rank.
        if self.training and getattr(self.blocks[0].ffn.router, "_balance", False):
            if not getattr(self, "_balance_logged", False):
                self._balance_logged = True
                import torch.distributed as _d  # noqa: PLC0415

                if not _d.is_initialized() or _d.get_rank() == 0:
                    _r0 = self.blocks[0].ffn.router
                    logger.info(
                        "MoE noaux_tc load balancing on: rate=%s across %d routers",
                        getattr(_r0, "_balance_rate", None),
                        len(self.layers),
                    )
            for _layer in self.blocks:
                _layer.ffn.router.update_load_balance_bias()

        for layer_idx, layer in enumerate(self.layers):
            attn_bias = (
                sliding_window_attn_mask
                if layer_idx in self.sliding_window_indices
                else full_attn_mask
            )
            layer_args = (
                streams,
                fc_output,
                block_freqs,
                ctx_freqs,
                self._mask_to_bias(attn_bias),
            )
            if self.grad_checkpoint and self.training:
                # Recompute this layer in backward to free its activations.
                # use_reentrant=False so any collective inside the MoE replays
                # correctly under the saved-tensor hooks.
                from torch.utils.checkpoint import checkpoint  # noqa: PLC0415

                streams = checkpoint(layer, *layer_args, use_reentrant=False)
            else:
                streams = layer(*layer_args)

        # Expert-utilisation counters: routing collapse (a handful of the experts
        # taking most of the tokens) does not show up in the loss curve, so report it
        # directly. Rank 0 only, roughly every 20th forward, and free when off.
        if getattr(self.blocks[0].ffn.router, "_log_load", False):
            self._eload_ctr = getattr(self, "_eload_ctr", 0) + 1
            import torch.distributed as _dist  # noqa: PLC0415

            _rk = _dist.get_rank() if _dist.is_initialized() else 0
            if _rk == 0 and self._eload_ctr % 20 == 1:
                for _n in range(len(self.blocks)):
                    _c = getattr(self.blocks[_n].ffn.router, "_sel_counts", None)
                    if _c is None:
                        continue
                    n_exp = _c.shape[0]
                    _cf = _c.float()
                    _tot = _cf.sum().clamp(min=1)
                    _p = _cf / _tot
                    _used = int((_c > 0).sum())
                    _tk = min(16, n_exp)
                    _tv, _ti = _cf.topk(_tk)
                    _top = float(_tv.sum() / _tot)
                    _ent = float(
                        -(_p[_p > 0] * _p[_p > 0].log()).sum()
                        / torch.log(torch.tensor(float(n_exp)))
                    )
                    # The hottest expert ids: the same ids across inits and datasets
                    # means router collapse; ids that follow the data do not.
                    logger.info(
                        "[MoE load L%d] used=%d/%d dead=%d effective=%.1f "
                        "top%d=%.2f entropy=%.3f hot=%s "
                        "(entropy 1.0 = uniform, low = collapsed)",
                        _n,
                        _used,
                        n_exp,
                        n_exp - _used,
                        float(torch.exp(-(_p[_p > 0] * _p[_p > 0].log()).sum())),
                        _tk,
                        _top,
                        _ent,
                        _ti.tolist(),
                    )

        hidden = self.norm(self.hc_head(streams))  # [1, TB, H]
        logits = self.lm_head(hidden)

        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )
        if not self.config.sample_from_anchor:
            # False: slot 0 is the given anchor rather than a prediction, so its
            # loss is masked. True: slot 0 is trained from the anchor's hidden state.
            aligned_loss_mask[:, :: self.block_size] = 0

        return hidden, logits, targets, aligned_loss_mask, anchored_block_indices

    @staticmethod
    def _mask_to_bias(mask: torch.Tensor | None) -> torch.Tensor | None:
        """Reshape the DFlash eager float mask to the sink attn_bias [1, TB, Sk]."""
        if mask is None:
            return None
        # eager float mask is [1, 1, TB, Sk] (or [1, TB, Sk]); collapse the head dim.
        while mask.dim() > _BIAS_RANK:
            mask = mask.squeeze(1)
        return mask


# Registering at import is what makes ``save_pretrained`` write the released layout and
# ``from_pretrained`` read it, with no converter in between. It is a no-op for every
# other model, and degrades to "keep module names" if the transformers API moves.
checkpoint_mapping.register()
