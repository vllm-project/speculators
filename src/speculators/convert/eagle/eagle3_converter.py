# speculators/convert/eagle/eagle3_converter.py

"""
Eagle-3 checkpoint converter with loguru logging.
"""

from pathlib import Path
from typing import Optional, Union

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, LlamaConfig, PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.convert.eagle.utils import (
    ensure_checkpoint_is_local,
    load_checkpoint_config,
    load_checkpoint_weights,
)
from speculators.models.eagle3 import Eagle3Speculator, Eagle3SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig


class Eagle3Converter:
    """
    Converter for Eagle3 checkpoints to speculators format.

    Handles weight remapping, embeddings replacement, and vLLM compatibility fixes.
    Produces production-ready models with standardized speculators_config metadata.
    """

    def convert(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        base_model: str,
        validate: bool = True,
        norm_before_residual: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        logger.info(f"Converting Eagle-3 checkpoint: {input_path}")

        local_checkpoint_path = ensure_checkpoint_is_local(input_path, cache_dir)

        eagle_config = load_checkpoint_config(local_checkpoint_path)
        weights = load_checkpoint_weights(local_checkpoint_path)
        logger.info(f"Loaded {len(weights)} weights")

        # Ensure target_vocab_size matches t2d tensor shape when present
        if "t2d" in weights and isinstance(weights["t2d"], torch.Tensor):
            eagle_config["target_vocab_size"] = int(weights["t2d"].numel())
        elif "target_vocab_size" in eagle_config:
            eagle_config["target_vocab_size"] = int(eagle_config["target_vocab_size"])

        config = self._build_eagle3_speculator_config(
            eagle_config,
            base_model,
            norm_before_residual,
        )

        # Process weights and ensure embeddings are properly handled
        processed_weights = self._process_checkpoint_weights(weights, base_model)

        saved_path = self._save_converted_checkpoint(
            config, processed_weights, output_path
        )
        logger.success(f"Saved to: {saved_path}")

        if validate:
            self._validate_converted_checkpoint(saved_path, base_model)

    def _process_checkpoint_weights(self, weights, base_model,
                                t2d_override=None, d2t_override=None):
        out = {}
        for k, v in weights.items():
            nk = k.replace("midlayer.", "layers.0.")  # keep as-is for others initially
            out[nk] = v

        # --- rename stray unprefixed LN tensors and final norm ---
        if "input_layernorm.weight" in out:
            out["layers.0.input_layernorm.weight"] = out.pop("input_layernorm.weight")
        if "hidden_norm.weight" in out:
            out["layers.0.hidden_norm.weight"] = out.pop("hidden_norm.weight")
        if "lm_head_layernorm.weight" in out:
            out["norm.weight"] = out.pop("lm_head_layernorm.weight")

        # sizes from verifier
        conf_dict, _ = PretrainedConfig.get_config_dict(base_model)
        T = int(conf_dict["vocab_size"])     # teacher vocab
        H = int(conf_dict["hidden_size"])
        N = int(weights.get("draft_vocab_size", 32000))

        # --- embeddings: ensure T x H, pull from verifier if missing/mismatched ---
        E = out.get("embed_tokens.weight")
        if E is None or E.shape[0] != T:
            ver = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)
            E = (ver.model.embed_tokens.weight
                if hasattr(ver, "model") else ver.embed_tokens.weight).data.clone()
            del ver
            out["embed_tokens.weight"] = E
        assert out["embed_tokens.weight"].shape[0] == T

        # --- mappings: MUST be provided/derived; otherwise fail loudly ---
        import numpy as np
        if t2d_override:
            t2d = np.load(t2d_override).astype(bool); assert t2d.size == T
            out["t2d"] = torch.from_numpy(t2d)
        if d2t_override:
            d2t = np.load(d2t_override).astype(np.int64); assert d2t.size == N
            out["d2t"] = torch.from_numpy(d2t)
        if "t2d" not in out and "d2t" not in out:
            raise RuntimeError("No t2d/d2t in checkpoint and no overrides provided.")

        # if only one is present, compute the other
        if "t2d" in out and "d2t" not in out:
            keep = out["t2d"].to(torch.bool).nonzero(as_tuple=False).view(-1).cpu().numpy()
            assert keep.size == N, "t2d True-count != draft vocab"
            base = np.arange(N, dtype=np.int64)
            out["d2t"] = torch.from_numpy(keep - base)
        if "d2t" in out and "t2d" not in out:
            d2t = out["d2t"].cpu().numpy()
            targets = np.arange(N, dtype=np.int64) + d2t
            t2d = np.zeros(T, dtype=bool); t2d[targets] = True
            out["t2d"] = torch.from_numpy(t2d)

        # --- sanity: t2d/d2t agree and are in-bounds ---
        d2t = out["d2t"].cpu().numpy()
        t2d = out["t2d"].cpu().numpy().astype(bool)
        targets = np.arange(N, dtype=np.int64) + d2t
        assert t2d.sum()==N and targets.min()>=0 and targets.max()<T and t2d[targets].all()

        # --- head: ensure N x H and aligned to t2d order ---
        L = out.get("lm_head.weight")
        if L is None:
            raise RuntimeError("lm_head.weight missing")
        if L.shape[0] == T:
            keep_idx = torch.from_numpy(np.nonzero(t2d)[0]).to(L.device)
            out["lm_head.weight"] = L.index_select(0, keep_idx)
        else:
            assert L.shape[0] == N, f"lm_head rows {L.shape[0]} != draft_vocab {N}"

        return out



    def _add_verifier_embeddings(
        self, weights: dict[str, torch.Tensor], base_model: str
    ) -> dict[str, torch.Tensor]:
        """
        Add embeddings from the verifier model to the checkpoint.
        """
        logger.info(f"Loading embeddings from verifier model: {base_model}")

        try:
            verifier = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch.float32
            )

            if hasattr(verifier, "model") and hasattr(verifier.model, "embed_tokens"):
                embed_tokens = verifier.model.embed_tokens.weight.data.clone()
            elif hasattr(verifier, "embed_tokens"):
                embed_tokens = verifier.embed_tokens.weight.data.clone()
            else:
                raise RuntimeError(
                    f"Could not find embed_tokens in verifier model {base_model}"
                )

            logger.info(f"Loaded embeddings with shape: {tuple(embed_tokens.shape)}")
            weights["embed_tokens.weight"] = embed_tokens

            del verifier
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to load embeddings from verifier: {e}")
            raise RuntimeError(
                f"Could not load embeddings from verifier model {base_model}. "
                "This is required for Eagle3 models without trained embeddings."
            ) from e

        return weights

    def _create_verifier_config(self, base_model: str) -> VerifierConfig:
        config_dict, _ = PretrainedConfig.get_config_dict(base_model)
        return VerifierConfig(
            name_or_path=base_model,
            architectures=config_dict.get("architectures", ["LlamaForCausalLM"]),
        )

    def _build_eagle3_speculator_config(
        self,
        eagle_config: dict,
        base_model: str,
        norm_before_residual: bool = False,
    ) -> Eagle3SpeculatorConfig:
        transformer_config = self._create_transformer_config_from_eagle(
            eagle_config, base_model
        )
        verifier_config = self._create_verifier_config(base_model)

        proposal_config = GreedyTokenProposalConfig(
            proposal_type="greedy",
            speculative_tokens=5,
        )

        speculators_config = SpeculatorsConfig(
            algorithm="eagle3",
            proposal_methods=[proposal_config],
            default_proposal_method="greedy",
            verifier=verifier_config,
        )

        return Eagle3SpeculatorConfig(
            transformer_layer_config=transformer_config,
            speculators_config=speculators_config,
            draft_vocab_size=int(eagle_config.get("draft_vocab_size", 32000)),
            norm_before_residual=norm_before_residual,
            target_hidden_size=eagle_config.get("target_hidden_size"),
            # NOTE: we do NOT pass target_vocab_size here; the model’s t2d/d2t
            # will be resized to the checkpoint shapes right before loading.
        )

    def _create_transformer_config_from_eagle(self, eagle_config: dict, base_model: str) -> LlamaConfig:
        conf_dict, _ = PretrainedConfig.get_config_dict(base_model)
        # teacher vocab (T)
        teacher_vocab = int(conf_dict.get("vocab_size"))
        draft_vocab = int(eagle_config.get("draft_vocab_size", 32000))  # keep for downstream

        return LlamaConfig(
            vocab_size=teacher_vocab,              # <-- IMPORTANT: T (e.g., 128256)
            hidden_size=eagle_config.get("hidden_size", conf_dict.get("hidden_size")),
            intermediate_size=eagle_config.get("intermediate_size", conf_dict.get("intermediate_size")),
            num_hidden_layers=1,
            num_attention_heads=eagle_config.get("num_attention_heads", conf_dict.get("num_attention_heads")),
            num_key_value_heads=eagle_config.get("num_key_value_heads", conf_dict.get("num_key_value_heads", conf_dict.get("num_attention_heads"))),
            hidden_act=eagle_config.get("hidden_act", conf_dict.get("hidden_act", "silu")),
            max_position_embeddings=conf_dict.get("max_position_embeddings", 8192),
            initializer_range=conf_dict.get("initializer_range", 0.02),
            rms_norm_eps=conf_dict.get("rms_norm_eps", 1e-5),
            use_cache=True,
            attention_bias=conf_dict.get("attention_bias", False),
            rope_theta=conf_dict.get("rope_theta", 500000.0),
            mlp_bias=conf_dict.get("mlp_bias", False),
            tie_word_embeddings=False,
        )


    # ---- NEW: make model mapping tensors match checkpoint before loading ----
    def _ensure_mapping_shapes_match(
        self, model: torch.nn.Module, ckpt: dict[str, torch.Tensor]
    ) -> None:
        """
        If the model's mapping tensors (t2d/d2t) have different shapes than
        the checkpoint, re-register them so load_state_dict won't fail.
        """
        def _resize(name: str) -> None:
            if name not in ckpt or not isinstance(ckpt[name], torch.Tensor):
                return
            tgt = ckpt[name]
            tgt_shape = tuple(tgt.shape)

            params = dict(model.named_parameters())
            buffs = dict(model.named_buffers())

            if name in params:
                cur = params[name]
                if tuple(cur.shape) != tgt_shape or cur.dtype != tgt.dtype:
                    logger.info(f"Resizing Parameter {name} from {tuple(cur.shape)}/{cur.dtype} to {tgt_shape}/{tgt.dtype}")
                    setattr(model, name, torch.nn.Parameter(tgt.new_empty(tgt_shape), requires_grad=cur.requires_grad))
            elif name in buffs:
                cur = buffs[name]
                if tuple(cur.shape) != tgt_shape or cur.dtype != tgt.dtype:
                    logger.info(f"Resizing buffer {name} from {tuple(cur.shape)}/{cur.dtype} to {tgt_shape}/{tgt.dtype}")
                    # delete then re-register to change shape/dtype
                    delattr(model, name)
                    model.register_buffer(name, tgt.new_empty(tgt_shape))
            else:
                logger.info(f"Registering missing buffer {name} with shape {tgt_shape}/{tgt.dtype}")
                model.register_buffer(name, tgt.new_empty(tgt_shape))

        _resize("t2d")
        _resize("d2t")

        # Helpful diagnostics
        try:
            t2d_m = dict(model.named_parameters()).get("t2d") or dict(model.named_buffers()).get("t2d")
            d2t_m = dict(model.named_parameters()).get("d2t") or dict(model.named_buffers()).get("d2t")
            if t2d_m is not None:
                logger.info(f"Model t2d shape now: {tuple(t2d_m.shape)} | ckpt: {tuple(ckpt['t2d'].shape) if 't2d' in ckpt else 'N/A'}")
            if d2t_m is not None:
                logger.info(f"Model d2t shape now: {tuple(d2t_m.shape)} | ckpt: {tuple(ckpt['d2t'].shape) if 'd2t' in ckpt else 'N/A'}")
        except Exception:
            pass

    def _save_converted_checkpoint(
        self,
        config: Eagle3SpeculatorConfig,
        weights: dict[str, torch.Tensor],
        output_dir: Union[str, Path],
    ) -> Path:
        model = Eagle3Speculator(
            config=config,
            verifier=None,
            verifier_attachment_mode="detached",
        )

        # Ensure mapping tensor shapes on the model match the checkpoint
        self._ensure_mapping_shapes_match(model, weights)

        # Load with strict=False (still errors on shape mismatch, which we avoid above)
        missing_unexp = model.load_state_dict(weights, strict=False)  # type: ignore[attr-defined]
        try:
            missing, unexpected = missing_unexp
        except Exception:
            missing, unexpected = [], []
        if missing:
            logger.warning(f"Missing keys after load: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys after load: {unexpected}")

        model.save_pretrained(str(output_dir))  # type: ignore[attr-defined]
        return Path(output_dir)

    def _validate_converted_checkpoint(
        self, checkpoint_path: Path, base_model: str
    ) -> None:
        logger.info("Validating converted Eagle-3 checkpoint...")
        try:
            Eagle3Speculator.from_pretrained(
                checkpoint_path,
                verifier=base_model,
                verifier_attachment_mode="detached",
            )
            logger.success("Validation succeeded")
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Validation failed: {e}")
            raise
