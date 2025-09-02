#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoConfig

def xavier_(shape, dtype):
    w = torch.empty(*shape, dtype=dtype)
    torch.nn.init.xavier_uniform_(w)
    return w

def kaiming_(shape, dtype):
    w = torch.empty(*shape, dtype=dtype)
    torch.nn.init.kaiming_uniform_(w, a=5**0.5)
    return w

def load_teacher_lm_head_rows(base_dir: str, keep_mask_bool: np.ndarray) -> torch.Tensor:
    # Load teacher lm_head.weight (rows=T, cols=H) and slice to kept rows (N, H).
    import json
    from safetensors import safe_open
    idx_path = Path(base_dir, "model.safetensors.index.json")
    if idx_path.exists():
        idx = json.loads(idx_path.read_text())
        shard = Path(base_dir, idx["weight_map"]["lm_head.weight"])
        with safe_open(str(shard), framework="pt", device="cpu") as sf:
            sl = sf.get_slice("lm_head.weight")
            T, H = sl.get_shape()
            W = sl[:, :H]
    else:
        # Fallback to pytorch shards (rare)
        pt_idx = Path(base_dir, "pytorch_model.bin.index.json")
        if not pt_idx.exists():
            raise FileNotFoundError("Could not find teacher lm_head.* index")
        idx = json.loads(pt_idx.read_text())
        shard = Path(base_dir, idx["weight_map"]["lm_head.weight"])
        W = torch.load(str(shard), map_location="cpu")["lm_head.weight"]

    keep = torch.from_numpy(keep_mask_bool.astype(np.bool_)).nonzero(as_tuple=False).view(-1)
    return W.index_select(0, keep)

def main():
    ap = argparse.ArgumentParser("Build an Eagle-3 drafter head ckpt for training.")
    ap.add_argument("--verifier", required=True, help="Path to teacher/verifier (HF folder).")
    ap.add_argument("--out", required=True, help="Output folder for the drafter head.")
    ap.add_argument("--draft-vocab", type=int, default=32000, help="N (draft vocab).")
    ap.add_argument("--k-fc", type=int, default=3, help="Concat factor for fc input (fc.weight is [H, k_fc*H]).")
    ap.add_argument("--attn-input-mult", type=int, default=2, help="q/k/v take input dim = attn_input_mult * H (default 2).")
    ap.add_argument("--dtype", choices=["bfloat16","float16","float32"], default="bfloat16")
    ap.add_argument("--t2d", type=str, default=None, help="Optional t2d.npy (bool mask length T). If set, slices lm_head.")
    ap.add_argument("--copy-embeddings", action="store_true", help="Also copy teacher embed_tokens.weight [T,H].")
    ap.add_argument("--save-mappings", action="store_true", help="Write mappings.safetensors with t2d (int64) and d2t (offsets).")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = AutoConfig.from_pretrained(args.verifier)
    H = int(getattr(cfg, "hidden_size"))
    I = int(getattr(cfg, "intermediate_size"))
    A = int(getattr(cfg, "num_attention_heads"))
    KV = int(getattr(cfg, "num_key_value_heads", A))
    T = int(getattr(cfg, "vocab_size"))
    head_dim = H // A
    kv_out = KV * head_dim
    N = int(args.draft_vocab)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    # ---- Build state dict ----
    sd = {}

    # FC projection (aux-history -> hidden)
    sd["fc.weight"] = xavier_((H, args.k_fc * H), dtype)

    # LayerNorms (weights only; RMSNorm-compatible if your Model uses it)
    for name in ("layers.0.input_layernorm.weight",
                 "layers.0.post_attention_layernorm.weight",
                 "layers.0.hidden_norm.weight",
                 "lm_head_layernorm.weight",
                 "norm.weight"):
        sd[name] = torch.ones(H, dtype=dtype)

    # Self-Attn (q/k/v from 2H input by default, o: HxH)
    in_attn = args.attn_input_mult * H
    sd["layers.0.self_attn.q_proj.weight"] = xavier_((H, in_attn), dtype)
    sd["layers.0.self_attn.k_proj.weight"] = xavier_((kv_out, in_attn), dtype)
    sd["layers.0.self_attn.v_proj.weight"] = xavier_((kv_out, in_attn), dtype)
    sd["layers.0.self_attn.o_proj.weight"] = xavier_((H, H), dtype)

    # MLP (SwiGLU-compatible shapes from config)
    sd["layers.0.mlp.gate_proj.weight"] = kaiming_((I, H), dtype)
    sd["layers.0.mlp.up_proj.weight"]   = kaiming_((I, H), dtype)
    sd["layers.0.mlp.down_proj.weight"] = kaiming_((H, I), dtype)

    # Embeddings (optional)
    if args.copy_embeddings:
        # copy teacher embeddings: embed_tokens.weight [T,H]
        from safetensors import safe_open
        # find a shard holding embed_tokens.weight
        idx_path = Path(args.verifier, "model.safetensors.index.json")
        if idx_path.exists():
            idx = json.loads(idx_path.read_text())
            # linear scan for the entry holding embed_tokens.weight
            shard = None
            for k, v in idx["weight_map"].items():
                if k == "model.embed_tokens.weight" or k == "embed_tokens.weight":
                    shard = v; break
            if shard is None:
                # try to guess typical shard name
                shard = list(idx["weight_map"].values())[0]
            with safe_open(str(Path(args.verifier, shard)), framework="pt", device="cpu") as sf:
                name = "model.embed_tokens.weight" if "model.embed_tokens.weight" in sf.keys() else "embed_tokens.weight"
                E = sf.get_tensor(name)
        else:
            # pytorch shards fallback
            raise RuntimeError("model.safetensors.index.json not found; embedding copy disabled.")
        sd["embed_tokens.weight"] = E.to(dtype)

    # lm_head (N,H): from teacher via t2d mask if provided, else random
    if args.t2d:
        t2d = np.load(args.t2d)
        if t2d.dtype != np.bool_:
            t2d = t2d.astype(bool)
        if t2d.shape[0] != T:
            raise ValueError(f"t2d length {t2d.shape[0]} != teacher vocab {T}")
        if t2d.sum() != N:
            raise ValueError(f"t2d true count {t2d.sum()} != draft vocab {N}")
        WN = load_teacher_lm_head_rows(args.verifier, t2d).to(dtype)
        sd["lm_head.weight"] = WN.contiguous()
    else:
        sd["lm_head.weight"] = xavier_((N, H), dtype)

    # ---- Save config.json ----
    head_cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "torch_dtype": args.dtype,
        # Keep vocab_size as TEACHER T if you copy embeddings (safer).
        # If you do NOT include embeddings, either value is fine for training.
        "vocab_size": int(T),
        "draft_vocab_size": int(N),

        "hidden_size": int(H),
        "intermediate_size": int(I),
        "num_hidden_layers": 1,
        "num_attention_heads": int(A),
        "num_key_value_heads": int(KV),
        "hidden_act": getattr(cfg, "hidden_act", "silu"),
        "rms_norm_eps": float(getattr(cfg, "rms_norm_eps", 1e-5)),
        "attention_bias": bool(getattr(cfg, "attention_bias", False)),
        "mlp_bias": bool(getattr(cfg, "mlp_bias", False)),
        "max_position_embeddings": int(getattr(cfg, "max_position_embeddings", 8192)),
        "rope_theta": float(getattr(cfg, "rope_theta", 500000.0)),
        "use_cache": True,
        "tie_word_embeddings": False,
        "bos_token_id": getattr(cfg, "bos_token_id", None),
        "eos_token_id": getattr(cfg, "eos_token_id", None),
        "pad_token_id": getattr(cfg, "pad_token_id", None),
    }
    (out / "config.json").write_text(json.dumps(head_cfg, indent=2))

    # ---- Save model.safetensors (single file; no index) ----
    save_file(sd, str(out / "model.safetensors"))

    # ---- Optional mappings.safetensors ----
    if args.save_mappings and args.t2d:
        t2d_mask = np.load(args.t2d).astype(bool)
        keep = np.nonzero(t2d_mask)[0].astype(np.int64)
        base = np.arange(keep.shape[0], dtype=np.int64)
        d2t_offsets = keep - base
        save_file({
            "t2d": torch.from_numpy(t2d_mask.astype(np.int64)),
            "d2t": torch.from_numpy(d2t_offsets),
        }, str(out / "mappings.safetensors"))

    # ---- Summary ----
    print(f"[OK] wrote {out}/model.safetensors")
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:40s} {tuple(v.shape)} {v.dtype}")
    print(f"[cfg] vocab_size(T)={T}  draft_vocab(N)={N}  H={H}  I={I}  heads={A}  kv_heads={KV}")

if __name__ == "__main__":
    main()
