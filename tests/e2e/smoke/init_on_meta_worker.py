"""torchrun worker for the --init-on-meta e2e test (test_init_on_meta.py).

Not a pytest module (not test_*); launched via torch.distributed.run. Mirrors the
trainer's real path (rank0 builds real, non-rank0 under build_on_meta, then
apply_fully_sharded + set_model_state_dict(broadcast_from_rank0=True)) and checks that
--init-on-meta changes nothing:
  1. non-rank0 params are on meta before broadcast (the memory win);
  2. every rank's shard is real + finite after broadcast;
  3. requires_grad matches across ranks (else FSDP2 grad-reduce hangs);
  4. gathered full weights equal rank0's originals.
Prints PASS / exits 0 on success. Usage: torchrun --nproc_per_node 2 <this> [tmp_dir].
"""

import contextlib
import os
import sys
import tempfile
import time

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from transformers import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig

from speculators.models.eagle3 import Eagle3DraftModel
from speculators.train.distributed import (
    apply_fully_sharded,
    build_on_meta,
    get_rank,
    get_world_size,
    maybe_destroy_distributed,
    maybe_setup_distributed,
)


def _tiny_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )


def _build_draft(verifier_dir: str) -> Eagle3DraftModel:
    # == scripts/train.py: build under build_on_meta on non-rank0 only.
    meta_ctx = build_on_meta() if get_rank() != 0 else contextlib.nullcontext()
    with meta_ctx:
        return Eagle3DraftModel.from_training_args(
            verifier_config=_tiny_config(),
            t2d=None,
            d2t=None,
            draft_vocab_size=64,
            norm_before_residual=False,
            ttt_steps=1,
            draft_attn_impl="eager",
            target_layer_ids=[0, 1],
            verifier_name_or_path=verifier_dir,
        )


def main() -> None:
    import numpy as np
    from torch.distributed.tensor import DTensor

    t0 = time.perf_counter()
    maybe_setup_distributed()
    world = get_world_size()
    if world < 2:
        raise SystemExit("run with torchrun --nproc_per_node >=2 (need >1 rank)")
    rank = get_rank()

    def tick(msg: str) -> None:
        if rank == 0:
            print(f"[+{time.perf_counter() - t0:5.1f}s] {msg}", flush=True)

    tick("distributed initialized")

    # rank0 writes a tiny verifier WITH weights so rank0's load_verifier_weights
    # succeeds; all ranks read it (torchrun ranks share the node filesystem). The
    # shared dir is passed in by the test (its tmp_path) so runs stay hermetic.
    shared_root = sys.argv[1] if len(sys.argv) > 1 else tempfile.gettempdir()
    verifier_dir = os.path.join(shared_root, "init_on_meta_tiny_verifier")
    if rank == 0:
        LlamaForCausalLM(_tiny_config()).save_pretrained(verifier_dir)
    dist.barrier()

    model = _build_draft(verifier_dir)
    tick("draft model built (rank0 real, non-rank0 on meta)")

    # rank0 is the single source of truth: ensure it has no nan params (safety net
    # that also decouples this check from verifier-loading details).
    if rank == 0:
        with torch.no_grad():
            for p in model.parameters():
                if p.is_meta:
                    raise AssertionError("rank0 unexpectedly built on meta")
                nan = torch.isnan(p)
                if nan.any():
                    p[nan] = torch.randn_like(p[nan]) * 0.02

    # (1) the optimization engaged: non-rank0 built on meta, rank0 did not.
    embed_is_meta = model.embed_tokens.weight.is_meta
    if rank == 0:
        assert not embed_is_meta, "rank0 should hold real weights"
    else:
        assert embed_is_meta, "non-rank0 must build on meta (--init-on-meta)"

    # Two snapshots: `bcast_src` is the broadcast source (set_model_state_dict mutates
    # its input dict into DTensors in place); `ref_np` is a frozen numpy copy for the
    # (3) compare, immune to that mutation.
    ref_np = (
        {
            k: v.detach().cpu().float().numpy().copy()
            for k, v in model.state_dict().items()
        }
        if rank == 0
        else {}
    )
    bcast_src = model.state_dict() if rank == 0 else {}

    # == trainer.py distributed setup (shard, then broadcast-materialize from rank0).
    apply_fully_sharded(model)
    set_model_state_dict(
        model,
        bcast_src,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            strict=False,
        ),
    )
    dist.barrier()
    tick("apply_fully_sharded + broadcast-materialize done")

    # (2) every rank's local shard is real and finite after the broadcast.
    for name, p in model.named_parameters():
        local = p.to_local() if isinstance(p, DTensor) else p
        assert not local.is_meta, f"[rank{rank}] {name} still on meta after broadcast"
        assert not torch.isnan(local.float()).any(), f"[rank{rank}] {name} has NaN"

    # (2b) requires_grad must match across ranks, or FSDP2 post_backward collects a
    #      different trainable-param set per rank and the first backward hangs.
    #      named_parameters() is identically ordered on every rank -> flags line up.
    flags = torch.tensor(
        [1 if p.requires_grad else 0 for _, p in model.named_parameters()],
        dtype=torch.int32,
        device="cuda",
    )
    gathered_flags = [torch.zeros_like(flags) for _ in range(world)]
    dist.all_gather(gathered_flags, flags)
    for other, gf in enumerate(gathered_flags):
        if not torch.equal(gf, flags):
            mism = [
                name
                for i, (name, _) in enumerate(model.named_parameters())
                if gf[i] != flags[i]
            ]
            raise AssertionError(
                f"[rank{rank}] requires_grad differs from rank{other} on {mism} "
                "-> FSDP2 gradient-reduction mismatch (training would hang)"
            )
    tick("requires_grad is consistent across ranks")

    # (3) full weights gathered from every rank's shard == rank0's originals.
    #     full_tensor() is a collective -> ALL ranks gather first, then rank0-only
    #     compares numpy (no distributed op inside `if rank == 0`, which would hang).
    def _gather_np(t: torch.Tensor):
        if isinstance(t, DTensor):  # sharded -> all-gather to a plain tensor
            t = t.full_tensor()
        return t.detach().cpu().float().numpy()

    gathered = {name: _gather_np(p) for name, p in model.named_parameters()}
    dist.barrier()
    tick("full weights gathered")

    if rank == 0:
        bad = [
            name
            for name, got in gathered.items()
            if not np.allclose(got, ref_np[name], rtol=1e-2, atol=1e-3)
        ]
        assert not bad, f"materialized weights differ from rank0 reference: {bad}"

    dist.barrier()
    if rank == 0:
        print(
            f"PASS ({world} ranks): non-rank0 built on meta, all ranks materialized "
            "real+finite weights, and the gathered state matches rank0 exactly.",
            flush=True,
        )
    maybe_destroy_distributed()


if __name__ == "__main__":
    main()
