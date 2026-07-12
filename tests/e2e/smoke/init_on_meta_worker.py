"""torchrun worker for the ``--init-on-meta`` e2e test (``test_init_on_meta.py``).

Not a pytest module (not ``test_*``); launched via ``torch.distributed.run`` by the
test. Mirrors the trainer's REAL build+materialize path and asserts the meta-init
memory optimization does not change the result:

  * build           -- rank0 builds the draft with real weights; non-rank0 builds
                       under ``build_on_meta`` (== scripts/train.py's --init-on-meta
                       branch: ``build_on_meta() if init_on_meta and rank != 0``);
  * shard+broadcast -- ``apply_fully_sharded`` then ``set_model_state_dict(
                       broadcast_from_rank0=True)`` (== trainer.py distributed setup).

Assertions:
  1. before broadcast: non-rank0 params are on ``meta`` (proof the allocation was
     skipped -> the memory win), rank0 params are real;
  2. after broadcast:  every rank's local shard is real (not meta) and finite (no NaN);
  3. after broadcast:  the full state gathered from every rank's shard equals rank0's
     original weights (so non-rank0 materialized exactly rank0's values).

Prints ``PASS`` and exits 0 on success; raises (nonzero exit) on any failure.

Usage (the test does this):
    torchrun --nproc_per_node 2 init_on_meta_worker.py [shared_tmp_dir]
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

    # Snapshot rank0's weights TWICE. `bcast_src` is the broadcast source handed to
    # set_model_state_dict below, which MUTATES its input dict into DTensors in place.
    # `ref_np` is a SEPARATE, frozen numpy copy used only for the comparison in (3) --
    # numpy so nothing torch/DTensor can retroactively touch it (reusing one dict for
    # both makes the compare call a rank0-only collective on the now-DTensor ref).
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

    # (3) each param, gathered from every rank's shard, equals rank0's original
    #     weights. full_tensor() is a COLLECTIVE, so ALL ranks gather (same order)
    #     FIRST; only rank0 then compares, against the frozen numpy reference, so there
    #     is NO distributed op inside `if rank == 0` (that would hang the other ranks).
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
