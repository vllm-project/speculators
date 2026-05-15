"""
Tests for model weight loading and initialization pathways.

Covers:
- Trainer.setup_model for single-GPU (fresh + resume)
- SingleGPUCheckpointer save/load round-trip
- from_pretrained save/load round-trip
- Weight precedence: checkpoint > pretrained > verifier > random init
- Distributed fresh init (FSDP + broadcast, mp.spawn)
- Distributed resume from checkpoint (mp.spawn)
- Distributed from_pretrained (mp.spawn)
"""

import copy
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors import safe_open
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from transformers.models.llama.configuration_llama import LlamaConfig

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.train.checkpointer import (
    DistributedCheckpointer,
    SingleGPUCheckpointer,
)
from speculators.train.trainer import Trainer, TrainerConfig
from tests.conftest import requires_cuda, requires_multi_gpu

# ---------------------------------------------------------------------------
# Tiny model constants
# ---------------------------------------------------------------------------

TINY_LLAMA_CONFIG = LlamaConfig(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=8,
    max_position_embeddings=32,
    rms_norm_eps=1e-6,  # type: ignore[arg-type] # (bad transformer's type hint, int instead of float)
    tie_word_embeddings=False,
    _attn_implementation="eager",  # type: ignore[call-arg]
)


# ---------------------------------------------------------------------------
# Helpers (used by both fixtures and mp.spawn workers)
# ---------------------------------------------------------------------------


def _make_eagle3_config(
    draft_vocab_size: int = 64,
    verifier_name_or_path: str | None = None,
) -> Eagle3SpeculatorConfig:
    return Eagle3SpeculatorConfig(
        transformer_layer_config=copy.deepcopy(TINY_LLAMA_CONFIG),
        draft_vocab_size=draft_vocab_size,
        norm_before_residual=False,
        embed_requires_grad=False,
        speculators_config=SpeculatorsConfig(
            algorithm="eagle3",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=1)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=verifier_name_or_path,
                architectures=["LlamaForCausalLM"],
            ),
        ),
    )


def _make_vocab_mappings(
    verifier_vocab_size: int = 64,
    draft_vocab_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create valid t2d and d2t tensors for testing.

    Selects the first `draft_vocab_size` tokens from the verifier vocab.
    t2d: bool[verifier_vocab_size] — True for tokens included in draft vocab.
    d2t: long[draft_vocab_size] — maps draft index to verifier index.
    """
    t2d = torch.zeros(verifier_vocab_size, dtype=torch.bool)
    t2d[:draft_vocab_size] = True
    d2t = torch.arange(draft_vocab_size, dtype=torch.long)
    return t2d, d2t


def _make_tiny_model() -> Eagle3DraftModel:
    """Create a tiny Eagle3 model with NaN weights filled."""
    model = Eagle3DraftModel(_make_eagle3_config())
    _fill_nan_weights(model)
    return model


def _fill_nan_weights(model: Eagle3DraftModel):
    """Replace NaN-initialized weights with deterministic values (simulates
    what load_verifier_weights does)."""
    with torch.no_grad():
        torch.nn.init.ones_(model.embed_tokens.weight)
        torch.nn.init.ones_(model.lm_head.weight)
        torch.nn.init.ones_(model.verifier_lm_head.weight)
        torch.nn.init.ones_(model.verifier_norm.weight)


def _make_trainer_no_init(
    model,
    *,
    is_distributed=False,
    resume_from_checkpoint=False,
    rank=0,
    local_rank=0,
    save_path="/tmp/test_ckpt",
    hidden_states_dtype=torch.bfloat16,
):
    """Create a Trainer instance bypassing __init__ to control setup order."""
    config = TrainerConfig(
        lr=1e-4,
        num_epochs=1,
        save_path=save_path,
        resume_from_checkpoint=resume_from_checkpoint,
        is_distributed=is_distributed,
        rank=rank,
        local_rank=local_rank,
        hidden_states_dtype=hidden_states_dtype,
    )
    trainer = Trainer.__new__(Trainer)
    trainer.model = model
    trainer.config = config
    trainer.rank = config.rank
    trainer.local_rank = config.local_rank
    trainer.is_distributed = config.is_distributed
    trainer.resume_from_checkpoint = config.resume_from_checkpoint
    trainer.train_loader = MagicMock(__len__=MagicMock(return_value=1))
    trainer.val_loader = None
    return trainer


def test_trainer_uses_rank_from_config(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setattr(Trainer, "setup_trainer", lambda self: None)
    monkeypatch.setattr(Trainer, "setup_model", lambda self: None)
    monkeypatch.setattr(Trainer, "setup_optimizer", lambda self: None)

    trainer = Trainer(
        MagicMock(),
        TrainerConfig(
            lr=1e-4,
            num_epochs=1,
            save_path=str(tmp_path),
            rank=3,
            local_rank=1,
        ),
        MagicMock(),
    )

    assert trainer.rank == 3
    assert trainer.local_rank == 1


def _param_checksums(state_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    """Compute per-key checksums for cross-rank comparison."""
    return {
        k: v.float().sum().item()
        for k, v in state_dict.items()
        if isinstance(v, torch.Tensor)
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def eagle3_config():
    return _make_eagle3_config()


@pytest.fixture
def tiny_model():
    """Tiny Eagle3 model on CPU with NaN weights filled."""
    return _make_tiny_model()


@pytest.fixture
def tiny_model_on_gpu(tiny_model):
    """Tiny Eagle3 model moved to cuda:0."""
    return tiny_model.to("cuda:0")


@pytest.fixture
def checkpoint_dir(tmp_path, tiny_model_on_gpu):
    """Save a checkpoint with trainable weights = 42.0, return the path."""
    with torch.no_grad():
        for p in tiny_model_on_gpu.parameters():
            if p.requires_grad:
                p.fill_(42.0)

    ckpt_dir = tmp_path / "ckpt"
    checkpointer = SingleGPUCheckpointer(ckpt_dir)
    optimizer = torch.optim.AdamW(tiny_model_on_gpu.parameters(), lr=1e-4)
    checkpointer.save_checkpoint(tiny_model_on_gpu, optimizer, epoch=0)
    return ckpt_dir


@pytest.fixture
def pretrained_dir(tmp_path, tiny_model):
    """Save a pretrained model with fc=66.0, lm_head=55.0, return the path."""
    with torch.no_grad():
        tiny_model.fc.weight.fill_(66.0)
        tiny_model.lm_head.weight.fill_(55.0)
    model_dir = tmp_path / "pretrained"
    tiny_model.save_pretrained(str(model_dir))
    return model_dir


@pytest.fixture
def mock_checkpointer():
    """Mock checkpointer with no previous checkpoint."""
    ckpt = MagicMock()
    ckpt.previous_epoch = -1
    return ckpt


# ===================================================================
# Single GPU — Fresh Init
# ===================================================================


@requires_cuda
def test_single_gpu_fresh_init(tiny_model, mock_checkpointer):
    """Fresh single-GPU setup: model moved to device, weights unchanged,
    no checkpoint loading."""
    state_before = {k: v.clone() for k, v in tiny_model.state_dict().items()}

    trainer = _make_trainer_no_init(
        tiny_model, is_distributed=False, hidden_states_dtype=torch.float
    )
    trainer.checkpointer = mock_checkpointer

    trainer.setup_model()

    # Weights should be unchanged (just moved to device)
    for k, v in tiny_model.state_dict().items():
        assert torch.allclose(v.cpu().float(), state_before[k].float()), (
            f"Weight {k} changed during fresh init"
        )

    # No checkpoint loading
    mock_checkpointer.load_model_state_dict.assert_not_called()


# ===================================================================
# Single GPU — Resume from Checkpoint
# ===================================================================


@requires_cuda
def test_single_gpu_resume(checkpoint_dir):
    """Resume from checkpoint: checkpoint weights loaded, verifier weights
    preserved (not overwritten by checkpoint since they're not saved)."""
    model = _make_tiny_model()
    with torch.no_grad():
        model.verifier_norm.weight.fill_(77.0)
        model.verifier_lm_head.weight.fill_(88.0)

    trainer = _make_trainer_no_init(
        model,
        is_distributed=False,
        resume_from_checkpoint=True,
        save_path=str(checkpoint_dir),
    )
    trainer.checkpointer = SingleGPUCheckpointer(checkpoint_dir)
    trainer.setup_model()

    # Trainable weights should match checkpoint (42.0, modulo bf16 round-trip)
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert torch.allclose(param.cpu().float(), torch.tensor(42.0), atol=0.5), (
                f"Trainable weight {name} not loaded from checkpoint"
            )

    # Verifier weights should be preserved (not in checkpoint)
    assert torch.allclose(
        model.verifier_norm.weight.cpu().float(), torch.tensor(77.0)
    ), "verifier_norm overwritten by checkpoint"
    assert torch.allclose(
        model.verifier_lm_head.weight.cpu().float(), torch.tensor(88.0)
    ), "verifier_lm_head overwritten by checkpoint"


# ===================================================================
# Checkpoint Save/Load Round-Trip
# ===================================================================


@requires_cuda
def test_checkpoint_save_load_round_trip(checkpoint_dir):
    """SingleGPUCheckpointer round-trip: trainable weights preserved, verifier
    keys not saved, expected files created."""
    # Verify files
    assert (checkpoint_dir / "0" / "model.safetensors").exists()
    assert (checkpoint_dir / "0" / "config.json").exists()
    assert (checkpoint_dir / "0" / "optimizer_state_dict.pt").exists()

    # Verify verifier-only keys not in saved safetensors
    with safe_open(
        str(checkpoint_dir / "0" / "model.safetensors"), framework="pt"
    ) as f:
        saved_keys = set(f.keys())
    for key in Eagle3DraftModel._keys_to_ignore_on_save:
        assert key not in saved_keys, f"{key} should not be saved"

    # Load into fresh model and verify trainable weights match
    model = _make_tiny_model()
    model.to("cuda:0")  # type: ignore[arg-type]
    checkpointer = SingleGPUCheckpointer(checkpoint_dir)
    checkpointer.load_model_state_dict(model)

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert torch.allclose(param.cpu().float(), torch.tensor(42.0), atol=0.5), (
                f"Trainable weight {name} not preserved in round-trip"
            )


# ===================================================================
# from_pretrained Round-Trip
# ===================================================================


def test_from_pretrained_round_trip(tiny_model):
    """from_pretrained round-trip: trainable weights preserved, ignored keys
    not in saved files, pretrained weights take precedence over verifier."""
    # Set trainable weights to known value
    with torch.no_grad():
        for p in tiny_model.parameters():
            if p.requires_grad:
                p.fill_(42.0)

    trainable_names = {n for n, p in tiny_model.named_parameters() if p.requires_grad}
    original_trainable = {
        k: v.clone() for k, v in tiny_model.state_dict().items() if k in trainable_names
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tiny_model.save_pretrained(tmpdir)

        # Verify _keys_to_ignore_on_save not in saved files
        with safe_open(f"{tmpdir}/model.safetensors", framework="pt") as f:
            saved_keys = set(f.keys())
        for key in Eagle3DraftModel._keys_to_ignore_on_save:
            assert key not in saved_keys, f"{key} should not be saved"

        # Load (mock load_verifier_weights to avoid HF downloads)
        with patch.object(Eagle3DraftModel, "load_verifier_weights"):
            loaded = Eagle3DraftModel.from_pretrained(tmpdir)

        # Trainable weights should match original
        for k, original_v in original_trainable.items():
            loaded_v = loaded.state_dict()[k]
            assert torch.allclose(loaded_v.float(), original_v.float(), atol=0.5), (
                f"Weight {k} not preserved in from_pretrained round-trip"
            )

        # lm_head was saved (it's trainable), so from_pretrained loads it.
        # Even if load_verifier_weights ran, the NaN guard would keep the
        # pretrained value since it's no longer NaN.
        assert not loaded.lm_head.weight.isnan().any(), (
            "lm_head should have pretrained value, not NaN"
        )


# ===================================================================
# Weight Precedence
# ===================================================================


@requires_cuda
def test_weight_precedence(eagle3_config, pretrained_dir, tmp_path):
    """Verify weight precedence: checkpoint > pretrained > verifier > random.

    Walks through the full chain in a single test."""

    # --- Level 5: Random init produces NaN for verifier-loaded weights ---
    model = Eagle3DraftModel(eagle3_config)
    assert model.embed_tokens.weight.isnan().all(), (
        "embed_tokens should be NaN after random init"
    )
    assert model.lm_head.weight.isnan().all(), "lm_head should be NaN after random init"
    # fc (trainable) should NOT be NaN — it's randomly initialized
    assert not model.fc.weight.isnan().any(), "fc should have random init, not NaN"

    # --- Level 4: Verifier fills NaN weights ---
    _fill_nan_weights(model)  # simulates load_verifier_weights
    assert not model.embed_tokens.weight.isnan().any(), (
        "embed_tokens should be filled by verifier"
    )
    assert not model.lm_head.weight.isnan().any(), (
        "lm_head should be filled by verifier"
    )

    # --- Level 2: Pretrained weights take precedence over verifier ---
    # pretrained_dir fixture saved lm_head=55.0, fc=66.0
    with patch.object(Eagle3DraftModel, "load_verifier_weights"):
        loaded = Eagle3DraftModel.from_pretrained(str(pretrained_dir))

    assert torch.allclose(loaded.lm_head.weight.float(), torch.tensor(55.0)), (
        "pretrained lm_head should not be overwritten by verifier"
    )
    assert torch.allclose(loaded.fc.weight.float(), torch.tensor(66.0)), (  # type: ignore[union-attr,arg-type]
        "pretrained fc should be preserved"
    )

    # --- Level 1: Checkpoint overrides everything ---
    loaded.to("cuda:0")  # type: ignore[arg-type]
    with torch.no_grad():
        loaded.fc.weight.fill_(99.0)  # type: ignore[union-attr,operator]  # checkpoint value
    ckpt_dir = str(tmp_path / "ckpt")
    checkpointer = SingleGPUCheckpointer(ckpt_dir)
    optimizer = torch.optim.AdamW(loaded.parameters(), lr=1e-4)
    checkpointer.save_checkpoint(loaded, optimizer, epoch=0)

    # Load checkpoint into a model that had pretrained value (66.0)
    with patch.object(Eagle3DraftModel, "load_verifier_weights"):
        model3 = Eagle3DraftModel.from_pretrained(str(pretrained_dir))
    model3.to("cuda:0")  # type: ignore[arg-type]
    checkpointer2 = SingleGPUCheckpointer(ckpt_dir)
    checkpointer2.load_model_state_dict(model3)

    assert torch.allclose(
        model3.fc.weight.cpu().float(),  # type: ignore[union-attr,arg-type]
        torch.tensor(99.0),
        atol=0.5,
    ), "checkpoint fc should override pretrained"


# ===================================================================
# Distributed helpers
# ===================================================================


def _dist_setup(rank, world_size):
    """Initialize distributed process group for testing."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _dist_teardown():
    """Clean up distributed process group."""
    dist.destroy_process_group()


def _get_full_state_dict_rank0(model):
    """Get unsharded full state dict from FSDP model (only populated on rank 0).

    All ranks must call this (it's a collective op), but only rank 0
    gets the actual tensors."""
    return get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )


# ===================================================================
# Distributed — Fresh Init
# ===================================================================


def _worker_distributed_fresh_init(rank, world_size, results_dir):
    """Worker for test_distributed_fresh_init."""
    _dist_setup(rank, world_size)
    try:
        model = _make_tiny_model()

        # Capture rank 0's pre-FSDP state dict for comparison
        pre_fsdp_checksums = _param_checksums(model.state_dict()) if rank == 0 else {}

        trainer = _make_trainer_no_init(
            model, is_distributed=True, rank=rank, local_rank=rank
        )
        trainer.checkpointer = MagicMock()
        trainer.checkpointer.previous_epoch = -1

        trainer.setup_model()

        # All ranks must call get_model_state_dict (collective op),
        # but only rank 0 gets the actual tensors
        full_sd = _get_full_state_dict_rank0(model)

        if rank == 0:
            checksums = _param_checksums(full_sd)
            has_nan = {
                k: v.isnan().any().item()
                for k, v in full_sd.items()
                if isinstance(v, torch.Tensor) and v.is_floating_point()
            }

            torch.save(
                {
                    "pre_fsdp_checksums": pre_fsdp_checksums,
                    "post_fsdp_checksums": checksums,
                    "has_nan": has_nan,
                },
                results_dir / "results.pt",
            )
    finally:
        _dist_teardown()


@requires_multi_gpu
def test_distributed_fresh_init(tmp_path):
    """Distributed fresh init: after setup_model, the gathered full state dict
    matches rank 0's original pre-FSDP weights and contains no NaN values.

    This verifies that set_model_state_dict(broadcast_from_rank0=True)
    correctly distributes rank 0's weights to all ranks, because
    get_model_state_dict gathers shards from ALL ranks to reconstruct
    the full dict on rank 0."""
    world_size = 2
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    mp.spawn(
        _worker_distributed_fresh_init,
        args=(world_size, results_dir),
        nprocs=world_size,
        join=True,
    )

    results = torch.load(results_dir / "results.pt", weights_only=False)

    # Post-FSDP gathered state dict should match pre-FSDP state dict from rank 0
    pre = results["pre_fsdp_checksums"]
    post = results["post_fsdp_checksums"]
    for key in pre:
        assert key in post, f"Key {key} missing after FSDP round-trip"
        assert pre[key] == pytest.approx(post[key], abs=1e-2), (
            f"Weight {key} changed during FSDP broadcast: "
            f"pre={pre[key]}, post={post[key]}"
        )

    # No NaN values in any float parameter
    for key, is_nan in results["has_nan"].items():
        assert not is_nan, f"Weight {key} is NaN after distributed setup"


# ===================================================================
# Distributed — Resume from Checkpoint
# ===================================================================


def _worker_distributed_resume(rank, world_size, ckpt_dir, results_dir):
    """Worker for test_distributed_resume."""
    _dist_setup(rank, world_size)
    try:
        model = _make_tiny_model()
        # Set verifier weights to known value before FSDP
        with torch.no_grad():
            model.verifier_norm.weight.fill_(77.0)
            model.verifier_lm_head.weight.fill_(88.0)

        trainer = _make_trainer_no_init(
            model,
            is_distributed=True,
            resume_from_checkpoint=True,
            rank=rank,
            local_rank=rank,
            save_path=ckpt_dir,
        )
        trainer.checkpointer = DistributedCheckpointer(ckpt_dir)
        trainer.setup_model()

        # All ranks must call (collective op), only rank 0 gets data
        full_sd = _get_full_state_dict_rank0(model)

        if rank == 0:
            checksums = _param_checksums(full_sd)
            verifier_norm_val = full_sd["verifier_norm.weight"].float().mean().item()
            verifier_lm_head_val = (
                full_sd["verifier_lm_head.weight"].float().mean().item()
            )

            torch.save(
                {
                    "checksums": checksums,
                    "verifier_norm_val": verifier_norm_val,
                    "verifier_lm_head_val": verifier_lm_head_val,
                },
                results_dir / "results.pt",
            )
    finally:
        _dist_teardown()


@requires_multi_gpu
def test_distributed_resume(checkpoint_dir, tmp_path):
    """Distributed resume: checkpoint weights loaded correctly, verifier
    weights preserved (not overwritten by checkpoint)."""
    world_size = min(torch.cuda.device_count(), 2)
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    mp.spawn(
        _worker_distributed_resume,
        args=(world_size, str(checkpoint_dir), results_dir),
        nprocs=world_size,
        join=True,
    )

    results = torch.load(results_dir / "results.pt", weights_only=False)

    # Verifier weights should be preserved (not in checkpoint)
    assert results["verifier_norm_val"] == pytest.approx(77.0, abs=0.1), (
        "verifier_norm overwritten by checkpoint"
    )
    assert results["verifier_lm_head_val"] == pytest.approx(88.0, abs=0.1), (
        "verifier_lm_head overwritten by checkpoint"
    )


# ===================================================================
# Distributed — from_pretrained
# ===================================================================


def _worker_distributed_from_pretrained(rank, world_size, model_dir, results_dir):
    """Worker for test_distributed_from_pretrained."""
    _dist_setup(rank, world_size)
    try:
        # Load model from pretrained (mock verifier loading)
        with patch.object(Eagle3DraftModel, "load_verifier_weights"):
            model = Eagle3DraftModel.from_pretrained(model_dir)
        _fill_nan_weights(model)  # type: ignore[arg-type]  # fill verifier weights post-load

        trainer = _make_trainer_no_init(
            model, is_distributed=True, rank=rank, local_rank=rank
        )
        trainer.checkpointer = MagicMock()
        trainer.checkpointer.previous_epoch = -1

        trainer.setup_model()

        # All ranks must call (collective op), only rank 0 gets data
        full_sd = _get_full_state_dict_rank0(model)

        if rank == 0:
            checksums = _param_checksums(full_sd)
            fc_val = full_sd["fc.weight"].float().mean().item()

            torch.save(
                {"checksums": checksums, "fc_val": fc_val},
                results_dir / "results.pt",
            )
    finally:
        _dist_teardown()


@requires_multi_gpu
def test_distributed_from_pretrained(pretrained_dir, tmp_path):
    """Model loaded via from_pretrained should have correct weights after FSDP
    setup, with pretrained weight values preserved through the broadcast."""
    world_size = min(torch.cuda.device_count(), 2)
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    mp.spawn(
        _worker_distributed_from_pretrained,
        args=(world_size, str(pretrained_dir), results_dir),
        nprocs=world_size,
        join=True,
    )

    results = torch.load(results_dir / "results.pt", weights_only=False)

    # Pretrained fc weight should be preserved through FSDP setup
    assert results["fc_val"] == pytest.approx(66.0, abs=0.5), (
        "Pretrained fc weight not preserved through FSDP broadcast"
    )


# ===================================================================
# Vocab Mapping Loading (t2d / d2t)
# ===================================================================

DRAFT_VOCAB_SIZE = 32  # < TINY_LLAMA_CONFIG.vocab_size (64)


@pytest.fixture
def draft_vocab_config():
    """Eagle3 config with draft_vocab_size < verifier_vocab_size."""
    return _make_eagle3_config(draft_vocab_size=DRAFT_VOCAB_SIZE)


@pytest.fixture
def vocab_mappings():
    """Valid (t2d, d2t) pair for verifier_vocab=64, draft_vocab=32."""
    assert TINY_LLAMA_CONFIG.vocab_size is not None  # typing
    return _make_vocab_mappings(
        verifier_vocab_size=TINY_LLAMA_CONFIG.vocab_size,
        draft_vocab_size=DRAFT_VOCAB_SIZE,
    )


def test_load_vocab_mappings(draft_vocab_config, vocab_mappings):
    """load_vocab_mappings stores t2d/d2t buffers correctly."""
    t2d, d2t = vocab_mappings
    model = Eagle3DraftModel(draft_vocab_config)

    # Before loading: buffers exist but are zeros
    assert model.t2d is not None
    assert not model.t2d.any(), "t2d should be all zeros before loading"
    assert model.d2t is not None
    assert (model.d2t == 0).all(), "d2t should be all zeros before loading"

    model.load_vocab_mappings(t2d, d2t)

    # After loading: buffers match inputs
    assert torch.equal(model.t2d, t2d), "t2d not loaded correctly"
    assert torch.equal(model.d2t, d2t), "d2t not loaded correctly"


def test_load_vocab_mappings_validation(draft_vocab_config, vocab_mappings):
    """load_vocab_mappings raises on invalid inputs."""
    t2d, d2t = vocab_mappings
    model = Eagle3DraftModel(draft_vocab_config)

    # Only one of t2d/d2t provided
    with pytest.raises(ValueError, match="Both t2d and d2t must be provided"):
        model.load_vocab_mappings(t2d, None)
    with pytest.raises(ValueError, match="Both t2d and d2t must be provided"):
        model.load_vocab_mappings(None, d2t)

    # Wrong t2d shape
    with pytest.raises(ValueError, match="t2d.shape"):
        model.load_vocab_mappings(torch.ones(10, dtype=torch.bool), d2t)

    # Wrong d2t shape
    with pytest.raises(ValueError, match="d2t.shape"):
        model.load_vocab_mappings(t2d, torch.zeros(10, dtype=torch.long))

    # Wrong number of True values in t2d
    assert TINY_LLAMA_CONFIG.vocab_size is not None  # typing
    bad_t2d = torch.ones(TINY_LLAMA_CONFIG.vocab_size, dtype=torch.bool)
    with pytest.raises(ValueError, match="non-zero values"):
        model.load_vocab_mappings(bad_t2d, d2t)


def test_load_vocab_mappings_not_needed():
    """load_vocab_mappings raises when vocab sizes match (no mapping needed)."""
    config = _make_eagle3_config(draft_vocab_size=64)  # same as verifier
    model = Eagle3DraftModel(config)
    t2d, d2t = _make_vocab_mappings(verifier_vocab_size=64, draft_vocab_size=64)

    with pytest.raises(RuntimeError, match="not needed"):
        model.load_vocab_mappings(t2d, d2t)


def test_from_training_args_loads_vocab_mappings(vocab_mappings):
    """from_training_args passes t2d/d2t through to load_vocab_mappings."""
    t2d, d2t = vocab_mappings

    with patch.object(Eagle3DraftModel, "load_verifier_weights"):
        model = Eagle3DraftModel.from_training_args(
            verifier_config=copy.deepcopy(TINY_LLAMA_CONFIG),
            t2d=t2d,
            d2t=d2t,
            draft_vocab_size=DRAFT_VOCAB_SIZE,
            norm_before_residual=False,
            ttt_steps=1,
            verifier_name_or_path="nm-testing/tinysmokellama-3.2",
        )

    assert model.t2d is not None, "t2d is None after from_training_args"
    assert model.d2t is not None, "d2t is None after from_training_args"
    assert torch.equal(model.t2d, t2d), "t2d not loaded via from_training_args"
    assert torch.equal(model.d2t, d2t), "d2t not loaded via from_training_args"


def test_from_pretrained_loads_vocab_mappings_from_kwargs(
    tmp_path, draft_vocab_config, vocab_mappings
):
    """from_pretrained loads t2d/d2t passed as kwargs."""
    t2d, d2t = vocab_mappings

    # Save a model without vocab mappings in the safetensors
    model = Eagle3DraftModel(draft_vocab_config)
    _fill_nan_weights(model)
    model_dir = tmp_path / "pretrained_no_vocab"
    model.save_pretrained(str(model_dir))

    # Load with t2d/d2t passed as kwargs
    with patch.object(Eagle3DraftModel, "load_verifier_weights"):
        loaded = Eagle3DraftModel.from_pretrained(str(model_dir), t2d=t2d, d2t=d2t)

    assert loaded.t2d is not None, "t2d is None after from_pretrained"
    assert loaded.d2t is not None, "d2t is None after from_pretrained"
    assert torch.equal(loaded.t2d, t2d), "t2d not loaded from kwargs in from_pretrained"  # type: ignore[arg-type]
    assert torch.equal(loaded.d2t, d2t), "d2t not loaded from kwargs in from_pretrained"  # type: ignore[arg-type]


def test_from_pretrained_loads_vocab_mappings_from_saved(
    tmp_path, draft_vocab_config, vocab_mappings
):
    """from_pretrained loads t2d/d2t from saved safetensors when not passed
    as kwargs."""
    t2d, d2t = vocab_mappings

    # Save model WITH vocab mappings loaded
    model = Eagle3DraftModel(draft_vocab_config)
    _fill_nan_weights(model)
    model.load_vocab_mappings(t2d, d2t)
    model_dir = tmp_path / "pretrained_with_vocab"
    model.save_pretrained(str(model_dir))

    # Verify t2d/d2t are in the saved safetensors
    with safe_open(str(model_dir / "model.safetensors"), framework="pt") as f:
        saved_keys = set(f.keys())
    assert "t2d" in saved_keys, "t2d should be saved in safetensors"
    assert "d2t" in saved_keys, "d2t should be saved in safetensors"

    # Load WITHOUT passing t2d/d2t — should come from safetensors
    with patch.object(Eagle3DraftModel, "load_verifier_weights"):
        loaded = Eagle3DraftModel.from_pretrained(str(model_dir))

    assert loaded.t2d is not None, "t2d is None after from_pretrained"
    assert loaded.d2t is not None, "d2t is None after from_pretrained"
    assert torch.equal(loaded.t2d, t2d), "t2d not loaded from saved safetensors"  # type: ignore[arg-type]
    assert torch.equal(loaded.d2t, d2t), "d2t not loaded from saved safetensors"  # type: ignore[arg-type]
