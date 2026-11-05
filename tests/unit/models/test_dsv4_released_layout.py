"""The DSV4 draft must save in the layout the released draft ships, key for key.

Two independent checks, because either one alone passes on a broken mapping. The round
trip catches a mapping that loses data, but not one that is self-consistently wrong: a
rule and its own inverse agree with each other whatever they emit. Comparing the emitted
key set against the released one is the check with standing.

The released key set is stated here rather than read from a checkpoint, so the test runs
anywhere. It is transcribed from the released draft's weight index.
"""

import pytest
import torch
import torch.distributed as dist

transformers = pytest.importorskip("transformers")

from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from torch.distributed.checkpoint.state_dict import (  # noqa: E402
    StateDictOptions,
    set_model_state_dict,
)

from speculators.config import SpeculatorsConfig, VerifierConfig  # noqa: E402
from speculators.models.dspark.config import (  # noqa: E402
    DSparkSpeculatorConfig,
)
from speculators.models.dspark.core import DSparkDraftModel  # noqa: E402
from speculators.models.dsv4_dspark import checkpoint_mapping  # noqa: E402
from speculators.models.dsv4_dspark.backbone import attention  # noqa: E402
from speculators.models.dsv4_dspark.core import (  # noqa: E402
    DSV4DSparkConfig,
    DSV4DSparkDraftModel,
    resolve_init_parts,
)
from speculators.proposals.greedy import GreedyTokenProposalConfig  # noqa: E402
from speculators.train.checkpointer import (  # noqa: E402
    load_safetensors_state_dict,
    state_dict_from_checkpoint,
)

N_LAYERS = 3
N_EXPERTS = 4  # the release has 256; the mapping does not depend on how many

# Per stage, what the released draft carries (checked against its weight index).
_PER_STAGE = [
    "attn.attn_sink",
    "attn.kv_norm.weight",
    "attn.q_norm.weight",
    "attn.wkv.weight",
    "attn.wo_a.weight",
    "attn.wo_b.weight",
    "attn.wq_a.weight",
    "attn.wq_b.weight",
    "attn_norm.weight",
    "ffn.gate.bias",
    "ffn.gate.weight",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w3.weight",
    "ffn_norm.weight",
    "hc_attn_base",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
]
_FIRST_STAGE_ONLY = ["main_norm.weight", "main_proj.weight"]
_LAST_STAGE_ONLY = [
    "confidence_head.proj.weight",
    "hc_head_base",
    "hc_head_fn",
    "hc_head_scale",
    "markov_head.markov_w1.weight",
    "markov_head.markov_w2.weight",
    "norm.weight",
]
# Speculators carries the draft<->target vocab maps as buffers; the release has no
# equivalent because its draft is not vocab-reduced. Extra keys, not missing ones.
_SPECULATORS_EXTRA = {"t2d", "d2t"}


def released_key_set(n_layers: int, n_experts: int) -> set[str]:
    keys = {"embed.weight", "head.weight"}
    for i in range(n_layers):
        keys |= {f"mtp.{i}.{k}" for k in _PER_STAGE}
        for e in range(n_experts):
            keys |= {f"mtp.{i}.ffn.experts.{e}.w{w}.weight" for w in (1, 2, 3)}
    keys |= {f"mtp.0.{k}" for k in _FIRST_STAGE_ONLY}
    keys |= {f"mtp.{n_layers - 1}.{k}" for k in _LAST_STAGE_ONLY}
    return keys


def tiny_model(n_layers: int = N_LAYERS, **recipe) -> DSV4DSparkDraftModel:
    """A DSV4 draft small enough for CPU, structurally identical to the real one."""
    layer_config = transformers.LlamaConfig(
        hidden_size=64,
        vocab_size=97,
        num_hidden_layers=n_layers,
        rms_norm_eps=1e-6,
        intermediate_size=64,
        num_attention_heads=2,
        sliding_window=8,
        layer_types=["sliding_attention"] * n_layers,
    )
    config = DSV4DSparkConfig(
        transformer_layer_config=layer_config,
        num_heads=2,
        head_dim=16,
        rope_head_dim=8,
        q_lora_rank=16,
        o_lora_rank=16,
        o_groups=2,
        window_size=8,
        n_routed_experts=N_EXPERTS,
        n_shared_experts=1,
        n_activated_experts=2,
        moe_inter_dim=32,
        hc_mult=2,
        markov_rank=8,
        block_size=3,
        mask_token_id=5,
        aux_hidden_state_layer_ids=list(range(n_layers)),
        **recipe,
        speculators_config=SpeculatorsConfig(
            algorithm="dsv4_dspark",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            # None keeps `load_verifier_weights` from reaching for a real verifier.
            verifier=VerifierConfig.from_config(layer_config, name_or_path=None),
        ),
    )
    torch.manual_seed(0)
    model = DSV4DSparkDraftModel(config)
    for param in model.parameters():
        # Distinctive values: a mis-mapped tensor of the right shape still fails.
        torch.nn.init.normal_(param, std=1.0)
    return model


def saved_keys(path) -> set[str]:
    keys: set[str] = set()
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as handle:
            keys |= set(handle.keys())
    return keys


@pytest.fixture
def registered():
    assert checkpoint_mapping.register(n_layers=N_LAYERS)


@pytest.mark.smoke
def test_saved_keys_match_the_released_draft(tmp_path, registered):
    model = tiny_model()
    model.save_pretrained(tmp_path)

    produced = saved_keys(tmp_path)
    expected = released_key_set(N_LAYERS, N_EXPERTS)

    assert produced - _SPECULATORS_EXTRA - expected == set(), (
        "keys the release does not have"
    )
    assert expected - produced == set(), "released keys that were not produced"


@pytest.mark.smoke
def test_round_trip_is_bit_identical(tmp_path, registered):
    model = tiny_model()
    # verifier_* are re-read from the verifier at load, so they are not saved at all.
    shared = {"verifier_lm_head.weight", "verifier_norm.weight"}
    before = {k: v.clone() for k, v in model.state_dict().items() if k not in shared}

    model.save_pretrained(tmp_path)
    reloaded = DSV4DSparkDraftModel.from_pretrained(tmp_path)
    after = {k: v for k, v in reloaded.state_dict().items() if k not in shared}

    assert set(before) == set(after)
    mismatched = [
        k for k, v in before.items() if not torch.equal(v, after[k].to(v.dtype))
    ]
    assert mismatched == []


@pytest.mark.smoke
def test_experts_are_stacked_in_the_module_and_per_expert_on_disk(tmp_path, registered):
    model = tiny_model()
    assert model.state_dict()["layers.0.ffn.experts.w1"].shape[0] == N_EXPERTS

    model.save_pretrained(tmp_path)
    on_disk = saved_keys(tmp_path)
    assert "layers.0.ffn.experts.w1" not in on_disk
    assert (
        sum(1 for k in on_disk if k.startswith("mtp.0.ffn.experts.")) == N_EXPERTS * 3
    )


@pytest.mark.smoke
def test_resume_reads_back_the_layout_it_wrote(tmp_path, registered):
    """The trainer resumes from raw safetensors keys, not through `from_pretrained`."""
    trained = tiny_model()
    trained.save_pretrained(tmp_path)

    raw = load_safetensors_state_dict(tmp_path / "model.safetensors", "cpu")
    assert any(k.startswith("mtp.") for k in raw), "expected the released layout"

    resumed = tiny_model()  # zeroed, so parameters must actually be overwritten
    for param in resumed.parameters():
        torch.nn.init.zeros_(param)
    converted = state_dict_from_checkpoint(resumed, raw)
    resumed.load_state_dict(converted, strict=False)

    shared = {"verifier_lm_head.weight", "verifier_norm.weight"}
    for key, value in trained.state_dict().items():
        if key in shared:
            continue
        assert torch.equal(value, resumed.state_dict()[key]), key


@pytest.mark.smoke
def test_the_distributed_resume_path_loads_the_same_weights(tmp_path, registered):
    """The FSDP checkpointer resumes through `set_model_state_dict`, which shards
    what it is given into DTensors before calling `load_state_dict` -- which is why
    the translation happens before that call and not inside it."""
    trained = tiny_model()
    trained.save_pretrained(tmp_path)
    raw = load_safetensors_state_dict(tmp_path / "model.safetensors", "cpu")

    dist.init_process_group(
        backend="gloo", store=dist.HashStore(), rank=0, world_size=1
    )
    try:
        resumed = tiny_model()
        for param in resumed.parameters():
            torch.nn.init.zeros_(param)
        set_model_state_dict(
            resumed,
            state_dict_from_checkpoint(resumed, raw),
            options=StateDictOptions(
                full_state_dict=True, broadcast_from_rank0=True, strict=False
            ),
        )
    finally:
        dist.destroy_process_group()

    shared = {"verifier_lm_head.weight", "verifier_norm.weight"}
    for key, value in trained.state_dict().items():
        if key in shared:
            continue
        assert torch.equal(value, resumed.state_dict()[key]), key


@pytest.mark.smoke
def test_a_released_checkpoint_that_covers_nothing_is_refused(registered):
    """strict=False is needed for the verifier weights; it must not hide a mismatch."""
    model = tiny_model()
    with pytest.raises(RuntimeError, match="no source"):
        model.state_dict_from_checkpoint({"mtp.0.attn.wq_a.weight": torch.zeros(1)})


@pytest.mark.smoke
def test_the_checkpointer_hook_is_a_no_op_for_other_models():
    """The hook in train/checkpointer.py must not change behaviour for other models."""
    layer_config = transformers.Qwen3Config(
        hidden_size=64,
        vocab_size=97,
        num_hidden_layers=2,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        sliding_window=8,
        layer_types=["full_attention"] * 2,
    )
    other = DSparkDraftModel(
        DSparkSpeculatorConfig(
            transformer_layer_config=layer_config,
            block_size=3,
            mask_token_id=5,
            markov_rank=8,
            aux_hidden_state_layer_ids=[0, 1],
            speculators_config=SpeculatorsConfig(
                algorithm="dspark",
                proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(layer_config, name_or_path=None),
            ),
        )
    )
    assert not hasattr(other, "state_dict_from_checkpoint")
    state = dict(other.state_dict())
    assert state_dict_from_checkpoint(other, state) is state  # same object, untouched


@pytest.mark.smoke
def test_module_layout_checkpoints_still_load(tmp_path, registered):
    """Every checkpoint written before this mapping existed uses module names with the
    experts stacked. Those keys match no rule, so they pass through and still load."""
    trained = tiny_model()
    shared = {"verifier_lm_head.weight", "verifier_norm.weight"}
    state = {k: v for k, v in trained.state_dict().items() if k not in shared}
    save_file(
        {k: v.contiguous() for k, v in state.items()}, tmp_path / "model.safetensors"
    )
    trained.config.save_pretrained(tmp_path)

    reloaded = DSV4DSparkDraftModel.from_pretrained(tmp_path)
    for key, value in state.items():
        assert torch.equal(value, reloaded.state_dict()[key]), key


@pytest.mark.smoke
def test_freeze_routed_experts_leaves_the_rest_trainable(registered):
    """`--freeze-experts`: the routed experts are ~99% of the parameters, and holding
    them read-only is what lets this model train without expert parallelism."""
    model = tiny_model()
    expert_names = {
        name
        for name, _ in model.named_parameters()
        if ".ffn.experts." in name and "shared" not in name
    }
    assert expert_names, "expected stacked routed-expert parameters"

    # The verifier-shared embedding and head are already frozen by design, so compare
    # against the state before the call rather than assuming everything else trains.
    before = {name: p.requires_grad for name, p in model.named_parameters()}
    frozen = model.freeze_routed_experts()
    assert frozen == len(expert_names)

    for name, param in model.named_parameters():
        expected = False if name in expert_names else before[name]
        assert param.requires_grad is expected, name
    assert any(before[name] for name in expert_names), "experts should start trainable"


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (None, frozenset()),
        ([], frozenset()),
        (["attn"], frozenset({"attn"})),
        (["attn", "norm"], frozenset({"attn", "norm"})),
        (["all"], frozenset({"attn", "moe", "hc", "norm"})),
    ],
)
def test_init_from_target_selects_parts(requested, expected):
    assert resolve_init_parts(requested) == expected


def test_init_from_target_rejects_an_unknown_part():
    with pytest.raises(ValueError, match="unknown part"):
        resolve_init_parts(["attn", "bogus"])


def test_chunked_attention_matches_the_unchunked_path(monkeypatch):
    """Chunking the query axis bounds the dense fp32 logit block, which is the largest
    allocation in the model. It has to be a memory change and nothing else."""
    torch.manual_seed(0)
    n, sq, heads, dim, sk = 1, 37, 4, 16, 53
    q = torch.randn(n, sq, heads, dim, requires_grad=True)
    k, v = torch.randn(n, sk, dim), torch.randn(n, sk, dim)
    sink, bias = torch.randn(heads), torch.randn(n, sq, sk)
    scale = dim**-0.5

    whole = attention.sink_block_attention(q, k, v, sink, scale, bias)
    grad_whole = torch.autograd.grad(whole.sum(), q, retain_graph=True)[0]

    # ~5 query rows per chunk, so the loop runs several times.
    monkeypatch.setattr(attention, "_ATTN_CHUNK_BYTES", heads * sk * 4 * 5)
    chunked = attention.sink_block_attention(q, k, v, sink, scale, bias)
    grad_chunked = torch.autograd.grad(chunked.sum(), q)[0]

    torch.testing.assert_close(whole, chunked, rtol=0, atol=1e-5)
    torch.testing.assert_close(grad_whole, grad_chunked, rtol=0, atol=1e-5)

    # and with no bias to slice
    torch.testing.assert_close(
        attention.sink_block_attention(q, k, v, sink, scale),
        attention._attend(q, k, v, sink, scale, None),
        rtol=0,
        atol=1e-5,
    )
