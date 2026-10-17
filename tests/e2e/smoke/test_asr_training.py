"""Qwen3-ASR uses the same online/offline training workflows as Qwen3-VL."""

import json
import wave

import openai
import pytest
from datasets import Dataset as HFDataset
from datasets import load_from_disk
from transformers import AutoProcessor

from hs_connectors import FileTransfer
from speculators.data_generation.offline import check_hidden_states
from speculators.data_generation.preprocessing import (
    _adapt_conv_for_vllm,
    build_speculator_training_dataset,
)
from speculators.data_generation.render_client import render_conversation
from speculators.data_generation.vllm_client import generate_hidden_states
from speculators.train.data import build_client_item
from tests.conftest import requires_cuda, requires_transformers_version
from tests.e2e.smoke.test_offline_training import run_offline_e2e
from tests.e2e.smoke.test_online_training import run_online_e2e
from tests.e2e.utils import ChatPrompts, launch_vllm_server_context

MODEL = "Qwen/Qwen3-ASR-0.6B-hf"
RESPONSE = "language English<asr_text>THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"


@pytest.fixture
def audio_conversations(tmp_path):
    """Small PCM fixtures cover resampling, stereo, and expanded audio lengths.

    Silence tests pipeline execution, not ASR quality or acceptance accuracy.
    Media is referenced by path, as with the dummy COCO images in the VL smoke.
    """
    conversations = []
    prompts: ChatPrompts = []
    for index, (sample_rate, channels, seconds) in enumerate(
        [(8_000, 1, 1), (16_000, 1, 2), (24_000, 2, 3)]
    ):
        path = tmp_path / f"audio #{index}.wav"
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(channels)
            audio.setsampwidth(2)
            audio.setframerate(sample_rate)
            audio.writeframes(bytes(sample_rate * seconds * channels * 2))
        conversation: list[dict] = [
            {"role": "system", "content": [{"type": "text", "text": ""}]},
            {"role": "user", "content": [{"type": "audio", "path": str(path)}]},
            {"role": "assistant", "content": [{"type": "text", "text": RESPONSE}]},
        ]
        conversations.append(conversation)
        prompts.append(_adapt_conv_for_vllm(conversation[:-1]))
    dataset_path = tmp_path / "conversations.jsonl"
    # Enough rows to exercise the normal train/validation split.
    dataset_path.write_text(
        "".join(
            json.dumps({"conversations": conv}) + "\n" for conv in conversations * 4
        ),
        encoding="utf-8",
    )
    return dataset_path, prompts


@pytest.mark.e2e
@pytest.mark.slow
@requires_cuda
@requires_transformers_version("5.13.0")
@pytest.mark.parametrize("workflow", ["online", "offline"])
def test_asr_training(tmp_path, audio_conversations, workflow):
    dataset_path, prompts = audio_conversations
    run = run_online_e2e if workflow == "online" else run_offline_e2e
    run(
        tmp_path,
        MODEL,
        str(dataset_path),
        max_samples=12,
        prompts=prompts,
        vllm_kwargs={"enforce_eager": True, "allowed_local_media_path": str(tmp_path)},
    )

    processor = AutoProcessor.from_pretrained(MODEL)
    prepared = load_from_disk(tmp_path / "data")
    assert len(prepared) == 12  # failed renders must not silently drop audio rows
    audio_counts = set()
    audio_token_id = processor.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    for row in prepared:
        ids, mask = row["input_ids"], row["loss_mask"].bool()
        audio_mask = ids == audio_token_id
        assert audio_mask.any()
        assert not mask[audio_mask].any()
        audio_counts.add(int(audio_mask.sum()))
        supervised = processor.decode(ids[mask].tolist())
        assert RESPONSE in supervised
        assert "<|im_end|>" in supervised
        assert row["messages"][0]["role"] == "system"
        assert row["messages"][1]["content"][0]["type"] == "audio_url"
    assert len(audio_counts) == 3
    state = json.loads((tmp_path / "checkpoints/0/training_state.json").read_text())
    assert state["global_step"] > 0
    metrics = json.loads((tmp_path / "results.json").read_text())["metrics"]
    assert metrics["num_drafts"] > 0


@pytest.mark.e2e
@pytest.mark.slow
@requires_cuda
@requires_transformers_version("5.13.0")
def test_asr_truncated_hidden_states(tmp_path, audio_conversations):
    """Extraction must retain the prepared audio/text prefix."""
    dataset_path, _ = audio_conversations
    conversation = json.loads(dataset_path.read_text().splitlines()[0])["conversations"]
    conversation[-1]["content"][0]["text"] = RESPONSE + " THE TRANSCRIPT" * 100
    processor = AutoProcessor.from_pretrained(MODEL)
    endpoint = "http://localhost:8321"
    hidden_states_path = tmp_path / "hidden_states"
    transfer = FileTransfer(hidden_states_path)
    with launch_vllm_server_context(
        MODEL,
        8321,
        str(hidden_states_path),
        enforce_eager=True,
        allowed_local_media_path=str(tmp_path),
    ):
        prepared = build_speculator_training_dataset(
            HFDataset.from_dict({"conversations": [conversation]}),
            processor,
            max_length=96,
            num_proc=1,
            render_endpoint=endpoint,
        )
        assert len(prepared) == 1
        row = prepared[0]
        item = build_client_item(row)
        full_ids = render_conversation(
            endpoint, item["messages"], add_generation_prompt=False
        )
        assert len(full_ids) > 96
        assert item["input_ids"] == full_ids[:96]
        assert row["loss_mask"].any()

        with openai.Client(base_url=f"{endpoint}/v1", api_key="unused") as client:
            handle, transfer_params = generate_hidden_states(
                client, MODEL, item, max_retries=0
            )
            try:
                states = transfer.get_generated(transfer_params)
                assert states is not None
                check_hidden_states(states, item["input_ids"])
            finally:
                transfer.delete(handle)
