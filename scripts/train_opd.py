"""Draft-OPD training controller.

On-policy distillation for speculative draft models (arXiv:2605.29343).
Drives the online loop: generate verified sequence via vLLM → run draft model
to get proposals → send draft-proposed sequences to vLLM for target hidden
states at rejected positions → compute acceptance-aware OPD loss → backprop.

Requires vLLM running with extract_hidden_states speculative method.

Usage:
    python scripts/train_opd.py \
        --vllm-base-url http://localhost:8000 \
        --model Qwen/Qwen3-4B \
        --prompt-data /path/to/prompts.jsonl \
        --from-pretrained /path/to/sft_checkpoint \
        --verifier-name-or-path /path/to/verifier \
        --epochs 8 --lr 3e-4 --gamma-opd 0.8 \
        --save-path /path/to/output
"""

import argparse
import json
import logging
import os

import openai
import torch
from safetensors.torch import load_file
from tqdm import tqdm

from speculators.model import SpeculatorModel
from speculators.models.opd_loss import build_accept_mask, compute_opd_metrics

logger = logging.getLogger(__name__)


def generate_verified_sequence(
    client: openai.Client,
    model: str,
    prompt_token_ids: list[int],
    max_response_tokens: int = 4096,
    timeout: float = 300.0,
) -> tuple[str, list[int]]:
    """Generate a response from the target model and extract hidden states.

    Returns (hidden_states_path, full_token_ids) where full_token_ids is
    the prompt + generated response.
    """
    res = client.completions.create(
        model=model,
        prompt=prompt_token_ids,
        max_tokens=max_response_tokens,
        extra_body={
            "return_token_ids": True,
            "kv_transfer_params": {"include_output_tokens": True},
        },
        timeout=timeout,
    )
    kv_params = getattr(res, "kv_transfer_params", None)
    if kv_params is None:
        raise RuntimeError("vLLM response missing kv_transfer_params")

    hs_path = kv_params["hidden_states_path"]
    prompt_token_ids_back = getattr(res.choices[0], "prompt_token_ids", None)
    output_text = res.choices[0].text
    # Full sequence = prompt + generated tokens
    # Load from the safetensors to get exact token_ids
    return hs_path


def request_replay_hidden_states(
    client: openai.Client,
    model: str,
    draft_sequence: list[int],
    timeout: float = 120.0,
) -> str:
    """Send a draft-proposed sequence to vLLM for target hidden state extraction.

    Uses max_tokens=1 (prefill only) since the full sequence is already
    provided as the prompt.
    """
    res = client.completions.create(
        model=model,
        prompt=draft_sequence,
        max_tokens=1,
        extra_body={
            "return_token_ids": True,
        },
        timeout=timeout,
    )
    kv_params = getattr(res, "kv_transfer_params", None)
    if kv_params is None:
        raise RuntimeError("vLLM response missing kv_transfer_params")
    return kv_params["hidden_states_path"]


def load_hidden_states_sample(
    hs_path: str, device: torch.device
) -> dict[str, torch.Tensor]:
    """Load hidden states from safetensors into training-ready format."""
    data = load_file(hs_path, device=str(device))
    hidden_states = data["hidden_states"]
    token_ids = data["token_ids"].long()
    seq_len = token_ids.shape[0]
    return {
        "hidden_states": hidden_states[:, :-1].flatten(1).unsqueeze(0),
        "input_ids": token_ids.unsqueeze(0),
        "verifier_last_hidden_states": hidden_states[:, -1].unsqueeze(0),
        "loss_mask": torch.ones(1, seq_len, device=device),
        "document_ids": torch.zeros(1, seq_len, dtype=torch.long, device=device),
    }


def replay_rejected_blocks(
    client: openai.Client,
    model_name: str,
    draft_model: torch.nn.Module,
    token_ids: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    anchor_positions: torch.Tensor,
    accept_mask: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """For rejected blocks, get correct target logits from vLLM.

    For each block with at least one rejection, constructs
    [context_up_to_anchor + draft_proposals] and sends to vLLM to get
    target hidden states conditioned on the draft prefix. Computes target
    logits via the shared LM head and replaces the corresponding positions
    in the target logits tensor.
    """
    num_anchors = anchor_positions.shape[0]
    accept_blocks = accept_mask.view(num_anchors, block_size)
    rejected_block_indices = torch.where(~accept_blocks.all(dim=-1))[0]

    if len(rejected_block_indices) == 0:
        return targets

    corrected_targets = targets.clone()
    draft_preds = logits.argmax(dim=-1)
    token_ids_flat = token_ids.squeeze(0)

    for block_idx in rejected_block_indices:
        anchor_pos = anchor_positions[block_idx].item()
        block_start = block_idx * block_size
        block_draft = draft_preds[0, block_start + 1 : block_start + block_size]

        draft_seq = torch.cat([
            token_ids_flat[: anchor_pos + 1],
            block_draft,
        ])

        try:
            hs_path = request_replay_hidden_states(
                client, model_name, draft_seq.tolist()
            )
            replay_data = load_file(hs_path, device=str(device))
            replay_hs = replay_data["hidden_states"]

            replay_block_hs = replay_hs[-(block_size - 1) :, -1]
            with torch.no_grad():
                replay_target_logits = draft_model.verifier_lm_head(
                    draft_model.verifier_norm(replay_block_hs)
                )
            corrected_targets[0, block_start + 1 : block_start + block_size] = (
                replay_target_logits
            )
        except Exception:
            logger.warning(
                "Failed to replay block at anchor %d, using verified targets",
                anchor_pos,
                exc_info=True,
            )

    return corrected_targets


def load_prompts(prompt_data_path: str) -> list[list[int]]:
    """Load prompt token IDs from a jsonl file.

    Each line should have an "input_ids" field with a list of token IDs.
    """
    prompts = []
    with open(prompt_data_path) as f:
        for line in f:
            item = json.loads(line)
            prompts.append(item["input_ids"])
    return prompts


def parse_args():
    parser = argparse.ArgumentParser(description="Draft-OPD training")
    parser.add_argument("--vllm-base-url", required=True)
    parser.add_argument("--model", required=True, help="Model name for vLLM requests")
    parser.add_argument(
        "--prompt-data", required=True,
        help="Path to jsonl file with prompt token IDs (each line has 'input_ids')",
    )
    parser.add_argument("--from-pretrained", required=True, help="SFT checkpoint path")
    parser.add_argument(
        "--verifier-name-or-path", required=True,
        help="Path to verifier model (for loading LM head weights)",
    )
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma-opd", type=float, default=0.8)
    parser.add_argument("--lambda-acc", type=float, default=1.0)
    parser.add_argument("--lambda-rej", type=float, default=1.0)
    parser.add_argument("--max-anchors", type=int, default=3072)
    parser.add_argument("--max-response-tokens", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading draft model from %s", args.from_pretrained)
    registry = SpeculatorModel.registry
    model_class = registry["dflash"]
    draft_model = model_class.from_pretrained(args.from_pretrained)
    draft_model.load_verifier_weights(args.verifier_name_or_path)
    draft_model = draft_model.to(device)
    block_size = draft_model.block_size

    optimizer = torch.optim.AdamW(
        draft_model.parameters(), lr=args.lr, weight_decay=0.01
    )

    prompts = load_prompts(args.prompt_data)
    if args.max_samples:
        prompts = prompts[: args.max_samples]
    logger.info("Loaded %d prompts", len(prompts))

    client = openai.Client(base_url=f"{args.vllm_base_url}/v1", api_key="dummy")

    total_steps = args.epochs * len(prompts)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0,
        total_iters=max(1, warmup_steps),
    )

    os.makedirs(args.save_path, exist_ok=True)
    global_step = 0

    for epoch in range(args.epochs):
        draft_model.train()
        epoch_metrics = {
            "loss": 0.0, "acc_loss": 0.0, "rej_loss": 0.0,
            "n_accepted": 0, "n_rejected": 0, "n_replayed": 0,
            "n_samples": 0,
        }

        pbar = tqdm(prompts, desc=f"Epoch {epoch}")
        for prompt_idx, prompt_token_ids in enumerate(pbar):

            # Step 1: Generate verified sequence + hidden states via vLLM
            try:
                hs_path = generate_verified_sequence(
                    client, args.model, prompt_token_ids,
                    max_response_tokens=args.max_response_tokens,
                )
                sample = load_hidden_states_sample(hs_path, device)
            except Exception:
                logger.warning(
                    "Failed to generate/load sample %d, skipping",
                    prompt_idx, exc_info=True,
                )
                continue

            # Step 2: Draft model forward (with grad)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                _, logits, targets, aligned_loss_mask, anchored_block_indices = (
                    draft_model._backbone_forward(
                        sample["hidden_states"],
                        sample["input_ids"],
                        sample["loss_mask"],
                        sample["verifier_last_hidden_states"],
                        sample["document_ids"],
                        max_anchors=args.max_anchors,
                    )
                )

            # Step 3: Build accept/reject mask
            accept_mask = build_accept_mask(
                logits, targets, block_size,
                draft_model.config.sample_from_anchor,
            )

            # Step 4: Replay rejected blocks via vLLM
            num_anchors = logits.shape[1] // block_size
            anchor_positions = anchored_block_indices[::block_size]
            accept_blocks = accept_mask.view(num_anchors, block_size)
            n_rejected_blocks = int((~accept_blocks.all(dim=-1)).sum().item())

            if n_rejected_blocks > 0:
                targets = replay_rejected_blocks(
                    client, args.model, draft_model,
                    sample["input_ids"], logits, targets,
                    anchor_positions, accept_mask, block_size, device,
                )
                epoch_metrics["n_replayed"] += n_rejected_blocks

            # Step 5: OPD loss + backprop
            loss, metrics = compute_opd_metrics(
                logits, targets, accept_mask, aligned_loss_mask,
                block_size, gamma=args.gamma_opd,
                lambda_acc=args.lambda_acc, lambda_rej=args.lambda_rej,
                sample_from_anchor=draft_model.config.sample_from_anchor,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(draft_model.parameters(), 1.0)
            optimizer.step()
            if global_step < warmup_steps:
                warmup_scheduler.step()
            else:
                scheduler.step()

            global_step += 1
            epoch_metrics["n_samples"] += 1
            epoch_metrics["loss"] += metrics["loss_sum"].item()
            epoch_metrics["acc_loss"] += metrics["acc_loss_sum"].item()
            epoch_metrics["rej_loss"] += metrics["rej_loss_sum"].item()
            epoch_metrics["n_accepted"] += metrics["n_accepted_sum"].item()
            epoch_metrics["n_rejected"] += metrics["n_rejected_sum"].item()

            if global_step % args.log_freq == 0:
                n = epoch_metrics["n_samples"]
                total_pos = epoch_metrics["n_accepted"] + epoch_metrics["n_rejected"]
                pbar.set_postfix({
                    "loss": f"{epoch_metrics['loss'] / n:.4f}",
                    "acc%": f"{100 * epoch_metrics['n_accepted'] / max(1, total_pos):.0f}",
                    "replayed": epoch_metrics["n_replayed"],
                })

        n = max(1, epoch_metrics["n_samples"])
        total_pos = epoch_metrics["n_accepted"] + epoch_metrics["n_rejected"]
        logger.info(
            "Epoch %d: loss=%.4f acc_loss=%.4f rej_loss=%.4f "
            "accept_rate=%.1f%% replayed_blocks=%d samples=%d",
            epoch,
            epoch_metrics["loss"] / n,
            epoch_metrics["acc_loss"] / n,
            epoch_metrics["rej_loss"] / n,
            100 * epoch_metrics["n_accepted"] / max(1, total_pos),
            epoch_metrics["n_replayed"],
            epoch_metrics["n_samples"],
        )

        draft_model.save_pretrained(f"{args.save_path}/epoch_{epoch}")
        logger.info("Saved checkpoint to %s/epoch_%d", args.save_path, epoch)

    draft_model.save_pretrained(args.save_path)
    logger.info("Training complete. Final model saved to %s", args.save_path)


if __name__ == "__main__":
    main()
