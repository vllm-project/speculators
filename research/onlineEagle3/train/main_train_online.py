# === set env BEFORE importing torch/vllm ===
import os
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("VLLM_TORCH_COMPILE", "0")  # avoid inductor meta-kernel asserts

import argparse
import json
import logging
import multiprocessing as mp
import queue
import random
import time
from math import ceil
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors
from torch import nn, optim
from torch.utils.data import DataLoader, IterableDataset as TorchIterableDataset

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed

from model.configs import EConfig
from model.llama_eagle3_full_grad import Model


# -------------------- args & constants --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--basepath", type=str, required=True)
parser.add_argument("--configpath", type=str, required=True)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--bs", type=int, default=7)
parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
parser.add_argument("--tmpdir", type=str, default="/tmp")
parser.add_argument("--cpdir", type=str, default="checkpoints")
parser.add_argument("--epoch", type=int, default=40)
parser.add_argument("--forward_num_total", type=int, default=3)
parser.add_argument("--ckpt_path", type=str, default=None)
parser.add_argument("--data_num", type=int, default=5000)
parser.add_argument("--log_every", type=int, default=10)
parser.add_argument("--quiet_vllm", action="store_true")
parser.add_argument("--hf_export_every", type=int, default=10, help="Export HF-style folder every N epochs (and at final epoch)")
args = parser.parse_args()

# reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# training/token window
TRAIN_TOKENS = int(os.environ.get("TRAIN_TOKENS", "1024"))
CHUNK_T = int(os.environ.get("VERIFIER_CHUNK_T", "256"))

# schedule figures
data_num = int(args.data_num)
train_frac = 1.0
total_steps = int(data_num * train_frac * (args.epoch + 1) / (args.bs * max(1, args.gradient_accumulation_steps)))
warm_steps = max(1, total_steps // 100)
train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "is_warmup": True,
    "num_epochs": args.epoch,
    "num_warmup_steps": warm_steps,
    "total_steps": total_steps,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 1.0,
    "config_path": args.configpath,
}


# -------------------- helpers --------------------
def _fmt_time(s: float) -> str:
    '''
    Format seconds as HhMmSs or MmSs or Ss.
    '''
    s = max(0, int(s))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def configure_gpu_visibility(gpu_ids: Optional[list[int]] = None) -> None:
    '''
    Set CUDA_VISIBLE_DEVICES to the specified gpu_ids list (or all if None).
    '''
    if gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))


def build_ds(tokenizer, split="train"):
    '''
    Build a tokenized dataset with loss masks from ShareGPT-style JSON.
    '''
    ds = load_dataset("json", data_files="ShareGPT_V4.3_unfiltered_cleaned_split.json")[split]
    ds = ds.shuffle(seed=42).select(range(0, data_num))
    original_columns = ds.column_names

    sep_assist = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    sep_user   = "<|eot_id|><|start_header_id|>user<|end_header_id|>"

    def preprocess(examples):
        out = {"conversation": [], "input_ids": [], "loss_mask": []}
        roles = {"human": "user", "gpt": "assistant"}

        for j in range(len(examples["id"])):
            messages = [
                {"role": "system", "content": "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024"},
            ]
            source = examples["conversations"][j]
            if roles.get(source[0]["from"], "user") != "user":
                source = source[1:]
            for s in source:
                role = roles[s["from"]]
                content = (" " + s["value"]) if s["from"] == "gpt" else s["value"]
                messages.append({"role": role, "content": content})

            conv = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id or tokenizer.unk_token_id

            ids = tokenizer(conv, return_tensors="pt", truncation=False, add_special_tokens=False).input_ids[0]
            loss_mask = torch.ones_like(ids)

            # build loss mask
            turns = conv.split(sep_user)
            if len(turns) > 1:
                turns[1] = turns[0] + sep_user + turns[1]
                turns = turns[1:]
            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if not turn:
                    break
                tlen = len(tokenizer(turn).input_ids)
                parts = turn.split(sep_assist)
                if len(parts) != 2:
                    break
                parts[0] += sep_assist
                inst_len = len(tokenizer(parts[0]).input_ids) - 1  # llama offset
                if i == 0:
                    loss_mask[cur_len: cur_len + inst_len - 2] = 0
                else:
                    loss_mask[cur_len - 3: cur_len + inst_len + 1] = 0
                cur_len += tlen
                if i != 0:
                    cur_len += 3
            loss_mask[cur_len:] = 0

            # crop to window
            ids = ids[-TRAIN_TOKENS:]
            loss_mask = loss_mask[-TRAIN_TOKENS:]

            out["conversation"].append(conv)
            out["input_ids"].append(ids[None, :])
            out["loss_mask"].append(loss_mask[None, :])

        return out

    ds = ds.map(preprocess, batched=True, remove_columns=original_columns, load_from_cache_file=False)
    ds.set_format(type="torch")
    return ds


def data_batches_loader(data, batch_size: int):
    batch = []
    while True:
        for item in data:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []


def online_data_generator(
    data_queue: mp.Queue,
    shutdown_event: mp.Event,
    verifier: str,
    data_path: str,
    batch_size: int = 32,
    gpu_ids: Optional[list[int]] = None,
    quiet: bool = True,
):
    # Confine this child process to the vLLM GPUs BEFORE importing vllm/torch.
    configure_gpu_visibility(gpu_ids)
    if quiet:
        logging.getLogger("vllm").setLevel(logging.ERROR)
        logging.getLogger("vllm.worker").setLevel(logging.ERROR)

    import vllm
    from vllm import SamplingParams

    try:
        tok = AutoTokenizer.from_pretrained(verifier, use_fast=False)
        ds = build_ds(tok)

        llm = vllm.LLM(
            model=verifier,
            enable_prefix_caching=False,
            tensor_parallel_size=len(gpu_ids) if gpu_ids else 1,
            disable_custom_all_reduce=True,
            dtype="bfloat16",
            max_model_len=8192,  # keep rotary cache at 8k to avoid compile meta-assert
            speculative_config={
                "model": "/cache/helen/speculators/research/onlineEagle3/meta-llama-3-70b-head-k3",
                "num_speculative_tokens": 1,
                "method": "eagle3",
            },
        )
        sp = SamplingParams(max_tokens=1)

        for prompt_batch in data_batches_loader(data=ds, batch_size=batch_size):
            if shutdown_event.is_set():
                break
            input_ids = [x["input_ids"] for x in prompt_batch]
            loss_mask = [x["loss_mask"] for x in prompt_batch]
            batched_ids = [ids[0].tolist() for ids in input_ids]

            try:
                outs = llm.generate(prompt_token_ids=batched_ids, sampling_params=sp, use_tqdm=False)
                for i, out in enumerate(outs):
                    T = len(batched_ids[i])
                    hs = out.hidden_states[:T]
                    aux = out.aux_hidden_states[:T]
                    item = {
                        "input_ids": input_ids[i],               # (1, T)
                        "loss_mask": loss_mask[i].cpu()[0],      # (T,)
                        "hidden_states": hs.unsqueeze(0),        # (1, T, H)
                        "aux_hidden_states": aux.unsqueeze(0),   # (1, T, H)
                    }
                    data_queue.put(item)
            except Exception as e:
                logging.error(f"[generator] vLLM batch error: {e}", exc_info=True)
                break

    except Exception as e:
        logging.error(f"[generator] fatal: {e}", exc_info=True)
        raise


class QueueDataset(TorchIterableDataset):
    def __init__(self, data_queue: mp.Queue, timeout: float = 1.0, eos_on_empty: bool = False):
        self.data_queue = data_queue
        self.timeout = timeout
        self.eos_on_empty = eos_on_empty

    def __iter__(self):
        while True:
            try:
                batch = self.data_queue.get(timeout=self.timeout)
                yield {"data": batch}
            except queue.Empty:
                if self.eos_on_empty:
                    break
            except Exception as err:
                logging.error(f"[loader] queue error: {err}")
                raise


def collate(batch):
    batch = [b["data"] for b in batch]

    # cap per-example
    for ex in batch:
        L = ex["input_ids"].shape[-1]
        if L > TRAIN_TOKENS:
            ex["input_ids"] = ex["input_ids"][..., -TRAIN_TOKENS:]
            ex["loss_mask"] = ex["loss_mask"][..., -TRAIN_TOKENS:]

    B = len(batch)
    T = min(max(i["input_ids"].shape[-1] for i in batch), TRAIN_TOKENS)

    input_ids = torch.zeros(B, T, dtype=torch.long)
    loss_mask = torch.zeros(B, T, dtype=torch.bool)
    for i in range(B):
        Li = min(len(batch[i]["input_ids"][0]), T)
        input_ids[i, :Li] = batch[i]["input_ids"][0][-Li:]
        loss_mask[i, :Li] = batch[i]["loss_mask"][-Li:].to(torch.bool)

    H_aux = batch[0]["aux_hidden_states"].shape[2]
    H_tch = batch[0]["hidden_states"].shape[2]
    aux_hidden_states = torch.zeros(B, T, H_aux, dtype=torch.bfloat16)
    hidden_states     = torch.zeros(B, T, H_tch, dtype=torch.bfloat16)
    for i in range(B):
        Li = min(len(batch[i]["aux_hidden_states"][0]), T)
        aux_hidden_states[i, :Li] = batch[i]["aux_hidden_states"][0][-Li:].to(torch.bfloat16)
        hidden_states[i, :Li]     = batch[i]["hidden_states"][0][-Li:].to(torch.bfloat16)

    return {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "aux_hidden_states": aux_hidden_states,
        "hidden_states": hidden_states,
    }


def get_gpu_ids_split(verifier_gpus: Union[float, int, list[int]],
                      train_gpus: Union[float, int, list[int]]):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("No GPUs available.")
    avail = list(range(torch.cuda.device_count()))

    if isinstance(verifier_gpus, int):
        ver_ids = avail[:verifier_gpus]
    elif isinstance(verifier_gpus, float):
        ver_ids = avail[: int(len(avail) * verifier_gpus)]
    else:
        if any(g not in avail for g in verifier_gpus):
            raise ValueError(f"Bad verifier GPUs: {verifier_gpus}")
        ver_ids = verifier_gpus

    rest = [g for g in avail if g not in ver_ids]
    if isinstance(train_gpus, int):
        tr_ids = rest[:train_gpus]
    elif isinstance(train_gpus, float):
        tr_ids = rest[int(len(rest) * train_gpus):]
    else:
        if any(g not in rest for g in train_gpus):
            raise ValueError(f"Bad training GPUs: {train_gpus}")
        tr_ids = train_gpus
    return ver_ids, tr_ids


# ---- HF-style save for the drafter ----
def save_hf_drafter(
    model_state: dict,
    teacher_cfg,                 # AutoConfig.from_pretrained(args.basepath)
    out_dir: str,
    draft_vocab_size: int = 32000,
    d2t_npy: Optional[str] = None,
    t2d_npy: Optional[str] = None,
):
    """
    Create a HF-ish folder that vLLM's Eagle3 drafter can load:
      - config.json (LlamaConfig-like + draft_vocab_size)
      - model.safetensors (weights)
      - mappings.safetensors (optional: d2t/t2d)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- 1) Build a Llama-style config for the drafter (1-layer head) ---
    cfg = {
        "model_type": "llama",
        "vocab_size": int(draft_vocab_size),
        "draft_vocab_size": int(draft_vocab_size),
        "hidden_size": int(getattr(teacher_cfg, "hidden_size")),
        "intermediate_size": int(getattr(teacher_cfg, "intermediate_size")),
        "num_hidden_layers": 1,
        "num_attention_heads": int(getattr(teacher_cfg, "num_attention_heads")),
        "num_key_value_heads": int(getattr(teacher_cfg, "num_key_value_heads", getattr(teacher_cfg, "num_attention_heads"))),
        "hidden_act": getattr(teacher_cfg, "hidden_act", "silu"),
        "rms_norm_eps": float(getattr(teacher_cfg, "rms_norm_eps", 1e-6)),
        "attention_bias": bool(getattr(teacher_cfg, "attention_bias", False)),
        "mlp_bias": bool(getattr(teacher_cfg, "mlp_bias", False)),
        "max_position_embeddings": int(getattr(teacher_cfg, "max_position_embeddings", 8192)),
        "rope_theta": float(getattr(teacher_cfg, "rope_theta", 500000.0)),
        # pass-through (vLLM will ignore unknowns)
        "use_cache": True,
        "tie_word_embeddings": False,
        "bos_token_id": getattr(teacher_cfg, "bos_token_id", None),
        "eos_token_id": getattr(teacher_cfg, "eos_token_id", None),
    }
    if getattr(teacher_cfg, "rope_scaling", None) is not None:
        try:
            cfg["rope_scaling"] = dict(teacher_cfg.rope_scaling)
        except Exception:
            cfg["rope_scaling"] = teacher_cfg.rope_scaling

    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # --- 2) Remap tensors ("midlayer." -> "layers.0.") and save ---
    weights_to_save = {}
    for k, v in model_state.items():
        if not isinstance(v, torch.Tensor):
            continue
        k2 = k
        if k2.startswith("midlayer."):
            k2 = k2.replace("midlayer.", "layers.0.")
        weights_to_save[k2] = v.detach().to("cpu")

    save_safetensors(weights_to_save, str(out / "model.safetensors"))

    # --- 3) (Optional) write mappings (d2t/t2d) ---
    extra = {}
    if d2t_npy is not None and Path(d2t_npy).exists():
        d2t = torch.from_numpy(np.load(d2t_npy)).to(torch.int64)
        assert d2t.ndim == 1, "d2t must be 1D"
        assert d2t.numel() == draft_vocab_size, f"d2t length {d2t.numel()} != draft_vocab {draft_vocab_size}"
        extra["d2t"] = d2t
    if t2d_npy is not None and Path(t2d_npy).exists():
        t2d = np.load(t2d_npy)
        t2d = torch.from_numpy(t2d.astype(np.int64, copy=False))  # store as int64
        assert t2d.ndim == 1, "t2d must be 1D"
        extra["t2d"] = t2d
    if extra:
        save_safetensors(extra, str(out / "mappings.safetensors"))

    print(f"[HF save] wrote {out}/config.json, model.safetensors" + (", mappings.safetensors" if extra else ""))


# -------------------- trainer --------------------
def train(data_queue: mp.Queue, args, train_config):
    """
    Single-process training wrapped by Accelerate (use with --num_processes 1).
    """
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=max(1, args.gradient_accumulation_steps),
    )
    device = accelerator.device
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(0)

    # ----- Build sliced verifier WT on this rank -----
    baseconfig = AutoConfig.from_pretrained(args.basepath)
    try:
        with open(os.path.join(args.basepath, "model.safetensors.index.json")) as f:
            idx = json.load(f)
        head_path = idx["weight_map"]["lm_head.weight"]
        with safe_open(os.path.join(args.basepath, head_path), framework="pt", device="cpu") as sf:
            sl = sf.get_slice("lm_head.weight")
            vocab_size, hidden_dim = sl.get_shape()
            lm_head_w_cpu = sl[:, :hidden_dim]
    except Exception:
        with open(os.path.join(args.basepath, "pytorch_model.bin.index.json")) as f:
            idx = json.load(f)
        head_path = idx["weight_map"]["lm_head.weight"]
        weights = torch.load(os.path.join(args.basepath, head_path), map_location="cpu")
        lm_head_w_cpu = weights["lm_head.weight"]

    map_tok_cpu = torch.from_numpy(np.load("t2d.npy")).bool()
    vocab_index = map_tok_cpu.nonzero(as_tuple=False).squeeze(-1).to(torch.long)  # CPU
    with torch.no_grad():
        W_sub_cpu = lm_head_w_cpu.index_select(0, vocab_index).contiguous()       # (V_sub, H) CPU
        WT = W_sub_cpu.to(torch.bfloat16, copy=True).to(device).t().contiguous()  # (H, V_sub) GPU
    del lm_head_w_cpu, W_sub_cpu, map_tok_cpu, vocab_index
    torch.cuda.empty_cache()

    # ----- drafter / optim / sched -----
    config = EConfig.from_pretrained(train_config["config_path"])
    model = Model(config, load_emb=True, path=args.basepath)
    optimizer = optim.AdamW(model.parameters(),
                            lr=train_config["lr"],
                            betas=(train_config["b1"], train_config["b2"]))
    scheduler = (get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(train_config["num_warmup_steps"]),
        num_training_steps=int(train_config["total_steps"]))
        if train_config["is_warmup"] else None
    )

    data_loader = DataLoader(
        QueueDataset(data_queue),
        batch_size=args.bs,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate,
    )

    model, optimizer, data_loader, scheduler = accelerator.prepare(
        model, optimizer, data_loader, scheduler
    )

    accum_steps = max(1, args.gradient_accumulation_steps)
    steps_per_epoch = ceil(data_num * 1.0 / (args.bs * accum_steps))
    max_steps = steps_per_epoch * accum_steps
    kldiv = nn.KLDivLoss(reduction="none")

    teacher_cfg = AutoConfig.from_pretrained(args.basepath)

    def kl_loss(verifier_probs, drafter_logits, loss_mask_btx1):
        out_logp = torch.log_softmax(drafter_logits.float(), dim=2)
        loss = kldiv(out_logp, verifier_probs)
        loss = torch.sum(torch.sum(loss_mask_btx1 * loss, dim=2)) / (loss_mask_btx1.sum() + 1e-5)
        return loss

    # ETA tracking
    global_step_counter = 0
    global_seconds_accum = 0.0
    last_log_t = time.perf_counter()
    last_log_step = 0

    for epoch in range(args.epoch):
        if accelerator.is_local_main_process:
            print(f"\n========== EPOCH {epoch+1}/{args.epoch} ==========")
        model.train()
        running = 0.0
        it = iter(data_loader)

        for step in range(1, max_steps + 1):
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                it = iter(data_loader)
                batch = next(it)

            input_ids  = batch["input_ids"].to(device, non_blocking=True)
            loss_mask  = batch["loss_mask"].to(device, non_blocking=True).float().unsqueeze(-1)  # (B,T,1)
            tgt_states = batch["hidden_states"].to(device, dtype=torch.bfloat16, non_blocking=True)
            aux_states = batch["aux_hidden_states"].to(device, dtype=torch.bfloat16, non_blocking=True)

            with accelerator.accumulate(model):
                # ----- verifier probs (chunk over time) -----
                with torch.no_grad():
                    B, T, Ht = tgt_states.shape
                    t_logits = []
                    for t0i in range(0, T, CHUNK_T):
                        t1i = min(T, t0i + CHUNK_T)
                        t_logits.append((tgt_states[:, t0i:t1i, :] @ WT).float())  # (B,Ct,Vsub)
                    verifier_logits = torch.cat(t_logits, dim=1)
                    verifier_probs  = torch.softmax(verifier_logits, dim=2)       # fp32

                # ----- drafter multi-forward unroll -----
                loss = torch.zeros((), device=device, dtype=torch.float32)
                hid_hist = []
                hid_proj = model.fc(aux_states)                                 # (B,T,H_model) bf16

                F = int(args.forward_num_total)
                with accelerator.autocast():
                    for _ in range(F):
                        pred_states = model(hid_proj, input_ids, None, hid_hist) # (B,T,H_model)
                        logits = model.lm_head_layernorm(pred_states)
                        logits = model.lm_head(logits.to(torch.bfloat16))        # (B,T,Vsub)
                        loss = loss + kl_loss(verifier_probs, logits, loss_mask)

                        hid_hist.append(hid_proj)
                        hid_proj = torch.cat([hid_proj[:, :1, :], pred_states[:, :-1, :]], dim=1)

                # handle non-finite loss gracefully
                if not torch.isfinite(loss):
                    if accelerator.is_local_main_process:
                        old_lr = optimizer.param_groups[0]["lr"]
                        new_lr = max(old_lr * 0.5, 1e-8)
                        print(f"[warn] non-finite loss detected. Lowering LR {old_lr:.2e} -> {new_lr:.2e} and skipping this micro-step.")
                        for g in optimizer.param_groups:
                            g["lr"] = new_lr  # <-- fixed
                    continue

                accelerator.backward(loss)
                running += float(loss)

                if accelerator.sync_gradients:
                    if train_config.get("grad_clip", None):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config["grad_clip"])
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

            # ---- ETA / speed print ----
            dt = time.perf_counter() - t0
            global_step_counter += 1
            global_seconds_accum += dt

            if accelerator.is_local_main_process and (step % max(1, args.log_every) == 0):
                now = time.perf_counter()
                steps_done_since = step - last_log_step
                avg_step = (now - last_log_t) / max(1, steps_done_since)
                epoch_steps_left = max_steps - step
                eta_epoch = epoch_steps_left * avg_step

                # global avg (smoothed across all steps so far)
                global_avg = global_seconds_accum / max(1, global_step_counter)
                epochs_left = (args.epoch - (epoch + 1))
                total_steps_left = epochs_left * max_steps + epoch_steps_left
                eta_total = total_steps_left * global_avg

                opt_step = (step + accum_steps - 1) // accum_steps
                curr_lr = optimizer.param_groups[0]["lr"]
                accelerator.print(
                    f"[eta] epoch {epoch+1}/{args.epoch} | step {step}/{max_steps} (opt {opt_step}/{steps_per_epoch}) | "
                    f"{avg_step*1000:.0f} ms/step | ETA epoch {_fmt_time(eta_epoch)} | ETA total {_fmt_time(eta_total)} | "
                    f"lr {curr_lr:.2e} | loss {loss.item():.4f}"
                )
                last_log_t = now
                last_log_step = step

            if step % 200 == 0:
                torch.cuda.empty_cache()

        if accelerator.is_local_main_process:
            accelerator.print(f"[epoch {epoch+1}] avg_loss={running / max_steps:.4f}")

            # ---- HF export every N epochs or at final epoch ----
            do_hf_export = ((epoch + 1) % max(1, args.hf_export_every) == 0) or ((epoch + 1) == args.epoch)
            if do_hf_export:
                hf_out = ckpt_dir / f"epoch_{epoch+1}-hf"
                # Get model state from accelerator
                model_state = accelerator.get_state_dict(model)
                save_hf_drafter(
                    model_state=model_state,
                    teacher_cfg=teacher_cfg,
                    out_dir=str(hf_out),
                    draft_vocab_size=32000,
                    d2t_npy="d2t.npy",   # optional: provide if present
                    t2d_npy="t2d.npy",   # optional: provide if present
                )
                print(f"[hf] wrote drafter folder: {hf_out}")

    if accelerator.is_local_main_process:
        print("finished training")


# -------------------- main --------------------
def main(
    verifier: str,
    data: str,
    verifier_batch_size: int,
    data_cache_limit: int = 16,
    verifier_gpus: Union[float, int, list[int]] = [2, 3],
    train_gpus: Union[float, int, list[int]] = [4, 5],
):
    ver_ids, tr_ids = verifier_gpus, train_gpus

    # Parent-only run info (avoid duplicate prints from spawn child)
    if mp.current_process().name == "MainProcess":
        print(f"training on {data_num} examples total")
        print(f"[config] TRAIN_TOKENS={TRAIN_TOKENS}  verifier_CHUNK_T={CHUNK_T}  bs={args.bs}  accum={max(1, args.gradient_accumulation_steps)}")
        steps_per_epoch = ceil(data_num / (args.bs * max(1, args.gradient_accumulation_steps)))
        print(f"[schedule] steps/epoch={steps_per_epoch} micro-steps, total_micro_steps={steps_per_epoch*args.epoch} warmup={warm_steps}")
        print("verifier", ver_ids)
        print("train", tr_ids)

    mp.set_start_method("spawn", force=True)
    ctx = mp.get_context("spawn")

    # Shared IPC (no Manager needed)
    data_queue = ctx.Queue(maxsize=int(data_cache_limit))
    shutdown_event = ctx.Event()

    # vLLM child must NOT be daemonic (it spawns its own workers)
    gen_proc = ctx.Process(
        target=online_data_generator,
        args=(data_queue, shutdown_event, verifier, data, verifier_batch_size, ver_ids, args.quiet_vllm or True),
        daemon=False,
    )
    gen_proc.start()

    # Make the training process only see training GPUs BEFORE touching CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, tr_ids))
    try:
        train(data_queue=data_queue, args=args, train_config=train_config)
    finally:
        shutdown_event.set()
        gen_proc.join(timeout=60)
        if gen_proc.is_alive():
            gen_proc.terminate()
            gen_proc.join()


if __name__ == "__main__":
    main(
        verifier=args.basepath,
        data="ShareGPT_V4.3_unfiltered_cleaned_split.json",
        verifier_batch_size=1,
    )
