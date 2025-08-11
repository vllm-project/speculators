import csv
import logging
import multiprocessing
import os
import queue
from collections.abc import Iterable
from multiprocessing.synchronize import Event
from pathlib import Path
from typing import Optional, Union

import torch
import vllm
from vllm import SamplingParams
from datasets import IterableDataset
from datasets import load_dataset
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def build_ds(
    tokenizer,
    split="train",
):
    ds = load_dataset("json", data_files="ShareGPT_V4.3_unfiltered_cleaned_split.json")
    ds = ds[split]
    ds = ds.shuffle(seed=42)
    ds1 = ds.select(range(0, 500))

    original_columns1 = ds1.column_names

    def preprocess(examples):
        new_examples = {"conversation": [], "input_ids": [], "loss_mask": []}
        for j in range(len(examples["id"])):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024"
                    ),
                },
            ]
            roles = {"human": "user", "gpt": "assistant"}
            source = examples["conversations"][j]
            if roles[source[0]["from"]] != "user":
                # Skip the first one if it is not from human
                source = source[1:]
            for _, sentence in enumerate(source):
                role = roles[sentence["from"]]

                if sentence["from"] == "gpt":
                    sentence["value"] = " " + sentence["value"]
                messages.append({"role": role, "content": sentence["value"]})
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=4096,
                add_special_tokens=False,
            ).input_ids[0]
            loss_mask = torch.ones_like(input_ids)

            sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            sep2 = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
            turns = conversation.split(sep2)

            turns[1] = turns[0] + sep2 + turns[1]
            turns = turns[1:]

            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

                # Ignore the user instructions
                if i == 0:
                    loss_mask[cur_len : cur_len + instruction_len - 2] = 0
                else:
                    loss_mask[cur_len - 3 : cur_len + instruction_len + 1] = 0
                cur_len += turn_len
                if i != 0:
                    cur_len += 3

            loss_mask[cur_len:] = 0

            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])

        return new_examples

    ds1 = ds1.map(
        preprocess,
        batched=True,
        remove_columns=original_columns1,
        load_from_cache_file=False,
    )

    ds1.set_format(type="torch")
    return ds1





def configure_gpu_visibility(gpu_ids: Optional[list[int]] = None) -> None:
    """
    Configure GPU visibility for the current process.

    :param gpu_ids: List of GPU IDs to make visible. If None, all GPUs are visible.
    """
    if gpu_ids is not None:
        gpu_str = ",".join(map(str, gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        logging.info(f"Configured GPU visibility to: {gpu_str}")
    else:
        logging.info("Using all available GPUs")


def data_batches_loader(data, batch_size: int) :
    """
    Load the source data into prompts to run for online data generation.
    This function should return a generator that yields the prompts for iteration.

    Currently, implemented as a base, simple case with data loaded from a csv file.

    :param data: Path to the source data file or Hugging Face dataset id.
    :param batch_size: Size of the batch to be processed.
    :return: An iterable of prompts.
    """

    batch = []
    while True:  # infinite looping over the data
        for item in data:
            
            batch.append(item)  # Assuming the prompt is in the first column
            if len(batch) >= batch_size:
                yield batch
                batch = []


def online_data_generator(
    data_queue: multiprocessing.Queue,
    shutdown_event: Event,
    verifier: str,
    data: Dataset,
    batch_size: int = 32,
    gpu_ids: Optional[list[int]] = None,
):
    """
    Function used to generate data through vLLM for online training.
    It handles instantiation of vLLM with the provided model,
    loading the data from a source file, running the data through vLLM in batches,
    putting the data into the queue for training, and handling shutdown events.

    :param data_queue: Queue to put generated data into
    :param shutdown_event: Event to signal shutdown
    :param verifier: Model name/path for vLLM
    :param data: Path to source data file
    :param batch_size: Batch size for processing
    :param env_vars: Environment variables to set
    :param gpu_ids: List of GPU IDs to make visible for vLLM process
    """
    # Configure GPU visibility for this process
    configure_gpu_visibility(gpu_ids)

    try:
        tokenizer = AutoTokenizer.from_pretrained(verifier, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            verifier, device_map="auto"
        )
        model.eval()
        # model = vllm.LLM(model=verifier,enable_prefix_caching=False)
        data=build_ds(tokenizer)
        sampling_params=SamplingParams(max_tokens=1)
        for prompt_batch in data_batches_loader(
            data=data,
            batch_size=batch_size,
        ):
            text=[x['conversation'] for x in prompt_batch]
            input_ids=[x['input_ids'] for x in prompt_batch]
            loss_mask=[x['loss_mask'] for x in prompt_batch]
            print(input_ids[0].shape)

            try:

                with torch.no_grad():
                    responses=model(input_ids[0], output_hidden_states=True)

                num_layers=len(model.model.layers)
                feature_fusion = [
                    responses.hidden_states[3],
                    responses.hidden_states[num_layers // 2 + 1],
                    responses.hidden_states[-3],
                ]

                hidden_states = torch.cat(feature_fusion, dim=-1)

                output={
                    "input_ids": input_ids[0].cpu()[0],
                    "hidden_state": hidden_states[0],
                    "loss_mask": loss_mask[0].cpu()[0],
                    "target": responses.logits.cpu()[0],
                }
                print(output)
                data_queue.put(output)
            except Exception as batch_err:
                logging.error(f"Error processing batch: {batch_err}")
                continue
    except Exception as process_err:
        logging.error(f"Error in online data generator: {process_err}")
        raise process_err


class QueueDataset(IterableDataset):
    def __init__(
        self,
        data_queue: multiprocessing.Queue,
        timeout: float = 1.0,
        eos_on_empty: bool = False,
    ):
        """
        Initialize the QueueDataset which sources data from a multiprocessing Queue.

        :param data_queue: The multiprocessing queue containing training data
        :param timeout: Timeout in seconds when getting data from queue
        :param eos_on_empty: Whether to stop iteration when queue is empty
        """
        self.data_queue = data_queue
        self.timeout = timeout
        self.eos_on_empty = eos_on_empty

    def __iter__(self):
        while True:
            try:
                data_item = self.data_queue.get(timeout=self.timeout)
                yield {"text": data_item}
            except queue.Empty:
                if self.eos_on_empty:
                    break
            except Exception as err:
                logging.error(f"Error getting data from queue: {err}")
                raise err


def train(
    data_queue: multiprocessing.Queue,
    gpu_ids: Optional[list[int]] = None,
    **kwargs,
):
    """
    Training function that runs in the main process and handles training
    using the data from the queue.

    :param data_queue: Queue containing training data
    :param gpu_ids: List of GPU IDs to make visible for training process
    :param kwargs: Additional training arguments
    """
    train_dataset = QueueDataset(data_queue=data_queue)
    configure_gpu_visibility(gpu_ids)

    # implement training logic here


def get_gpu_ids_split(
    verifier_gpus: Union[float, int, list[int]],
    train_gpus: Union[float, int, list[int]],
) -> tuple[list[int], list[int]]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("No GPUs available for training.")

    available_gpus = list(range(torch.cuda.device_count()))

    if isinstance(verifier_gpus, int):
        verifier_gpu_ids = available_gpus[:verifier_gpus]
    elif isinstance(verifier_gpus, float):
        verifier_gpu_ids = available_gpus[: int(len(available_gpus) * verifier_gpus)]
    elif isinstance(verifier_gpus, list) and any(
        gpu_id not in available_gpus for gpu_id in verifier_gpus
    ):
        raise ValueError(f"Some verifier GPU IDs {verifier_gpus} are not available.")

    available_gpus = list(set(available_gpus) - set(verifier_gpu_ids))

    if isinstance(train_gpus, int):
        train_gpu_ids = available_gpus[train_gpus:]
    elif isinstance(train_gpus, float):
        train_gpu_ids = available_gpus[ int(len(available_gpus) * train_gpus):]
    elif isinstance(train_gpus, list) and any(
        gpu_id not in available_gpus for gpu_id in train_gpus
    ):
        raise ValueError(f"Some training GPU IDs {train_gpus} are not available.")

    return verifier_gpu_ids, train_gpu_ids


def main(
    verifier: str,
    data: str,
    verifier_batch_size: int,
    data_cache_limit: int = 1e6,
    verifier_gpus: Union[
        float, int, list[int]
    ] = 0.5,  # Default to half of available GPUs
    train_gpus: Union[float, int, list[int]] = 0.5,  # Default to half of available GPUs
    **train_kwargs,
) -> None:
    """
    Main function that runs in the main process and handles running training.

    :param verifier: Model name/path for vLLM verifier
    :param data: Path to source data file
    :param verifier_batch_size: Batch size for verifier processing
    :param data_cache_limit: Maximum size of data cache queue
    :param verifier_env_vars: Environment variables for verifier process
    :param verifier_gpu_ids: List of GPU IDs for vLLM data generation process
    :param train_gpu_ids: List of GPU IDs for training process
    :param train_kwargs: Additional training arguments
    """
    verifier_gpu_ids, train_gpu_ids = get_gpu_ids_split(
        verifier_gpus,
        train_gpus,
    )

    with multiprocessing.Manager() as manager:
        data_queue = multiprocessing.Queue(maxsize=int(data_cache_limit))
        data_gen_shutdown_event = manager.Event()

        data_gen_process = multiprocessing.Process(
            target=online_data_generator,
            args=(
                data_queue,
                data_gen_shutdown_event,
                verifier,
                data,
                verifier_batch_size,
                verifier_gpu_ids,
            ),
        )
        data_gen_process.start()

        try:
            train(data_queue=data_queue, gpu_ids=train_gpu_ids, **train_kwargs)
        finally:
            data_gen_shutdown_event.set()
            data_gen_process.join()


if __name__ == "__main__":
    main("meta-llama/Llama-3.1-8B-Instruct","ShareGPT_V4.3_unfiltered_cleaned_split.json", 1)