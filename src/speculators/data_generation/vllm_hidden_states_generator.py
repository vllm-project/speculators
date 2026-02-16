"""Extract hidden states from intermediate layers during prefill using vLLM."""

import torch
from transformers import AutoConfig, AutoTokenizer
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_groups_uniform_spec,
    get_kv_cache_config_from_groups,
    get_request_block_hasher,
    init_none_hash,
    unify_hybrid_kv_cache_specs,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from speculators.utils.util import empty_cache, is_npu_available, mem_get_info

from .logging_utils import PipelineLogger

__all__ = ["VllmHiddenStatesGenerator"]

# Constants
CACHE_MEMORY_FRACTION = 0.2  # Fraction of GPU memory for KV cache
VLLM_BLOCK_SIZE = 128 if is_npu_available() else 16  # Block size for KV cache
MAX_NUM_SEQS = 32  # Maximum sequences for prefill-only workload
MIN_MAX_BATCHED_TOKENS = 8192  # Minimum batched tokens threshold
MAX_DECODE_TOKENS = 1  # Maximum tokens to generate (prefill only)
SAMPLING_TEMPERATURE = 0.0  # Temperature for sampling (greedy)
INITIAL_ARRIVAL_TIME = 0.0  # Initial request arrival time

log = PipelineLogger(__name__)


class VllmHiddenStatesGenerator:
    """Extracts hidden states from intermediate layers during prefill only.

    This module provides a generator for extracting hidden states from
    transformer models during the prefill phase using VLLM's inference engine.
    It is designed for generating training data for speculative decoding models
    like EAGLE3.

    The generator:
    - Uses VLLM's multiprocess executor for efficient batch inference
    - Patches model forward pass to capture intermediate layer hidden states
    - Operates in prefill-only mode (max_tokens=1) for data generation
    - Supports tensor parallelism for large models
    - Automatically manages KV cache and memory allocation

    Example:
        generator = VllmHiddenStatesGenerator(
            model_path="meta-llama/Llama-3.1-8B-Instruct",
            layer_ids=[10, 20, 30],
            tensor_parallel_size=2
        )

        results = generator.generate(token_ids)
        for result in results:
            input_ids = result["input_ids"]
            hidden_states = result["hidden_states"]  # List of tensors per layer`
    """

    def __init__(
        self,
        model_path: str,
        layer_ids: list[int] | None = None,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.8,
        tensor_parallel_size: int = 1,
        max_num_batched_tokens: int | None = None,
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self._request_counter = 0

        log.info(f"Initializing hidden states generator for {model_path}")
        log.info(f"Tensor parallel size: {tensor_parallel_size}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(config, "num_hidden_layers"):
            num_layers = config.num_hidden_layers
        elif hasattr(config, "text_config"):
            num_layers = config.text_config.num_hidden_layers
        else:
            raise ValueError("Cannot determine num_layers from config")

        log.info(f"Model has {num_layers} layers")

        if layer_ids is None:
            self.layer_ids = [2, num_layers // 2, num_layers - 3, num_layers - 1]
            log.info(
                f"Auto-selected layers: {self.layer_ids} "
                f"(from {num_layers} total layers)"
            )
        else:
            self.layer_ids = layer_ids
            log.info(f"Using specified layers: {layer_ids}")

        for layer_id in self.layer_ids:
            if layer_id < 0 or layer_id >= num_layers:
                raise ValueError(
                    f"Layer index {layer_id} out of bounds [0, {num_layers - 1}]"
                )

        self.vllm_config = self._create_vllm_config(
            model_path=model_path,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_num_batched_tokens=max_num_batched_tokens,
        )

        log.info("Initializing executor...")
        self.executor = MultiprocExecutor(vllm_config=self.vllm_config)

        log.info("Setting up hidden states capture...")
        self._setup_capture()

        log.info("Creating scheduler...")
        kv_cache_spec_list = self.executor.collective_rpc("get_kv_cache_spec")
        kv_cache_spec = kv_cache_spec_list[0]
        # Normalize hybrid KV cache specs for models with non-uniform attention
        # (e.g., GPT-OSS with sliding/full attention layers)
        unify_hybrid_kv_cache_specs(kv_cache_spec)
        kv_cache_groups = _get_kv_cache_groups_uniform_spec(kv_cache_spec)

        free_memory, _ = mem_get_info()
        cache_memory = int(free_memory * gpu_memory_utilization * CACHE_MEMORY_FRACTION)

        kv_cache_config = get_kv_cache_config_from_groups(
            vllm_config=self.vllm_config,
            kv_cache_groups=kv_cache_groups,
            available_memory=cache_memory,
        )

        self.vllm_config.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        structured_output_manager = StructuredOutputManager(
            vllm_config=self.vllm_config
        )

        self.scheduler = Scheduler(
            vllm_config=self.vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=VLLM_BLOCK_SIZE,
        )

        log.info("Initializing KV cache on all workers...")
        kv_cache_configs = [kv_cache_config] * tensor_parallel_size
        self.executor.initialize_from_config(kv_cache_configs)

        # Create block hasher for request KV cache management
        # Following vLLM's pattern in v1/engine/core.py
        caching_hash_fn = get_hash_fn_by_name(
            self.vllm_config.cache_config.prefix_caching_hash_algo
        )
        init_none_hash(caching_hash_fn)

        self.block_hasher = get_request_block_hasher(
            self.vllm_config.cache_config.block_size,
            caching_hash_fn,
        )

    def _create_vllm_config(
        self,
        model_path: str,
        max_model_len: int,
        gpu_memory_utilization: float,
        tensor_parallel_size: int,
        max_num_batched_tokens: int | None = None,
    ) -> VllmConfig:
        """Create VllmConfig with hidden states worker extension"""
        cache_config = CacheConfig(
            block_size=VLLM_BLOCK_SIZE,
            gpu_memory_utilization=gpu_memory_utilization,
            # disable to prevent cache state leakage
            enable_prefix_caching=False,
        )

        # For prefill-only workloads, use conservative scheduler limits
        # to reduce warmup memory allocation. max_num_seqs controls the
        # warmup allocation size (see gpu_worker.py:441-444).
        # We set it to a small value since we only do prefill in batches.
        max_num_seqs = MAX_NUM_SEQS
        if not max_num_batched_tokens:
            max_num_batched_tokens = max(MIN_MAX_BATCHED_TOKENS, max_model_len)

        return VllmConfig(
            model_config=ModelConfig(
                model=model_path,
                tokenizer=model_path,
                trust_remote_code=True,
                dtype="auto",
                max_model_len=max_model_len,
                enforce_eager=True,
            ),
            cache_config=cache_config,
            parallel_config=ParallelConfig(
                tensor_parallel_size=tensor_parallel_size,
                worker_extension_cls="speculators.data_generation.custom_worker.HiddenStatesWorkerExtension",
            ),
            scheduler_config=SchedulerConfig(
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                is_encoder_decoder=False,
            ),
            device_config=DeviceConfig(),
            load_config=LoadConfig(),
        )

    def _setup_capture(self):
        self.executor.collective_rpc(
            "_setup_hidden_states_capture",
            args=(self.layer_ids,),
        )

    def generate(self, token_ids: list[list[int]] | torch.Tensor) -> list[dict]:  # noqa: PLR0912, PLR0915
        """Extract hidden states from prefill phase only.

        Args:
            token_ids: Batch of token ID sequences as list[list[int]] or Tensor

        Returns:
            List of dicts with keys: input_ids, hidden_states, loss_mask
        """
        if isinstance(token_ids, torch.Tensor):
            input_ids_list = token_ids.tolist()
        else:
            if not token_ids:
                raise ValueError("token_ids cannot be empty")
            input_ids_list = token_ids

        log.debug(f"Generating hidden states for {len(input_ids_list)} sequences")
        # Account for max_tokens=1 in sampling params
        # (vLLM enforces: len(prompt) + max_tokens <= max_model_len)
        max_len = self.vllm_config.model_config.max_model_len - MAX_DECODE_TOKENS
        input_ids_list = [ids[:max_len] for ids in input_ids_list]

        # Track request IDs and prompt lengths for proper token attribution
        request_id_to_idx = {}
        request_id_to_prompt_len = {}

        for i, ids in enumerate(input_ids_list):
            # Ensure ids is a list (not tensor) for vLLM Request
            ids_list = ids.tolist() if isinstance(ids, torch.Tensor) else ids
            req_id = f"req_{self._request_counter}_{i}"
            request_id_to_idx[req_id] = i
            request_id_to_prompt_len[req_id] = len(ids_list)

            req = Request(
                request_id=req_id,
                prompt_token_ids=ids_list,
                sampling_params=SamplingParams(
                    max_tokens=MAX_DECODE_TOKENS, temperature=SAMPLING_TEMPERATURE
                ),
                pooling_params=None,
                eos_token_id=self.tokenizer.eos_token_id,
                arrival_time=INITIAL_ARRIVAL_TIME,
                block_hasher=self.block_hasher,
            )
            self.scheduler.add_request(req)

        # Increment to ensure unique request IDs across calls
        # (prevents KV cache corruption with delayed block freeing)
        self._request_counter += 1
        self.executor.collective_rpc("_reset_capture")

        # Track progress for each request to distinguish prefill from decode
        request_num_computed = dict.fromkeys(request_id_to_idx, 0)
        schedule_iterations = 0
        all_prefill_complete = False

        while (
            scheduler_output := self.scheduler.schedule()
        ).total_num_scheduled_tokens > 0 and not all_prefill_complete:
            schedule_iterations += 1

            # Calculate prefill tokens for each request (ignore decode tokens)
            prefill_metadata = {}
            for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
                num_already_computed = request_num_computed[req_id]
                num_prompt = request_id_to_prompt_len[req_id]
                num_prefill = max(0, min(num_tokens, num_prompt - num_already_computed))

                if num_prefill > 0:
                    prefill_metadata[req_id] = num_prefill

                request_num_computed[req_id] += num_tokens

            all_prefill_complete = all(
                request_num_computed[req_id] >= request_id_to_prompt_len[req_id]
                for req_id in request_id_to_idx
            )

            if prefill_metadata:
                self.executor.collective_rpc(
                    "_set_request_metadata", args=(prefill_metadata,)
                )

            model_output = self.executor.execute_model(scheduler_output)
            self.executor.sample_tokens(model_output)

        # Abort all requests (prefill complete, don't need decode)
        self.scheduler.finish_requests(
            list(request_id_to_idx.keys()), RequestStatus.FINISHED_ABORTED
        )

        # Get captured states organized by request ID
        request_states_dict = self.executor.collective_rpc(
            "_get_captured_states",
            unique_reply_rank=0,
        )

        if not request_states_dict:
            raise RuntimeError("Failed to capture hidden states from worker")

        log.debug(f"Captured states for {len(request_states_dict)} requests")

        # Map results back to original input order
        results = []
        for req_id in sorted(request_id_to_idx.keys()):
            i = request_id_to_idx[req_id]

            if req_id not in request_states_dict:
                raise RuntimeError(
                    f"Request {req_id} not found in captured states. "
                    f"Available: {list(request_states_dict.keys())}"
                )

            layer_states = [h.clone().cpu() for h in request_states_dict[req_id]]
            input_ids_tensor = torch.as_tensor(input_ids_list[i], dtype=torch.long)

            results.append(
                {
                    "input_ids": input_ids_tensor,
                    "hidden_states": layer_states,
                    "loss_mask": None,
                }
            )

        empty_cache()
        return results

    def __del__(self):
        if hasattr(self, "executor"):
            try:
                self.executor.shutdown()
            except Exception:
                log.warning("Exception during executor shutdown")
