"""
vLLM Hidden States Generator

Extracts hidden states from intermediate layers during prefill.
Uses vLLM's MultiprocExecutor for proper multi-GPU tensor parallelism.

Usage:
    from vllm_hidden_states_generator import VllmHiddenStatesGenerator

    # Auto-select layers
    generator = VllmHiddenStatesGenerator(
        model_path="Qwen/Qwen2.5-7B",
    )

    # Or manually specify layers
    generator = VllmHiddenStatesGenerator(
        model_path="Qwen/Qwen2.5-7B",
        layer_ids=[2, 14, 24],
    )

    # Multi-GPU tensor parallelism
    generator = VllmHiddenStatesGenerator(
        model_path="meta-llama/Llama-3.1-70B",
        tensor_parallel_size=4,
    )

    # Generate with token IDs
    token_ids = [1, 234, 567, 890]  # Single sequence
    data = generator.generate(token_ids=token_ids)

    # Or batch of sequences
    token_ids_batch = [[1, 234, 567], [1, 890, 123, 456]]
    data = generator.generate(token_ids=token_ids_batch)
"""

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
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_groups_uniform_spec,
    get_kv_cache_config_from_groups,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .logging_utils import PipelineLogger

log = PipelineLogger(__name__)


class VllmHiddenStatesGenerator:
    """
    Extracts hidden states from intermediate layers during prefill only.
    Uses MultiprocExecutor for proper multi-GPU tensor parallelism.
    """

    layer_ids: list[int]  # Populated in __init__
    model_path: str
    tensor_parallel_size: int
    _request_counter: int

    def __init__(
        self,
        model_path: str,
        layer_ids: list[int] | None = None,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.8,
        tensor_parallel_size: int = 1,
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self._request_counter = 0  # For unique request IDs across batches

        log.info(f"Initializing hidden states generator for {model_path}")
        log.info(f"Tensor parallel size: {tensor_parallel_size}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(config, "num_hidden_layers"):
            num_layers = config.num_hidden_layers
        elif hasattr(config, "text_config"):
            num_layers = config.text_config.num_hidden_layers
        else:
            raise ValueError("Cannot determine num_layers from config")

        log.info(f"Model has {num_layers} layers")

        # Auto-select layers if not specified
        # Matches EAGLE3 pattern:
        # - Feature fusion: hidden_states[3], hidden_states[num_layers // 2 + 1],
        #   hidden_states[-3]
        # - Target (last layer): hidden_states[-1]
        # - Note: hidden_states includes embedding (index 0),
        #   so we subtract 1 to get layer indices
        if layer_ids is None:
            self.layer_ids = [2, num_layers // 2, num_layers - 3, num_layers - 1]
            log.info(
                f"Auto-selected layers: {self.layer_ids} "
                f"(from {num_layers} total layers)"
            )
        else:
            self.layer_ids = layer_ids
            log.info(f"Using specified layers: {layer_ids}")

        # Validate layer indices
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
        )

        log.info("Initializing executor...")
        self.executor = MultiprocExecutor(vllm_config=self.vllm_config)

        log.info("Setting up hidden states capture...")
        self._setup_capture()

        log.info("Creating scheduler...")
        kv_cache_spec_list = self.executor.collective_rpc("get_kv_cache_spec")
        kv_cache_spec = kv_cache_spec_list[0]
        kv_cache_groups = _get_kv_cache_groups_uniform_spec(kv_cache_spec)

        # TODO: Optimize KV cache allocation for prefill-only workloads
        # Currently allocating based on available memory, but we only need minimal cache
        free_memory, total_memory = torch.cuda.mem_get_info()
        cache_memory = int(free_memory * gpu_memory_utilization * 0.2)

        kv_cache_config = get_kv_cache_config_from_groups(
            vllm_config=self.vllm_config,
            kv_cache_groups=kv_cache_groups,
            kv_cache_specs=kv_cache_spec,
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
        )

        log.info("Initializing KV cache on all workers...")
        kv_cache_configs = [kv_cache_config] * tensor_parallel_size
        self.executor.initialize_from_config(kv_cache_configs)

    def _create_vllm_config(
        self,
        model_path: str,
        max_model_len: int,
        gpu_memory_utilization: float,
        tensor_parallel_size: int,
    ) -> VllmConfig:
        """Create VllmConfig with hidden states worker extension"""
        cache_config = CacheConfig(
            block_size=16,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        # For prefill-only workloads, use conservative scheduler limits
        # to reduce warmup memory allocation. max_num_seqs controls the
        # warmup allocation size (see gpu_worker.py:441-444).
        # We set it to a small value since we only do prefill in batches.
        max_num_seqs = 32  # Reduced from 256
        max_num_batched_tokens = max(8192, max_model_len)  # Reduced from 65536

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
            ),
            device_config=DeviceConfig(),
            load_config=LoadConfig(),
        )

    def _setup_capture(self):
        """Setup hidden states capture on all workers"""
        # Call setup on all workers via RPC
        self.executor.collective_rpc(
            "_setup_hidden_states_capture",
            args=(self.layer_ids,),
        )

    def generate(
        self, token_ids: list[int] | list[list[int]] | torch.Tensor
    ) -> list[dict]:
        """
        Extract hidden states from prefill phase only (zero decode).

        Args:
            token_ids: Token IDs as either:
                - List[int]: Single sequence of token IDs
                - List[List[int]]: Batch of token ID sequences
                - torch.Tensor: Tensor of shape (batch_size, seq_len) or (seq_len,)

        Returns:
            List of dicts, one per sequence, each containing:
                - input_ids: torch.Tensor of token IDs
                - hidden_states: List[torch.Tensor] of hidden states (one per layer)
                - loss_mask: None (to be filled by caller)
        """
        if isinstance(token_ids, torch.Tensor):
            input_ids_list = [row.tolist() for row in token_ids]
        elif (
            isinstance(token_ids, list)
            and len(token_ids) > 0
            and isinstance(token_ids[0], int)
        ):
            input_ids_list = [token_ids]
        elif isinstance(token_ids, list) and len(token_ids) > 0:
            input_ids_list = token_ids
        else:
            raise ValueError(
                f"token_ids must be non-empty List[int], List[List[int]], "
                f"or torch.Tensor, got {type(token_ids)}"
            )

        log.debug(f"Generating hidden states for {len(input_ids_list)} sequences")

        max_len = self.vllm_config.model_config.max_model_len - 1
        input_ids_list = [ids[:max_len] for ids in input_ids_list]

        requests = []
        for i, ids in enumerate(input_ids_list):
            input_ids_for_req = ids.tolist() if isinstance(ids, torch.Tensor) else ids

            req = Request(
                request_id=f"req_{self._request_counter}_{i}",
                prompt_token_ids=input_ids_for_req,
                sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
                pooling_params=None,
                eos_token_id=self.tokenizer.eos_token_id,
                arrival_time=0.0,
            )
            requests.append(req)

        self._request_counter += 1

        for req in requests:
            self.scheduler.add_request(req)

        self.executor.collective_rpc("_enable_capture")

        schedule_iterations = 0
        while True:
            scheduler_output = self.scheduler.schedule()

            if scheduler_output.total_num_scheduled_tokens == 0:
                break

            schedule_iterations += 1
            log.debug(
                f"Scheduler iteration {schedule_iterations} - tokens: "
                f"{scheduler_output.total_num_scheduled_tokens}"
            )

            self.executor.execute_model(scheduler_output)

            for req_id in scheduler_output.num_scheduled_tokens:
                self.scheduler.finish_requests([req_id], RequestStatus.FINISHED_ABORTED)

        # Get captured states from driver worker
        captured_states_list = self.executor.collective_rpc(
            "_get_captured_states",
            unique_reply_rank=0,
        )
        aux_hidden_states = captured_states_list[0]

        self.executor.collective_rpc("_disable_capture")

        if not aux_hidden_states:
            raise RuntimeError("Failed to capture hidden states from worker")

        log.debug(f"Successfully captured {len(aux_hidden_states)} layers")

        seq_lens = [len(ids) for ids in input_ids_list]
        results = []
        offset = 0
        for i, seq_len in enumerate(seq_lens):
            # Clone slices and move to CPU to free GPU memory immediately
            # This prevents GPU memory accumulation across batches
            layer_states = [
                h[offset : offset + seq_len].clone().cpu() for h in aux_hidden_states
            ]

            # Convert to tensor efficiently
            input_ids_tensor = (
                input_ids_list[i]
                if isinstance(input_ids_list[i], torch.Tensor)
                else torch.as_tensor(input_ids_list[i], dtype=torch.long)
            )

            results.append(
                {
                    "input_ids": input_ids_tensor,
                    "hidden_states": layer_states,  # List of layer tensors
                    "loss_mask": None,
                }
            )
            offset += seq_len

        # Explicitly delete GPU tensors to free memory immediately
        del aux_hidden_states
        torch.cuda.empty_cache()

        return results

    def __del__(self):
        if hasattr(self, "executor"):
            try:
                self.executor.shutdown()
            except Exception:
                pass
