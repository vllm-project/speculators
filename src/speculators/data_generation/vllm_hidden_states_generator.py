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
from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_groups_uniform_spec,
    get_kv_cache_config_from_groups,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .logging_utils import PipelineLogger

__all__ = ["VllmHiddenStatesGenerator"]

log = PipelineLogger(__name__)


class VllmHiddenStatesGenerator:
    """Extracts hidden states from intermediate layers during prefill only."""

    layer_ids: list[int]
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
        )

        log.info("Initializing executor...")
        self.executor = MultiprocExecutor(vllm_config=self.vllm_config)

        log.info("Setting up hidden states capture...")
        self._setup_capture()

        log.info("Creating scheduler...")
        kv_cache_spec_list = self.executor.collective_rpc("get_kv_cache_spec")
        kv_cache_spec = kv_cache_spec_list[0]
        kv_cache_groups = _get_kv_cache_groups_uniform_spec(kv_cache_spec)

        free_memory, _ = torch.cuda.mem_get_info()
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
        max_num_seqs = 32
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
        self.executor.collective_rpc(
            "_setup_hidden_states_capture",
            args=(self.layer_ids,),
        )

    def generate(self, token_ids: list[list[int]] | torch.Tensor) -> list[dict]:
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
        max_len = self.vllm_config.model_config.max_model_len - 1
        input_ids_list = [ids[:max_len] for ids in input_ids_list]

        for i, ids in enumerate(input_ids_list):
            # Ensure ids is a list (not tensor) for vLLM Request
            ids_list = ids.tolist() if isinstance(ids, torch.Tensor) else ids
            req = Request(
                request_id=f"req_{self._request_counter}_{i}",
                prompt_token_ids=ids_list,
                sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
                pooling_params=None,
                eos_token_id=self.tokenizer.eos_token_id,
                arrival_time=0.0,
            )
            self.scheduler.add_request(req)

        # Increment to ensure unique request IDs across calls
        # (prevents KV cache corruption with delayed block freeing)
        self._request_counter += 1

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
            layer_states = [
                h[offset : offset + seq_len].clone().cpu() for h in aux_hidden_states
            ]

            input_ids_tensor = torch.as_tensor(input_ids_list[i], dtype=torch.long)

            results.append(
                {
                    "input_ids": input_ids_tensor,
                    "hidden_states": layer_states,
                    "loss_mask": None,
                }
            )
            offset += seq_len

        del aux_hidden_states
        torch.cuda.empty_cache()

        return results

    def __del__(self):
        if hasattr(self, "executor"):
            try:
                self.executor.shutdown()
            except Exception:
                pass
