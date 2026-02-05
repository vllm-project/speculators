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
from vllm.v1.engine.input_processor import InputProcessor
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

    def __init__(  # noqa: PLR0915
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
        self._input_processor: InputProcessor | None = None

        log.info(f"Initializing hidden states generator for {model_path}")
        log.info(f"Tensor parallel size: {tensor_parallel_size}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if not hasattr(self.tokenizer, "max_token_id"):
            try:
                max_token_id = self.tokenizer.vocab_size - 1
            except Exception:
                max_token_id = len(self.tokenizer) - 1
            self.tokenizer.max_token_id = max_token_id

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
    ) -> VllmConfig:
        """Create VllmConfig with hidden states worker extension"""
        cache_config = CacheConfig(
            block_size=VLLM_BLOCK_SIZE,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        # For prefill-only workloads, use conservative scheduler limits
        # to reduce warmup memory allocation. max_num_seqs controls the
        # warmup allocation size (see gpu_worker.py:441-444).
        # We set it to a small value since we only do prefill in batches.
        max_num_seqs = MAX_NUM_SEQS
        max_num_batched_tokens = max(
            MIN_MAX_BATCHED_TOKENS, max_model_len
        )  # Reduced from 65536

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

    def _get_input_processor(self) -> InputProcessor:
        # Lazily construct the vLLM input processor for multimodal requests.
        if self._input_processor is None:
            self._input_processor = InputProcessor(
                self.vllm_config, tokenizer=self.tokenizer
            )
        return self._input_processor

    def generate(  # noqa: PLR0912, PLR0915
        self,
        token_ids: list[list[int]] | torch.Tensor,
        prompt_texts: list[str] | None = None,
        multimodal_inputs: list[dict] | None = None,
        request_ids: list[str] | None = None,
    ) -> list[dict]:
        """Extract hidden states from prefill phase only.

        Args:
            token_ids: Batch of token ID sequences as list[list[int]] or Tensor
            prompt_texts: Optional prompt strings for multimodal processing
            multimodal_inputs: Optional multi-modal data aligned to each prompt

        Returns:
            List of dicts with keys: input_ids, hidden_states, loss_mask
        """
        if isinstance(token_ids, torch.Tensor):
            input_ids_list = token_ids.tolist()
        else:
            if not token_ids:
                raise ValueError("token_ids cannot be empty")
            input_ids_list = token_ids

        # Guard against multimodal misalignment: prompt text + mm inputs must be
        # present and batch-aligned with token IDs, or vLLM will mix samples.
        if multimodal_inputs is not None:
            if prompt_texts is None:
                raise ValueError("prompt_texts is required for multimodal inputs")
            if len(prompt_texts) != len(input_ids_list):
                raise ValueError(
                    "prompt_texts length must match token_ids length for multimodal"
                )
            if len(multimodal_inputs) != len(input_ids_list):
                raise ValueError("multimodal_inputs length must match token_ids length")

        log.debug(f"Generating hidden states for {len(input_ids_list)} sequences")
        # Account for max_tokens=1 in sampling params
        # (vLLM enforces: len(prompt) + max_tokens <= max_model_len)
        max_len = self.vllm_config.model_config.max_model_len - 1
        if multimodal_inputs is None:
            input_ids_list = [ids[:max_len] for ids in input_ids_list]

        for i, ids in enumerate(input_ids_list):
            # Ensure ids is a list (not tensor) for vLLM Request
            ids_list = ids.tolist() if isinstance(ids, torch.Tensor) else ids
            if request_ids is not None:
                request_id = request_ids[i]
            else:
                request_id = f"req_{self._request_counter}_{i}"
            sampling_params = SamplingParams(
                max_tokens=MAX_DECODE_TOKENS, temperature=SAMPLING_TEMPERATURE
            )

            mm_item = None if multimodal_inputs is None else multimodal_inputs[i]
            if multimodal_inputs is None or not mm_item:
                req = Request(
                    request_id=request_id,
                    prompt_token_ids=ids_list,
                    sampling_params=sampling_params,
                    pooling_params=None,
                    eos_token_id=self.tokenizer.eos_token_id,
                    arrival_time=INITIAL_ARRIVAL_TIME,
                    block_hasher=self.block_hasher,
                )
            else:
                # Provide per-request UUIDs for multimodal items to avoid
                # cache collisions that can drop mm_feature.data in vLLM.
                assert prompt_texts is not None
                prompt = {
                    "prompt": prompt_texts[i],
                    "multi_modal_data": mm_item,
                    "multi_modal_uuids": {"image": [f"{request_id}_img0"]},
                }
                processor = self._get_input_processor()
                engine_req = processor.process_inputs(
                    request_id=request_id,
                    prompt=prompt,
                    params=sampling_params,
                    arrival_time=INITIAL_ARRIVAL_TIME,
                )
                # vLLM should return multimodal features; if missing, the prompt
                # likely lacks image placeholders or preprocessing failed.
                if engine_req.mm_features is None:
                    raise ValueError(
                        "Multimodal request missing mm_features; "
                        "prompt may lack multimodal placeholders."
                    )
                for mm_index, mm_feature in enumerate(engine_req.mm_features):
                    mm_data = mm_feature.data
                    # Guard against empty feature payloads (would crash later).
                    if mm_data is None:
                        raise ValueError(
                            "Multimodal feature data is None; "
                            f"feature_index={mm_index} modality={mm_feature.modality}"
                        )
                    if mm_feature.modality == "image":
                        # Ensure image preprocessing produced grid sizes.
                        if (
                            not hasattr(mm_data, "get")
                            or mm_data.get("image_grid_thw") is None
                        ):
                            keys = (
                                list(mm_data.keys())
                                if hasattr(mm_data, "keys")
                                else type(mm_data)
                            )
                            raise ValueError(
                                "Multimodal image feature missing image_grid_thw; "
                                f"feature_index={mm_index} keys={keys}"
                            )
                prompt_token_ids = engine_req.prompt_token_ids
                if prompt_token_ids is None:
                    raise ValueError("Multimodal request missing prompt_token_ids")
                if len(prompt_token_ids) > max_len:
                    raise ValueError(
                        "Multimodal prompt exceeds max_model_len; reduce seq length"
                    )
                # Use vLLM tokenization for multimodal prompts to keep
                # image-expanded tokens consistent with model inputs.
                input_ids_list[i] = prompt_token_ids
                req = Request.from_engine_core_request(
                    engine_req, block_hasher=self.block_hasher
                )
            self.scheduler.add_request(req)

        # Increment to ensure unique request IDs across calls
        # (prevents KV cache corruption with delayed block freeing)
        self._request_counter += 1
        self.executor.collective_rpc("_reset_capture")

        schedule_iterations = 0
        while (
            scheduler_output := self.scheduler.schedule()
        ).total_num_scheduled_tokens > 0:
            schedule_iterations += 1
            log.debug(
                f"Scheduler iteration {schedule_iterations} - tokens: "
                f"{scheduler_output.total_num_scheduled_tokens}"
            )

            model_output = self.executor.execute_model(scheduler_output)
            self.executor.sample_tokens(model_output)

            for req_id in scheduler_output.num_scheduled_tokens:
                self.scheduler.finish_requests([req_id], RequestStatus.FINISHED_ABORTED)

        # Get captured states from driver worker
        aux_hidden_states = self.executor.collective_rpc(
            "_get_captured_states",
            unique_reply_rank=0,
        )

        if not aux_hidden_states or len(aux_hidden_states) == 0:
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
        empty_cache()

        return results

    def __del__(self):
        if hasattr(self, "executor"):
            try:
                self.executor.shutdown()
            except Exception:
                log.warning("Exception during executor shutdown")
