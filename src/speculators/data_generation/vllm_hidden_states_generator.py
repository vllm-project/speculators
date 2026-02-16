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
from .vllm_hidden_states_utils import (
    build_generation_results,
    ensure_tokenizer_max_token_id,
    get_embed_length_mismatch,
    get_request_id,
    infer_num_hidden_layers,
    normalize_token_ids_batch,
    resolve_layer_ids,
    sequence_lengths,
    to_token_id_list,
    truncate_text_only_batch,
    validate_multimodal_batch_alignment,
    validate_multimodal_engine_features,
)

__all__ = ["VllmHiddenStatesGenerator"]

# Constants
CACHE_MEMORY_FRACTION = 0.2  # Fraction of GPU memory for KV cache
VLLM_BLOCK_SIZE = 128 if is_npu_available() else 16  # Block size for KV cache
MAX_NUM_SEQS = 32  # Maximum sequences for prefill-only workload
MIN_MAX_BATCHED_TOKENS = 32768  # Minimum batched tokens threshold
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
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.8,
        tensor_parallel_size: int = 1,
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self._request_counter = 0
        self._mm_request_counter = 0
        self._input_processor: InputProcessor | None = None

        log.info(f"Initializing hidden states generator for {model_path}")
        log.info(f"Tensor parallel size: {tensor_parallel_size}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        ensure_tokenizer_max_token_id(self.tokenizer)

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        num_layers = infer_num_hidden_layers(config)

        log.info(f"Model has {num_layers} layers")
        self.layer_ids = resolve_layer_ids(layer_ids, num_layers)
        if layer_ids is None:
            log.info(
                f"Auto-selected layers: {self.layer_ids} "
                f"(from {num_layers} total layers)"
            )
        else:
            log.info(f"Using specified layers: {self.layer_ids}")

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
                worker_extension_cls=(
                    "speculators.data_generation.custom_worker."
                    "HiddenStatesWorkerExtension"
                ),
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
                self.vllm_config
            )
        return self._input_processor

    def _process_multimodal_inputs_with_truncation(
        self,
        *,
        request_id: str,
        prompt_text: str,
        multimodal_item: dict,
        sampling_params: SamplingParams,
    ):
        """Build multimodal engine request and fail fast on max-length overflow."""
        processor = self._get_input_processor()
        unique_suffix = f"{self._mm_request_counter}"
        self._mm_request_counter += 1
        engine_request_id = f"{request_id}__mm_{unique_suffix}"
        image_uuid = f"{engine_request_id}_img0"

        prompt = {
            "prompt": prompt_text,
            "multi_modal_data": multimodal_item,
            "multi_modal_uuids": {"image": [image_uuid]},
        }
        try:
            return processor.process_inputs(
                request_id=engine_request_id,
                prompt=prompt,
                params=sampling_params,
                arrival_time=INITIAL_ARRIVAL_TIME,
            )
        except ValueError as exc:
            msg = str(exc)
            if "longer than the maximum model length" not in msg:
                raise
            raise ValueError(
                "Multimodal prompt exceeds max_model_len and strict mode is enabled; "
                f"request_id={request_id}. {msg}"
            ) from exc

    def get_multimodal_prompt_token_count(
        self,
        *,
        request_id: str,
        prompt_text: str,
        multimodal_item: dict,
    ) -> int:
        """Return exact multimodal prompt token count for strict pre-check."""
        sampling_params = SamplingParams(
            max_tokens=MAX_DECODE_TOKENS, temperature=SAMPLING_TEMPERATURE
        )
        engine_req = self._process_multimodal_inputs_with_truncation(
            request_id=request_id,
            prompt_text=prompt_text,
            multimodal_item=multimodal_item,
            sampling_params=sampling_params,
        )
        validate_multimodal_engine_features(engine_req)
        prompt_token_ids = engine_req.prompt_token_ids
        if prompt_token_ids is None:
            raise ValueError("Multimodal request missing prompt_token_ids")
        max_len = self.vllm_config.model_config.max_model_len - MAX_DECODE_TOKENS
        if len(prompt_token_ids) > max_len:
            raise ValueError(
                "Multimodal prompt token ids exceed prefill-safe limit in strict mode: "
                f"len={len(prompt_token_ids)} max_len={max_len} request_id={request_id}"
            )
        return len(prompt_token_ids)

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
        input_ids_list = normalize_token_ids_batch(token_ids)
        validate_multimodal_batch_alignment(
            input_ids_list,
            prompt_texts=prompt_texts,
            multimodal_inputs=multimodal_inputs,
        )

        log.debug(f"Generating hidden states for {len(input_ids_list)} sequences")
        # Account for max_tokens=1 in sampling params for pure text mode.
        # Multimodal mode is handled in strict mode and raises on overflow.
        max_len = self.vllm_config.model_config.max_model_len - 1
        if multimodal_inputs is None:
            input_ids_list = truncate_text_only_batch(input_ids_list, max_len=max_len)

        for i, ids in enumerate(input_ids_list):
            # Ensure ids is a list (not tensor) for vLLM Request
            ids_list = to_token_id_list(ids)
            request_id = get_request_id(request_ids, self._request_counter, i)
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
                engine_req = self._process_multimodal_inputs_with_truncation(
                    request_id=request_id,
                    prompt_text=prompt_texts[i],
                    multimodal_item=mm_item,
                    sampling_params=sampling_params,
                )
                validate_multimodal_engine_features(engine_req)
                prompt_token_ids = engine_req.prompt_token_ids
                if prompt_token_ids is None:
                    raise ValueError("Multimodal request missing prompt_token_ids")
                if len(prompt_token_ids) > max_len:
                    raise ValueError(
                        "Multimodal prompt token ids exceed prefill-safe limit in "
                        f"strict mode: len={len(prompt_token_ids)} max_len={max_len} "
                        f"request_id={request_id}"
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
        captured_input_embeds = self.executor.collective_rpc(
            "_get_captured_input_embeds",
            unique_reply_rank=0,
        )

        if not aux_hidden_states or len(aux_hidden_states) == 0:
            raise RuntimeError("Failed to capture hidden states from worker")

        log.debug(f"Successfully captured {len(aux_hidden_states)} layers")

        seq_lens = sequence_lengths(input_ids_list)
        mismatch, expected_num_tokens, actual_num_tokens = get_embed_length_mismatch(
            captured_input_embeds,
            seq_lens,
        )
        if mismatch:
            batch_max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
            max_model_len = self.vllm_config.model_config.max_model_len
            shortfall = expected_num_tokens - actual_num_tokens
            expected_exceeds_batch_cap = expected_num_tokens > batch_max_tokens
            seq_lens_top = sorted(seq_lens, reverse=True)[:8]
            mismatch_context = (
                "Mismatch details: "
                f"num_requests={len(input_ids_list)}, "
                f"schedule_iterations={schedule_iterations}, "
                f"max_model_len={max_model_len}, "
                f"batch_max_tokens={batch_max_tokens}, "
                f"expected_exceeds_batch_cap={expected_exceeds_batch_cap}, "
                f"shortfall={shortfall}, "
                f"min_seq_len={min(seq_lens)}, "
                f"max_seq_len={max(seq_lens)}, "
                f"top_seq_lens={seq_lens_top}."
            )
            # Fallback: some vLLM schedules chunk prefill when total prompt
            # tokens exceed max_num_batched_tokens; our prefill-only abort
            # path can then capture only the first chunk embeddings.
            # Regenerate this batch sample-by-sample to keep exact alignment.
            if len(input_ids_list) > 1:
                log.warning(
                    "Captured input embeddings length mismatch in batched mode "
                    f"(expected={expected_num_tokens}, actual={actual_num_tokens}); "
                    f"{mismatch_context} "
                    "Try reducing data_generation batch_size or increasing total token "
                    "budget (max_model_len / max_num_batched_tokens); "
                    "falling back to per-sample generation for this batch."
                )
                fallback_results: list[dict] = []
                for i in range(len(input_ids_list)):
                    sample_prompt = None if prompt_texts is None else [prompt_texts[i]]
                    sample_mm = (
                        None if multimodal_inputs is None else [multimodal_inputs[i]]
                    )
                    sample_req_id = None if request_ids is None else [request_ids[i]]
                    fallback_results.extend(
                        self.generate(
                            [input_ids_list[i]],
                            prompt_texts=sample_prompt,
                            multimodal_inputs=sample_mm,
                            request_ids=sample_req_id,
                        )
                    )
                return fallback_results

            # Single-sample mismatch should not happen; fail fast for visibility.
            raise RuntimeError(
                "Captured input embeddings length mismatch: "
                f"expected={expected_num_tokens}, actual={actual_num_tokens}. "
                f"{mismatch_context}"
            )

        results = build_generation_results(
            aux_hidden_states=aux_hidden_states,
            input_ids_list=input_ids_list,
            captured_input_embeds=captured_input_embeds,
        )
        empty_cache()

        return results

    def __del__(self):
        if hasattr(self, "executor"):
            try:
                self.executor.shutdown()
            except Exception:
                log.warning("Exception during executor shutdown")
