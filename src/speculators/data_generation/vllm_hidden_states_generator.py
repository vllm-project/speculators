"""
vLLM Hidden States Generator

Extracts hidden states from intermediate layers during prefill.
Uses vLLM's MultiprocExecutor for proper multi-GPU tensor parallelism.

Usage:
    from vllm_hidden_states_generator import VllmHiddenStatesGenerator

    # Auto-select layers (early=3, middle=num_layers//2+1, late=num_layers-3)
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

    data = generator.generate(prompts=["Your prompt here"])
"""

import torch
import logging
from typing import List, Dict
from transformers import AutoTokenizer, AutoConfig

from vllm.config import VllmConfig, ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig, DeviceConfig, LoadConfig
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

logger = logging.getLogger(__name__)


class VllmHiddenStatesGenerator:
    """
    Extracts hidden states from intermediate layers during prefill only.
    Uses MultiprocExecutor for proper multi-GPU tensor parallelism.
    """

    def __init__(
        self,
        model_path: str,
        layer_ids: List[int] = None,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.8,
        tensor_parallel_size: int = 1,
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size

        logger.info(f"Initializing hidden states generator for {model_path}")
        logger.info(f"Tensor parallel size: {tensor_parallel_size}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get model config
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(config, 'num_hidden_layers'):
            num_layers = config.num_hidden_layers
        elif hasattr(config, 'text_config'):
            num_layers = config.text_config.num_hidden_layers
        else:
            raise ValueError("Cannot determine num_layers from config")

        logger.info(f"Model has {num_layers} layers")

        # Auto-select layers if not specified
        # Matches EAGLE3 pattern: hidden_states[3], hidden_states[num_layers // 2 + 1], hidden_states[-3]
        # Since hidden_states includes embedding (index 0), we subtract 1 to get layer indices
        if layer_ids is None:
            self.layer_ids = [2, num_layers // 2, num_layers - 3]
            logger.info(f"Auto-selected layers: {self.layer_ids} (from {num_layers} total layers)")
        else:
            self.layer_ids = layer_ids
            logger.info(f"Using specified layers: {layer_ids}")

        # Create VllmConfig
        self.vllm_config = self._create_vllm_config(
            model_path=model_path,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        )

        # Initialize executor (handles both single and multi-GPU)
        logger.info("Initializing executor...")
        self.executor = MultiprocExecutor(vllm_config=self.vllm_config)

        # Setup hidden states capture on all workers
        logger.info("Setting up hidden states capture...")
        self._setup_capture()

        # Create scheduler
        logger.info("Creating scheduler...")
        from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups, _get_kv_cache_groups_uniform_spec

        # Get KV cache config from driver worker
        kv_cache_spec_list = self.executor.collective_rpc("get_kv_cache_spec")
        kv_cache_spec = kv_cache_spec_list[0]

        kv_cache_groups = _get_kv_cache_groups_uniform_spec(kv_cache_spec)

        # Get available memory
        free_memory, total_memory = torch.cuda.mem_get_info()
        cache_memory = int(free_memory * gpu_memory_utilization * 0.8)

        kv_cache_config = get_kv_cache_config_from_groups(
            vllm_config=self.vllm_config,
            kv_cache_groups=kv_cache_groups,
            kv_cache_specs=kv_cache_spec,
            available_memory=cache_memory,
        )

        # Set num_gpu_blocks in cache_config (required by Scheduler)
        self.vllm_config.cache_config.num_gpu_blocks = kv_cache_config.num_blocks

        structured_output_manager = StructuredOutputManager(vllm_config=self.vllm_config)

        from vllm.v1.core.sched.scheduler import Scheduler
        self.scheduler = Scheduler(
            vllm_config=self.vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
        )

        # Initialize KV cache on all workers
        # Each worker needs its own config (indexed by rpc_rank)
        logger.info("Initializing KV cache on all workers...")
        kv_cache_configs = [kv_cache_config] * tensor_parallel_size
        self.executor.initialize_from_config(kv_cache_configs)


        logger.info("Initialization complete")

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
                max_num_seqs=256,
                max_model_len=max_model_len,
                max_num_batched_tokens=8192,
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

    def generate(self, prompts: List[str]) -> Dict:
        """
        Extract hidden states from prefill phase only (zero decode).
        """
        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]

        logger.info(f"Generating hidden states for {len(prompts)} prompts")

        # Tokenize with padding for batching
        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.vllm_config.model_config.max_model_len,
        )
        input_ids_list = tokenized["input_ids"].tolist()

        # Store the original tensor for return
        input_ids_tensor = tokenized["input_ids"]

        # Create requests
        requests = []
        for i, input_ids in enumerate(input_ids_list):
            req = Request(
                request_id=f"req_{i}",
                prompt_token_ids=input_ids,
                sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
                pooling_params=None,
                eos_token_id=self.tokenizer.eos_token_id,
                arrival_time=0.0,
            )
            requests.append(req)

        # Add to scheduler
        for req in requests:
            self.scheduler.add_request(req)

        # Enable capture on all workers
        self.executor.collective_rpc("_enable_capture")

        # Schedule
        scheduler_output = self.scheduler.schedule()

        if scheduler_output.total_num_scheduled_tokens == 0:
            logger.warning("No tokens scheduled")
            self.executor.collective_rpc("_disable_capture")
            return {
                'input_ids': tokenized['input_ids'],
                'aux_hidden_states': None,
                'layer_ids': self.layer_ids,
            }

        # Execute model (prefill only)
        self.executor.execute_model(scheduler_output)

        # Get captured states from driver worker
        captured_states_list = self.executor.collective_rpc(
            "_get_captured_states",
            unique_reply_rank=0,  # Only get from driver worker
        )
        aux_hidden_states = captured_states_list[0]

        # Disable capture
        self.executor.collective_rpc("_disable_capture")

        # Remove requests
        for req in requests:
            self.scheduler.finish_requests([req.request_id], RequestStatus.FINISHED_ABORTED)

        logger.info(f"Successfully captured {len(aux_hidden_states) if aux_hidden_states else 0} layers")

        # Reshape captured states from (total_tokens, hidden_dim) to (batch_size, seq_len, hidden_dim)
        # vLLM batches all tokens together, we need to unflatten based on actual sequence lengths
        if aux_hidden_states:
            # Get actual sequence lengths from input_ids
            seq_lens = [len(ids) for ids in input_ids_list]

            # Reshape each captured layer
            reshaped_states = []
            for h in aux_hidden_states:
                # h is (total_tokens, hidden_dim), split by sequence lengths
                batch_states = []
                offset = 0
                for seq_len in seq_lens:
                    batch_states.append(h[offset:offset + seq_len])
                    offset += seq_len
                # Stack to (batch_size, seq_len, hidden_dim)
                reshaped = torch.stack(batch_states)
                reshaped_states.append(reshaped)
            aux_hidden_states = reshaped_states

        # Format output to match EAGLE3:
        # - input_ids: (seq_len,)
        # - hidden_state: (seq_len, hidden_size * 3) - concatenated aux layers
        # - loss_mask: (seq_len,) - user provided
        # - target: (seq_len, hidden_size) - final layer output

        results = []
        for i in range(len(prompts)):
            # Concatenate auxiliary hidden states
            if aux_hidden_states:
                hidden_state = torch.cat([h[i] for h in aux_hidden_states], dim=-1)
            else:
                hidden_state = None

            result = {
                'input_ids': input_ids_tensor[i],  # Use the stored tensor
                'hidden_state': hidden_state,
                'loss_mask': None,  # User needs to provide this externally
                'target': None,  # TODO: capture final layer if needed
            }
            results.append(result)

        # Return single dict if single prompt, otherwise list
        return results[0] if len(results) == 1 else results

    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'executor'):
            self.executor.shutdown()
