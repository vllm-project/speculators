"""
Configuration registries for data generation pipeline.

This module provides centralized configuration for:
- Chat templates for different model formats
- Dataset loading configurations
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable


# ============================================================================
# Chat Template Configurations
# ============================================================================

@dataclass
class ChatTemplate:
    """Chat template configuration for a specific model format"""
    name: str
    system_prompt: str
    user_header: str
    assistant_header: str
    end_of_turn_token: str
    bos_token: Optional[str] = None
    eos_token: Optional[str] = None


CHAT_TEMPLATES: Dict[str, ChatTemplate] = {
    "llama3": ChatTemplate(
        name="llama3",
        system_prompt="You are a helpful assistant.",
        user_header="<|start_header_id|>user<|end_header_id|>\n\n",
        assistant_header="<|start_header_id|>assistant<|end_header_id|>\n\n",
        end_of_turn_token="<|eot_id|>",
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
    ),
    "qwen2": ChatTemplate(
        name="qwen2",
        system_prompt="You are a helpful assistant.",
        user_header="<|im_start|>user\n",
        assistant_header="<|im_start|>assistant\n",
        end_of_turn_token="<|im_end|>\n",
        bos_token="",
        eos_token="<|endoftext|>",
    ),
    "vicuna": ChatTemplate(
        name="vicuna",
        system_prompt="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.",
        user_header="USER: ",
        assistant_header="ASSISTANT: ",
        end_of_turn_token="\n",
        bos_token="<s>",
        eos_token="</s>",
    ),
    "chatml": ChatTemplate(
        name="chatml",
        system_prompt="You are a helpful assistant.",
        user_header="<|im_start|>user\n",
        assistant_header="<|im_start|>assistant\n",
        end_of_turn_token="<|im_end|>\n",
        bos_token="",
        eos_token="<|im_end|>",
    ),
    "mistral": ChatTemplate(
        name="mistral",
        system_prompt="",
        user_header="[INST] ",
        assistant_header="",
        end_of_turn_token=" [/INST]",
        bos_token="<s>",
        eos_token="</s>",
    ),
}


def format_conversation(
    conversation: List[Dict[str, str]],
    template: ChatTemplate,
) -> str:
    """
    Format a ShareGPT-style conversation using a chat template.

    Args:
        conversation: List of messages with 'from'/'role' and 'value'/'content' fields
        template: The chat template to use

    Returns:
        Formatted conversation string
    """
    formatted = ""

    # Add BOS token if present
    if template.bos_token:
        formatted += template.bos_token

    # Add system prompt (for some templates)
    if template.system_prompt:
        formatted += f"{template.assistant_header}{template.system_prompt}{template.end_of_turn_token}"

    # Format each turn
    for turn in conversation:
        role = turn.get("from", turn.get("role", ""))
        content = turn.get("value", turn.get("content", ""))

        if role in ["user", "human"]:
            formatted += f"{template.user_header}{content}{template.end_of_turn_token}"
        elif role in ["assistant", "gpt", "model"]:
            formatted += f"{template.assistant_header}{content}{template.end_of_turn_token}"

    return formatted


# ============================================================================
# Dataset Configurations
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for loading a dataset"""
    name: str
    hf_path: str
    split: str
    normalize_fn: Optional[Callable[[Dict], Dict]] = None


def _normalize_ultrachat(example: Dict) -> Dict:
    """Convert ultrachat 'messages' field to 'conversations'"""
    if 'messages' in example:
        return {'conversations': example['messages']}
    return example


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    'sharegpt': DatasetConfig(
        name='sharegpt',
        hf_path='Aeala/ShareGPT_Vicuna_unfiltered',
        split='train',
    ),
    'ultrachat': DatasetConfig(
        name='ultrachat',
        hf_path='HuggingFaceH4/ultrachat_200k',
        split='train',
        normalize_fn=_normalize_ultrachat,
    ),
}
