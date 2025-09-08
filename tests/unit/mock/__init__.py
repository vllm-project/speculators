from .hf_factory import (
    MockPretrainedTransformersFactory,
    PretrainedBundle,
    mock_llama3_2m_config_dict,
    mock_llama3_2m_state_dict,
)
from .safeai_eagle import mock_eagle3_config_dict, mock_eagle3_state_dict

__all__ = [
    "MockPretrainedTransformersFactory",
    "PretrainedBundle",
    "mock_eagle3_config_dict",
    "mock_eagle3_state_dict",
    "mock_llama3_2m_config_dict",
    "mock_llama3_2m_state_dict",
]

# Expose all sub-plugins within the mock package to pytest
pytest_plugins = [
    "tests.unit.mock.hf_factory",
    "tests.unit.mock.safeai_eagle",
]
