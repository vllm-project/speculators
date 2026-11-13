import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples/quantization/dspark_nvfp4_gptq_imatrix.py"
)


def test_nvfp4_recipe_uses_strict_imatrix_mse(monkeypatch):
    preset = Mock()
    modifier = Mock()
    quantization = SimpleNamespace(preset_name_to_scheme=preset)
    oneshot = SimpleNamespace(Oneshot=object)
    gptq = SimpleNamespace(GPTQModifier=modifier)
    entrypoints = SimpleNamespace(__path__=[], oneshot=oneshot)
    modifiers = SimpleNamespace(__path__=[], gptq=gptq)
    stubs = {
        "compressed_tensors": SimpleNamespace(__path__=[], quantization=quantization),
        "compressed_tensors.quantization": quantization,
        "llmcompressor": SimpleNamespace(
            __path__=[], entrypoints=entrypoints, modifiers=modifiers
        ),
        "llmcompressor.entrypoints": entrypoints,
        "llmcompressor.entrypoints.oneshot": oneshot,
        "llmcompressor.modifiers": modifiers,
        "llmcompressor.modifiers.gptq": gptq,
    }
    scheme = SimpleNamespace(weights=SimpleNamespace())
    preset.return_value = scheme
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    with patch.dict(sys.modules, stubs):
        module = runpy.run_path(str(SCRIPT))
        assert module["_nvfp4_recipe"]() is modifier.return_value

    preset.assert_called_once_with("NVFP4", targets=["Linear"])
    assert scheme.weights.observer == "imatrix_mse"
    assert scheme.weights.observer_kwargs == {"strict": True}
    kwargs = modifier.call_args.kwargs
    assert kwargs["config_groups"] == {"group_0": scheme}
    assert kwargs["ignore"] is module["IGNORE"]
    assert "targets" not in kwargs
