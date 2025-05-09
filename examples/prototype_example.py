from torch.nn import Module

from speculators import SpeculatorModel


def from_other_library_file_example():
    source = "path/to/other/library/model"
    speculator = SpeculatorModel.from_pretrained(source)
    speculator.save("path/to/save/speculator_model", push_to_hub=True)


def from_other_library_model_example():
    model = Module()
    config = {} or "path/to/config.json"

    speculator = SpeculatorModel.from_pretrained(model, config)
    speculator.save("path/to/save/speculator_model", push_to_hub=True)


def from_speculators_example():
    source = "path/to/speculators/model"  # {config.json, safetensors}
    speculator = SpeculatorModel.from_pretrained(source)
    speculator.save("path/to/save/speculator_model", push_to_hub=True)


def from_scratch_config_example():
    config = {}
    speculator = SpeculatorModel.from_config(config)
    speculator.save("path/to/save/speculator_model", push_to_hub=True)


def create_from_scratch_example():
    speculator = Eagle3Speculator(config={})
    speculator.save("path/to/save/speculator_model", push_to_hub=True)


def lifecycle_example():
    speculator = SpeculatorModel.from_pretrained("path/to/speculator_model")
    speculator.attach_verifier(
        PreTrainedModel.from_pretrained("path/to/verifier_model")
    )
    tokenzier = PreTrainedTokenizer.from_pretrained("path/to/verifier_model")
    input_ids = tokenzier("Hello, how are you?", return_tensors="pt").input_ids
    outputs = speculator.generate(input_ids)
    print(outputs)
