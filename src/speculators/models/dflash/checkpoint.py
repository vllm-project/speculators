"""DFlash-family checkpoint names, independent of physical module ownership."""

from copy import copy, deepcopy
from typing import ClassVar

from speculators.model import SpeculatorModel

_BACKBONE_PREFIXES = {
    name: f"backbone.{name}"
    for name in ("layers", "fc", "hidden_norm", "norm", "rotary_emb")
}


def _rename_key(key: str, prefixes: dict[str, str]) -> str:
    for source, target in prefixes.items():
        prefix = f"{source}."
        if key.startswith(prefix):
            return f"{target}.{key.removeprefix(prefix)}"
    return key


def _validate_checkpoint_keys(missing, unexpected):
    if missing or unexpected:
        raise ValueError(
            f"Invalid draft checkpoint: missing learned weights {sorted(missing)}; "
            f"unexpected weights {sorted(unexpected)}"
        )


class DFlashCheckpointModel(SpeculatorModel):
    """Adapt DFlash-family checkpoint names at serialization boundaries."""

    hierarchical_checkpoint: ClassVar[bool] = False
    legacy_correction_name: ClassVar[str | None] = None

    @classmethod
    def _checkpoint_prefixes(cls) -> dict[str, str]:
        if cls.hierarchical_checkpoint:
            return {}
        prefixes = dict(_BACKBONE_PREFIXES)
        if cls.legacy_correction_name is not None:
            prefixes[cls.legacy_correction_name] = "correction"
        return prefixes

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        mapping = {
            rf"^{source}\.": f"{target}."
            for source, target in cls._checkpoint_prefixes().items()
        }
        mapping.update(kwargs.pop("key_mapping", {}) or {})
        return super().from_pretrained(*args, key_mapping=mapping, **kwargs)

    @staticmethod
    def _validate_pretrained_loading_info(loading_info):
        _validate_checkpoint_keys(
            loading_info["missing_keys"], loading_info["unexpected_keys"]
        )

    @classmethod
    def _save_prefixes(cls) -> dict[str, str]:
        return {target: source for source, target in cls._checkpoint_prefixes().items()}

    def prepare_checkpoint_state_dict(self, state_dict):
        """Adapt and validate weights before single-device or FSDP resume."""
        save_prefixes = self._save_prefixes()
        expected = {_rename_key(name, save_prefixes) for name in self.state_dict()}
        allowed_missing = set(self._keys_to_ignore_on_load_missing)
        keys = state_dict.keys()
        _validate_checkpoint_keys(
            expected - keys - allowed_missing,
            keys - expected,
        )
        prefixes = self._checkpoint_prefixes()
        if not prefixes:
            return state_dict
        return {
            _rename_key(name, prefixes): tensor for name, tensor in state_dict.items()
        }

    def save_pretrained(
        self,
        save_directory,
        is_main_process=True,
        state_dict=None,
        push_to_hub=False,
        max_shard_size="50GB",
        variant=None,
        token=None,
        save_peft_format=True,
        save_original_format=True,  # noqa: ARG002 - model type owns conversion
        distributed_checkpoint=False,
        **kwargs,
    ):
        """Save weights in the layout assigned to this model family."""
        if "checkpoint_format" in kwargs:
            raise ValueError("Checkpoint layout is fixed by the model type")
        state_dict = self.state_dict() if state_dict is None else state_dict
        save_prefixes = self._save_prefixes()
        exported = {
            _rename_key(name, save_prefixes): tensor
            for name, tensor in state_dict.items()
        }
        # Transformers updates config metadata while writing the checkpoint.
        view = copy(self)
        view.config = deepcopy(self.config)
        return super(DFlashCheckpointModel, view).save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            state_dict=exported,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            save_original_format=False,
            distributed_checkpoint=distributed_checkpoint,
            **kwargs,
        )
