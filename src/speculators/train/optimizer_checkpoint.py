"""Validate optimizer parameter identities across model layout migrations."""

import json
from pathlib import Path


def optimizer_metadata(model, optimizers, *, distributed):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return {
        "version": 1,
        "distributed": distributed,
        "optimizer_types": [
            f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
            for optimizer in optimizers
        ],
        "parameter_groups": [
            [
                [names[id(parameter)] for parameter in group["params"]]
                for group in optimizer.param_groups
            ]
            for optimizer in optimizers
        ],
    }


def save_optimizer_metadata(path: Path, model, optimizers, *, distributed):
    if getattr(model, "requires_optimizer_metadata", False):
        path.write_text(
            json.dumps(
                optimizer_metadata(model, optimizers, distributed=distributed), indent=2
            )
            + "\n"
        )


def _verify_groups(groups, expected_names, name_field, normalize):
    actual_names = [group.get(name_field) for group in groups]
    if any(
        names is None or any(not isinstance(name, str) for name in names)
        for names in actual_names
    ):
        raise ValueError(
            "Cannot safely resume an unnamed legacy optimizer checkpoint after "
            "backbone composition. Initialize weights with --from-pretrained "
            "instead of resuming this training state."
        )
    mapped = [[normalize(name) for name in names] for names in actual_names]
    if mapped != expected_names:
        raise ValueError("Legacy optimizer parameter identities or ordering differ")
    return [
        dict(group, **{name_field: names})
        for group, names in zip(groups, mapped, strict=True)
    ]


def prepare_optimizer_state(path: Path, model, optimizers, state, *, distributed):
    """Accept verified identities, never infer positional identity from shapes.

    New checkpoints carry metadata even for optimizers built without names.
    Legacy named checkpoints can migrate in place without copying tensor data.
    Unnamed legacy states require an explicit weights-only initialization.
    """
    if not getattr(model, "requires_optimizer_metadata", False):
        return state
    expected = optimizer_metadata(model, optimizers, distributed=distributed)
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(
                "Optimizer checkpoint metadata does not match the model, optimizer "
                "types, or parameter group ordering. Initialize weights with "
                "--from-pretrained instead of resuming this training state."
            )
        return state

    normalize = model.checkpoint_parameter_name
    expected_groups = expected["parameter_groups"]

    if distributed:
        groups = _verify_groups(
            state["param_groups"],
            [
                group
                for optimizer_groups in expected_groups
                for group in optimizer_groups
            ],
            "params",
            normalize,
        )
        mapped_state = {}
        allowed = {name for group in groups for name in group["params"]}
        for name, value in state["state"].items():
            destination = normalize(name)
            if destination in mapped_state or destination not in allowed:
                raise ValueError(f"Invalid optimizer parameter identity: {name!r}")
            mapped_state[destination] = value
        return dict(state, state=mapped_state, param_groups=groups)

    states = state if isinstance(state, list) else [state]
    if len(states) != len(expected_groups):
        raise ValueError("Legacy checkpoint optimizer count differs")
    prepared = [
        dict(
            item,
            param_groups=_verify_groups(
                item["param_groups"], groups, "param_names", normalize
            ),
        )
        for item, groups in zip(states, expected_groups, strict=True)
    ]
    return prepared if isinstance(state, list) else prepared[0]
