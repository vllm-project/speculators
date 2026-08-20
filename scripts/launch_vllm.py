import argparse
import json
import os
import sys
import warnings

try:
    from hs_connectors import HiddenStatesBackend

    _backend_registry: dict[str, type[HiddenStatesBackend]] = dict(
        HiddenStatesBackend.registry  # type: ignore[misc]
    )
except ImportError:
    _backend_registry = {}  # type: ignore[assignment]


if "file" not in _backend_registry:
    # Vendored File backend in case hs_connectors is not available
    class _InlineFileBackend:
        @staticmethod
        def add_launch_args(parser: argparse.ArgumentParser) -> None:
            parser.add_argument(
                "--hidden-states-path",
                type=str,
                default="/tmp/hidden_states",  # noqa: S108
                help=(
                    "The directory to save hidden states to. "
                    "Default '/tmp/hidden_states'"
                ),
            )

        @staticmethod
        def build_kv_transfer_config(args: argparse.Namespace) -> dict:
            return {
                "kv_connector": "ExampleHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "shared_storage_path": args.hidden_states_path,
                },
            }

    _backend_registry["file"] = _InlineFileBackend  # type: ignore[assignment]


# Front-end throughput defaults, applied unless the flag is passed after "--".
# prepare_data drives /v1/chat/completions/render with ~2 requests per
# assistant turn; vLLM's stock front end handles them in one API process whose
# renderer pool is a single thread (--renderer-num-workers defaults to 1), so
# the render stage serializes there no matter how many client workers run.
# These flags only scale the HTTP front end (template + tokenize); the engine
# and hidden-states connector are untouched.
#
# Sizing, measured on an H100 node (2x EPYC 9654, 384 CPUs), Qwen3-0.6B.
#
# The front end is sized against the client, not the machine: prepare_data
# renders from --num-preprocessing-workers forked processes, each blocking on
# one call at a time, so that worker count is how many renders are ever in
# flight. Both sides therefore derive from the same CPU count and stay paired
# without this script having to read prepare_data's setting -- it cannot, since
# it runs inside the vLLM virtualenv where speculators is not installed
# (see #958 / #1008). Keep _preprocessing_workers below in sync with
# speculators.data_generation.preprocessing.default_preprocessing_workers.
#
#   8 workers  +  4x2  ->  1,702 renders/s   (vLLM stock 1x1: 633)
#  64 workers  + 16x2  ->  8,997
# 128 workers  + 32x2  -> 16,878             <- where 384 CPUs lands
# 256 workers  + 32x2  -> 15,400             (past the knee)
#
# Raising either side alone is worse than raising both: at 8 workers a 16x1
# front end measured 1,586 renders/s against 1,702 for 4x2, at three times the
# memory. How the slots are split matters too -- 8 threads in one process reach
# only 1.2x stock, because chat-template rendering is Python and serializes on
# the GIL, while 8 single-threaded processes cost ~1.2 GB RSS each and still
# trail 4 processes of 2 threads.
RENDERER_THREADS_PER_SERVER = 2
WORKERS_PER_API_SERVER = 4
MAX_API_SERVERS = 32
CPUS_PER_API_SERVER = 4
# Mirrors of the preprocessing-side constants; see the sync note above.
MIN_PREPROCESSING_WORKERS = 8
MAX_PREPROCESSING_WORKERS = 128
CPUS_PER_PREPROCESSING_WORKER = 3


def _usable_cpu_count() -> int:
    """Number of CPUs this process may actually run on.

    Not ``os.cpu_count()``: that reports the whole machine and ignores CPU
    affinity, so a datagen job confined to a few cores of a large node would
    still start a front end per idle core it cannot use.
    """
    if hasattr(os, "process_cpu_count"):  # Python 3.13+
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):  # Linux
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _preprocessing_workers(cpus: int) -> int:
    """What prepare_data will pick for --num-preprocessing-workers here."""
    return max(
        MIN_PREPROCESSING_WORKERS,
        min(MAX_PREPROCESSING_WORKERS, cpus // CPUS_PER_PREPROCESSING_WORKER),
    )


def render_throughput_defaults(cpus: int | None = None) -> dict[str, str]:
    """Front-end sizing for this host.

    One API server per WORKERS_PER_API_SERVER renders in flight, so the front
    end grows with the clients rather than with the machine, and never past what
    the local cores can run: on a small host extra front ends only take cores
    away from the preprocessing workers.
    """
    if cpus is None:
        cpus = _usable_cpu_count()
    api_servers = max(
        1,
        min(
            MAX_API_SERVERS,
            _preprocessing_workers(cpus) // WORKERS_PER_API_SERVER,
            cpus // CPUS_PER_API_SERVER,
        ),
    )
    return {
        "--api-server-count": str(api_servers),
        "--renderer-num-workers": str(RENDERER_THREADS_PER_SERVER),
    }


def _with_render_defaults(vllm_args: list[str]) -> list[str]:
    """Prepend the render throughput defaults for flags the caller did not set.

    Defaults go before ``vllm_args`` so an explicit flag always wins (argparse
    keeps the last occurrence), even for spellings the presence check misses.
    """
    extra: list[str] = []
    for flag, value in render_throughput_defaults().items():
        # vLLM's parser accepts dash and underscore spellings, plus "=" form.
        names = {flag, "--" + flag[2:].replace("-", "_")}
        if any(
            arg == name or arg.startswith(f"{name}=")
            for arg in vllm_args
            for name in names
        ):
            continue
        extra += [flag, value]
    return [*extra, *vllm_args]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch vLLM for hidden states extraction",
        usage=(
            "launch_vllm.py [-h] MODEL [--hidden-states-backend BACKEND] "
            "[--target-layer-ids TARGET_LAYER_IDS [TARGET_LAYER_IDS ...]] -- *VLLM_ARGS"
        ),
    )
    parser.add_argument(
        "model", type=str, help="Model name or path to extract hidden states from"
    )

    parser.add_argument(
        "--hidden-states-backend",
        choices=list(_backend_registry.keys()),
        default="file",
        help=(
            "Hidden states transfer backend. Each backend may add its own "
            "CLI arguments (see below). Default: 'file'."
        ),
    )
    for backend_cls in _backend_registry.values():
        backend_cls.add_launch_args(parser)

    parser.add_argument(
        "--target-layer-ids",
        type=int,
        nargs="+",
        help=(
            "(Optional) A (space separated) list of integer layer ids. Defaults to "
            "[2, num_hidden_layers // 2, num_hidden_layers - 3]. "
            "Note: if set, you must also pass the same value into the training process"
        ),
    )
    parser.add_argument(
        "--include-last-layer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Append the last layer (num_hidden_layers) to "
            "target_layer_ids for verifier hidden states extraction. Default: True"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be executed without running it",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Allow custom model configuration code while resolving hidden-state "
            "layer ids. Pass the same flag after '--' for vLLM itself."
        ),
    )
    return parser.parse_known_args()


def main():
    args, vllm_args = parse_args()
    if "--" in vllm_args:
        vllm_args.remove("--")

    from transformers import AutoConfig  # noqa: PLC0415

    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if hasattr(config, "text_config"):
        config = config.text_config
    num_hidden_layers = config.num_hidden_layers

    if args.target_layer_ids:
        target_layer_ids = args.target_layer_ids
        if args.include_last_layer and num_hidden_layers not in target_layer_ids:
            target_layer_ids.append(num_hidden_layers)
        warnings.warn(
            f"Using custom target layer ids {target_layer_ids}. These "
            "must also be explicitly passed into the training script.",
            stacklevel=2,
        )
    else:
        target_layer_ids = [
            2,
            num_hidden_layers // 2,
            num_hidden_layers - 3,
            num_hidden_layers,
        ]
    # Layer id ``num_hidden_layers`` (the final hidden state) is valid: the
    # default above and --include-last-layer both emit it.
    if (
        min(target_layer_ids) < 0
        or max(target_layer_ids) > num_hidden_layers
        or len(set(target_layer_ids)) != len(target_layer_ids)
    ):
        raise ValueError(
            f"Invalid target layer ids {target_layer_ids}; ids must be "
            f"distinct and within [0, {num_hidden_layers}]."
        )

    speculative_config = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {"eagle_aux_hidden_state_layer_ids": target_layer_ids}
        },
    }
    backend_cls = _backend_registry[args.hidden_states_backend]
    kv_transfer_config = backend_cls.build_kv_transfer_config(args)

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--speculative_config",
        json.dumps(speculative_config),
        "--kv_transfer_config",
        json.dumps(kv_transfer_config),
        *_with_render_defaults(vllm_args),
    ]

    print("Running command:")
    print(" ".join(cmd))

    if not args.dry_run:
        os.execvp(cmd[0], cmd)  # noqa: S606


if __name__ == "__main__":
    main()
