"""End-to-end tests for backend arg auto-registration.

Verifies that backend-contributed train args survive the full
``resolve -> flatten`` and ``dump_yaml -> --config`` pipelines.
"""

import textwrap

from speculators.train.config.schema import TrainConfig


def test_mooncake_args_survive_from_sources():
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m", "hidden_states_backend": "mooncake"},
        backend_cli={"mooncake_master": "10.0.0.1:50051"},
        argv=["train.py"],
    )
    flat = cfg.flatten()
    assert flat["mooncake_master"] == "10.0.0.1:50051"
    assert flat["mooncake_protocol"] == "tcp"  # default backfilled


def test_mooncake_defaults_backfilled_for_selected_backend():
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m", "hidden_states_backend": "mooncake"},
        argv=["train.py"],
    )
    flat = cfg.flatten()
    assert flat["mooncake_master"] == "127.0.0.1:50051"
    assert flat["mooncake_metadata_server"] == "P2PHANDSHAKE"
    assert flat["mooncake_protocol"] == "tcp"
    assert flat["mooncake_global_segment_gib"] == 4.0
    assert flat["mooncake_local_buffer_gib"] == 2.0


def test_file_backend_defaults_backfilled():
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m", "hidden_states_backend": "file"},
        argv=["train.py"],
    )
    flat = cfg.flatten()
    assert flat["hidden_states_path"] is None  # FileBackend default


def test_unselected_backend_args_not_backfilled():
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m", "hidden_states_backend": "file"},
        argv=["train.py"],
    )
    flat = cfg.flatten()
    assert "mooncake_master" not in flat


def test_backend_args_provenance():
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m", "hidden_states_backend": "mooncake"},
        backend_cli={"mooncake_master": "10.0.0.1:50051"},
        argv=["train.py"],
    )
    assert cfg.provenance["mooncake_master"] == "flag"
    assert "mooncake_protocol" not in cfg.provenance  # default, no provenance entry


def test_backend_args_in_yaml_roundtrip(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        textwrap.dedent("""\
        train:
          verifier:
            verifier_name_or_path: m
          data:
            hidden_states_backend: mooncake
          backend:
            mooncake_master: "10.0.0.1:50051"
            mooncake_protocol: rdma
        """)
    )

    cfg = TrainConfig.from_sources(
        cli={}, config_path=str(yaml_file), argv=["train.py"]
    )
    flat = cfg.flatten()
    assert flat["mooncake_master"] == "10.0.0.1:50051"
    assert flat["mooncake_protocol"] == "rdma"
    assert cfg.provenance["mooncake_master"] == "yaml"
    assert cfg.provenance["mooncake_protocol"] == "yaml"


def test_backend_cli_beats_yaml(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        textwrap.dedent("""\
        train:
          verifier:
            verifier_name_or_path: m
          data:
            hidden_states_backend: mooncake
          backend:
            mooncake_master: "yaml-addr:50051"
        """)
    )

    cfg = TrainConfig.from_sources(
        cli={},
        config_path=str(yaml_file),
        backend_cli={"mooncake_master": "cli-addr:50051"},
        argv=["train.py"],
    )
    assert cfg.flatten()["mooncake_master"] == "cli-addr:50051"
    assert cfg.provenance["mooncake_master"] == "flag"


def test_dump_yaml_includes_backend_block():
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m", "hidden_states_backend": "mooncake"},
        backend_cli={"mooncake_master": "10.0.0.1:50051"},
        argv=["train.py"],
    )
    yaml_str = cfg.dump_yaml()
    assert "backend:" in yaml_str
    assert "mooncake_master" in yaml_str


def test_resolve_with_mooncake_args():
    cfg = TrainConfig.resolve(
        [
            "--verifier-name-or-path",
            "m",
            "--hidden-states-backend",
            "mooncake",
            "--mooncake-master",
            "10.0.0.1:50051",
        ]
    )
    flat = cfg.flatten()
    assert flat["mooncake_master"] == "10.0.0.1:50051"
    assert flat["mooncake_protocol"] == "tcp"


def test_resolve_file_backend_hidden_states_path():
    cfg = TrainConfig.resolve(
        [
            "--verifier-name-or-path",
            "m",
            "--hidden-states-path",
            "/data/hs",
        ]
    )
    flat = cfg.flatten()
    assert flat["hidden_states_path"] == "/data/hs"


def test_from_flat_roundtrip_with_backend_args():
    cfg = TrainConfig.from_sources(
        cli={"verifier_name_or_path": "m", "hidden_states_backend": "mooncake"},
        backend_cli={"mooncake_master": "10.0.0.1:50051"},
        argv=["train.py"],
    )
    flat = cfg.flatten()
    cfg2 = TrainConfig.from_flat(flat)
    assert cfg2.flatten()["mooncake_master"] == "10.0.0.1:50051"
