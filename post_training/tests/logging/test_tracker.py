from __future__ import annotations

import pytest

from post_training.logging import NullTracker, TrackingConfig, build_tracker, resolve_tracking_config
from post_training.logging import tracker as tracker_module


class DummyWandbConfig(dict):
    def update(self, payload, allow_val_change=False):
        self["allow_val_change"] = allow_val_change
        super().update(payload)


class DummyWandbRun:
    def __init__(self) -> None:
        self.config = DummyWandbConfig()
        self.summary: dict[str, object] = {}
        self.logged: list[tuple[dict[str, object], int | None]] = []
        self.exit_code: int | None = None

    def log(self, payload, step=None):
        self.logged.append((payload, step))

    def finish(self, exit_code=None):
        self.exit_code = exit_code


class DummyWandbModule:
    def __init__(self) -> None:
        self.init_calls: list[dict[str, object]] = []
        self.run = DummyWandbRun()

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.run


class DummyCometExperiment:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.parameters: list[dict[str, object]] = []
        self.metrics: list[tuple[dict[str, object], int | None]] = []
        self.others: list[tuple[str, object]] = []
        self.tags: list[str] = []
        self.name: str | None = None
        self.ended = False

    def set_name(self, name: str) -> None:
        self.name = name

    def add_tags(self, tags: list[str]) -> None:
        self.tags.extend(tags)

    def log_parameters(self, payload):
        self.parameters.append(payload)

    def log_metrics(self, payload, step=None):
        self.metrics.append((payload, step))

    def log_other(self, key, value):
        self.others.append((key, value))

    def end(self):
        self.ended = True


class DummyCometModule:
    def __init__(self) -> None:
        self.instances: list[DummyCometExperiment] = []

    def Experiment(self, **kwargs):
        instance = DummyCometExperiment(**kwargs)
        self.instances.append(instance)
        return instance


def test_resolve_tracking_config_defaults_to_disabled() -> None:
    config = resolve_tracking_config({})

    assert config == TrackingConfig()


def test_resolve_tracking_config_supports_logging_alias() -> None:
    config = resolve_tracking_config(
        {
            "logging": {
                "enabled": True,
                "backend": "wandb",
                "project": "demo-project",
                "tags": ["sft"],
            }
        }
    )

    assert config.enabled is True
    assert config.backend == "wandb"
    assert config.project == "demo-project"
    assert config.tags == ("sft",)


def test_resolve_tracking_config_rejects_duplicate_sections() -> None:
    with pytest.raises(ValueError, match="Use only one of 'tracking' or 'logging'"):
        resolve_tracking_config(
            {
                "tracking": {"enabled": False},
                "logging": {"enabled": False},
            }
        )


def test_build_tracker_returns_null_tracker_when_disabled(tmp_path) -> None:
    tracker = build_tracker({}, stage_name="multi_molecule_sft", output_dir=tmp_path / "run")

    assert isinstance(tracker, NullTracker)


def test_build_tracker_rejects_unknown_backend(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported tracking backend"):
        build_tracker(
            {
                "tracking": {
                    "enabled": True,
                    "backend": "unknown",
                    "project": "demo-project",
                }
            },
            stage_name="multi_molecule_sft",
            output_dir=tmp_path / "run",
        )


def test_build_tracker_raises_when_wandb_package_is_missing(monkeypatch, tmp_path) -> None:
    def import_module(name: str):
        raise ImportError(name)

    monkeypatch.setattr(tracker_module.importlib, "import_module", import_module)

    with pytest.raises(RuntimeError, match="wandb"):
        build_tracker(
            {
                "tracking": {
                    "enabled": True,
                    "backend": "wandb",
                    "project": "demo-project",
                }
            },
            stage_name="multi_molecule_sft",
            output_dir=tmp_path / "run",
        )


def test_build_tracker_raises_when_comet_package_is_missing(monkeypatch, tmp_path) -> None:
    def import_module(name: str):
        raise ImportError(name)

    monkeypatch.setattr(tracker_module.importlib, "import_module", import_module)

    with pytest.raises(RuntimeError, match="comet-ml"):
        build_tracker(
            {
                "tracking": {
                    "enabled": True,
                    "backend": "comet",
                    "project": "demo-project",
                }
            },
            stage_name="molecule_wise_ppo",
            output_dir=tmp_path / "run",
        )


def test_wandb_tracker_logs_config_metrics_and_summary(monkeypatch, tmp_path) -> None:
    wandb_module = DummyWandbModule()

    def import_module(name: str):
        assert name == "wandb"
        return wandb_module

    monkeypatch.setattr(tracker_module.importlib, "import_module", import_module)

    tracker = build_tracker(
        {
            "seed": 7,
            "tracking": {
                "enabled": True,
                "backend": "wandb",
                "project": "demo-project",
                "workspace": "demo-team",
                "tags": ["baseline"],
            },
        },
        stage_name="multi_molecule_sft",
        output_dir=tmp_path / "sft_run",
    )

    tracker.log_config({"resolved_config": {"training": {"num_epochs": 3}}})
    tracker.log_metrics({"train_loss": 1.25, "epoch": 1, "label": "ignored"}, step=4, prefix="sft")
    tracker.log_summary({"output_dir": str(tmp_path / "sft_run")}, prefix="sft")
    tracker.finish(status="success")

    assert wandb_module.init_calls[0]["entity"] == "demo-team"
    assert wandb_module.init_calls[0]["name"] == "multi_molecule_sft-sft_run-seed7"
    assert wandb_module.init_calls[0]["tags"] == ["multi_molecule_sft", "baseline"]
    assert wandb_module.run.config["resolved_config"]["training"]["num_epochs"] == 3
    assert wandb_module.run.logged == [({"sft/train_loss": 1.25, "sft/epoch": 1}, 4)]
    assert wandb_module.run.summary["sft/output_dir"] == str(tmp_path / "sft_run")
    assert wandb_module.run.summary["run_status"] == "success"
    assert wandb_module.run.exit_code == 0


def test_comet_tracker_logs_config_metrics_and_summary(monkeypatch, tmp_path) -> None:
    comet_module = DummyCometModule()

    def import_module(name: str):
        assert name == "comet_ml"
        return comet_module

    monkeypatch.setattr(tracker_module.importlib, "import_module", import_module)

    tracker = build_tracker(
        {
            "seed": 11,
            "tracking": {
                "enabled": True,
                "backend": "comet",
                "project": "demo-project",
                "workspace": "demo-workspace",
                "run_name": "ppo-debug",
                "tags": ["debug"],
            },
        },
        stage_name="molecule_wise_ppo",
        output_dir=tmp_path / "ppo_run",
    )

    tracker.log_config({"resolved_config": {"ppo": {"ppo_iterations": 2}}})
    tracker.log_metrics({"mean_reward": 2.5, "iteration": 1, "status": "ignored"}, step=1, prefix="ppo")
    tracker.log_summary({"output_dir": str(tmp_path / "ppo_run")}, prefix="ppo")
    tracker.finish(status="failed")

    experiment = comet_module.instances[0]
    assert experiment.kwargs["project_name"] == "demo-project"
    assert experiment.kwargs["workspace"] == "demo-workspace"
    assert experiment.name == "ppo-debug"
    assert experiment.tags == ["molecule_wise_ppo", "debug"]
    assert experiment.parameters[0]["resolved_config.ppo.ppo_iterations"] == 2
    assert experiment.metrics == [({"ppo/mean_reward": 2.5, "ppo/iteration": 1}, 1)]
    assert ("ppo/output_dir", str(tmp_path / "ppo_run")) in experiment.others
    assert ("run_status", "failed") in experiment.others
    assert experiment.ended is True
