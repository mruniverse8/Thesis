import json
from types import SimpleNamespace

import pytest
import torch

import post_training.sft_multi.trainer as trainer_module


class DummyTokenizer:
    def __len__(self) -> int:
        return 321


class DummyTracker:
    def __init__(self) -> None:
        self.config_payloads: list[dict] = []
        self.metric_calls: list[tuple[dict, int | None, str | None]] = []
        self.summary_payloads: list[tuple[dict, str | None]] = []
        self.status: str | None = None

    def log_config(self, payload) -> None:
        self.config_payloads.append(dict(payload))

    def log_metrics(self, payload, *, step=None, prefix=None) -> None:
        self.metric_calls.append((dict(payload), step, prefix))

    def log_summary(self, payload, *, prefix=None) -> None:
        self.summary_payloads.append((dict(payload), prefix))

    def finish(self, *, status) -> None:
        self.status = status


class DummyProgressBar:
    def __init__(self, items) -> None:
        self._items = list(items)
        self.postfixes: list[dict[str, str]] = []

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def set_postfix(self, **kwargs) -> None:
        self.postfixes.append(kwargs)


class DummyModel(torch.nn.Module):
    def __init__(self, losses: list[float]) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self._losses = iter(losses)

    def forward(self, **kwargs):
        del kwargs
        loss_value = next(self._losses)
        return SimpleNamespace(loss=(self.weight * 0) + self.weight.new_tensor(loss_value))


class DummyT5:
    model = DummyModel([4.0, 2.0, 1.0])

    @classmethod
    def from_pretrained(cls, name: str) -> DummyModel:
        assert name == "demo-model"
        return cls.model


class DummyScheduler:
    def __init__(self) -> None:
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1

    def get_last_lr(self) -> list[float]:
        return [0.01 * max(self.step_count, 1)]


def test_run_multi_molecule_sft_logs_step_and_epoch_metrics(monkeypatch, tmp_path) -> None:
    tracker = DummyTracker()
    history_snapshots: list[list[dict]] = []
    checkpoint_calls: list[dict] = []
    progress_bars: list[DummyProgressBar] = []

    train_loader = [{"input_ids": torch.ones(2, 1, dtype=torch.long)} for _ in range(3)]
    validation_loader = [{"input_ids": torch.ones(2, 1, dtype=torch.long)}]
    loaders = iter((train_loader, validation_loader))

    def fake_tqdm(items, **kwargs):
        del kwargs
        progress_bar = DummyProgressBar(items)
        progress_bars.append(progress_bar)
        return progress_bar

    monkeypatch.setattr(trainer_module, "prepare_sft_tokenizers", lambda config: (DummyTokenizer(), {}))
    monkeypatch.setattr(trainer_module, "assert_tokenizer_matches_model_vocab", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer_module, "T5ForConditionalGeneration", DummyT5)
    monkeypatch.setattr(trainer_module, "build_multi_molecule_dataloader", lambda *args, **kwargs: next(loaders))
    monkeypatch.setattr(trainer_module, "choose_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(trainer_module, "resolve_mixed_precision", lambda *args, **kwargs: "no")
    monkeypatch.setattr(trainer_module, "move_tensor_batch_to_device", lambda batch, device: batch)
    monkeypatch.setattr(trainer_module, "evaluate_loss", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(
        trainer_module,
        "get_linear_schedule_with_warmup",
        lambda *args, **kwargs: DummyScheduler(),
    )
    monkeypatch.setattr(trainer_module, "prepare_sft_output_dir", lambda *args, **kwargs: tmp_path)
    monkeypatch.setattr(trainer_module, "build_tracker", lambda *args, **kwargs: tracker)
    monkeypatch.setattr(
        trainer_module,
        "write_sft_history",
        lambda output_dir, history: history_snapshots.append([dict(item) for item in history]),
    )
    monkeypatch.setattr(
        trainer_module,
        "save_sft_checkpoint",
        lambda **kwargs: checkpoint_calls.append(dict(kwargs)),
    )
    monkeypatch.setattr(trainer_module, "clip_grad_norm_", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer_module, "tqdm", fake_tqdm)

    summary = trainer_module.run_multi_molecule_sft(
        {
            "seed": 7,
            "model": {"name": "demo-model"},
            "data": {
                "train_file": "train.jsonl",
                "validation_file": "validation.jsonl",
                "max_source_length": 16,
                "max_target_length": 16,
                "num_workers": 0,
            },
            "training": {
                "output_dir": str(tmp_path),
                "device": "cpu",
                "mixed_precision": "no",
                "per_device_train_batch_size": 2,
                "per_device_eval_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "num_epochs": 1,
                "learning_rate": 1.0e-3,
                "weight_decay": 0.0,
                "warmup_ratio": 0.0,
                "max_grad_norm": 1.0,
                "log_every": 2,
                "save_every_epochs": 1,
            },
        }
    )

    assert tracker.status == "success"
    assert [step for _, step, _ in tracker.metric_calls] == [2, 3, 3]
    assert [prefix for _, _, prefix in tracker.metric_calls] == ["sft", "sft", "sft"]

    first_step_metrics, second_step_metrics, epoch_metrics = [
        payload for payload, _, _ in tracker.metric_calls
    ]
    assert first_step_metrics["epoch"] == 1
    assert first_step_metrics["global_step"] == 2
    assert first_step_metrics["train_loss_step"] == pytest.approx(3.0)
    assert first_step_metrics["learning_rate"] == pytest.approx(0.02)

    assert second_step_metrics["epoch"] == 1
    assert second_step_metrics["global_step"] == 3
    assert second_step_metrics["train_loss_step"] == pytest.approx(1.0)
    assert second_step_metrics["learning_rate"] == pytest.approx(0.03)

    assert epoch_metrics["epoch"] == 1
    assert epoch_metrics["global_step"] == 3
    assert epoch_metrics["train_loss_epoch"] == pytest.approx(7.0 / 3.0)
    assert epoch_metrics["validation_loss"] == pytest.approx(0.5)
    assert epoch_metrics["learning_rate"] == pytest.approx(0.03)

    assert progress_bars[0].postfixes == [{"train_loss": "3.0000", "lr": "2.00e-02"}]
    assert history_snapshots == [
        [
            {
                "epoch": 1,
                "global_step": 3,
                "train_loss": pytest.approx(7.0 / 3.0),
                "validation_loss": 0.5,
                "learning_rate": 0.03,
            }
        ]
    ]
    assert [call["checkpoint_dir"].name for call in checkpoint_calls] == ["last", "epoch-01", "best"]
    assert checkpoint_calls[-1]["create_archive"] is True

    assert summary["best_validation_loss"] == pytest.approx(0.5)
    assert summary["history"][0]["train_loss"] == pytest.approx(7.0 / 3.0)

    run_summary_payload = json.loads((tmp_path / "run_summary.json").read_text())
    assert run_summary_payload["best_validation_loss"] == pytest.approx(0.5)
    assert run_summary_payload["history"][0]["train_loss"] == pytest.approx(7.0 / 3.0)
