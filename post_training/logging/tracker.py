from __future__ import annotations

from collections.abc import Mapping
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _is_mapping(value: object) -> bool:
    return isinstance(value, Mapping)


def _normalize_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_tags(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("Tracking tags must be a list of strings.")
    tags: list[str] = []
    for item in value:
        text = _normalize_text(item)
        if text is None:
            continue
        tags.append(text)
    return tuple(tags)


def make_json_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_serializable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _prefix_payload_keys(payload: Mapping[str, Any], prefix: str | None) -> dict[str, Any]:
    if not prefix:
        return {str(key): value for key, value in payload.items()}
    return {f"{prefix}/{key}": value for key, value in payload.items()}


def _normalize_metric_payload(payload: Mapping[str, Any], prefix: str | None = None) -> dict[str, float | int]:
    normalized: dict[str, float | int] = {}
    for key, value in _prefix_payload_keys(payload, prefix).items():
        if isinstance(value, bool):
            normalized[key] = int(value)
            continue
        if isinstance(value, (int, float)):
            normalized[key] = value
    return normalized


def _flatten_payload(payload: Mapping[str, Any], prefix: str | None = None) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        key_text = str(key)
        full_key = f"{prefix}.{key_text}" if prefix else key_text
        if _is_mapping(value):
            flattened.update(_flatten_payload(value, prefix=full_key))
            continue
        serialized = make_json_serializable(value)
        if isinstance(serialized, (list, dict)):
            flattened[full_key] = json.dumps(serialized, ensure_ascii=False, sort_keys=True)
            continue
        flattened[full_key] = serialized
    return flattened


def _build_default_run_name(stage_name: str, output_dir: str | Path, seed: object | None) -> str:
    output_name = Path(output_dir).name or "run"
    seed_suffix = f"-seed{seed}" if seed is not None else ""
    return f"{stage_name}-{output_name}{seed_suffix}"


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    enabled: bool = False
    backend: str | None = None
    project: str | None = None
    workspace: str | None = None
    run_name: str | None = None
    tags: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "TrackingConfig":
        if payload is None:
            return cls()
        if not _is_mapping(payload):
            raise ValueError("Tracking config must be a mapping.")

        enabled = bool(payload.get("enabled", False))
        backend = _normalize_text(payload.get("backend"))
        if backend is not None:
            backend = backend.lower()
        project = _normalize_text(payload.get("project"))
        workspace = _normalize_text(payload.get("workspace"))
        run_name = _normalize_text(payload.get("run_name"))
        tags = _normalize_tags(payload.get("tags", ()))

        if enabled:
            if backend not in {"wandb", "comet"}:
                raise ValueError(
                    "Unsupported tracking backend. Expected one of: wandb, comet."
                )
            if project is None:
                raise ValueError("Tracking project is required when tracking is enabled.")

        return cls(
            enabled=enabled,
            backend=backend,
            project=project,
            workspace=workspace,
            run_name=run_name,
            tags=tags,
        )


def resolve_tracking_config(config: Mapping[str, Any]) -> TrackingConfig:
    tracking_payload = config.get("tracking")
    logging_payload = config.get("logging")
    if tracking_payload is not None and logging_payload is not None:
        raise ValueError("Use only one of 'tracking' or 'logging' in the config.")
    selected_payload = tracking_payload if tracking_payload is not None else logging_payload
    return TrackingConfig.from_payload(selected_payload)


class BaseTracker:
    def log_config(self, payload: Mapping[str, Any]) -> None:
        del payload

    def log_metrics(
        self,
        payload: Mapping[str, Any],
        *,
        step: int | None = None,
        prefix: str | None = None,
    ) -> None:
        del payload, step, prefix

    def log_summary(self, payload: Mapping[str, Any], *, prefix: str | None = None) -> None:
        del payload, prefix

    def finish(self, *, status: str) -> None:
        del status


class NullTracker(BaseTracker):
    pass


class WandbTracker(BaseTracker):
    def __init__(
        self,
        *,
        sdk: Any,
        tracking_config: TrackingConfig,
        stage_name: str,
        output_dir: str | Path,
    ) -> None:
        entity = tracking_config.workspace
        try:
            self._run = sdk.init(
                project=tracking_config.project,
                entity=entity,
                name=tracking_config.run_name,
                tags=[stage_name, *tracking_config.tags],
                job_type=stage_name,
                dir=str(output_dir),
                mode="online",
                anonymous="never",
            )
        except Exception as exc:  # pragma: no cover - backend-dependent details
            raise RuntimeError(f"Failed to initialize wandb tracking: {exc}") from exc

    def log_config(self, payload: Mapping[str, Any]) -> None:
        self._run.config.update(make_json_serializable(dict(payload)), allow_val_change=True)

    def log_metrics(
        self,
        payload: Mapping[str, Any],
        *,
        step: int | None = None,
        prefix: str | None = None,
    ) -> None:
        metrics = _normalize_metric_payload(payload, prefix=prefix)
        if not metrics:
            return
        self._run.log(metrics, step=step)

    def log_summary(self, payload: Mapping[str, Any], *, prefix: str | None = None) -> None:
        self._run.summary.update(
            make_json_serializable(_prefix_payload_keys(dict(payload), prefix))
        )

    def finish(self, *, status: str) -> None:
        self._run.summary["run_status"] = status
        self._run.finish(exit_code=0 if status == "success" else 1)


class CometTracker(BaseTracker):
    def __init__(
        self,
        *,
        sdk: Any,
        tracking_config: TrackingConfig,
        stage_name: str,
    ) -> None:
        try:
            self._experiment = sdk.Experiment(
                project_name=tracking_config.project,
                workspace=tracking_config.workspace,
                auto_metric_logging=False,
                auto_param_logging=False,
                parse_args=False,
            )
        except Exception as exc:  # pragma: no cover - backend-dependent details
            raise RuntimeError(f"Failed to initialize comet tracking: {exc}") from exc
        if tracking_config.run_name is not None:
            self._experiment.set_name(tracking_config.run_name)
        self._experiment.add_tags([stage_name, *tracking_config.tags])

    def log_config(self, payload: Mapping[str, Any]) -> None:
        self._experiment.log_parameters(_flatten_payload(make_json_serializable(dict(payload))))

    def log_metrics(
        self,
        payload: Mapping[str, Any],
        *,
        step: int | None = None,
        prefix: str | None = None,
    ) -> None:
        metrics = _normalize_metric_payload(payload, prefix=prefix)
        if not metrics:
            return
        self._experiment.log_metrics(metrics, step=step)

    def log_summary(self, payload: Mapping[str, Any], *, prefix: str | None = None) -> None:
        for key, value in _prefix_payload_keys(make_json_serializable(dict(payload)), prefix).items():
            self._experiment.log_other(key, value)

    def finish(self, *, status: str) -> None:
        self._experiment.log_other("run_status", status)
        self._experiment.end()


def build_tracker(
    config: Mapping[str, Any],
    *,
    stage_name: str,
    output_dir: str | Path,
) -> BaseTracker:
    tracking_config = resolve_tracking_config(config)
    if not tracking_config.enabled:
        return NullTracker()

    run_name = tracking_config.run_name or _build_default_run_name(
        stage_name,
        output_dir,
        config.get("seed"),
    )
    resolved_tracking_config = TrackingConfig(
        enabled=tracking_config.enabled,
        backend=tracking_config.backend,
        project=tracking_config.project,
        workspace=tracking_config.workspace,
        run_name=run_name,
        tags=tracking_config.tags,
    )

    if resolved_tracking_config.backend == "wandb":
        try:
            sdk = importlib.import_module("wandb")
        except ImportError as exc:
            raise RuntimeError(
                "wandb tracking is enabled but the 'wandb' package is not installed."
            ) from exc
        return WandbTracker(
            sdk=sdk,
            tracking_config=resolved_tracking_config,
            stage_name=stage_name,
            output_dir=output_dir,
        )

    if resolved_tracking_config.backend == "comet":
        try:
            sdk = importlib.import_module("comet_ml")
        except ImportError as exc:
            raise RuntimeError(
                "Comet tracking is enabled but the 'comet-ml' package is not installed."
            ) from exc
        return CometTracker(
            sdk=sdk,
            tracking_config=resolved_tracking_config,
            stage_name=stage_name,
        )

    raise ValueError("Unsupported tracking backend. Expected one of: wandb, comet.")
