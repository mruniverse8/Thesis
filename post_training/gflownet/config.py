from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from post_training.shared.sequence import STAGE_SEPARATOR


@dataclass(frozen=True)
class GFlowNetRolloutConfig:
    max_source_length: int = 512
    max_stage_new_tokens: int = 128
    max_molecules_per_sequence: int = 8
    max_sequence_length: int = 2560
    temperature: float = 1.0
    top_p: float = 1.0
    terminate_on_invalid_stage: bool = True
    stage_separator: str = STAGE_SEPARATOR


@dataclass(frozen=True)
class ReplayConfig:
    capacity: int = 0
    replay_batch_size: int = 0
    max_total_action_tokens: int = 50_000
    with_replacement: bool = False


@dataclass(frozen=True)
class GFlowNetConfig:
    output_dir: str = "outputs/post_training_gflownet"
    gflownet_iterations: int = 200
    batch_size: int = 64
    objective: str = "tb"
    learning_rate: float = 5.0e-5
    max_grad_norm: float = 1.0
    invalid_terminal_reward: float = 1.0e-4
    save_every_iterations: int = 10
    use_lora: bool = True
    freeze_base_model_without_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q", "v")
    rollout: GFlowNetRolloutConfig = field(default_factory=GFlowNetRolloutConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GFlowNetConfig":
        rollout_payload = dict(payload.get("rollout", {}))
        replay_payload = dict(payload.get("replay", {}))
        target_modules = tuple(payload.get("target_modules", ("q", "v")))
        objective = str(payload.get("objective", cls.objective)).strip().lower()
        if objective == "subtb":
            raise ValueError(
                "objective 'subtb' is disabled pending a learned state-flow redesign."
            )
        if objective not in {"tb", "db"}:
            raise ValueError("objective must be one of: tb, db.")

        max_stage_new_tokens = rollout_payload.get(
            "max_stage_new_tokens",
            rollout_payload.get(
                "max_new_tokens",
                GFlowNetRolloutConfig.max_stage_new_tokens,
            ),
        )
        return cls(
            output_dir=str(payload.get("output_dir", cls.output_dir)),
            gflownet_iterations=int(
                payload.get("gflownet_iterations", cls.gflownet_iterations)
            ),
            batch_size=max(1, int(payload.get("batch_size", cls.batch_size))),
            objective=objective,
            learning_rate=float(payload.get("learning_rate", cls.learning_rate)),
            max_grad_norm=float(payload.get("max_grad_norm", cls.max_grad_norm)),
            invalid_terminal_reward=max(
                1.0e-12,
                float(
                    payload.get(
                        "invalid_terminal_reward",
                        cls.invalid_terminal_reward,
                    )
                ),
            ),
            save_every_iterations=max(
                1,
                int(payload.get("save_every_iterations", cls.save_every_iterations)),
            ),
            use_lora=bool(payload.get("use_lora", cls.use_lora)),
            freeze_base_model_without_lora=bool(
                payload.get(
                    "freeze_base_model_without_lora",
                    cls.freeze_base_model_without_lora,
                )
            ),
            lora_rank=int(payload.get("lora_rank", cls.lora_rank)),
            lora_alpha=int(payload.get("lora_alpha", cls.lora_alpha)),
            lora_dropout=float(payload.get("lora_dropout", cls.lora_dropout)),
            target_modules=target_modules,
            rollout=GFlowNetRolloutConfig(
                max_source_length=int(
                    rollout_payload.get(
                        "max_source_length",
                        GFlowNetRolloutConfig.max_source_length,
                    )
                ),
                max_stage_new_tokens=max(1, int(max_stage_new_tokens)),
                max_molecules_per_sequence=max(
                    1,
                    int(
                        rollout_payload.get(
                            "max_molecules_per_sequence",
                            GFlowNetRolloutConfig.max_molecules_per_sequence,
                        )
                    ),
                ),
                max_sequence_length=max(
                    1,
                    int(
                        rollout_payload.get(
                            "max_sequence_length",
                            GFlowNetRolloutConfig.max_sequence_length,
                        )
                    ),
                ),
                temperature=float(
                    rollout_payload.get(
                        "temperature",
                        GFlowNetRolloutConfig.temperature,
                    )
                ),
                top_p=float(rollout_payload.get("top_p", GFlowNetRolloutConfig.top_p)),
                terminate_on_invalid_stage=bool(
                    rollout_payload.get(
                        "terminate_on_invalid_stage",
                        GFlowNetRolloutConfig.terminate_on_invalid_stage,
                    )
                ),
                stage_separator=str(
                    rollout_payload.get(
                        "stage_separator",
                        GFlowNetRolloutConfig.stage_separator,
                    )
                ),
            ),
            replay=ReplayConfig(
                capacity=max(0, int(replay_payload.get("capacity", ReplayConfig.capacity))),
                replay_batch_size=max(
                    0,
                    int(
                        replay_payload.get(
                            "replay_batch_size",
                            ReplayConfig.replay_batch_size,
                        )
                    ),
                ),
                max_total_action_tokens=max(
                    1,
                    int(
                        replay_payload.get(
                            "max_total_action_tokens",
                            ReplayConfig.max_total_action_tokens,
                        )
                    ),
                ),
                with_replacement=bool(
                    replay_payload.get(
                        "with_replacement",
                        ReplayConfig.with_replacement,
                    )
                ),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_modules"] = list(self.target_modules)
        return payload


def build_gflownet_config(config: dict[str, Any]) -> GFlowNetConfig:
    gflownet_payload = dict(config.get("gflownet", {}))
    training_config = config.get("training", {})
    model_config = config.get("model", {})

    gflownet_payload.setdefault("output_dir", training_config["output_dir"])
    gflownet_payload.setdefault(
        "save_every_iterations",
        training_config.get("save_every_iterations", GFlowNetConfig.save_every_iterations),
    )
    gflownet_payload.setdefault(
        "use_lora",
        model_config.get("use_lora", GFlowNetConfig.use_lora),
    )
    gflownet_payload.setdefault(
        "freeze_base_model_without_lora",
        model_config.get(
            "freeze_base_model_without_lora",
            GFlowNetConfig.freeze_base_model_without_lora,
        ),
    )
    gflownet_payload.setdefault(
        "lora_rank",
        model_config.get("lora_rank", GFlowNetConfig.lora_rank),
    )
    gflownet_payload.setdefault(
        "lora_alpha",
        model_config.get("lora_alpha", GFlowNetConfig.lora_alpha),
    )
    gflownet_payload.setdefault(
        "lora_dropout",
        model_config.get("lora_dropout", GFlowNetConfig.lora_dropout),
    )
    gflownet_payload.setdefault(
        "target_modules",
        model_config.get("target_modules", list(GFlowNetConfig.target_modules)),
    )

    rollout_payload = dict(gflownet_payload.get("rollout", {}))
    rollout_payload.setdefault(
        "max_source_length",
        config.get("data", {}).get(
            "max_source_length",
            GFlowNetRolloutConfig.max_source_length,
        ),
    )
    gflownet_payload["rollout"] = rollout_payload
    return GFlowNetConfig.from_dict(gflownet_payload)
