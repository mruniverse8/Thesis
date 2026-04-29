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
    decoding_strategy: str = "sample"
    num_beams: int = 1
    length_penalty: float = 1.0
    early_stopping: bool = True
    temperature: float = 0.8
    top_p: float = 0.95
    constrained_decoding: bool = True
    terminate_on_invalid_stage: bool = False
    append_probability: float = 0.30
    invalid_append_probability: float = 0.0
    stage_separator: str = STAGE_SEPARATOR
    selfies_dict_path: str = "molecules/dict/selfies_dict.txt"


@dataclass(frozen=True)
class ReplayConfig:
    enabled: bool = False
    buffer_type: str = "experimental_mixture"
    capacity: int = 256
    replay_fraction: float | None = 0.75
    replay_batch_size: int = 0
    max_total_action_tokens: int = 50_000
    with_replacement: bool = False
    recent_fraction: float = 0.40
    reward_fraction: float = 0.40
    uniform_fraction: float = 0.20
    tb_residual_fraction: float = 0.20
    reward_temperature: float = 1.0
    tb_residual_temperature: float = 1.0
    recent_window_size: int = 64
    max_invalid_fraction: float = 0.20
    max_duplicate_fraction: float = 0.10


@dataclass(frozen=True)
class TargetGuidanceConfig:
    enabled: bool = True
    on_policy_fraction: float = 0.25
    target_prefix_rollout_fraction: float = 0.50
    target_teacher_fraction: float = 0.25
    teacher_stage_strategy: str = "random"
    prefix_stage_strategy: str = "random"
    shuffle_target_selfies_list: bool = False

    def __post_init__(self) -> None:
        fractions = (
            float(self.on_policy_fraction),
            float(self.target_prefix_rollout_fraction),
            float(self.target_teacher_fraction),
        )
        if any(value < 0.0 for value in fractions):
            raise ValueError("target_guidance fractions must be non-negative.")
        if bool(self.enabled) and abs(sum(fractions) - 1.0) > 1.0e-6:
            raise ValueError("enabled target_guidance fractions must sum to 1.0.")
        valid_strategies = {"random"}
        if str(self.teacher_stage_strategy).strip().lower() not in valid_strategies:
            raise ValueError("target_guidance.teacher_stage_strategy must be: random.")
        if str(self.prefix_stage_strategy).strip().lower() not in valid_strategies:
            raise ValueError("target_guidance.prefix_stage_strategy must be: random.")


@dataclass(frozen=True)
class ParallelTrainingConfig:
    enabled: bool = False
    mode: str = "replicated_scoring"
    devices: tuple[str, ...] = ("cuda:0", "cuda:1")
    strict: bool = True

    def __post_init__(self) -> None:
        normalized_mode = str(self.mode).strip().lower()
        if normalized_mode != "replicated_scoring":
            raise ValueError("parallel_training.mode must be: replicated_scoring.")
        normalized_devices = tuple(str(device).strip() for device in self.devices)
        if any(not device for device in normalized_devices):
            raise ValueError("parallel_training.devices must not contain empty values.")
        if bool(self.enabled) and len(normalized_devices) < 2:
            raise ValueError("enabled parallel_training requires at least two devices.")
        object.__setattr__(self, "mode", normalized_mode)
        object.__setattr__(self, "devices", normalized_devices)


@dataclass(frozen=True)
class GFlowNetConfig:
    output_dir: str = "outputs/post_training_gflownet"
    seed: int = 42
    start_iteration: int = 0
    gflownet_iterations: int = 200
    batch_size: int = 64
    max_optimization_trajectories_per_iter: int | None = None
    scoring_microbatch_size: int = 4
    objective: str = "tb"
    learning_rate: float = 5.0e-5
    warmup_ratio: float = 0.0
    max_grad_norm: float = 1.0
    diagnostic_log_every_iterations: int = 25
    trajectory_preview_every_iterations: int = 25
    trajectory_preview_num_samples: int = 3
    trajectory_preview_max_chars: int = 480
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
    target_guidance: TargetGuidanceConfig = field(default_factory=TargetGuidanceConfig)
    parallel_training: ParallelTrainingConfig = field(default_factory=ParallelTrainingConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GFlowNetConfig":
        rollout_payload = dict(payload.get("rollout", {}))
        replay_payload = dict(payload.get("replay", {}))
        target_guidance_payload = dict(payload.get("target_guidance", {}))
        parallel_training_payload = dict(payload.get("parallel_training", {}))
        target_modules = tuple(payload.get("target_modules", ("q", "v")))
        objective = str(payload.get("objective", cls.objective)).strip().lower()
        if objective not in {"tb", "db", "subtb"}:
            raise ValueError("objective must be one of: tb, db, subtb.")
        max_optimization_trajectories_per_iter = payload.get(
            "max_optimization_trajectories_per_iter",
            cls.max_optimization_trajectories_per_iter,
        )
        if max_optimization_trajectories_per_iter is not None:
            max_optimization_trajectories_per_iter = max(
                1,
                int(max_optimization_trajectories_per_iter),
            )
        decoding_strategy = str(
            rollout_payload.get(
                "decoding_strategy",
                GFlowNetRolloutConfig.decoding_strategy,
            )
        ).strip().lower()
        if decoding_strategy not in {"sample", "beam"}:
            raise ValueError("rollout.decoding_strategy must be one of: sample, beam.")

        max_stage_new_tokens = rollout_payload.get(
            "max_stage_new_tokens",
            rollout_payload.get(
                "max_new_tokens",
                GFlowNetRolloutConfig.max_stage_new_tokens,
            ),
        )
        replay_fraction_payload = replay_payload.get("replay_fraction")
        legacy_replay_batch_size = max(
            0,
            int(
                replay_payload.get(
                    "replay_batch_size",
                    ReplayConfig.replay_batch_size,
                )
            ),
        )
        if replay_fraction_payload is None and "replay_batch_size" in replay_payload:
            replay_fraction: float | None = None
        else:
            replay_fraction = max(
                0.0,
                min(
                    float(
                        replay_fraction_payload
                        if replay_fraction_payload is not None
                        else ReplayConfig.replay_fraction
                    ),
                    1.0 - 1.0e-6,
                ),
            )
        parallel_training_devices = parallel_training_payload.get(
            "devices",
            ParallelTrainingConfig.devices,
        )
        if isinstance(parallel_training_devices, str):
            parallel_training_devices = (parallel_training_devices,)
        return cls(
            output_dir=str(payload.get("output_dir", cls.output_dir)),
            seed=int(payload.get("seed", cls.seed)),
            start_iteration=max(
                0,
                int(payload.get("start_iteration", cls.start_iteration)),
            ),
            gflownet_iterations=int(
                payload.get("gflownet_iterations", cls.gflownet_iterations)
            ),
            batch_size=max(1, int(payload.get("batch_size", cls.batch_size))),
            max_optimization_trajectories_per_iter=(
                max_optimization_trajectories_per_iter
            ),
            scoring_microbatch_size=max(
                1,
                int(
                    payload.get(
                        "scoring_microbatch_size",
                        cls.scoring_microbatch_size,
                    )
                ),
            ),
            objective=objective,
            learning_rate=float(payload.get("learning_rate", cls.learning_rate)),
            warmup_ratio=max(
                0.0,
                min(float(payload.get("warmup_ratio", cls.warmup_ratio)), 1.0),
            ),
            max_grad_norm=float(payload.get("max_grad_norm", cls.max_grad_norm)),
            diagnostic_log_every_iterations=max(
                1,
                int(
                    payload.get(
                        "diagnostic_log_every_iterations",
                        cls.diagnostic_log_every_iterations,
                    )
                ),
            ),
            trajectory_preview_every_iterations=max(
                1,
                int(
                    payload.get(
                        "trajectory_preview_every_iterations",
                        cls.trajectory_preview_every_iterations,
                    )
                ),
            ),
            trajectory_preview_num_samples=max(
                1,
                int(
                    payload.get(
                        "trajectory_preview_num_samples",
                        cls.trajectory_preview_num_samples,
                    )
                ),
            ),
            trajectory_preview_max_chars=max(
                32,
                int(
                    payload.get(
                        "trajectory_preview_max_chars",
                        cls.trajectory_preview_max_chars,
                    )
                ),
            ),
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
                decoding_strategy=decoding_strategy,
                num_beams=max(
                    1,
                    int(
                        rollout_payload.get(
                            "num_beams",
                            GFlowNetRolloutConfig.num_beams,
                        )
                    ),
                ),
                length_penalty=max(
                    0.0,
                    float(
                        rollout_payload.get(
                            "length_penalty",
                            GFlowNetRolloutConfig.length_penalty,
                        )
                    ),
                ),
                early_stopping=bool(
                    rollout_payload.get(
                        "early_stopping",
                        GFlowNetRolloutConfig.early_stopping,
                    )
                ),
                temperature=float(
                    rollout_payload.get(
                        "temperature",
                        GFlowNetRolloutConfig.temperature,
                    )
                ),
                top_p=float(rollout_payload.get("top_p", GFlowNetRolloutConfig.top_p)),
                constrained_decoding=bool(
                    rollout_payload.get(
                        "constrained_decoding",
                        GFlowNetRolloutConfig.constrained_decoding,
                    )
                ),
                terminate_on_invalid_stage=True,
                append_probability=max(
                    0.0,
                    min(
                        float(
                            rollout_payload.get(
                                "append_probability",
                                GFlowNetRolloutConfig.append_probability,
                            )
                        ),
                        1.0,
                    ),
                ),
                invalid_append_probability=max(
                    0.0,
                    min(
                        float(
                            rollout_payload.get(
                                "invalid_append_probability",
                                GFlowNetRolloutConfig.invalid_append_probability,
                            )
                        ),
                        1.0,
                    ),
                ),
                stage_separator=str(
                    rollout_payload.get(
                        "stage_separator",
                        GFlowNetRolloutConfig.stage_separator,
                    )
                ),
                selfies_dict_path=str(
                    rollout_payload.get(
                        "selfies_dict_path",
                        GFlowNetRolloutConfig.selfies_dict_path,
                    )
                ),
            ),
            replay=ReplayConfig(
                enabled=bool(replay_payload.get("enabled", ReplayConfig.enabled)),
                buffer_type=str(
                    replay_payload.get(
                        "buffer_type",
                        ReplayConfig.buffer_type,
                    )
                ).strip().lower(),
                capacity=max(0, int(replay_payload.get("capacity", ReplayConfig.capacity))),
                replay_fraction=replay_fraction,
                replay_batch_size=legacy_replay_batch_size,
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
                recent_fraction=max(
                    0.0,
                    float(
                        replay_payload.get(
                            "recent_fraction",
                            ReplayConfig.recent_fraction,
                        )
                    ),
                ),
                reward_fraction=max(
                    0.0,
                    float(
                        replay_payload.get(
                            "reward_fraction",
                            ReplayConfig.reward_fraction,
                        )
                    ),
                ),
                uniform_fraction=max(
                    0.0,
                    float(
                        replay_payload.get(
                            "uniform_fraction",
                            ReplayConfig.uniform_fraction,
                        )
                    ),
                ),
                tb_residual_fraction=max(
                    0.0,
                    float(
                        replay_payload.get(
                            "tb_residual_fraction",
                            ReplayConfig.tb_residual_fraction,
                        )
                    ),
                ),
                reward_temperature=max(
                    1.0e-6,
                    float(
                        replay_payload.get(
                            "reward_temperature",
                            ReplayConfig.reward_temperature,
                        )
                    ),
                ),
                tb_residual_temperature=max(
                    1.0e-6,
                    float(
                        replay_payload.get(
                            "tb_residual_temperature",
                            ReplayConfig.tb_residual_temperature,
                        )
                    ),
                ),
                recent_window_size=max(
                    1,
                    int(
                        replay_payload.get(
                            "recent_window_size",
                            ReplayConfig.recent_window_size,
                        )
                    ),
                ),
                max_invalid_fraction=max(
                    0.0,
                    min(
                        float(
                            replay_payload.get(
                                "max_invalid_fraction",
                                ReplayConfig.max_invalid_fraction,
                            )
                        ),
                        1.0,
                    ),
                ),
                max_duplicate_fraction=max(
                    0.0,
                    min(
                        float(
                            replay_payload.get(
                                "max_duplicate_fraction",
                                ReplayConfig.max_duplicate_fraction,
                            )
                        ),
                        1.0,
                    ),
                ),
            ),
            target_guidance=TargetGuidanceConfig(
                enabled=bool(
                    target_guidance_payload.get(
                        "enabled",
                        TargetGuidanceConfig.enabled,
                    )
                ),
                on_policy_fraction=float(
                    target_guidance_payload.get(
                        "on_policy_fraction",
                        TargetGuidanceConfig.on_policy_fraction,
                    )
                ),
                target_prefix_rollout_fraction=float(
                    target_guidance_payload.get(
                        "target_prefix_rollout_fraction",
                        TargetGuidanceConfig.target_prefix_rollout_fraction,
                    )
                ),
                target_teacher_fraction=float(
                    target_guidance_payload.get(
                        "target_teacher_fraction",
                        TargetGuidanceConfig.target_teacher_fraction,
                    )
                ),
                teacher_stage_strategy=str(
                    target_guidance_payload.get(
                        "teacher_stage_strategy",
                        TargetGuidanceConfig.teacher_stage_strategy,
                    )
                ).strip().lower(),
                prefix_stage_strategy=str(
                    target_guidance_payload.get(
                        "prefix_stage_strategy",
                        TargetGuidanceConfig.prefix_stage_strategy,
                    )
                ).strip().lower(),
                shuffle_target_selfies_list=bool(
                    target_guidance_payload.get(
                        "shuffle_target_selfies_list",
                        TargetGuidanceConfig.shuffle_target_selfies_list,
                    )
                ),
            ),
            parallel_training=ParallelTrainingConfig(
                enabled=bool(
                    parallel_training_payload.get(
                        "enabled",
                        ParallelTrainingConfig.enabled,
                    )
                ),
                mode=str(
                    parallel_training_payload.get(
                        "mode",
                        ParallelTrainingConfig.mode,
                    )
                ),
                devices=tuple(
                    parallel_training_devices
                ),
                strict=bool(
                    parallel_training_payload.get(
                        "strict",
                        ParallelTrainingConfig.strict,
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

    if "seed" not in gflownet_payload and "seed" in config:
        gflownet_payload["seed"] = config["seed"]
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
