from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from reward_utils.rewards import RewardBreakdown

from post_training.shared.sequence import STAGE_SEPARATOR


@dataclass(frozen=True)
class RolloutGenerationConfig:
    max_source_length: int = 512
    max_stage_new_tokens: int = 128
    max_molecules_per_sequence: int = 8
    max_sequence_length: int = 2560
    temperature: float = 0.8
    top_p: float = 0.95
    constrained_decoding: bool = True
    terminate_on_invalid_stage: bool = True
    stage_separator: str = STAGE_SEPARATOR
    selfies_dict_path: str = "molecules/dict/selfies_dict.txt"


@dataclass(frozen=True)
class PPOConfig:
    output_dir: str = "outputs/post_training_ppo"
    ppo_iterations: int = 200
    batch_size: int = 128
    mini_batch_size: int = 8
    ppo_epochs_per_batch: int = 4
    learning_rate: float = 5.0e-5
    kl_penalty: float = 0.01
    clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    save_every_iterations: int = 10
    diagnostic_log_every_optimizer_steps: int = 25
    trajectory_preview_every_iterations: int = 25
    num_trajectory_samples_to_log: int = 3
    trajectory_preview_max_chars: int = 240
    use_lora: bool = True
    freeze_base_model_without_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q", "v")
    rollout: RolloutGenerationConfig = field(default_factory=RolloutGenerationConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PPOConfig":
        rollout_payload = dict(payload.get("rollout", {}))
        target_modules = tuple(payload.get("target_modules", ("q", "v")))
        return cls(
            output_dir=str(payload.get("output_dir", cls.output_dir)),
            ppo_iterations=int(payload.get("ppo_iterations", cls.ppo_iterations)),
            batch_size=int(payload.get("batch_size", cls.batch_size)),
            mini_batch_size=int(payload.get("mini_batch_size", cls.mini_batch_size)),
            ppo_epochs_per_batch=int(
                payload.get("ppo_epochs_per_batch", cls.ppo_epochs_per_batch)
            ),
            learning_rate=float(payload.get("learning_rate", cls.learning_rate)),
            kl_penalty=float(payload.get("kl_penalty", cls.kl_penalty)),
            clip_range=float(payload.get("clip_range", cls.clip_range)),
            value_loss_coef=float(payload.get("value_loss_coef", cls.value_loss_coef)),
            entropy_coef=float(payload.get("entropy_coef", cls.entropy_coef)),
            max_grad_norm=float(payload.get("max_grad_norm", cls.max_grad_norm)),
            save_every_iterations=int(
                payload.get("save_every_iterations", cls.save_every_iterations)
            ),
            diagnostic_log_every_optimizer_steps=max(
                1,
                int(
                    payload.get(
                        "diagnostic_log_every_optimizer_steps",
                        cls.diagnostic_log_every_optimizer_steps,
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
            num_trajectory_samples_to_log=max(
                1,
                int(
                    payload.get(
                        "num_trajectory_samples_to_log",
                        cls.num_trajectory_samples_to_log,
                    )
                ),
            ),
            trajectory_preview_max_chars=max(
                1,
                int(
                    payload.get(
                        "trajectory_preview_max_chars",
                        cls.trajectory_preview_max_chars,
                    )
                ),
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
            rollout=RolloutGenerationConfig(
                max_source_length=int(
                    rollout_payload.get(
                        "max_source_length",
                        RolloutGenerationConfig.max_source_length,
                    )
                ),
                max_stage_new_tokens=int(
                    rollout_payload.get(
                        "max_stage_new_tokens",
                        RolloutGenerationConfig.max_stage_new_tokens,
                    )
                ),
                max_molecules_per_sequence=int(
                    rollout_payload.get(
                        "max_molecules_per_sequence",
                        RolloutGenerationConfig.max_molecules_per_sequence,
                    )
                ),
                max_sequence_length=int(
                    rollout_payload.get(
                        "max_sequence_length",
                        RolloutGenerationConfig.max_sequence_length,
                    )
                ),
                temperature=float(
                    rollout_payload.get("temperature", RolloutGenerationConfig.temperature)
                ),
                top_p=float(rollout_payload.get("top_p", RolloutGenerationConfig.top_p)),
                constrained_decoding=bool(
                    rollout_payload.get(
                        "constrained_decoding",
                        RolloutGenerationConfig.constrained_decoding,
                    )
                ),
                terminate_on_invalid_stage=bool(
                    rollout_payload.get(
                        "terminate_on_invalid_stage",
                        RolloutGenerationConfig.terminate_on_invalid_stage,
                    )
                ),
                stage_separator=str(
                    rollout_payload.get("stage_separator", RolloutGenerationConfig.stage_separator)
                ),
                selfies_dict_path=str(
                    rollout_payload.get(
                        "selfies_dict_path",
                        RolloutGenerationConfig.selfies_dict_path,
                    )
                ),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_modules"] = list(self.target_modules)
        return payload


def build_ppo_config(config: dict[str, Any]) -> PPOConfig:
    ppo_payload = dict(config.get("ppo", {}))
    training_config = config.get("training", {})
    model_config = config.get("model", {})

    ppo_payload.setdefault("output_dir", training_config["output_dir"])
    ppo_payload.setdefault(
        "save_every_iterations",
        training_config.get("save_every_iterations", PPOConfig.save_every_iterations),
    )
    ppo_payload.setdefault("use_lora", model_config.get("use_lora", PPOConfig.use_lora))
    ppo_payload.setdefault(
        "freeze_base_model_without_lora",
        model_config.get(
            "freeze_base_model_without_lora",
            PPOConfig.freeze_base_model_without_lora,
        ),
    )
    ppo_payload.setdefault("lora_rank", model_config.get("lora_rank", PPOConfig.lora_rank))
    ppo_payload.setdefault("lora_alpha", model_config.get("lora_alpha", PPOConfig.lora_alpha))
    ppo_payload.setdefault("lora_dropout", model_config.get("lora_dropout", PPOConfig.lora_dropout))
    ppo_payload.setdefault(
        "target_modules",
        model_config.get("target_modules", list(PPOConfig.target_modules)),
    )

    rollout_payload = dict(ppo_payload.get("rollout", {}))
    rollout_payload.setdefault(
        "max_source_length",
        config.get("data", {}).get("max_source_length", RolloutGenerationConfig.max_source_length),
    )
    ppo_payload["rollout"] = rollout_payload
    return PPOConfig.from_dict(ppo_payload)


@dataclass(frozen=True)
class StageTrajectory:
    rollout_id: str
    example_id: str
    prompt_text: str
    description: str
    target_selfies_list: tuple[str, ...]
    stage_index: int
    decoder_prefix_text: str
    stage_text: str
    sampled_selfies: str | None
    stop_token: str | None
    termination_reason: str
    action_token_ids: tuple[int, ...]
    action_logprob_sum_old: float
    reference_logprob_sum: float
    value_old: float
    reward_breakdown: RewardBreakdown
    reward: float
    entropy_sum_old: float
    is_valid: bool
    is_duplicate: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        breakdown = self.reward_breakdown
        return {
            "rollout_id": self.rollout_id,
            "example_id": self.example_id,
            "prompt_text": self.prompt_text,
            "description": self.description,
            "target_selfies_list": list(self.target_selfies_list),
            "stage_index": self.stage_index,
            "decoder_prefix_text": self.decoder_prefix_text,
            "stage_text": self.stage_text,
            "sampled_selfies": self.sampled_selfies,
            "stop_token": self.stop_token,
            "termination_reason": self.termination_reason,
            "action_token_ids": list(self.action_token_ids),
            "action_logprob_sum_old": self.action_logprob_sum_old,
            "reference_logprob_sum": self.reference_logprob_sum,
            "value_old": self.value_old,
            "reward": self.reward,
            "entropy_sum_old": self.entropy_sum_old,
            "is_valid": self.is_valid,
            "is_duplicate": self.is_duplicate,
            "metadata": dict(self.metadata),
            "reward_breakdown": {
                "candidate": breakdown.candidate.canonical_smiles,
                "match_reward": breakdown.match.reward,
                "diversity_reward": breakdown.diversity.reward,
                "total_reward": breakdown.total_reward,
                "amplified_reward": breakdown.amplified_reward,
            },
        }
