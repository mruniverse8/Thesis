import pytest
import torch
from types import SimpleNamespace

pytest.importorskip("selfies")
pytest.importorskip("rdkit")

from reward_utils.defaults import CHEBI20_REWARD_CONFIG
from src.constants import EOM_TOKEN

from post_training.gflownet.config import GFlowNetRolloutConfig
from post_training.gflownet.rollout import (
    build_sampled_stage_trajectory_from_generation,
    sample_stage,
    sample_stage_trajectories_for_example,
)
from post_training.shared.decoding import StageTokenConstraints
from post_training.shared.sequence import build_stage_prefix


class DummyTokenizer:
    def __init__(self, id_to_token: dict[int, str], token_to_id: dict[str, int]) -> None:
        self.id_to_token = id_to_token
        self.token_to_id = token_to_id
        self.eos_token = "</s>"

    def __call__(
        self,
        text,
        truncation=False,
        max_length=None,
        return_tensors=None,
        add_special_tokens=True,
        return_attention_mask=False,
    ):
        del truncation, max_length, add_special_tokens, return_attention_mask
        remaining = str(text)
        token_ids: list[int] = []
        known_tokens = sorted(self.token_to_id, key=len, reverse=True)
        while remaining:
            matched_token = next(
                (token for token in known_tokens if remaining.startswith(token)),
                None,
            )
            if matched_token is None:
                token_ids = [1, 2]
                break
            token_ids.append(int(self.token_to_id[matched_token]))
            remaining = remaining[len(matched_token) :]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([token_ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
            }
        return {
            "input_ids": token_ids,
            "attention_mask": [1] * len(token_ids),
        }

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(self.id_to_token[int(token_id)] for token_id in token_ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return int(self.token_to_id[token])


class FixedRandom:
    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def random(self) -> float:
        if not self._values:
            raise AssertionError("No more RNG values were available.")
        return float(self._values.pop(0))


class PrefixLogitPolicyModel:
    def __init__(
        self,
        logits_by_action_prefix: dict[tuple[int, ...], dict[int, float]],
        *,
        vocab_size: int = 16,
        eos_token_id: int | None = 99,
    ) -> None:
        self.logits_by_action_prefix = logits_by_action_prefix
        self.config = SimpleNamespace(eos_token_id=eos_token_id)
        self.calls: list[tuple[int, ...]] = []
        self.vocab_size = vocab_size

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        return_dict: bool,
    ) -> SimpleNamespace:
        del input_ids, attention_mask, return_dict
        action_prefix = tuple(int(token_id) for token_id in decoder_input_ids[0, 1:].tolist())
        self.calls.append(action_prefix)
        logits = torch.full((1, decoder_input_ids.size(1), self.vocab_size), -20.0)
        for token_id, score in self.logits_by_action_prefix.get(action_prefix, {}).items():
            logits[:, -1, int(token_id)] = float(score)
        return SimpleNamespace(logits=logits)


class PrefixLogitModel:
    def __init__(
        self,
        logits_by_action_prefix: dict[tuple[int, ...], dict[int, float]],
        *,
        vocab_size: int = 16,
        eos_token_id: int | None = 99,
    ) -> None:
        self.policy_model = PrefixLogitPolicyModel(
            logits_by_action_prefix,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
        )


def test_build_sampled_stage_trajectory_from_generation_keeps_stop_out_of_action_ids() -> None:
    trajectory = build_sampled_stage_trajectory_from_generation(
        example={
            "id": "example-1",
            "prompt": "prompt",
            "description": "description",
            "target_selfies_list": ["[C][C][O]"],
        },
        rollout_id="rollout-1",
        stage_index=1,
        decoder_prefix_text="",
        previous_sampled_selfies=(),
        stage_text="<bom>[C][C][O]<eom>",
        sampled_selfies="[C][C][O]",
        action_token_ids=(1, 2),
        metadata={"raw_stage_text": "<bom>[C][C][O]<eom>"},
        stop_token=EOM_TOKEN,
        termination_reason="stop_token",
        reward_config=CHEBI20_REWARD_CONFIG,
        invalid_terminal_reward=1.0e-4,
    )

    assert trajectory.action_token_ids == (1, 2)
    assert trajectory.stage_text == "<bom>[C][C][O]<eom>"
    assert trajectory.sampled_selfies == "[C][C][O]"
    assert trajectory.stop_token == EOM_TOKEN
    assert trajectory.termination_reason == "stop_token"
    assert trajectory.prefix_rewards[-1] == pytest.approx(trajectory.terminal_reward)
    assert trajectory.metadata["raw_stage_text"] == "<bom>[C][C][O]<eom>"
    assert trajectory.previous_sampled_selfies == ()
    assert trajectory.to_dict()["previous_sampled_selfies"] == []


def test_sample_stage_trajectories_for_example_updates_prefix_after_sampled_stage(monkeypatch) -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[C][C][O]",
            3: "<eom>",
            4: "<bom>",
            5: "[C][C][N]",
        },
        {EOM_TOKEN: 3},
    )

    class DummyModel:
        def __init__(self) -> None:
            self.policy_model = type(
                "Policy",
                (),
                {"config": type("Config", (), {"decoder_start_token_id": 0, "eos_token_id": 99})()},
            )()

    samples = iter(
        [
            {
                "stage_text": "<bom>[C][C][O]<eom>",
                "sampled_selfies": "[C][C][O]",
                "action_token_ids": (1, 2),
                "stop_token": EOM_TOKEN,
                "termination_reason": "stop_token",
            },
            {
                "stage_text": "<bom>[C][C][N]<eom>",
                "sampled_selfies": "[C][C][N]",
                "action_token_ids": (4, 5),
                "stop_token": EOM_TOKEN,
                "termination_reason": "stop_token",
            },
        ]
    )
    monkeypatch.setattr("post_training.gflownet.rollout.sample_stage", lambda *args, **kwargs: next(samples))

    trajectories = sample_stage_trajectories_for_example(
        DummyModel(),
        tokenizer,
        {
            "id": "example-1",
            "prompt": "prompt",
            "description": "description",
            "target_selfies_list": ["[C][C][O]", "[C][C][N]"],
        },
        rollout_id="rollout-1",
        generation_config=GFlowNetRolloutConfig(
            max_molecules_per_sequence=2,
            append_probability=1.0,
        ),
        reward_config=CHEBI20_REWARD_CONFIG,
        invalid_terminal_reward=1.0e-4,
        device=torch.device("cpu"),
    )

    assert len(trajectories) == 2
    assert trajectories[0].decoder_prefix_text == ""
    assert trajectories[1].decoder_prefix_text == build_stage_prefix(["[C][C][O]"])


def test_sample_stage_trajectories_for_example_ignores_invalid_stage_when_configured(
    monkeypatch,
) -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<eom>",
        },
        {EOM_TOKEN: 1},
    )

    class DummyModel:
        def __init__(self) -> None:
            self.policy_model = type(
                "Policy",
                (),
                {"config": type("Config", (), {"decoder_start_token_id": 0, "eos_token_id": 99})()},
            )()

    monkeypatch.setattr(
        "post_training.gflownet.rollout.sample_stage",
        lambda *args, **kwargs: {
            "stage_text": "<eom>",
            "sampled_selfies": None,
            "action_token_ids": (),
            "stop_token": EOM_TOKEN,
            "termination_reason": "stop_token",
        },
    )

    trajectories = sample_stage_trajectories_for_example(
        DummyModel(),
        tokenizer,
        {
            "id": "example-1",
            "prompt": "prompt",
            "description": "description",
            "target_selfies_list": ["[C][C][O]", "[C][C][N]"],
        },
        rollout_id="rollout-1",
        generation_config=GFlowNetRolloutConfig(
            max_molecules_per_sequence=2,
            append_probability=1.0,
            terminate_on_invalid_stage=True,
        ),
        reward_config=CHEBI20_REWARD_CONFIG,
        invalid_terminal_reward=1.0e-4,
        device=torch.device("cpu"),
    )

    assert trajectories == []


def test_sample_stage_trajectories_for_example_appends_invalid_stage_when_probability_allows(
    monkeypatch,
) -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<eom>",
        },
        {EOM_TOKEN: 1},
    )

    class DummyModel:
        def __init__(self) -> None:
            self.policy_model = type(
                "Policy",
                (),
                {"config": type("Config", (), {"decoder_start_token_id": 0, "eos_token_id": 99})()},
            )()

    monkeypatch.setattr(
        "post_training.gflownet.rollout.sample_stage",
        lambda *args, **kwargs: {
            "stage_text": "<eom>",
            "sampled_selfies": None,
            "action_token_ids": (),
            "stop_token": EOM_TOKEN,
            "termination_reason": "stop_token",
        },
    )

    trajectories = sample_stage_trajectories_for_example(
        DummyModel(),
        tokenizer,
        {
            "id": "example-1",
            "prompt": "prompt",
            "description": "description",
            "target_selfies_list": ["[C][C][O]", "[C][C][N]"],
        },
        rollout_id="rollout-1",
        generation_config=GFlowNetRolloutConfig(
            max_molecules_per_sequence=1,
            append_probability=0.0,
            invalid_append_probability=0.25,
            terminate_on_invalid_stage=True,
        ),
        reward_config=CHEBI20_REWARD_CONFIG,
        invalid_terminal_reward=1.0e-4,
        device=torch.device("cpu"),
        rng=FixedRandom([0.1]),
    )

    assert len(trajectories) == 1
    assert trajectories[0].is_valid is False
    assert trajectories[0].terminal_reward == pytest.approx(
        1.0e-4 * CHEBI20_REWARD_CONFIG.penalty_invalid
    )


def test_sample_stage_trajectories_for_example_skips_invalid_stage_when_probability_rejects(
    monkeypatch,
) -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<eom>",
        },
        {EOM_TOKEN: 1},
    )

    class DummyModel:
        def __init__(self) -> None:
            self.policy_model = type(
                "Policy",
                (),
                {"config": type("Config", (), {"decoder_start_token_id": 0, "eos_token_id": 99})()},
            )()

    monkeypatch.setattr(
        "post_training.gflownet.rollout.sample_stage",
        lambda *args, **kwargs: {
            "stage_text": "<eom>",
            "sampled_selfies": None,
            "action_token_ids": (),
            "stop_token": EOM_TOKEN,
            "termination_reason": "stop_token",
        },
    )

    trajectories = sample_stage_trajectories_for_example(
        DummyModel(),
        tokenizer,
        {
            "id": "example-1",
            "prompt": "prompt",
            "description": "description",
            "target_selfies_list": ["[C][C][O]", "[C][C][N]"],
        },
        rollout_id="rollout-1",
        generation_config=GFlowNetRolloutConfig(
            max_molecules_per_sequence=1,
            append_probability=1.0,
            invalid_append_probability=0.25,
            terminate_on_invalid_stage=True,
        ),
        reward_config=CHEBI20_REWARD_CONFIG,
        invalid_terminal_reward=1.0e-4,
        device=torch.device("cpu"),
        rng=FixedRandom([0.9]),
    )

    assert trajectories == []


def test_sample_stage_trajectories_for_example_keeps_invalid_sample_out_of_prefix_history(
    monkeypatch,
) -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[C][C][O]",
            3: "<eom>",
            4: "<bom>",
            5: "[C][C][N]",
        },
        {EOM_TOKEN: 3},
    )

    class DummyModel:
        def __init__(self) -> None:
            self.policy_model = type(
                "Policy",
                (),
                {"config": type("Config", (), {"decoder_start_token_id": 0, "eos_token_id": 99})()},
            )()

    samples = iter(
        [
            {
                "stage_text": "<bom>[C][C][O]<eom>",
                "sampled_selfies": "[C][C][O]",
                "action_token_ids": (1, 2),
                "stop_token": EOM_TOKEN,
                "termination_reason": "stop_token",
            },
            {
                "stage_text": "<bom>[C][C][N]<eom>",
                "sampled_selfies": "[C][C][N]",
                "action_token_ids": (4, 5),
                "stop_token": EOM_TOKEN,
                "termination_reason": "stop_token",
            },
        ]
    )

    def fake_reward_summary(
        candidate_selfies,
        *,
        targets,
        previous_candidates,
        num_prefix_states,
        reward_config,
        invalid_terminal_reward,
    ):
        del targets, reward_config
        terminal_reward = 2.0 if candidate_selfies == "[C][C][N]" else invalid_terminal_reward
        return SimpleNamespace(
            reward_breakdown={"previous_candidates": tuple(previous_candidates)},
            prefix_rewards=tuple([invalid_terminal_reward] * (num_prefix_states - 1) + [terminal_reward]),
            terminal_reward=terminal_reward,
            is_valid_terminal=candidate_selfies == "[C][C][N]",
            is_duplicate_terminal=False,
        )

    monkeypatch.setattr("post_training.gflownet.rollout.sample_stage", lambda *args, **kwargs: next(samples))
    monkeypatch.setattr("post_training.gflownet.rollout.score_stage_terminal_reward", fake_reward_summary)

    trajectories = sample_stage_trajectories_for_example(
        DummyModel(),
        tokenizer,
        {
            "id": "example-1",
            "prompt": "prompt",
            "description": "description",
            "target_selfies_list": ["[C][C][O]", "[C][C][N]"],
        },
        rollout_id="rollout-1",
        generation_config=GFlowNetRolloutConfig(
            max_molecules_per_sequence=2,
            append_probability=1.0,
            terminate_on_invalid_stage=False,
        ),
        reward_config=CHEBI20_REWARD_CONFIG,
        invalid_terminal_reward=1.0e-4,
        device=torch.device("cpu"),
    )

    assert len(trajectories) == 1
    assert trajectories[0].is_valid is True
    assert trajectories[0].decoder_prefix_text == ""
    assert trajectories[0].previous_sampled_selfies == ()


def test_sample_stage_trajectories_for_example_appends_stage_two_plus_only_when_probability_allows_it(
    monkeypatch,
) -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[C][C][O]",
            3: "<eom>",
            4: "<bom>",
            5: "[C][C][N]",
            6: "<bom>",
            7: "[C][O][O]",
        },
        {EOM_TOKEN: 3},
    )

    class DummyModel:
        def __init__(self) -> None:
            self.policy_model = type(
                "Policy",
                (),
                {"config": type("Config", (), {"decoder_start_token_id": 0, "eos_token_id": 99})()},
            )()

    base_samples = [
        {
            "stage_text": "<bom>[C][C][O]<eom>",
            "sampled_selfies": "[C][C][O]",
            "action_token_ids": (1, 2),
            "stop_token": EOM_TOKEN,
            "termination_reason": "stop_token",
        },
        {
            "stage_text": "<bom>[C][C][N]<eom>",
            "sampled_selfies": "[C][C][N]",
            "action_token_ids": (4, 5),
            "stop_token": EOM_TOKEN,
            "termination_reason": "stop_token",
        },
        {
            "stage_text": "<bom>[C][O][O]<eom>",
            "sampled_selfies": "[C][O][O]",
            "action_token_ids": (6, 7),
            "stop_token": EOM_TOKEN,
            "termination_reason": "stop_token",
        },
    ]

    def run_with_rng(rng_values: list[float]) -> list:
        samples = iter(base_samples)
        monkeypatch.setattr(
            "post_training.gflownet.rollout.sample_stage",
            lambda *args, **kwargs: next(samples),
        )
        return sample_stage_trajectories_for_example(
            DummyModel(),
            tokenizer,
            {
                "id": "example-1",
                "prompt": "prompt",
                "description": "description",
                "target_selfies_list": ["[C][C][O]", "[C][C][N]", "[C][O][O]"],
            },
            rollout_id="rollout-1",
            generation_config=GFlowNetRolloutConfig(
                max_molecules_per_sequence=3,
                append_probability=0.30,
            ),
            reward_config=CHEBI20_REWARD_CONFIG,
            invalid_terminal_reward=1.0e-4,
            device=torch.device("cpu"),
            rng=FixedRandom(rng_values),
        )

    appended = run_with_rng([0.1, 0.1])
    assert len(appended) == 3
    assert [trajectory.stage_index for trajectory in appended] == [1, 2, 3]

    skipped = run_with_rng([0.9, 0.1])
    assert len(skipped) == 2
    assert [trajectory.stage_index for trajectory in skipped] == [1, 3]
    assert skipped[1].decoder_prefix_text == build_stage_prefix(["[C][C][O]", "[C][C][N]"])


def test_sample_stage_trajectories_for_example_can_keep_only_last_valid_stage(
    monkeypatch,
) -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[C][C][O]",
            3: "<eom>",
            4: "<bom>",
            5: "[C][C][N]",
            6: "<bom>",
            7: "[C][O][O]",
        },
        {EOM_TOKEN: 3},
    )

    class DummyModel:
        def __init__(self) -> None:
            self.policy_model = type(
                "Policy",
                (),
                {"config": type("Config", (), {"decoder_start_token_id": 0, "eos_token_id": 99})()},
            )()

    samples = iter(
        [
            {
                "stage_text": "<bom>[C][C][O]<eom>",
                "sampled_selfies": "[C][C][O]",
                "action_token_ids": (1, 2),
                "stop_token": EOM_TOKEN,
                "termination_reason": "stop_token",
            },
            {
                "stage_text": "<bom>[C][C][N]<eom>",
                "sampled_selfies": "[C][C][N]",
                "action_token_ids": (4, 5),
                "stop_token": EOM_TOKEN,
                "termination_reason": "stop_token",
            },
            {
                "stage_text": "<bom>[C][O][O]<eom>",
                "sampled_selfies": "[C][O][O]",
                "action_token_ids": (6, 7),
                "stop_token": EOM_TOKEN,
                "termination_reason": "stop_token",
            },
        ]
    )
    monkeypatch.setattr("post_training.gflownet.rollout.sample_stage", lambda *args, **kwargs: next(samples))

    trajectories = sample_stage_trajectories_for_example(
        DummyModel(),
        tokenizer,
        {
            "id": "example-1",
            "prompt": "prompt",
            "description": "description",
            "target_selfies_list": ["[C][C][O]", "[C][C][N]", "[C][O][O]"],
        },
        rollout_id="rollout-1",
        generation_config=GFlowNetRolloutConfig(
            max_molecules_per_sequence=3,
            append_probability=0.0,
        ),
        reward_config=CHEBI20_REWARD_CONFIG,
        invalid_terminal_reward=1.0e-4,
        device=torch.device("cpu"),
        rng=FixedRandom([]),
        return_last_valid_trajectory_only=True,
    )

    assert len(trajectories) == 1
    assert trajectories[0].stage_index == 3
    assert trajectories[0].decoder_prefix_text == build_stage_prefix(["[C][C][O]", "[C][C][N]"])
    assert trajectories[0].previous_sampled_selfies == ("[C][C][O]", "[C][C][N]")


def test_sample_stage_trajectories_for_example_last_only_skips_invalid_final_stage(
    monkeypatch,
) -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[C][C][O]",
            3: "<eom>",
            4: "<bom>",
            5: "[C][C][N]",
        },
        {EOM_TOKEN: 3},
    )

    class DummyModel:
        def __init__(self) -> None:
            self.policy_model = type(
                "Policy",
                (),
                {"config": type("Config", (), {"decoder_start_token_id": 0, "eos_token_id": 99})()},
            )()

    samples = iter(
        [
            {
                "stage_text": "<bom>[C][C][O]<eom>",
                "sampled_selfies": "[C][C][O]",
                "action_token_ids": (1, 2),
                "stop_token": EOM_TOKEN,
                "termination_reason": "stop_token",
            },
            {
                "stage_text": "<bom>[C][C][N]",
                "sampled_selfies": None,
                "action_token_ids": (4, 5),
                "stop_token": None,
                "termination_reason": "max_stage_new_tokens",
            },
        ]
    )
    monkeypatch.setattr("post_training.gflownet.rollout.sample_stage", lambda *args, **kwargs: next(samples))

    trajectories = sample_stage_trajectories_for_example(
        DummyModel(),
        tokenizer,
        {
            "id": "example-1",
            "prompt": "prompt",
            "description": "description",
            "target_selfies_list": ["[C][C][O]", "[C][C][N]"],
        },
        rollout_id="rollout-1",
        generation_config=GFlowNetRolloutConfig(
            max_molecules_per_sequence=3,
            append_probability=0.0,
        ),
        reward_config=CHEBI20_REWARD_CONFIG,
        invalid_terminal_reward=1.0e-4,
        device=torch.device("cpu"),
        rng=FixedRandom([]),
        return_last_valid_trajectory_only=True,
    )

    assert [trajectory.stage_index for trajectory in trajectories] == [1]
    assert trajectories[0].is_valid is True
    assert trajectories[0].termination_reason == "stop_token"


def test_sample_stage_trajectories_for_example_last_only_returns_empty_when_no_valid_stage(
    monkeypatch,
) -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<eom>",
        },
        {EOM_TOKEN: 1},
    )

    class DummyModel:
        def __init__(self) -> None:
            self.policy_model = type(
                "Policy",
                (),
                {"config": type("Config", (), {"decoder_start_token_id": 0, "eos_token_id": 99})()},
            )()

    monkeypatch.setattr(
        "post_training.gflownet.rollout.sample_stage",
        lambda *args, **kwargs: {
            "stage_text": "<eom>",
            "sampled_selfies": None,
            "action_token_ids": (),
            "stop_token": EOM_TOKEN,
            "termination_reason": "stop_token",
        },
    )

    trajectories = sample_stage_trajectories_for_example(
        DummyModel(),
        tokenizer,
        {
            "id": "example-1",
            "prompt": "prompt",
            "description": "description",
            "target_selfies_list": ["[C][C][O]"],
        },
        rollout_id="rollout-1",
        generation_config=GFlowNetRolloutConfig(
            max_molecules_per_sequence=3,
            append_probability=0.0,
            terminate_on_invalid_stage=True,
        ),
        reward_config=CHEBI20_REWARD_CONFIG,
        invalid_terminal_reward=1.0e-4,
        device=torch.device("cpu"),
        rng=FixedRandom([]),
        return_last_valid_trajectory_only=True,
    )

    assert trajectories == []


def test_sample_stage_trajectories_for_example_keeps_skipped_early_break_trajectory(
    monkeypatch,
) -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[C][C][O]",
            3: "<eom>",
            4: "<bom>",
            5: "[C][C][N]",
        },
        {EOM_TOKEN: 3},
    )

    class DummyModel:
        def __init__(self) -> None:
            self.policy_model = type(
                "Policy",
                (),
                {"config": type("Config", (), {"decoder_start_token_id": 0, "eos_token_id": 99})()},
            )()

    samples = iter(
        [
            {
                "stage_text": "<bom>[C][C][O]<eom>",
                "sampled_selfies": "[C][C][O]",
                "action_token_ids": (1, 2),
                "stop_token": EOM_TOKEN,
                "termination_reason": "stop_token",
            },
            {
                "stage_text": "<bom>[C][C][N]",
                "sampled_selfies": "[C][C][N]",
                "action_token_ids": (4, 5),
                "stop_token": None,
                "termination_reason": "max_stage_new_tokens",
            },
        ]
    )
    monkeypatch.setattr("post_training.gflownet.rollout.sample_stage", lambda *args, **kwargs: next(samples))

    trajectories = sample_stage_trajectories_for_example(
        DummyModel(),
        tokenizer,
        {
            "id": "example-1",
            "prompt": "prompt",
            "description": "description",
            "target_selfies_list": ["[C][C][O]", "[C][C][N]"],
        },
        rollout_id="rollout-1",
        generation_config=GFlowNetRolloutConfig(
            max_molecules_per_sequence=3,
            append_probability=0.30,
        ),
        reward_config=CHEBI20_REWARD_CONFIG,
        invalid_terminal_reward=1.0e-4,
        device=torch.device("cpu"),
        rng=FixedRandom([0.9]),
    )

    assert [trajectory.stage_index for trajectory in trajectories] == [1, 2]
    assert trajectories[1].termination_reason == "max_stage_new_tokens"


def test_sample_stage_trajectories_for_example_does_not_duplicate_appended_early_break_trajectory(
    monkeypatch,
) -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[C][C][O]",
            3: "<eom>",
            4: "<bom>",
            5: "[C][C][N]",
        },
        {EOM_TOKEN: 3},
    )

    class DummyModel:
        def __init__(self) -> None:
            self.policy_model = type(
                "Policy",
                (),
                {"config": type("Config", (), {"decoder_start_token_id": 0, "eos_token_id": 99})()},
            )()

    samples = iter(
        [
            {
                "stage_text": "<bom>[C][C][O]<eom>",
                "sampled_selfies": "[C][C][O]",
                "action_token_ids": (1, 2),
                "stop_token": EOM_TOKEN,
                "termination_reason": "stop_token",
            },
            {
                "stage_text": "<bom>[C][C][N]",
                "sampled_selfies": "[C][C][N]",
                "action_token_ids": (4, 5),
                "stop_token": None,
                "termination_reason": "max_stage_new_tokens",
            },
        ]
    )
    monkeypatch.setattr("post_training.gflownet.rollout.sample_stage", lambda *args, **kwargs: next(samples))

    trajectories = sample_stage_trajectories_for_example(
        DummyModel(),
        tokenizer,
        {
            "id": "example-1",
            "prompt": "prompt",
            "description": "description",
            "target_selfies_list": ["[C][C][O]", "[C][C][N]"],
        },
        rollout_id="rollout-1",
        generation_config=GFlowNetRolloutConfig(
            max_molecules_per_sequence=3,
            append_probability=0.30,
        ),
        reward_config=CHEBI20_REWARD_CONFIG,
        invalid_terminal_reward=1.0e-4,
        device=torch.device("cpu"),
        rng=FixedRandom([0.1]),
    )

    assert [trajectory.stage_index for trajectory in trajectories] == [1, 2]
    assert sum(trajectory.stage_index == 2 for trajectory in trajectories) == 1


def test_sample_stage_trajectories_for_example_uses_max_molecules_for_planned_stage_count(
    monkeypatch,
) -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[C][C][O]",
            3: "<eom>",
            4: "<bom>",
            5: "[C][C][N]",
            6: "<bom>",
            7: "[C][O][O]",
        },
        {EOM_TOKEN: 3},
    )

    class DummyModel:
        def __init__(self) -> None:
            self.policy_model = type(
                "Policy",
                (),
                {"config": type("Config", (), {"decoder_start_token_id": 0, "eos_token_id": 99})()},
            )()

    samples = iter(
        [
            {
                "stage_text": "<bom>[C][C][O]<eom>",
                "sampled_selfies": "[C][C][O]",
                "action_token_ids": (1, 2),
                "stop_token": EOM_TOKEN,
                "termination_reason": "stop_token",
            },
            {
                "stage_text": "<bom>[C][C][N]<eom>",
                "sampled_selfies": "[C][C][N]",
                "action_token_ids": (4, 5),
                "stop_token": EOM_TOKEN,
                "termination_reason": "stop_token",
            },
            {
                "stage_text": "<bom>[C][O][O]<eom>",
                "sampled_selfies": "[C][O][O]",
                "action_token_ids": (6, 7),
                "stop_token": EOM_TOKEN,
                "termination_reason": "stop_token",
            },
        ]
    )
    monkeypatch.setattr("post_training.gflownet.rollout.sample_stage", lambda *args, **kwargs: next(samples))

    trajectories = sample_stage_trajectories_for_example(
        DummyModel(),
        tokenizer,
        {
            "id": "example-1",
            "prompt": "prompt",
            "description": "description",
            "target_selfies_list": ["[C][C][O]"],
        },
        rollout_id="rollout-1",
        generation_config=GFlowNetRolloutConfig(
            max_molecules_per_sequence=3,
            append_probability=1.0,
        ),
        reward_config=CHEBI20_REWARD_CONFIG,
        invalid_terminal_reward=1.0e-4,
        device=torch.device("cpu"),
    )

    assert [trajectory.stage_index for trajectory in trajectories] == [1, 2, 3]


def test_sample_stage_keeps_raw_action_ids_when_projection_is_invalid() -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[C]",
            3: "<eom>",
        },
        {
            "<bom>": 1,
            "[C]": 2,
            EOM_TOKEN: 3,
        },
    )

    class DummyPolicyModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(eos_token_id=99)

        def __call__(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            decoder_input_ids: torch.Tensor,
            return_dict: bool,
        ) -> SimpleNamespace:
            del input_ids, attention_mask, return_dict
            logits = torch.full((1, decoder_input_ids.size(1), 8), -20.0)
            current_length = decoder_input_ids.size(1)
            if current_length == 1:
                logits[:, -1, 1] = 10.0
            else:
                logits[:, -1, 2] = 10.0
            return SimpleNamespace(logits=logits)

    class DummyModel:
        def __init__(self) -> None:
            self.policy_model = DummyPolicyModel()

    stage_sample = sample_stage(
        DummyModel(),
        tokenizer,
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        decoder_prefix_ids=torch.tensor([[0]], dtype=torch.long),
        generation_config=GFlowNetRolloutConfig(max_stage_new_tokens=2, top_p=1.0),
    )

    assert stage_sample["action_token_ids"] == (1, 2)
    assert stage_sample["sampled_selfies"] is None
    assert stage_sample["stop_token"] is None
    assert stage_sample["termination_reason"] == "max_stage_new_tokens"
    assert stage_sample["metadata"]["raw_action_token_ids"] == (1, 2)
    assert stage_sample["metadata"]["used_raw_action_ids_for_invalid_projection"] is True
    assert stage_sample["metadata"]["projection_failure_reason"] == "invalid_raw_stage_text"


def test_sample_stage_enforces_bom_and_masks_language_tokens() -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[C]",
            3: "<eom>",
            4: "ordinary",
        },
        {
            "<bom>": 1,
            "[C]": 2,
            EOM_TOKEN: 3,
            "ordinary": 4,
        },
    )
    token_constraints = StageTokenConstraints(
        bom_token_id=1,
        eom_token_id=3,
        content_token_ids=(2,),
    )

    class DummyPolicyModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(eos_token_id=99)

        def __call__(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            decoder_input_ids: torch.Tensor,
            return_dict: bool,
        ) -> SimpleNamespace:
            del input_ids, attention_mask, return_dict
            logits = torch.full((1, decoder_input_ids.size(1), 8), -20.0)
            current_length = decoder_input_ids.size(1)
            if current_length == 1:
                logits[:, -1, 4] = 10.0
                logits[:, -1, 1] = 0.0
            elif current_length == 2:
                logits[:, -1, 4] = 10.0
                logits[:, -1, 2] = 0.0
            else:
                logits[:, -1, 4] = 10.0
                logits[:, -1, 3] = 6.0
                logits[:, -1, 2] = 0.0
            return SimpleNamespace(logits=logits)

    class DummyModel:
        def __init__(self) -> None:
            self.policy_model = DummyPolicyModel()
            self._stage_token_constraints = token_constraints

        def get_stage_token_constraints(self):
            return self._stage_token_constraints

    stage_sample = sample_stage(
        DummyModel(),
        tokenizer,
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        decoder_prefix_ids=torch.tensor([[0]], dtype=torch.long),
        generation_config=GFlowNetRolloutConfig(max_stage_new_tokens=4),
    )

    assert stage_sample["action_token_ids"] == (1, 2)
    assert stage_sample["stop_token"] == EOM_TOKEN
    assert stage_sample["termination_reason"] == "stop_token"
    assert stage_sample["stage_text"] == "<bom>[C]<eom>"
    assert stage_sample["sampled_selfies"] == "[C]"
    assert stage_sample["metadata"]["projection_applied"] is True
    assert stage_sample["metadata"]["projection_failure_reason"] is None
    assert stage_sample["metadata"]["raw_action_token_ids"] == (1, 2)


def test_sample_stage_beam_search_selects_best_completed_eom_beam() -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[C]",
            3: "[O]",
            4: "<eom>",
        },
        {
            "<bom>": 1,
            "[C]": 2,
            "[O]": 3,
            EOM_TOKEN: 4,
        },
    )
    token_constraints = StageTokenConstraints(
        bom_token_id=1,
        eom_token_id=4,
        content_token_ids=(2, 3),
    )
    model = PrefixLogitModel(
        {
            (): {1: 8.0},
            (1,): {2: 8.0, 3: 7.0},
            (1, 2): {4: 9.0},
            (1, 3): {4: 1.0},
        },
        vocab_size=8,
    )

    stage_sample = sample_stage(
        model,
        tokenizer,
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        decoder_prefix_ids=torch.tensor([[0]], dtype=torch.long),
        generation_config=GFlowNetRolloutConfig(
            decoding_strategy="beam",
            num_beams=2,
            max_stage_new_tokens=4,
        ),
        stage_token_constraints=token_constraints,
    )

    assert stage_sample["stage_text"] == "<bom>[C]<eom>"
    assert stage_sample["sampled_selfies"] == "[C]"
    assert stage_sample["action_token_ids"] == (1, 2)
    assert stage_sample["stop_token"] == EOM_TOKEN
    assert stage_sample["termination_reason"] == "stop_token"
    assert stage_sample["metadata"]["decoding_strategy"] == "beam"
    assert stage_sample["metadata"]["num_beams"] == 2
    assert stage_sample["metadata"]["beam_rank"] == 0
    assert stage_sample["metadata"]["raw_action_token_ids"] == (1, 2)


def test_sample_stage_beam_search_cuts_beam_at_eom() -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[C]",
            3: "<eom>",
            4: "[O]",
        },
        {
            "<bom>": 1,
            "[C]": 2,
            EOM_TOKEN: 3,
            "[O]": 4,
        },
    )
    token_constraints = StageTokenConstraints(
        bom_token_id=1,
        eom_token_id=3,
        content_token_ids=(2, 4),
    )
    model = PrefixLogitModel(
        {
            (): {1: 8.0},
            (1,): {2: 8.0},
            (1, 2): {3: 8.0},
            (1, 2, 3): {4: 8.0},
        },
        vocab_size=8,
    )

    stage_sample = sample_stage(
        model,
        tokenizer,
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        decoder_prefix_ids=torch.tensor([[0]], dtype=torch.long),
        generation_config=GFlowNetRolloutConfig(
            decoding_strategy="beam",
            num_beams=1,
            max_stage_new_tokens=5,
        ),
        stage_token_constraints=token_constraints,
    )

    assert stage_sample["stage_text"] == "<bom>[C]<eom>"
    assert stage_sample["action_token_ids"] == (1, 2)
    assert stage_sample["termination_reason"] == "stop_token"
    assert model.policy_model.calls == [(), (1,), (1, 2)]


def test_sample_stage_beam_search_respects_stage_token_constraints() -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[C]",
            3: "<eom>",
            4: "ordinary",
        },
        {
            "<bom>": 1,
            "[C]": 2,
            EOM_TOKEN: 3,
            "ordinary": 4,
        },
    )
    token_constraints = StageTokenConstraints(
        bom_token_id=1,
        eom_token_id=3,
        content_token_ids=(2,),
    )
    model = PrefixLogitModel(
        {
            (): {4: 10.0, 1: 0.0},
            (1,): {4: 10.0, 2: 0.0},
            (1, 2): {4: 10.0, 3: 0.0},
        },
        vocab_size=8,
    )

    stage_sample = sample_stage(
        model,
        tokenizer,
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        decoder_prefix_ids=torch.tensor([[0]], dtype=torch.long),
        generation_config=GFlowNetRolloutConfig(
            decoding_strategy="beam",
            num_beams=2,
            max_stage_new_tokens=4,
        ),
        stage_token_constraints=token_constraints,
    )

    assert stage_sample["stage_text"] == "<bom>[C]<eom>"
    assert stage_sample["sampled_selfies"] == "[C]"
    assert stage_sample["action_token_ids"] == (1, 2)
    assert stage_sample["termination_reason"] == "stop_token"
    assert 4 not in stage_sample["metadata"]["raw_action_token_ids"]


def test_sample_stage_beam_search_stops_at_eos_token() -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[C]",
            3: "<eom>",
            5: "</s>",
        },
        {
            "<bom>": 1,
            "[C]": 2,
            EOM_TOKEN: 3,
            "</s>": 5,
        },
    )
    model = PrefixLogitModel(
        {
            (): {5: 10.0, 1: 0.0},
        },
        vocab_size=8,
        eos_token_id=5,
    )

    stage_sample = sample_stage(
        model,
        tokenizer,
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        decoder_prefix_ids=torch.tensor([[0]], dtype=torch.long),
        generation_config=GFlowNetRolloutConfig(
            decoding_strategy="beam",
            num_beams=1,
            max_stage_new_tokens=4,
        ),
    )

    assert stage_sample["sampled_selfies"] is None
    assert stage_sample["action_token_ids"] == ()
    assert stage_sample["stop_token"] == tokenizer.eos_token
    assert stage_sample["termination_reason"] == "eos_token"
    assert stage_sample["metadata"]["raw_action_token_ids"] == ()


def test_sample_stage_beam_search_returns_unfinished_stage_at_max_stage_tokens() -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[C]",
            3: "<eom>",
        },
        {
            "<bom>": 1,
            "[C]": 2,
            EOM_TOKEN: 3,
        },
    )
    token_constraints = StageTokenConstraints(
        bom_token_id=1,
        eom_token_id=3,
        content_token_ids=(2,),
    )
    model = PrefixLogitModel(
        {
            (): {1: 8.0},
            (1,): {2: 8.0},
            (1, 2): {3: 8.0},
        },
        vocab_size=8,
    )

    stage_sample = sample_stage(
        model,
        tokenizer,
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        decoder_prefix_ids=torch.tensor([[0]], dtype=torch.long),
        generation_config=GFlowNetRolloutConfig(
            decoding_strategy="beam",
            num_beams=1,
            max_stage_new_tokens=2,
        ),
        stage_token_constraints=token_constraints,
    )

    assert stage_sample["sampled_selfies"] is None
    assert stage_sample["action_token_ids"] == (1, 2)
    assert stage_sample["stop_token"] is None
    assert stage_sample["termination_reason"] == "max_stage_new_tokens"
    assert stage_sample["metadata"]["used_raw_action_ids_for_invalid_projection"] is True


def test_sample_stage_beam_search_returns_unfinished_stage_at_max_sequence_length() -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[C]",
            3: "<eom>",
        },
        {
            "<bom>": 1,
            "[C]": 2,
            EOM_TOKEN: 3,
        },
    )
    token_constraints = StageTokenConstraints(
        bom_token_id=1,
        eom_token_id=3,
        content_token_ids=(2,),
    )
    model = PrefixLogitModel(
        {
            (): {1: 8.0},
            (1,): {2: 8.0},
        },
        vocab_size=8,
    )

    stage_sample = sample_stage(
        model,
        tokenizer,
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        decoder_prefix_ids=torch.tensor([[0]], dtype=torch.long),
        generation_config=GFlowNetRolloutConfig(
            decoding_strategy="beam",
            num_beams=1,
            max_stage_new_tokens=4,
            max_sequence_length=4,
        ),
        stage_token_constraints=token_constraints,
    )

    assert stage_sample["sampled_selfies"] is None
    assert stage_sample["action_token_ids"] == (1,)
    assert stage_sample["stop_token"] is None
    assert stage_sample["termination_reason"] == "max_sequence_length"
    assert model.policy_model.calls == [()]


def test_sample_stage_explicit_sample_strategy_preserves_sampling_path() -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[C]",
            3: "<eom>",
        },
        {
            "<bom>": 1,
            "[C]": 2,
            EOM_TOKEN: 3,
        },
    )
    token_constraints = StageTokenConstraints(
        bom_token_id=1,
        eom_token_id=3,
        content_token_ids=(2,),
    )
    model = PrefixLogitModel(
        {
            (): {1: 8.0},
            (1,): {2: 8.0},
            (1, 2): {3: 8.0},
        },
        vocab_size=8,
    )

    stage_sample = sample_stage(
        model,
        tokenizer,
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        decoder_prefix_ids=torch.tensor([[0]], dtype=torch.long),
        generation_config=GFlowNetRolloutConfig(
            decoding_strategy="sample",
            num_beams=1,
            max_stage_new_tokens=4,
            top_p=1.0,
        ),
        stage_token_constraints=token_constraints,
    )

    assert stage_sample["stage_text"] == "<bom>[C]<eom>"
    assert stage_sample["termination_reason"] == "stop_token"
    assert "beam_score" not in stage_sample["metadata"]


def test_sample_stage_projects_explicit_hydrogens_to_no_h_prefix_sequence() -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[CH",
            3: "4]",
            4: "<eom>",
            5: "[C]",
            6: "ordinary",
        },
        {
            "<bom>": 1,
            "[CH": 2,
            "4]": 3,
            EOM_TOKEN: 4,
            "[C]": 5,
            "ordinary": 6,
        },
    )
    token_constraints = StageTokenConstraints(
        bom_token_id=1,
        eom_token_id=4,
        content_token_ids=(2, 3, 5),
    )

    class DummyPolicyModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(eos_token_id=99)

        def __call__(
            self,
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            decoder_input_ids: torch.Tensor,
            return_dict: bool,
        ) -> SimpleNamespace:
            del input_ids, attention_mask, return_dict
            logits = torch.full((1, decoder_input_ids.size(1), 12), -20.0)
            current_length = decoder_input_ids.size(1)
            if current_length == 1:
                logits[:, -1, 1] = 0.0
            elif current_length == 2:
                logits[:, -1, 2] = 0.0
            elif current_length == 3:
                logits[:, -1, 3] = 0.0
            else:
                logits[:, -1, 4] = 6.0
                logits[:, -1, 5] = 0.0
            return SimpleNamespace(logits=logits)

    class DummyModel:
        def __init__(self) -> None:
            self.policy_model = DummyPolicyModel()
            self._stage_token_constraints = token_constraints

        def get_stage_token_constraints(self):
            return self._stage_token_constraints

    stage_sample = sample_stage(
        DummyModel(),
        tokenizer,
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        decoder_prefix_ids=torch.tensor([[0]], dtype=torch.long),
        generation_config=GFlowNetRolloutConfig(max_stage_new_tokens=5),
    )

    assert stage_sample["stage_text"] == "<bom>[C]<eom>"
    assert stage_sample["sampled_selfies"] == "[C]"
    assert stage_sample["action_token_ids"] == (1, 5)
    assert stage_sample["stop_token"] == EOM_TOKEN
    assert stage_sample["termination_reason"] == "stop_token"
    assert stage_sample["metadata"]["raw_stage_text"] == "<bom>[CH4]<eom>"
    assert stage_sample["metadata"]["raw_sampled_selfies"] == "[CH4]"
    assert stage_sample["metadata"]["raw_action_token_ids"] == (1, 2, 3)
    assert stage_sample["metadata"]["projection_applied"] is True
    assert stage_sample["metadata"]["projection_changed"] is True
    assert stage_sample["metadata"]["projection_failure_reason"] is None
    assert len(stage_sample["metadata"]["raw_action_token_ids"]) > len(stage_sample["action_token_ids"])

    trajectory = build_sampled_stage_trajectory_from_generation(
        example={
            "id": "example-1",
            "prompt": "prompt",
            "description": "description",
            "target_selfies_list": ["[C]"],
        },
        rollout_id="rollout-1",
        stage_index=1,
        decoder_prefix_text="",
        previous_sampled_selfies=(),
        stage_text=stage_sample["stage_text"],
        sampled_selfies=stage_sample["sampled_selfies"],
        action_token_ids=stage_sample["action_token_ids"],
        metadata=stage_sample["metadata"],
        stop_token=stage_sample["stop_token"],
        termination_reason=stage_sample["termination_reason"],
        reward_config=CHEBI20_REWARD_CONFIG,
        invalid_terminal_reward=1.0e-4,
    )

    assert trajectory.action_token_ids == (1, 5)
    assert trajectory.prefix_states == ((), (1,), (1, 5))
    assert len(trajectory.prefix_rewards) == len(trajectory.action_token_ids) + 1
    assert trajectory.metadata["raw_action_token_ids"] == (1, 2, 3)
