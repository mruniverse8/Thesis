import pytest
import torch

pytest.importorskip("selfies")
pytest.importorskip("rdkit")

from reward_utils.defaults import CHEBI20_REWARD_CONFIG
from src.constants import EOM_TOKEN

from post_training.gflownet.config import GFlowNetRolloutConfig
from post_training.gflownet.rollout import (
    build_sampled_stage_trajectory_from_generation,
    sample_stage_trajectories_for_example,
)
from post_training.shared.sequence import build_stage_prefix


class DummyTokenizer:
    def __init__(self, id_to_token: dict[int, str], token_to_id: dict[str, int]) -> None:
        self.id_to_token = id_to_token
        self.token_to_id = token_to_id
        self.eos_token = "</s>"

    def __call__(
        self,
        _text,
        truncation=False,
        max_length=None,
        return_tensors=None,
        add_special_tokens=True,
    ):
        del truncation, max_length, return_tensors, add_special_tokens
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(self.id_to_token[int(token_id)] for token_id in token_ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return int(self.token_to_id[token])


def test_build_sampled_stage_trajectory_from_generation_keeps_stop_out_of_action_ids() -> None:
    tokenizer = DummyTokenizer(
        {
            1: "<bom>",
            2: "[C][C][O]",
            3: "<eom>",
        },
        {EOM_TOKEN: 3},
    )
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
        previous_valid_selfies=(),
        tokenizer=tokenizer,
        action_token_ids=(1, 2),
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


def test_sample_stage_trajectories_for_example_updates_prefix_after_valid_stage(monkeypatch) -> None:
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
        generation_config=GFlowNetRolloutConfig(max_molecules_per_sequence=2),
        reward_config=CHEBI20_REWARD_CONFIG,
        invalid_terminal_reward=1.0e-4,
        device=torch.device("cpu"),
    )

    assert len(trajectories) == 2
    assert trajectories[0].decoder_prefix_text == ""
    assert trajectories[1].decoder_prefix_text == build_stage_prefix(["[C][C][O]"])


def test_sample_stage_trajectories_for_example_stops_after_invalid_stage_when_configured(
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
            terminate_on_invalid_stage=True,
        ),
        reward_config=CHEBI20_REWARD_CONFIG,
        invalid_terminal_reward=1.0e-4,
        device=torch.device("cpu"),
    )

    assert len(trajectories) == 1
    assert trajectories[0].is_valid is False
