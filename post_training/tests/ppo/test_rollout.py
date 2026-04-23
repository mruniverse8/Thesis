import pytest
import torch
from types import SimpleNamespace

pytest.importorskip("transformers")

from transformers import T5Config, T5ForConditionalGeneration

from src.constants import EOM_TOKEN

from post_training.ppo.config import RolloutGenerationConfig
from post_training.ppo.rollout import compute_action_stats, sample_rollout_for_example, sample_stage
from post_training.shared.decoding import (
    StageTokenConstraints,
    mask_logits_to_allowed_token_ids,
)
from post_training.shared.sequence import build_stage_prefix


def _compute_action_stats_naively(
    model: T5ForConditionalGeneration,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    action_token_ids: list[int],
    stage_token_constraints: StageTokenConstraints | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    running_decoder_input_ids = decoder_input_ids
    total_logprob = torch.zeros((), device=input_ids.device)
    total_entropy = torch.zeros((), device=input_ids.device)

    for token_id in action_token_ids:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=running_decoder_input_ids,
            return_dict=True,
        )
        next_logits = outputs.logits[:, -1, :]
        if stage_token_constraints is not None and stage_token_constraints.enabled:
            allowed_token_ids = stage_token_constraints.allowed_token_ids_for_prefix(
                action_token_ids[: len(running_decoder_input_ids[0]) - len(decoder_input_ids[0])]
            )
            next_logits = mask_logits_to_allowed_token_ids(next_logits, allowed_token_ids)
        log_probs = torch.log_softmax(next_logits, dim=-1)
        probabilities = torch.softmax(next_logits, dim=-1)
        total_logprob = total_logprob + log_probs[0, token_id]
        entropy_terms = torch.where(
            probabilities > 0,
            probabilities * log_probs,
            torch.zeros_like(probabilities),
        )
        total_entropy = total_entropy - entropy_terms.sum(dim=-1).squeeze(0)

        next_token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=input_ids.device)
        running_decoder_input_ids = torch.cat([running_decoder_input_ids, next_token_tensor], dim=1)

    return total_logprob, total_entropy


def test_compute_action_stats_matches_naive_teacher_forcing() -> None:
    torch.manual_seed(0)
    config = T5Config(
        vocab_size=32,
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    model = T5ForConditionalGeneration(config)
    model.eval()

    input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    decoder_input_ids = torch.tensor([[0, 6, 7]], dtype=torch.long)
    action_token_ids = [8, 9, 10]

    with torch.no_grad():
        optimized = compute_action_stats(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            action_token_ids=action_token_ids,
        )
        naive = _compute_action_stats_naively(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            action_token_ids=action_token_ids,
        )

    assert torch.allclose(optimized[0], naive[0], atol=1.0e-6)
    assert torch.allclose(optimized[1], naive[1], atol=1.0e-6)


def test_compute_action_stats_returns_zeros_for_empty_action_sequence() -> None:
    config = T5Config(
        vocab_size=32,
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    model = T5ForConditionalGeneration(config)

    input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    decoder_input_ids = torch.tensor([[0, 6, 7]], dtype=torch.long)

    logprob_sum, entropy_sum = compute_action_stats(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        action_token_ids=[],
    )

    assert logprob_sum.item() == 0.0
    assert entropy_sum.item() == 0.0


def test_compute_action_stats_matches_naive_teacher_forcing_with_constraints() -> None:
    torch.manual_seed(0)
    config = T5Config(
        vocab_size=32,
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    model = T5ForConditionalGeneration(config)
    model.eval()

    input_ids = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    decoder_input_ids = torch.tensor([[0]], dtype=torch.long)
    action_token_ids = [6, 8, 7]
    token_constraints = StageTokenConstraints(
        bom_token_id=6,
        eom_token_id=7,
        content_token_ids=(8,),
    )

    with torch.no_grad():
        optimized = compute_action_stats(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            action_token_ids=action_token_ids,
            stage_token_constraints=token_constraints,
        )
        naive = _compute_action_stats_naively(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            action_token_ids=action_token_ids,
            stage_token_constraints=token_constraints,
        )

    assert torch.allclose(optimized[0], naive[0], atol=1.0e-6)
    assert torch.allclose(optimized[1], naive[1], atol=1.0e-6)


class DummyTokenizer:
    def __init__(self, id_to_token: dict[int, str], token_to_id: dict[str, int]) -> None:
        self.id_to_token = dict(id_to_token)
        self.token_to_id = dict(token_to_id)

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
        return_tensors: str | None = None,
    ):
        del add_special_tokens, return_attention_mask
        remaining = str(text)
        token_ids: list[int] = []
        known_tokens = sorted(self.token_to_id, key=len, reverse=True)
        while remaining:
            matched_token = next(
                (token for token in known_tokens if remaining.startswith(token)),
                None,
            )
            if matched_token is None:
                raise ValueError(f"Unable to tokenize {text!r}; next chunk was {remaining!r}.")
            token_ids.append(int(self.token_to_id[matched_token]))
            remaining = remaining[len(matched_token) :]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}
        return {"input_ids": token_ids}

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


def _patch_rollout_encoders(monkeypatch) -> None:
    monkeypatch.setattr(
        "post_training.ppo.rollout.encode_prompt",
        lambda *args, **kwargs: {
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
        },
    )
    monkeypatch.setattr(
        "post_training.ppo.rollout.encode_decoder_prefix",
        lambda *args, **kwargs: torch.tensor([[0]], dtype=torch.long),
    )


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
            self.policy_model = self
            self._stage_token_constraints = token_constraints

        def get_stage_token_constraints(self):
            return self._stage_token_constraints

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

    stage_sample = sample_stage(
        DummyPolicyModel(),
        tokenizer,
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        decoder_prefix_ids=torch.tensor([[0]], dtype=torch.long),
        generation_config=RolloutGenerationConfig(max_stage_new_tokens=4),
    )

    assert stage_sample["action_token_ids"] == [1, 2, 3]
    assert stage_sample["stop_token"] == EOM_TOKEN
    assert stage_sample["termination_reason"] == "stop_token"
    assert stage_sample["stage_text"] == "<bom>[C]<eom>"
    assert stage_sample["sampled_selfies"] == "[C]"
    assert stage_sample["metadata"]["projection_applied"] is True
    assert stage_sample["metadata"]["projection_failure_reason"] is None
    assert stage_sample["metadata"]["raw_action_token_ids"] == (1, 2, 3)


def test_sample_stage_projects_explicit_hydrogens_to_no_h_action_ids() -> None:
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
            self.policy_model = self
            self._stage_token_constraints = token_constraints

        def get_stage_token_constraints(self):
            return self._stage_token_constraints

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

    stage_sample = sample_stage(
        DummyPolicyModel(),
        tokenizer,
        input_ids=torch.tensor([[1, 2]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1]], dtype=torch.long),
        decoder_prefix_ids=torch.tensor([[0]], dtype=torch.long),
        generation_config=RolloutGenerationConfig(max_stage_new_tokens=5),
    )

    assert stage_sample["stage_text"] == "<bom>[C]<eom>"
    assert stage_sample["sampled_selfies"] == "[C]"
    assert stage_sample["action_token_ids"] == [1, 5, 4]
    assert stage_sample["stop_token"] == EOM_TOKEN
    assert stage_sample["termination_reason"] == "stop_token"
    assert stage_sample["metadata"]["raw_stage_text"] == "<bom>[CH4]<eom>"
    assert stage_sample["metadata"]["raw_sampled_selfies"] == "[CH4]"
    assert stage_sample["metadata"]["raw_action_token_ids"] == (1, 2, 3, 4)
    assert stage_sample["metadata"]["projection_applied"] is True
    assert stage_sample["metadata"]["projection_changed"] is True
    assert stage_sample["metadata"]["projection_failure_reason"] is None
    assert len(stage_sample["metadata"]["raw_action_token_ids"]) > len(stage_sample["action_token_ids"])


def test_sample_rollout_for_example_continues_after_invalid_stage_when_sampling_stops_normally(
    monkeypatch,
) -> None:
    class DummyPolicyValueModel:
        def __init__(self) -> None:
            self.policy_model = SimpleNamespace(config=SimpleNamespace(decoder_start_token_id=0))

        def compute_values(self, **kwargs) -> torch.Tensor:
            del kwargs
            return torch.tensor([0.0], dtype=torch.float32)

    samples = iter(
        [
            {
                "stage_text": "<eom>",
                "sampled_selfies": None,
                "action_token_ids": (),
                "action_logprob_sum": 0.0,
                "entropy_sum": 0.0,
                "stop_token": EOM_TOKEN,
                "termination_reason": "stop_token",
                "metadata": {},
            },
            {
                "stage_text": "<bom>[C][C][N]<eom>",
                "sampled_selfies": "[C][C][N]",
                "action_token_ids": [4, 5],
                "action_logprob_sum": 0.0,
                "entropy_sum": 0.0,
                "stop_token": EOM_TOKEN,
                "termination_reason": "stop_token",
                "metadata": {},
            },
        ]
    )

    def fake_reward_breakdown(candidate_selfies, *, targets, previous_candidates, config):
        del targets, previous_candidates, config
        is_valid = bool(candidate_selfies)
        reward = 2.0 if is_valid else 1.0e-4
        return SimpleNamespace(
            amplified_reward=reward,
            total_reward=reward,
            match=SimpleNamespace(reward=reward / 2.0),
            diversity=SimpleNamespace(reward=0.0),
            is_duplicate=False,
            candidate=SimpleNamespace(is_valid=is_valid, canonical_smiles="C"),
        )

    _patch_rollout_encoders(monkeypatch)
    monkeypatch.setattr("post_training.ppo.rollout.sample_stage", lambda *args, **kwargs: next(samples))
    monkeypatch.setattr(
        "post_training.ppo.rollout.compute_action_stats",
        lambda *args, **kwargs: (torch.zeros(()), torch.zeros(())),
    )
    monkeypatch.setattr("post_training.ppo.rollout.score_stage_reward", fake_reward_breakdown)

    trajectories = sample_rollout_for_example(
        DummyPolicyValueModel(),
        object(),
        object(),
        {
            "id": "example-1",
            "prompt": "prompt",
            "description": "description",
            "target_selfies_list": ["[C][C][O]", "[C][C][N]"],
        },
        rollout_id="rollout-1",
        generation_config=RolloutGenerationConfig(
            max_molecules_per_sequence=2,
            append_probability=1.0,
        ),
        device=torch.device("cpu"),
        rng=FixedRandom([0.0]),
    )

    assert len(trajectories) == 2
    assert trajectories[0].is_valid is False
    assert trajectories[1].decoder_prefix_text == ""


def test_sample_rollout_for_example_trusts_invalid_sampled_selfies_for_prefix_history(
    monkeypatch,
) -> None:
    class DummyPolicyValueModel:
        def __init__(self) -> None:
            self.policy_model = SimpleNamespace(config=SimpleNamespace(decoder_start_token_id=0))

        def compute_values(self, **kwargs) -> torch.Tensor:
            del kwargs
            return torch.tensor([0.0], dtype=torch.float32)

    samples = iter(
        [
            {
                "stage_text": "<bom>[C][C][O]<eom>",
                "sampled_selfies": "[C][C][O]",
                "action_token_ids": [1, 2],
                "action_logprob_sum": 0.0,
                "entropy_sum": 0.0,
                "stop_token": EOM_TOKEN,
                "termination_reason": "stop_token",
                "metadata": {},
            },
            {
                "stage_text": "<bom>[C][C][N]<eom>",
                "sampled_selfies": "[C][C][N]",
                "action_token_ids": [4, 5],
                "action_logprob_sum": 0.0,
                "entropy_sum": 0.0,
                "stop_token": EOM_TOKEN,
                "termination_reason": "stop_token",
                "metadata": {},
            },
        ]
    )

    def fake_reward_breakdown(candidate_selfies, *, targets, previous_candidates, config):
        del targets, config
        is_valid = candidate_selfies == "[C][C][N]"
        reward = 2.0 if is_valid else 1.0e-4
        return SimpleNamespace(
            amplified_reward=reward,
            total_reward=reward,
            match=SimpleNamespace(reward=reward / 2.0),
            diversity=SimpleNamespace(reward=0.0),
            is_duplicate=False,
            candidate=SimpleNamespace(is_valid=is_valid, canonical_smiles="C"),
            previous_candidates=tuple(previous_candidates),
        )

    _patch_rollout_encoders(monkeypatch)
    monkeypatch.setattr("post_training.ppo.rollout.sample_stage", lambda *args, **kwargs: next(samples))
    monkeypatch.setattr(
        "post_training.ppo.rollout.compute_action_stats",
        lambda *args, **kwargs: (torch.zeros(()), torch.zeros(())),
    )
    monkeypatch.setattr("post_training.ppo.rollout.score_stage_reward", fake_reward_breakdown)

    trajectories = sample_rollout_for_example(
        DummyPolicyValueModel(),
        object(),
        object(),
        {
            "id": "example-1",
            "prompt": "prompt",
            "description": "description",
            "target_selfies_list": ["[C][C][O]", "[C][C][N]"],
        },
        rollout_id="rollout-1",
        generation_config=RolloutGenerationConfig(
            max_molecules_per_sequence=2,
            append_probability=1.0,
        ),
        device=torch.device("cpu"),
        rng=FixedRandom([0.0]),
    )

    assert len(trajectories) == 2
    assert trajectories[0].is_valid is False
    assert trajectories[1].decoder_prefix_text == build_stage_prefix(["[C][C][O]"])


def test_sample_rollout_for_example_appends_stage_two_plus_only_when_probability_allows_it(
    monkeypatch,
) -> None:
    class DummyPolicyValueModel:
        def __init__(self) -> None:
            self.policy_model = SimpleNamespace(config=SimpleNamespace(decoder_start_token_id=0))

        def compute_values(self, **kwargs) -> torch.Tensor:
            del kwargs
            return torch.tensor([0.0], dtype=torch.float32)

    base_samples = [
        {
            "stage_text": "<bom>[C][C][O]<eom>",
            "sampled_selfies": "[C][C][O]",
            "action_token_ids": [1, 2],
            "action_logprob_sum": 0.0,
            "entropy_sum": 0.0,
            "stop_token": EOM_TOKEN,
            "termination_reason": "stop_token",
            "metadata": {},
        },
        {
            "stage_text": "<bom>[C][C][N]<eom>",
            "sampled_selfies": "[C][C][N]",
            "action_token_ids": [4, 5],
            "action_logprob_sum": 0.0,
            "entropy_sum": 0.0,
            "stop_token": EOM_TOKEN,
            "termination_reason": "stop_token",
            "metadata": {},
        },
        {
            "stage_text": "<bom>[C][O][O]<eom>",
            "sampled_selfies": "[C][O][O]",
            "action_token_ids": [6, 7],
            "action_logprob_sum": 0.0,
            "entropy_sum": 0.0,
            "stop_token": EOM_TOKEN,
            "termination_reason": "stop_token",
            "metadata": {},
        },
    ]

    def fake_reward_breakdown(candidate_selfies, *, targets, previous_candidates, config):
        del targets, previous_candidates, config
        return SimpleNamespace(
            amplified_reward=2.0,
            total_reward=2.0,
            match=SimpleNamespace(reward=1.0),
            diversity=SimpleNamespace(reward=0.0),
            is_duplicate=False,
            candidate=SimpleNamespace(is_valid=bool(candidate_selfies), canonical_smiles="C"),
        )

    def run_with_rng(rng_values: list[float]) -> list:
        samples = iter(base_samples)
        _patch_rollout_encoders(monkeypatch)
        monkeypatch.setattr("post_training.ppo.rollout.sample_stage", lambda *args, **kwargs: next(samples))
        monkeypatch.setattr(
            "post_training.ppo.rollout.compute_action_stats",
            lambda *args, **kwargs: (torch.zeros(()), torch.zeros(())),
        )
        monkeypatch.setattr("post_training.ppo.rollout.score_stage_reward", fake_reward_breakdown)
        return sample_rollout_for_example(
            DummyPolicyValueModel(),
            object(),
            object(),
            {
                "id": "example-1",
                "prompt": "prompt",
                "description": "description",
                "target_selfies_list": ["[C][C][O]", "[C][C][N]", "[C][O][O]"],
            },
            rollout_id="rollout-1",
            generation_config=RolloutGenerationConfig(
                max_molecules_per_sequence=3,
                append_probability=0.30,
            ),
            device=torch.device("cpu"),
            rng=FixedRandom(rng_values),
        )

    appended = run_with_rng([0.1, 0.1])
    assert [trajectory.stage_index for trajectory in appended] == [1, 2, 3]

    skipped = run_with_rng([0.9, 0.1])
    assert [trajectory.stage_index for trajectory in skipped] == [1, 3]
    assert skipped[1].decoder_prefix_text == build_stage_prefix(["[C][C][O]", "[C][C][N]"])


def test_sample_rollout_for_example_uses_max_molecules_for_planned_stage_count(
    monkeypatch,
) -> None:
    class DummyPolicyValueModel:
        def __init__(self) -> None:
            self.policy_model = SimpleNamespace(config=SimpleNamespace(decoder_start_token_id=0))

        def compute_values(self, **kwargs) -> torch.Tensor:
            del kwargs
            return torch.tensor([0.0], dtype=torch.float32)

    samples = iter(
        [
            {
                "stage_text": "<bom>[C][C][O]<eom>",
                "sampled_selfies": "[C][C][O]",
                "action_token_ids": [1, 2],
                "action_logprob_sum": 0.0,
                "entropy_sum": 0.0,
                "stop_token": EOM_TOKEN,
                "termination_reason": "stop_token",
                "metadata": {},
            },
            {
                "stage_text": "<bom>[C][C][N]<eom>",
                "sampled_selfies": "[C][C][N]",
                "action_token_ids": [4, 5],
                "action_logprob_sum": 0.0,
                "entropy_sum": 0.0,
                "stop_token": EOM_TOKEN,
                "termination_reason": "stop_token",
                "metadata": {},
            },
            {
                "stage_text": "<bom>[C][O][O]<eom>",
                "sampled_selfies": "[C][O][O]",
                "action_token_ids": [6, 7],
                "action_logprob_sum": 0.0,
                "entropy_sum": 0.0,
                "stop_token": EOM_TOKEN,
                "termination_reason": "stop_token",
                "metadata": {},
            },
        ]
    )

    def fake_reward_breakdown(candidate_selfies, *, targets, previous_candidates, config):
        del targets, previous_candidates, config
        return SimpleNamespace(
            amplified_reward=2.0,
            total_reward=2.0,
            match=SimpleNamespace(reward=1.0),
            diversity=SimpleNamespace(reward=0.0),
            is_duplicate=False,
            candidate=SimpleNamespace(is_valid=bool(candidate_selfies), canonical_smiles="C"),
        )

    _patch_rollout_encoders(monkeypatch)
    monkeypatch.setattr("post_training.ppo.rollout.sample_stage", lambda *args, **kwargs: next(samples))
    monkeypatch.setattr(
        "post_training.ppo.rollout.compute_action_stats",
        lambda *args, **kwargs: (torch.zeros(()), torch.zeros(())),
    )
    monkeypatch.setattr("post_training.ppo.rollout.score_stage_reward", fake_reward_breakdown)

    trajectories = sample_rollout_for_example(
        DummyPolicyValueModel(),
        object(),
        object(),
        {
            "id": "example-1",
            "prompt": "prompt",
            "description": "description",
            "target_selfies_list": ["[C][C][O]"],
        },
        rollout_id="rollout-1",
        generation_config=RolloutGenerationConfig(
            max_molecules_per_sequence=3,
            append_probability=1.0,
        ),
        device=torch.device("cpu"),
        rng=FixedRandom([0.0, 0.0]),
    )

    assert [trajectory.stage_index for trajectory in trajectories] == [1, 2, 3]
