import torch

from successor.preference_objective import masked_sequence_logprob


def test_masked_sequence_logprob_ignores_prompt_tokens():
    logits = torch.tensor([[[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]]])
    labels = torch.tensor([[-100, 1, 0]])
    score = masked_sequence_logprob(logits, labels)
    logp = torch.log_softmax(logits, dim=-1)
    expected = logp[0, 0, 1] + logp[0, 1, 0]
    assert torch.allclose(score, expected.unsqueeze(0))


def test_masked_sequence_logprob_rejects_empty_supervision():
    logits = torch.zeros((1, 3, 2))
    labels = torch.full((1, 3), -100)
    try:
        masked_sequence_logprob(logits, labels)
    except ValueError as exc:
        assert "supervised" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")
