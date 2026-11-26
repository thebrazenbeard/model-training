import torch

from successor.preference_objective import (
    offline_dpo_margin_coefficient,
    offline_dpo_surrogate,
    pad_labeled_sequences,
)


def test_offline_dpo_surrogate_gradient_matches_serial_coefficients():
    chosen = torch.tensor([-2.0, -3.0, -4.0], requires_grad=True)
    rejected = torch.tensor([-3.5, -2.5, -5.0], requires_grad=True)
    ref_chosen = torch.tensor([-2.2, -3.1, -4.4])
    ref_rejected = torch.tensor([-3.2, -2.7, -4.8])

    expected = offline_dpo_margin_coefficient(
        chosen.detach(), rejected.detach(), ref_chosen, ref_rejected, beta=0.1
    )
    loss = offline_dpo_surrogate(
        chosen, rejected, ref_chosen, ref_rejected, beta=0.1
    )
    loss.backward()

    assert torch.allclose(chosen.grad, expected)
    assert torch.allclose(rejected.grad, -expected)


def test_pad_labeled_sequences_preserves_masks_and_padding():
    ids = [
        torch.tensor([10, 11, 12, 13]),
        torch.tensor([20, 21]),
    ]
    labels = [
        torch.tensor([-100, -100, 12, 13]),
        torch.tensor([-100, 21]),
    ]

    batch_ids, batch_labels, attention = pad_labeled_sequences(
        ids, labels, pad_token_id=0
    )

    assert batch_ids.tolist() == [[10, 11, 12, 13], [20, 21, 0, 0]]
    assert batch_labels.tolist() == [[-100, -100, 12, 13], [-100, 21, -100, -100]]
    assert attention.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
