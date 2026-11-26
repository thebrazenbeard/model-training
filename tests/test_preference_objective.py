import torch

from successor.preference_objective import offline_dpo_loss


def test_offline_dpo_is_neutral_when_policy_matches_reference():
    loss = offline_dpo_loss(
        policy_chosen=torch.tensor([3.0]),
        policy_rejected=torch.tensor([1.0]),
        reference_chosen=torch.tensor([3.0]),
        reference_rejected=torch.tensor([1.0]),
        beta=0.1,
    )
    assert torch.allclose(loss, torch.tensor(0.69314718), atol=1e-6)


def test_offline_dpo_rewards_better_chosen_margin():
    neutral = offline_dpo_loss(torch.tensor([3.0]), torch.tensor([1.0]), torch.tensor([3.0]), torch.tensor([1.0]), beta=0.1)
    improved = offline_dpo_loss(torch.tensor([4.0]), torch.tensor([1.0]), torch.tensor([3.0]), torch.tensor([1.0]), beta=0.1)
    assert improved < neutral


def test_margin_coefficient_matches_direct_autograd():
    from successor.preference_objective import offline_dpo_margin_coefficient
    pc=torch.tensor([1.2],requires_grad=True)
    pr=torch.tensor([0.4],requires_grad=True)
    rc=torch.tensor([0.7])
    rr=torch.tensor([0.3])
    direct=offline_dpo_loss(pc,pr,rc,rr,beta=0.2)
    direct.backward()
    expected_pc=pc.grad.detach().clone()
    expected_pr=pr.grad.detach().clone()
    coeff=offline_dpo_margin_coefficient(pc.detach(),pr.detach(),rc,rr,beta=0.2)
    assert torch.allclose(coeff,expected_pc)
    assert torch.allclose(-coeff,expected_pr)
