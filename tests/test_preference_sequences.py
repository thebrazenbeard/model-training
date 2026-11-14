import pytest

from successor.preference_objective import completion_labels


def test_completion_labels_masks_exact_prompt_prefix():
    labels = completion_labels([1, 2, 3], [1, 2, 3, 4, 5])
    assert labels == [-100, -100, -100, 4, 5]


def test_completion_labels_rejects_prefix_mismatch():
    with pytest.raises(ValueError, match="prefix"):
        completion_labels([1, 9], [1, 2, 3])


def test_completion_labels_rejects_no_completion():
    with pytest.raises(ValueError, match="completion"):
        completion_labels([1, 2], [1, 2])
