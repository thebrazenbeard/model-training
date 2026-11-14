from successor.training_data import assistant_only_labels


def test_assistant_only_labels_mask_prompt_prefix():
    input_ids = [10, 11, 12, 13, 14]
    labels = assistant_only_labels(input_ids, prompt_token_count=3)
    assert labels == [-100, -100, -100, 13, 14]


def test_assistant_only_labels_rejects_invalid_boundary():
    import pytest
    with pytest.raises(ValueError):
        assistant_only_labels([1, 2], prompt_token_count=3)
