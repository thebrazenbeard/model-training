from __future__ import annotations


def assistant_only_labels(input_ids, prompt_token_count: int):
    if prompt_token_count < 0 or prompt_token_count > len(input_ids):
        raise ValueError("prompt_token_count is outside input_ids")
    return [-100] * prompt_token_count + list(input_ids[prompt_token_count:])
