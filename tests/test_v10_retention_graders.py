from __future__ import annotations

from successor.experiments.retention_graders import (
    check_instruction_contract,
    invalid_instruction_output,
    reference_coding_source,
    reference_instruction_output,
    run_python_test_contract,
)


def test_instruction_contract_reference_passes_and_invalid_fails() -> None:
    contracts = [
        {"type": "token_exact_count", "token": "V001", "count": 2},
        {"type": "prefix_suffix", "prefix": "START:", "suffix": ":END"},
        {"type": "line_prefixes", "prefixes": ["A-", "B-", "C-"]},
        {"type": "json_object_exact_keys", "keys": ["a", "b"]},
        {
            "type": "forbidden_character_and_required_token",
            "forbidden": ",",
            "required_token": "V002",
        },
    ]
    for contract in contracts:
        assert check_instruction_contract(
            reference_instruction_output(contract), contract
        )
        assert not check_instruction_contract(
            invalid_instruction_output(contract), contract
        )


def test_all_coding_reference_families_execute_contracts() -> None:
    fixtures = {
        "rotate_left": ("solve_x", ["assert solve_x([1,2,3], 1) == [2,3,1]"]),
        "count_vowels": ("solve_x", ["assert solve_x('AbcE') == 2"]),
        "unique_preserve": ("solve_x", ["assert solve_x([1,2,1]) == [1,2]"]),
        "digit_sum": ("solve_x", ["assert solve_x(1234) == 10"]),
        "reverse_words": ("solve_x", ["assert solve_x('a b c') == 'c b a'"]),
        "clamp_values": ("solve_x", ["assert solve_x([-1,3,9],0,5) == [0,3,5]"]),
        "pairwise_sums": ("solve_x", ["assert solve_x([1,2,4]) == [3,6]"]),
        "flatten_once": ("solve_x", ["assert solve_x([[1],[2,3]]) == [1,2,3]"]),
        "chunk_list": ("solve_x", ["assert solve_x([1,2,3],2) == [[1,2],[3]]"]),
        "frequency_map": ("solve_x", ["assert solve_x(['a','a','b']) == {'a':2,'b':1}"]),
    }
    for family, (name, tests) in fixtures.items():
        run_python_test_contract(reference_coding_source(family, name), tests)
