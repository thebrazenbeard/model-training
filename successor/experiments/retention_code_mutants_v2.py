from __future__ import annotations


def _fn(name: str, args: str, body: str) -> str:
    return f"def {name}({args}):\n    {body}\n"


def mutant_sources(family: str, function_name: str) -> dict[str, str]:
    if family == "rotate_left":
        return {
            "right_rotation": _fn(
                function_name,
                "values, k",
                "return [] if not values else list(values[-(k % len(values)):]) + list(values[:-(k % len(values))])",
            ),
            "no_modulo": _fn(
                function_name,
                "values, k",
                "return [] if not values else list(values[k:]) + list(values[:k])",
            ),
        }
    if family == "count_vowels":
        return {
            "includes_y": _fn(
                function_name,
                "text",
                "return sum(ch.lower() in 'aeiouy' for ch in text)",
            ),
            "case_sensitive": _fn(
                function_name,
                "text",
                "return sum(ch in 'aeiou' for ch in text)",
            ),
        }
    if family == "unique_preserve":
        return {
            "sorted_set": _fn(
                function_name,
                "values",
                "return sorted(set(values))",
            ),
            "last_occurrence_order": _fn(
                function_name,
                "values",
                "return list(dict.fromkeys(reversed(values)))[::-1]",
            ),
        }
    if family == "digit_sum":
        return {
            "off_by_one": _fn(
                function_name,
                "value",
                "return sum(int(ch) for ch in str(value)) + (1 if value else 0)",
            ),
            "digit_product": _fn(
                function_name,
                "value",
                "import math; return math.prod(int(ch) for ch in str(value))",
            ),
        }
    if family == "reverse_words":
        return {
            "reverse_each_word": _fn(
                function_name,
                "text",
                "return ' '.join(word[::-1] for word in text.split())",
            ),
            "normalize_only": _fn(
                function_name,
                "text",
                "return ' '.join(text.split())",
            ),
        }
    if family == "clamp_values":
        return {
            "lower_bound_only": _fn(
                function_name,
                "values, low, high",
                "return [max(low, item) for item in values]",
            ),
            "upper_bound_only": _fn(
                function_name,
                "values, low, high",
                "return [min(high, item) for item in values]",
            ),
        }
    if family == "pairwise_sums":
        return {
            "disjoint_pairs": _fn(
                function_name,
                "values",
                "return [values[i] + values[i + 1] for i in range(0, len(values) - 1, 2)]",
            ),
            "wraparound_extra_pair": _fn(
                function_name,
                "values",
                "return ([values[i] + values[i + 1] for i in range(len(values) - 1)] + ([values[-1] + values[0]] if len(values) > 1 else []))",
            ),
        }
    if family == "flatten_once":
        return {
            "first_group_only": _fn(
                function_name,
                "groups",
                "return list(groups[0]) if groups else []",
            ),
            "deduplicate_flattened": _fn(
                function_name,
                "groups",
                "return list(dict.fromkeys(item for group in groups for item in group))",
            ),
        }
    if family == "chunk_list":
        return {
            "drop_remainder": _fn(
                function_name,
                "values, n",
                "return [values[i:i+n] for i in range(0, len(values) - (len(values) % n), n)]",
            ),
            "overlapping_windows": _fn(
                function_name,
                "values, n",
                "return [values[i:i+n] for i in range(0, len(values)) if values[i:i+n]]",
            ),
        }
    if family == "frequency_map":
        return {
            "presence_only": _fn(
                function_name,
                "values",
                "return {item: 1 for item in values}",
            ),
            "off_by_one_counts": _fn(
                function_name,
                "values",
                "return {item: values.count(item) + 1 for item in values}",
            ),
        }
    raise ValueError(f"unknown coding family: {family}")
