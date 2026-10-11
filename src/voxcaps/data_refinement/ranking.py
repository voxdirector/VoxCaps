"""Optional ordering from existing scores or preferences; no annotation workflow."""

from __future__ import annotations

import math
from collections import defaultdict


def score_value(record: dict, field: str) -> float:
    value = record
    for key in field.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ValueError("ranking_score_field_missing")
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("ranking_score_must_be_finite_numeric")
    return float(value)


def rank_scores(records, *, field: str = "score", descending: bool = True) -> dict:
    items, seen = [], set()
    for record in records:
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("ranking_ids_must_be_unique_nonempty_strings")
        seen.add(identifier)
        items.append((identifier, score_value(record, field)))
    items.sort(key=lambda item: (-item[1] if descending else item[1], item[0]))
    ranked, previous, rank = [], None, 0
    for position, (identifier, score) in enumerate(items, 1):
        if score != previous:
            rank = position
        ranked.append({"id": identifier, "score": score, "rank": rank})
        previous = score
    return {
        "method": "sort_existing_scores",
        "score_field": field,
        "descending": descending,
        "ranking": ranked,
    }


def rank_preferences(pairs) -> dict:
    nodes, children, indegree = set(), defaultdict(set), {}
    for pair in pairs:
        preferred, dispreferred = pair.get("preferred"), pair.get("dispreferred")
        if (
            any(not isinstance(key, str) or not key for key in (preferred, dispreferred))
            or preferred == dispreferred
        ):
            raise ValueError("invalid_preference_pair")
        nodes.update((preferred, dispreferred))
        children[preferred].add(dispreferred)
    indegree = dict.fromkeys(nodes, 0)
    for parent in children:
        for child in children[parent]:
            indegree[child] += 1
    current = sorted(key for key in nodes if indegree[key] == 0)
    tiers, consumed = [], 0
    while current:
        tiers.append(current)
        consumed += len(current)
        following = []
        for parent in current:
            for child in children[parent]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    following.append(child)
        current = sorted(following)
    if consumed != len(nodes):
        raise ValueError("preference_graph_contains_cycle")
    return {"method": "topological_preference_tiers", "tiers": tiers}
