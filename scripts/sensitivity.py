#!/usr/bin/env python3
"""Local stress test of role thresholds and triage weights; no ground truth needed."""
import argparse
import json
from pathlib import Path

from moneygraph.analysis import analyze
from moneygraph.data import load
from moneygraph.pipeline import read_config


ROLE_THRESHOLDS = ("consolidator_min_in", "distributor_min_out")
WEIGHTS = {
    "priority_seed": 0.30,
    "priority_collection": 0.25,
    "priority_volume": 0.15,
    "priority_bridge": 0.15,
    "priority_distribution": 0.10,
    "priority_temporal": 0.05,
}


def ranked_gids(features, scores, count):
    ranked = features[["gid"]].copy()
    ranked["score"] = scores.round(6)
    return [str(gid) for gid in ranked.sort_values(
        ["score", "gid"], ascending=[False, True], kind="stable"
    ).head(count).gid]


def compare_top(baseline, candidate):
    before, after = set(baseline), set(candidate)
    return {
        "overlap": len(before & after),
        "entered": sorted(after - before, key=int),
        "left": sorted(before - after, key=int),
        "same_order": baseline == candidate,
    }


def report(nodes, edges, tx, cfg, top_n=20):
    _, base, _, _, _ = analyze(nodes, edges, tx, cfg)
    count = min(top_n, len(base))
    base_top = ranked_gids(base, base.priority_score, count)
    base_roles = base.set_index("gid").role
    thresholds = []
    for key in ROLE_THRESHOLDS:
        for shift in (-1, 1):
            value = cfg[key] + shift
            if value < 1:
                continue
            changed_cfg = {**cfg, key: value}
            _, variant, _, _, _ = analyze(nodes, edges, tx, changed_cfg)
            current = variant.set_index("gid").role
            differences = [str(gid) for gid in base_roles.index if base_roles[gid] != current[gid]]
            thresholds.append({
                "parameter": key, "baseline": cfg[key], "value": value,
                "changed_roles": len(differences), "changed_gids": differences,
            })

    weights = []
    for key, weight in WEIGHTS.items():
        for factor in (0.8, 1.2):
            # Only one factor changes; rescale to retain a total weight of one.
            scores = (base.priority_score + (factor - 1) * base[key]) / (1 + (factor - 1) * weight)
            candidate = ranked_gids(base, scores, count)
            weights.append({
                "factor": key, "multiplier": factor,
                **compare_top(base_top, candidate),
            })
    return {
        "note": "Сценарии чувствительности, не оценка accuracy или вероятности виновности.",
        "top_n": count, "baseline_top_gids": base_top,
        "role_thresholds": thresholds, "priority_weights": weights,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/sensitivity.json"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top должен быть положительным")
    cfg = read_config(args.config)
    result = report(*load(args.data, cfg), cfg, args.top)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Отчёт: {args.out}; изменения ролей: "
          f"{[x['changed_roles'] for x in result['role_thresholds']]}; "
          f"минимальное пересечение топ-{result['top_n']}: "
          f"{min(x['overlap'] for x in result['priority_weights'])}")


if __name__ == "__main__":
    main()
