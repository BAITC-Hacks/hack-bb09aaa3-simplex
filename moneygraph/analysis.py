"""Deterministic structural features, transparent roles and triage scores."""
import math

import networkx as nx
import numpy as np
import pandas as pd

from .temporal import temporal_features

ROLES = ("consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral")


def build_graph(nodes, edges):
    graph = nx.DiGraph()
    graph.add_nodes_from(int(gid) for gid in nodes.gid)  # Include all 19 isolated seeds.
    for r in edges.itertuples(index=False):
        graph.add_edge(int(r.src), int(r.dst), sum_kzt=float(r.sum_kzt), n_tx=int(r.n_tx))
    return graph


def partitions(graph, cfg):
    components = sorted(nx.weakly_connected_components(graph), key=lambda c: (-len(c), min(c)))
    projection = nx.Graph()
    projection.add_nodes_from(sorted(graph))
    for u, v, d in graph.edges(data=True):
        if u == v:
            continue
        old = projection.get_edge_data(u, v, {}).get("amount", 0)
        projection.add_edge(u, v, amount=old + d["sum_kzt"])
    for _, _, d in projection.edges(data=True):
        d["weight"] = math.log1p(d["amount"])
    groups = []
    for component in components:
        sub = projection.subgraph(sorted(component)).copy()
        if sub.number_of_edges() == 0:
            groups.extend({gid} for gid in sorted(component))
        else:
            groups.extend(nx.community.louvain_communities(
                sub, weight="weight", resolution=cfg["louvain_resolution"], seed=cfg["random_seed"]))
    groups.sort(key=lambda c: (-len(c), min(c)))
    clusters = {gid: i for i, c in enumerate(groups) for gid in c}
    component_ids = {gid: i for i, c in enumerate(components) for gid in c}
    return clusters, component_ids


def positive_scale(series):
    """Zero remains zero; log scale capped at the 95th percentile of positive values."""
    positive = series[series > 0]
    if len(positive) == 0:
        return series * 0.0
    return (np.log1p(series.clip(lower=0)) / np.log1p(positive.quantile(0.95))).clip(0, 1)


def role_for(r, cfg):
    """Eligibility gates precede confidence. Returned scores are NOT probabilities."""
    scores = {}
    boundary = bool(r["truncated_by_depth"])
    ratio_ok = not r["is_seed"] and r["in_kzt"] > 0 and not boundary
    ratio = r["pass_through"]
    if r["in_deg"] >= cfg["consolidator_min_in"]:
        scores["consolidator"] = 0.50 + 0.35 * min(r["in_deg"] / 12, 1) + 0.15 * min(r["seed_reach"] / 4, 1)
    if r["out_deg"] >= cfg["distributor_min_out"]:
        scores["distributor"] = 0.55 + 0.45 * min(r["out_deg"] / 60, 1)
    if ratio_ok and r["out_deg"] > 0 and cfg["transit_ratio_min"] <= ratio <= cfg["transit_ratio_max"]:
        scores["transit"] = 0.50 + 0.25 * (1 - abs(1 - ratio)) + 0.25 * r["fast_1_2d_share"]
    if (ratio_ok and ratio <= cfg["terminal_ratio_max"] and
            r["temporal_eligible_kzt"] > 0 and not r["late_inflow"]):
        scores["terminal"] = 0.50 + 0.20 * (1 - ratio / max(cfg["terminal_ratio_max"], 1e-9))
    if (not boundary and r["in_deg"] >= 2 and r["out_deg"] >= 2 and
            r["seed_reach"] >= cfg["coordinator_min_seeds"] and
            r["external_clusters"] >= cfg["coordinator_min_clusters"] and
            r["betweenness"] > 0 and r["betweenness_pct"] >= cfg["coordinator_betweenness_percentile"]):
        scores["coordinator"] = 0.55 + 0.20 * min(r["seed_reach"] / 5, 1) + 0.25 * r["betweenness_pct"]
    if boundary:
        scores = {role: min(score, 0.60) for role, score in scores.items()}
    if not scores:
        return "peripheral", 0.15 if boundary or r["is_isolate"] else 0.35, {}
    # A coordinating structural pattern is more specific than its degree patterns.
    # Otherwise choose highest score; tuple order makes ties deterministic.
    role = "coordinator" if "coordinator" in scores else max(scores, key=scores.get)
    return role, round(scores[role], 6), {key: round(value, 6) for key, value in scores.items()}


def evidence_for(r):
    base = {
        "consolidator": f"Признаки сбора: {r['in_deg']} плательщиков, вход {r['in_kzt']:,.0f} KZT; пути от {r['seed_reach']} seed.",
        "distributor": f"Веер: {r['out_deg']} получателей, {r['out_tx']} переводов; исходящие {r['out_kzt']:,.0f} KZT.",
        "transit": f"Транзитная гипотеза: выход/вход {r['pass_through']:.2f}; FIFO 1–2 дня {r['fast_1_2d_share']:.0%} наблюдаемого входа.",
        "terminal": f"Гипотеза стока: вход {r['in_kzt']:,.0f} KZT, выход/вход {r['pass_through']:.2f}; колено {r['depth']}.",
        "coordinator": f"Кандидат-мост: пути от {r['seed_reach']} seed; {r['external_clusters']} внешних кластеров; посредничество P{r['betweenness_pct'] * 100:.0f}.",
        "peripheral": f"Недостаточно признаков: входящих связей {r['in_deg']}, исходящих {r['out_deg']}.",
    }[r["role"]]
    if r["truncated_by_depth"]:
        base += " Обрыв на 4-м колене: исходящие неизвестны."
    elif r["is_isolate"]:
        base += " Изолированный seed; движение денег не наблюдается."
    elif r["is_seed"]:
        base += " Seed: вход неполон, отношение сумм не используется."
    elif r["role"] == "terminal":
        base += " Остаток и межбанк неизвестны."
    elif r["late_inflow"]:
        base += " Конец периода ограничивает наблюдение."
    else:
        base += " Только наблюдаемые переводы."
    return base[:200]


def analyze(nodes, edges, tx, cfg):
    graph = build_graph(nodes, edges)
    cluster_map, component_map = partitions(graph, cfg)
    features = nodes.copy()
    for name, degree in [
        ("in_deg", graph.in_degree()), ("out_deg", graph.out_degree()),
        ("in_kzt", graph.in_degree(weight="sum_kzt")), ("out_kzt", graph.out_degree(weight="sum_kzt")),
        ("in_tx", graph.in_degree(weight="n_tx")), ("out_tx", graph.out_degree(weight="n_tx")),
    ]:
        features[name] = features.gid.map(dict(degree)).fillna(0)
    features["cluster_id"] = features.gid.map(cluster_map)
    features["component_id"] = features.gid.map(component_map)
    features["truncated_by_depth"] = (features.depth == cfg["max_depth"]) & (features.out_deg == 0)
    features["is_isolate"] = (features.in_deg + features.out_deg) == 0
    features["pass_through"] = features.out_kzt / features.in_kzt.replace(0, np.nan)
    features["balance_incomplete"] = features.is_seed | (features.out_kzt > features.in_kzt)
    # Distances are hops. Monetary amounts are NOT shortest-path distances.
    between = nx.betweenness_centrality(graph, normalized=True, weight=None)
    features["betweenness"] = features.gid.map(between)
    features["betweenness_pct"] = features.betweenness.where(features.betweenness > 0).rank(pct=True).fillna(0)
    ancestors = {gid: set() for gid in graph}
    seeds = sorted(int(gid) for gid in nodes.loc[nodes.is_seed, "gid"])
    for seed in seeds:
        for gid, distance in nx.single_source_shortest_path_length(graph, seed, cutoff=cfg["max_depth"]).items():
            if distance > 0:
                ancestors[gid].add(seed)
    features["seed_reach"] = features.gid.map({gid: len(s) for gid, s in ancestors.items()})
    features["direct_seed_payers"] = features.gid.map({gid: sum(u in seeds for u in graph.predecessors(gid)) for gid in graph})
    features["external_clusters"] = features.gid.map({
        gid: len({cluster_map[n] for n in set(graph.predecessors(gid)) | set(graph.successors(gid))} - {cluster_map[gid]}) for gid in graph})
    features["reciprocal_neighbors"] = features.gid.map({gid: len((set(graph.predecessors(gid)) & set(graph.successors(gid))) - {gid}) for gid in graph})
    features = features.merge(temporal_features(nodes, tx, cfg["period_end"], cfg["temporal_window_days"]), on="gid", validate="one_to_one")
    for col in ["in_deg", "out_deg", "in_tx", "out_tx"]:
        features[col] = features[col].astype("int64")
    assignments = [role_for(row, cfg) for row in features.to_dict("records")]
    features["role"] = [a[0] for a in assignments]
    features["role_score"] = [a[1] for a in assignments]
    role_candidates = {str(gid): a[2] for gid, a in zip(features.gid, assignments)}
    features["priority_seed"] = 0.30 * positive_scale(features.seed_reach)
    features["priority_collection"] = 0.25 * positive_scale(features.in_deg)
    features["priority_volume"] = 0.15 * positive_scale(features[["in_kzt", "out_kzt"]].max(axis=1))
    features["priority_bridge"] = 0.15 * positive_scale(features.betweenness)
    features["priority_distribution"] = 0.10 * positive_scale(features.out_deg)
    features["priority_temporal"] = 0.05 * features.fast_1_2d_share.where(~features.is_seed & ~features.truncated_by_depth, 0)
    contribution_cols = [c for c in features if c.startswith("priority_")]
    features["priority_score"] = features[contribution_cols].sum(axis=1).clip(0, 1).round(6)
    features["evidence"] = [evidence_for(r) for r in features.to_dict("records")]
    required = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
    features = features[required + [c for c in features if c not in required]]
    ranked = features.sort_values(["priority_score", "gid"], ascending=[False, True], kind="stable")
    top = ranked.head(max(20, cfg["top_n"]))[["gid", "role", "priority_score", "evidence"]].rename(columns={"evidence": "why"}).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    by_gid = features.set_index("gid")
    top["why"] = [
        f"{why} Приоритет: seed {by_gid.loc[gid, 'priority_seed']:.3f}; сбор {by_gid.loc[gid, 'priority_collection']:.3f}; "
        f"объём {by_gid.loc[gid, 'priority_volume']:.3f}; мост {by_gid.loc[gid, 'priority_bridge']:.3f}; "
        f"веер {by_gid.loc[gid, 'priority_distribution']:.3f}; время {by_gid.loc[gid, 'priority_temporal']:.3f}."
        for gid, why in zip(top.gid, top.why)]
    internal = {int(cid): 0.0 for cid in features.cluster_id.unique()}
    for r in edges.itertuples(index=False):
        if cluster_map[r.src] == cluster_map[r.dst]:
            internal[cluster_map[r.src]] += r.sum_kzt
    cluster_rows = []
    for cid, group in features.groupby("cluster_id", sort=True):
        leaders = group.sort_values(["priority_score", "gid"], ascending=[False, True]).head(5)
        counts = group.role.value_counts()
        hypothesis = (f"Структурное сообщество: сборщиков {counts.get('consolidator', 0)}, "
                      f"распределителей {counts.get('distributor', 0)}, мостов {counts.get('coordinator', 0)}; "
                      f"seed {int(group.is_seed.sum())}. Гипотеза связности, не доказанная группа.")
        if group.is_isolate.all():
            hypothesis = "Изолированный seed: 0 наблюдаемых связей; нужны дополнительные переводы."
        cluster_rows.append({"cluster_id": int(cid), "n_nodes": len(group), "n_seed": int(group.is_seed.sum()),
                             "sum_kzt_internal": round(internal[cid], 2),
                             "top_gids": ";".join(map(str, leaders.gid)), "hypothesis": hypothesis})
    clusters = pd.DataFrame(cluster_rows)
    return graph, features, clusters, top, role_candidates


def removal_scenarios(graph, features, top, cutoff=4):
    """Reachability of surviving endpoints only; a structural scenario, not a forecast."""
    seeds = set(features.loc[features.is_seed, "gid"])
    baseline = {s: set(nx.single_source_shortest_path_length(graph, s, cutoff=cutoff)) - {s} for s in seeds}
    rows = []
    for count in (1, 3, 5):
        removed = set(top.head(count).gid)
        remaining = set(graph) - removed
        sub = graph.subgraph(remaining)
        before, after = 0, 0
        for seed in seeds - removed:
            before += len(baseline[seed] & remaining)
            after += len(set(nx.single_source_shortest_path_length(sub, seed, cutoff=cutoff)) - {seed})
        components = list(nx.weakly_connected_components(sub))
        rows.append({"n_removed": len(removed), "removed_gids": list(map(str, sorted(removed))),
                     "removed_seeds": len(seeds & removed), "surviving_pairs_before": before,
                     "surviving_pairs_after": after, "lost_reachability_share": (before - after) / before if before else 0,
                     "components_after": len(components), "largest_component_after": max(map(len, components), default=0)})
    return rows
