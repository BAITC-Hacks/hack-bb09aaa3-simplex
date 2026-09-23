import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time

import networkx as nx
import numpy as np
import pandas as pd

from .analysis import ROLES, analyze, removal_scenarios
from .data import DataError, load, require

ROOT = Path(__file__).resolve().parents[1]


def read_config(path=None):
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if path:
        overrides = json.loads(Path(path).read_text(encoding="utf-8"))
        require(not (set(overrides) - set(cfg)), "Неизвестные параметры config: " + str(set(overrides) - set(cfg)))
        cfg.update(overrides)
    integer_keys = ["max_depth", "temporal_window_days", "consolidator_min_in", "distributor_min_out",
                    "coordinator_min_seeds", "coordinator_min_clusters", "top_n", "random_seed"]
    for key in integer_keys:
        require(type(cfg[key]) is int and cfg[key] >= (0 if key == "random_seed" else 1), f"{key}: неверное целое значение")
    require(cfg["top_n"] >= 20, "top_n должен быть >= 20")
    require(cfg["temporal_window_days"] == 2, "MVP поддерживает temporal_window_days=2; для другого окна измените объяснения")
    require(cfg["max_depth"] == 4, "MVP настроен на 4 колена; для другого обхода измените модель цензурирования")
    for key in ["terminal_ratio_max", "coordinator_betweenness_percentile"]:
        require(isinstance(cfg[key], (float, int)) and 0 <= cfg[key] <= 1, f"{key}: требуется 0..1")
    require(0 < cfg["transit_ratio_min"] < 1 < cfg["transit_ratio_max"], "Неверные границы transit_ratio")
    require(cfg["terminal_ratio_max"] < cfg["transit_ratio_min"], "Пороги terminal/transit пересекаются")
    require(cfg["louvain_resolution"] > 0 and cfg["min_transaction_kzt"] > 0, "Порог суммы и resolution должны быть положительными")
    require(pd.Timestamp(cfg["period_start"]) <= pd.Timestamp(cfg["period_end"]), "Неверный период")
    return cfg


def json_clean(obj):
    if isinstance(obj, dict):
        return {str(k): json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    return obj


def records(frame, ids=()):
    result = frame.to_dict("records")
    for row in result:
        for col in ids:
            row[col] = str(row[col])  # GIDs exceed Number.MAX_SAFE_INTEGER in JS.
    return json_clean(result)


def validate_outputs(nodes, features, clusters, top):
    require(len(features) == len(nodes) and set(features.gid) == set(nodes.gid), "Потеря узлов на выходе")
    require(not features.gid.duplicated().any(), "Повторяющиеся узлы на выходе")
    require(features.role.isin(ROLES).all(), "Неизвестная роль")
    require(features.role_score.between(0, 1).all() and features.priority_score.between(0, 1).all(), "Скор вне 0..1")
    require(features.evidence.str.len().between(1, 200).all(), "evidence должен содержать 1..200 символов")
    require(set(features.cluster_id) == set(clusters.cluster_id), "Несогласованные cluster_id")
    require(int(clusters.n_nodes.sum()) == len(nodes), "Кластеры не покрывают все узлы")
    require(len(top) >= min(20, len(nodes)), "Недостаточно узлов в топ-листе")
    require(top.priority_score.is_monotonic_decreasing, "Топ не отсортирован")
    require(not ((features.role == "terminal") & features.truncated_by_depth).any(), "Граничный узел ошибочно назван terminal")


def run(data_dir, out_dir, config_path=None):
    started = time.perf_counter()
    cfg = read_config(config_path)
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    nodes, edges, tx = load(data_dir, cfg)
    graph, features, clusters, top, candidates = analyze(nodes, edges, tx, cfg)
    validate_outputs(nodes, features, clusters, top)
    audit = {
        "nodes": len(nodes), "edges": len(edges), "transactions": len(tx),
        "seeds": int(nodes.is_seed.sum()), "turnover_kzt": round(float(edges.sum_kzt.sum()), 2),
        "period_start": cfg["period_start"], "period_end": cfg["period_end"],
        "components_with_isolates": nx.number_weakly_connected_components(graph),
        "isolates": int(features.is_isolate.sum()),
        "components_with_edges": sum(graph.subgraph(c).number_of_edges() > 0 for c in nx.weakly_connected_components(graph)),
        "clusters": len(clusters), "truncated": int(features.truncated_by_depth.sum()),
        "seeds_without_outgoing": int((features.is_seed & (features.out_deg == 0)).sum()),
        "out_exceeds_in": int((features.out_kzt > features.in_kzt).sum()),
        "out_exceeds_positive_in": int(((features.out_kzt > features.in_kzt) & (features.in_kzt > 0)).sum()),
        "out_without_observed_in": int(((features.out_kzt > 0) & (features.in_kzt == 0)).sum()),
        "identical_transaction_rows_preserved": int(tx.duplicated().sum()),
        "roles": features.role.value_counts().to_dict(),
        "warnings": [
            "Роли — проверяемые гипотезы; role_score — сила правил, не вероятность виновности или точность модели.",
            "4-е колено обрезано. Отсутствие исходящих переводов не подтверждает удержание денег.",
            "Входящие извне, межбанк, платежи <5 000 KZT и остатки не видны. Полный баланс не вычисляется.",
            "FIFO за 1–2 дня — совместимость дат и сумм, не доказанная трассировка одних и тех же денег; порядок внутри дня неизвестен.",
            "365 890 012.01 KZT — оборот по рёбрам: одна сумма может учитываться на нескольких этапах цепочки.",
            "Кластеры показывают структурную связность, а не доказанное участие в одной группе.",
        ],
    }
    # Keep warning applicable to synthetic and future datasets as well.
    audit["warnings"][4] = f"{audit['turnover_kzt']:,.2f} KZT — оборот по рёбрам: деньги могут учитываться на нескольких этапах цепочки."
    payload = {
        "audit": audit, "config": cfg,
        "nodes": records(features, ("gid",)), "edges": records(edges, ("src", "dst")),
        "clusters": records(clusters), "top": records(top, ("gid",)), "role_candidates": candidates,
        "removal": removal_scenarios(graph, features, top, cfg["max_depth"]),
        "daily": {},
    }
    daily_in = tx.groupby(["dst", "date"]).sum_kzt.sum()
    daily_out = tx.groupby(["src", "date"]).sum_kzt.sum()
    for (gid, date), value in daily_in.items():
        payload["daily"].setdefault(str(gid), {}).setdefault(date.strftime("%Y-%m-%d"), {"in": 0, "out": 0})["in"] = float(value)
    for (gid, date), value in daily_out.items():
        payload["daily"].setdefault(str(gid), {}).setdefault(date.strftime("%Y-%m-%d"), {"in": 0, "out": 0})["out"] = float(value)
    template = (ROOT / "moneygraph" / "web" / "dashboard.html").read_text(encoding="utf-8")
    manifest = {
        "version": "1.0.0", "python": platform.python_version(), "config": cfg,
        "packages": {name: importlib.metadata.version(name) for name in ["pandas", "numpy", "pyarrow", "networkx"]},
        "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data_dir.glob("*.parquet"))},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".moneygraph-", dir=out_dir) as temp:
        temp = Path(temp)
        for name, frame in [("nodes_roles", features), ("clusters", clusters), ("top_nodes", top)]:
            frame.to_csv(temp / f"{name}.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
        # Write data once to measure the full work, then record elapsed time in the artifacts.
        encoded = json.dumps(json_clean(payload), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        (temp / "dashboard.html").write_text(template.replace("__MONEYGRAPH_DATA__", encoded.replace("<", "\\u003c")), encoding="utf-8")
        elapsed = time.perf_counter() - started
        audit["elapsed_seconds"] = round(elapsed, 3)
        manifest["elapsed_seconds"] = audit["elapsed_seconds"]
        encoded = json.dumps(json_clean(payload), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        (temp / "graph.json").write_text(encoded, encoding="utf-8")
        (temp / "dashboard.html").write_text(template.replace("__MONEYGRAPH_DATA__", encoded.replace("<", "\\u003c")), encoding="utf-8")
        for name, value in [("audit", audit), ("manifest", manifest)]:
            (temp / f"{name}.json").write_text(json.dumps(json_clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        for file in temp.iterdir():
            os.replace(file, out_dir / file.name)
    return audit
