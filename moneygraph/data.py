"""Strict input contract. Never silently drop accounts or transactions."""
from pathlib import Path

import numpy as np
import pandas as pd


class DataError(ValueError):
    """An actionable problem with the input dataset."""


def require(condition, message):
    if not condition:
        raise DataError(message)


def validate(nodes, edges, tx, cfg):
    required = {
        "nodes": (nodes, ["gid", "depth", "is_seed"]),
        "edges": (edges, ["src", "dst", "sum_kzt", "n_tx", "depth"]),
        "transactions": (tx, ["src", "dst", "date", "sum_kzt"]),
    }
    for name, (frame, cols) in required.items():
        require(set(cols).issubset(frame.columns), f"{name}: обязательные поля {cols}")
        require(not frame[cols].isna().any().any(), f"{name}: найдены пропуски")
    require(len(nodes) > 0, "nodes: пустой список клиентов")
    require(not nodes.gid.duplicated().any(), "nodes: повторяющиеся gid")
    require(not edges.duplicated(["src", "dst"]).any(), "edges: повторяющиеся пары src/dst")
    for frame, cols in [(nodes, ["gid", "depth"]), (edges, ["src", "dst", "depth", "n_tx"]), (tx, ["src", "dst"])]:
        for col in cols:
            require(pd.api.types.is_integer_dtype(frame[col]), f"{col}: нужен целочисленный тип, не float/string")
            require(frame[col].between(np.iinfo(np.int64).min, np.iinfo(np.int64).max).all(), f"{col}: вне int64")
    require(pd.api.types.is_bool_dtype(nodes.is_seed), "is_seed: требуется bool")
    require(nodes.depth.between(0, cfg["max_depth"]).all(), "nodes.depth: неверное колено")
    require(edges.depth.between(1, cfg["max_depth"]).all(), "edges.depth: неверное колено")
    require((nodes.is_seed == (nodes.depth == 0)).all(), "is_seed и depth=0 не согласованы")
    require((edges.n_tx > 0).all(), "n_tx: требуется положительное число")
    known = set(nodes.gid)
    for name, frame in [("edges", edges), ("transactions", tx)]:
        require((set(frame.src) | set(frame.dst)).issubset(known), f"{name}: gid отсутствует в nodes")
        require(pd.api.types.is_numeric_dtype(frame.sum_kzt), f"{name}.sum_kzt: нужен числовой тип")
        require(np.isfinite(frame.sum_kzt).all() and (frame.sum_kzt > 0).all(), f"{name}: неверные суммы")
    require((tx.sum_kzt >= cfg["min_transaction_kzt"]).all(), "transactions: сумма ниже порога сбора")
    try:
        tx = tx.copy()
        tx["date"] = pd.to_datetime(tx.date, errors="raise")
    except (ValueError, TypeError) as exc:
        raise DataError(f"transactions.date: неверная дата: {exc}") from exc
    require((tx.date == tx.date.dt.normalize()).all(), "Ожидались даты без времени; проверьте временную модель")
    require(tx.date.between(pd.Timestamp(cfg["period_start"]), pd.Timestamp(cfg["period_end"])).all(), "Транзакции вне периода config.json")
    agg = tx.groupby(["src", "dst"], as_index=False).agg(tx_sum=("sum_kzt", "sum"), tx_count=("sum_kzt", "size"))
    merged = edges.merge(agg, on=["src", "dst"], how="outer", indicator=True)
    require((merged._merge == "both").all(), "edges и transactions: не совпадают пары")
    require(np.allclose(merged.sum_kzt, merged.tx_sum, rtol=0, atol=0.01), "edges и transactions: не совпадают суммы (допуск 0.01 KZT)")
    require((merged.n_tx == merged.tx_count).all(), "edges и transactions: не совпадает n_tx")
    # Same-day same-amount payments can be legitimate. No transaction ID => no deduplication.
    nodes = nodes.sort_values("gid", kind="stable").reset_index(drop=True)
    edges = edges.sort_values(["src", "dst"], kind="stable").reset_index(drop=True)
    tx = tx.sort_values(["date", "src", "dst", "sum_kzt"], kind="stable").reset_index(drop=True)
    return nodes, edges, tx


def load(data_dir: Path, cfg):
    frames = []
    for name in ["nodes", "edges", "transactions"]:
        path = data_dir / f"{name}.parquet"
        require(path.is_file(), f"Не найден {path}. Распакуйте data (1).zip в папку проекта.")
        frames.append(pd.read_parquet(path))
    return validate(*frames, cfg)
