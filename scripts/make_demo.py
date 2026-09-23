"""Generate a small, entirely fictional public dataset. No hackathon data is copied."""
import argparse
from pathlib import Path

import networkx as nx
import pandas as pd


def make_demo(out):
    base = 900000000000000000
    rows = []

    def pay(src, dst, day, amount):
        rows.append((base + src, base + dst, f"2026-07-{day:02d}", float(amount)))

    # Independent seeds converge; the next two hops have observable temporal patterns.
    for seed in (1, 2, 3):
        pay(seed, 10, 3, 50000)
    pay(10, 11, 4, 140000)
    pay(11, 12, 5, 135000)
    pay(12, 13, 6, 5000)  # depth=4 is censored, not a proven terminal.
    pay(4, 20, 8, 100000)
    for receiver in range(21, 36):
        pay(20, receiver, 9, 5000)
    pay(5, 40, 28, 30000)
    pay(40, 41, 31, 25000)  # late incoming prevents a terminal claim.
    pay(2, 50, 10, 10000)
    pay(50, 51, 10, 10000)  # same-day order is unknown.
    pay(51, 50, 12, 5000)
    tx = pd.DataFrame(rows, columns=["src", "dst", "date", "sum_kzt"])
    tx["date"] = pd.to_datetime(tx.date).dt.date
    graph = nx.DiGraph()
    graph.add_edges_from(zip(tx.src, tx.dst))
    seeds = {base + i for i in range(1, 7)}  # seed 6 has no edges.
    graph.add_nodes_from(seeds)
    depths = {gid: 4 for gid in graph}
    for seed in sorted(seeds):
        for gid, depth in nx.single_source_shortest_path_length(graph, seed, cutoff=4).items():
            depths[gid] = min(depths[gid], depth)
    nodes = pd.DataFrame([{"gid": gid, "depth": depths[gid], "is_seed": gid in seeds} for gid in sorted(graph)])
    edges = tx.groupby(["src", "dst"], as_index=False).agg(sum_kzt=("sum_kzt", "sum"), n_tx=("sum_kzt", "size"))
    edges["depth"] = edges.src.map(lambda gid: min(4, depths[gid] + 1)).astype("int8")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in [("nodes", nodes), ("edges", edges), ("transactions", tx)]:
        frame.to_parquet(out / f"{name}.parquet", index=False)
    return nodes, edges, tx


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data-demo"))
    args = ap.parse_args()
    nodes, edges, tx = make_demo(args.out)
    print(f"Fictional demo: {len(nodes)} nodes, {len(edges)} edges, {len(tx)} transactions -> {args.out}")
