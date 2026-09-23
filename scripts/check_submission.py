"""Validate the mandatory exports against the source dataset (no fixed GIDs)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moneygraph.analysis import ROLES


def check(data, out, expected_nodes=2248):
    source = pd.read_parquet(data / "nodes.parquet")
    edges = pd.read_parquet(data / "edges.parquet")
    nodes = pd.read_csv(out / "nodes_roles.csv", dtype={"gid": "int64"})
    clusters = pd.read_csv(out / "clusters.csv", dtype={"top_gids": str})
    top = pd.read_csv(out / "top_nodes.csv", dtype={"gid": "int64"})

    def require(ok, message):
        if not ok:
            raise ValueError(message)

    for df, columns in [(nodes, ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]),
                        (clusters, ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"]),
                        (top, ["rank", "gid", "role", "priority_score", "why"])]:
        require(set(columns).issubset(df.columns), f"Missing columns: {columns}")
        require(not df[columns].isna().any().any(), "Empty mandatory cells")
    require(expected_nodes == 0 or len(nodes) == expected_nodes, "Unexpected node count")
    require(len(nodes) == len(source) and set(nodes.gid) == set(source.gid), "Nodes missing or changed")
    require(not nodes.gid.duplicated().any(), "Duplicate node IDs")
    require(nodes.role.isin(ROLES).all(), "Unknown roles")
    require(nodes.role_score.between(0, 1).all() and nodes.priority_score.between(0, 1).all(), "Invalid score")
    require(nodes.evidence.str.len().between(1, 200).all() and nodes.evidence.str.contains(r"\d").all(), "Invalid evidence")
    require(pd.api.types.is_integer_dtype(nodes.cluster_id), "cluster_id must be integer")
    require(set(nodes.cluster_id) == set(clusters.cluster_id), "Unknown cluster IDs")
    require(not clusters.cluster_id.duplicated().any(), "Duplicate cluster IDs")
    require(clusters.hypothesis.str.strip().str.len().gt(0).all(), "Empty cluster hypothesis")
    sizes = nodes.groupby("cluster_id").size()
    seed_ids = set(source.loc[source.is_seed, "gid"])
    mapping = nodes.set_index("gid").cluster_id.to_dict()
    for c in clusters.itertuples(index=False):
        members = nodes[nodes.cluster_id == c.cluster_id]
        require(c.n_nodes == sizes[c.cluster_id], "Wrong cluster size")
        require(c.n_seed == sum(members.gid.isin(seed_ids)), "Wrong seed count")
        require(set(map(int, c.top_gids.split(";"))).issubset(set(members.gid)), "Cluster leaders outside cluster")
        internal = edges[edges.src.map(mapping).eq(c.cluster_id) & edges.dst.map(mapping).eq(c.cluster_id)].sum_kzt.sum()
        require(abs(internal - c.sum_kzt_internal) <= .01, "Wrong internal turnover")
    require(len(top) >= min(20, len(nodes)), "Top has fewer than 20 nodes")
    expected = nodes.sort_values(["priority_score", "gid"], ascending=[False, True]).head(len(top))
    require(list(top.gid) == list(expected.gid), "Top is not the ranking of node scores")
    require(list(top.role) == list(expected.role), "Top roles disagree")
    require(np.allclose(top.priority_score, expected.priority_score), "Top scores disagree")
    require(list(top['rank']) == list(range(1, len(top) + 1)), "Rank sequence invalid")
    require(top.why.str.strip().str.len().gt(0).all(), "Empty why")
    require(not ((nodes.role == "terminal") & nodes.truncated_by_depth).any(), "Censored terminal")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for path in data.glob("*.parquet"):
        require(manifest["inputs_sha256"][path.name] == hashlib.sha256(path.read_bytes()).hexdigest(), "Stale results: source hash changed")
    require(manifest["elapsed_seconds"] < 300, "Runtime >= 5 minutes")
    graph = json.loads((out / "graph.json").read_text(encoding="utf-8"))
    require(all(type(n["gid"]) is str for n in graph["nodes"]), "JSON GIDs are not strings")
    require({n["gid"] for n in graph["nodes"]} == set(map(str, source.gid)), "JSON GIDs lost precision")
    require((out / "dashboard.html").is_file(), "Missing dashboard")
    return {"nodes": len(nodes), "clusters": len(clusters), "top": len(top), "seconds": manifest["elapsed_seconds"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--expected-nodes", type=int, default=2248, help="0 = accept dataset size")
    args = ap.parse_args()
    try:
        print("PASS", check(args.data, args.out, args.expected_nodes))
    except (ValueError, OSError, KeyError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
