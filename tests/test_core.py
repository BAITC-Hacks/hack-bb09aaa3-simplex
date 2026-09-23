import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from moneygraph.analysis import analyze, role_for
from moneygraph.data import DataError, load, validate
from moneygraph.pipeline import read_config, run
from moneygraph.temporal import temporal_features
from scripts.make_demo import make_demo
from scripts.sensitivity import compare_top, report


class RoleSafetyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = read_config()
        self.base = dict(is_seed=False, in_kzt=100000, out_kzt=0, in_deg=1, out_deg=0,
                         truncated_by_depth=False, pass_through=0, is_isolate=False,
                         seed_reach=1, external_clusters=0, betweenness=0, betweenness_pct=0,
                         fast_1_2d_share=0, temporal_eligible_kzt=100000, late_inflow=False)

    def test_censored_node_never_terminal(self):
        role, score, candidates = role_for({**self.base, "truncated_by_depth": True}, self.cfg)
        self.assertEqual(role, "peripheral")
        self.assertNotIn("terminal", candidates)

    def test_boundary_can_have_positive_collection_evidence(self):
        role, score, _ = role_for({**self.base, "truncated_by_depth": True, "in_deg": 12}, self.cfg)
        self.assertEqual(role, "consolidator")
        self.assertLessEqual(score, .6)

    def test_seed_not_assigned_balance_based_roles(self):
        for ratio in (0, 1, 20):
            with self.subTest(ratio=ratio):
                _, _, candidates = role_for({**self.base, "is_seed": True, "pass_through": ratio, "out_deg": 1}, self.cfg)
                self.assertNotIn("terminal", candidates)
                self.assertNotIn("transit", candidates)

    def test_late_incoming_blocks_terminal(self):
        self.assertEqual(role_for(self.base, self.cfg)[0], "terminal")
        self.assertEqual(role_for({**self.base, "late_inflow": True}, self.cfg)[0], "peripheral")

    def test_coordinator_requires_multiple_independent_structural_signals(self):
        row = {**self.base, "in_deg": 3, "out_deg": 10, "seed_reach": 3, "external_clusters": 3,
               "betweenness": .05, "betweenness_pct": .99}
        self.assertEqual(role_for(row, self.cfg)[0], "coordinator")
        self.assertNotIn("coordinator", role_for({**row, "seed_reach": 1}, self.cfg)[2])


class TemporalTests(unittest.TestCase):
    def features(self, rows):
        tx = pd.DataFrame(rows, columns=["src", "dst", "date", "sum_kzt"])
        tx["date"] = pd.to_datetime(tx.date)
        return temporal_features(pd.DataFrame({"gid": [2]}), tx, "2026-07-31").iloc[0]

    def test_same_day_is_not_ordered(self):
        r = self.features([(1, 2, "2026-07-01", 10000), (2, 3, "2026-07-01", 10000)])
        self.assertEqual(r.fast_1_2d_share, 0)

    def test_one_incoming_amount_cannot_be_spent_twice(self):
        r = self.features([(1, 2, "2026-07-01", 10000), (2, 3, "2026-07-02", 10000), (2, 4, "2026-07-03", 10000)])
        self.assertEqual(r.temporal_matched_kzt, 10000)
        self.assertEqual(r.fast_1_2d_share, 1)

    def test_future_incoming_cannot_fund_past_outgoing(self):
        r = self.features([(2, 3, "2026-07-01", 10000), (1, 2, "2026-07-02", 10000)])
        self.assertEqual(r.temporal_matched_kzt, 0)

    def test_window_expiry_and_period_censoring(self):
        r = self.features([(1, 2, "2026-07-01", 10000), (2, 3, "2026-07-04", 10000), (1, 2, "2026-07-31", 20000)])
        self.assertEqual(r.temporal_eligible_kzt, 10000)
        self.assertEqual(r.temporal_matched_kzt, 0)
        self.assertTrue(r.late_inflow)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.nodes, self.edges, self.tx = make_demo(self.data)
        self.cfg = read_config()

    def tearDown(self):
        self.temp.cleanup()

    def test_reject_mismatched_edge_sum_and_count(self):
        for col in ["sum_kzt", "n_tx"]:
            edges = self.edges.copy()
            edges.loc[0, col] += 100
            with self.subTest(col=col), self.assertRaises(DataError):
                validate(self.nodes, edges, self.tx, self.cfg)

    def test_reject_unknown_id_and_float_id(self):
        edges = self.edges.copy()
        edges.loc[0, "src"] = 123
        with self.assertRaises(DataError):
            validate(self.nodes, edges, self.tx, self.cfg)
        nodes = self.nodes.copy()
        nodes["gid"] = nodes.gid.astype(float)
        with self.assertRaises(DataError):
            validate(nodes, self.edges, self.tx, self.cfg)

    def test_identical_transactions_are_preserved(self):
        tx = pd.concat([self.tx, self.tx.iloc[[0]]], ignore_index=True)
        edges = self.edges.copy()
        mask = (edges.src == tx.iloc[0].src) & (edges.dst == tx.iloc[0].dst)
        edges.loc[mask, "sum_kzt"] += tx.iloc[0].sum_kzt
        edges.loc[mask, "n_tx"] += 1
        _, _, checked = validate(self.nodes, edges, tx, self.cfg)
        self.assertEqual(len(checked), len(self.tx) + 1)

    def test_end_to_end_exact_ids_isolates_exports_and_determinism(self):
        first, second = self.root / "first", self.root / "second"
        audit = run(self.data, first)
        run(self.data, second)
        for filename in ["nodes_roles.csv", "clusters.csv", "top_nodes.csv"]:
            self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())
        roles = pd.read_csv(first / "nodes_roles.csv", dtype={"gid": "int64"})
        self.assertEqual(set(roles.gid), set(self.nodes.gid))
        self.assertEqual(audit["isolates"], 1)
        self.assertEqual(roles.loc[roles.is_isolate, "role"].tolist(), ["peripheral"])
        self.assertTrue(roles.evidence.str.len().between(1, 200).all())
        self.assertFalse(((roles.role == "terminal") & roles.truncated_by_depth).any())
        self.assertGreaterEqual(len(pd.read_csv(first / "top_nodes.csv")), 20)
        payload = json.loads((first / "graph.json").read_text(encoding="utf-8"))
        self.assertTrue(all(type(n["gid"]) is str for n in payload["nodes"]))
        self.assertEqual({n["gid"] for n in payload["nodes"]}, set(map(str, self.nodes.gid)))
        self.assertTrue(all(type(e["src"]) is str and type(e["dst"]) is str for e in payload["edges"]))
        self.assertNotIn("__MONEYGRAPH_DATA__", (first / "dashboard.html").read_text(encoding="utf-8"))
        for scenario in payload["removal"]:
            self.assertLessEqual(scenario["surviving_pairs_after"], scenario["surviving_pairs_before"])

    def test_row_order_does_not_change_roles_or_clusters(self):
        inputs = load(self.data, self.cfg)
        _, features, clusters, top, _ = analyze(*inputs, self.cfg)
        shuffled = validate(*(df.sample(frac=1, random_state=9) for df in inputs), self.cfg)
        _, actual, actual_clusters, actual_top, _ = analyze(*shuffled, self.cfg)
        pd.testing.assert_frame_equal(features, actual)
        pd.testing.assert_frame_equal(clusters, actual_clusters)
        pd.testing.assert_frame_equal(top, actual_top)

    def test_bad_config_fails(self):
        config = self.root / "bad.json"
        config.write_text('{"top_n": 3}', encoding="utf-8")
        with self.assertRaises(DataError):
            read_config(config)

    def test_sensitivity_report_preserves_ids_and_input_config(self):
        before = self.cfg.copy()
        result = report(*load(self.data, self.cfg), self.cfg)
        self.assertEqual(self.cfg, before)
        self.assertEqual(len(result["role_thresholds"]), 4)
        self.assertEqual(len(result["priority_weights"]), 12)
        self.assertTrue(set(result["baseline_top_gids"]).issubset(set(map(str, self.nodes.gid))))
        self.assertTrue(all(0 <= item["overlap"] <= result["top_n"] for item in result["priority_weights"]))
        self.assertTrue(all(set(item["changed_gids"]).issubset(set(map(str, self.nodes.gid)))
                            for item in result["role_thresholds"]))

    def test_top_comparison_reports_changes(self):
        self.assertEqual(compare_top(["10", "20"], ["20", "30"]),
                         {"overlap": 1, "entered": ["30"], "left": ["10"], "same_order": False})


if __name__ == "__main__":
    unittest.main()
