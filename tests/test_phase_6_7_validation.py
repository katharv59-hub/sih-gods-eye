"""Phase 6.7 Final Integration, Held-Out Accuracy & Deterministic Validation Suite.

Covers comprehensive E2E validation for:
1. Tool 9 (get_trajectory_clusters)
2. Tool 10 (get_behavioral_prediction)
3. TrajectoryClusteringEngine route structure recovery
4. BehavioralAnomalyDetector scoring differential
5. MarkovBehavioralPredictor held-out temporal accuracy & coverage
6. Determinism & Read-Only Invariants
7. Privacy & Evidence Integrity
8. Query Semantics & Natural Language Routing
"""

from __future__ import annotations

import time
import pytest

from gods_eye.behavioral.anomaly import BehavioralAnomalyDetector
from gods_eye.behavioral.clustering import TrajectoryClusteringEngine
from gods_eye.behavioral.extractor import TrajectorySequence, TrajectorySequenceExtractor
from gods_eye.behavioral.predictor import MarkovBehavioralPredictor
from gods_eye.events.event_store import SQLiteEventStore
from gods_eye.memory.graph_store import SQLiteGraphStore
from gods_eye.reasoning.engine import ReasoningEngine
from gods_eye.reasoning.planner import NLQPlanner, RuleBasedPlanner
from gods_eye.reasoning.tools import ToolDispatcher
from gods_eye.schemas.behavioral import TrajectoryCluster
from gods_eye.schemas.reasoning import QueryStatus, ToolCall, UserQuery


@pytest.fixture
def behavioral_stack():
    es = SQLiteEventStore(":memory:")
    gs = SQLiteGraphStore(":memory:")
    dispatcher = ToolDispatcher(event_store=es, graph_store=gs)
    planner = NLQPlanner()
    engine = ReasoningEngine(planner=planner, dispatcher=dispatcher)
    yield es, gs, dispatcher, planner, engine
    es.close()
    gs.close()


# ── 1. TOOL 9 END-TO-END VALIDATION ─────────────────────────────────────────
class TestTool9EndToEndValidation:
    def test_tool9_complete_suite(self, behavioral_stack):
        es, gs, dispatcher, planner, engine = behavioral_stack

        # A. Empty GraphStore
        call_empty = ToolCall("c_t9_empty", "get_trajectory_clusters", {})
        res_empty = dispatcher.dispatch(call_empty)
        assert res_empty.success is True
        assert res_empty.data["status"] == "INSUFFICIENT_EVIDENCE"
        assert res_empty.data["clusters"] == []

        # B & C & D & E. Seed multiple trajectories with 2 distinct clusters
        for c in ["cam_01", "cam_02", "cam_03", "cam_04"]:
            gs.upsert_camera_node(c)

        # Cluster 1: cam_01 -> cam_02 -> cam_03 (3 identities)
        for i in range(1, 4):
            gid = f"gid_c1_{i}"
            gs.add_transition_edge(f"t_c1_{i}_1", gid, "cam_01", "cam_02", 1_000 + i, 1.0)
            gs.add_transition_edge(f"t_c1_{i}_2", gid, "cam_02", "cam_03", 2_000 + i, 1.0)

        # Cluster 2: cam_01 -> cam_04 (3 identities)
        for i in range(1, 4):
            gid = f"gid_c2_{i}"
            gs.add_transition_edge(f"t_c2_{i}_1", gid, "cam_01", "cam_04", 1_000 + i, 1.0)

        # Noise: cam_03 -> cam_04 (1 identity)
        gs.add_transition_edge("t_noise_1", "gid_noise", "cam_03", "cam_04", 1_500, 1.0)

        call = ToolCall("c_t9_full", "get_trajectory_clusters", {"min_samples": 2, "eps": 0.3})
        res = dispatcher.dispatch(call)

        assert res.success is True
        clusters = res.data["clusters"]
        assert len(clusters) >= 2

        # K. Privacy output validation: no global_id or global_ids in cluster payloads
        for cl in clusters:
            assert "global_id" not in cl
            assert "global_ids" not in cl
            assert "unique_identity_count" in cl

        # L. Read-only invariant check
        nodes_count = gs.count_nodes()
        edges_count = gs.count_edges()

        dispatcher.dispatch(call)
        assert gs.count_nodes() == nodes_count
        assert gs.count_edges() == edges_count

        # M. Evidence correctness
        assert len(res.evidence_list) == len(clusters)
        for ev in res.evidence_list:
            assert ev.source_store == "graph_store"
            assert ev.record_type == "trajectory_cluster"

        # N & O. Planner and Reasoning Engine E2E
        q = UserQuery("q_t9_e2e", "Show common trajectory clusters for cam_01", 1_700_000_000_000_000_000)
        r_result = engine.process_query(q)
        assert r_result.status == QueryStatus.SUCCESS
        assert len(r_result.evidence) >= 1


# ── 2. TOOL 10 END-TO-END VALIDATION ────────────────────────────────────────
class TestTool10EndToEndValidation:
    def test_tool10_complete_suite(self, behavioral_stack):
        es, gs, dispatcher, planner, engine = behavioral_stack

        # E & G. Unknown camera / insufficient evidence
        call_unk = ToolCall("c_t10_unk", "get_behavioral_prediction", {"current_camera_id": "cam_unknown"})
        res_unk = dispatcher.dispatch(call_unk)
        assert res_unk.success is True
        assert res_unk.data["predictions"] == []
        assert res_unk.data["overall_confidence"] == 0.0

        # Seed Markov transitions: cam_01 -> cam_02 (70%), cam_01 -> cam_03 (30%)
        gs.upsert_camera_node("cam_01")
        gs.upsert_camera_node("cam_02")
        gs.upsert_camera_node("cam_03")

        for i in range(7):
            gs.add_transition_edge(f"t_70_{i}", f"gid_a_{i}", "cam_01", "cam_02", 1_000 + i)
        for i in range(3):
            gs.add_transition_edge(f"t_30_{i}", f"gid_b_{i}", "cam_01", "cam_03", 2_000 + i)

        # H & I & J & K & L. Tool 10 query
        call = ToolCall("c_t10_pred", "get_behavioral_prediction", {"current_camera_id": "cam_01", "top_k": 2})
        res = dispatcher.dispatch(call)

        assert res.success is True
        preds = res.data["predictions"]
        assert len(preds) == 2
        assert preds[0]["camera_id"] == "cam_02"
        assert pytest.approx(preds[0]["probability"], 0.01) == 0.70
        assert preds[1]["camera_id"] == "cam_03"
        assert pytest.approx(preds[1]["probability"], 0.01) == 0.30

        # Probability normalization check
        sum_prob = sum(p["probability"] for p in preds)
        assert pytest.approx(sum_prob, 0.01) == 1.0

        # M. Read-only invariant check
        edges_before = gs.count_edges()
        dispatcher.dispatch(call)
        assert gs.count_edges() == edges_before

        # N. Evidence correctness
        assert len(res.evidence_list) == 2
        for ev in res.evidence_list:
            assert ev.source_store == "graph_store"
            assert ev.record_type == "behavioral_prediction"

        # O & P. Planner and Reasoning Engine E2E
        q = UserQuery("q_t10_e2e", "Where is someone likely to go from cam_01?", 1_700_000_000_000_000_000)
        r_result = engine.process_query(q)
        assert r_result.status == QueryStatus.SUCCESS
        assert len(r_result.evidence) == 2


# ── 3. HELD-OUT PREDICTION ACCURACY ─────────────────────────────────────────
class TestHeldOutPredictionAccuracy:
    def test_temporal_train_test_split_accuracy(self):
        gs = SQLiteGraphStore(":memory:")
        for c in ["c1", "c2", "c3", "c4"]:
            gs.upsert_camera_node(c)

        # Train set (t = 1,000 to 5,000 ns): 80% c1->c2, 20% c1->c3
        for i in range(80):
            gs.add_transition_edge(f"tr_train_1_{i}", f"g_tr_{i}", "c1", "c2", 1_000 + i)
        for i in range(20):
            gs.add_transition_edge(f"tr_train_2_{i}", f"g_tr_b_{i}", "c1", "c3", 2_000 + i)

        # Fit model ONLY on train range
        predictor = MarkovBehavioralPredictor()
        predictor.fit_from_graph_store(gs, start_ns=1_000, end_ns=5_000)

        # Test set (t = 6,000 to 10,000 ns): 10 test transitions from c1 (8 to c2, 2 to c3)
        test_transitions = [
            ("c1", "c2") for _ in range(8)
        ] + [
            ("c1", "c3") for _ in range(2)
        ]

        evaluated = 0
        correct_top1 = 0
        correct_top3 = 0

        for current_cam, actual_next in test_transitions:
            res = predictor.predict_next(current_cam, timestamp_ns=7_000, top_k=3)
            if res.predictions:
                evaluated += 1
                top1_cam = res.predictions[0].camera_id
                top3_cams = [p.camera_id for p in res.predictions]
                if top1_cam == actual_next:
                    correct_top1 += 1
                if actual_next in top3_cams:
                    correct_top3 += 1

        total_test_cases = len(test_transitions)
        coverage = evaluated / total_test_cases if total_test_cases > 0 else 0.0
        accuracy_top1 = correct_top1 / evaluated if evaluated > 0 else 0.0
        accuracy_top3 = correct_top3 / evaluated if evaluated > 0 else 0.0

        assert coverage == 1.0
        assert accuracy_top1 == 0.80
        assert accuracy_top3 == 1.00

        gs.close()


# ── 4. TRAJECTORY CLUSTERING RECOVERY ───────────────────────────────────────
class TestTrajectoryClusteringRecovery:
    def test_recovers_known_route_groups(self):
        # Known Route A: cam_01 -> cam_02 -> cam_03
        # Known Route B: cam_01 -> cam_04 -> cam_05
        seqs = []
        for i in range(5):
            seqs.append(TrajectorySequence(
                camera_sequence=["cam_01", "cam_02", "cam_03"],
                timestamps_ns=[1000, 2000, 3000],
                total_duration_s=2.0,
                transition_count=2,
                unique_camera_count=3,
            ))
        for i in range(5):
            seqs.append(TrajectorySequence(
                camera_sequence=["cam_01", "cam_04", "cam_05"],
                timestamps_ns=[1000, 2000, 3000],
                total_duration_s=2.0,
                transition_count=2,
                unique_camera_count=3,
            ))

        engine = TrajectoryClusteringEngine(eps=0.3, min_samples=3)
        clusters = engine.cluster(seqs)

        assert len(clusters) == 2
        sequences_in_clusters = sum(c.occurrence_count for c in clusters)
        assert sequences_in_clusters == 10


# ── 5. ANOMALY DETECTOR SCORING ─────────────────────────────────────────────
class TestBehavioralAnomalyDetectorValidation:
    def test_anomaly_scoring_differential(self):
        norm_seq = TrajectorySequence(
            camera_sequence=["c1", "c2", "c3"],
            timestamps_ns=[1000, 2000, 3000],
            total_duration_s=2.0,
            transition_count=2,
            unique_camera_count=3,
        )
        anom_seq = TrajectorySequence(
            camera_sequence=["c1", "cX", "cZ"],
            timestamps_ns=[1000, 2000, 3000],
            total_duration_s=2.0,
            transition_count=2,
            unique_camera_count=3,
        )

        cluster = TrajectoryCluster(
            cluster_id="cl_norm",
            camera_sequence=["c1", "c2", "c3"],
            occurrence_count=3,
            unique_identity_count=3,
            mean_duration_s=2.0,
            confidence=0.9,
        )
        detector = BehavioralAnomalyDetector(threshold=0.5)

        res_norm = detector.detect(norm_seq, [cluster])
        res_anom = detector.detect(anom_seq, [cluster])

        assert res_norm.anomaly_score < res_anom.anomaly_score
        assert res_norm.is_anomalous is False
        assert res_anom.is_anomalous is True


# ── 6. DETERMINISM & READ-ONLY AUDIT ────────────────────────────────────────
class TestDeterminismAndReadOnlyAudit:
    def test_functional_determinism_across_runs(self, behavioral_stack):
        es, gs, dispatcher, planner, engine = behavioral_stack

        gs.upsert_camera_node("cam_01")
        gs.upsert_camera_node("cam_02")
        gs.add_transition_edge("t1", "gid_1", "cam_01", "cam_02", 1000)

        call9 = ToolCall("c9", "get_trajectory_clusters", {})
        call10 = ToolCall("c10", "get_behavioral_prediction", {"current_camera_id": "cam_01"})

        res9_a = dispatcher.dispatch(call9)
        res9_b = dispatcher.dispatch(call9)

        res10_a = dispatcher.dispatch(call10)
        res10_b = dispatcher.dispatch(call10)

        assert res9_a.data == res9_b.data
        assert res10_a.data == res10_b.data


# ── 7. QUERY SEMANTICS AUDIT ─────────────────────────────────────────────────
class TestQuerySemanticsAudit:
    def test_natural_language_variations_and_ambiguity(self):
        planner = RuleBasedPlanner()

        # Tool 9 phrasing variations
        for text in [
            "What are the common movement patterns?",
            "Show common routes for cam_01",
            "What trajectory clusters exist?",
        ]:
            q = UserQuery("q", text, 1000)
            st, calls, exp = planner.plan(q)
            assert st == QueryStatus.SUCCESS
            assert calls[0].tool_name == "get_trajectory_clusters"

        # Tool 10 phrasing variations
        for text in [
            "Where does someone usually go after cam_01?",
            "What camera is likely next after cam_01?",
            "What are the likely next destinations from cam_01?",
        ]:
            q = UserQuery("q", text, 1000)
            st, calls, exp = planner.plan(q)
            assert st == QueryStatus.SUCCESS
            assert calls[0].tool_name == "get_behavioral_prediction"

        # Adversarial Ambiguity (missing camera context)
        q_amb = UserQuery("q_amb", "Where will they go?", 1000)
        st_amb, calls_amb, exp_amb = planner.plan(q_amb)
        assert st_amb == QueryStatus.AMBIGUOUS
        assert len(calls_amb) == 0

        # Unparseable query fallback
        q_inv = UserQuery("q_inv", "What happens next?", 1000)
        st_inv, calls_inv, exp_inv = planner.plan(q_inv)
        assert st_inv == QueryStatus.INVALID_QUERY
        assert len(calls_inv) == 0
