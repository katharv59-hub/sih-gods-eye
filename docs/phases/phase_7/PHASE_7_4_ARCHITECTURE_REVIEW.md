# Sub-Phase 7.4 — Hypothesis Engine & Tool 12 Architecture & Contract Review

## Executive Summary

- **Target Phase**: Sub-Phase 7.4 — Hypothesis Tree Engine & Tool 12 (`get_hypothesis_tree`)
- **Repository Baseline Commit**: `fa10968a681986bd54f6b03dbedf629066c0dc4b` (`v0.6.0-phase6-freeze`)
- **Current Test Count**: **492 / 492 passing tests**
- **Sub-Phase 7.3 Status**: Tool 11 (`get_situational_risk`) COMPLETE; Tool 12 (`get_hypothesis_tree`) **BLOCKED due to CONTRACT GAP**.
- **Review Objective**: Perform a read-only architecture and contract gap review for Tool 12 without inventing implementation details or modifying source files.
- **Overall Recommendation**: **BLOCKED UNTIL CONTRACT IS COMPLETE**

---

## 1. Authoritative Document Traceability Matrix

An audit of existing repository specifications (`GODS_EYE_MASTER_SPEC.md`, Phase 7 architecture/contract/completion reviews, and schemas) classifies the current status of Tool 12 requirements:

| Dimension | Master Spec Reference | Existing Implementation | Classification | Status & Gap Analysis |
|---|---|---|---|---|
| `Hypothesis` Schema | §4, §18 Canonical Schemas | `gods_eye/schemas/situational.py` | **SPECIFIED** | Frozen dataclass `Hypothesis` implemented in Sub-Phase 7.1 |
| Privacy Model | §2 Rule 6, §18 | `gods_eye/schemas/situational.py` | **SPECIFIED** | Strictly uses `subject_ref`; forbids `global_id` / `subject_global_id` |
| Determinism & Immutability | §2 Rule 8, §4 | Dataclass `frozen=True` | **SPECIFIED** | Immutable collection tuples, deterministic equality |
| `HypothesisGenerator` Engine | §18 Architectural Horizon | None (`gods_eye/situational/`) | **UNSPECIFIED** | No hypothesis generation engine exists in the codebase |
| Hypothesis Ranking Equation | §18 | None | **UNSPECIFIED** | Mathematical ranking formula for candidate hypotheses is undefined |
| Competing Candidate Selection | §5, §18 | None | **UNSPECIFIED** | Algorithm to select and bound `competing_candidates` is undefined |
| Hypothesis Tree Structure | §18 Horizon | None | **UNSPECIFIED** | Node parent-child hierarchy schema for `get_hypothesis_tree` is undefined |
| Tree Expansion & Pruning | §18 | None | **UNSPECIFIED** | Maximum tree depth ($D$), branching factor ($B$), and confidence thresholds are undefined |
| Causal vs Correlational Claims | §2 Rule 8 | `explanation` text string | **PARTIALLY SPECIFIED** | Explanations must be factual summaries, not intent inferences |
| Tool 12 NLQ Intent Patterns | §14 Planner | `RuleBasedPlanner` | **UNSPECIFIED** | Intent regex patterns for hypothesis queries are not registered |

---

## 2. Detailed Contract Gap Analysis for Tool 12

### Gap 1: Missing Hypothesis Generator Engine (`HypothesisGenerator`)
Sub-Phase 7.1 defined the frozen `Hypothesis` data contract, and Sub-Phase 7.2 implemented `SituationalRiskEvaluator` (which produces `RiskSignal`). However, no `HypothesisGenerator` engine exists to evaluate evidence signals and generate candidate `Hypothesis` instances.

### Gap 2: Missing Hypothesis Tree Schema (`HypothesisTree` / `HypothesisNode`)
Tool 12 is named `get_hypothesis_tree`. However, Phase 7.1 schemas only define a flat `Hypothesis` dataclass. No tree structure contract (`HypothesisNode` with `child_hypotheses: tuple[...]` or `parent_id: Optional[str]`) exists in `gods_eye/schemas/situational.py`.

### Gap 3: Missing Deterministic Pruning & Branching Boundaries
Generating hypothesis trees over evidence spaces has exponential worst-case complexity $O(B^D)$ where $B$ is branching factor and $D$ is depth. Without hard contract bounds on $D \le 3$, $B \le 5$, and minimum candidate confidence $\ge 0.20$, tree generation risks unbound latency.

---

## 3. Required Pre-Implementation Contracts

Before Tool 12 (`get_hypothesis_tree`) can safely be implemented in Sub-Phase 7.4, the following canonical contracts must be established:

### 3.1 Proposed `HypothesisNode` Schema Contract
```python
@dataclass(frozen=True)
class HypothesisNode:
    node_id: str
    hypothesis: Hypothesis
    parent_node_id: Optional[str]
    children: tuple[HypothesisNode, ...]
    depth: int
```

### 3.2 Proposed `HypothesisGenerator` Engine Specification
```python
class HypothesisGenerator:
    """Deterministic candidate hypothesis generation engine."""

    def __init__(self, generator_version: str = "v7.4.0") -> None:
        self._generator_version = generator_version

    def generate_hypotheses(
        self,
        evidence_items: tuple[Evidence, ...],
        max_depth: int = 3,
        max_branching: int = 5,
        min_confidence: float = 0.20,
    ) -> tuple[HypothesisNode, ...]:
        """Generate a deterministic forest of candidate hypothesis trees."""
        ...
```

---

## 4. Privacy & Determinism Boundaries

- **Privacy Constraint**: `HypothesisNode` and `Hypothesis` payloads MUST NOT expose `global_id`, `subject_global_id`, ReID embeddings, or gallery images.
- **Determinism Constraint**: Tree sorting and candidate ordering MUST be 100% deterministic using stable field sorting (`sorted(children, key=lambda n: n.hypothesis.confidence, reverse=True)`).
- **Read-Only Constraint**: Tree generation must operate strictly in-memory without side-effects on underlying persistent stores (`SQLiteEventStore`, `SQLiteGraphStore`).

---

## 5. Proposed Sub-Phase 7.4 Work Breakdown

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SUB-PHASE 7.4.1: Hypothesis Tree Schemas & Generator Engine                 │
│ - Create HypothesisNode dataclass in gods_eye/schemas/situational.py        │
│ - Implement HypothesisGenerator in gods_eye/situational/hypothesis.py      │
│ - Unit tests in tests/test_hypothesis_generator.py                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ SUB-PHASE 7.4.2: Tool 12 Integration & NLQ Router                           │
│ - Register get_hypothesis_tree in ToolDispatcher (gods_eye/reasoning/tools)│
│ - Register Tool 12 routing in RuleBasedPlanner (gods_eye/reasoning/planner) │
│ - Integration tests in tests/test_situational_tool12.py                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Verification of Existing Invariants

- **Tool 1–11 Frozen Status**: Tools 1–11 baseline behavior in `ToolDispatcher` and `RuleBasedPlanner` remains 100% untouched.
- **Phase 6 Freeze**: Core Phase 1–6 engines, databases, tracking, ReID, and reasoning cores remain frozen.

---

## 7. Final Preflight Recommendation

### Recommendation: **BLOCKED UNTIL CONTRACT IS COMPLETE**

Tool 12 implementation remains **BLOCKED** until formal `HypothesisNode` tree schemas and `HypothesisGenerator` engine contracts are specified and approved in a Sub-Phase 7.4 Contract Review.
