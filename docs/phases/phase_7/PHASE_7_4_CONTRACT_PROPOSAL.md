# Sub-Phase 7.4 — Tool 12 Contract Proposal & Design Specification

## Executive Summary

- **Document Target**: Sub-Phase 7.4 — Tool 12 (`get_hypothesis_tree`) Contract Design Proposal
- **Repository Commit Baseline**: `fa10968a681986bd54f6b03dbedf629066c0dc4b` (`v0.6.0-phase6-freeze`)
- **Current Test Suite**: **492 / 492 passing tests**
- **Implementation Status**: **STRICT CONTRACT DESIGN ONLY. Zero runtime code, zero schema edits, zero test edits.**
- **Tool Status**:
  - Tool 11 (`get_situational_risk`): **FROZEN & VERIFIED**
  - Tool 12 (`get_hypothesis_tree`): **BLOCKED PENDING CONTRACT APPROVAL**

---

## 1. Decision Classification Taxonomy

Every architectural decision and contract rule in this proposal is explicitly classified according to the following taxonomy:

- **Category A**: Directly required by `GODS_EYE_MASTER_SPEC.md`.
- **Category B**: Required for consistency with an existing frozen contract (`gods_eye/schemas/situational.py`, `gods_eye/schemas/reasoning.py`, `SituationalRiskEvaluator`).
- **Category C**: Proposed engineering design (deterministic rule engine pattern).
- **Category D**: Open design decision requiring explicit human approval.

---

## 2. Core Semantic Definitions & Invariants

### 2.1 What is a "Hypothesis" in God's Eye?
A **Hypothesis** is a structured candidate explanation of an observed situational state or anomaly over a temporal window. It links pseudonymous subject references, evidence records, identity confidence scores, and competing candidate assignments. (**Category B**)

### 2.2 Semantic Distinctions
- **Observation**: Raw perception sighting at camera $C$ and time $t$. (**Category B**)
- **Evidence**: Provenance-tracked record linking an observation or event to underlying memory stores (`source_store`, `record_id`, `timestamp_ns`). (**Category B**)
- **Risk Signal**: Evaluated operational risk level (`LOW`..`CRITICAL`) and rule-severity score over a temporal window. (**Category B**)
- **Prediction**: Markov transition forecast of next-camera arrival probability. (**Category B**)
- **Hypothesis**: Candidate structural explanation linking multiple evidence items and competing identity assignments to explain a trajectory deviation or situational risk signal. (**Category B**)

### 2.3 Valid Hypothesis Constraints
A valid hypothesis MUST satisfy the frozen `Hypothesis` dataclass invariants:
- `hypothesis_id`: Non-empty string.
- `hypothesis_type`: Non-empty string.
- `primary_subject_ref`: Optional non-empty pseudonymous string (`subject_ref`).
- `identity_confidence`: Bounded float $\in [0.0, 1.0]$.
- `competing_candidates`: Immutable tuple of `IdentityCandidate` objects.
- `confidence`: Bounded float $\in [0.0, 1.0]$.
- `evidence_ids`: Immutable tuple of non-empty strings.
- `explanation`: Non-empty factual string summary.
- **Absence of Sensitive Fields**: MUST NOT contain `global_id`, `subject_global_id`, or `confidence_band`. (**Category B**)

### 2.4 Required Input Evidence
A hypothesis CANNOT be fabricated. It MUST be grounded in at least 1 valid canonical `Evidence` item retrieved from `SQLiteEventStore`, `SQLiteGraphStore`, or `TimelineEngine`. (**Category B**)

### 2.5 Prohibition of Causal and Intent Claims
Hypotheses MUST remain strictly factual structural descriptions of movement and anomaly data (e.g. "Trajectory deviation across Cameras 1 -> 3 with dwell elevation"). Claims of "criminal intent", "malicious threat", "suspicious motive", or human mental state are **STRICTLY PROHIBITED**. (**Category A - Master Spec §2 Rule 8**)

### 2.6 Competing Identity Candidates
Competing identities are represented as a tuple of `IdentityCandidate(subject_ref, confidence)` objects, sorted deterministically by descending confidence. Opaque pseudonymous `subject_ref` strings MUST be used. Raw `global_id` strings MUST NOT be exposed. (**Category B**)

### 2.7 Confidence Semantics
- `identity_confidence`: Bounded scalar $[0.0, 1.0]$ representing ReID/tracking association confidence.
- `confidence`: Bounded scalar $[0.0, 1.0]$ representing structural evidence support. (**Category B**)

### 2.8 Uncertainty Representation
Uncertainty is represented deterministically by listing alternative `competing_candidates` and competing `Hypothesis` nodes. Parametric confidence intervals ("95% CI") or Gaussian noise estimates are explicitly rejected. (**Category B**)

---

## 3. Hypothesis Tree Architecture & Topology

### 3.1 Definition of "Hypothesis Tree"
A **Hypothesis Tree** is a hierarchical forest structure where root nodes represent primary macro hypotheses explaining a situational risk or anomaly, and child nodes represent sub-hypotheses explaining specific temporal or spatial sub-components. (**Category C**)

### 3.2 Graph Topology
The structure is a **Ranked Forest of Bounded Trees** (a Directed Acyclic Graph mapped cleanly into tree node representation). (**Category C**)

### 3.3 Parent-Child Relationship
- **Parent Node**: Explains a composite situational anomaly (e.g. "Restricted Zone Dwell Anomaly").
- **Child Node**: Explains a constituent sub-component (e.g. "Transition Delay on Camera 2 -> Camera 3"). (**Category C**)

### 3.4 Child Node Generation Criteria
A child hypothesis is generated when a parent hypothesis relies on compound evidence items that can be decomposed into distinct spatial or temporal sub-anomalies. (**Category C**)

---

## 4. Ranking, Sorting & Pruning Contracts

### 4.1 Ranking Score Formula
Candidate hypotheses are ranked using a deterministic composite score:
$$S = \text{confidence} \times \text{identity\_confidence}$$
(**Category C**)

### 4.2 Deterministic Tie-Breaking Rules
When two hypothesis nodes have identical composite scores $S$:
1. Higher `confidence` wins.
2. Higher `identity_confidence` wins.
3. Lexicographically smaller `hypothesis_id` wins.
(**Category B - Consistency with Phase 4/5/6 composite keys**)

### 4.3 Depth, Branching & Size Bounds
- **Maximum Depth ($D$)**: $3$ (**Category C**)
- **Maximum Branching Factor ($B$)**: $5$ (**Category C**)
- **Maximum Total Nodes Per Tree**: $25$ (**Category C**)

### 4.4 Pruning Rules
1. Any candidate node with `confidence < min_confidence` (default $0.20$) is pruned.
2. At each tree level, only the top $B=5$ ranked child nodes are retained; lower-ranked children are pruned. (**Category C**)

### 4.5 Minimum Evidence & Insufficient Evidence Handling
- Minimum 1 valid `Evidence` item per hypothesis node.
- If zero evidence exists in the requested window, return `ToolResult(success=True, data={"hypothesis_trees": [], "count": 0}, evidence_list=[])` with `QueryStatus.INSUFFICIENT_EVIDENCE`. (**Category B**)

---

## 5. Inter-Module Integration & Data Flow

### 5.1 Interaction with Tool 11 (`get_situational_risk`)
Tool 11 evaluates overall operational risk signals. Tool 12 (`get_hypothesis_tree`) decomposes high-risk evaluation windows into structured hypothesis trees to explain the underlying risk drivers. Tool 12 can consume `RiskSignal` records emitted by `SituationalRiskEvaluator`. (**Category C**)

### 5.2 Interaction with Phase 6 Behavioral Predictor
Tool 12 consumes `PredictionResult` outputs from `MarkovBehavioralPredictor` as evidence when constructing transition sub-hypotheses. It does NOT mutate predictor transition matrices. (**Category B**)

### 5.3 Evidence Attachment
Evidence IDs are extracted directly from input `Evidence` objects and attached to `Hypothesis.evidence_ids`. (**Category B**)

---

## 6. Privacy, Security & Determinism Contracts

### 6.1 Privacy Preservation
- `subject_ref` pseudonyms used exclusively in public contracts.
- Zero raw `global_id` UUIDs, `subject_global_id`, face/body ReID embeddings, or identity gallery images may appear in `ToolResult` payloads or explanations. (**Category A & B**)

### 6.2 Determinism Contract
- 100% deterministic rule-based generation. Zero random number calls, zero LLMs in hypothesis core.
- Tested by executing 100 repeated evaluation calls and asserting byte-for-byte equality. (**Category B**)

### 6.3 Anti-Hallucination Guarantee
All hypothesis explanation strings are generated via deterministic templates reporting verified evidence attributes. Non-deterministic LLM text generation is prohibited in the core engine. (**Category A & B**)

---

## 7. Tool 12 Interface Specifications

### 7.1 Proposed Schema Additions (`gods_eye/schemas/situational.py`)
```python
@dataclass(frozen=True)
class HypothesisNode:
    node_id: str
    hypothesis: Hypothesis
    parent_node_id: Optional[str]
    children: tuple[HypothesisNode, ...]
    depth: int

@dataclass(frozen=True)
class HypothesisTree:
    tree_id: str
    root_node: HypothesisNode
    total_nodes: int
    max_depth: int
```
(**Category C**)

### 7.2 Tool 12 Input Contract
- `window_start_ns`: Integer, required, $\ge 0$.
- `window_end_ns`: Integer, required, $\ge 0$, duration $\le 24\text{h}$.
- `subject_ref`: Optional string pseudonym filter.
- `max_depth`: Optional integer, default $3$, max $5$.
- `min_confidence`: Optional float, default $0.20$.
(**Category C**)

### 7.3 Tool 12 Output Contract
Returns `ToolResult`:
- `data`: `{"hypothesis_trees": [tree.to_dict() for tree in trees], "count": len(trees)}`
- `evidence_list`: Canonical `Evidence` objects referenced in the trees.
(**Category C**)

---

## 8. OPEN DECISIONS REQUIRING HUMAN APPROVAL

The following architectural choices are explicitly classified as **Category D (Open Decisions)** and require explicit human approval before Sub-Phase 7.4 implementation begins:

| Decision ID | Open Architectural Choice | RECOMMENDED DEFAULT | Rationale |
|---|---|---|---|
| **D-01** | Tree Topology: Maximum Tree Depth ($D$) | **$D = 3$** | Prevents exponential node expansion while allowing macro $\to$ micro hypothesis decomposition. |
| **D-02** | Tree Topology: Maximum Branching Factor ($B$) | **$B = 5$** | Bounds child hypotheses per node to top 5 candidates. |
| **D-03** | Pruning Threshold: Minimum Node Confidence | **$\text{min\_confidence} = 0.20$** | Filters out weak, unsupported candidate hypotheses. |
| **D-04** | Node Schema: Include `HypothesisNode` & `HypothesisTree` in `gods_eye/schemas/situational.py` | **Approve Schema Addition** | Provides clean, typed dataclass structure for Tool 12 output. |
| **D-05** | Engine Design: Pure Deterministic Rule Engine vs ML | **Pure Deterministic Rule Engine** | Satisfies all Master Spec requirements with zero training overhead, 100% determinism, and sub-millisecond latency. |

---

## 9. Out of Scope

- LLM narrative generation or intent inferencing
- Database schema changes or migrations
- Alert/notification dispatch
- Sub-Phase 7.5 features

---

## 10. Conclusion & Verification Baseline

This contract proposal establishes the complete specification required to unblock Sub-Phase 7.4 once Category D decisions are approved.

```text
SOURCE FILES MODIFIED: 0
TEST FILES MODIFIED: 0
TOOL 11: FROZEN
TOOL 12: NOT IMPLEMENTED
PHASE 6: FROZEN
PHASE 7.1–7.3: FROZEN
```
