# Task 4 — Identity Gallery & Identity Lifecycle: Validation Report

**Status:** COMPLETE
**Phase:** 2 — Identity Persistence Layer
**Task Scope:** Task 4 only (persistent identities). No tracking, no matching, no pipeline integration.
**Date:** 2026-07-07

---

## 1. Architecture Summary

Task 4 introduces the **persistent identity substrate** for Phase 2: a minimal,
immutable identity record and the three collaborators that manage its lifetime.
It deliberately performs **no** appearance matching, similarity search, tracking,
or pipeline wiring — those are later tasks.

The design centers on **immutability**. An `Identity` is a frozen dataclass with a
read-only embedding. Nothing mutates an identity in place; every lifecycle step
returns a *new* `Identity`. This makes state transitions deterministic and free of
aliasing bugs, and lets the gallery/store treat identities as plain values.

```
                    ┌──────────────────────────────┐
                    │        identity.py           │
                    │  Identity (frozen) +         │
                    │  LifecycleState enum +       │
                    │  create_identity() factory   │
                    └──────────────┬───────────────┘
                                   │ value type
          ┌────────────────────────┼────────────────────────┐
          │                        │                         │
┌─────────▼──────────┐  ┌──────────▼───────────┐  ┌──────────▼───────────┐
│ identity_lifecycle │  │  identity_gallery.py │  │  identity_store.py   │
│  LifecycleManager  │  │   IdentityGallery    │  │  IdentityStore (ABC) │
│  NEW→ACTIVE→LOST   │  │  in-memory storage,  │  │  InMemory / JsonFile │
│  →EXPIRED (pure    │  │  duplicate-protected │  │  save/load/delete/   │
│  transitions)      │  │  (no matching)       │  │  exists (+serialize) │
└────────────────────┘  └──────────────────────┘  └──────────────────────┘
```

**Separation of concerns (each component does exactly one thing):**

| Component | Owns | Explicitly does NOT own |
|-----------|------|-------------------------|
| `identity.py` | The immutable record + state enum + factory | Any behavior/transitions |
| `identity_lifecycle.py` | Deterministic state transitions | Storage, persistence, matching |
| `identity_gallery.py` | In-memory store/retrieve/enumerate | Matching, similarity, lifecycle, persistence |
| `identity_store.py` | Persistence + serialization | Matching, lifecycle, in-memory indexing |

**Note on module placement.** The Task 4 brief specifies `gods_eye/reid/` for
these files, so they live alongside the Task 1–2 Re-ID modules. This differs from
the older `PHASE_2_IMPLEMENTATION_PLAN.md` draft (which sketched a separate
`gods_eye/identity/` package and a richer `schemas/identity.py` model). The
pre-existing `schemas/identity.py` was **left untouched** to avoid breaking Phase 1
contracts; Task 4 ships a self-contained model in `reid/identity.py` using the
NEW/ACTIVE/LOST/EXPIRED states required by this task.

---

## 2. Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `gods_eye/reid/identity.py` | ~180 | Immutable `Identity`, `LifecycleState`, `create_identity`, `new_global_id` |
| `gods_eye/reid/identity_gallery.py` | ~180 | `IdentityGallery` in-memory container + duplicate/not-found errors |
| `gods_eye/reid/identity_store.py` | ~250 | `IdentityStore` ABC, `InMemory`/`JsonFile` impls, serialization |
| `gods_eye/reid/identity_lifecycle.py` | ~200 | `LifecycleManager` deterministic transitions |
| `tests/test_identity.py` | ~430 | 44 unit tests across all components |
| `docs/phases/phase_2/TASK_4_VALIDATION.md` | — | This report |

**Modified:** `gods_eye/reid/__init__.py` — exports the new Task 4 symbols.

---

## 3. Public APIs

### `identity.py`

```python
class LifecycleState(Enum):        # NEW | ACTIVE | LOST | EXPIRED
TERMINAL_STATES: frozenset[LifecycleState]   # {EXPIRED}

@dataclass(frozen=True)
class Identity:
    global_id: str
    embedding: np.ndarray          # read-only float32, copied on construction
    created_ns: int
    last_seen_ns: int
    observation_count: int
    state: LifecycleState
    confidence: float
    expire_at_ns: int | None = None

    @property
    def is_terminal(self) -> bool
    def with_updates(self, **changes) -> Identity

def new_global_id() -> str
def create_identity(embedding, timestamp_ns, *, confidence=0.0,
                    global_id=None) -> Identity   # → state NEW, count 1
```

*Tracker IDs are intentionally absent — ADR-004: identity is permanently separated
from tracker-local IDs, which belong to ByteTrack.*

### `identity_lifecycle.py`

```python
class InvalidTransitionError(Exception)

class LifecycleManager:
    def __init__(self, settings: Settings)
    # explicit transitions (each returns a NEW Identity)
    def activate(identity, timestamp_ns) -> Identity        # NEW → ACTIVE
    def refresh(identity, timestamp_ns, *, embedding=None,
                confidence=None) -> Identity                # NEW/ACTIVE/LOST → ACTIVE
    def mark_lost(identity, timestamp_ns) -> Identity       # ACTIVE → LOST (sets expire_at_ns)
    def expire(identity) -> Identity                        # LOST → EXPIRED
    # time-driven
    def should_mark_lost(identity, current_ns) -> bool
    def should_expire(identity, current_ns) -> bool
    def advance(identity, current_ns) -> Identity           # applies ≤1 due transition
```

### `identity_gallery.py`

```python
class DuplicateIdentityError(Exception)
class IdentityNotFoundError(Exception)

class IdentityGallery:
    def insert(identity) -> None          # raises DuplicateIdentityError
    def update(identity) -> None          # raises IdentityNotFoundError
    def upsert(identity) -> None          # insert-or-replace, never raises
    def remove(global_id) -> Identity     # raises IdentityNotFoundError
    def clear() -> None
    def get(global_id) -> Identity | None
    def contains(global_id) -> bool
    def identities() -> list[Identity]
    def ids() -> list[str]
    def __len__ / __contains__ / __iter__
```

### `identity_store.py`

```python
SCHEMA_VERSION: int = 1
def identity_to_dict(identity) -> dict
def identity_from_dict(data) -> Identity        # raises ValueError on bad/old schema

class IdentityStore(ABC):
    def save(identity) -> None
    def load(global_id) -> Identity | None
    def load_all() -> list[Identity]
    def delete(global_id) -> bool               # True if a record was removed
    def exists(global_id) -> bool

class InMemoryIdentityStore(IdentityStore)      # dict-backed, non-durable
class JsonFileIdentityStore(IdentityStore)      # one atomic <id>.json per identity
```

The ABC is the extension point: a future database backend implements the same
five methods with no change to callers.

---

## 4. Lifecycle Diagram

```
                         create_identity()
                                │
                                ▼
                          ┌───────────┐
                          │    NEW    │
                          └─────┬─────┘
                                │ activate()
                                ▼
        refresh() ┌───────────────────────────┐
        (re-obs) ─┤          ACTIVE           │
                  └─────┬───────────────▲──────┘
      mark_lost()       │               │  refresh()  (re-link)
   (idle ≥ lost_timeout)│               │  clears expire_at_ns
                        ▼               │
                  ┌───────────┐         │
                  │   LOST    ├─────────┘
                  └─────┬─────┘
        expire()        │
   (now ≥ expire_at_ns  │
    = lost_at + TTL)    ▼
                  ┌───────────┐
                  │  EXPIRED  │  ── terminal (refresh/expire forbidden)
                  └───────────┘
```

**Transition guards (all deterministic, all raise `InvalidTransitionError` on
illegal source state):**

| Transition | Trigger | Guard |
|-----------|---------|-------|
| NEW → ACTIVE | `activate()` | source must be NEW |
| {NEW,ACTIVE,LOST} → ACTIVE | `refresh()` | source must **not** be EXPIRED |
| ACTIVE → LOST | `mark_lost()` / `advance()` when idle ≥ `identity_lost_timeout_s` | source must be ACTIVE |
| LOST → EXPIRED | `expire()` / `advance()` when `now ≥ expire_at_ns` | source must be LOST |

Timeouts resolve from `Settings`: `identity_lost_timeout_s` (default 300s),
`identity_ttl_s` (default 86400s / 24h).

---

## 5. Test Results

**Command:** `python -m pytest tests/test_identity.py -q`

```
44 passed in 5.71s
```

**Full suite (regression check):** `python -m pytest -q`

```
138 passed, 3 warnings in 9.84s
```

(94 pre-existing tests + 44 new = 138; the warnings are pre-existing torchreid /
supervision deprecation notices, unrelated to Task 4.)

### Coverage of required validation points

| Required validation | Tests |
|---------------------|-------|
| UUID uniqueness | `test_new_global_id_is_unique` (10k ids), `test_created_identities_have_unique_ids` |
| Identity creation | `test_create_identity_fields`, `test_create_identity_explicit_id` |
| Immutability | `test_identity_is_frozen`, `test_embedding_is_read_only`, `test_embedding_is_copied_not_aliased`, `test_with_updates_returns_new_object` |
| Lifecycle transitions | `test_activate*`, `test_refresh*`, `test_mark_lost*`, `test_relink_lost_to_active`, `test_expire*`, `test_refresh_expired_forbidden` |
| Gallery insert/update/remove | `test_gallery_insert_and_get`, `test_gallery_update*`, `test_gallery_remove*`, `test_gallery_upsert`, `test_gallery_enumerate`, `test_gallery_clear` |
| Duplicate protection | `test_gallery_duplicate_protection` |
| Store serialization | `test_serialization_round_trip`, `test_serialization_bad_version_raises`, `test_serialization_malformed_raises`, `test_in_memory_store_*`, `test_json_file_store_*` |
| Expiration logic | `test_advance_active_to_lost`, `test_advance_lost_to_expired`, `test_advance_*_before_*`, `test_should_predicates` |
| End-to-end | `test_full_lifecycle_persisted` |

### Type safety

**Command:** `python -m mypy --strict gods_eye/`

```
Success: no issues found in 39 source files
```

(35 files at baseline + 4 new Task 4 modules = 39.)

---

## 6. Known Limitations

These are **intentional** scope boundaries for Task 4, deferred to later tasks:

1. **No matching / similarity search.** The gallery is a pure key-value container.
   Appearance matching (cosine similarity, top-k, threshold) lives in the Task 2
   `Matcher` and will be composed in during a later task.
2. **No LRU eviction / `gallery_max_size` enforcement.** The gallery grows
   unbounded. Capacity management and eviction policy are out of scope here.
3. **No pipeline / IdentityWorker integration.** Nothing consumes tracking results
   or connects cameras. The identity substrate is standalone.
4. **No concurrency control.** `IdentityGallery` is single-writer by design and
   performs no internal locking. Cross-thread use must be externally synchronized
   (deferred to a later phase, per spec §3.2.4).
5. **`JsonFileIdentityStore` is a reference backend, not a production database.**
   It is durable and atomic per-record, but does no indexing, batching, or
   compaction. The `IdentityStore` ABC exists precisely so a real database can
   replace it without touching callers.
6. **No EMA embedding update.** `refresh()` can replace an embedding wholesale but
   does not implement the ADR-003 EMA (α=0.9) blend; that belongs with the
   matching/gallery-search integration task.
7. **Coexistence with `schemas/identity.py`.** A separate, richer identity schema
   exists from Phase 1 groundwork. Task 4 does not reconcile the two models; a
   future task may unify them if the pipeline requires a single canonical type.

---

## 7. Stop Condition Check

| Condition | Status |
|-----------|--------|
| Identity model exists | ✅ `reid/identity.py` |
| Gallery exists | ✅ `reid/identity_gallery.py` |
| Lifecycle manager exists | ✅ `reid/identity_lifecycle.py` |
| Store exists | ✅ `reid/identity_store.py` |
| Tests pass | ✅ 44 new / 138 total, 0 failures |
| `mypy --strict` passes | ✅ 39 files, 0 errors |
| Validation report exists | ✅ this document |

**Task 4 is complete. Task 5 not started (per stop condition).**
