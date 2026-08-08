"""Re-ID layer — OSNet embedding extraction for identity persistence.

Phase 2 deliverable: Converts person crops into 512-dimensional
appearance embeddings using OSNet (ADR-001), computes cosine
similarity (ADR-002), and provides stateless matching primitives.

Modules:
    embedding_extractor: Abstract interface for all embedding extractors.
    osnet_extractor: Concrete OSNet implementation via torchreid.
    similarity: Cosine similarity engine with zero-vector safety.
    matcher: Stateless matching primitives (threshold, top-k ranking).
    identity: Immutable persistent identity model + lifecycle states.
    identity_gallery: In-memory storage/retrieval of identities.
    identity_store: Persistence abstraction (in-memory / JSON-file).
    identity_lifecycle: Deterministic lifecycle state machine.
    identity_mapper: Identity Mapping Layer — Phase 2 orchestrator (Task 7).
"""

from gods_eye.reid.embedding_extractor import EmbeddingExtractor
from gods_eye.reid.identity import (
    TERMINAL_STATES,
    Identity,
    LifecycleState,
    create_identity,
    new_global_id,
)
from gods_eye.reid.identity_gallery import (
    DuplicateIdentityError,
    IdentityGallery,
    IdentityNotFoundError,
)
from gods_eye.reid.identity_lifecycle import (
    InvalidTransitionError,
    LifecycleManager,
)
from gods_eye.reid.identity_store import (
    SCHEMA_VERSION,
    IdentityStore,
    InMemoryIdentityStore,
    JsonFileIdentityStore,
    identity_from_dict,
    identity_to_dict,
)
from gods_eye.reid.matcher import MatchResult, Matcher
from gods_eye.reid.osnet_extractor import OSNetExtractor
from gods_eye.reid.similarity import (
    cosine_similarity,
    cosine_similarity_matrix,
    top_k_indices,
)

# Identity Mapper types are lazy-imported to avoid a circular dependency:
# reid/__init__ → identity_mapper → pipeline → reid/__init__
_IDENTITY_MAPPER_NAMES = {
    "IdentityMapper", "IdentityResult", "IdentityTransition", "TransitionType",
}


def __getattr__(name: str) -> object:
    if name in _IDENTITY_MAPPER_NAMES:
        from gods_eye.reid import identity_mapper as _im

        return getattr(_im, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Re-ID (Tasks 1-2)
    "EmbeddingExtractor",
    "OSNetExtractor",
    "cosine_similarity",
    "cosine_similarity_matrix",
    "top_k_indices",
    "Matcher",
    "MatchResult",
    # Identity model + lifecycle (Task 4)
    "Identity",
    "LifecycleState",
    "TERMINAL_STATES",
    "create_identity",
    "new_global_id",
    "IdentityGallery",
    "DuplicateIdentityError",
    "IdentityNotFoundError",
    "LifecycleManager",
    "InvalidTransitionError",
    "IdentityStore",
    "InMemoryIdentityStore",
    "JsonFileIdentityStore",
    "identity_to_dict",
    "identity_from_dict",
    "SCHEMA_VERSION",
    # Identity Mapper Layer (Task 7) — lazy-loaded
    "IdentityMapper",
    "IdentityResult",
    "IdentityTransition",
    "TransitionType",
]
