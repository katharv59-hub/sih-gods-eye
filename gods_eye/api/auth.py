"""Token-based Authentication — Phase 10.4 (SIH 26187).

Reuses GODS_EYE_API_TOKEN and GODS_EYE_SCOPE pattern from the master spec.
"""

from __future__ import annotations

import os
from typing import Optional


def verify_token(token: Optional[str]) -> bool:
    """Verify API token against GODS_EYE_API_TOKEN environment variable.

    Args:
        token: Token to verify.

    Returns:
        True if token matches or no token is configured (dev mode).
    """
    expected = os.environ.get("GODS_EYE_API_TOKEN")
    if expected is None:
        # No token configured — allow all access (dev mode)
        return True
    return token == expected


def check_scope(token: Optional[str], required_scope: str) -> bool:
    """Check if token has required scope.

    Args:
        token: API token.
        required_scope: Required scope (e.g. "behavioral", "alerts").

    Returns:
        True if scope is available.
    """
    if not verify_token(token):
        return False

    configured_scope = os.environ.get("GODS_EYE_SCOPE", "")
    if not configured_scope:
        # No scope restriction configured — allow all
        return True

    return required_scope in configured_scope.split(",")
