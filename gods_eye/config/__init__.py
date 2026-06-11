"""Configuration loader — §6 and §14 of GODS_EYE_MASTER_SPEC.

All configuration is resolved from environment variables with typed defaults.
No hidden globals — all state is passed explicitly.
"""

from gods_eye.config.settings import Settings

__all__ = ["Settings"]
