"""Model definitions for CalibFuse.

Only the main model class :class:`~nets.fusion.CalibFuse` is exposed;
import submodules from their own modules.
"""

from .fusion import CalibFuse

__all__ = ["CalibFuse"]
