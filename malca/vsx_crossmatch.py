"""
Backward compatibility shim for vsx_crossmatch.

This module maintains the old import path for backward compatibility.
New code should use: from malca.vsx.crossmatch import ...
"""

from malca.vsx.crossmatch import *  # noqa: F401, F403

import warnings
warnings.warn(
    "Importing from malca.vsx_crossmatch is deprecated. "
    "Use 'from malca.vsx.crossmatch import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)
