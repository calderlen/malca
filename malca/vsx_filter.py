"""
Backward compatibility shim for vsx_filter.

This module maintains the old import path for backward compatibility.
New code should use: from malca.vsx.filter import ...
"""

from malca.vsx.filter import *  # noqa: F401, F403

import warnings
warnings.warn(
    "Importing from malca.vsx_filter is deprecated. "
    "Use 'from malca.vsx.filter import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)
