"""
Backward compatibility shim for vsx_reproducibility.

This module maintains the old import path for backward compatibility.
New code should use: from malca.vsx.reproducibility import ...
"""

from malca.vsx.reproducibility import *  # noqa: F401, F403

import warnings
warnings.warn(
    "Importing from malca.vsx_reproducibility is deprecated. "
    "Use 'from malca.vsx.reproducibility import ...' instead.",
    DeprecationWarning,
    stacklevel=2
)
