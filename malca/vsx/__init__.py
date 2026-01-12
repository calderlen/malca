"""
VSX (Variable Star Index) catalog tools.

This subpackage provides tools for working with the VSX catalog:
- filter: Clean VSX catalog by variability class
- crossmatch: Crossmatch ASAS-SN with VSX (proper motion corrected)
- reproducibility: Test recovery of VSX objects

Backward compatibility imports:
These allow existing code to continue using the old import paths.
"""

# Backward compatibility - allow old import paths to still work
from malca.vsx.filter import *  # noqa: F401, F403
from malca.vsx.crossmatch import *  # noqa: F401, F403
from malca.vsx.reproducibility import *  # noqa: F401, F403

__all__ = [
    # Re-export everything from submodules for backward compatibility
]
