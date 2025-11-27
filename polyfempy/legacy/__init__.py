"""Legacy API: Original PolyFEM Python bindings.

This package contains the original C++ bindings API including Problem, Problems,
Selection, and command-line tools. These are preserved for backward compatibility
and advanced use cases.

The new recommended API is in polyfempy.api with simplified interfaces.
"""

# These imports are currently disabled in the main __init__.py
# Uncomment to use the legacy API:

# from .Problem import Problem
# from .Problems import (
#     Franke,
#     GenericScalar,
#     Gravity,
#     Torsion,
#     GenericTensor,
#     Flow,
#     DrivenCavity,
#     FlowWithObstacle,
# )
# from .Selection import Selection

__all__ = []


