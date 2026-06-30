"""Lightweight time-domain adjoint helpers built on top of Meep.

TIDE-Adj intentionally keeps Meep objects visible: users still define sources,
geometry, materials, and simulations with Meep, while TIDE-Adj provides a small
driver for point-monitor time-domain adjoint objectives.
"""

__version__ = "0.5.0a1"

from .chunking import (
    SourceBoundaryDecision,
    SourceBoundaryPolicy,
    adjoint_source_boundary_workaround,
    regularize_source_size_and_amplitude,
    resolve_source_boundary_workaround,
    safe_chunk_layout,
)
from .coords import centered_grid_coords
from .sampling_grid import FastFieldGrid, FastGradientGrid, native_sampler_available
from .specs import DesignGrid, PointTarget, SimulationSpec
from .tda_objective import TDAObjective

__all__ = [
    "__version__",
    "SourceBoundaryDecision",
    "SourceBoundaryPolicy",
    "adjoint_source_boundary_workaround",
    "regularize_source_size_and_amplitude",
    "resolve_source_boundary_workaround",
    "safe_chunk_layout",
    "centered_grid_coords",
    "FastFieldGrid",
    "FastGradientGrid",
    "DesignGrid",
    "PointTarget",
    "SimulationSpec",
    "TDAObjective",
    "native_sampler_available",
]
