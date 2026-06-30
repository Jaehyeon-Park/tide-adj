from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import meep as mp
import numpy as np

from .coords import centered_grid_coords
from .objectives import _epsilon_from_medium


@dataclass
class DesignGrid:
    """Bundle a Meep MaterialGrid with its TIDE-Adj sampling metadata."""

    material_grid: mp.MaterialGrid
    center: mp.Vector3
    size: mp.Vector3
    shape: Tuple[int, int]
    background: Optional[mp.Medium] = None
    design_material: Optional[mp.Medium] = None
    material_factor: Optional[float] = None

    def __post_init__(self) -> None:
        nx, ny = self.shape
        self.spacing = (self.size.x / nx, self.size.y / ny)
        self.coords_x, self.coords_y = centered_grid_coords(
            center=self.center,
            shape=self.shape,
            spacing=self.spacing,
        )
        self.cell_area = self.spacing[0] * self.spacing[1]

        if self.material_factor is None:
            if self.background is None or self.design_material is None:
                raise ValueError("DesignGrid requires material_factor or both background and design_material")
            self.material_factor = (
                _epsilon_from_medium(self.design_material)
                - _epsilon_from_medium(self.background)
            )

    def update_weights(self, x: np.ndarray) -> None:
        """Write a flat design vector into the bundled Meep MaterialGrid."""
        self.material_grid.update_weights(np.asarray(x).reshape(self.shape))


@dataclass
class SimulationSpec:
    """Reusable Meep Simulation construction arguments."""

    cell_size: mp.Vector3
    boundary_layers: Sequence = ()
    geometry: Sequence = ()
    sources: Union[Sequence, Callable[[], Sequence]] = ()
    resolution: float = 10
    geometry_center: Optional[mp.Vector3] = None
    chunk_layout: Optional = None
    dimensions: Optional[int] = None
    eps_averaging: Optional[bool] = None

    def _resolve_sources(self, sources=None):
        selected = self.sources if sources is None else sources
        return selected() if callable(selected) else selected

    def make(self, sources=None) -> mp.Simulation:
        """Create a Meep Simulation, optionally replacing the source list."""
        kwargs = {
            "cell_size": self.cell_size,
            "boundary_layers": list(self.boundary_layers),
            "geometry": list(self.geometry),
            "sources": self._resolve_sources(sources),
            "resolution": self.resolution,
        }
        if self.geometry_center is not None:
            kwargs["geometry_center"] = self.geometry_center
        if self.chunk_layout is not None:
            kwargs["chunk_layout"] = self.chunk_layout
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        if self.eps_averaging is not None:
            kwargs["eps_averaging"] = self.eps_averaging
        return mp.Simulation(**kwargs)


@dataclass(frozen=True)
class PointTarget:
    """Point monitor and matching adjoint-source settings."""

    position: mp.Vector3
    component: int
    adjoint_source_size: Optional[mp.Vector3] = None
    adjoint_source_amplitude: float = 1.0
