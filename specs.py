from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import meep as mp
import numpy as np

from .coords import centered_grid_coords
from .objectives import _epsilon_from_medium


@dataclass
class DesignGrid:
    """Bundle a Meep MaterialGrid with TIDE-Adj design-grid metadata.

    Args:
        material_grid: Meep ``MaterialGrid`` whose weights are optimized.
        center: Physical center of the design region.
        size: Physical size of the design region.
        shape: Number of design variables as ``(nx, ny)``.
        background: Background medium used to infer ``d epsilon / d rho``.
        design_material: Design medium used to infer ``d epsilon / d rho``.
        material_factor: Explicit ``d epsilon / d rho``. If omitted,
            ``background`` and ``design_material`` are required.

    Attributes:
        spacing: Physical pixel spacing as ``(dx, dy)``.
        coords_x: x coordinates used for design-region field sampling.
        coords_y: y coordinates used for design-region field sampling.
        cell_area: Area represented by one design variable.
    """

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
        """Write a flat design vector into the bundled Meep MaterialGrid.

        Args:
            x: Flat design vector with ``shape[0] * shape[1]`` entries.
        """
        self.material_grid.update_weights(np.asarray(x).reshape(self.shape))


@dataclass
class SimulationSpec:
    """Reusable Meep ``Simulation`` construction arguments.

    Args:
        cell_size: Meep simulation cell size.
        boundary_layers: Boundary layers passed to ``mp.Simulation``.
        geometry: Geometry objects passed to ``mp.Simulation``.
        sources: Default source list, or a zero-argument callable returning one.
        resolution: Meep spatial resolution.
        geometry_center: Optional Meep ``geometry_center``.
        chunk_layout: Optional MPI chunk layout passed to Meep.
        dimensions: Optional simulation dimensionality.
        eps_averaging: Optional Meep subpixel averaging flag.
    """

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
        """Create a Meep ``Simulation``.

        Args:
            sources: Optional replacement source list. When omitted, the
                bundled ``sources`` value is used.

        Returns:
            Newly constructed Meep simulation.
        """
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
    """Point monitor and matching adjoint-source settings.

    Args:
        position: Physical point where the forward field is monitored and the
            adjoint source is placed.
        component: Meep field component, e.g. ``mp.Ez``.
        adjoint_source_size: Meep source size for adjoint injection.
        adjoint_source_amplitude: Scalar amplitude multiplier for the adjoint
            source.
    """

    position: mp.Vector3
    component: int
    adjoint_source_size: Optional[mp.Vector3] = None
    adjoint_source_amplitude: float = 1.0
