from typing import Optional

import meep as mp
import numpy as np
from numpy.typing import NDArray
import scipy.interpolate as spi


def _epsilon_from_medium(medium: mp.Medium) -> float:
    epsilon_diag = getattr(medium, "epsilon_diag", None)
    if epsilon_diag is None:
        raise ValueError("material_factor is required for materials without epsilon_diag")
    return float(epsilon_diag.x)


class _PointTarget:
    """Point-monitor target adapter for TDAObjective.

    This is the time-domain counterpart of Meep adjoint's objective-quantity
    objects: it knows how to record the forward monitor signal, place the
    corresponding adjoint point source, and scale the final design gradient.
    The scalar FoM itself is supplied to TDAObjective.
    """

    def __init__(
        self,
        monitor_position: mp.Vector3,
        component: int,
        cell_area: float,
        adjoint_source_size: Optional[mp.Vector3] = None,
        adjoint_source_amplitude: float = 1.0,
        background: Optional[mp.Medium] = None,
        design_material: Optional[mp.Medium] = None,
        material_factor: Optional[float] = None,
    ) -> None:
        """Define a point monitor and its matching time-domain adjoint source.

        Args:
            monitor_position: Physical point at which the forward field is
                sampled and the adjoint source is placed.
            component: Meep field component, e.g. ``mp.Ez``.
            cell_area: Area represented by one 2D design variable. The final
                gradient is multiplied by this value.
            adjoint_source_size: Meep source size for the adjoint injection.
                Use zero size for a point source or a small line segment to
                regularize MPI boundary placement.
            adjoint_source_amplitude: Scalar multiplier for the adjoint source.
            background: Background medium used to infer permittivity contrast.
            design_material: Design material used to infer permittivity
                contrast.
            material_factor: Optional explicit ``d epsilon / d rho``. If this
                is supplied, ``background`` and ``design_material`` are not
                required.
        """
        if material_factor is None:
            if background is None or design_material is None:
                raise ValueError("TDAObjective requires material_factor or both background and design_material")
            material_factor = _epsilon_from_medium(design_material) - _epsilon_from_medium(background)

        self.monitor_position = monitor_position
        self.component = component
        self.cell_area = cell_area
        self.adjoint_source_size = adjoint_source_size if adjoint_source_size is not None else mp.Vector3()
        self.adjoint_source_amplitude = adjoint_source_amplitude
        self.material_factor = material_factor

    def record_monitor(self, sim: mp.Simulation) -> complex:
        """Sample the monitored field component during the forward run.

        Args:
            sim: Active forward Meep simulation.

        Returns:
            Complex field value at ``monitor_position``.
        """
        return sim.get_field_point(self.component, self.monitor_position)

    def adjoint_sources(
        self,
        adjoint_signal: NDArray[np.complex128],
        actual_time: float,
    ) -> list:
        """Create the time-reversed adjoint source from ``dFoM/dE(t)``.

        Args:
            adjoint_signal: Continuous-time adjoint signal sampled on the
                forward monitor time grid.
            actual_time: Final time reached by the forward simulation.

        Returns:
            List of Meep sources to pass back into the simulation factory.
        """
        adj_sig = adjoint_signal[::-1].copy()
        t_adj = np.linspace(0, actual_time, len(adj_sig))
        adj_interp = spi.interp1d(t_adj, adj_sig, kind="cubic", fill_value=0j, bounds_error=False)
        return [
            mp.Source(
                mp.CustomSource(src_func=lambda t: complex(adj_interp(t))),
                component=self.component,
                center=self.monitor_position,
                size=self.adjoint_source_size,
                amplitude=self.adjoint_source_amplitude,
            )
        ]

    def gradient(
        self,
        grad_grid: NDArray[np.complex128],
        dt_eff: float,
    ) -> NDArray[np.float64]:
        """Convert the accumulated field product into a flat design gradient.

        Args:
            grad_grid: Complex field-product grid accumulated during the
                adjoint run.
            dt_eff: Sampling interval used for the field-product integral.

        Returns:
            Flattened real-valued gradient with one entry per design variable.
        """
        return grad_grid.real.flatten() * self.cell_area * dt_eff * self.material_factor
