"""Fast field sampling helpers for Meep time-domain adjoint loops.

The public behavior mirrors repeated `Simulation.get_field_point` calls on a
rectangular design grid. When the optional native extension is available, the
same sampling and accumulation are performed through Meep's lower-level field
object to reduce Python-call overhead.
"""

import meep as mp
from meep.simulation import py_v3_to_vec
import numpy as np
from numpy.typing import NDArray
from typing import Any, List, Optional, Sequence

try:
    from . import native_sampler
except Exception:
    native_sampler = None

_native_sampler_enabled = native_sampler is not None


def native_sampler_available() -> bool:
    """Return whether the optional native sampler is currently usable.

    Returns:
        ``True`` when ``tide_adj.native_sampler`` was imported and has not failed
        during the current process. TIDE-Adj automatically falls back to pure
        Python Meep sampling if this becomes ``False``.
    """
    return _native_sampler_enabled


def _simulation_has_symmetry(sim: mp.Simulation) -> bool:
    return bool(getattr(sim, "symmetries", []))


def make_field_vecs(sim: mp.Simulation, coords_x: Sequence[float], coords_y: Sequence[float]) -> List[Any]:
    """Convert physical coordinates into Meep internal field vectors.

    Args:
        sim: Active Meep simulation.
        coords_x: x coordinates of the 2D sampling grid.
        coords_y: y coordinates of the 2D sampling grid.

    Returns:
        Flat list of Meep internal vectors ordered as ``for x`` then ``for y``.
    """
    return [
        py_v3_to_vec(sim.dimensions, mp.Vector3(xv, yv), sim.is_cylindrical)
        for xv in coords_x
        for yv in coords_y
    ]


class FastFieldGrid:
    """Sample one Meep field component on a fixed 2D point grid.

    Args:
        sim: Active Meep simulation whose fields will be sampled.
        component: Meep field component, e.g. ``mp.Ez``.
        coords_x: x coordinates of the sampling grid.
        coords_y: y coordinates of the sampling grid.

    Attributes:
        shape: Output grid shape ``(len(coords_x), len(coords_y))``.
    """

    def __init__(
        self,
        sim: mp.Simulation,
        component: int,
        coords_x: Sequence[float],
        coords_y: Sequence[float],
    ) -> None:
        self.sim = sim
        self.component = component
        self.coords_x = list(coords_x)
        self.coords_y = list(coords_y)
        self.shape = (len(self.coords_x), len(self.coords_y))
        self.has_symmetry = _simulation_has_symmetry(sim)
        self.field_vecs = None
        self.plan = None

    def ensure_plan(self) -> Optional[Any]:
        """Create the native sampling plan once per simulation when possible.

        Returns:
            Native sampler plan object, or ``None`` when the native extension is
            unavailable and TIDE-Adj should use the Python fallback path.
        """
        global _native_sampler_enabled
        if self.has_symmetry:
            return None
        if self.plan is None and _native_sampler_enabled and hasattr(native_sampler, "create_component_grid_plan"):
            try:
                self.plan = native_sampler.create_component_grid_plan(
                    int(self.sim.fields.this),
                    self.coords_x,
                    self.coords_y,
                    int(self.component),
                )
            except Exception:
                _native_sampler_enabled = False
        return self.plan

    def ensure_field_vecs(self) -> List[Any]:
        """Create Python fallback coordinate vectors once per simulation.

        Returns:
            Flat list of Meep internal field vectors used by
            ``fields.get_field_from_comp``.
        """
        if self.field_vecs is None:
            self.field_vecs = make_field_vecs(self.sim, self.coords_x, self.coords_y)
        return self.field_vecs

    def sample(self) -> NDArray[np.complex128]:
        """Sample the field grid.

        Returns:
            Complex-valued array with shape ``(len(coords_x), len(coords_y))``.
            Values match repeated Meep field sampling at the same physical
            coordinates.
        """
        global _native_sampler_enabled
        sample_plan = self.ensure_plan()
        if _native_sampler_enabled and sample_plan is not None and hasattr(native_sampler, "sample_component_grid_plan_allreduced"):
            try:
                return native_sampler.sample_component_grid_plan_allreduced(sample_plan)
            except Exception:
                _native_sampler_enabled = False

        if _native_sampler_enabled:
            try:
                if self.has_symmetry and hasattr(native_sampler, "sample_component_grid"):
                    return native_sampler.sample_component_grid(
                        int(self.sim.fields.this),
                        self.coords_x,
                        self.coords_y,
                        int(self.component),
                    )
                if hasattr(native_sampler, "sample_component_grid_allreduced"):
                    return native_sampler.sample_component_grid_allreduced(
                        int(self.sim.fields.this),
                        self.coords_x,
                        self.coords_y,
                        int(self.component),
                    )
                return native_sampler.sample_component_grid(
                    int(self.sim.fields.this),
                    self.coords_x,
                    self.coords_y,
                    int(self.component),
                )
            except Exception:
                _native_sampler_enabled = False

        get_field = self.sim.fields.get_field_from_comp
        return np.array(
            [get_field(self.component, field_vec) for field_vec in self.ensure_field_vecs()],
            dtype=complex,
        ).reshape(self.shape)

    def product(self, multiplier: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Sample the field and multiply it by a grid-shaped factor.

        Args:
            multiplier: Array broadcastable to ``self.shape``.

        Returns:
            ``sample() * multiplier`` as a complex grid.
        """
        global _native_sampler_enabled
        if self.has_symmetry:
            return self.sample() * multiplier
        if _native_sampler_enabled and hasattr(native_sampler, "accumulate_component_product_allreduced"):
            try:
                return native_sampler.accumulate_component_product_allreduced(
                    int(self.sim.fields.this),
                    self.coords_x,
                    self.coords_y,
                    int(self.component),
                    np.asarray(multiplier, dtype=np.complex128),
                )
            except Exception:
                _native_sampler_enabled = False

        return self.sample() * multiplier

    def accumulate_product(
        self,
        multiplier: NDArray[np.complex128],
        accumulator: NDArray[np.complex128],
    ) -> bool:
        """Accumulate ``field * multiplier`` into ``accumulator``.

        Args:
            multiplier: Grid-shaped factor, typically ``dE/dt`` from the
                forward run.
            accumulator: Complex grid updated in place.

        Returns:
            ``True`` if native in-place accumulation succeeded, otherwise
            ``False`` so the caller can use the Python fallback.
        """
        global _native_sampler_enabled
        if self.has_symmetry:
            return False
        sample_plan = self.ensure_plan()
        if _native_sampler_enabled and sample_plan is not None and hasattr(native_sampler, "accumulate_component_product_plan_local_inplace"):
            try:
                native_sampler.accumulate_component_product_plan_local_inplace(
                    sample_plan,
                    np.asarray(multiplier, dtype=np.complex128),
                    accumulator,
                )
                return True
            except Exception:
                _native_sampler_enabled = False

        if _native_sampler_enabled and hasattr(native_sampler, "accumulate_component_product_local_inplace"):
            try:
                native_sampler.accumulate_component_product_local_inplace(
                    int(self.sim.fields.this),
                    self.coords_x,
                    self.coords_y,
                    int(self.component),
                    np.asarray(multiplier, dtype=np.complex128),
                    accumulator,
                )
                return True
            except Exception:
                _native_sampler_enabled = False
        return False

    @staticmethod
    def supports_local_accumulation() -> bool:
        """Return whether native rank-local accumulation and reduction exist."""
        return (
            native_sampler_available()
            and hasattr(native_sampler, "accumulate_component_product_local_inplace")
            and hasattr(native_sampler, "reduce_complex_grid_sum")
        )

    @staticmethod
    def reduce(local_grid: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """MPI-sum a rank-local complex grid when the native reducer exists.

        Args:
            local_grid: Rank-local complex accumulator.

        Returns:
            MPI-summed grid on all ranks, or ``local_grid`` when no native
            reducer is available.
        """
        global _native_sampler_enabled
        if _native_sampler_enabled and hasattr(native_sampler, "reduce_complex_grid_sum"):
            try:
                return native_sampler.reduce_complex_grid_sum(
                    np.asarray(local_grid, dtype=np.complex128)
                )
            except Exception:
                _native_sampler_enabled = False
        return local_grid


class FastGradientGrid:
    """Rank-local gradient accumulator with an MPI reduction at finalize().

    Args:
        sim: Active adjoint Meep simulation.
        component: Meep field component sampled from the adjoint fields.
        coords_x: x coordinates of the design grid.
        coords_y: y coordinates of the design grid.
    """

    def __init__(
        self,
        sim: mp.Simulation,
        component: int,
        coords_x: Sequence[float],
        coords_y: Sequence[float],
    ) -> None:
        self.field = FastFieldGrid(sim, component, coords_x, coords_y)
        self.local = np.zeros(self.field.shape, dtype=complex)
        self.use_local = FastFieldGrid.supports_local_accumulation() and not self.field.has_symmetry
        self.needs_reduce = False

    def accumulate(self, multiplier: NDArray[np.complex128]) -> None:
        """Add one time-slice contribution to the local gradient accumulator.

        Args:
            multiplier: Grid-shaped forward factor multiplied by the current
                adjoint field sample.
        """
        if self.use_local:
            if not self.field.accumulate_product(multiplier, self.local):
                raise RuntimeError("native local gradient accumulation failed")
            self.needs_reduce = True
        else:
            self.local[:] += self.field.product(multiplier)

    def finalize(self) -> NDArray[np.complex128]:
        """Return the accumulated complex gradient grid.

        Returns:
            MPI-reduced grid when native rank-local accumulation was used,
            otherwise the local Python accumulator.
        """
        if self.needs_reduce:
            return FastFieldGrid.reduce(self.local)
        return self.local
