import gc
from typing import Callable, Optional, Sequence, Tuple

from autograd import grad
import autograd.numpy as npa
import meep as mp
import numpy as np

from .sampling_grid import FastFieldGrid, FastGradientGrid
from .objectives import _PointTarget
from .specs import DesignGrid, PointTarget, SimulationSpec


def _default_intensity_fom(monitor_history: np.ndarray, sample_dt: float):
    """Default point-signal objective: 0.5 * integral |E(t)|^2 dt."""
    return 0.5 * npa.sum(npa.abs(monitor_history) ** 2) * sample_dt


class TDAObjective:
    """Meep-style callable optimization problem for time-domain point objectives.

    Users provide Meep simulation construction through `sim_factory`, design
    updates through `update_design`, and point-target options describing where
    the forward monitor and adjoint source live. The scalar FoM can be
    customized with `fom_fn`; when no adjoint signal is supplied, autograd
    differentiates `fom_fn(E_t, dt)` with respect to the sampled monitor
    history.

    Sign convention: ``evaluate`` / ``fom_and_grad`` return
    ``(fom, d fom / d rho)`` with the maximization sign, matching
    ``MultiTDAObjective`` and Meep adjoint's ``OptimizationProblem``. Negate
    both in the optimizer callback when driving a minimizer such as nlopt or
    ``scipy.optimize.minimize``.
    """

    def __init__(
        self,
        update_design: Optional[Callable[[np.ndarray], None]] = None,
        coords_x: Optional[Sequence[float]] = None,
        coords_y: Optional[Sequence[float]] = None,
        t_final: Optional[float] = None,
        sim_factory: Optional[Callable[..., mp.Simulation]] = None,
        monitor_position: Optional[mp.Vector3] = None,
        component: Optional[int] = None,
        cell_area: Optional[float] = None,
        adjoint_source_size: Optional[mp.Vector3] = None,
        adjoint_source_amplitude: Optional[float] = None,
        background: Optional[mp.Medium] = None,
        design_material: Optional[mp.Medium] = None,
        material_factor: Optional[float] = None,
        fom_fn: Optional[Callable[[np.ndarray, float], float]] = None,
        adjoint_signal_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
        dt: Optional[float] = None,
        resolution: Optional[float] = None,
        sampling_interval: int = 1,
        design: Optional[DesignGrid] = None,
        simulation: Optional[SimulationSpec] = None,
        target: Optional[PointTarget] = None,
    ) -> None:
        """Create a callable time-domain adjoint optimization problem.

        Args:
            update_design: Function that writes the design vector ``x`` into
                the Meep geometry or material grid.
            coords_x: Physical x coordinates for design-grid field sampling.
            coords_y: Physical y coordinates for design-grid field sampling.
            t_final: Forward simulation end time.
            sim_factory: Function returning a Meep ``Simulation``. It is called
                with no arguments for the forward run and with a source list for
                the adjoint run.
            monitor_position: Physical point at which the forward field is
                sampled and the adjoint source is placed.
            component: Meep field component, e.g. ``mp.Ez``.
            cell_area: Area represented by one 2D design variable.
            adjoint_source_size: Meep source size for adjoint injection.
            adjoint_source_amplitude: Scalar multiplier for the adjoint source.
            background: Background medium used to infer permittivity contrast.
            design_material: Design material used to infer permittivity
                contrast.
            material_factor: Optional explicit ``d epsilon / d rho``. If this
                is supplied, ``background`` and ``design_material`` are not
                required.
            fom_fn: Optional scalar objective ``fom_fn(E_t, dt)``. It must use
                autograd-compatible operations if ``adjoint_signal_fn`` is not
                supplied.
            adjoint_signal_fn: Optional manual ``dFoM/dE(t)`` provider. Use
                this for objectives that autograd cannot differentiate.
            dt: Optional explicit Meep time step. If omitted, ``0.5 /
                resolution`` is used.
            resolution: Optional Meep resolution used to infer ``dt`` when the
                created simulation does not expose one.
            sampling_interval: Number of Meep time steps between design-grid
                field samples.
            design: Optional ``DesignGrid`` bundle. When supplied, it fills
                ``update_design``, ``coords_x``, ``coords_y``, ``cell_area``,
                and ``material_factor`` unless those are explicitly supplied.
            simulation: Optional ``SimulationSpec`` bundle. When supplied, it
                fills ``sim_factory`` and ``resolution`` unless those are
                explicitly supplied.
            target: Optional ``PointTarget`` bundle. When supplied, it fills
                ``monitor_position``, ``component``, ``adjoint_source_size``,
                and ``adjoint_source_amplitude`` unless those are explicitly
                supplied.
        """
        if design is not None:
            update_design = update_design if update_design is not None else design.update_weights
            coords_x = coords_x if coords_x is not None else design.coords_x
            coords_y = coords_y if coords_y is not None else design.coords_y
            cell_area = cell_area if cell_area is not None else design.cell_area
            material_factor = material_factor if material_factor is not None else design.material_factor
            background = background if background is not None else design.background
            design_material = design_material if design_material is not None else design.design_material
        if simulation is not None:
            sim_factory = sim_factory if sim_factory is not None else simulation.make
            resolution = resolution if resolution is not None else simulation.resolution
        if target is not None:
            monitor_position = monitor_position if monitor_position is not None else target.position
            component = component if component is not None else target.component
            adjoint_source_size = (
                adjoint_source_size
                if adjoint_source_size is not None
                else target.adjoint_source_size
            )
            adjoint_source_amplitude = (
                adjoint_source_amplitude
                if adjoint_source_amplitude is not None
                else target.adjoint_source_amplitude
            )
        if adjoint_source_amplitude is None:
            adjoint_source_amplitude = 1.0

        missing = [
            name for name, value in (
                ("update_design", update_design),
                ("coords_x", coords_x),
                ("coords_y", coords_y),
                ("t_final", t_final),
                ("sim_factory", sim_factory),
                ("monitor_position", monitor_position),
                ("component", component),
                ("cell_area", cell_area),
            )
            if value is None
        ]
        if missing:
            raise ValueError("TDAObjective missing required inputs: " + ", ".join(missing))

        self.update_design = update_design
        self.sim_factory = sim_factory
        self.coords_x = coords_x
        self.coords_y = coords_y
        self.t_final = float(t_final)
        self.dt = dt
        self.resolution = resolution
        self.sampling_interval = sampling_interval
        self.objective = _PointTarget(
            monitor_position=monitor_position,
            component=component,
            cell_area=cell_area,
            adjoint_source_size=adjoint_source_size,
            adjoint_source_amplitude=adjoint_source_amplitude,
            background=background,
            design_material=design_material,
            material_factor=material_factor,
        )
        self.fom_fn = fom_fn if fom_fn is not None else _default_intensity_fom
        self.adjoint_signal_fn = adjoint_signal_fn

    def time_step(self, sim: mp.Simulation) -> float:
        """Return the time step used by TIDE-Adj sampling.

        Args:
            sim: Forward Meep simulation created by ``sim_factory``.

        Returns:
            Explicit ``dt`` if supplied, otherwise Meep's default Courant time
            step ``0.5 / resolution``.
        """
        if self.dt is not None:
            return self.dt
        resolution = self.resolution if self.resolution is not None else getattr(sim, "resolution", None)
        if resolution is None:
            raise ValueError("TDAObjective requires dt, resolution, or a Simulation with a resolution attribute")
        return 0.5 / resolution

    def __call__(self, x: np.ndarray, need_gradient: bool = True):
        """Evaluate the objective and optionally its design gradient.

        Args:
            x: Flat design vector.
            need_gradient: If ``True``, run the adjoint simulation and return a
                gradient. If ``False``, run only the forward simulation.

        Returns:
            ``(fom, gradient)``. ``gradient`` is ``None`` when
            ``need_gradient=False``.
        """
        return self.evaluate(x, need_gradient=need_gradient)

    def fom(self, x: np.ndarray) -> float:
        """Evaluate only the scalar objective value.

        Args:
            x: Flat design vector passed to ``update_design``.

        Returns:
            Scalar FoM returned by the forward-only path.
        """
        value, _ = self.evaluate(x, need_gradient=False)
        return value

    def fom_and_grad(self, x: np.ndarray):
        """Evaluate both the scalar objective and the flat design gradient.

        Args:
            x: Flat design vector passed to ``update_design``.

        Returns:
            ``(objective_value, gradient)`` where ``gradient`` is a flat real
            array over the design variables.
        """
        return self.evaluate(x, need_gradient=True)

    def _fom_value_and_adjoint_signal(
        self,
        monitor_history: np.ndarray,
        sample_dt: float,
    ) -> Tuple[float, np.ndarray]:
        """Compute FoM and the continuous-time adjoint source signal.

        Args:
            monitor_history: Forward point-monitor samples ``E(t_n)``.
            sample_dt: Time step between adjacent monitor samples.

        Returns:
            Scalar FoM and sampled continuous-time adjoint signal
            ``dFoM/dE(t)``.
        """
        monitor_history = np.asarray(monitor_history)
        objective_value = self.fom_fn(monitor_history, sample_dt)

        if self.adjoint_signal_fn is not None:
            adjoint_signal = self.adjoint_signal_fn(monitor_history, sample_dt)
        else:
            # autograd differentiates the Riemann-sum objective with respect to
            # sampled values; divide by dt to recover dFoM/dE(t).
            d_fom_d_samples = grad(self.fom_fn, 0)(monitor_history, sample_dt)
            adjoint_signal = d_fom_d_samples / sample_dt

        return float(objective_value), np.asarray(adjoint_signal)

    def evaluate(self, x: np.ndarray, need_gradient: bool = True):
        """Run forward/adjoint Meep simulations for one design vector.

        Args:
            x: Flat design vector passed to ``update_design``.
            need_gradient: Whether to run the adjoint simulation.

        Returns:
            ``(objective_value, gradient)``. The gradient is a flat real array
            from the objective model, or ``None`` for value-only evaluation.
        """
        self.update_design(x)

        sim_fwd = self.sim_factory()
        dt = self.time_step(sim_fwd)
        e_mon_t = []
        e_fld_t = [] if need_gradient else None

        sample_count = {"count": 0}
        fwd_field = {"obj": None}

        def record_fwd(s):
            e_mon_t.append(self.objective.record_monitor(s))
            if need_gradient and sample_count["count"] % self.sampling_interval == 0:
                if fwd_field["obj"] is None:
                    fwd_field["obj"] = FastFieldGrid(
                        s,
                        self.objective.component,
                        self.coords_x,
                        self.coords_y,
                    )
                e_fld_t.append(fwd_field["obj"].sample())
            sample_count["count"] += 1

        if need_gradient:
            sim_fwd.run(record_fwd, until=self.t_final)
        else:
            sim_fwd.run(record_fwd, until=self.t_final)

        actual_time = sim_fwd.round_time()
        monitor_history = np.array(e_mon_t)
        e_mon_t.clear()
        sim_fwd.reset_meep()
        del sim_fwd
        gc.collect()

        objective_value = None
        adjoint_signal = None
        if need_gradient:
            objective_value, adjoint_signal = self._fom_value_and_adjoint_signal(monitor_history, dt)
        else:
            objective_value = float(self.fom_fn(monitor_history, dt))

        gradient = None
        if need_gradient:
            field_history = np.array(e_fld_t)
            e_fld_t.clear()
            del e_fld_t
            dt_eff = dt * self.sampling_interval

            adjoint_sources = self.objective.adjoint_sources(adjoint_signal, actual_time)
            del monitor_history
            gc.collect()

            sim_adj = self.sim_factory(adjoint_sources)
            adj_status = {"step": 0, "sample": 0}
            adj_grad = None

            def accum_adj(s):
                nonlocal adj_grad
                if adj_status["step"] % self.sampling_interval == 0:
                    idx = len(field_history) - 1 - adj_status["sample"]
                    if idx >= 0:
                        if idx == 0:
                            dE_dt_t = (field_history[1] - field_history[0]) / dt_eff
                        elif idx == len(field_history) - 1:
                            dE_dt_t = (field_history[-1] - field_history[-2]) / dt_eff
                        else:
                            dE_dt_t = (field_history[idx + 1] - field_history[idx - 1]) / (2 * dt_eff)

                        if adj_grad is None:
                            adj_grad = FastGradientGrid(s, self.objective.component, self.coords_x, self.coords_y)
                        adj_grad.accumulate(dE_dt_t)
                    adj_status["sample"] += 1
                adj_status["step"] += 1

            sim_adj.run(accum_adj, until=actual_time)
            grad_grid = adj_grad.finalize()
            sim_adj.reset_meep()
            del sim_adj, field_history, adjoint_sources
            gc.collect()

            gradient = self.objective.gradient(grad_grid, dt_eff)
            del grad_grid
            gc.collect()
        else:
            del monitor_history
            gc.collect()

        return objective_value, gradient
