import gc
import os
import tempfile
from typing import Callable, Optional, Sequence

import meep as mp
import numpy as np
import scipy.interpolate as spi
from scipy.ndimage import convolve1d

from .objectives import _epsilon_from_medium
from .sampling_grid import FastFieldGrid
from .specs import DesignGrid, PointTarget, SimulationSpec


def bandpass_kernel(f_low: float, f_high: float, dt: float, length: int) -> np.ndarray:
    """Return a windowed ideal bandpass kernel normalized at band center."""
    n = np.arange(length)
    tau = (n - (length - 1) / 2) * dt
    kernel = (
        2 * f_high * np.sinc(2 * f_high * tau)
        - 2 * f_low * np.sinc(2 * f_low * tau)
    )
    kernel *= np.hamming(kernel.size)

    f_center = 0.5 * (f_low + f_high)
    response = np.sum(kernel * np.exp(-2j * np.pi * f_center * tau)) * dt
    gain = np.abs(response)
    if gain > 0:
        kernel /= gain
    return kernel


def temporal_convolve_signal(signal: np.ndarray, kernel: np.ndarray, dt: float) -> np.ndarray:
    return convolve1d(signal, kernel, axis=0, mode="constant", cval=0.0) * dt


def auto_pixel_chunk(
    n_pixels: int,
    *,
    nproc: Optional[int] = None,
    target_chunks_per_rank: int = 8,
    min_pixel_chunk: int = 16,
    max_pixel_chunk: int = 128,
) -> int:
    if n_pixels <= 0:
        raise ValueError("n_pixels must be positive")
    if nproc is None:
        nproc = mp.count_processors()
    if nproc <= 0:
        raise ValueError("nproc must be positive")
    if target_chunks_per_rank <= 0:
        raise ValueError("target_chunks_per_rank must be positive")
    if min_pixel_chunk <= 0 or max_pixel_chunk <= 0:
        raise ValueError("pixel chunk limits must be positive")
    if min_pixel_chunk > max_pixel_chunk:
        raise ValueError("min_pixel_chunk must be <= max_pixel_chunk")

    chunk = int(np.ceil(n_pixels / (nproc * target_chunks_per_rank)))
    return int(np.clip(chunk, min_pixel_chunk, max_pixel_chunk))


class MultiTDAObjective:
    """Multi-band time-domain adjoint objective using temporal convolution.

    The class evaluates several wavelength bands from one broadband time-domain
    run by filtering monitor histories with bandpass kernels. The resulting
    band objectives are combined by a user-provided scalarization function.

    Sign convention: ``evaluate`` / ``fom_and_grad`` return
    ``(total_fom, d total_fom / d rho)`` with the maximization sign, matching
    ``TDAObjective`` and Meep adjoint's ``OptimizationProblem``. Negate both in
    the optimizer callback when driving a minimizer such as nlopt or
    ``scipy.optimize.minimize``.
    """

    def __init__(
        self,
        update_design: Optional[Callable[[np.ndarray], None]] = None,
        sim_factory: Optional[Callable[..., mp.Simulation]] = None,
        coords_x: Optional[Sequence[float]] = None,
        coords_y: Optional[Sequence[float]] = None,
        t_final: Optional[float] = None,
        monitor_positions: Optional[Sequence[mp.Vector3]] = None,
        component: Optional[int] = None,
        wavelength_bands: Optional[Sequence[tuple[float, float]]] = None,
        weights: Optional[Sequence[float]] = None,
        kernel_length: Optional[int] = None,
        pixel_chunk="auto",
        target_chunks_per_rank: int = 8,
        min_pixel_chunk: int = 16,
        max_pixel_chunk: int = 128,
        adjoint_source_size: Optional[mp.Vector3] = None,
        adjoint_source_amplitude: Optional[float] = None,
        background: Optional[mp.Medium] = None,
        design_material: Optional[mp.Medium] = None,
        material_factor: Optional[float] = None,
        cell_area: Optional[float] = None,
        dt: Optional[float] = None,
        resolution: Optional[float] = None,
        scalarization_fn: Optional[Callable[[np.ndarray], tuple]] = None,
        history_dtype=np.complex128,
        design: Optional[DesignGrid] = None,
        simulation: Optional[SimulationSpec] = None,
        targets: Optional[Sequence[PointTarget]] = None,
    ) -> None:
        """Create a multi-band temporal-convolution adjoint problem.

        Args:
            update_design: Function that writes the flat design vector ``x`` into
                the Meep design object, usually a ``MaterialGrid``.
            sim_factory: Function returning a Meep ``Simulation``. It is called
                with no arguments for the forward run and with a source list for
                the adjoint run.
            coords_x: Physical x coordinates where the forward and adjoint fields
                are sampled over the design region.
            coords_y: Physical y coordinates where the forward and adjoint fields
                are sampled over the design region.
            t_final: Main forward run time before the extra filter tail is added.
            monitor_positions: Point-monitor positions. The current API expects
                one monitor position per wavelength band.
            component: Meep field component to monitor, inject, and sample, e.g.
                ``mp.Ez``.
            wavelength_bands: Wavelength intervals ``(lambda_min, lambda_max)``.
                Each interval defines one bandpass temporal-convolution kernel.
            weights: Per-band amplitude weights applied to monitor filtering and
                gradient kernels.
            kernel_length: Number of time samples in each bandpass kernel.
            pixel_chunk: Number of design pixels processed at once during
                temporal-convolution gradient evaluation, or ``"auto"``.
            target_chunks_per_rank: Target number of pixel chunks assigned to each
                MPI rank when ``pixel_chunk="auto"``.
            min_pixel_chunk: Lower bound for automatic pixel chunk size.
            max_pixel_chunk: Upper bound for automatic pixel chunk size.
            adjoint_source_size: Meep source size for each adjoint monitor source.
            adjoint_source_amplitude: Scalar amplitude multiplier for each
                adjoint monitor source.
            background: Background medium used to infer ``d epsilon / d rho``.
            design_material: Design medium used to infer ``d epsilon / d rho``.
            material_factor: Explicit ``d epsilon / d rho``. If supplied,
                ``background`` and ``design_material`` are not required.
            cell_area: Area represented by one 2D design variable.
            dt: Explicit time step used for sampling and convolution.
            resolution: Meep resolution used to infer ``dt=0.5/resolution`` when
                ``dt`` is omitted.
            scalarization_fn: Function that maps band FoMs to a scalar FoM and
                band coefficients. It should return ``(total_fom, band_coeffs)``
                or ``(total_fom, band_coeffs, info)``. ``band_coeffs`` must be
                ``d total_fom / d band_objectives``.
            history_dtype: Complex dtype used for temporary field-history memmaps.
            design: Optional ``DesignGrid`` bundle. When supplied, it fills
                design-grid sampling, update, cell area, and material factor
                inputs unless those are explicitly supplied.
            simulation: Optional ``SimulationSpec`` bundle. When supplied, it
                fills ``sim_factory`` and ``resolution`` unless those are
                explicitly supplied.
            targets: Optional sequence of ``PointTarget`` bundles. The current
                implementation expects one target per wavelength band and one
                shared field component/source size/source amplitude.
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
        if targets is not None:
            targets = list(targets)
            if not targets:
                raise ValueError("targets must not be empty")
            components = {target.component for target in targets}
            if len(components) != 1:
                raise ValueError("targets must use the same component")
            size_keys = {
                None if target.adjoint_source_size is None else (
                    target.adjoint_source_size.x,
                    target.adjoint_source_size.y,
                    target.adjoint_source_size.z,
                )
                for target in targets
            }
            if len(size_keys) != 1:
                raise ValueError("targets must use the same adjoint_source_size")
            amplitudes = {float(target.adjoint_source_amplitude) for target in targets}
            if len(amplitudes) != 1:
                raise ValueError("targets must use the same adjoint_source_amplitude")

            monitor_positions = monitor_positions if monitor_positions is not None else [
                target.position for target in targets
            ]
            component = component if component is not None else targets[0].component
            adjoint_source_size = (
                adjoint_source_size
                if adjoint_source_size is not None
                else targets[0].adjoint_source_size
            )
            adjoint_source_amplitude = (
                adjoint_source_amplitude
                if adjoint_source_amplitude is not None
                else targets[0].adjoint_source_amplitude
            )
        if cell_area is None:
            cell_area = 1.0
        if adjoint_source_amplitude is None:
            adjoint_source_amplitude = 1.0

        missing = [
            name for name, value in (
                ("update_design", update_design),
                ("sim_factory", sim_factory),
                ("coords_x", coords_x),
                ("coords_y", coords_y),
                ("t_final", t_final),
                ("monitor_positions", monitor_positions),
                ("component", component),
                ("wavelength_bands", wavelength_bands),
                ("weights", weights),
                ("kernel_length", kernel_length),
            )
            if value is None
        ]
        if missing:
            raise ValueError("MultiTDAObjective missing required inputs: " + ", ".join(missing))

        self.update_design = update_design
        self.sim_factory = sim_factory
        self.coords_x = list(coords_x)
        self.coords_y = list(coords_y)
        self.t_final = float(t_final)
        self.dt = self._resolve_dt(dt, resolution)
        self.filter_time = kernel_length * self.dt
        self.run_time = self.t_final + self.filter_time
        self.monitor_positions = list(monitor_positions)
        self.component = component
        self.weights = np.asarray(weights, dtype=float)
        self.pixel_chunk = self._resolve_pixel_chunk(
            pixel_chunk,
            target_chunks_per_rank=target_chunks_per_rank,
            min_pixel_chunk=min_pixel_chunk,
            max_pixel_chunk=max_pixel_chunk,
        )
        self.adjoint_source_size = adjoint_source_size if adjoint_source_size is not None else mp.Vector3()
        self.adjoint_source_amplitude = float(adjoint_source_amplitude)
        self.cell_area = float(cell_area)
        self.scalarization_fn = scalarization_fn if scalarization_fn is not None else self._weighted_sum_scalarization
        self.history_dtype = history_dtype

        if material_factor is None:
            if background is None or design_material is None:
                raise ValueError("MultiTDAObjective requires material_factor or both background and design_material")
            material_factor = _epsilon_from_medium(design_material) - _epsilon_from_medium(background)
        self.material_factor = float(material_factor)

        self.kernels = []
        for lam_min, lam_max in wavelength_bands:
            f_low = 1 / lam_max
            f_high = 1 / lam_min
            self.kernels.append(bandpass_kernel(f_low, f_high, self.dt, kernel_length))

        if len(self.monitor_positions) != len(self.kernels):
            raise ValueError("monitor_positions must match the number of wavelength bands")
        if self.weights.size != len(self.kernels):
            raise ValueError("weights must match the number of wavelength bands")

        self.weighted_kernels = [
            weight * kernel for weight, kernel in zip(self.weights, self.kernels)
        ]
        self.last_band_objectives = None
        self.last_band_losses = None
        self.last_band_coeffs = None
        self.last_scalarization_info = None
        self.last_smooth_min = None
        self.last_total_fom = None

    @staticmethod
    def _resolve_dt(dt: Optional[float], resolution: Optional[float]) -> float:
        if dt is not None:
            return float(dt)
        if resolution is not None:
            return 0.5 / float(resolution)
        raise ValueError("MultiTDAObjective requires dt or resolution")

    def _resolve_pixel_chunk(
        self,
        pixel_chunk,
        *,
        target_chunks_per_rank: int,
        min_pixel_chunk: int,
        max_pixel_chunk: int,
    ) -> int:
        if isinstance(pixel_chunk, str):
            if pixel_chunk != "auto":
                raise ValueError('pixel_chunk must be a positive integer or "auto"')
            return auto_pixel_chunk(
                len(self.coords_x) * len(self.coords_y),
                target_chunks_per_rank=target_chunks_per_rank,
                min_pixel_chunk=min_pixel_chunk,
                max_pixel_chunk=max_pixel_chunk,
            )
        pixel_chunk = int(pixel_chunk)
        if pixel_chunk <= 0:
            raise ValueError("pixel_chunk must be positive")
        return pixel_chunk

    @staticmethod
    def _weighted_sum_scalarization(band_objectives: np.ndarray):
        return float(np.sum(band_objectives)), np.ones_like(band_objectives)

    @staticmethod
    def _parse_scalarization_result(result, band_objectives: np.ndarray):
        if not isinstance(result, tuple) or len(result) not in (2, 3):
            raise ValueError("scalarization_fn must return (total_fom, band_coeffs) or (total_fom, band_coeffs, info)")
        total_fom, band_coeffs = result[:2]
        info = result[2] if len(result) == 3 else None
        band_coeffs = np.asarray(band_coeffs, dtype=float)
        if band_coeffs.shape != band_objectives.shape:
            raise ValueError("scalarization_fn band_coeffs must match band_objectives shape")
        return float(total_fom), band_coeffs, info

    def _make_history_memmap(self, shape):
        tmp = tempfile.NamedTemporaryFile(prefix="tide_adj_history_", suffix=".dat", delete=False)
        path = tmp.name
        tmp.close()
        return np.memmap(path, dtype=self.history_dtype, mode="w+", shape=shape), path

    @staticmethod
    def _cleanup_history_memmaps(*history_memmaps):
        for history, path in history_memmaps:
            if history is not None:
                history.flush()
            if path is not None:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass

    def __call__(self, x: np.ndarray, need_gradient: bool = True):
        return self.evaluate(x, need_gradient=need_gradient)

    def fom(self, x: np.ndarray) -> float:
        value, _ = self.evaluate(x, need_gradient=False)
        return value

    def fom_and_grad(self, x: np.ndarray):
        return self.evaluate(x, need_gradient=True)

    def evaluate(self, x: np.ndarray, need_gradient: bool = True):
        self.update_design(x)

        sim_fwd = self.sim_factory()
        monitor_history = []
        n_expected = int(np.ceil(self.run_time / self.dt)) + 8
        grid_shape = (len(self.coords_x), len(self.coords_y))
        fwd_history = None
        fwd_history_path = None
        if need_gradient:
            fwd_history, fwd_history_path = self._make_history_memmap((n_expected, *grid_shape))
        fwd_grid = {"obj": None}
        fwd_count = {"value": 0}

        def record_fwd(sim):
            monitor_history.append([
                sim.get_field_point(self.component, monitor_position)
                for monitor_position in self.monitor_positions
            ])
            if need_gradient:
                if fwd_grid["obj"] is None:
                    fwd_grid["obj"] = FastFieldGrid(sim, self.component, self.coords_x, self.coords_y)
                if fwd_count["value"] >= fwd_history.shape[0]:
                    raise RuntimeError("forward field history buffer is too small")
                fwd_history[fwd_count["value"]] = fwd_grid["obj"].sample().astype(self.history_dtype, copy=False)
                fwd_count["value"] += 1

        sim_fwd.run(record_fwd, until=self.run_time)
        actual_time = sim_fwd.round_time()
        monitor_history = np.asarray(monitor_history, dtype=self.history_dtype)
        if need_gradient:
            fwd_history.flush()
            fwd_history = fwd_history[:fwd_count["value"]]
        sim_fwd.reset_meep()
        del sim_fwd
        gc.collect()

        filtered_monitors = self.filter_monitor_signals(monitor_history)
        band_objectives = 0.5 * np.sum(np.abs(filtered_monitors) ** 2, axis=0) * self.dt
        band_losses = -band_objectives
        total_fom, band_coeffs, scalarization_info = self._parse_scalarization_result(
            self.scalarization_fn(band_objectives),
            band_objectives,
        )
        self.last_band_objectives = band_objectives
        self.last_band_losses = band_losses
        self.last_band_coeffs = band_coeffs
        self.last_scalarization_info = scalarization_info
        self.last_smooth_min = (
            scalarization_info.get("smooth_min")
            if isinstance(scalarization_info, dict)
            else None
        )
        self.last_total_fom = total_fom
        if not need_gradient:
            return total_fom, None

        adj_signals = filtered_monitors * band_coeffs[None, :]
        adj_signals = adj_signals[::-1]
        t_array = np.linspace(0, actual_time, adj_signals.shape[0])

        def source_func(interp):
            return lambda t: complex(interp(t))

        adj_interps = [
            spi.interp1d(t_array, adj_signals[:, band_index], kind="cubic", fill_value=0j, bounds_error=False)
            for band_index in range(adj_signals.shape[1])
        ]

        adjoint_source = [
            mp.Source(
                mp.CustomSource(src_func=source_func(adj_interp)),
                component=self.component,
                center=monitor_position,
                size=self.adjoint_source_size,
                amplitude=self.adjoint_source_amplitude,
            )
            for monitor_position, adj_interp in zip(self.monitor_positions, adj_interps)
        ]

        sim_adj = self.sim_factory(adjoint_source)
        adj_history, adj_history_path = self._make_history_memmap((fwd_history.shape[0], *grid_shape))
        adj_grid = {"obj": None}
        adj_count = {"value": 0}

        def record_adj(sim):
            if adj_grid["obj"] is None:
                adj_grid["obj"] = FastFieldGrid(sim, self.component, self.coords_x, self.coords_y)
            if adj_count["value"] < adj_history.shape[0]:
                adj_history[adj_count["value"]] = adj_grid["obj"].sample().astype(self.history_dtype, copy=False)
            adj_count["value"] += 1

        sim_adj.run(record_adj, until=actual_time)
        adj_history.flush()
        sim_adj.reset_meep()
        del sim_adj, adjoint_source, adj_interps, t_array
        gc.collect()

        adj_history = adj_history[:min(adj_count["value"], adj_history.shape[0])]
        n_common = min(fwd_history.shape[0], adj_history.shape[0])
        fwd_history = fwd_history[:n_common]
        adj_history = adj_history[:n_common]
        gradient_kernel = np.sum(
            [coeff * kernel for coeff, kernel in zip(band_coeffs, self.weighted_kernels)],
            axis=0,
        )
        gradient = self.band_gradient(fwd_history, adj_history, gradient_kernel)

        self._cleanup_history_memmaps((fwd_history, fwd_history_path), (adj_history, adj_history_path))
        del fwd_history, adj_history, monitor_history, filtered_monitors
        gc.collect()
        return total_fom, gradient

    def filter_monitor_signals(self, signals: np.ndarray) -> np.ndarray:
        filtered = []
        for band_index, kernel in enumerate(self.kernels):
            filtered.append(
                temporal_convolve_signal(
                    signals[:, band_index],
                    self.weights[band_index] * kernel,
                    self.dt,
                )
            )
        return np.column_stack(filtered)

    def band_gradient(self, fwd_history: np.ndarray, adj_history: np.ndarray, gradient_kernel: np.ndarray):
        n_time = fwd_history.shape[0]
        n_x = len(self.coords_x)
        n_y = len(self.coords_y)
        n_pixels = n_x * n_y
        rank = mp.my_rank()
        nproc = mp.count_processors()
        fwd_flat = fwd_history.reshape(n_time, n_pixels)
        adj_flat = adj_history.reshape(n_time, n_pixels)
        local_grad = np.zeros(n_pixels, dtype=np.complex128)
        for chunk_id, start in enumerate(range(0, n_pixels, self.pixel_chunk)):
            if chunk_id % nproc != rank:
                continue
            stop = min(start + self.pixel_chunk, n_pixels)
            fwd_filtered = convolve1d(
                fwd_flat[:, start:stop],
                gradient_kernel,
                axis=0,
                mode="constant",
                cval=0.0,
            ) * self.dt

            dE_dt = np.empty_like(fwd_filtered)
            dE_dt[0] = (fwd_filtered[1] - fwd_filtered[0]) / self.dt
            dE_dt[-1] = (fwd_filtered[-1] - fwd_filtered[-2]) / self.dt
            dE_dt[1:-1] = (fwd_filtered[2:] - fwd_filtered[:-2]) / (2 * self.dt)

            local_grad[start:stop] += np.sum(adj_flat[::-1, start:stop] * dE_dt, axis=0)
            del fwd_filtered, dE_dt
            gc.collect()

        grad_total = np.zeros_like(local_grad)
        mp.comm.Allreduce(local_grad, grad_total, op=mp.MPI.SUM)
        grad_total *= self.dt * self.cell_area * self.material_factor
        return grad_total.real.flatten()
