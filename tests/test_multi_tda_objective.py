import meep as mp
import numpy as np

import tide_adj as tp


def _make_minimal_multi_tda(**kwargs):
    params = dict(
        update_design=lambda _: None,
        sim_factory=lambda sources=None: None,
        coords_x=[0.0],
        coords_y=[0.0],
        t_final=1.0,
        monitor_positions=[mp.Vector3()],
        component=mp.Ez,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=9,
        pixel_chunk=1,
        adjoint_source_size=mp.Vector3(),
        adjoint_source_amplitude=1.0,
        material_factor=3.0,
        cell_area=1.0,
        dt=0.05,
    )
    params.update(kwargs)
    return tp.MultiTDAObjective(**params)


def test_multi_tda_objective_accepts_numpy_kernel_window_options():
    rectangular = _make_minimal_multi_tda(kernel_window="rectangular")
    hamming = _make_minimal_multi_tda(kernel_window="hamming")
    kaiser_4 = _make_minimal_multi_tda(
        kernel_window="kaiser",
        kernel_window_params={"beta": 4.0},
    )
    kaiser_8 = _make_minimal_multi_tda(
        kernel_window="kaiser",
        kernel_window_params={"beta": 8.0},
    )

    assert not np.allclose(rectangular.kernels[0], hamming.kernels[0])
    assert not np.allclose(kaiser_4.kernels[0], kaiser_8.kernels[0])


def test_multi_tda_objective_exports_and_filters_signals():
    obj = tp.MultiTDAObjective(
        update_design=lambda _: None,
        sim_factory=lambda sources=None: None,
        coords_x=[0.0],
        coords_y=[0.0],
        t_final=1.0,
        monitor_positions=[mp.Vector3()],
        component=mp.Ez,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=9,
        pixel_chunk=1,
        adjoint_source_size=mp.Vector3(),
        adjoint_source_amplitude=1.0,
        background=mp.Medium(epsilon=1.0),
        design_material=mp.Medium(epsilon=4.0),
        cell_area=1.0,
        dt=0.05,
    )

    signal = np.zeros((32, 1), dtype=np.complex128)
    signal[16, 0] = 1.0
    filtered = obj.filter_monitor_signals(signal)

    assert filtered.shape == signal.shape
    assert np.iscomplexobj(filtered)
    assert obj.last_total_fom is None


class _FakeSimulation:
    def __init__(self, dt, samples):
        self.dt = dt
        self.samples = samples
        self.index = 0

    def run(self, callback, until):
        for index in range(self.samples):
            self.index = index
            callback(self)

    def get_field_point(self, component, position):
        return np.exp(1j * 0.1 * self.index)

    def round_time(self):
        return self.dt * (self.samples - 1)

    def reset_meep(self):
        pass


def test_multi_tda_objective_value_only_path_sets_band_state():
    dt = 0.05

    def scalarization_fn(band_objectives):
        return 2.0 * np.sum(band_objectives), 2.0 * np.ones_like(band_objectives)

    obj = tp.MultiTDAObjective(
        update_design=lambda _: None,
        sim_factory=lambda sources=None: _FakeSimulation(dt, 48),
        coords_x=[0.0],
        coords_y=[0.0],
        t_final=1.0,
        monitor_positions=[mp.Vector3()],
        component=mp.Ez,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=9,
        pixel_chunk=1,
        adjoint_source_size=mp.Vector3(),
        adjoint_source_amplitude=1.0,
        material_factor=3.0,
        cell_area=1.0,
        dt=dt,
        scalarization_fn=scalarization_fn,
    )

    value, gradient = obj.evaluate(np.array([0.0]), need_gradient=False)

    assert gradient is None
    assert np.isfinite(value)
    assert obj.last_band_objectives.shape == (1,)
    assert obj.last_band_losses.shape == (1,)
    assert obj.last_band_coeffs.shape == (1,)
    assert obj.last_band_coeffs[0] == 2.0
    assert obj.last_total_fom == 2.0 * obj.last_band_objectives[0]
    assert obj.last_total_fom is not None


def test_multi_tda_objective_auto_pixel_chunk():
    assert tp.auto_pixel_chunk(
        20,
        nproc=1,
        target_chunks_per_rank=10,
        min_pixel_chunk=3,
        max_pixel_chunk=7,
    ) == 3

    obj = tp.MultiTDAObjective(
        update_design=lambda _: None,
        sim_factory=lambda sources=None: None,
        coords_x=list(range(10)),
        coords_y=list(range(2)),
        t_final=1.0,
        monitor_positions=[mp.Vector3()],
        component=mp.Ez,
        wavelength_bands=[(0.4, 0.5)],
        weights=[1.0],
        kernel_length=9,
        pixel_chunk="auto",
        target_chunks_per_rank=10,
        min_pixel_chunk=3,
        max_pixel_chunk=7,
        adjoint_source_size=mp.Vector3(),
        adjoint_source_amplitude=1.0,
        material_factor=3.0,
        cell_area=1.0,
        dt=0.05,
    )

    assert obj.pixel_chunk == 3


def test_multi_tda_objective_accepts_bundled_design_simulation_and_targets():
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=4.0)
    design = tp.DesignGrid(
        material_grid=mp.MaterialGrid(mp.Vector3(2, 2), air, dielectric),
        center=mp.Vector3(),
        size=mp.Vector3(2.0, 2.0),
        shape=(2, 2),
        background=air,
        design_material=dielectric,
    )
    simulation = tp.SimulationSpec(
        cell_size=mp.Vector3(1.0, 1.0),
        boundary_layers=[],
        geometry=[],
        sources=[],
        resolution=10,
    )
    targets = [
        tp.PointTarget(position=mp.Vector3(0.0, 0.1), component=mp.Ez),
        tp.PointTarget(position=mp.Vector3(0.0, 0.2), component=mp.Ez),
    ]

    obj = tp.MultiTDAObjective(
        design=design,
        simulation=simulation,
        targets=targets,
        t_final=1.0,
        wavelength_bands=[(0.4, 0.5), (0.5, 0.6)],
        weights=[1.0, 1.0],
        kernel_length=9,
        pixel_chunk=1,
        dt=0.05,
    )

    assert obj.coords_x == design.coords_x
    assert obj.coords_y == design.coords_y
    assert obj.monitor_positions == [target.position for target in targets]
    assert obj.component == mp.Ez
    assert obj.cell_area == design.cell_area
    assert obj.material_factor == design.material_factor


if __name__ == "__main__":
    test_multi_tda_objective_accepts_numpy_kernel_window_options()
    test_multi_tda_objective_exports_and_filters_signals()
    test_multi_tda_objective_value_only_path_sets_band_state()
    test_multi_tda_objective_auto_pixel_chunk()
    test_multi_tda_objective_accepts_bundled_design_simulation_and_targets()
