import numpy as np
import meep as mp

import tide_adj as tp


def test_design_grid_bundles_material_grid_coordinates_and_updates():
    air = mp.Medium(epsilon=1.0)
    dielectric = mp.Medium(epsilon=4.0)
    material_grid = mp.MaterialGrid(mp.Vector3(2, 3), air, dielectric)

    design = tp.DesignGrid(
        material_grid=material_grid,
        center=mp.Vector3(0.5, -0.5),
        size=mp.Vector3(2.0, 3.0),
        shape=(2, 3),
        background=air,
        design_material=dielectric,
    )

    design.update_weights(np.linspace(0.0, 1.0, 6))

    assert len(design.coords_x) == 2
    assert len(design.coords_y) == 3
    assert design.cell_area == 1.0
    assert design.material_factor == 3.0


def test_simulation_spec_uses_default_or_override_sources():
    default_sources = [
        mp.Source(
            mp.GaussianSource(frequency=1.0, fwidth=0.1),
            component=mp.Ez,
            center=mp.Vector3(),
        )
    ]
    override_sources = [
        mp.Source(
            mp.GaussianSource(frequency=1.2, fwidth=0.1),
            component=mp.Ez,
            center=mp.Vector3(),
        )
    ]
    spec = tp.SimulationSpec(
        cell_size=mp.Vector3(1.0, 1.0),
        boundary_layers=[],
        geometry=[],
        sources=lambda: default_sources,
        resolution=10,
    )

    default_sim = spec.make()
    override_sim = spec.make(override_sources)

    assert default_sim.sources is default_sources
    assert override_sim.sources is override_sources


def test_tda_objective_accepts_bundled_design_simulation_and_target():
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
    target = tp.PointTarget(
        position=mp.Vector3(0.0, 0.25),
        component=mp.Ez,
        adjoint_source_size=mp.Vector3(),
        adjoint_source_amplitude=2.0,
    )

    obj = tp.TDAObjective(
        design=design,
        simulation=simulation,
        target=target,
        t_final=1.0,
        dt=0.05,
    )

    assert obj.coords_x == design.coords_x
    assert obj.coords_y == design.coords_y
    assert obj.objective.monitor_position == target.position
    assert obj.objective.component == mp.Ez
    assert obj.objective.cell_area == design.cell_area
    assert obj.objective.adjoint_source_amplitude == 2.0


if __name__ == "__main__":
    test_design_grid_bundles_material_grid_coordinates_and_updates()
    test_simulation_spec_uses_default_or_override_sources()
    test_tda_objective_accepts_bundled_design_simulation_and_target()
