# TIDE-Adj: Time-Domain Inverse-Design Extensions for Adjoint Optimization

[![Release](https://img.shields.io/github/v/release/Jaehyeon-Park/tide-adj?include_prereleases&label=release)](https://github.com/Jaehyeon-Park/tide-adj/releases)
[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-pre--release-orange.svg)](https://github.com/Jaehyeon-Park/tide-adj/releases/tag/v0.5.0-alpha.1)

TIDE-Adj (`tide-adj` as a project name, `tide_adj` as a Python import) is a
small helper layer for time-domain adjoint optimization with Meep. It does not
hide Meep objects. Users still create Meep sources, geometry, materials, and
simulations directly, while TIDE-Adj provides field sampling, point-monitor
objectives, and MPI source-boundary workarounds.

## Status

This is a pre-release research package. APIs, examples, and native-sampler
build details may change before the first stable release.

Current pre-release: `v0.5.0-alpha.1`  
Python package version: `0.5.0a1`

## License

TIDE-Adj is licensed under the GNU General Public License version 2.0 only
(`GPL-2.0-only`).

TIDE-Adj depends on Meep, which is distributed under the GNU General Public
License version 2.0. TIDE-Adj is an independent project and is not affiliated
with or endorsed by the Meep project.

## Files

```text
tide_adj/
├── __init__.py
├── chunking.py
├── coords.py
├── fastmeep_grid.py  # compatibility shim
├── sampling_grid.py
├── native_sampler.cpp
├── native_sampler.py  # compatibility wrapper
├── native_sampler.pyi
├── specs.py
├── multi_tda_objective.py
├── objectives.py
├── tda_objective.py
└── build_native_sampler.sh
```

GitHub source releases contain the Python sources and native-sampler source
code. Build the native sampler after cloning the repository or unpacking a
source release.

## Environment

Use a conda environment with MPI-enabled Meep and the Python dependencies used
by the example scripts.

```bash
conda create -n <env-name> -c conda-forge \
  python=3.11 \
  "pymeep=*=mpi_mpich_*" \
  mpi4py mpich cxx-compiler \
  nlopt scipy matplotlib autograd numpy
conda activate <env-name>
```

For an existing environment:

```bash
conda activate <env-name>
conda install -c conda-forge \
  "pymeep=*=mpi_mpich_*" \
  mpi4py mpich cxx-compiler \
  nlopt scipy matplotlib autograd numpy
```

Check the environment:

```bash
python -c "import meep as mp; print(mp.__version__)"
which python
which mpic++
```

## Native Sampler

TIDE-Adj can use a C++ extension for faster field sampling. Build it in the
target Python/Meep environment.

```bash
cd <tide-adj-source-directory>
chmod +x build_native_sampler.sh
MEEP_CONDA_PREFIX="$CONDA_PREFIX" ./build_native_sampler.sh
```

If TIDE-Adj is used directly from source, verify from the directory that
contains the `tide_adj/` package directory:

```bash
cd <directory-containing-tide_adj>
python -c "import tide_adj; print(tide_adj.__version__); print(tide_adj.native_sampler_available())"
python -c "import tide_adj.native_sampler as ns; print(ns.__file__)"
```

`native_sampler_available()` should print `True`.
If `ns.__file__` ends with `native_sampler.py`, TIDE-Adj is using a
compatibility wrapper around a pre-rename binary. Rebuild the native sampler
when the matching compiler is available.

## Basic Imports

Run scripts from the directory containing `tide_adj/`, or add that directory to
`PYTHONPATH`.

```python
import meep as mp
import numpy as np
import tide_adj as tp
```

## Design Grid Coordinates

Use `centered_grid_coords` to make sampling coordinates aligned with a Meep
design region.

```python
coords_x, coords_y = tp.centered_grid_coords(
    center=design_center,
    shape=(nx, ny),
    spacing=(dx, dy),
)
```

These coordinates can be passed to `TDAObjective` or to `FastFieldGrid` /
`FastGradientGrid` directly.

For less repetitive objective setup, use the bundled input objects:

| Object | Bundles |
| --- | --- |
| `DesignGrid` | `MaterialGrid`, design-region center/size/shape, sampling coordinates, cell area, and `d epsilon / d rho`. |
| `SimulationSpec` | Common `mp.Simulation` constructor inputs such as cell size, PML layers, geometry, default sources, resolution, `geometry_center`, and `chunk_layout`. |
| `PointTarget` | Point-monitor position, field component, and matching adjoint-source size/amplitude. |

`SimulationSpec.make()` behaves like the examples' usual `make_sim(sources=None)`
factory: no argument creates the forward simulation, and an explicit source list
creates the adjoint simulation.

## MPI Source-Boundary Workaround

Meep ordinary `mp.Source` deposition can depend on MPI chunk boundaries for
zero-size point sources and sources with a zero-size axis. TIDE-Adj provides a
helper for point-monitor adjoint sources:

```python
source_boundary_decision = tp.adjoint_source_boundary_workaround(
    cell_size=cell_size,
    geometry_center=geometry_center,
    resolution=resolution,
    source_position=source_center,
    monitor_positions=monitor_positions,
    dimensions=2,
)

chunk_layout = source_boundary_decision.chunk_layout
adjoint_source_size = source_boundary_decision.source_size
adjoint_source_amplitude = source_boundary_decision.source_amplitude
```

Default behavior:

```text
mode = auto
protected_gap_cells = 2
min_split_spacing_cells = 8
finite_source_width_cells = 2
prefer_finite_above_resolution = 80
layout_axis = mp.Y
check_axes = (mp.X, mp.Y) in 2D
base adjoint source = point source, size=(0, 0), amplitude=1
```

The helper first tries to place a safe MPI chunk layout with boundaries outside
the protected gap around the source and monitor positions. If the layout is not
safe, or if the resolution is at least 80, it falls back to a finite source.

Fallback rules for a point adjoint source in 2D:

```text
X boundary risk      -> size=(2dx, 0),      amplitude=1/(2dx)
Y boundary risk      -> size=(0, 2dx),      amplitude=1/(2dx)
X and Y boundary risk -> size=(2dx, 2dx),   amplitude=1/(4dx^2)
```

Use the returned `chunk_layout` in every forward and adjoint `mp.Simulation`
created for the same problem.

```python
def make_sim(sources=None):
    return mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=fwd_sources if sources is None else sources,
        resolution=resolution,
        geometry_center=geometry_center,
        chunk_layout=chunk_layout,
    )
```

For lower-level control, use `SourceBoundaryPolicy` and
`resolve_source_boundary_workaround` directly.

## TDAObjective Example

Minimal shape of a TIDE-Adj time-domain adjoint setup:

```python
design = tp.DesignGrid(
    material_grid=design_variables,
    center=design_center,
    size=design_region_size,
    shape=(nx, ny),
    background=air,
    design_material=design_material,
)

simulation = tp.SimulationSpec(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=fwd_sources,
    resolution=resolution,
    geometry_center=geometry_center,
    chunk_layout=chunk_layout,
)

target = tp.PointTarget(
    position=monitor_position,
    component=mp.Ez,
    adjoint_source_size=adjoint_source_size,
    adjoint_source_amplitude=adjoint_source_amplitude,
)


tda = tp.TDAObjective(
    design=design,
    simulation=simulation,
    target=target,
    t_final=T_f,
    resolution=resolution,
)

fom, gradient = tda.fom_and_grad(x)
```

`TDAObjective` default objective is:

```text
0.5 * integral |E_monitor(t)|^2 dt
```

For a custom scalar objective, pass `fom_fn(monitor_history, sample_dt)`. Use
autograd-compatible operations unless you also provide `adjoint_signal_fn`.

### TDAObjective Inputs

`TDAObjective` follows the same style as Meep objects: users pass ordinary Meep
objects and explicit physical coordinates rather than hidden configuration.

| Input | Meaning |
| --- | --- |
| `design` | Optional `DesignGrid`. Fills `update_design`, `coords_x`, `coords_y`, `cell_area`, and `material_factor`. |
| `simulation` | Optional `SimulationSpec`. Fills `sim_factory` and `resolution`. |
| `target` | Optional `PointTarget`. Fills `monitor_position`, `component`, `adjoint_source_size`, and `adjoint_source_amplitude`. |
| `update_design` | Function that writes the flat design vector into the Meep design object. Usually calls `MaterialGrid.update_weights(...)`. |
| `coords_x`, `coords_y` | Physical sampling coordinates over the design region. Use `tp.centered_grid_coords(...)` when the design grid is rectangular. |
| `t_final` | Forward simulation end time. |
| `sim_factory` | Function returning `mp.Simulation`. Called with no argument for the forward run and with an adjoint source list for the adjoint run. |
| `monitor_position` | Physical point where the forward monitor signal is sampled and where the adjoint source is placed. |
| `component` | Meep field component, e.g. `mp.Ez`. Current examples are 2D TMz/Ez-oriented. |
| `cell_area` | Area represented by one 2D design variable. Used to scale the final gradient. |
| `adjoint_source_size` | Meep source size for the adjoint source. Use `mp.Vector3()` for a point source or the source-boundary workaround result. |
| `adjoint_source_amplitude` | Scalar amplitude multiplier for the adjoint source. Use the source-boundary workaround result when finite-source fallback is selected. |
| `background`, `design_material` | Materials used to infer `d epsilon / d rho`. |
| `material_factor` | Explicit `d epsilon / d rho`. If supplied, `background` and `design_material` are not required. |
| `fom_fn` | Optional scalar objective `fom_fn(monitor_history, sample_dt)`. |
| `adjoint_signal_fn` | Optional manual provider for `dFoM/dE(t)`. Use this when `fom_fn` is not autograd-compatible. |
| `dt` | Explicit sampling time step. |
| `resolution` | Used to infer `dt = 0.5 / resolution` when `dt` is omitted. |
| `sampling_interval` | Number of Meep time steps between design-grid field samples. |

## MultiTDAObjective Example

`MultiTDAObjective` is a multi-band temporal-convolution objective. It
evaluates several wavelength bands from one broadband time-domain run and
combines the resulting per-band FoMs through a user-defined scalarization
function.

```python
def scalarization_fn(band_objectives):
    total_fom = np.sum(band_objectives)
    band_coeffs = np.ones_like(band_objectives)
    return total_fom, band_coeffs


multi_tda = tp.MultiTDAObjective(
    design=design,
    simulation=simulation,
    targets=[
        tp.PointTarget(
            position=pos,
            component=mp.Ez,
            adjoint_source_size=adjoint_source_size,
            adjoint_source_amplitude=adjoint_source_amplitude,
        )
        for pos in monitor_positions
    ],
    t_final=T_f,
    wavelength_bands=[(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8)],
    weights=band_weights,
    kernel_length=2001,
    kernel_window="hamming",
    dt=dt,
    scalarization_fn=scalarization_fn,
)
```

`scalarization_fn` defines how per-band FoMs become one scalar FoM. It must
return the scalar FoM and the derivative of that scalar with respect to each
band FoM:

```python
total_fom, band_coeffs = scalarization_fn(band_objectives)
```

For optional diagnostics, return a third item:

```python
return total_fom, band_coeffs, {"name": value}
```

`MultiTDAObjective` returns `(total_fom, d total_fom / d rho)` with the
maximization sign, matching `TDAObjective` and Meep adjoint's
`OptimizationProblem`. Negate both in the optimizer callback when driving a
minimizer such as nlopt or `scipy.optimize.minimize`. `band_coeffs` are used to
scale each band adjoint source and gradient kernel.

### MultiTDAObjective Inputs

| Input | Meaning |
| --- | --- |
| `design` | Optional `DesignGrid`. Fills design update, sampling coordinates, cell area, and material factor. |
| `simulation` | Optional `SimulationSpec`. Fills the simulation factory and resolution. |
| `targets` | Optional list of `PointTarget` objects. Current implementation expects one target per wavelength band and one shared component/source size/source amplitude. |
| `update_design` | Same role as in `TDAObjective`: writes the design vector into the active Meep design object. |
| `sim_factory` | Same role as in `TDAObjective`: returns forward or adjoint `mp.Simulation` objects. |
| `coords_x`, `coords_y` | Physical design-grid sampling coordinates. These define where forward and adjoint fields are sampled for the design gradient. |
| `t_final` | Main forward simulation time. The class internally runs until `t_final + kernel_length * dt` so the temporal filter has enough tail. |
| `monitor_positions` | Point-monitor locations. Current implementation expects one monitor position per wavelength band. |
| `component` | Meep field component used for monitor sampling, adjoint source injection, and design-grid field sampling. |
| `wavelength_bands` | List of `(lambda_min, lambda_max)` intervals. Each interval creates one bandpass temporal-convolution kernel. |
| `weights` | Per-band amplitude weights. These are applied to monitor filtering and the gradient kernel. |
| `kernel_length` | Number of time samples in the temporal-convolution bandpass kernel. Larger values give narrower filtering but increase runtime and temporary storage. |
| `kernel_window` | NumPy window used to taper each bandpass kernel. Supported values are `"rectangular"`/`None`, `"hamming"`, `"hann"`, `"blackman"`, `"bartlett"`, and `"kaiser"`. Default is `"hamming"`. |
| `kernel_window_params` | Optional window parameters. Currently only `"kaiser"` uses this, with `{"beta": value}`. |
| `pixel_chunk` | Design-pixel block size for temporal-convolution gradient evaluation. Defaults to `"auto"` and usually should be omitted. |
| `target_chunks_per_rank` | Target number of pixel chunks per MPI rank when `pixel_chunk="auto"`. Default is `8`. |
| `min_pixel_chunk`, `max_pixel_chunk` | Lower and upper bounds for automatic `pixel_chunk`. Defaults are `16` and `128`. |
| `adjoint_source_size` | Meep source size for each adjoint monitor source. Usually the value returned by `adjoint_source_boundary_workaround`. |
| `adjoint_source_amplitude` | Amplitude multiplier for each adjoint source. Usually the value returned by `adjoint_source_boundary_workaround`. |
| `background`, `design_material` | Materials used to infer `d epsilon / d rho`. |
| `material_factor` | Explicit `d epsilon / d rho`. If supplied, `background` and `design_material` are not required. |
| `cell_area` | Area represented by one 2D design variable. |
| `dt` | Explicit time step used for monitor integration, temporal convolution, and gradient scaling. |
| `resolution` | Used to infer `dt = 0.5 / resolution` when `dt` is omitted. |
| `scalarization_fn` | User-defined scalarization function. It receives `band_objectives` and returns `(total_fom, band_coeffs)` or `(total_fom, band_coeffs, info)`. |
| `history_dtype` | Complex dtype for temporary forward/adjoint field-history memmaps. Default is `np.complex128`. |

## Running

Until package installation metadata is added, run from a location where
`tide_adj` is importable. For a source checkout, this is the parent directory of
the `tide_adj/` package directory. Alternatively, set `PYTHONPATH` to that
parent directory.

Single process:

```bash
cd <directory-containing-tide_adj>
python <path-to-your-script.py>
```

MPI:

```bash
cd <directory-containing-tide_adj>
mpirun -np 4 python <path-to-your-script.py>
```

From another working directory:

```bash
PYTHONPATH=<directory-containing-tide_adj> python <path-to-your-script.py>
PYTHONPATH=<directory-containing-tide_adj> mpirun -np 4 python <path-to-your-script.py>
```

Check what TIDE-Adj selected:

```text
TIDE-Adj source boundary policy: method=layout, ...
TIDE-Adj source boundary policy: method=finite, ...
```

## Troubleshooting

If native sampler is unavailable:

```bash
cd <tide-adj-source-directory>
MEEP_CONDA_PREFIX="$CONDA_PREFIX" ./build_native_sampler.sh
```

If `mpic++` is missing:

```bash
which mpic++
conda install -c conda-forge mpi4py mpich cxx-compiler
```

If `import tide_adj` fails, run from the parent directory of `tide_adj/`:

```bash
cd <directory-containing-tide_adj>
python -c "import tide_adj; print(tide_adj.__file__)"
```

If MPI execution differs from serial, first inspect the printed source-boundary
decision and verify that the same `chunk_layout` is used for forward and adjoint
simulations.
