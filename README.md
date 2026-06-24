# TIDE-Adj

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
conda create -n meep-latest -c conda-forge \
  python=3.11 \
  "pymeep=*=mpi_mpich_*" \
  mpi4py mpich cxx-compiler \
  nlopt scipy matplotlib autograd numpy
conda activate meep-latest
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
def update_design(x):
    design_variables.update_weights(np.asarray(x))


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


tda = tp.TDAObjective(
    update_design=update_design,
    coords_x=coords_x,
    coords_y=coords_y,
    t_final=T_f,
    sim_factory=make_sim,
    monitor_position=monitor_position,
    component=mp.Ez,
    cell_area=dx * dy,
    adjoint_source_size=adjoint_source_size,
    adjoint_source_amplitude=adjoint_source_amplitude,
    background=air,
    design_material=design_material,
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
