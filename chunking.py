from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import meep as mp
import numpy as np


_AXIS_NAMES = ("x", "y", "z")


@dataclass(frozen=True)
class SourceBoundaryPolicy:
    mode: str = "auto"
    protected_gap_cells: float = 2.0
    min_split_spacing_cells: float = 8.0
    finite_source_width_cells: float = 2.0
    finite_source_axes: Optional[Tuple[str, ...]] = None
    prefer_finite_above_resolution: Optional[float] = None
    margin_cells: float = 1.0
    neighbor_gap_cells: float = 0.25


@dataclass(frozen=True)
class SourceBoundaryDecision:
    method: str
    reason: str
    chunk_layout: Optional[mp.BinaryPartition]
    source_size: mp.Vector3
    source_amplitude: complex
    changed_axes: Tuple[str, ...] = ()
    dimensions: Optional[int] = None
    resolution: Optional[float] = None
    finite_source_width_cells: Optional[float] = None
    finite_source_axes: Tuple[str, ...] = ()
    boundary_axes: Tuple[str, ...] = ()

    def regularized_source(
        self,
        source_size: mp.Vector3,
        source_amplitude: complex = 1.0,
    ) -> Tuple[mp.Vector3, complex, Tuple[str, ...]]:
        """Apply this decision's finite-source fallback to another source."""
        if self.method != "finite":
            return source_size, source_amplitude, ()
        if self.dimensions is None or self.resolution is None or self.finite_source_width_cells is None:
            raise ValueError("finite-source decision is missing regularization metadata")
        return regularize_source_size_and_amplitude(
            source_size,
            source_amplitude,
            dimensions=self.dimensions,
            resolution=self.resolution,
            width_cells=self.finite_source_width_cells,
            axes=self.finite_source_axes,
        )


def _axis_index(axis) -> int:
    if axis == mp.X:
        return 0
    if axis == mp.Y:
        return 1
    if axis == mp.Z:
        return 2
    raise ValueError("split_axis must be one of mp.X, mp.Y, or mp.Z")


def _axis_coord(v: mp.Vector3, axis) -> float:
    return (float(v.x), float(v.y), float(v.z))[_axis_index(axis)]


def _axis_name(axis) -> str:
    if isinstance(axis, str):
        axis = axis.lower()
        if axis in _AXIS_NAMES:
            return axis
    else:
        if axis == 0 or axis == mp.X:
            return "x"
        if axis == 1 or axis == mp.Y:
            return "y"
        if axis == 2 or axis == mp.Z:
            return "z"
    raise ValueError("source axes must be x/y/z, 0/1/2, or mp.X/mp.Y/mp.Z")


def _normalize_axis_names(axes: Sequence, dimensions: int) -> Tuple[str, ...]:
    axis_names = []
    for axis in axes:
        axis_name = _axis_name(axis)
        axis_index = _AXIS_NAMES.index(axis_name)
        if axis_index >= dimensions:
            raise ValueError("source axis is outside the simulated dimensions")
        if axis_name not in axis_names:
            axis_names.append(axis_name)
    return tuple(axis_names)


def _source_axes_for_boundary(
    policy: SourceBoundaryPolicy,
    split_axes: Sequence,
    dimensions: int,
) -> Tuple[str, ...]:
    if policy.finite_source_axes is not None:
        return _normalize_axis_names(policy.finite_source_axes, dimensions)
    return _normalize_axis_names(split_axes, dimensions)


def _unique_split_axes(split_axis, split_axes: Optional[Sequence]) -> Tuple:
    axes = [split_axis] if split_axes is None else list(split_axes)
    if _axis_name(split_axis) not in {_axis_name(axis) for axis in axes}:
        axes.insert(0, split_axis)
    unique_axes = []
    seen = set()
    for axis in axes:
        axis_name = _axis_name(axis)
        if axis_name not in seen:
            unique_axes.append(axis)
            seen.add(axis_name)
    return tuple(unique_axes)


def _split_spacing_cells(
    *,
    cell_size: mp.Vector3,
    resolution: float,
    num_proc: int,
    split_axis,
    margin_cells: float,
) -> float:
    length = _axis_coord(cell_size, split_axis)
    usable_length = length - 2 * margin_cells / resolution
    return resolution * usable_length / num_proc


def _vector_from_coords(coords) -> mp.Vector3:
    return mp.Vector3(float(coords[0]), float(coords[1]), float(coords[2]))


def _build_binary_partition(axis, splits: Sequence[float], ranks: Sequence[int]):
    if len(ranks) == 1:
        return ranks[0]
    mid = len(ranks) // 2
    return [
        (axis, float(splits[mid - 1])),
        _build_binary_partition(axis, splits[: mid - 1], ranks[:mid]),
        _build_binary_partition(axis, splits[mid:], ranks[mid:]),
    ]


def safe_chunk_layout(
    *,
    cell_size: mp.Vector3,
    geometry_center: mp.Vector3,
    resolution: float,
    protected_points: Sequence[mp.Vector3],
    num_proc: Optional[int] = None,
    split_axis=mp.Y,
    protected_gap_cells: float = 2.0,
    margin_cells: float = 1.0,
    neighbor_gap_cells: float = 0.25,
) -> Optional[mp.BinaryPartition]:
    """Create a 1D MPI chunk layout avoiding protected source/monitor points."""
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if num_proc is None:
        num_proc = mp.count_processors()
    if num_proc <= 1:
        return None

    length = _axis_coord(cell_size, split_axis)
    center = _axis_coord(geometry_center, split_axis)
    margin = margin_cells / resolution
    low = center - 0.5 * length + margin
    high = center + 0.5 * length - margin
    if not low < high:
        raise ValueError("cell size is too small for the requested chunk-layout margin")

    protected = np.array([_axis_coord(point, split_axis) for point in protected_points], dtype=float)
    protected_gap = protected_gap_cells / resolution
    neighbor_gap = neighbor_gap_cells / resolution
    ideal_splits = low + np.arange(1, num_proc) * (high - low) / num_proc
    split_spacing = (high - low) / num_proc
    search_step = 0.5 / resolution
    max_offset = max(0.5 * split_spacing - neighbor_gap, protected_gap + search_step)
    offsets = [0.0]
    for step_index in range(1, int(np.ceil(max_offset / search_step)) + 1):
        offset = step_index * search_step
        offsets.extend([-offset, offset])

    splits = []
    for split_index, target in enumerate(ideal_splits):
        left_bound = low if split_index == 0 else splits[-1] + neighbor_gap
        right_bound = high if split_index == ideal_splits.size - 1 else ideal_splits[split_index + 1] - neighbor_gap
        candidates = [
            target + offset
            for offset in offsets
            if left_bound < target + offset < right_bound
        ]
        candidates.sort(key=lambda value: abs(value - target))
        for candidate in candidates:
            if protected.size == 0 or np.all(np.abs(candidate - protected) >= protected_gap):
                splits.append(candidate)
                break
        else:
            raise ValueError(
                f"could not place a safe MPI chunk boundary near {target:.6g} "
                f"for {num_proc} ranks"
            )

    return mp.BinaryPartition(data=_build_binary_partition(split_axis, splits, list(range(num_proc))))


def regularize_source_size_and_amplitude(
    source_size: mp.Vector3,
    source_amplitude: complex,
    *,
    dimensions: int,
    resolution: float,
    width_cells: float = 2.0,
    axes: Optional[Sequence] = None,
) -> Tuple[mp.Vector3, complex, Tuple[str, ...]]:
    """Replace zero-size simulated axes by a finite width and renormalize amplitude."""
    if dimensions not in (1, 2, 3):
        raise ValueError("dimensions must be 1, 2, or 3")
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if width_cells <= 0:
        raise ValueError("width_cells must be positive")

    width = width_cells / resolution
    coords = [float(source_size.x), float(source_size.y), float(source_size.z)]
    target_axes = (
        tuple(_AXIS_NAMES[:dimensions])
        if axes is None
        else _normalize_axis_names(axes, dimensions)
    )
    scale = 1.0
    changed = []
    for axis_name in target_axes:
        axis_index = _AXIS_NAMES.index(axis_name)
        if coords[axis_index] == 0.0:
            coords[axis_index] = width
            scale *= width
            changed.append(axis_name)

    if scale != 1.0:
        source_amplitude = source_amplitude / scale
    return _vector_from_coords(coords), source_amplitude, tuple(changed)


def resolve_source_boundary_workaround(
    *,
    cell_size: mp.Vector3,
    geometry_center: mp.Vector3,
    resolution: float,
    protected_points: Sequence[mp.Vector3],
    source_size: mp.Vector3,
    source_amplitude: complex = 1.0,
    dimensions: int = 2,
    num_proc: Optional[int] = None,
    split_axis=mp.Y,
    split_axes: Optional[Sequence] = None,
    policy: Optional[SourceBoundaryPolicy] = None,
) -> SourceBoundaryDecision:
    """Choose safe chunk layout when feasible, otherwise regularize source size."""
    if policy is None:
        policy = SourceBoundaryPolicy()
    if policy.mode not in ("auto", "layout", "finite"):
        raise ValueError("policy.mode must be 'auto', 'layout', or 'finite'")
    if num_proc is None:
        num_proc = mp.count_processors()
    checked_split_axes = _unique_split_axes(split_axis, split_axes)
    boundary_axes = _normalize_axis_names(checked_split_axes, dimensions)
    finite_source_axes = _source_axes_for_boundary(policy, checked_split_axes, dimensions)

    force_finite = policy.mode == "finite"
    if policy.mode == "auto" and policy.prefer_finite_above_resolution is not None:
        force_finite = resolution >= policy.prefer_finite_above_resolution

    if force_finite:
        new_size, new_amplitude, changed_axes = regularize_source_size_and_amplitude(
            source_size,
            source_amplitude,
            dimensions=dimensions,
            resolution=resolution,
            width_cells=policy.finite_source_width_cells,
            axes=finite_source_axes,
        )
        return SourceBoundaryDecision(
            method="finite",
            reason="finite source selected by policy",
            chunk_layout=None,
            source_size=new_size,
            source_amplitude=new_amplitude,
            changed_axes=changed_axes,
            dimensions=dimensions,
            resolution=resolution,
            finite_source_width_cells=policy.finite_source_width_cells,
            finite_source_axes=finite_source_axes,
            boundary_axes=boundary_axes,
        )

    if num_proc <= 1:
        return SourceBoundaryDecision(
            method="serial",
            reason="single process run does not need a custom chunk layout",
            chunk_layout=None,
            source_size=source_size,
            source_amplitude=source_amplitude,
            dimensions=dimensions,
            resolution=resolution,
            finite_source_width_cells=policy.finite_source_width_cells,
            finite_source_axes=finite_source_axes,
            boundary_axes=boundary_axes,
        )

    checked_layouts = {}
    checked_spacings = {}
    risky_split_axes = []
    for axis in checked_split_axes:
        spacing = _split_spacing_cells(
            cell_size=cell_size,
            resolution=resolution,
            num_proc=num_proc,
            split_axis=axis,
            margin_cells=policy.margin_cells,
        )
        checked_spacings[_axis_name(axis)] = spacing
        if spacing < policy.min_split_spacing_cells:
            risky_split_axes.append(axis)
            continue
        try:
            checked_layouts[_axis_name(axis)] = safe_chunk_layout(
                cell_size=cell_size,
                geometry_center=geometry_center,
                resolution=resolution,
                protected_points=protected_points,
                num_proc=num_proc,
                split_axis=axis,
                protected_gap_cells=policy.protected_gap_cells,
                margin_cells=policy.margin_cells,
                neighbor_gap_cells=policy.neighbor_gap_cells,
            )
        except ValueError:
            risky_split_axes.append(axis)

    if risky_split_axes:
        if policy.mode == "layout":
            risky_names = ", ".join(_axis_name(axis) for axis in risky_split_axes)
            raise ValueError(f"could not place safe MPI chunk boundaries for axes: {risky_names}")
        boundary_axes = _normalize_axis_names(risky_split_axes, dimensions)
        finite_source_axes = _source_axes_for_boundary(policy, risky_split_axes, dimensions)
        new_size, new_amplitude, changed_axes = regularize_source_size_and_amplitude(
            source_size,
            source_amplitude,
            dimensions=dimensions,
            resolution=resolution,
            width_cells=policy.finite_source_width_cells,
            axes=finite_source_axes,
        )
        return SourceBoundaryDecision(
            method="finite",
            reason=f"safe chunk layout skipped; risky boundary axes={boundary_axes}",
            chunk_layout=None,
            source_size=new_size,
            source_amplitude=new_amplitude,
            changed_axes=changed_axes,
            dimensions=dimensions,
            resolution=resolution,
            finite_source_width_cells=policy.finite_source_width_cells,
            finite_source_axes=finite_source_axes,
            boundary_axes=boundary_axes,
        )

    split_spacing_cells = checked_spacings[_axis_name(split_axis)]
    should_try_layout = policy.mode == "layout" or split_spacing_cells >= policy.min_split_spacing_cells

    if should_try_layout:
        return SourceBoundaryDecision(
            method="layout",
            reason=f"safe chunk layout selected; split spacing is {split_spacing_cells:.3g} cells",
            chunk_layout=checked_layouts[_axis_name(split_axis)],
            source_size=source_size,
            source_amplitude=source_amplitude,
            dimensions=dimensions,
            resolution=resolution,
            finite_source_width_cells=policy.finite_source_width_cells,
            finite_source_axes=finite_source_axes,
            boundary_axes=boundary_axes,
        )

    new_size, new_amplitude, changed_axes = regularize_source_size_and_amplitude(
        source_size,
        source_amplitude,
        dimensions=dimensions,
        resolution=resolution,
        width_cells=policy.finite_source_width_cells,
        axes=finite_source_axes,
    )
    return SourceBoundaryDecision(
        method="finite",
        reason=f"safe chunk layout skipped; split spacing is {split_spacing_cells:.3g} cells",
        chunk_layout=None,
        source_size=new_size,
        source_amplitude=new_amplitude,
        changed_axes=changed_axes,
        dimensions=dimensions,
        resolution=resolution,
        finite_source_width_cells=policy.finite_source_width_cells,
        finite_source_axes=finite_source_axes,
        boundary_axes=boundary_axes,
    )


def adjoint_source_boundary_workaround(
    *,
    cell_size: mp.Vector3,
    geometry_center: mp.Vector3,
    resolution: float,
    source_position: mp.Vector3,
    monitor_positions: Sequence[mp.Vector3],
    dimensions: int = 2,
    source_size: Optional[mp.Vector3] = None,
    source_amplitude: complex = 1.0,
    num_proc: Optional[int] = None,
    layout_axis=mp.Y,
    check_axes: Optional[Sequence] = None,
    policy: Optional[SourceBoundaryPolicy] = None,
) -> SourceBoundaryDecision:
    """Return chunk layout and adjoint-source size/amplitude for point monitors."""
    if source_size is None:
        source_size = mp.Vector3()
    if check_axes is None:
        check_axes = (mp.X, mp.Y) if dimensions >= 2 else (mp.X,)
    if policy is None:
        policy = SourceBoundaryPolicy(
            mode="auto",
            protected_gap_cells=2.0,
            min_split_spacing_cells=8.0,
            finite_source_width_cells=2.0,
            prefer_finite_above_resolution=80,
        )
    return resolve_source_boundary_workaround(
        cell_size=cell_size,
        geometry_center=geometry_center,
        resolution=resolution,
        protected_points=[source_position] + list(monitor_positions),
        source_size=source_size,
        source_amplitude=source_amplitude,
        dimensions=dimensions,
        num_proc=num_proc,
        split_axis=layout_axis,
        split_axes=check_axes,
        policy=policy,
    )
