from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


def create_component_grid_plan(
    fields_addr: int,
    coords_x: Sequence[float],
    coords_y: Sequence[float],
    component: int,
) -> Any:
    """Precompute native sampling metadata for one field component grid.

    Args:
        fields_addr: Integer address of Meep's low-level ``fields`` object.
        coords_x: x coordinates of the 2D sampling grid.
        coords_y: y coordinates of the 2D sampling grid.
        component: Meep field component integer, e.g. ``int(mp.Ez)``.

    Returns:
        Opaque native plan reused by plan-based sampling functions.
    """
    ...


def sample_component_grid_plan_allreduced(
    plan: Any,
) -> NDArray[np.complex128]:
    """Sample a component grid using a precomputed native plan.

    Args:
        plan: Opaque object returned by ``create_component_grid_plan``.

    Returns:
        Complex sampled field grid after MPI all-reduction.
    """
    ...


def sample_component_grid(
    fields_addr: int,
    coords_x: Sequence[float],
    coords_y: Sequence[float],
    component: int,
) -> NDArray[np.complex128]:
    """Sample a Meep field component on a 2D coordinate grid.

    Args:
        fields_addr: Integer address of Meep's low-level ``fields`` object.
        coords_x: x coordinates of the 2D sampling grid.
        coords_y: y coordinates of the 2D sampling grid.
        component: Meep field component integer.

    Returns:
        Complex sampled field grid.
    """
    ...


def sample_component_grid_allreduced(
    fields_addr: int,
    coords_x: Sequence[float],
    coords_y: Sequence[float],
    component: int,
) -> NDArray[np.complex128]:
    """Sample a component grid and MPI-all-reduce rank-local contributions.

    Args:
        fields_addr: Integer address of Meep's low-level ``fields`` object.
        coords_x: x coordinates of the 2D sampling grid.
        coords_y: y coordinates of the 2D sampling grid.
        component: Meep field component integer.

    Returns:
        Complex sampled field grid with MPI rank contributions combined.
    """
    ...


def accumulate_component_product_allreduced(
    fields_addr: int,
    coords_x: Sequence[float],
    coords_y: Sequence[float],
    component: int,
    multiplier: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Return MPI-all-reduced ``sampled_field * multiplier``.

    Args:
        fields_addr: Integer address of Meep's low-level ``fields`` object.
        coords_x: x coordinates of the 2D sampling grid.
        coords_y: y coordinates of the 2D sampling grid.
        component: Meep field component integer.
        multiplier: Complex grid multiplied pointwise with the sampled field.

    Returns:
        Complex product grid after MPI all-reduction.
    """
    ...


def accumulate_component_product_local_inplace(
    fields_addr: int,
    coords_x: Sequence[float],
    coords_y: Sequence[float],
    component: int,
    multiplier: NDArray[np.complex128],
    accumulator: NDArray[np.complex128],
) -> None:
    """Accumulate ``sampled_field * multiplier`` into ``accumulator`` in place.

    Args:
        fields_addr: Integer address of Meep's low-level ``fields`` object.
        coords_x: x coordinates of the 2D sampling grid.
        coords_y: y coordinates of the 2D sampling grid.
        component: Meep field component integer.
        multiplier: Complex grid multiplied pointwise with the sampled field.
        accumulator: Complex grid updated in place on the local rank.
    """
    ...


def accumulate_component_product_plan_local_inplace(
    plan: Any,
    multiplier: NDArray[np.complex128],
    accumulator: NDArray[np.complex128],
) -> None:
    """Plan-based in-place accumulation of ``sampled_field * multiplier``.

    Args:
        plan: Opaque object returned by ``create_component_grid_plan``.
        multiplier: Complex grid multiplied pointwise with the sampled field.
        accumulator: Complex grid updated in place on the local rank.
    """
    ...


def reduce_complex_grid_sum(
    local_grid: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """MPI-sum a complex grid over all ranks.

    Args:
        local_grid: Rank-local complex grid.

    Returns:
        Complex grid after MPI summation.
    """
    ...


def sample_ez_grid(
    fields_addr: int,
    coords_x: Sequence[float],
    coords_y: Sequence[float],
    component: int = ...,
) -> NDArray[np.complex128]:
    """Backward-compatible Ez grid sampler.

    Args:
        fields_addr: Integer address of Meep's low-level ``fields`` object.
        coords_x: x coordinates of the 2D sampling grid.
        coords_y: y coordinates of the 2D sampling grid.
        component: Meep field component integer. Defaults to Ez in the native
            module.

    Returns:
        Complex sampled Ez grid.
    """
    ...
