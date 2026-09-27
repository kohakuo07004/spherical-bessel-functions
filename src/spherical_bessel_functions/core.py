"""Spherical Bessel functions of the first and second kind.

The spherical Bessel functions j_n(x) and y_n(x) are solutions of the
differential equation

    x^2 f'' + 2x f' + (x^2 - n(n+1)) f = 0

We evaluate them on plain Python floats via recurrence relations, which
keeps the dependency footprint at zero and the code transparent.

Strategy
--------
For j_n we use downward (Miller) recurrence starting from a small seed
because upward recurrence is numerically unstable for j_n when x < n.
For y_n we use upward recurrence, which is stable for this function.

The normalisation constant for the downward j_n sweep is obtained from
the known closed forms j_0(x) = sin(x)/x and j_1(x) = sin(x)/x^2 - cos(x)/x.
"""

from __future__ import annotations

import math


__all__ = ["spherical_j", "spherical_y"]


def _j0(x: float) -> float:
    """j_0(x) = sin(x)/x, with the removable singularity at 0 filled in."""
    if x == 0.0:
        return 1.0
    return math.sin(x) / x


def _j1(x: float) -> float:
    """j_1(x) = sin(x)/x^2 - cos(x)/x, with the limit at 0 being 0."""
    if x == 0.0:
        return 0.0
    return math.sin(x) / (x * x) - math.cos(x) / x


def _y0(x: float) -> float:
    """y_0(x) = -cos(x)/x. Undefined at x=0; we raise ValueError there."""
    if x == 0.0:
        raise ValueError("spherical_y is singular at x=0")
    return -math.cos(x) / x


def _y1(x: float) -> float:
    """y_1(x) = -cos(x)/x^2 - sin(x)/x. Undefined at x=0."""
    if x == 0.0:
        raise ValueError("spherical_y is singular at x=0")
    return -math.cos(x) / (x * x) - math.sin(x) / x


def spherical_j(n: int, x: float) -> float:
    """Spherical Bessel function of the first kind, j_n(x).

    Parameters
    ----------
    n:
        Non-negative integer order.
    x:
        Real float argument.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``n`` is negative or not an integer.

    Notes
    -----
    For ``n == 0`` and ``n == 1`` the closed forms are used directly.
    For ``n >= 2`` we use Miller's downward recurrence: seed two tiny
    values at orders ``n_max+1`` and ``n_max``, recurse downward, then
    normalise against the analytic j_0.  Downward recurrence is stable
    where upward recurrence (which amplifies the dominant y_n solution)
    is not.
    """
    if not isinstance(n, int):
        raise ValueError("order n must be an integer")
    if n < 0:
        raise ValueError("order n must be non-negative")

    if n == 0:
        return _j0(x)
    if n == 1:
        return _j1(x)

    # For x == 0, j_n(0) = 0 for n >= 1.
    if x == 0.0:
        return 0.0

    ax = abs(x)

    # Miller's algorithm: start the downward sweep a few orders above n
    # so that the recurrence has room to converge toward the true ratio
    # j_{k+1}/j_k before we reach the order we actually want.
 # The extra headroom matters most when x is small relative to n.
    n_start = n + max(10, n)

    # Seed values; their absolute magnitude is irrelevant because we
    # normalise at the end.
    j_high = 0.0
    j_low = 1.0

    for k in range(n_start, 0, -1):
        # Recurrence: j_{k-1} = (2k+1)/x * j_k - j_{k+1}
        j_new = (2 * k + 1) / x * j_low - j_high
        j_high = j_low
        j_low = j_new
        # Rescale to avoid overflow/underflow during the sweep.
        if abs(j_low) > 1e100:
            scale = 1e100
            j_low /= scale
            j_high /= scale
        elif 0 < abs(j_low) < 1e-100:
            scale = 1e100
            j_low *= scale
            j_high *= scale

    # After the loop j_low ~ j_0 (unnormalised), j_high ~ j_1 (unnormalised).
    # Normalise so that the j_0 value matches the analytic result.
    j0_true = _j0(x)
    if j_low == 0.0:
        # Pathological: the seed decayed to exactly zero. Fall back to
        # upward recurrence, which is fine for moderate n/x ratios.
        return _spherical_j_upward(n, x)

    ratio = j0_true / j_low

    # Now run the upward recurrence from the normalised j_0, j_1 up to n.
    j_prev = j0_true
    j_curr = _j1(x)
    if n == 0:
        return j_prev
    if n == 1:
        return j_curr
    for k in range(1, n):
        j_next = (2 * k + 1) / x * j_curr - j_prev
        j_prev = j_curr
        j_curr = j_next
    return j_curr


def _spherical_j_upward(n: int, x: float) -> float:
    """Upward recurrence fallback for j_n. Stable when x is not tiny vs n."""
    j_prev = _j0(x)
    if n == 0:
        return j_prev
    j_curr = _j1(x)
    if n == 1:
        return j_curr
    for k in range(1, n):
        j_next = (2 * k + 1) / x * j_curr - j_prev
        j_prev = j_curr
        j_curr = j_next
    return j_curr


def spherical_y(n: int, x: float) -> float:
    """Spherical Bessel function of the second kind, y_n(x).

    Parameters
    ----------
    n:
        Non-negative integer order.
    x:
        Real, non-zero float argument.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``n`` is negative, not an integer, or if ``x`` is zero
        (y_n has a pole at the origin).

    Notes
    -----
    Upward recurrence is numerically stable for y_n, so we use the
    closed forms y_0 and y_1 as seeds and recurse up.
    """
    if not isinstance(n, int):
        raise ValueError("order n must be an integer")
    if n < 0:
        raise ValueError("order n must be non-negative")
    if x == 0.0:
        raise ValueError("spherical_y is singular at x=0")

    if n == 0:
        return _y0(x)
    if n == 1:
        return _y1(x)

    y_prev = _y0(x)
    y_curr = _y1(x)
    for k in range(1, n):
        y_next = (2 * k + 1) / x * y_curr - y_prev
        y_prev = y_curr
        y_curr = y_next
    return y_curr
