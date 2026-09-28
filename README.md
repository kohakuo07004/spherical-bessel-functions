# spherical_bessel_functions

Evaluates spherical Bessel functions j_n(x) and y_n(x) of integer order on plain Python floats, using recurrence relations with no third-party dependencies.

## Usage

```python
from spherical_bessel_functions import spherical_j, spherical_y

spherical_j(0, 1.0)   # 0.8414709848078965
spherical_j(3, 2.5)   # 0.022437...
spherical_y(2, 1.0)   # -0.325674...
```

Exported names: `spherical_j(n, x)` and `spherical_y(n, x)`, where `n` is a non-negative integer and `x` is a float.

## Why this exists

Scientific Python stacks usually pull in SciPy for these functions, which is a heavy dependency when all you need is a handful of recurrence-based evaluations. This library provides the same values using only the standard library, at the cost of some performance and the most aggressive edge cases (very high order, very small argument). The trade-off is simplicity and zero install footprint.

## Algorithm

For `j_n`, upward recurrence is numerically unstable when `x` is small relative to `n`, because it amplifies the `y_n` component. We therefore use Miller's downward recurrence: seed tiny values at a higher order, sweep down, and normalise against the analytic `j_0(x) = sin(x)/x`. For `y_n`, upward recurrence is stable, so we seed from `y_0` and `y_1` and recurse up.

## Edge cases

- `spherical_j(n, 0.0)` returns `1.0` for `n == 0` and `0.0` for `n >= 1`, matching the analytic limits.
- `spherical_y(n, 0.0)` raises `ValueError`, since `y_n` has a pole at the origin.
- Negative or non-integer `n` raises `ValueError`.
- For very large `n` relative to `x`, the downward recurrence headroom may be insufficient; this library targets moderate `n` and `x` (roughly `n` up to a few tens).
