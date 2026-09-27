import math
import unittest

from spherical_bessel_functions.core import spherical_j, spherical_y


class TestSphericalJ(unittest.TestCase):
    def test_j0_at_zero(self):
        self.assertAlmostEqual(spherical_j(0, 0.0), 1.0, places=12)

    def test_j1_at_zero(self):
        self.assertAlmostEqual(spherical_j(1, 0.0), 0.0, places=12)

    def test_jn_at_zero_for_n_ge_2(self):
        for n in [2, 3, 5, 10]:
            self.assertAlmostEqual(spherical_j(n, 0.0), 0.0, places=12)

    def test_j0_known_value(self):
        # j_0(pi) = sin(pi)/pi ~ 0
        val = spherical_j(0, math.pi)
        self.assertAlmostEqual(val, 0.0, places=12)

    def test_j0_one(self):
        # j_0(1) = sin(1)
        self.assertAlmostEqual(spherical_j(0, 1.0), math.sin(1.0), places=12)

    def test_j1_known_value(self):
        # j_1(1) = sin(1) - cos(1)
        expected = math.sin(1.0) - math.cos(1.0)
        self.assertAlmostEqual(spherical_j(1, 1.0), expected, places=12)

    def test_j2_recurrence_check(self):
        # j_2(x) = (3/x) j_1(x) - j_0(x)
        x = 2.5
        expected = (3.0 / x) * spherical_j(1, x) - spherical_j(0, x)
        self.assertAlmostEqual(spherical_j(2, x), expected, places=10)

    def test_j5_against_upward(self):
        # For moderate x, upward and downward should agree.
        x = 5.0
        # Compute via independent upward recurrence
        j0 = math.sin(x) / x
        j1 = math.sin(x) / (x * x) - math.cos(x) / x
        j_prev, j_curr = j0, j1
        for k in range(1, 5):
            j_next = (2 * k + 1) / x * j_curr - j_prev
            j_prev, j_curr = j_curr, j_next
        self.assertAlmostEqual(spherical_j(5, x), j_curr, places=10)

    def test_j10_moderate_x(self):
        # Just check it runs and is finite.
        val = spherical_j(10, 10.0)
        self.assertTrue(math.isfinite(val))

    def test_negative_order_raises(self):
        with self.assertRaises(ValueError):
            spherical_j(-1, 1.0)

    def test_non_integer_order_raises(self):
        with self.assertRaises(ValueError):
            spherical_j(2.5, 1.0)


class TestSphericalY(unittest.TestCase):
    def test_y0_known_value(self):
        # y_0(x) = -cos(x)/x
        x = 1.0
        expected = -math.cos(x) / x
        self.assertAlmostEqual(spherical_y(0, x), expected, places=12)

    def test_y1_known_value(self):
        # y_1(x) = -cos(x)/x^2 - sin(x)/x
        x = 1.0
        expected = -math.cos(x) / (x * x) - math.sin(x) / x
        self.assertAlmostEqual(spherical_y(1, x), expected, places=12)

    def test_y2_recurrence_check(self):
        # y_2(x) = (3/x) y_1(x) - y_0(x)
        x = 2.5
        expected = (3.0 / x) * spherical_y(1, x) - spherical_y(0, x)
        self.assertAlmostEqual(spherical_y(2, x), expected, places=10)

    def test_y5_upward_independent(self):
        x = 3.0
        y0 = -math.cos(x) / x
        y1 = -math.cos(x) / (x * x) - math.sin(x) / x
        y_prev, y_curr = y0, y1
        for k in range(1, 5):
            y_next = (2 * k + 1) / x * y_curr - y_prev
            y_prev, y_curr = y_curr, y_next
        self.assertAlmostEqual(spherical_y(5, x), y_curr, places=10)

    def test_y_at_zero_raises(self):
        with self.assertRaises(ValueError):
            spherical_y(0, 0.0)

    def test_y_negative_order_raises(self):
        with self.assertRaises(ValueError):
            spherical_y(-1, 1.0)

    def test_y_non_integer_order_raises(self):
        with self.assertRaises(ValueError):
            spherical_y(1.5, 1.0)


class TestWronskian(unittest.TestCase):
    def test_wronskian_identity(self):
        # j_n(x) y_{n-1}(x) - j_{n-1}(x) y_n(x) = 1/x^2
        x = 4.0
        for n in range(1, 6):
            lhs = (
                spherical_j(n, x) * spherical_y(n - 1, x)
                - spherical_j(n - 1, x) * spherical_y(n, x)
            )
            self.assertAlmostEqual(lhs, 1.0 / (x * x), places=8)


if __name__ == "__main__":
    unittest.main()
