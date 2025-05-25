# tests/test_replace_values.py

import time
import unittest
from termcolor import colored
from tabulate import tabulate
from clabtoolkit.parcellationtools.Parcellation import (
    replace_values,
)  # Adjust path if needed


# --- Decorator for test categories ---
def test_category(category):
    def decorator(func):
        setattr(func, "_test_category", category)
        return func

    return decorator


# --- Test class ---
class TestReplaceValues(unittest.TestCase):

    def setUp(self):
        self.results = []

    def _record_result(self, name, passed, duration, error=""):
        self.results.append([name, "✔" if passed else "✘", f"{duration:.4f}s", error])

    @test_category("unit")
    def test_basic_value_replacement(self):
        name = "test_basic_value_replacement"
        start = time.time()
        try:
            data = [1, 2, 3, 4]
            result = replace_values(data, {2: 20, 4: 40})
            expected = [1, 20, 3, 40]
            self.assertEqual(result, expected)
            self._record_result(name, True, time.time() - start)
        except Exception as e:
            self._record_result(name, False, time.time() - start, str(e))
            raise

    @test_category("unit")
    def test_dict_replacement_with_non_matching_keys(self):
        name = "test_dict_replacement_with_non_matching_keys"
        start = time.time()
        try:
            data = ["a", "b", "c"]
            result = replace_values(data, {"b": "beta", "x": "unknown"})
            expected = ["a", "beta", "c"]
            self.assertEqual(result, expected)
            self._record_result(name, True, time.time() - start)
        except Exception as e:
            self._record_result(name, False, time.time() - start, str(e))
            raise

    @test_category("unit")
    def test_empty_inputs(self):
        name = "test_empty_inputs"
        start = time.time()
        try:
            result = replace_values([], {})
            expected = []
            self.assertEqual(result, expected)
            self._record_result(name, True, time.time() - start)
        except Exception as e:
            self._record_result(name, False, time.time() - start, str(e))
            raise

    @classmethod
    def tearDownClass(cls):
        print(
            "\n"
            + colored("Test Summary for test_replace_values.py", "cyan", attrs=["bold"])
        )
        headers = ["Test Name", "Status", "Time", "Error"]
        print(tabulate(cls.results, headers=headers, tablefmt="grid"))
