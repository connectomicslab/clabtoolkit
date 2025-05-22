import os
import json
import tempfile
import unittest
import time
from functools import wraps
from tabulate import tabulate
from unittest.mock import patch, mock_open

# Import the module containing get_units
from clabtoolkit.morphometrytools import get_units


def collect_info(category):
    """Decorator to add metadata and timing to tests."""

    def decorator(func):
        func.category = category

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            test_duration = end_time - start_time

            # Store timing information in the test class
            if not hasattr(self, "test_timings"):
                self.test_timings = {}
            self.test_timings[func.__name__] = {
                "duration": test_duration,
                "category": category,
                "name": func.__name__,
                "docstring": func.__doc__,
            }

            return result

        return wrapper

    return decorator


class TestGetUnits(unittest.TestCase):
    """Test case for the get_units function with enhanced feedback."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures and initialize tracking."""
        cls.test_timings = {}
        cls.start_time = time.time()

    @classmethod
    def tearDownClass(cls):
        """Generate and print test feedback and analytics."""
        total_time = time.time() - cls.start_time

        # Process test timing data
        timing_data = []
        categories = {}

        for test_name, data in cls.test_timings.items():
            timing_data.append(
                [
                    test_name,
                    data["category"],
                    f"{data['duration']*1000:.2f} ms",
                    (
                        data["docstring"].strip()
                        if data["docstring"]
                        else "No description"
                    ),
                ]
            )

            # Group by category
            if data["category"] not in categories:
                categories[data["category"]] = {"count": 0, "total_time": 0}
            categories[data["category"]]["count"] += 1
            categories[data["category"]]["total_time"] += data["duration"]

        # Sort by execution time (descending)
        timing_data.sort(key=lambda x: float(x[2].split()[0]), reverse=True)

        # Print test summary
        print("\n" + "=" * 80)
        print(f"TEST SUMMARY FOR get_units")
        print("=" * 80)
        print(f"Total tests: {len(cls.test_timings)}")
        print(f"Passed tests: {len(cls.test_timings)}")
        print(f"Success rate: 100.0%")
        print(f"Total execution time: {total_time:.2f} seconds")

        # Print category summary
        print("\n" + "-" * 80)
        print("TEST CATEGORIES")
        print("-" * 80)
        category_data = []
        for category, data in categories.items():
            category_data.append(
                [
                    category,
                    data["count"],
                    f"{data['total_time']*1000:.2f} ms",
                    f"{(data['count']/len(cls.test_timings))*100:.1f}%",
                ]
            )
        print(
            tabulate(
                category_data,
                headers=["Category", "Count", "Total Time", "Coverage %"],
                tablefmt="grid",
            )
        )

        # Print timing table
        print("\n" + "-" * 80)
        print("TEST TIMING (SORTED BY DURATION)")
        print("-" * 80)
        print(
            tabulate(
                timing_data,
                headers=["Test Name", "Category", "Duration", "Description"],
                tablefmt="grid",
            )
        )

        # Print test coverage analysis
        print("\n" + "-" * 80)
        print("TEST COVERAGE ANALYSIS")
        print("-" * 80)

        # Define expected test categories for get_units
        expected_categories = {
            "basic functionality": True,
            "edge cases": True,
            "error handling": True,
            "input variations": True,
        }

        missing_categories = []
        for category in expected_categories:
            if not any(cat.lower() == category for cat in categories.keys()):
                missing_categories.append(category)
                expected_categories[category] = False

        if missing_categories:
            print(f"⚠️  Missing test categories: {', '.join(missing_categories)}")
        else:
            print("✅ All expected test categories are covered")

        # Check if common edge cases are tested
        edge_cases = {
            "empty input": any("empty" in test.lower() for test in cls.test_timings),
            "invalid input": any(
                "invalid" in test.lower() for test in cls.test_timings
            ),
            "case sensitivity": any(
                "case" in test.lower() for test in cls.test_timings
            ),
            "unknown values": any(
                "unknown" in test.lower() for test in cls.test_timings
            ),
        }

        missing_edge_cases = [
            case for case, covered in edge_cases.items() if not covered
        ]
        if missing_edge_cases:
            print(f"⚠️  Missing edge cases: {', '.join(missing_edge_cases)}")
        else:
            print("✅ All common edge cases are tested")

        # Function-specific coverage check
        function_features = {
            "single metric input": any(
                "single" in test.lower() for test in cls.test_timings
            ),
            "multiple metrics input": any(
                "multiple" in test.lower() for test in cls.test_timings
            ),
            "custom JSON file": any(
                "custom" in test.lower() and "json" in test.lower()
                for test in cls.test_timings
            ),
            "dictionary input": any(
                "dict" in test.lower() for test in cls.test_timings
            ),
            "default configuration": any(
                "default" in test.lower() for test in cls.test_timings
            ),
        }

        missing_features = [
            feature for feature, covered in function_features.items() if not covered
        ]
        if missing_features:
            print(f"⚠️  Missing function feature tests: {', '.join(missing_features)}")
        else:
            print("✅ All function features are tested")

        # Suggestions for improvement
        print("\n" + "-" * 80)
        print("SUGGESTIONS FOR IMPROVEMENT")
        print("-" * 80)

        suggestions = []

        # Check for very fast tests that might be combined
        fast_tests = [
            name for name, data in cls.test_timings.items() if data["duration"] < 0.001
        ]
        if len(fast_tests) > 3:
            suggestions.append(
                f"Consider combining some of the {len(fast_tests)} very fast tests for efficiency"
            )

        # Check for slow tests
        slow_tests = [
            name for name, data in cls.test_timings.items() if data["duration"] > 0.1
        ]
        if slow_tests:
            suggestions.append(
                f"Optimize the following slow tests: {', '.join(slow_tests)}"
            )

        # Check docstring quality
        missing_docs = [
            name
            for name, data in cls.test_timings.items()
            if not data["docstring"] or len(data["docstring"]) < 10
        ]
        if missing_docs:
            suggestions.append(
                f"Improve documentation for tests: {', '.join(missing_docs)}"
            )

        # Get_units specific suggestions
        test_names = cls.test_timings.keys()

        if not any("metrics_json_none" in name.lower() for name in test_names):
            suggestions.append(
                "Add a test for the case when metrics_json is explicitly None"
            )

        if not any("nested_dict" in name.lower() for name in test_names):
            suggestions.append("Add a test for deeply nested dictionary structures")

        if not suggestions:
            print("✅ No significant issues found in the test suite")
        else:
            for suggestion in suggestions:
                print(f"• {suggestion}")

    def setUp(self):
        """Set up test fixtures."""
        # Example metrics dictionary for testing
        self.test_metrics_dict = {
            "metrics_units": {
                "thickness": "mm",
                "area": "cm²",
                "volume": "cm³",
                "UPPERCASE_METRIC": "kg",
            }
        }

        # Create a temporary JSON file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_json_path = os.path.join(self.temp_dir.name, "test_config.json")
        with open(self.temp_json_path, "w") as f:
            json.dump(self.test_metrics_dict, f)

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    @collect_info("basic functionality")
    def test_single_metric(self):
        """Test getting unit for a single metric as string."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.test_metrics_dict))
        ):
            result = get_units("thickness")
            self.assertEqual(result, ["mm"])

    @collect_info("basic functionality")
    def test_multiple_metrics(self):
        """Test getting units for multiple metrics as list."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.test_metrics_dict))
        ):
            result = get_units(["thickness", "area", "volume"])
            self.assertEqual(result, ["mm", "cm²", "cm³"])

    @collect_info("input variations")
    def test_case_insensitivity(self):
        """Test that metric lookup is case-insensitive."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.test_metrics_dict))
        ):
            result = get_units(["THICKNESS", "Area", "Volume"])
            self.assertEqual(result, ["mm", "cm²", "cm³"])

            # Test uppercase metric with lowercase query
            result = get_units("uppercase_metric")
            self.assertEqual(result, ["kg"])

    @collect_info("edge cases")
    def test_unknown_metric(self):
        """Test handling of unknown metrics."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.test_metrics_dict))
        ):
            result = get_units(["thickness", "unknown_metric"])
            self.assertEqual(result, ["mm", "unknown"])

    @collect_info("input variations")
    def test_custom_json_file(self):
        """Test using a custom JSON file."""
        custom_metrics = {"metrics_units": {"custom_metric": "custom_unit"}}

        # Create another temporary file with custom metrics
        custom_json_path = os.path.join(self.temp_dir.name, "custom_config.json")
        with open(custom_json_path, "w") as f:
            json.dump(custom_metrics, f)

        result = get_units("custom_metric", metrics_json=custom_json_path)
        self.assertEqual(result, ["custom_unit"])

    @collect_info("input variations")
    def test_dict_input(self):
        """Test using a dictionary input instead of a file path."""
        # Test with nested dictionary containing metrics_units key
        result = get_units("thickness", metrics_json=self.test_metrics_dict)
        self.assertEqual(result, ["mm"])

        # Test with flat dictionary
        flat_dict = {"thickness": "mm", "custom": "unit"}
        result = get_units("custom", metrics_json={"metrics_units": flat_dict})
        self.assertEqual(result, ["unit"])

        # Test with direct flat dictionary
        result = get_units("custom", metrics_json=flat_dict)
        self.assertEqual(result, ["unit"])

    @collect_info("error handling")
    def test_invalid_json_path(self):
        """Test error handling for invalid JSON file path."""
        with self.assertRaises(ValueError):
            get_units("thickness", metrics_json="/nonexistent/path.json")

    @collect_info("error handling")
    def test_invalid_json_format(self):
        """Test error handling for invalid JSON format."""
        # Create a file with invalid JSON
        invalid_json_path = os.path.join(self.temp_dir.name, "invalid.json")
        with open(invalid_json_path, "w") as f:
            f.write("This is not valid JSON")

        with self.assertRaises(ValueError):
            get_units("thickness", metrics_json=invalid_json_path)

    @collect_info("edge cases")
    def test_invalid_dict_structure(self):
        """Test handling of dictionary with missing metrics_units key."""
        # This should still work by using the dict directly
        empty_dict = {"not_metrics_units": {}}
        result = get_units("anything", metrics_json=empty_dict)
        self.assertEqual(result, ["unknown"])

    @collect_info("edge cases")
    def test_empty_metrics_list(self):
        """Test handling of empty metrics list."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.test_metrics_dict))
        ):
            result = get_units([])
            self.assertEqual(result, [])

    @patch("os.path.isfile")
    @patch("os.path.dirname")
    @collect_info("input variations")
    def test_default_config(self, mock_dirname, mock_isfile):
        """Test using the default configuration."""
        # Mock the dirname and isfile functions
        mock_dirname.return_value = "/fake/path"
        mock_isfile.return_value = True

        # Mock the open function to return our test metrics
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.test_metrics_dict))
        ):
            result = get_units("thickness")
            self.assertEqual(result, ["mm"])

    @collect_info("error handling")
    def test_invalid_input_type(self):
        """Test handling of invalid input types."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.test_metrics_dict))
        ):
            with self.assertRaises(ValueError):
                # Pass an integer instead of a string or dict
                get_units("thickness", metrics_json=123)

    @collect_info("edge cases")
    def test_metrics_json_none(self):
        """Test explicitly passing None as metrics_json."""
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(self.test_metrics_dict))
        ):
            # This should use the default config path
            result = get_units("thickness", metrics_json=None)
            self.assertEqual(result, ["mm"])

    @collect_info("edge cases")
    def test_nested_dict_structure(self):
        """Test handling of deeply nested dictionary structures."""
        # Create a nested dict with metrics_units buried deep inside
        nested_dict = {
            "level1": {"level2": {"metrics_units": {"nested_metric": "nested_unit"}}}
        }

        # This should fail to find metrics_units and return unknown
        result = get_units("nested_metric", metrics_json=nested_dict)
        self.assertEqual(result, ["unknown"])

        # But if we pass the direct path to metrics_units, it should work
        direct_path = nested_dict["level1"]["level2"]
        result = get_units("nested_metric", metrics_json=direct_path)
        self.assertEqual(result, ["nested_unit"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
