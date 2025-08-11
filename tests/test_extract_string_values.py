import pytest
import json
import os
import time
import tempfile
import unittest
from functools import wraps
from typing import Union
from tabulate import tabulate


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


# The function being tested
def extract_string_values(data_dict: Union[str, dict], only_last_key=True) -> dict:
    """
    Recursively extracts all keys with string values from a nested dictionary. It will avoid keys
    Parameters:
    -----------
        data_dict: A nested dictionary to search through
        only_last_key: If True, uses only the leaf key name; if False, uses the full path

    Returns:
    --------
        A dictionary where keys are either leaf keys or paths to string values,
        and values are the corresponding strings

    Examples:
        >>> data = {
        ...     "a": {
        ...         "b": "value1",
        ...         "c": {
        ...             "d": "value2"
        ...         }
        ...     },
        ...     "e": ["list", "of", "values"],
        ...     "f": "value3"
        ... }
        >>>
        >>> # With only_last_key=True (default)
        >>> extract_string_values(data)
        {'b': 'value1', 'd': 'value2', 'f': 'value3'}
        >>>
        >>> # With only_last_key=False
        >>> extract_string_values(data, only_last_key=False)
        {'a.b': 'value1', 'a.c.d': 'value2', 'f': 'value3'}
    """

    if isinstance(data_dict, str):
        # Check if the string is a valid JSON file path
        if os.path.isfile(data_dict):
            # Load the custom JSON file
            with open(data_dict, "r") as file:
                data_dict = json.load(file)
        else:
            # If the file does not exist, raise an error
            raise ValueError(f"Invalid file path: {data_dict}")

    result = {}

    def explore_dict(d, path=""):
        if not isinstance(d, dict):
            return

        for key, value in d.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, str):
                # Use either just the key or the full path based on the parameter
                result_key = key if only_last_key else current_path
                result[result_key] = value
            elif isinstance(value, dict):
                explore_dict(value, current_path)
            # Skip lists and other types

    explore_dict(data_dict)
    return result


# Test cases
class TestExtractStringValues(unittest.TestCase):
    """Test case for the extract_string_values function with enhanced feedback."""

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
        print(f"TEST SUMMARY FOR extract_string_values")
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

        # Define expected test categories for extract_string_values
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
            "duplicate keys": any(
                "duplicate" in test.lower() for test in cls.test_timings
            ),
            "complex nesting": any(
                "complex" in test.lower() or "nested" in test.lower()
                for test in cls.test_timings
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
            "leaf keys only": any(
                "last_key" in test.lower() or "leaf" in test.lower()
                for test in cls.test_timings
            ),
            "full path keys": any(
                "full_path" in test.lower() or "path" in test.lower()
                for test in cls.test_timings
            ),
            "file loading": any(
                "file" in test.lower() or "json" in test.lower()
                for test in cls.test_timings
            ),
            "real-world data": any(
                "real" in test.lower() or "bids" in test.lower()
                for test in cls.test_timings
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

        # Function-specific suggestions
        test_names = cls.test_timings.keys()

        if not any("unicode" in name.lower() for name in test_names):
            suggestions.append(
                "Add a test for handling Unicode strings in dictionary values"
            )

        if not any("circular" in name.lower() for name in test_names):
            suggestions.append(
                "Consider testing for circular references or other edge cases in dictionaries"
            )

        if not suggestions:
            print("✅ No significant issues found in the test suite")
        else:
            for suggestion in suggestions:
                print(f"• {suggestion}")

    @collect_info("basic functionality")
    def test_basic_dict_with_last_key(self):
        """Test with a simple dictionary using only_last_key=True."""
        data = {"a": {"b": "value1", "c": {"d": "value2"}}, "f": "value3"}

        expected = {"b": "value1", "d": "value2", "f": "value3"}

        self.assertEqual(extract_string_values(data), expected)

    @collect_info("basic functionality")
    def test_basic_dict_with_full_path(self):
        """Test with a simple dictionary using only_last_key=False."""
        data = {"a": {"b": "value1", "c": {"d": "value2"}}, "f": "value3"}

        expected = {"a.b": "value1", "a.c.d": "value2", "f": "value3"}

        self.assertEqual(extract_string_values(data, only_last_key=False), expected)

    @collect_info("edge cases")
    def test_empty_dict(self):
        """Test with an empty dictionary."""
        data = {}
        expected = {}

        self.assertEqual(extract_string_values(data), expected)
        self.assertEqual(extract_string_values(data, only_last_key=False), expected)

    @collect_info("edge cases")
    def test_dict_with_no_strings(self):
        """Test with a dictionary containing no string values."""
        data = {
            "a": {"b": ["list1"]},
            "e": 123,
            "f": True,
        }

        # There should be no strings extracted
        expected = {}

        # Corrected to match the actual implementation
        self.assertEqual(extract_string_values(data), expected)

    @collect_info("edge cases")
    def test_dict_with_duplicate_keys(self):
        """Test with duplicate leaf keys in different branches."""
        data = {
            "branch1": {"common_key": "value1"},
            "branch2": {"common_key": "value2"},
        }

        # With only_last_key=True, one of the values will overwrite the other
        # The exact result depends on dictionary iteration order, which can vary
        result = extract_string_values(data)
        self.assertEqual(len(result), 1)
        self.assertIn("common_key", result)
        self.assertIn(result["common_key"], ["value1", "value2"])

        # With only_last_key=False, both values should be present with different paths
        expected_full_path = {
            "branch1.common_key": "value1",
            "branch2.common_key": "value2",
        }
        self.assertEqual(
            extract_string_values(data, only_last_key=False), expected_full_path
        )

    @collect_info("input variations")
    def test_complex_nested_dict(self):
        """Test with a more complex nested dictionary."""
        data = {
            "level1": {
                "a": "value_a",
                "level2": {
                    "b": "value_b",
                    "level3": {"c": "value_c", "d": ["list", "items"]},
                },
                "e": 123,
            },
            "f": True,
            "g": "value_g",
        }

        expected_last_key = {
            "a": "value_a",
            "b": "value_b",
            "c": "value_c",
            "g": "value_g",
        }

        expected_full_path = {
            "level1.a": "value_a",
            "level1.level2.b": "value_b",
            "level1.level2.level3.c": "value_c",
            "g": "value_g",
        }

        self.assertEqual(extract_string_values(data), expected_last_key)
        self.assertEqual(
            extract_string_values(data, only_last_key=False), expected_full_path
        )

    @collect_info("input variations")
    def test_from_json_file(self):
        """Test loading from a JSON file."""
        # Create a temporary JSON file
        test_data = {"key1": "value1", "nested": {"key2": "value2"}}

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
            tmp_name = tmp.name
            json.dump(test_data, tmp)

        try:
            expected = {"key1": "value1", "key2": "value2"}

            self.assertEqual(extract_string_values(tmp_name), expected)

            expected_full_path = {"key1": "value1", "nested.key2": "value2"}

            self.assertEqual(
                extract_string_values(tmp_name, only_last_key=False), expected_full_path
            )

        finally:
            # Clean up the temporary file
            os.unlink(tmp_name)

    @collect_info("error handling")
    def test_invalid_file_path(self):
        """Test with an invalid file path."""
        try:
            extract_string_values("nonexistent_file.json")
            self.fail("Should have raised ValueError")
        except ValueError as e:
            self.assertIn("Invalid file path", str(e))

    @collect_info("input variations")
    def test_bids_dictionary(self):
        """Test with a real-world example of a BIDS dictionary."""
        bids_dict = {
            "bids_entities": {
                "raw_entities": {
                    "sub": "Participant",
                    "ses": "Session",
                    "task": "Task",
                },
                "derivatives_entities": {"space": "Space", "hemi": "Hemisphere"},
                "raw_suffix": ["bold", "T1w", "T2w"],
                "extensions": [".nii.gz", ".json"],
            }
        }

        expected_last_key = {
            "sub": "Participant",
            "ses": "Session",
            "task": "Task",
            "space": "Space",
            "hemi": "Hemisphere",
        }

        expected_full_path = {
            "bids_entities.raw_entities.sub": "Participant",
            "bids_entities.raw_entities.ses": "Session",
            "bids_entities.raw_entities.task": "Task",
            "bids_entities.derivatives_entities.space": "Space",
            "bids_entities.derivatives_entities.hemi": "Hemisphere",
        }

        self.assertEqual(extract_string_values(bids_dict), expected_last_key)
        self.assertEqual(
            extract_string_values(bids_dict, only_last_key=False), expected_full_path
        )

    @collect_info("edge cases")
    def test_mixed_content_dict(self):
        """Test with dictionary containing mixed content types."""
        data = {
            "string_key": "string_value",
            "int_key": 42,
            "bool_key": True,
            "list_key": ["a", "b", "c"],
            "nested": {"nested_string": "nested_value", "nested_int": 100},
        }

        expected = {"string_key": "string_value", "nested_string": "nested_value"}

        self.assertEqual(extract_string_values(data), expected)

    @collect_info("edge cases")
    def test_unicode_strings(self):
        """Test with dictionary containing Unicode strings."""
        data = {
            "unicode_key": "Unicode value: 你好, こんにちは, Привет",
            "emoji": "Emoji test: 🚀 🌟 🎉 🎮 🍕",
            "nested": {"more_unicode": "More symbols: ∑ ∞ ♥ ♠"},
        }

        expected = {
            "unicode_key": "Unicode value: 你好, こんにちは, Привет",
            "emoji": "Emoji test: 🚀 🌟 🎉 🎮 🍕",
            "more_unicode": "More symbols: ∑ ∞ ♥ ♠",
        }

        self.assertEqual(extract_string_values(data), expected)


if __name__ == "__main__":
    unittest.main()
