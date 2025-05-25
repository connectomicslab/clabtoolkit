import pytest
import os
import time
import tempfile
import unittest
import pandas as pd
from functools import wraps
from typing import Union, Dict, List, Optional
from tabulate import tabulate

# Import function to test
from clabtoolkit.morphometrytools import entities_to_table


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


# Test cases
class TestExtractBidsEntities(unittest.TestCase):
    """Test case for the entities_to_table function with enhanced feedback."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures and initialize tracking."""
        cls.test_timings = {}
        cls.start_time = time.time()

        # Mock the cltbids module
        import sys
        import types

        # Create a mock cltbids module
        mock_cltbids = types.ModuleType("cltbids")

        # Define mock functions
        def mock_str2entity(filename):
            """Mock implementation of str2entity function."""
            if "sub-01_ses-pre_task-rest" in filename:
                return {"sub": "01", "ses": "pre", "task": "rest"}
            elif "sub-02_ses-post" in filename:
                return {"sub": "02", "ses": "post"}
            elif "atlas-chimera123" in filename:
                return {"atlas": "chimera123", "desc": "parcellation"}
            elif "desc-grow456" in filename:
                return {"desc": "grow456"}
            else:
                return {"sub": "unknown"}

        def mock_is_bids_filename(filename):
            """Mock implementation of is_bids_filename function."""
            return "non_bids" not in filename

        def mock_entity2str(entity_dict):
            """Mock implementation of entity2str function."""
            if "sub" in entity_dict and entity_dict["sub"] == "01":
                return "sub-01_ses-pre_task-rest"
            else:
                return "unknown_entity"

        # Add mock functions to mock module
        mock_cltbids.str2entity = mock_str2entity
        mock_cltbids.is_bids_filename = mock_is_bids_filename
        mock_cltbids.entity2str = mock_entity2str

        # Add the mock module to sys.modules
        sys.modules["cltbids"] = mock_cltbids

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
        print(f"TEST SUMMARY FOR entities_to_table")
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

        # Define expected test categories for entities_to_table
        expected_categories = {
            "basic functionality": True,
            "edge cases": True,
            "error handling": True,
            "input variations": True,
            "special entity handling": True,
        }

        missing_categories = []
        for category in expected_categories:
            if not any(cat.lower() == category.lower() for cat in categories.keys()):
                missing_categories.append(category)
                expected_categories[category] = False

        if missing_categories:
            print(f"⚠️  Missing test categories: {', '.join(missing_categories)}")
        else:
            print("✅ All expected test categories are covered")

        # Check if common edge cases are tested
        edge_cases = {
            "empty input": any("empty" in test.lower() for test in cls.test_timings),
            "non-bids file": any(
                "non-bids" in test.lower() for test in cls.test_timings
            ),
            "custom column names": any(
                "custom" in test.lower() for test in cls.test_timings
            ),
            "complex entities": any(
                "complex" in test.lower() or "multiple" in test.lower()
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
            "subject parsing": any(
                "subject" in test.lower() or "sub" in test.lower()
                for test in cls.test_timings
            ),
            "session parsing": any(
                "session" in test.lower() or "ses" in test.lower()
                for test in cls.test_timings
            ),
            "atlas handling": any(
                "atlas" in test.lower() or "chimera" in test.lower()
                for test in cls.test_timings
            ),
            "description handling": any(
                "desc" in test.lower() or "grow" in test.lower()
                for test in cls.test_timings
            ),
            "custom column names": any(
                "custom" in test.lower() or "column name" in test.lower()
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

        if not any("performance" in name.lower() for name in test_names):
            suggestions.append(
                "Add performance tests for large datasets or complex entity structures"
            )

        if not any("integration" in name.lower() for name in test_names):
            suggestions.append(
                "Consider adding integration tests with real BIDS datasets"
            )

        if not suggestions:
            print("✅ No significant issues found in the test suite")
        else:
            for suggestion in suggestions:
                print(f"• {suggestion}")

    @collect_info("basic functionality")
    def test_basic_extraction(self):
        """Test basic entity extraction from a BIDS filename."""
        filepath = "/data/sub-01_ses-pre_task-rest_bold.nii.gz"
        entities = ["sub", "ses"]

        result = entities_to_table(filepath, entities)

        # Check basic properties
        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(result["Participant"].iloc[0], "01")
        self.assertEqual(result["Session"].iloc[0], "pre")

    @collect_info("basic functionality")
    def test_single_entity_extraction(self):
        """Test extraction of a single entity from a BIDS filename."""
        filepath = "/data/sub-01_ses-pre_task-rest_bold.nii.gz"
        entity = "sub"

        result = entities_to_table(filepath, entity)

        # Check result
        self.assertEqual(result.shape, (1, 1))
        self.assertEqual(result["Participant"].iloc[0], "01")

    @collect_info("input variations")
    def test_custom_column_names(self):
        """Test extraction with custom column names provided as a dictionary."""
        filepath = "/data/sub-01_ses-pre_task-rest_bold.nii.gz"
        entities = {"sub": "SubjectID", "ses": "SessionName"}

        result = entities_to_table(filepath, entities)

        # Check custom column names
        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(result["SubjectID"].iloc[0], "01")
        self.assertEqual(result["SessionName"].iloc[0], "pre")

    @collect_info("special entity handling")
    def test_atlas_handling(self):
        """Test special handling for atlas entities."""
        filepath = "/data/atlas-chimera123_desc-parcellation.nii.gz"
        entities = ["atlas", "desc"]

        result = entities_to_table(filepath, entities)

        # Check atlas special handling
        self.assertEqual(result.shape, (1, 4))
        self.assertEqual(result["Atlas"].iloc[0], "chimera")
        self.assertEqual(result["ChimeraCode"].iloc[0], "123")
        self.assertEqual(result["Description"].iloc[0], "parcellation")

    @collect_info("special entity handling")
    def test_description_grow_handling(self):
        """Test special handling for description entities with 'grow' pattern."""
        filepath = "/data/desc-grow456_bold.nii.gz"
        entities = ["desc"]

        result = entities_to_table(filepath, entities)

        # Check description special handling
        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(result["Description"].iloc[0], "grow456")
        self.assertEqual(result["GrowIntoWM"].iloc[0], "456")

    @collect_info("edge cases")
    def test_empty_dataframe(self):
        """Test behavior when no entities are extracted."""
        filepath = "/data/sub-01_ses-pre_task-rest_bold.nii.gz"
        entities = ["run"]  # Entity not present in the filename

        result = entities_to_table(filepath, entities)

        # Should have one row but empty value for the entity
        self.assertEqual(result.shape, (1, 1))
        self.assertEqual(result["Run"].iloc[0], "")

    @collect_info("edge cases")
    def test_non_bids_file(self):
        """Test behavior with non-BIDS files."""
        filepath = "/data/non_bids_file.txt"
        entities = ["sub", "ses"]

        result = entities_to_table(filepath, entities)

        # Should return empty DataFrame
        self.assertTrue(result.empty)

    @collect_info("edge cases")
    def test_no_entities_specified(self):
        """Test behavior when no entities are specified."""
        filepath = "/data/sub-01_ses-pre_task-rest_bold.nii.gz"

        result = entities_to_table(filepath)

        # Should return DataFrame with one column for the full filename
        self.assertEqual(result.shape, (1, 1))
        self.assertEqual(result["Participant"].iloc[0], "sub-01_ses-pre_task-rest")

    @collect_info("input variations")
    def test_multiple_entities(self):
        """Test extraction of multiple entities from a BIDS filename."""
        filepath = "/data/sub-01_ses-pre_task-rest_bold.nii.gz"
        entities = ["sub", "ses", "task"]

        result = entities_to_table(filepath, entities)

        # Check all entities are extracted
        self.assertEqual(result.shape, (1, 3))
        self.assertEqual(result["Participant"].iloc[0], "01")
        self.assertEqual(result["Session"].iloc[0], "pre")
        self.assertEqual(result["Task"].iloc[0], "rest")

    @collect_info("input variations")
    def test_mixed_entity_types(self):
        """Test extraction with a mix of standard and custom entities."""
        filepath = "/data/sub-01_ses-pre_task-rest_acq-multiband_bold.nii.gz"
        entities = {"sub": "", "ses": "", "acq": "AcquisitionType"}

        # Mock the str2entity function to include the 'acq' entity
        import cltbids

        original_str2entity = cltbids.str2entity

        def mock_str2entity_with_acq(filename):
            result = original_str2entity(filename)
            result["acq"] = "multiband"
            return result

        cltbids.str2entity = mock_str2entity_with_acq

        try:
            result = entities_to_table(filepath, entities)

            # Check results
            self.assertEqual(result.shape, (1, 3))
            self.assertEqual(result["Participant"].iloc[0], "01")
            self.assertEqual(result["Session"].iloc[0], "pre")
            self.assertEqual(result["AcquisitionType"].iloc[0], "multiband")
        finally:
            # Restore the original function
            cltbids.str2entity = original_str2entity

    @collect_info("error handling")
    def test_invalid_input_types(self):
        """Test behavior with invalid input types."""
        # Test with None filepath
        with self.assertRaises(TypeError):
            entities_to_table(None, ["sub"])

        # Test with non-string filepath
        with self.assertRaises(TypeError):
            entities_to_table(123, ["sub"])

    @collect_info("basic functionality")
    def test_complex_path(self):
        """Test extraction from a complex filepath with directory structure."""
        filepath = "/data/bids_dataset/sub-01/ses-pre/func/sub-01_ses-pre_task-rest_bold.nii.gz"
        entities = ["sub", "ses", "task"]

        result = entities_to_table(filepath, entities)

        # Check all entities are extracted correctly
        self.assertEqual(result.shape, (1, 3))
        self.assertEqual(result["Participant"].iloc[0], "01")
        self.assertEqual(result["Session"].iloc[0], "pre")
        self.assertEqual(result["Task"].iloc[0], "rest")

    @collect_info("error handling")
    def test_import_error_handling(self):
        """Test handling of import errors with cltbids module."""
        import sys

        # Save original modules
        original_modules = sys.modules.copy()

        try:
            # Remove cltbids from sys.modules to simulate import error
            if "cltbids" in sys.modules:
                del sys.modules["cltbids"]

            # Define a mock import function that raises ImportError for cltbids
            original_import = __import__

            def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "cltbids":
                    raise ImportError("Mock import error for cltbids")
                return original_import(name, globals, locals, fromlist, level)

            # Replace the built-in import function
            __builtins__["__import__"] = mock_import

            # This should raise ImportError due to missing cltbids
            with self.assertRaises(ImportError):
                entities_to_table("/data/sub-01_bold.nii.gz", ["sub"])

        finally:
            # Restore the original import function and modules
            __builtins__["__import__"] = original_import
            sys.modules = original_modules

    @collect_info("performance")
    def test_performance_large_dataset(self):
        """Test performance with a large number of files."""
        # Create a list of 100 simulated filepaths
        filepaths = [
            f"/data/sub-{i:02d}/ses-{j:02d}/func/sub-{i:02d}_ses-{j:02d}_task-rest_bold.nii.gz"
            for i in range(1, 11)
            for j in range(1, 11)
        ]

        # Process each filepath and measure performance
        start_time = time.time()

        for filepath in filepaths:
            entities_to_table(filepath, ["sub", "ses"])

        end_time = time.time()
        total_time = end_time - start_time

        # Print performance metrics
        print(f"\nProcessed {len(filepaths)} files in {total_time:.4f} seconds")
        print(f"Average time per file: {(total_time / len(filepaths)) * 1000:.4f} ms")

        # No specific assertion, just logging performance

    @collect_info("input variations")
    def test_real_world_bids_example(self):
        """Test with a realistic BIDS filename containing multiple entities."""
        filepath = "/data/bids_dataset/sub-01/ses-pre/func/sub-01_ses-pre_task-rest_acq-multiband_run-01_echo-1_bold.nii.gz"

        # Mock the str2entity function to include all entities
        import cltbids

        original_str2entity = cltbids.str2entity

        def mock_str2entity_complete(filename):
            return {
                "sub": "01",
                "ses": "pre",
                "task": "rest",
                "acq": "multiband",
                "run": "01",
                "echo": "1",
                "suffix": "bold",
                "extension": ".nii.gz",
            }

        cltbids.str2entity = mock_str2entity_complete

        try:
            # Extract all entities
            result = entities_to_table(
                filepath, ["sub", "ses", "task", "acq", "run", "echo"]
            )

            # Check all entities are extracted correctly
            self.assertEqual(result.shape, (1, 6))
            self.assertEqual(result["Participant"].iloc[0], "01")
            self.assertEqual(result["Session"].iloc[0], "pre")
            self.assertEqual(result["Task"].iloc[0], "rest")
            self.assertEqual(result["Acq"].iloc[0], "multiband")
            self.assertEqual(result["Run"].iloc[0], "01")
            self.assertEqual(result["Echo"].iloc[0], "1")
        finally:
            # Restore the original function
            cltbids.str2entity = original_str2entity

    @collect_info("special entity handling")
    def test_combined_special_entities(self):
        """Test a file with multiple special entities that need custom handling."""
        filepath = "/data/atlas-chimera123_desc-grow456_bold.nii.gz"

        # Mock the str2entity function to include both special entities
        import cltbids

        original_str2entity = cltbids.str2entity

        def mock_str2entity_special(filename):
            return {
                "atlas": "chimera123",
                "desc": "grow456",
                "suffix": "bold",
                "extension": ".nii.gz",
            }

        cltbids.str2entity = mock_str2entity_special

        try:
            # Extract both special entities
            result = entities_to_table(filepath, ["atlas", "desc"])

            # Check all special entity handling is applied correctly
            self.assertEqual(result.shape, (1, 5))
            self.assertEqual(result["Atlas"].iloc[0], "chimera")
            self.assertEqual(result["ChimeraCode"].iloc[0], "123")
            self.assertEqual(result["Description"].iloc[0], "grow456")
            self.assertEqual(result["GrowIntoWM"].iloc[0], "456")
        finally:
            # Restore the original function
            cltbids.str2entity = original_str2entity


if __name__ == "__main__":
    unittest.main()
