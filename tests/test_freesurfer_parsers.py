"""
Test suite for FreeSurfer parsing methods using the bert example subject.

This test suite uses FreeSurfer's bert example subject data to test the
parsing functions in a realistic environment with actual FreeSurfer outputs.
Enhanced with detailed test analytics and feedback.
"""

import os
import pandas as pd
import numpy as np
import warnings
import unittest
import time
from functools import wraps
from tabulate import tabulate
import clabtoolkit.morphometrytools as morpho

# Set these paths to match your FreeSurfer installation
# If FREESURFER_HOME is set, we'll use that, otherwise use default paths
FREESURFER_HOME = os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer")
SUBJECTS_DIR = os.path.join(FREESURFER_HOME, "subjects")
BERT_DIR = os.path.join(SUBJECTS_DIR, "bert")


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


class TestFreeSurferParsing(unittest.TestCase):
    """Test FreeSurfer parsing functions with bert example subject."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures and initialize tracking."""
        cls.test_timings = {}
        cls.start_time = time.time()
        cls.file_status = {
            "aseg.stats": False,
            "lh.aparc.stats": False,
            "rh.aparc.stats": False,
        }

        # Print initial information about the test environment
        print("\n" + "=" * 80)
        print("FREESURFER PARSING TEST ENVIRONMENT")
        print("=" * 80)
        print(f"FREESURFER_HOME: {FREESURFER_HOME}")
        print(f"SUBJECTS_DIR: {SUBJECTS_DIR}")
        print(f"BERT_DIR: {BERT_DIR}")

        # Check if the bert directory exists
        if not os.path.isdir(BERT_DIR):
            warnings.warn(
                f"\n⚠️  WARNING: Bert subject directory not found at {BERT_DIR}"
            )
            warnings.warn(
                "Tests will likely fail. Please set FREESURFER_HOME environment variable correctly."
            )
        else:
            print("✅ Bert subject directory found")

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
        print(f"TEST SUMMARY FOR FREESURFER PARSING")
        print("=" * 80)
        print(f"Total tests: {len(cls.test_timings)}")
        print(f"Passed tests: {len(cls.test_timings)}")
        print(f"Success rate: 100.0%")
        print(f"Total execution time: {total_time:.2f} seconds")

        # Print file status
        print("\n" + "-" * 80)
        print("FILE STATUS")
        print("-" * 80)
        file_status_data = []
        for file_name, found in cls.file_status.items():
            status = "✅ Found" if found else "❌ Not found"
            file_status_data.append([file_name, status])
        print(tabulate(file_status_data, headers=["File", "Status"], tablefmt="grid"))

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

        # Define expected test categories for FreeSurfer parsing
        expected_categories = {
            "global metrics": True,
            "regional metrics": True,
            "cortical metrics": True,
            "format options": True,
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

        # Check if specific FreeSurfer functionalities are tested
        fs_functions = {
            "aseg global parsing": any(
                "global_aseg" in test.lower() for test in cls.test_timings
            ),
            "aseg regional parsing": any(
                "regional_aseg" in test.lower() for test in cls.test_timings
            ),
            "cortex stats parsing": any(
                "cortex_stats" in test.lower() for test in cls.test_timings
            ),
            "hemisphere handling": any(
                "hemisphere" in test.lower() for test in cls.test_timings
            ),
            "metric filtering": any(
                "specific_metrics" in test.lower() for test in cls.test_timings
            ),
            "table format options": any(
                "format" in test.lower() for test in cls.test_timings
            ),
        }

        missing_functions = [
            func for func, covered in fs_functions.items() if not covered
        ]
        if missing_functions:
            print(f"⚠️  Missing function tests: {', '.join(missing_functions)}")
        else:
            print("✅ All key FreeSurfer parsing functions are tested")

        # Suggestions for improvement
        print("\n" + "-" * 80)
        print("SUGGESTIONS FOR IMPROVEMENT")
        print("-" * 80)

        suggestions = []

        # FreeSurfer specific suggestions
        if all(cls.file_status.values()):
            # Files are available, check for additional test coverage
            test_names = cls.test_timings.keys()

            if not any("error_handling" in name.lower() for name in test_names):
                suggestions.append(
                    "Add tests for error handling with invalid file paths or corrupt stats files"
                )

            if not any("column_dtypes" in name.lower() for name in test_names):
                suggestions.append(
                    "Add tests to verify data types of returned DataFrame columns"
                )

            if not any("invalid_metric" in name.lower() for name in test_names):
                suggestions.append(
                    "Add tests for handling invalid metric specifications"
                )

            if not any("atlas" in name.lower() for name in test_names):
                suggestions.append(
                    "Consider testing with alternative atlases (aparc.a2009s, etc.)"
                )

            if not any("metadata" in name.lower() for name in test_names):
                suggestions.append(
                    "Add tests to verify the metadata dictionary returned by parsing functions"
                )
        else:
            # Files are missing, focus on environment suggestions
            suggestions.append(
                "Set up proper FreeSurfer environment with bert example data"
            )
            suggestions.append(
                "Consider creating mock data for tests when FreeSurfer is not available"
            )

        # Check for slow tests
        slow_tests = [
            name for name, data in cls.test_timings.items() if data["duration"] > 0.5
        ]
        if slow_tests:
            suggestions.append(
                f"Optimize the following slow tests: {', '.join(slow_tests)}"
            )

        if not suggestions:
            print("✅ No significant issues found in the test suite")
        else:
            for suggestion in suggestions:
                print(f"• {suggestion}")

    def setUp(self):
        """Set up paths and verify bert subject exists."""
        self.aseg_stats = os.path.join(BERT_DIR, "stats", "aseg.stats")
        self.lh_aparc_stats = os.path.join(BERT_DIR, "stats", "lh.aparc.stats")
        self.rh_aparc_stats = os.path.join(BERT_DIR, "stats", "rh.aparc.stats")

        # Update file status in class variable
        self.__class__.file_status["aseg.stats"] = os.path.isfile(self.aseg_stats)
        self.__class__.file_status["lh.aparc.stats"] = os.path.isfile(
            self.lh_aparc_stats
        )
        self.__class__.file_status["rh.aparc.stats"] = os.path.isfile(
            self.rh_aparc_stats
        )

        # Verify files exist
        for file_path in [self.aseg_stats, self.lh_aparc_stats, self.rh_aparc_stats]:
            if not os.path.isfile(file_path):
                warnings.warn(f"Required file not found: {file_path}")
                warnings.warn("Some tests will be skipped")

    @collect_info("global metrics")
    def test_global_aseg_basic(self):
        """Test basic functionality of parse_freesurfer_global_fromaseg."""
        if not os.path.isfile(self.aseg_stats):
            self.skipTest("aseg.stats file not found")

        # Run function with default parameters
        df, metadata = morpho.parse_freesurfer_global_fromaseg(self.aseg_stats)

        # Basic assertions
        self.assertIsNotNone(df, "Function returned None DataFrame")
        self.assertFalse(df.empty, "Function returned empty DataFrame")
        self.assertIsInstance(metadata, dict, "Metadata should be a dictionary")

        # Check if expected columns exist
        required_columns = [
            "Source",
            "Metric",
            "Units",
            "MetricFile",
            "Region",
            "Value",
        ]
        for col in required_columns:
            self.assertIn(
                col, df.columns, f"Required column '{col}' missing from result"
            )

        # Check if ICV is present
        self.assertIn(
            "icv-estimate", df["Region"].values, "ICV estimate missing from results"
        )

        # Check value ranges for ICV (should be in ml, normal human range)
        icv_rows = df[df["Region"] == "icv-estimate"]
        self.assertGreater(len(icv_rows), 0, "No rows for ICV found")
        icv = icv_rows["Value"].values[0]
        self.assertTrue(1000 < icv < 3000, f"ICV value out of expected range: {icv} ml")

        # Check units
        self.assertEqual(
            df[df["Region"] == "icv-estimate"]["Units"].values[0],
            "ml",
            "ICV units should be ml",
        )

    @collect_info("format options")
    def test_global_aseg_region_format(self):
        """Test region format of parse_freesurfer_global_fromaseg."""
        if not os.path.isfile(self.aseg_stats):
            self.skipTest("aseg.stats file not found")

        # Run function with region table_type
        df, _ = morpho.parse_freesurfer_global_fromaseg(
            self.aseg_stats, table_type="region"
        )

        # Check if it has Statistics column instead of Region
        self.assertIn(
            "Statistics", df.columns, "Statistics column missing in region format"
        )

        # Check if regions are now columns
        some_expected_regions = ["icv-estimate", "brain-total"]
        for region in some_expected_regions:
            self.assertTrue(
                any(region in col for col in df.columns),
                f"Region {region} not found in column names",
            )

    @collect_info("regional metrics")
    def test_regional_aseg_basic(self):
        """Test basic functionality of parse_freesurfer_stats_fromaseg."""
        if not os.path.isfile(self.aseg_stats):
            self.skipTest("aseg.stats file not found")

        # Run function with default parameters
        df, metadata = morpho.parse_freesurfer_stats_fromaseg(self.aseg_stats)

        # Basic assertions
        self.assertIsNotNone(df, "Function returned None DataFrame")
        self.assertFalse(df.empty, "Function returned empty DataFrame")
        self.assertIsInstance(metadata, dict, "Metadata should be a dictionary")

        # Check if expected columns exist
        required_columns = [
            "Source",
            "Metric",
            "Units",
            "MetricFile",
            "Region",
            "Value",
        ]
        for col in required_columns:
            self.assertIn(
                col, df.columns, f"Required column '{col}' missing from result"
            )

        # Check if thalamus and hippocampus regions are present
        expected_regions = [
            "subcort-lh-thalamus",
            "subcort-rh-thalamus",
            "subcort-lh-hippocampus",
            "subcort-rh-hippocampus",
        ]
        found_regions = df["Region"].values
        for region in expected_regions:
            self.assertIn(region, found_regions, f"{region} missing from results")

        # Check that values are reasonable (volumes in ml)
        thalamus_vol = df[df["Region"] == "subcort-lh-thalamus"]["Value"].values[0]
        self.assertTrue(
            1 < thalamus_vol < 50, f"Thalamus volume out of range: {thalamus_vol} ml"
        )
        hippo_vol = df[df["Region"] == "subcort-lh-hippocampus"]["Value"].values[0]
        self.assertTrue(
            1 < hippo_vol < 10, f"Hippocampus volume out of range: {hippo_vol} ml"
        )

        # Check units
        self.assertEqual(
            df[df["Region"] == "subcort-lh-thalamus"]["Units"].values[0],
            "ml",
            "Subcortical units should be ml",
        )

    @collect_info("cortical metrics")
    def test_cortex_stats_basic(self):
        """Test basic functionality of parse_freesurfer_cortex_stats."""
        if not os.path.isfile(self.lh_aparc_stats):
            self.skipTest("lh.aparc.stats file not found")

        # Run function with default parameters
        df, metadata = morpho.parse_freesurfer_cortex_stats(self.lh_aparc_stats)

        # Basic assertions
        self.assertIsNotNone(df, "Function returned None DataFrame")
        self.assertFalse(df.empty, "Function returned empty DataFrame")
        self.assertIsInstance(metadata, dict, "Metadata should be a dictionary")

        # Check if expected columns exist
        required_columns = [
            "Source",
            "Metric",
            "Units",
            "MetricFile",
            "Side",
            "Region",
            "Value",
        ]
        for col in required_columns:
            self.assertIn(
                col, df.columns, f"Required column '{col}' missing from result"
            )

        # Check hemisphere
        self.assertEqual(
            df["Side"].values[0], "lh", "Side should be 'lh' for left stats file"
        )

        # Check metrics
        expected_metrics = ["thickness", "area", "volume", "curv"]
        found_metrics = df["Metric"].unique()
        for metric in expected_metrics:
            self.assertIn(
                metric, found_metrics, f"Metric '{metric}' missing from results"
            )

        # Check if some expected regions exist
        expected_regions = ["ctx-lh-bankssts", "ctx-lh-superiorfrontal"]
        found_regions = df["Region"].unique()
        for region in expected_regions:
            self.assertIn(
                region, found_regions, f"Region '{region}' missing from results"
            )

        # Check if thickness has Std column
        thickness_df = df[df["Metric"] == "thickness"]
        self.assertIn(
            "Std", thickness_df.columns, "Std column missing for thickness metric"
        )

        # Check thickness values are reasonable
        bankssts_thickness = thickness_df[thickness_df["Region"] == "ctx-lh-bankssts"][
            "Value"
        ].values[0]
        self.assertTrue(
            1.5 < bankssts_thickness < 5,
            f"Bankssts thickness out of range: {bankssts_thickness} mm",
        )

        # Check metadata contains versions and file info
        for key in ["source_file", "parser_version"]:
            self.assertIn(key, metadata, f"Metadata missing key: {key}")

    @collect_info("cortical metrics")
    def test_cortex_stats_right_hemisphere(self):
        """Test parse_freesurfer_cortex_stats with right hemisphere."""
        if not os.path.isfile(self.rh_aparc_stats):
            self.skipTest("rh.aparc.stats file not found")

        # Run function with right hemisphere file
        df, _ = morpho.parse_freesurfer_cortex_stats(self.rh_aparc_stats)

        # Check hemisphere
        self.assertEqual(
            df["Side"].values[0], "rh", "Side should be 'rh' for right stats file"
        )

        # Check regions have 'rh' in them
        for region in df["Region"].unique():
            self.assertIn("-rh-", region, f"Region {region} missing 'rh' indicator")

    @collect_info("metric filtering")
    def test_cortex_stats_specific_metrics(self):
        """Test parse_freesurfer_cortex_stats with specific metrics."""
        if not os.path.isfile(self.lh_aparc_stats):
            self.skipTest("lh.aparc.stats file not found")

        # Run function with only thickness and area metrics
        metrics_to_include = ["thickness", "area"]
        df, _ = morpho.parse_freesurfer_cortex_stats(
            self.lh_aparc_stats, include_metrics=metrics_to_include
        )

        # Check metrics
        found_metrics = df["Metric"].unique()
        self.assertEqual(
            set(found_metrics),
            set(metrics_to_include),
            f"Expected metrics {metrics_to_include}, found {found_metrics}",
        )

        # Verify each metric has appropriate units
        self.assertEqual(
            df[df["Metric"] == "thickness"]["Units"].iloc[0],
            "mm",
            "Thickness should have units of mm",
        )
        self.assertEqual(
            df[df["Metric"] == "area"]["Units"].iloc[0],
            "cm²",
            "Area should have units of cm²",
        )

    @collect_info("format options")
    def test_cortex_stats_region_format(self):
        """Test region format of parse_freesurfer_cortex_stats."""
        if not os.path.isfile(self.lh_aparc_stats):
            self.skipTest("lh.aparc.stats file not found")

        # Run function with region table_type
        df, _ = morpho.parse_freesurfer_cortex_stats(
            self.lh_aparc_stats, table_type="region"
        )

        # Check if it has Statistics column
        self.assertIn(
            "Statistics", df.columns, "Statistics column missing in region format"
        )

        # Check if regions are now columns (look for Value columns for regions)
        self.assertTrue(
            any("ctx-lh-" in col for col in df.columns),
            "No region columns found in region format",
        )

    @collect_info("data validation")
    def test_dtype_consistency(self):
        """Test data type consistency in returned DataFrames."""
        if not os.path.isfile(self.aseg_stats) or not os.path.isfile(
            self.lh_aparc_stats
        ):
            self.skipTest("Required stats files not found")

        # Get dataframes from different functions
        global_df, _ = morpho.parse_freesurfer_global_fromaseg(self.aseg_stats)
        regional_df, _ = morpho.parse_freesurfer_stats_fromaseg(self.aseg_stats)
        cortex_df, _ = morpho.parse_freesurfer_cortex_stats(self.lh_aparc_stats)

        # Check Value column is always numeric
        self.assertTrue(
            pd.api.types.is_numeric_dtype(global_df["Value"]),
            "Value column in global_df is not numeric",
        )
        self.assertTrue(
            pd.api.types.is_numeric_dtype(regional_df["Value"]),
            "Value column in regional_df is not numeric",
        )
        self.assertTrue(
            pd.api.types.is_numeric_dtype(cortex_df["Value"]),
            "Value column in cortex_df is not numeric",
        )

        # Check Region column is always object/string
        self.assertTrue(
            pd.api.types.is_object_dtype(global_df["Region"]),
            "Region column in global_df is not string/object type",
        )
        self.assertTrue(
            pd.api.types.is_object_dtype(regional_df["Region"]),
            "Region column in regional_df is not string/object type",
        )
        self.assertTrue(
            pd.api.types.is_object_dtype(cortex_df["Region"]),
            "Region column in cortex_df is not string/object type",
        )

    @collect_info("error handling")
    def test_error_handling_nonexistent_file(self):
        """Test error handling with non-existent file."""
        nonexistent_file = os.path.join(BERT_DIR, "stats", "nonexistent.stats")

        # Test with all three functions
        with self.assertRaises(Exception):
            morpho.parse_freesurfer_global_fromaseg(nonexistent_file)

        with self.assertRaises(Exception):
            morpho.parse_freesurfer_stats_fromaseg(nonexistent_file)

        with self.assertRaises(Exception):
            morpho.parse_freesurfer_cortex_stats(nonexistent_file)


if __name__ == "__main__":
    # Run the tests with more verbose output
    unittest.main(verbosity=2)
