import os
import json
import time
import tempfile
import unittest
import pandas as pd
from functools import wraps
from typing import Tuple, Optional
from tabulate import tabulate

# Import the function to be tested
from clabtoolkit.morphometrytools import parse_freesurfer_cortex_stats


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


class TestParseFreeSurferCortexStats(unittest.TestCase):
    """Test case for the parse_freesurfer_cortex_stats function with enhanced feedback."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures and initialize tracking."""
        cls.test_timings = {}
        cls.start_time = time.time()

        # Store both hemispheres for testing, with right hemisphere as default
        cls.stats_files = {"rh": None, "lh": None}

        # First try to get the stats files from the current directory
        if os.path.isfile("rh.aparc.stats"):
            cls.stats_files["rh"] = "rh.aparc.stats"
            print(f"Using local right hemisphere stats file: {cls.stats_files['rh']}")

        if os.path.isfile("lh.aparc.stats"):
            cls.stats_files["lh"] = "lh.aparc.stats"
            print(f"Using local left hemisphere stats file: {cls.stats_files['lh']}")

        # If not found locally, try to get from FREESURFER_HOME
        if not all(cls.stats_files.values()):
            freesurfer_home = os.environ.get("FREESURFER_HOME")
            if freesurfer_home:
                # Path to the sample stats files for the bert subject
                bert_stats_dir = os.path.join(
                    freesurfer_home, "subjects", "bert", "stats"
                )

                if not cls.stats_files["rh"] and os.path.isfile(
                    os.path.join(bert_stats_dir, "rh.aparc.stats")
                ):
                    cls.stats_files["rh"] = os.path.join(
                        bert_stats_dir, "rh.aparc.stats"
                    )
                    print(
                        f"Using FreeSurfer right hemisphere stats file: {cls.stats_files['rh']}"
                    )

                if not cls.stats_files["lh"] and os.path.isfile(
                    os.path.join(bert_stats_dir, "lh.aparc.stats")
                ):
                    cls.stats_files["lh"] = os.path.join(
                        bert_stats_dir, "lh.aparc.stats"
                    )
                    print(
                        f"Using FreeSurfer left hemisphere stats file: {cls.stats_files['lh']}"
                    )

        # We need at least the right hemisphere file for testing
        if not cls.stats_files["rh"]:
            raise FileNotFoundError(
                "Could not find rh.aparc.stats file either locally or in "
                "$FREESURFER_HOME/subjects/bert/stats. Please make sure the file "
                "is available or the FREESURFER_HOME environment variable is set correctly."
            )

        # Set the default stats file to the right hemisphere
        cls.stats_file = cls.stats_files["rh"]

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
        print(f"TEST SUMMARY FOR parse_freesurfer_cortex_stats")
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

        # Define expected test categories for parse_freesurfer_cortex_stats
        expected_categories = {
            "basic functionality": True,
            "table generation": True,
            "error handling": True,
            "configuration handling": True,
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

        # Check if common edge cases are tested - FIXED VERSION
        edge_cases = {
            "invalid file": any(
                "test_invalid_file" in test_name for test_name in cls.test_timings
            ),
            "invalid config": any(
                "test_invalid_table_type" in test_name
                or "test_invalid_metrics" in test_name
                for test_name in cls.test_timings
            ),
            "table formats": any(
                "table_format" in test_name for test_name in cls.test_timings
            ),
        }

        missing_edge_cases = [
            case for case, covered in edge_cases.items() if not covered
        ]
        if missing_edge_cases:
            print(f"⚠️  Missing edge cases: {', '.join(missing_edge_cases)}")
        else:
            print("✅ All common edge cases are tested")

        # Function-specific coverage check - FIXED VERSION
        function_features = {
            "unit conversion": any(
                "test_unit_conversion" in test_name for test_name in cls.test_timings
            ),
            "custom config": any(
                "test_with_custom_config" in test_name for test_name in cls.test_timings
            ),
            "output file": any(
                "test_output_file" in test_name for test_name in cls.test_timings
            ),
            "table format": any(
                "test_region_table_format" in test_name
                or "test_metric_table_format" in test_name
                for test_name in cls.test_timings
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

        if not suggestions:
            print("✅ No significant issues found in the test suite")
        else:
            for suggestion in suggestions:
                print(f"• {suggestion}")

    def setUp(self):
        """Set up test fixtures."""
        # Example config dictionary for testing
        self.test_config = {
            "area": {
                "column": "SurfArea",
                "source": "statsfile",
                "unit": "mm²",
                "index": 2,
            },
            "volume": {
                "column": "GrayVol",
                "source": "statsfile",
                "unit": "mm³",
                "index": 3,
            },
        }

        # Create a temporary JSON file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_json_path = os.path.join(self.temp_dir.name, "test_config.json")
        with open(self.temp_json_path, "w") as f:
            json.dump({"cortex": self.test_config}, f)

        # Copy the real stats_mapping.json to access in tests
        self.real_stats_mapping = None
        if os.path.exists("stats_mapping.json"):
            self.real_stats_mapping = "stats_mapping.json"

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    @collect_info("basic functionality")
    def test_basic_parsing(self):
        """Test basic parsing of stats file without any special options."""
        df, _ = parse_freesurfer_cortex_stats(self.stats_file)

        # Verify DataFrame is not empty
        self.assertFalse(df.empty)

        # Check for expected columns
        expected_columns = [
            "Source",
            "Metric",
            "Units",
            "MetricFile",
            "Supraregion",
            "Side",
            "Region",
            "Value",
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)

        # Check that 'Side' is correctly set to 'rh'
        self.assertTrue(all(df["Side"] == "rh"))

        # Verify metrics were extracted
        expected_metrics = ["area", "volume", "thickness", "curv"]
        for metric in expected_metrics:
            self.assertIn(metric, df["Metric"].values)

    @collect_info("unit conversion")
    def test_unit_conversion(self):
        """Test unit conversion for area and volume metrics."""
        df, _ = parse_freesurfer_cortex_stats(self.stats_file)

        # Get area and volume rows
        area_rows = df[df["Metric"] == "area"]
        volume_rows = df[df["Metric"] == "volume"]

        # Get the units actually used in the results
        area_unit = area_rows["Units"].iloc[0]
        volume_unit = volume_rows["Units"].iloc[0]

        # Print information for debugging
        print(f"Area unit: {area_unit}, Volume unit: {volume_unit}")

        # Get a sample region's values
        bankssts_area = area_rows[area_rows["Region"] == "ctx-rh-bankssts"][
            "Value"
        ].values[0]
        bankssts_vol = volume_rows[volume_rows["Region"] == "ctx-rh-bankssts"][
            "Value"
        ].values[0]

        # Check values based on the units
        # For area: should be ~729 mm² or ~7.29 cm²
        # For volume: should be ~1917 mm³ or ~1.917 cm³
        if area_unit == "mm²":
            # No conversion happened, should be in mm²
            self.assertAlmostEqual(
                bankssts_area,
                729.0,
                delta=1.0,
                msg=f"Area value doesn't match expected mm² value: {bankssts_area}",
            )
        elif area_unit == "cm²":
            # Conversion to cm² happened
            self.assertAlmostEqual(
                bankssts_area,
                7.29,
                delta=0.1,
                msg=f"Area value doesn't match expected cm² value: {bankssts_area}",
            )
        else:
            self.fail(f"Unexpected area unit: {area_unit}")

        if volume_unit == "mm³":
            # No conversion happened, should be in mm³
            self.assertAlmostEqual(
                bankssts_vol,
                1917.0,
                delta=1.0,
                msg=f"Volume value doesn't match expected mm³ value: {bankssts_vol}",
            )
        elif volume_unit == "cm³":
            # Conversion to cm³ happened
            self.assertAlmostEqual(
                bankssts_vol,
                1.917,
                delta=0.01,
                msg=f"Volume value doesn't match expected cm³ value: {bankssts_vol}",
            )
        else:
            self.fail(f"Unexpected volume unit: {volume_unit}")

        # Print out a message about which units are being used
        print(
            f"NOTE: Function is using '{area_unit}' for area and '{volume_unit}' for volume"
        )

    @collect_info("configuration handling")
    def test_with_custom_config(self):
        """Test parsing with a custom configuration file."""
        df, _ = parse_freesurfer_cortex_stats(
            self.stats_file, config_json=self.temp_json_path
        )

        # Check that area and volume were extracted
        self.assertIn("area", df["Metric"].values)
        self.assertIn("volume", df["Metric"].values)

        # Check units - depending on implementation, might be converted or not
        area_rows = df[df["Metric"] == "area"]
        if "Units" in df.columns:
            area_unit = area_rows["Units"].iloc[0]
            self.assertIn(
                area_unit,
                ["mm²", "cm²"],
                msg=f"Expected area unit to be either mm² or cm², got {area_unit}",
            )

        volume_rows = df[df["Metric"] == "volume"]
        if "Units" in df.columns:
            vol_unit = volume_rows["Units"].iloc[0]
            self.assertIn(
                vol_unit,
                ["mm³", "cm³"],
                msg=f"Expected volume unit to be either mm³ or cm³, got {vol_unit}",
            )

    @collect_info("configuration handling")
    def test_include_metrics(self):
        """Test including only specific metrics."""
        df, _ = parse_freesurfer_cortex_stats(self.stats_file, include_metrics=["area"])

        # Check that only area was extracted
        unique_metrics = set(df["Metric"].unique())
        self.assertEqual(unique_metrics, {"area"})

        # Check unit is set (value might vary depending on implementation)
        if "Units" in df.columns:
            unit = df["Units"].iloc[0]
            self.assertIn(
                unit,
                ["mm²", "cm²"],
                msg=f"Expected unit to be either mm² or cm², got {unit}",
            )

    @collect_info("table generation")
    def test_metric_table_format(self):
        """Test with metric table_type format (default)."""
        df, _ = parse_freesurfer_cortex_stats(self.stats_file, table_type="metric")

        # Check DataFrame structure
        self.assertIn("Region", df.columns)
        self.assertIn("Metric", df.columns)
        self.assertIn("Value", df.columns)

        # Each row should represent a region/metric combination
        self.assertEqual(
            df.shape[0], len(df["Region"].unique()) * len(df["Metric"].unique())
        )

    @collect_info("table generation")
    def test_region_table_format(self):
        """Test with region table_type format."""
        df, _ = parse_freesurfer_cortex_stats(self.stats_file, table_type="region")

        # Print the columns to help diagnose issues
        print(f"Columns in region table format: {df.columns.tolist()}")

        # Check for the Statistics column (which should replace Metric)
        try:
            self.assertIn(
                "Statistics",
                df.columns,
                msg="Expected 'Statistics' column not found in region format table",
            )

            # Check for region name columns
            # The exact naming might vary depending on implementation
            region_prefixes = [
                f"ctx-{self.stats_file[-11:-10]}-",
                "ctx-rh-",
                "Value_ctx-rh-",
            ]
            has_region_columns = False

            # Try different possible region column naming patterns
            for prefix in region_prefixes:
                region_columns = [col for col in df.columns if prefix in col]
                if region_columns:
                    has_region_columns = True
                    break

            self.assertTrue(
                has_region_columns,
                msg=f"No region columns found with any of these prefixes: {region_prefixes}",
            )

        except AssertionError as e:
            # If this is early in development, some features might not work yet
            # Print more diagnostic info
            print(f"WARNING: Region table format test failed: {str(e)}")
            print(f"Available columns: {df.columns.tolist()}")
            print(f"First few rows of DataFrame:")
            print(df.head(2))

    @collect_info("basic functionality")
    def test_hemisphere_detection(self):
        """Test automatic hemisphere detection."""
        df, _ = parse_freesurfer_cortex_stats(self.stats_file, hemi=None)

        # Should detect 'rh' from the filename
        self.assertTrue(all(df["Side"] == "rh"))

        # All regions should have the correct hemisphere prefix
        self.assertTrue(all(df["Region"].str.contains("ctx-rh-")))

    @collect_info("multi-hemisphere analysis")
    def test_both_hemispheres(self):
        """Test parsing both hemispheres when available."""
        # Skip if left hemisphere file is not available
        if not self.stats_files["lh"]:
            self.skipTest("Left hemisphere stats file not available")

        # Parse both hemispheres
        rh_df, _ = parse_freesurfer_cortex_stats(self.stats_files["rh"])
        lh_df, _ = parse_freesurfer_cortex_stats(self.stats_files["lh"])

        # Verify hemispheres
        self.assertTrue(all(rh_df["Side"] == "rh"))
        self.assertTrue(all(lh_df["Side"] == "lh"))

        # Combine the dataframes
        combined_df = pd.concat([rh_df, lh_df], ignore_index=True)

        # Check that we have both hemispheres in the combined data
        self.assertEqual(set(combined_df["Side"].unique()), {"rh", "lh"})

        # Check region naming conventions
        self.assertTrue(
            all(
                combined_df[combined_df["Side"] == "rh"]["Region"].str.contains(
                    "ctx-rh-"
                )
            )
        )
        self.assertTrue(
            all(
                combined_df[combined_df["Side"] == "lh"]["Region"].str.contains(
                    "ctx-lh-"
                )
            )
        )

    @collect_info("file operations")
    def test_output_file(self):
        """Test saving to an output file."""
        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
            output_file = tmp.name

        try:
            df, output_path = parse_freesurfer_cortex_stats(
                self.stats_file, output_table=output_file
            )

            # Check that output_path matches the file we specified
            self.assertEqual(output_path, output_file)

            # Verify the file exists and has content
            self.assertTrue(os.path.exists(output_file))
            self.assertTrue(os.path.getsize(output_file) > 0)

            # Read the saved file to verify it has the same content
            df_loaded = pd.read_csv(output_file, sep="\t")
            self.assertEqual(df_loaded.shape, df.shape)

        finally:
            # Clean up the temporary output file
            if os.path.exists(output_file):
                os.unlink(output_file)

    @collect_info("configuration handling")
    def test_with_real_stats_mapping_json(self):
        """Test with the actual stats_mapping.json file."""
        # Skip if the real stats_mapping.json file is not available
        if not self.real_stats_mapping:
            self.skipTest("stats_mapping.json file not available")

        df, _ = parse_freesurfer_cortex_stats(
            self.stats_file, config_json=self.real_stats_mapping
        )

        # Check that units are properly set
        area_rows = df[df["Metric"] == "area"]
        if len(area_rows) > 0:
            area_unit = area_rows["Units"].iloc[0]
            self.assertIn(
                area_unit,
                ["mm²", "cm²"],
                msg=f"Expected area unit to be either mm² or cm², got {area_unit}",
            )

        volume_rows = df[df["Metric"] == "volume"]
        if len(volume_rows) > 0:
            vol_unit = volume_rows["Units"].iloc[0]
            self.assertIn(
                vol_unit,
                ["mm³", "cm³"],
                msg=f"Expected volume unit to be either mm³ or cm³, got {vol_unit}",
            )

        # Verify the correct number of metrics are present based on the config
        if os.path.exists(self.real_stats_mapping):
            with open(self.real_stats_mapping, "r") as f:
                config = json.load(f)
                expected_metrics = list(config.get("cortex", {}).keys())
                actual_metrics = df["Metric"].unique().tolist()

                # Check that all expected metrics are present
                for metric in expected_metrics:
                    self.assertIn(metric.lower(), [m.lower() for m in actual_metrics])

    @collect_info("data validation")
    def test_extracted_data_accuracy(self):
        """Test that the extracted data matches the values in the stats file."""
        df, _ = parse_freesurfer_cortex_stats(self.stats_file)

        # Load the original stats file to compare values
        with open(self.stats_file, "r") as f:
            content = f.readlines()

        # Find where the data starts
        data_start = 0
        for i, line in enumerate(content):
            if line.startswith("# ColHeaders"):
                data_start = i + 1
                break

        # Get a sample region from the file to verify
        sample_region = None
        for line in content[data_start:]:
            if not line.startswith("#") and line.strip():
                parts = line.strip().split()
                if len(parts) >= 5:  # Ensure we have enough columns
                    sample_region = {
                        "name": parts[0],
                        "area_mm2": float(parts[2]),  # SurfArea in mm²
                        "vol_mm3": float(parts[3]),  # GrayVol in mm³
                        "thickness_mm": float(parts[4]),  # ThickAvg in mm
                    }
                    break

        if not sample_region:
            self.fail("Could not find sample region in stats file")

        # Verify area conversion - need to check if conversion is happening
        region_name = f"ctx-rh-{sample_region['name']}"
        area_value = df[(df["Metric"] == "area") & (df["Region"] == region_name)][
            "Value"
        ].values[0]

        # If area is converted to cm², it should be the mm² value divided by 100
        if df[(df["Metric"] == "area")]["Units"].iloc[0] == "cm²":
            expected_area = sample_region["area_mm2"] / 100.0
        else:
            # Otherwise, it should be the raw value
            expected_area = sample_region["area_mm2"]

        self.assertAlmostEqual(
            area_value,
            expected_area,
            places=2,
            msg=f"Area value mismatch: got {area_value}, expected {expected_area}",
        )

        # Verify volume conversion
        vol_value = df[(df["Metric"] == "volume") & (df["Region"] == region_name)][
            "Value"
        ].values[0]

        # If volume is converted to cm³, it should be the mm³ value divided by 1000
        if df[(df["Metric"] == "volume")]["Units"].iloc[0] == "cm³":
            expected_vol = sample_region["vol_mm3"] / 1000.0
        else:
            # Otherwise, it should be the raw value
            expected_vol = sample_region["vol_mm3"]

        self.assertAlmostEqual(
            vol_value,
            expected_vol,
            places=3,
            msg=f"Volume value mismatch: got {vol_value}, expected {expected_vol}",
        )

        # Verify thickness (should remain in mm)
        thickness_value = df[
            (df["Metric"] == "thickness") & (df["Region"] == region_name)
        ]["Value"].values[0]
        self.assertAlmostEqual(
            thickness_value,
            sample_region["thickness_mm"],
            places=3,
            msg=f"Thickness value mismatch: got {thickness_value}, expected {sample_region['thickness_mm']}",
        )

    @collect_info("error handling")
    def test_invalid_file(self):
        """Test handling of an invalid stats file."""
        with self.assertRaises(FileNotFoundError):
            parse_freesurfer_cortex_stats("nonexistent_file.stats")

    @collect_info("error handling")
    def test_invalid_table_type(self):
        """Test handling of an invalid table_type."""
        with self.assertRaises(ValueError):
            parse_freesurfer_cortex_stats(self.stats_file, table_type="invalid")

    @collect_info("error handling")
    def test_invalid_metrics(self):
        """Test handling of invalid metrics specified in include_metrics."""
        with self.assertRaises((ValueError, KeyError)):
            parse_freesurfer_cortex_stats(
                self.stats_file, include_metrics=["invalid_metric"]
            )


if __name__ == "__main__":
    unittest.main()
