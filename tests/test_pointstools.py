"""Tests for clabtoolkit.pointstools (the plotting method is not covered)."""

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import pytest

import clabtoolkit.pointstools as cltpts
from clabtoolkit.pointstools import PointCloud

# Six points, one per region
POINTS = np.array(
    [
        [-30.0, 10.0, 5.0],
        [-10.0, -20.0, 15.0],
        [0.0, 0.0, 0.0],
        [10.0, 20.0, -5.0],
        [30.0, -10.0, 25.0],
        [40.0, 5.0, -15.0],
    ]
)
COLORS = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff"]
NAMES = ["a", "b", "c", "d", "e", "f"]

# A rotation of 90 degrees around Z, a scaling of 2 along Z and a translation
AFFINE = np.array(
    [[0.0, -1.0, 0.0, 5.0], [1.0, 0.0, 0.0, -3.0], [0.0, 0.0, 2.0, 7.0], [0, 0, 0, 1.0]]
)


####################################################################################################
# Fixtures and helpers
####################################################################################################
@pytest.fixture(autouse=True)
def _quiet():
    """Silence the warnings of the functions."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture
def cloud():
    """Six regions (one per point) and a scalar map with the values 0, 10, ..., 50."""
    pc = PointCloud(
        POINTS.copy(), region_colors=COLORS, region_names=NAMES, name="regions"
    )
    pc.add_point_data(np.arange(6) * 10.0, name="val")
    return pc


@pytest.fixture
def plain():
    """Six points sharing one region, without any map but the default one."""
    return PointCloud(POINTS.copy(), name="plain")


@pytest.fixture
def lut_file(tmp_path):
    path = tmp_path / "hemispheres.lut"
    path.write_text(
        "1  Left-Hemisphere   220  20  60  0\n2  Right-Hemisphere   30 144 255  0\n"
    )
    return path


def quiet_call(function, *args, **kwargs):
    """Call a function that prints progress messages and return what it returns."""
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def apply(points, affine):
    return points @ affine[:3, :3].T + affine[:3, 3]


####################################################################################################
# Creation
####################################################################################################
class TestCreation:
    def test_empty(self):
        pc = PointCloud()
        assert pc.coords is None and len(pc) == 0 and pc.name == "default"
        assert pc.point_data == {} and np.array_equal(pc.affine, np.eye(4))
        assert list(pc.colortables) == ["default"]
        assert repr(pc) == "PointCloud(name='default', n_points=0, n_attributes=0)"

    def test_one_region_for_all_the_points(self, plain):
        assert len(plain) == 6 and np.array_equal(plain.coords, POINTS)
        assert plain.point_data["default"].tolist() == [1] * 6
        table = plain.colortables["default"]
        assert table["names"] == ["region_1"]
        # The default gray #BFBDBD, as 0-1 values, an opaque alpha and the region value
        assert table["color_table"][0, :3] == pytest.approx(
            np.array([191, 189, 189]) / 255
        )
        assert table["color_table"][0, 3:].tolist() == [1.0, 1.0]
        assert repr(plain) == "PointCloud(name='plain', n_points=6, n_attributes=1)"

    def test_one_region_per_point(self, cloud):
        assert cloud.point_data["default"].tolist() == [1, 2, 3, 4, 5, 6]
        table = cloud.colortables["default"]
        assert table["names"] == NAMES
        assert table["color_table"][:, 4].tolist() == [1, 2, 3, 4, 5, 6]
        assert table["color_table"][:2, :3].tolist() == [[1, 0, 0], [0, 1, 0]]

    @pytest.mark.parametrize(
        "color",
        ["#ff0000", "red", (255, 0, 0), [255, 0, 0], np.array([255, 0, 0])],
    )
    def test_a_single_color_in_any_form(self, color):
        pc = PointCloud(POINTS, region_colors=color)
        assert pc.colortables["default"]["color_table"][0, :3].tolist() == [1, 0, 0]
        assert pc.point_data["default"].tolist() == [1] * 6

    def test_alpha(self):
        pc = PointCloud(POINTS, region_colors="#ff0000", alpha=0.25)
        assert pc.colortables["default"]["color_table"][0, 3] == 0.25
        assert (
            PointCloud(POINTS, alpha=1).colortables["default"]["color_table"][0, 3]
            == 1.0
        )

    def test_a_single_name_can_be_a_string(self):
        pc = PointCloud(POINTS, region_names="only")
        assert pc.colortables["default"]["names"] == ["only"]

    def test_affine_and_name(self):
        pc = PointCloud(POINTS, affine=np.diag([2.0, 2.0, 2.0, 1.0]), name="mine")
        assert pc.name == "mine" and pc.affine[0, 0] == 2.0

    def test_points_from_a_dataframe(self):
        frame = pd.DataFrame(POINTS, columns=["X", "Y", "Z"])
        frame["ignored"] = 1.0
        pc = PointCloud(frame)
        assert np.array_equal(pc.coords, POINTS) and list(pc.point_data) == ["default"]

    def test_two_dimensional_points_get_a_zero_z(self):
        pc = PointCloud(POINTS[:, :2])
        assert pc.coords.shape == (6, 3) and (pc.coords[:, 2] == 0).all()
        assert np.array_equal(pc.coords[:, :2], POINTS[:, :2])

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"points": np.zeros((4, 4))}, "shape"),
            ({"points": np.zeros(3)}, "shape"),
            ({"points": [[1, 2, 3]]}, "numpy array or pandas DataFrame"),
            ({"points": pd.DataFrame({"X": [1.0], "Y": [2.0]})}, "'X', 'Y', and 'Z'"),
            ({"points": POINTS, "alpha": 2}, "Alpha"),
            ({"points": POINTS, "alpha": -0.1}, "Alpha"),
            (
                {
                    "points": POINTS,
                    "region_colors": ["red", "blue"],
                    "region_names": ["a"],
                },
                "number of region names",
            ),
            (
                {"points": POINTS, "region_colors": ["red", "blue"]},
                "must be 1 or equal",
            ),
        ],
    )
    def test_errors(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            PointCloud(**kwargs)

    def test_copy_is_independent(self, cloud):
        clone = cloud.copy()
        clone.coords[0, 0] = 999
        clone.point_data["val"][0] = 999
        clone.colortables["default"]["names"][0] = "changed"
        clone.name = "clone"
        assert cloud.coords[0, 0] == -30 and cloud.point_data["val"][0] == 0
        assert cloud.colortables["default"]["names"][0] == "a"
        assert cloud.name == "regions"


####################################################################################################
# Point data and maps
####################################################################################################
class TestPointData:
    def test_add_point_data(self, plain):
        plain.add_point_data(np.arange(6), name="index")
        assert plain.point_data["index"].tolist() == list(range(6))
        assert plain.list_maps() == ["default", "index"]

    def test_dtype(self, plain):
        plain.add_point_data(np.arange(6), name="labels", dtype=np.int16)
        assert plain.point_data["labels"].dtype == np.int16

    def test_the_data_is_copied(self, plain):
        values = np.arange(6.0)
        plain.add_point_data(values, name="values")
        values[0] = 99
        assert plain.point_data["values"][0] == 0

    def test_lists_become_arrays(self, plain):
        plain.add_point_data([1, 2, 3, 4, 5, 6], name="listed")
        assert isinstance(plain.point_data["listed"], np.ndarray)

    def test_a_map_can_be_replaced(self, plain):
        plain.add_point_data(np.zeros(6), name="values")
        plain.add_point_data(np.ones(6), name="values")
        assert plain.point_data["values"].tolist() == [1.0] * 6

    def test_errors(self, plain):
        with pytest.raises(ValueError, match=r"Data length \(5\)"):
            plain.add_point_data(np.zeros(5), name="short")
        with pytest.raises(ValueError, match="empty point cloud"):
            PointCloud().add_point_data(np.zeros(3), name="x")

    def test_list_maps(self, cloud):
        assert cloud.list_maps() == ["default", "val"]
        assert PointCloud().list_maps() is None  # No maps at all


####################################################################################################
# Geometry
####################################################################################################
class TestGeometry:
    def test_bounds(self, cloud):
        assert cloud.get_bounds() == {"x": (-30, 40), "y": (-20, 20), "z": (-15, 25)}

    def test_centroid(self, cloud):
        assert cloud.get_centroid() == pytest.approx(POINTS.mean(axis=0))
        assert cloud.get_centroid() == pytest.approx([20 / 3, 5 / 6, 25 / 6])

    def test_bounds_and_centroid_of_an_empty_cloud(self):
        assert PointCloud().get_bounds() == {"x": (0, 0), "y": (0, 0), "z": (0, 0)}
        assert PointCloud().get_centroid().tolist() == [0, 0, 0]

    def test_apply_affine(self, cloud):
        assert cloud.apply_affine(AFFINE) is None
        assert cloud.coords == pytest.approx(apply(POINTS, AFFINE))
        assert np.array_equal(cloud.affine, AFFINE)  # The transformation is accumulated
        assert cloud.point_data["val"].tolist() == [0, 10, 20, 30, 40, 50]

    def test_the_inverse_undoes_the_affine(self, cloud):
        cloud.apply_affine(AFFINE)
        cloud.apply_affine(AFFINE, inverse=True)
        assert cloud.coords == pytest.approx(POINTS)
        assert cloud.affine == pytest.approx(np.eye(4))

    def test_affines_accumulate_in_order(self, cloud):
        scaling = np.diag([2.0, 2.0, 2.0, 1.0])
        cloud.apply_affine(AFFINE)
        cloud.apply_affine(scaling)
        assert cloud.affine == pytest.approx(scaling @ AFFINE)
        assert cloud.coords == pytest.approx(apply(apply(POINTS, AFFINE), scaling))

    def test_a_transformed_copy(self, cloud):
        moved = cloud.apply_affine(AFFINE, inplace=False)
        assert isinstance(moved, PointCloud) and moved is not cloud
        assert moved.coords == pytest.approx(apply(POINTS, AFFINE))
        assert np.array_equal(moved.affine, AFFINE)
        assert np.array_equal(cloud.coords, POINTS) and np.array_equal(
            cloud.affine, np.eye(4)
        )
        assert moved.point_data["val"].tolist() == [0, 10, 20, 30, 40, 50]

    def test_affine_of_an_empty_cloud(self):
        assert PointCloud().apply_affine(AFFINE) is None
        assert isinstance(PointCloud().apply_affine(AFFINE, inplace=False), PointCloud)

    def test_affine_needs_a_4x4_matrix(self, cloud):
        with pytest.raises(ValueError, match="4x4"):
            cloud.apply_affine(np.eye(3))


class TestFilterByBounds:
    def test_one_axis_is_inclusive(self, cloud):
        kept = cloud.filter_by_bounds(x_range=(-10, 30), inplace=False)
        assert kept.coords[:, 0].tolist() == [-10, 0, 10, 30]  # Both limits are kept

    def test_maps_follow_the_points(self, cloud):
        kept = cloud.filter_by_bounds(x_range=(-10, 30), inplace=False)
        assert kept.point_data["val"].tolist() == [10, 20, 30, 40]
        assert kept.point_data["default"].tolist() == [2, 3, 4, 5]

    def test_several_axes(self, cloud):
        kept = cloud.filter_by_bounds(
            x_range=(-10, 30), y_range=(-20, 10), z_range=(0, 30), inplace=False
        )
        assert kept.coords.tolist() == [[-10, -20, 15], [0, 0, 0], [30, -10, 25]]

    def test_open_limits(self, cloud):
        left = cloud.filter_by_bounds(x_range=(-np.inf, 0), inplace=False)
        assert left.coords[:, 0].tolist() == [-30, -10, 0]

    def test_a_new_cloud_leaves_the_source_alone(self, cloud):
        kept = cloud.filter_by_bounds(z_range=(0, 30), inplace=False)
        assert kept is not cloud and len(kept) == 4 and len(cloud) == 6

    def test_in_place(self, cloud):
        assert cloud.filter_by_bounds(z_range=(0, 30)) is None
        assert len(cloud) == 4 and cloud.point_data["val"].tolist() == [0, 10, 20, 40]
        # The color table is not touched, even if some regions disappear
        assert len(cloud.colortables["default"]["names"]) == 6

    def test_no_limits_keeps_everything(self, cloud):
        assert len(cloud.filter_by_bounds(inplace=False)) == 6

    def test_nothing_inside_the_bounds(self, cloud):
        empty = cloud.filter_by_bounds(x_range=(1000, 2000), inplace=False)
        assert len(empty) == 0 and empty.point_data["val"].size == 0

    def test_empty_cloud(self):
        assert PointCloud().filter_by_bounds(x_range=(0, 1)) is None
        assert isinstance(
            PointCloud().filter_by_bounds(x_range=(0, 1), inplace=False), PointCloud
        )


####################################################################################################
# Filtering by condition
####################################################################################################
class TestFilter:
    @pytest.mark.parametrize(
        "condition, expected",
        [
            ("Z > 0", [0, 1, 4]),
            ("Z >= 0", [0, 1, 2, 4]),
            ("X <= 0", [0, 1, 2]),
            ("Y != 0", [0, 1, 3, 4, 5]),
            ("val > 20", [3, 4, 5]),
            ("val == 0", [0]),
            ("10 <= val <= 30", [1, 2, 3]),
            ("-10 < X < 30", [2, 3]),
        ],
    )
    def test_conditions(self, cloud, condition, expected):
        kept = quiet_call(cloud.filter, condition, inplace=False)
        assert np.array_equal(kept.coords, POINTS[expected])
        assert kept.point_data["val"].tolist() == [10.0 * i for i in expected]

    def test_a_new_cloud_is_returned_by_default_in_place_otherwise(self, cloud):
        kept = quiet_call(cloud.filter, "val >= 30", inplace=False)
        assert kept is not cloud and kept.name == "regions_filtered"
        assert len(cloud) == 6  # The source is untouched
        assert quiet_call(cloud.filter, "val >= 30") is None
        assert len(cloud) == 3 and cloud.name == "regions"

    def test_the_affine_is_kept(self, cloud):
        cloud.apply_affine(AFFINE)
        kept = quiet_call(cloud.filter, "val >= 30", inplace=False)
        assert np.array_equal(kept.affine, AFFINE) and kept.affine is not cloud.affine

    def test_the_color_table_keeps_the_regions_in_use(self, cloud):
        kept = quiet_call(cloud.filter, "val >= 30", inplace=False)
        table = kept.colortables["default"]
        assert kept.point_data["default"].tolist() == [4, 5, 6]
        assert table["names"] == ["d", "e", "f"]
        assert table["color_table"][:, 4].tolist() == [4, 5, 6]

    def test_other_color_tables(self, cloud):
        cloud.add_point_data(np.array([1, 1, 2, 2, 3, 3]), name="group")
        cloud.colortables["group"] = {
            "names": ["g1", "g2", "g3"],
            "color_table": np.array(
                [[1, 0, 0, 1, 1], [0, 1, 0, 1, 2], [0, 0, 1, 1, 3.0]]
            ),
            "lookup_table": None,
        }
        cloud.colortables["unrelated"] = {
            "names": ["u"],
            "color_table": np.array([[1, 1, 1, 1, 1.0]]),
            "lookup_table": None,
        }
        kept = quiet_call(cloud.filter, "group >= 2", inplace=False)
        assert kept.colortables["group"]["names"] == ["g2", "g3"]
        assert kept.colortables["group"]["color_table"][:, 4].tolist() == [2, 3]
        assert kept.colortables["unrelated"]["names"] == ["u"]  # Not tied to a map
        assert kept.colortables["default"]["names"] == ["c", "d", "e", "f"]

    def test_nothing_matches(self, cloud):
        with pytest.raises(ValueError, match="No points match"):
            quiet_call(cloud.filter, "val > 1000")
        assert len(cloud) == 6

    def test_unknown_attribute(self, cloud):
        with pytest.raises(
            ValueError,
            match=r"Attribute 'nope' not found.*\['X', 'Y', 'Z', 'default', 'val'\]",
        ):
            cloud.filter("nope > 1")

    def test_empty_cloud(self):
        with pytest.raises(ValueError, match="empty point cloud"):
            PointCloud().filter("X > 1")


####################################################################################################
# Tables
####################################################################################################
class TestDataFrame:
    def test_coordinates_and_maps(self, cloud):
        frame = cloud.to_dataframe()
        assert list(frame.columns) == ["X", "Y", "Z", "default", "val"]
        assert frame[["X", "Y", "Z"]].to_numpy() == pytest.approx(POINTS)
        assert frame["val"].tolist() == [0, 10, 20, 30, 40, 50]

    def test_coordinates_only(self, cloud):
        assert list(cloud.to_dataframe(include_data=False).columns) == ["X", "Y", "Z"]

    def test_with_the_colortable(self, cloud):
        frame = cloud.to_dataframe(include_colortable=True)
        assert list(frame.columns) == ["index", "name", "color", "X", "Y", "Z", "val"]
        assert frame["index"].tolist() == [1, 2, 3, 4, 5, 6]
        assert frame["name"].tolist() == NAMES
        assert frame["color"].tolist() == COLORS

    def test_with_the_colortable_and_no_other_data(self, cloud):
        frame = cloud.to_dataframe(include_colortable=True, include_data=False)
        assert list(frame.columns) == ["index", "name", "color", "X", "Y", "Z"]

    def test_values_missing_from_the_colortable(self, cloud):
        cloud.point_data["default"][1] = 99
        row = cloud.to_dataframe(include_colortable=True).iloc[1]
        assert (row["index"], row["name"], row["color"]) == (
            99,
            "auto-roi-000099",
            "#000000",
        )

    def test_another_colortable(self, cloud, lut_file):
        cloud.add_point_data(np.where(cloud.coords[:, 0] < 0, 1, 2), name="hemisphere")
        cloud.load_colortable(lut_file, map_name="hemisphere")
        frame = cloud.to_dataframe(
            include_colortable=True, colortable_name="hemisphere"
        )
        assert (
            frame["name"].tolist()
            == ["Left-Hemisphere"] * 2 + ["Right-Hemisphere"] * 4  # X < 0 for 2 points
        )
        assert frame["color"].tolist()[0] == "#dc143c"
        assert "hemisphere" not in frame.columns and "default" in frame.columns

    def test_errors(self, cloud):
        with pytest.raises(ValueError, match="Colortable 'nope' not found"):
            cloud.to_dataframe(include_colortable=True, colortable_name="nope")
        cloud.colortables["without_data"] = cloud.colortables["default"]
        with pytest.raises(ValueError, match="Point data for 'without_data' not found"):
            cloud.to_dataframe(include_colortable=True, colortable_name="without_data")

    def test_empty_cloud(self):
        assert list(PointCloud().to_dataframe().columns) == ["X", "Y", "Z"]
        frame = PointCloud().to_dataframe(include_colortable=True)
        assert (
            list(frame.columns) == ["index", "name", "color", "X", "Y", "Z"]
            and frame.empty
        )


####################################################################################################
# Files
####################################################################################################
class TestSaveAndLoad:
    @pytest.mark.parametrize("as_str", [True, False])
    def test_npy_round_trip(self, tmp_path, cloud, as_str):
        file = tmp_path / "cloud.npy"
        cloud.apply_affine(AFFINE)
        cloud.save(str(file) if as_str else file)
        assert [p.name for p in tmp_path.iterdir()] == ["cloud.npy"]  # No .npz appended
        loaded = PointCloud.load(str(file) if as_str else file)
        assert loaded.name == "regions" and len(loaded) == 6
        assert loaded.coords == pytest.approx(cloud.coords)
        assert loaded.affine == pytest.approx(AFFINE)
        assert set(loaded.point_data) == {"default", "val"}
        assert loaded.point_data["val"].tolist() == [0, 10, 20, 30, 40, 50]

    @pytest.mark.parametrize("fmt, separator", [("csv", ","), ("txt", "\t")])
    def test_text_files(self, tmp_path, cloud, fmt, separator):
        file = tmp_path / f"cloud.{fmt}"
        cloud.save(file, format=fmt)
        lines = file.read_text().splitlines()
        assert lines[0] == separator.join(["X", "Y", "Z", "default", "val"])
        assert len(lines) == 7
        loaded = PointCloud.load(file, format=fmt)
        assert loaded.coords == pytest.approx(POINTS)
        assert set(loaded.point_data) == {"default", "val"}
        assert loaded.point_data["val"].tolist() == [0, 10, 20, 30, 40, 50]

    def test_text_files_can_include_the_colortable(self, tmp_path, cloud):
        cloud.save(tmp_path / "cloud.csv", format="csv", include_colortable=True)
        header = (tmp_path / "cloud.csv").read_text().splitlines()[0]
        assert header == "index,name,color,X,Y,Z,val"

    def test_a_table_written_by_another_program(self, tmp_path):
        table = pd.DataFrame(
            {
                "X": [10.0, -12.5, 3.2],
                "Y": [5.1, 8.0, -20.4],
                "Z": [0.0, 14.3, 7.7],
                "fa": [0.45, 0.61, 0.38],
            }
        )
        table.to_csv(tmp_path / "seeds.txt", sep="\t", index=False)
        loaded = PointCloud.load(tmp_path / "seeds.txt", format="txt")
        assert len(loaded) == 3 and loaded.point_data["fa"].tolist() == [
            0.45,
            0.61,
            0.38,
        ]
        assert loaded.coords[1].tolist() == [-12.5, 8.0, 14.3]

    @pytest.mark.parametrize("fmt", ["csv", "txt"])
    def test_the_colortable_written_with_the_table_is_restored(
        self, tmp_path, cloud, fmt
    ):
        cloud.save(tmp_path / f"cloud.{fmt}", format=fmt, include_colortable=True)
        loaded = PointCloud.load(tmp_path / f"cloud.{fmt}", format=fmt)
        table = loaded.colortables["default"]
        assert table["names"] == NAMES
        assert table["color_table"][:, 4].tolist() == [1, 2, 3, 4, 5, 6]
        assert table["color_table"][:, :3] == pytest.approx(
            cloud.colortables["default"]["color_table"][:, :3]
        )
        assert loaded.point_data["default"].tolist() == [1, 2, 3, 4, 5, 6]
        # The columns of the table are not kept as maps
        assert set(loaded.point_data) == {"default", "val"}

    def test_a_reloaded_cloud_behaves_like_the_original(self, tmp_path, cloud):
        cloud.save(tmp_path / "cloud.csv", format="csv", include_colortable=True)
        loaded = PointCloud.load(tmp_path / "cloud.csv", format="csv")
        assert np.array_equal(
            loaded.get_pointwise_colors(), cloud.get_pointwise_colors()
        )
        original = cloud.to_dataframe(include_colortable=True)
        reloaded = loaded.to_dataframe(include_colortable=True)
        pd.testing.assert_frame_equal(reloaded, original, check_dtype=False)
        quiet_call(loaded.get_info)  # Used to fail with the string columns

    def test_only_the_regions_in_use_are_restored(self, tmp_path, cloud):
        kept = quiet_call(cloud.filter, "val >= 30", inplace=False)
        kept.save(tmp_path / "kept.csv", format="csv", include_colortable=True)
        loaded = PointCloud.load(tmp_path / "kept.csv", format="csv")
        assert loaded.colortables["default"]["names"] == ["d", "e", "f"]
        assert loaded.point_data["default"].tolist() == [4, 5, 6]

    def test_a_table_of_another_map_keeps_its_columns_as_maps(
        self, tmp_path, cloud, lut_file
    ):
        cloud.add_point_data(np.where(cloud.coords[:, 0] < 0, 1, 2), name="hemisphere")
        cloud.load_colortable(lut_file, map_name="hemisphere")
        cloud.save(
            tmp_path / "hemi.csv",
            format="csv",
            include_colortable=True,
            colortable_name="hemisphere",
        )
        loaded = PointCloud.load(tmp_path / "hemi.csv", format="csv")
        # The file does not say which map the table describes: nothing is restored
        assert {"index", "name", "color", "default", "val"} <= set(loaded.point_data)
        assert loaded.colortables["default"]["names"] == ["region_1"]
        quiet_call(loaded.get_info)  # The string maps do not break it either

    def test_errors(self, tmp_path, cloud):
        with pytest.raises(ValueError, match="Cannot save an empty"):
            PointCloud().save(tmp_path / "empty.npy")
        with pytest.raises(ValueError, match="Unsupported format"):
            cloud.save(tmp_path / "cloud.xyz", format="xyz")
        cloud.save(tmp_path / "cloud.npy")
        with pytest.raises(ValueError, match="Unsupported format"):
            PointCloud.load(tmp_path / "cloud.npy", format="xyz")
        with pytest.raises(FileNotFoundError):
            PointCloud.load(tmp_path / "missing.npy")
        (tmp_path / "other.csv").write_text("a,b\n1,2\n")
        with pytest.raises(ValueError, match="X, Y, Z"):
            PointCloud.load(tmp_path / "other.csv", format="csv")


####################################################################################################
# Colortables
####################################################################################################
class TestLoadColortable:
    @pytest.fixture
    def hemispheres(self, cloud):
        cloud.add_point_data(np.where(cloud.coords[:, 0] < 0, 1, 2), name="hemisphere")
        return cloud

    @pytest.mark.parametrize("as_str", [True, False])
    def test_names_colors_and_values(self, hemispheres, lut_file, as_str):
        hemispheres.load_colortable(
            str(lut_file) if as_str else lut_file, map_name="hemisphere"
        )
        table = hemispheres.colortables["hemisphere"]
        assert table["names"] == ["Left-Hemisphere", "Right-Hemisphere"]
        assert table["color_table"][:, :3] == pytest.approx(
            np.array([[220, 20, 60], [30, 144, 255]]) / 255
        )
        assert table["color_table"][:, 3].tolist() == [1.0, 1.0]
        assert table["color_table"][:, 4].tolist() == [1, 2]  # The values of the map

    def test_scalar_opacity(self, hemispheres, lut_file):
        hemispheres.load_colortable(lut_file, map_name="hemisphere", opacity=0.3)
        assert hemispheres.colortables["hemisphere"]["color_table"][:, 3].tolist() == [
            0.3,
            0.3,
        ]

    def test_opacity_per_entry(self, hemispheres, lut_file):
        hemispheres.load_colortable(
            lut_file, map_name="hemisphere", opacity=np.array([1.0, 0.3])
        )
        assert hemispheres.colortables["hemisphere"]["color_table"][:, 3].tolist() == [
            1.0,
            0.3,
        ]

    def test_opacity_of_the_wrong_length_uses_its_first_value(
        self, hemispheres, lut_file
    ):
        hemispheres.load_colortable(
            lut_file, map_name="hemisphere", opacity=np.array([0.5])
        )
        assert hemispheres.colortables["hemisphere"]["color_table"][:, 3].tolist() == [
            0.5,
            0.5,
        ]

    def test_colors_come_from_the_new_table(self, hemispheres, lut_file):
        hemispheres.load_colortable(lut_file, map_name="hemisphere")
        colors = hemispheres.get_pointwise_colors("hemisphere")
        assert colors[0].tolist() == [220, 20, 60, 255]
        assert colors[-1].tolist() == [30, 144, 255, 255]

    def test_a_table_for_a_map_that_does_not_exist_yet(self, hemispheres, lut_file):
        hemispheres.load_colortable(lut_file, map_name="future")
        assert hemispheres.colortables["future"]["names"] == [
            "Left-Hemisphere",
            "Right-Hemisphere",
        ]

    def test_errors(self, tmp_path, hemispheres, lut_file):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            hemispheres.load_colortable(tmp_path / "missing.lut", map_name="hemisphere")
        with pytest.raises(ValueError, match="Unsupported lut_type"):
            hemispheres.load_colortable(lut_file, map_name="hemisphere", lut_type="tsv")
        three = tmp_path / "three.lut"
        three.write_text("1 a 1 2 3 0\n2 b 4 5 6 0\n3 c 7 8 9 0\n")
        with pytest.raises(ValueError, match="does not cover all IDs"):
            hemispheres.load_colortable(three, map_name="hemisphere")


####################################################################################################
# Appending and merging
####################################################################################################
class TestAppend:
    @pytest.fixture
    def pair(self):
        first = PointCloud(POINTS[:4].copy(), name="first")
        first.add_point_data(np.arange(4.0), name="val")
        second = PointCloud(POINTS[4:].copy(), name="second")
        second.add_point_data(np.arange(2.0) + 100, name="other")
        return first, second

    def test_append_returns_a_new_cloud(self, pair):
        first, second = pair
        joined = first.append(second, inplace=False)
        assert isinstance(joined, PointCloud) and joined.name == "first"
        assert np.array_equal(joined.coords, POINTS)
        assert (len(first), len(second)) == (4, 2)

    def test_append_in_place(self, pair):
        first, second = pair
        assert first.append(second) is None
        assert len(first) == 6 and np.array_equal(first.coords, POINTS)

    def test_the_plus_operator(self, pair):
        first, second = pair
        joined = first + second
        assert np.array_equal(joined.coords, POINTS) and len(first) == 4

    def test_maps_that_are_not_in_both_are_filled(self, pair):
        joined = pair[0].append(pair[1], inplace=False)
        assert set(joined.point_data) == {"default", "val", "other"}
        assert joined.point_data["val"][:4].tolist() == [0, 1, 2, 3]
        assert np.isnan(joined.point_data["val"][4:]).all()
        assert np.isnan(joined.point_data["other"][:4]).all()
        assert joined.point_data["other"][4:].tolist() == [100, 101]

    def test_fill_value(self, pair):
        joined = pair[0].append(pair[1], inplace=False, fill_value=-1.0)
        assert joined.point_data["val"].tolist() == [0, 1, 2, 3, -1, -1]
        assert joined.point_data["other"].tolist() == [-1, -1, -1, -1, 100, 101]

    def test_default_map_of_single_region_clouds(self, pair):
        joined = pair[0].append(pair[1], inplace=False)
        assert joined.point_data["default"].tolist() == [1] * 6

    def test_appending_to_an_empty_cloud(self, pair):
        empty = PointCloud()
        empty.append(pair[0])
        assert len(empty) == 4 and set(empty.point_data) == {"default", "val"}
        assert len(PointCloud().append(pair[0], inplace=False)) == 4

    def test_appending_an_empty_cloud(self, pair):
        assert pair[0].append(PointCloud()) is None and len(pair[0]) == 4
        assert len(pair[0].append(PointCloud(), inplace=False)) == 4

    def test_different_affines_warn_and_keep_the_first(self, pair):
        moved = pair[1].apply_affine(AFFINE, inplace=False)
        with pytest.warns(UserWarning, match="Affine matrices differ"):
            joined = pair[0].append(
                moved, inplace=False, handle_colortable_conflicts="skip"
            )
        assert np.array_equal(joined.affine, np.eye(4))

    def test_conflicting_color_tables_warn_by_default(self, pair):
        with pytest.warns(UserWarning, match="Keeping colortable from first"):
            joined = pair[0].append(pair[1], inplace=False)
        assert list(joined.colortables) == ["default"]

    @pytest.mark.parametrize(
        "mode, tables",
        [
            ("skip", ["default"]),
            ("overwrite", ["default"]),
            ("rename", ["default", "default_2"]),
        ],
    )
    def test_other_ways_of_handling_conflicting_color_tables(self, pair, mode, tables):
        first, second = pair
        second.colortables["default"]["names"] = ["from_second"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            joined = first.append(
                second, inplace=False, handle_colortable_conflicts=mode
            )
        assert sorted(joined.colortables) == sorted(tables)
        expected_first = "from_second" if mode == "overwrite" else "region_1"
        assert joined.colortables["default"]["names"] == [expected_first]
        if mode == "rename":
            assert joined.colortables["default_2"]["names"] == ["from_second"]

    def test_new_color_tables_are_added(self, pair):
        first, second = pair
        second.colortables["extra"] = second.colortables["default"]
        joined = first.append(second, inplace=False, handle_colortable_conflicts="skip")
        assert "extra" in joined.colortables

    def test_errors(self, pair):
        with pytest.raises(ValueError, match="Can only append another PointCloud"):
            pair[0].append("not a cloud")
        with pytest.raises(ValueError, match="two empty"):
            PointCloud().append(PointCloud())
        with pytest.raises(ValueError, match="Unknown handle_colortable_conflicts"):
            pair[0].append(pair[1], inplace=False, handle_colortable_conflicts="nope")


class TestMerge:
    @pytest.fixture
    def pair(self):
        first = PointCloud(POINTS[:4].copy(), name="left")
        first.add_point_data(np.arange(4.0), name="val")
        second = PointCloud(POINTS[4:].copy(), name="right")
        return first, second

    def test_points_and_labels(self, pair):
        merged = cltpts.merge_pointclouds(list(pair))
        assert isinstance(merged, PointCloud) and np.array_equal(merged.coords, POINTS)
        assert merged.point_data["point_id"].tolist() == [1, 1, 1, 1, 2, 2]
        table = merged.colortables["point_id"]
        assert table["names"] == ["left", "right"]  # Named after the point clouds
        assert table["color_table"][:, 4].tolist() == [1, 2]

    def test_other_maps_are_kept_and_filled(self, pair):
        merged = cltpts.merge_pointclouds(list(pair))
        assert merged.point_data["val"][:4].tolist() == [0, 1, 2, 3]
        assert np.isnan(merged.point_data["val"][4:]).all()

    def test_the_sources_are_untouched(self, pair):
        cltpts.merge_pointclouds(list(pair))
        assert (len(pair[0]), len(pair[1])) == (4, 2)
        assert "point_id" not in pair[0].point_data

    def test_repeated_names_are_replaced(self, pair):
        pair[1].name = "left"
        merged = cltpts.merge_pointclouds(list(pair))
        assert merged.colortables["point_id"]["names"] == [
            "pointcloud_1",
            "pointcloud_2",
        ]

    def test_map_name(self, pair):
        merged = cltpts.merge_pointclouds(list(pair), map_name="source")
        assert "source" in merged.point_data and "source" in merged.colortables
        assert "point_id" not in merged.point_data

    def test_given_color_table(self, pair):
        table = {
            "names": ["L", "R"],
            "color_table": np.array([[1, 0, 0, 1, 10], [0, 0, 1, 1, 20.0]]),
        }
        merged = cltpts.merge_pointclouds(list(pair), color_table=table)
        assert np.unique(merged.point_data["point_id"]).tolist() == [10, 20]
        assert merged.colortables["point_id"]["names"] == ["L", "R"]

    def test_from_files(self, tmp_path, pair):
        pair[0].save(tmp_path / "left.npy")
        pair[1].save(tmp_path / "right.npy")
        merged = cltpts.merge_pointclouds(
            [tmp_path / "left.npy", str(tmp_path / "right.npy")]
        )
        assert len(merged) == 6 and merged.colortables["point_id"]["names"] == [
            "left",
            "right",
        ]

    def test_a_single_cloud_and_no_clouds(self, pair):
        merged = cltpts.merge_pointclouds([pair[0]])
        assert len(merged) == 4 and merged.point_data["point_id"].tolist() == [1] * 4
        assert cltpts.merge_pointclouds([]) is None

    def test_errors(self, pair):
        first, second = pair
        with pytest.raises(TypeError, match="must be a list"):
            cltpts.merge_pointclouds(first)
        with pytest.raises(TypeError, match="PointCloud objects"):
            cltpts.merge_pointclouds([first, 3])
        with pytest.raises(TypeError, match="dictionary"):
            cltpts.merge_pointclouds([first, second], color_table=[1])
        with pytest.raises(ValueError, match="must contain the keys"):
            cltpts.merge_pointclouds([first, second], color_table={"names": ["a", "b"]})
        wrong_names = {"names": ["only"], "color_table": np.zeros((2, 5))}
        with pytest.raises(ValueError, match="'names'"):
            cltpts.merge_pointclouds([first, second], color_table=wrong_names)
        wrong_rows = {"names": ["a", "b"], "color_table": np.zeros((1, 5))}
        with pytest.raises(ValueError, match="Number of rows"):
            cltpts.merge_pointclouds([first, second], color_table=wrong_rows)


####################################################################################################
# Colors and information
####################################################################################################
class TestPointwiseColors:
    def test_regions_use_their_colortable(self, cloud):
        colors = cloud.get_pointwise_colors()
        assert colors.dtype == np.uint8 and colors.shape == (6, 4)
        assert colors[:, :3].tolist() == [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
        ]
        assert (colors[:, 3] == 255).all()

    def test_opacity_of_the_colortable(self):
        pc = PointCloud(POINTS, region_colors="#ff0000", alpha=0.25)
        assert pc.get_pointwise_colors()[0].tolist() == [255, 0, 0, 64]

    def test_single_color_cloud(self, plain):
        colors = plain.get_pointwise_colors()
        assert np.unique(colors, axis=0).tolist() == [[191, 189, 189, 255]]

    def test_scalar_maps_use_a_colormap(self, cloud):
        colors = cloud.get_pointwise_colors("val", colormap="viridis")
        assert colors.dtype == np.uint8 and colors.shape == (6, 4)
        assert colors[0].tolist() == [68, 1, 84, 255]  # Lowest value
        assert colors[-1, :3] == pytest.approx([253, 231, 36], abs=1)  # Highest value
        gray = cloud.get_pointwise_colors("val", colormap="gray")
        assert gray[0, :3].tolist() == [0, 0, 0] and gray[-1, :3].tolist() == [
            255,
            255,
            255,
        ]

    def test_color_limits(self, cloud):
        colors = cloud.get_pointwise_colors("val", vmin=100, vmax=200)
        assert np.unique(colors, axis=0).tolist() == [
            [68, 1, 84, 255]
        ]  # All below vmin

    def test_values_outside_a_range_use_the_range_color(self, cloud):
        colors = cloud.get_pointwise_colors(
            "val",
            colormap="gray",
            range_min=10,
            range_max=30,
            range_color=(1, 2, 3, 255),
        )
        assert colors[:, :3].tolist() == [
            [1, 2, 3],  # 0 is below the range
            [0, 0, 0],
            [128, 128, 128],
            [255, 255, 255],
            [1, 2, 3],  # 40 and 50 are above it
            [1, 2, 3],
        ]

    def test_unknown_map(self, cloud):
        with pytest.raises(ValueError, match=r"Overlay 'nope' not found.*default, val"):
            cloud.get_pointwise_colors("nope")


class TestGetInfo:
    def test_overview(self, cloud, capsys):
        cloud.get_info()
        out = capsys.readouterr().out
        assert "POINT CLOUD EXPLORATION" in out
        assert len({len(line) for line in out.splitlines()}) == 1  # The box is aligned
        for text in (
            "Name: regions",
            "Points: 6",
            "AFFINE TRANSFORMATION MATRIX",
            "X: -30.0000 to 40.0000  (range: 70.0000)",
            "Y: -20.0000 to 20.0000  (range: 40.0000)",
            "Z: -15.0000 to 25.0000  (range: 40.0000)",
            "Centroid: [6.6667, 0.8333, 4.1667]",
            "SCALAR DATA PER POINT (2 maps)",
            "COLORTABLES (1 available)",
            "• default",
        ):
            assert text in out
        val = next(line for line in out.splitlines() if " val " in line)
        assert "Min:   0.0000" in val and "Max:  50.0000" in val

    def test_maps_that_are_not_numbers_are_listed_without_statistics(
        self, cloud, capsys
    ):
        cloud.add_point_data(np.array(list("abcdef")), name="labels")
        cloud.add_point_data(np.array([True, False] * 3), name="flag")
        cloud.get_info()
        lines = capsys.readouterr().out.splitlines()
        assert "SCALAR DATA PER POINT (4 maps)" in "\n".join(lines)
        labels = next(line for line in lines if " labels " in line)
        assert "(non-numeric)" in labels and "Min" not in labels
        flag = next(line for line in lines if " flag " in line)
        assert (
            "Min:   0.0000" in flag and "Max:   1.0000" in flag
        )  # Booleans are numbers
        assert len({len(line) for line in lines}) == 1

    def test_a_single_map_is_not_plural(self, plain, capsys):
        plain.get_info()
        assert "SCALAR DATA PER POINT (1 map)" in capsys.readouterr().out

    def test_empty_cloud(self, capsys):
        PointCloud(name="empty").get_info()
        out = capsys.readouterr().out
        assert "Name: empty" in out and "Points: 0" in out
        assert "Not available (empty point cloud)" in out
        assert (
            "SCALAR DATA PER POINT (0 maps)" in out
            and "No scalar data available" in out
        )


####################################################################################################
# Smoothing curves
####################################################################################################
class TestSmoothCurveCoordinates:
    @staticmethod
    def roughness(curve):
        return float(np.sum(np.diff(curve, n=2, axis=0) ** 2))

    @pytest.fixture
    def zigzag(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 30, 40)
        return np.c_[x, rng.normal(0, 1, 40), np.zeros(40)]

    def test_smoothing_reduces_the_roughness(self, zigzag):
        smooth = cltpts.smooth_curve_coordinates(zigzag)
        assert smooth.shape == zigzag.shape
        assert self.roughness(smooth) < 0.1 * self.roughness(zigzag)

    def test_the_input_is_not_modified(self, zigzag):
        original = zigzag.copy()
        smooth = cltpts.smooth_curve_coordinates(zigzag)
        assert smooth is not zigzag and np.array_equal(zigzag, original)

    def test_more_smoothing_is_smoother(self, zigzag):
        base = self.roughness(cltpts.smooth_curve_coordinates(zigzag))
        assert (
            self.roughness(cltpts.smooth_curve_coordinates(zigzag, iterations=3)) < base
        )
        assert self.roughness(cltpts.smooth_curve_coordinates(zigzag, sigma=3.0)) < base
        wide = cltpts.smooth_curve_coordinates(zigzag, sigma=3.0, window_size=9)
        narrow = cltpts.smooth_curve_coordinates(zigzag, sigma=3.0, window_size=5)
        assert self.roughness(wide) < self.roughness(narrow)

    def test_zero_iterations_change_nothing(self, zigzag):
        assert np.array_equal(
            cltpts.smooth_curve_coordinates(zigzag, iterations=0), zigzag
        )

    def test_a_straight_line_is_not_changed_away_from_its_ends(self):
        line = np.c_[np.linspace(0, 30, 20), np.full(20, 5.0), np.full(20, 5.0)]
        smooth = cltpts.smooth_curve_coordinates(line, sigma=2.0)
        assert smooth[2:-2] == pytest.approx(line[2:-2])  # Where the window is complete
        assert smooth[:, 1:] == pytest.approx(5.0)

    def test_the_ends_are_pulled_inwards(self):
        line = np.c_[np.linspace(0, 10, 11), np.zeros(11), np.zeros(11)]
        smooth = cltpts.smooth_curve_coordinates(line)
        assert smooth[0, 0] > line[0, 0] and smooth[-1, 0] < line[-1, 0]

    def test_a_constant_curve_stays_constant(self):
        assert cltpts.smooth_curve_coordinates(np.full((10, 3), 7.0)) == pytest.approx(
            7.0
        )

    def test_curves_shorter_than_the_window(self):
        curve = np.array([[0.0, 0, 0], [1, 1, 1], [2, 2, 2]])
        smooth = cltpts.smooth_curve_coordinates(curve)
        assert smooth[1] == pytest.approx([1, 1, 1])  # Symmetric around the middle
        assert smooth[0, 0] == pytest.approx(0.504, abs=1e-3)
        assert smooth[2, 0] == pytest.approx(1.496, abs=1e-3)
        assert cltpts.smooth_curve_coordinates(np.array([[1.0, 2, 3]])).tolist() == [
            [1, 2, 3]
        ]

    def test_integer_coordinates_are_smoothed_as_floats(self):
        curve = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        smooth = cltpts.smooth_curve_coordinates(curve)
        assert smooth.dtype == np.float64
        assert smooth[0, 0] == pytest.approx(0.504, abs=1e-3)
        assert smooth[2, 0] == pytest.approx(1.496, abs=1e-3)
        # The same result as for the equivalent floating point curve
        assert np.array_equal(
            smooth, cltpts.smooth_curve_coordinates(curve.astype(float))
        )

    def test_lists_are_accepted(self):
        smooth = cltpts.smooth_curve_coordinates([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        assert isinstance(smooth, np.ndarray) and smooth[2, 0] == pytest.approx(
            1.496, abs=1e-3
        )

    def test_floating_point_precision_is_kept(self):
        curve = np.c_[np.linspace(0, 10, 11), np.zeros(11), np.zeros(11)]
        assert (
            cltpts.smooth_curve_coordinates(curve.astype(np.float32)).dtype
            == np.float32
        )
        assert cltpts.smooth_curve_coordinates(curve).dtype == np.float64
