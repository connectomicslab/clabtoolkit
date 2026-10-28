"""Tests for clabtoolkit.morphometrytools."""

import json
import sys
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

import clabtoolkit.connectivitytools as cltconn
import clabtoolkit.freesurfertools as cltfree
import clabtoolkit.morphometrytools as cltmorph
import clabtoolkit.parcellationtools as cltparc
import clabtoolkit.surfacetools as cltsurf

GLOBAL = "brain-brain-wholebrain"


####################################################################################################
# Helpers and fixtures
####################################################################################################
def grid_mesh(n=3, spacing=1.0):
    """Flat n x n grid of vertices on z=0, two triangles per cell."""
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    coords = np.c_[xs.ravel(), ys.ravel(), np.zeros(n * n)] * spacing
    faces = []
    for row in range(n - 1):
        for col in range(n - 1):
            a = row * n + col
            b, c, d = a + 1, a + n, a + n + 1
            faces += [[a, b, c], [b, d, c]]
    return coords, np.array(faces)


def make_annot(codes, names, table_codes, hemi="lh"):
    """Annotation built by hand: one color-table row per region name."""
    annot = cltfree.AnnotParcellation()
    annot.codes = np.array(codes)
    annot.regnames = list(names)
    table = np.zeros((len(names), 5), dtype=int)
    table[:, 0] = np.arange(len(names)) * 40
    table[:, 4] = table_codes
    annot.regtable = table
    annot.hemi = hemi
    return annot


def vertex_areas(coords, faces):
    """Every vertex gets a third of the area of each of its triangles (explicit loop)."""
    areas = np.zeros(len(coords))
    for tri in faces:
        v1, v2, v3 = coords[tri]
        area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
        areas[tri] += area / 3
    return areas


def value_of(df, region, column="Value"):
    """Value of a region in a table with one row per region."""
    rows = df[df["Region"] == region]
    assert len(rows) == 1, f"{region} should appear once in {df['Region'].tolist()}"
    return float(rows[column].iloc[0])


@pytest.fixture(autouse=True)
def _quiet():
    """Silence the warnings that the table builders emit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture
def cm_units(monkeypatch):
    """Report areas in cm2 and volumes in cm3, instead of the mm2 / mm3 of config.json."""
    lookup = dict(cltmorph._default_units_lookup())
    lookup.update({"area": "cm2", "volume": "cm3"})
    monkeypatch.setattr(cltmorph, "_default_units_lookup", lambda: lookup)


@pytest.fixture
def grid():
    """3 x 3 grid, 1 mm apart, as a Surface: 9 vertices, 8 triangles, 4 mm2 in total."""
    coords, faces = grid_mesh(3, 1.0)
    return cltsurf.Surface(vertices=coords, faces=faces)


@pytest.fixture
def big_grid():
    """The same grid with 10 mm between vertices: 400 mm2 in total."""
    coords, faces = grid_mesh(3, 10.0)
    return cltsurf.Surface(vertices=coords, faces=faces)


@pytest.fixture
def annot():
    """Region A has the vertices 0, 3 and 6 and region B the other six."""
    return make_annot([10, 20, 20, 10, 20, 20, 10, 20, 20], ["regA", "regB"], [10, 20])


@pytest.fixture
def annot_with_medialwall():
    """Rows of the grid: regA (bottom), regB (middle) and the medial wall (top)."""
    return make_annot(
        [10, 10, 10, 20, 20, 20, 30, 30, 30],
        ["regA", "regB", "medialwall"],
        [10, 20, 30],
    )


####################################################################################################
class TestGetUnits:
    def test_default_units(self):
        assert cltmorph.get_units("thickness") == ["mm"]
        assert cltmorph.get_units(["thickness", "nvertices", "md"]) == [
            "mm",
            "au",
            "mm2/s",
        ]

    def test_case_insensitive(self):
        assert cltmorph.get_units("THICKNESS") == ["mm"]

    def test_unknown_metric_is_reported_as_unknown(self):
        assert cltmorph.get_units(["not_a_metric", "thickness"]) == ["unknown", "mm"]

    def test_dict_with_metrics_units_key(self):
        config = {"metrics_units": {"Custom": "kg"}}
        assert cltmorph.get_units("custom", metrics_json=config) == ["kg"]

    def test_flat_dict(self):
        assert cltmorph.get_units("custom", metrics_json={"custom": "kg"}) == ["kg"]

    def test_dict_replaces_the_default_units(self):
        assert cltmorph.get_units("thickness", metrics_json={"custom": "kg"}) == [
            "unknown"
        ]

    def test_json_file(self, tmp_path):
        path = tmp_path / "units.json"
        path.write_text(json.dumps({"metrics_units": {"custom": "kg"}}))
        assert cltmorph.get_units("custom", metrics_json=path) == ["kg"]
        assert cltmorph.get_units("custom", metrics_json=str(path)) == ["kg"]

    def test_missing_file_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid JSON file path"):
            cltmorph.get_units("custom", metrics_json=tmp_path / "missing.json")

    def test_invalid_json_is_rejected(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not json")
        with pytest.raises(ValueError, match="Invalid JSON format"):
            cltmorph.get_units("custom", metrics_json=path)

    def test_file_without_metrics_units_is_rejected(self, tmp_path):
        path = tmp_path / "other.json"
        path.write_text(json.dumps({"something": 1}))
        with pytest.raises(ValueError, match="metrics_units"):
            cltmorph.get_units("custom", metrics_json=path)

    def test_invalid_type_is_rejected(self):
        with pytest.raises(ValueError, match="metrics_json must be"):
            cltmorph.get_units("custom", metrics_json=5)


####################################################################################################
class TestConvertFromMm:
    @pytest.mark.parametrize(
        "unit, factor",
        [
            ("mm2", 1.0),
            ("cm2", 100.0),
            ("dm2", 1e4),
            ("m2", 1e6),
            ("um2", 1e-6),
            ("μm2", 1e-6),
            ("cm²", 100.0),
            ("cm^2", 100.0),
            ("CM2", 100.0),
        ],
    )
    def test_area_units(self, unit, factor):
        value, returned = cltmorph.convert_from_mm(250.0, "area", {"area": unit})
        assert value == pytest.approx(250.0 / factor)
        assert returned == unit

    @pytest.mark.parametrize(
        "unit, factor",
        [
            ("mm3", 1.0),
            ("cm3", 1e3),
            ("cc", 1e3),
            ("ml", 1e3),
            ("l", 1e6),
            ("dm3", 1e6),
            ("m3", 1e9),
            ("um3", 1e-9),
            ("cm³", 1e3),
            ("cm^3", 1e3),
        ],
    )
    def test_volume_units(self, unit, factor):
        value, returned = cltmorph.convert_from_mm(1500.0, "volume", {"volume": unit})
        assert value == pytest.approx(1500.0 / factor)
        assert returned == unit

    def test_scalar_returns_a_float(self):
        value, _ = cltmorph.convert_from_mm(100, "area", {"area": "cm2"})
        assert isinstance(value, float)
        assert value == 1.0

    def test_numpy_scalar_returns_a_float(self):
        value, _ = cltmorph.convert_from_mm(np.float64(100), "area", {"area": "cm2"})
        assert isinstance(value, float)

    def test_array_returns_an_array(self):
        values, unit = cltmorph.convert_from_mm(
            np.array([100.0, 250.0]), "area", {"area": "cm2"}
        )
        assert isinstance(values, np.ndarray)
        np.testing.assert_allclose(values, [1.0, 2.5])
        assert unit == "cm2"

    def test_list_returns_an_array(self):
        values, _ = cltmorph.convert_from_mm([100, 300], "area", {"area": "cm2"})
        np.testing.assert_allclose(values, [1.0, 3.0])

    def test_input_array_is_not_modified(self):
        original = np.array([100.0, 250.0])
        cltmorph.convert_from_mm(original, "area", {"area": "cm2"})
        np.testing.assert_array_equal(original, [100.0, 250.0])

    def test_metric_name_is_case_insensitive(self):
        value, _ = cltmorph.convert_from_mm(100.0, "AREA", {"area": "cm2"})
        assert value == 1.0

    def test_default_config_is_used(self, cm_units):
        assert cltmorph.convert_from_mm(100.0, "area") == (1.0, "cm2")
        assert cltmorph.convert_from_mm(1500.0, "volume") == (1.5, "cm3")

    def test_only_area_and_volume_can_be_converted(self):
        with pytest.raises(ValueError, match="only handles 'area' and 'volume'"):
            cltmorph.convert_from_mm(1.0, "thickness")

    def test_metric_without_unit_in_the_configuration(self):
        with pytest.raises(ValueError, match="No unit defined"):
            cltmorph.convert_from_mm(1.0, "area", {"volume": "cm3"})

    @pytest.mark.parametrize(
        "metric, unit",
        [
            ("area", "cm3"),
            ("volume", "cm2"),
            ("area", "ml"),
            ("area", "kg2"),
            ("volume", "cm"),
            ("area", ""),
        ],
    )
    def test_unit_of_another_dimension_is_rejected(self, metric, unit):
        with pytest.raises(ValueError, match="Unsupported unit"):
            cltmorph.convert_from_mm(1.0, metric, {metric: unit})


####################################################################################################
class TestStatsFromVector:
    def test_default_statistics(self):
        # mean, median, std, min, max
        result = cltmorph.stats_from_vector([1, 2, 3, 4, 5])
        assert result == pytest.approx([3.0, 3.0, np.sqrt(2.0), 1.0, 5.0])

    def test_std_is_the_population_std(self):
        assert cltmorph.stats_from_vector([2, 4], ["std"]) == [1.0]

    def test_count_and_sum(self):
        assert cltmorph.stats_from_vector([1, 2, 3, 4, 5], ["count", "sum"]) == [
            5.0,
            15.0,
        ]

    def test_value_is_the_mean(self):
        assert cltmorph.stats_from_vector([1, 2, 6], ["value"]) == [3.0]
        assert cltmorph.stats_from_vector([1, 2, 6], ["mean"]) == [3.0]

    def test_statistics_come_back_in_the_requested_order(self):
        assert cltmorph.stats_from_vector([1, 2, 6], ["max", "min", "sum"]) == [
            6.0,
            1.0,
            9.0,
        ]

    def test_repeated_statistics_are_kept(self):
        assert cltmorph.stats_from_vector([1, 3], ["mean", "mean"]) == [2.0, 2.0]

    def test_names_are_case_insensitive(self):
        assert cltmorph.stats_from_vector([1, 3], ["MEAN", "Max"]) == [2.0, 3.0]

    def test_tuple_of_statistics(self):
        assert cltmorph.stats_from_vector([1, 3], ("min", "max")) == [1.0, 3.0]

    def test_zeros_are_ignored_by_default(self):
        assert cltmorph.stats_from_vector([0, 0, 1, 2, 3], ["mean", "count"]) == [
            2.0,
            3.0,
        ]

    def test_zeros_are_kept_when_asked(self):
        result = cltmorph.stats_from_vector(
            [0, 0, 1, 2, 3], ["mean", "count", "min"], nonzeros_only=False
        )
        assert result == pytest.approx([1.2, 5.0, 0.0])

    def test_negative_values_are_not_zeros(self):
        assert cltmorph.stats_from_vector([-2, 0, 2, 4], ["mean", "count"]) == [
            4 / 3,
            3.0,
        ]

    def test_multidimensional_input_is_flattened(self):
        assert cltmorph.stats_from_vector(np.arange(6).reshape(2, 3), ["sum"]) == [15.0]

    def test_integer_input_gives_floats(self):
        result = cltmorph.stats_from_vector(np.array([1, 2], dtype=np.int8), ["sum"])
        assert result == [3.0]
        assert isinstance(result[0], float)

    def test_empty_input(self):
        result = cltmorph.stats_from_vector(
            [], ["mean", "median", "std", "min", "max", "count", "sum"]
        )
        assert result[-2:] == [0.0, 0.0]
        assert all(np.isnan(v) for v in result[:-2])

    def test_all_zero_input_ignoring_zeros(self):
        result = cltmorph.stats_from_vector([0, 0, 0], ["mean", "count", "max"])
        assert result == [0.0, 0.0, 0.0]

    def test_all_zero_input_keeping_zeros(self):
        result = cltmorph.stats_from_vector(
            [0, 0, 0], ["mean", "count"], nonzeros_only=False
        )
        assert result == [0.0, 3.0]

    def test_nan_propagates(self):
        assert np.isnan(cltmorph.stats_from_vector([1, np.nan], ["mean"])[0])

    def test_unsupported_statistic_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported statistics: variance"):
            cltmorph.stats_from_vector([1, 2], ["mean", "variance"])

    def test_statistics_must_be_a_list(self):
        with pytest.raises(TypeError, match="list or tuple"):
            cltmorph.stats_from_vector([1, 2], "mean")


####################################################################################################
class TestAreaFromMesh:
    def test_right_triangle(self):
        coords = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0]])
        total, per_triangle = cltmorph.area_from_mesh(coords, np.array([[0, 1, 2]]))
        assert total == pytest.approx(6.0)
        np.testing.assert_allclose(per_triangle, [6.0])

    def test_square_made_of_two_triangles(self):
        coords = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]])
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        total, per_triangle = cltmorph.area_from_mesh(coords, faces)
        assert total == pytest.approx(100.0)
        np.testing.assert_allclose(per_triangle, [50.0, 50.0])

    def test_total_is_a_float_and_triangles_an_array(self):
        coords, faces = grid_mesh(3)
        total, per_triangle = cltmorph.area_from_mesh(coords, faces)
        assert isinstance(total, float)
        assert per_triangle.shape == (len(faces),)
        assert total == pytest.approx(per_triangle.sum())

    def test_grid_area(self):
        coords, faces = grid_mesh(4, spacing=2.0)
        total, per_triangle = cltmorph.area_from_mesh(coords, faces)
        assert total == pytest.approx(36.0)  # (3 cells * 2 mm)^2
        np.testing.assert_allclose(per_triangle, 2.0)

    def test_triangle_out_of_the_plane(self):
        # Face of a cube corner, with sides of sqrt(2): area sqrt(3) / 2
        coords = np.eye(3)
        total, _ = cltmorph.area_from_mesh(coords, np.array([[0, 1, 2]]))
        assert total == pytest.approx(np.sqrt(3) / 2)

    def test_rigid_motion_does_not_change_the_area(self):
        rng = np.random.default_rng(0)
        coords = rng.normal(size=(20, 3))
        faces = rng.integers(0, 20, size=(30, 3))
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        moved = coords @ rotation.T + np.array([5.0, -3.0, 100.0])
        assert cltmorph.area_from_mesh(moved, faces)[0] == pytest.approx(
            cltmorph.area_from_mesh(coords, faces)[0]
        )

    def test_scaling_scales_the_area_quadratically(self):
        coords, faces = grid_mesh(3)
        base = cltmorph.area_from_mesh(coords, faces)[0]
        assert cltmorph.area_from_mesh(coords * 3, faces)[0] == pytest.approx(9 * base)

    def test_vertex_order_does_not_change_the_area(self):
        coords = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0]])
        assert cltmorph.area_from_mesh(coords, np.array([[2, 1, 0]]))[0] == (
            pytest.approx(6.0)
        )

    def test_degenerate_triangle_has_no_area(self):
        coords = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        assert cltmorph.area_from_mesh(coords, np.array([[0, 1, 2]]))[0] == 0.0

    def test_thin_triangle_keeps_its_precision(self):
        # Heron's formula gives 0 or noise here
        coords = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1e-9, 0]])
        total, _ = cltmorph.area_from_mesh(coords, np.array([[0, 1, 2]]))
        assert total == pytest.approx(0.5e-9, rel=1e-6)

    def test_areas_are_given_in_the_configured_units(self, cm_units):
        coords = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [10, 10, 0]])
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        total, per_triangle = cltmorph.area_from_mesh(coords, faces)
        assert total == pytest.approx(1.0)  # 100 mm2 = 1 cm2
        np.testing.assert_allclose(per_triangle, [0.5, 0.5])

    def test_input_is_not_modified(self):
        coords, faces = grid_mesh(3)
        coords_before, faces_before = coords.copy(), faces.copy()
        cltmorph.area_from_mesh(coords, faces)
        np.testing.assert_array_equal(coords, coords_before)
        np.testing.assert_array_equal(faces, faces_before)

    def test_integer_coordinates(self):
        coords = np.array([[0, 0, 0], [4, 0, 0], [0, 3, 0]], dtype=np.int32)
        assert cltmorph.area_from_mesh(coords, np.array([[0, 1, 2]]))[0] == 6.0

    @pytest.mark.parametrize("shape", [(3, 2), (3,), (2, 3, 3)])
    def test_coordinates_must_be_n_by_3(self, shape):
        with pytest.raises(ValueError, match="coords must have shape"):
            cltmorph.area_from_mesh(np.zeros(shape), np.array([[0, 1, 2]]))

    @pytest.mark.parametrize("shape", [(1, 4), (3,)])
    def test_faces_must_be_m_by_3(self, shape):
        with pytest.raises(ValueError, match="faces must have shape"):
            cltmorph.area_from_mesh(np.zeros((4, 3)), np.zeros(shape, dtype=int))

    @pytest.mark.parametrize("face", [[0, 1, 3], [0, 1, -1]])
    def test_invalid_vertex_indices_are_rejected(self, face):
        with pytest.raises(ValueError, match="invalid vertex indices"):
            cltmorph.area_from_mesh(np.zeros((3, 3)), np.array([face]))


####################################################################################################
class TestEulerFromMesh:
    TETRAHEDRON = (
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
    )
    OCTAHEDRON_FACES = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [5, 1, 2],
            [5, 2, 3],
            [5, 3, 4],
            [5, 4, 1],
        ]
    )

    def test_sphere_topology(self):
        assert cltmorph.euler_from_mesh(*self.TETRAHEDRON) == 2
        assert cltmorph.euler_from_mesh(np.zeros((6, 3)), self.OCTAHEDRON_FACES) == 2

    def test_returns_a_python_int(self):
        assert isinstance(cltmorph.euler_from_mesh(*self.TETRAHEDRON), int)

    def test_single_triangle(self):
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        assert cltmorph.euler_from_mesh(coords, np.array([[0, 1, 2]])) == 1

    def test_disk_is_one(self):
        coords, faces = grid_mesh(5)
        assert cltmorph.euler_from_mesh(coords, faces) == 1

    def test_sphere_with_a_hole_is_a_disk(self):
        assert (
            cltmorph.euler_from_mesh(np.zeros((6, 3)), self.OCTAHEDRON_FACES[1:]) == 1
        )

    def test_two_disjoint_triangles(self):
        coords = np.zeros((6, 3))
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        assert cltmorph.euler_from_mesh(coords, faces) == 2

    def test_torus_is_zero(self):
        n = 4
        faces = []
        for row in range(n):
            for col in range(n):
                a = row * n + col
                b = row * n + (col + 1) % n
                c = ((row + 1) % n) * n + col
                d = ((row + 1) % n) * n + (col + 1) % n
                faces += [[a, b, c], [b, d, c]]
        assert cltmorph.euler_from_mesh(np.zeros((n * n, 3)), np.array(faces)) == 0

    def test_shared_edges_are_counted_once(self):
        # Two triangles sharing an edge: V=4, E=5, F=2
        coords = np.zeros((4, 3))
        faces = np.array([[0, 1, 2], [1, 3, 2]])
        assert cltmorph.euler_from_mesh(coords, faces) == 1

    def test_isolated_vertices_count(self):
        coords = np.zeros((5, 3))
        assert cltmorph.euler_from_mesh(coords, np.array([[0, 1, 2]])) == 3

    def test_vertex_positions_are_irrelevant(self):
        coords, faces = self.TETRAHEDRON
        assert cltmorph.euler_from_mesh(coords * 100, faces) == 2

    @pytest.mark.parametrize("shape", [(3, 2), (3,)])
    def test_coordinates_must_be_n_by_3(self, shape):
        with pytest.raises(ValueError, match="coords must have shape"):
            cltmorph.euler_from_mesh(np.zeros(shape), np.array([[0, 1, 2]]))

    @pytest.mark.parametrize("shape", [(1, 4), (3,)])
    def test_faces_must_be_m_by_3(self, shape):
        with pytest.raises(ValueError, match="faces must have shape"):
            cltmorph.euler_from_mesh(np.zeros((4, 3)), np.zeros(shape, dtype=int))

    @pytest.mark.parametrize("face", [[0, 1, 3], [0, 1, -1]])
    def test_invalid_vertex_indices_are_rejected(self, face):
        with pytest.raises(ValueError, match="invalid vertex indices"):
            cltmorph.euler_from_mesh(np.zeros((3, 3)), np.array([face]))


####################################################################################################
class TestComputeEulerFromSurf:
    def test_grid_is_a_disk(self, grid):
        df, path = cltmorph.compute_euler_fromsurf(grid, "lh", add_bids_entities=False)
        assert path is None
        assert df["Region"].tolist() == ["ctx-lh-hemisphere"]
        assert value_of(df, "ctx-lh-hemisphere") == 1
        assert df["Metric"].tolist() == ["euler"]
        assert df["Units"].tolist() == ["au"]

    def test_sphere_has_euler_number_two(self):
        sphere = cltsurf.Surface.simulate_surface(seed=1, shape="sphere", maps=None)
        df, _ = cltmorph.compute_euler_fromsurf(sphere, "rh", add_bids_entities=False)
        assert value_of(df, "ctx-rh-hemisphere") == 2

    def test_hemisphere_in_the_table(self, grid):
        df, _ = cltmorph.compute_euler_fromsurf(grid, "rh", add_bids_entities=False)
        assert df["Hemisphere"].tolist() == ["rh"]
        assert df["Supraregion"].tolist() == ["ctx"]

    def test_surface_type_is_stored_in_the_source(self, grid):
        df, _ = cltmorph.compute_euler_fromsurf(
            grid, "lh", surf_type="white", add_bids_entities=False
        )
        assert df["Source"].tolist() == ["white"]

    def test_surface_type_comes_from_the_extension(self, tmp_path, grid):
        path = tmp_path / "lh.pial"
        grid.save_surface(str(path))
        df, _ = cltmorph.compute_euler_fromsurf(path, "lh", add_bids_entities=False)
        assert df["Source"].tolist() == ["pial"]
        assert df["MetricFile"].tolist() == [str(path)]
        assert value_of(df, "ctx-lh-hemisphere") == 1

    def test_explicit_surface_type_wins_over_the_extension(self, tmp_path, grid):
        path = tmp_path / "lh.pial"
        grid.save_surface(str(path))
        df, _ = cltmorph.compute_euler_fromsurf(
            str(path), "lh", surf_type="custom", add_bids_entities=False
        )
        assert df["Source"].tolist() == ["custom"]

    def test_region_table(self, grid):
        df, _ = cltmorph.compute_euler_fromsurf(
            grid, "lh", table_type="region", add_bids_entities=False
        )
        assert df["Statistics"].tolist() == ["Value"]
        assert df["ctx-lh-hemisphere"].tolist() == [1]

    def test_table_is_saved(self, tmp_path, grid):
        out = tmp_path / "euler.tsv"
        df, path = cltmorph.compute_euler_fromsurf(
            grid, "lh", output_table=out, add_bids_entities=False
        )
        assert path == str(out)
        saved = pd.read_csv(out, sep="\t")
        assert saved["Region"].tolist() == ["ctx-lh-hemisphere"]
        assert saved["Value"].tolist() == [1]

    def test_missing_surface(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Surface file not found"):
            cltmorph.compute_euler_fromsurf(tmp_path / "lh.white", "lh")

    def test_invalid_surface_type(self):
        with pytest.raises(TypeError, match="surf_file must be"):
            cltmorph.compute_euler_fromsurf(np.zeros((3, 3)), "lh")

    def test_invalid_table_type(self, grid):
        with pytest.raises(ValueError, match="Invalid table_type"):
            cltmorph.compute_euler_fromsurf(grid, "lh", table_type="wide")

    def test_output_directory_must_exist(self, tmp_path, grid):
        with pytest.raises(FileNotFoundError, match="Directory does not exist"):
            cltmorph.compute_euler_fromsurf(
                grid, "lh", output_table=tmp_path / "missing" / "euler.tsv"
            )

    def test_output_table_must_be_a_path(self, grid):
        with pytest.raises(TypeError, match="output_table must be"):
            cltmorph.compute_euler_fromsurf(grid, "lh", output_table=5)


####################################################################################################
class TestComputeRegAreaFromSurf:
    def test_areas_of_the_grid(self, grid, annot):
        # A has 6 incident triangles and B the 8 triangles' remaining share: 1 and 3 mm2
        df, path = cltmorph.compute_reg_area_fromsurf(
            grid, annot, "lh", add_bids_entities=False
        )
        assert path is None
        assert df["Region"].tolist() == [
            "ctx-lh-hemisphere",
            "ctx-lh-regA",
            "ctx-lh-regB",
        ]
        assert value_of(df, "ctx-lh-regA") == pytest.approx(1.0)
        assert value_of(df, "ctx-lh-regB") == pytest.approx(3.0)
        assert value_of(df, "ctx-lh-hemisphere") == pytest.approx(4.0)

    def test_table_metadata(self, grid, annot):
        df, _ = cltmorph.compute_reg_area_fromsurf(
            grid, annot, "lh", surf_type="white", add_bids_entities=False
        )
        assert set(df["Source"]) == {"white"}
        assert set(df["Metric"]) == {"area"}
        assert set(df["Units"]) == {"mm2"}
        assert set(df["Supraregion"]) == {"ctx"}
        assert set(df["Hemisphere"]) == {"lh"}

    def test_regions_add_up_to_the_surface_area(self):
        sphere = cltsurf.Surface.simulate_surface(seed=3, shape="sphere", maps=None)
        annot = cltfree.AnnotParcellation.simulate_annotation(sphere, n_regions=6)
        df, _ = cltmorph.compute_reg_area_fromsurf(
            sphere, annot, "lh", add_bids_entities=False
        )
        total = cltmorph.area_from_mesh(sphere.mesh.points, sphere.get_faces())[0]
        regions = df[df["Region"] != "ctx-lh-hemisphere"]
        assert len(regions) == 6
        assert regions["Value"].sum() == pytest.approx(total)
        assert value_of(df, "ctx-lh-hemisphere") == pytest.approx(total)

    def test_areas_follow_the_vertex_area_definition(self):
        sphere = cltsurf.Surface.simulate_surface(seed=4, shape="sphere", maps=None)
        annot = cltfree.AnnotParcellation.simulate_annotation(sphere, n_regions=5)
        df, _ = cltmorph.compute_reg_area_fromsurf(
            sphere, annot, "lh", add_bids_entities=False
        )
        per_vertex = vertex_areas(
            np.asarray(sphere.mesh.points, dtype=float), sphere.get_faces()
        )
        for name, code in zip(annot.regnames, annot.regtable[:, 4], strict=True):
            expected = per_vertex[annot.codes == code].sum()
            assert value_of(df, f"ctx-lh-{name}") == pytest.approx(expected)

    def test_medial_wall_is_excluded_by_default(self, grid, annot_with_medialwall):
        df, _ = cltmorph.compute_reg_area_fromsurf(
            grid, annot_with_medialwall, "lh", add_bids_entities=False
        )
        assert "ctx-lh-medialwall" not in df["Region"].tolist()
        # Bottom row vertices: 1, 3, 2 triangles... the rest is hand counted below
        per_vertex = vertex_areas(*grid_mesh(3, 1.0))
        expected = per_vertex[:6].sum()
        assert value_of(df, "ctx-lh-hemisphere") == pytest.approx(expected)
        assert value_of(df, "ctx-lh-hemisphere") < 4.0

    @pytest.mark.parametrize("name", ["medialwall", "Unknown", "corpuscallosum"])
    def test_non_anatomical_regions_are_excluded_by_default(self, grid, name):
        annot = make_annot(
            [10, 10, 10, 20, 20, 20, 30, 30, 30], ["regA", "regB", name], [10, 20, 30]
        )
        by_default, _ = cltmorph.compute_reg_nvertices_fromsurf(
            grid, annot, "lh", add_bids_entities=False
        )
        everything, _ = cltmorph.compute_reg_nvertices_fromsurf(
            grid, annot, "lh", include_unknown=True, add_bids_entities=False
        )
        assert by_default["Region"].tolist() == [
            "ctx-lh-hemisphere",
            "ctx-lh-regA",
            "ctx-lh-regB",
        ]
        assert value_of(by_default, "ctx-lh-hemisphere") == 6
        assert value_of(everything, f"ctx-lh-{name}") == 3

    def test_medial_wall_can_be_included(self, grid, annot_with_medialwall):
        df, _ = cltmorph.compute_reg_area_fromsurf(
            grid,
            annot_with_medialwall,
            "lh",
            include_unknown=True,
            add_bids_entities=False,
        )
        per_vertex = vertex_areas(*grid_mesh(3, 1.0))
        assert value_of(df, "ctx-lh-medialwall") == pytest.approx(per_vertex[6:].sum())
        assert value_of(df, "ctx-lh-hemisphere") == pytest.approx(4.0)

    def test_without_the_global_row(self, grid, annot):
        df, _ = cltmorph.compute_reg_area_fromsurf(
            grid, annot, "lh", include_global=False, add_bids_entities=False
        )
        assert df["Region"].tolist() == ["ctx-lh-regA", "ctx-lh-regB"]

    def test_region_table(self, grid, annot):
        df, _ = cltmorph.compute_reg_area_fromsurf(
            grid, annot, "lh", table_type="region", add_bids_entities=False
        )
        assert df["Statistics"].tolist() == ["Value"]
        assert df["ctx-lh-regA"].iloc[0] == pytest.approx(1.0)
        assert df["ctx-lh-regB"].iloc[0] == pytest.approx(3.0)
        assert df["ctx-lh-hemisphere"].iloc[0] == pytest.approx(4.0)

    def test_areas_are_given_in_the_configured_units(self, cm_units, big_grid, annot):
        df, _ = cltmorph.compute_reg_area_fromsurf(
            big_grid, annot, "lh", add_bids_entities=False
        )
        assert set(df["Units"]) == {"cm2"}
        assert value_of(df, "ctx-lh-regA") == pytest.approx(1.0)  # 100 mm2
        assert value_of(df, "ctx-lh-hemisphere") == pytest.approx(4.0)  # 400 mm2

    def test_files_give_the_same_result_as_objects(self, tmp_path, annot):
        sphere = cltsurf.Surface.simulate_surface(seed=5, shape="sphere", maps=None)
        annot = cltfree.AnnotParcellation.simulate_annotation(sphere, n_regions=4)
        surf_path, annot_path = tmp_path / "lh.white", tmp_path / "lh.aparc.annot"
        sphere.save_surface(str(surf_path))
        annot.save_annotation(str(annot_path))

        from_files, _ = cltmorph.compute_reg_area_fromsurf(
            surf_path, annot_path, "lh", add_bids_entities=False
        )
        from_objects, _ = cltmorph.compute_reg_area_fromsurf(
            sphere, annot, "lh", add_bids_entities=False
        )
        assert from_files["MetricFile"].iloc[0] == str(surf_path)
        np.testing.assert_allclose(
            from_files["Value"].to_numpy(), from_objects["Value"].to_numpy(), rtol=1e-5
        )

    def test_annotation_object_is_not_modified(self, grid, annot_with_medialwall):
        codes = annot_with_medialwall.codes.copy()
        cltmorph.compute_reg_area_fromsurf(
            grid, annot_with_medialwall, "lh", add_bids_entities=False
        )
        np.testing.assert_array_equal(annot_with_medialwall.codes, codes)
        assert annot_with_medialwall.regnames == ["regA", "regB", "medialwall"]
        assert annot_with_medialwall.regtable.shape == (3, 5)

    def test_bids_entities_come_from_the_annotation_file(self, tmp_path, grid, annot):
        path = tmp_path / "sub-01_ses-M00_hemi-L_desc-x_dseg.annot"
        # The table is built from the annotation file name
        sphere = cltsurf.Surface.simulate_surface(seed=5, shape="sphere", maps=None)
        sim = cltfree.AnnotParcellation.simulate_annotation(sphere, n_regions=3)
        sim.save_annotation(str(path))
        df, _ = cltmorph.compute_reg_area_fromsurf(sphere, path, "lh")
        assert df["Participant"].unique().tolist() == ["01"]
        assert df["Session"].unique().tolist() == ["M00"]

    def test_table_is_saved(self, tmp_path, grid, annot):
        out = tmp_path / "area.tsv"
        df, path = cltmorph.compute_reg_area_fromsurf(
            grid, annot, "lh", output_table=out, add_bids_entities=False
        )
        assert path == str(out)
        saved = pd.read_csv(out, sep="\t")
        assert saved["Region"].tolist() == df["Region"].tolist()
        np.testing.assert_allclose(saved["Value"], df["Value"])

    def test_annotation_must_match_the_surface(self, grid):
        annot = make_annot([10, 20, 20], ["regA", "regB"], [10, 20])
        with pytest.raises(ValueError, match="annotation has 3 vertices"):
            cltmorph.compute_reg_area_fromsurf(grid, annot, "lh")

    def test_repeated_region_names_are_rejected(self, grid):
        annot = make_annot(
            [10, 20, 20, 10, 20, 20, 10, 20, 20], ["same", "same"], [10, 20]
        )
        with pytest.raises(ValueError, match="share the same name"):
            cltmorph.compute_reg_area_fromsurf(grid, annot, "lh")

    def test_missing_files(self, tmp_path, grid, annot):
        with pytest.raises(FileNotFoundError, match="Annotation file not found"):
            cltmorph.compute_reg_area_fromsurf(grid, tmp_path / "x.annot", "lh")
        with pytest.raises(FileNotFoundError, match="Surface file not found"):
            cltmorph.compute_reg_area_fromsurf(tmp_path / "lh.white", annot, "lh")

    def test_invalid_input_types(self, grid, annot):
        with pytest.raises(TypeError, match="parc_file must be"):
            cltmorph.compute_reg_area_fromsurf(grid, np.zeros(9), "lh")
        with pytest.raises(TypeError, match="surf_file must be"):
            cltmorph.compute_reg_area_fromsurf(np.zeros((9, 3)), annot, "lh")

    def test_invalid_table_type(self, grid, annot):
        with pytest.raises(ValueError, match="Invalid table_type"):
            cltmorph.compute_reg_area_fromsurf(grid, annot, "lh", table_type="wide")


####################################################################################################
class TestComputeRegNverticesFromSurf:
    def test_counts_of_the_grid(self, grid, annot):
        df, path = cltmorph.compute_reg_nvertices_fromsurf(
            grid, annot, "lh", add_bids_entities=False
        )
        assert path is None
        assert value_of(df, "ctx-lh-regA") == 3
        assert value_of(df, "ctx-lh-regB") == 6
        assert value_of(df, "ctx-lh-hemisphere") == 9
        assert set(df["Metric"]) == {"nvertices"}
        assert set(df["Units"]) == {"au"}

    def test_counts_are_integers(self, grid, annot):
        df, _ = cltmorph.compute_reg_nvertices_fromsurf(
            grid, annot, "lh", add_bids_entities=False
        )
        assert pd.api.types.is_integer_dtype(df["Value"])

    def test_surface_type_is_stored_in_the_source(self, grid, annot):
        df, _ = cltmorph.compute_reg_nvertices_fromsurf(
            grid, annot, "lh", surf_type="pial", add_bids_entities=False
        )
        assert set(df["Source"]) == {"pial"}

    def test_regions_add_up_to_the_vertices(self):
        sphere = cltsurf.Surface.simulate_surface(seed=3, shape="sphere", maps=None)
        annot = cltfree.AnnotParcellation.simulate_annotation(sphere, n_regions=7)
        df, _ = cltmorph.compute_reg_nvertices_fromsurf(
            sphere, annot, "lh", add_bids_entities=False
        )
        regions = df[df["Region"] != "ctx-lh-hemisphere"]
        assert regions["Value"].sum() == sphere.mesh.n_points
        assert value_of(df, "ctx-lh-hemisphere") == sphere.mesh.n_points

    def test_medial_wall_is_excluded_by_default(self, grid, annot_with_medialwall):
        df, _ = cltmorph.compute_reg_nvertices_fromsurf(
            grid, annot_with_medialwall, "lh", add_bids_entities=False
        )
        assert "ctx-lh-medialwall" not in df["Region"].tolist()
        assert value_of(df, "ctx-lh-hemisphere") == 6

    def test_medial_wall_can_be_included(self, grid, annot_with_medialwall):
        df, _ = cltmorph.compute_reg_nvertices_fromsurf(
            grid,
            annot_with_medialwall,
            "lh",
            include_unknown=True,
            add_bids_entities=False,
        )
        assert value_of(df, "ctx-lh-medialwall") == 3
        assert value_of(df, "ctx-lh-hemisphere") == 9

    def test_without_the_global_row(self, grid, annot):
        df, _ = cltmorph.compute_reg_nvertices_fromsurf(
            grid, annot, "lh", include_global=False, add_bids_entities=False
        )
        assert "ctx-lh-hemisphere" not in df["Region"].tolist()

    def test_region_table(self, grid, annot):
        df, _ = cltmorph.compute_reg_nvertices_fromsurf(
            grid, annot, "lh", table_type="region", add_bids_entities=False
        )
        assert df["ctx-lh-regA"].tolist() == [3]
        assert df["ctx-lh-regB"].tolist() == [6]

    def test_table_is_saved(self, tmp_path, grid, annot):
        out = tmp_path / "nvert.tsv"
        _, path = cltmorph.compute_reg_nvertices_fromsurf(
            grid, annot, "lh", output_table=out, add_bids_entities=False
        )
        assert path == str(out)
        assert pd.read_csv(out, sep="\t")["Value"].tolist() == [9, 3, 6]

    def test_annotation_must_match_the_surface(self, grid):
        annot = make_annot([10, 20], ["regA", "regB"], [10, 20])
        with pytest.raises(ValueError, match="annotation has 2 vertices"):
            cltmorph.compute_reg_nvertices_fromsurf(grid, annot, "lh")

    def test_repeated_region_names_are_rejected(self, grid):
        annot = make_annot(
            [10, 20, 20, 10, 20, 20, 10, 20, 20], ["same", "same"], [10, 20]
        )
        with pytest.raises(ValueError, match="share the same name"):
            cltmorph.compute_reg_nvertices_fromsurf(grid, annot, "lh")

    def test_invalid_table_type(self, grid, annot):
        with pytest.raises(ValueError, match="Invalid table_type"):
            cltmorph.compute_reg_nvertices_fromsurf(grid, annot, "lh", table_type="x")


####################################################################################################
class TestComputeRegValFromAnnot:
    VALUES = np.arange(9, dtype=float)

    def test_statistics_of_the_regions(self, annot):
        df, _, path = cltmorph.compute_reg_val_fromannot(
            self.VALUES, annot, "lh", metric="thickness", add_bids_entities=False
        )
        assert path is None
        # regA = vertices 0, 3, 6 and regB = vertices 1, 2, 4, 5, 7, 8
        a = df[df["Region"] == "ctx-lh-regA"].iloc[0]
        assert a["Value"] == pytest.approx(3.0)
        assert a["Median"] == pytest.approx(3.0)
        assert a["Std"] == pytest.approx(np.sqrt(6.0))
        assert (a["Min"], a["Max"]) == (0.0, 6.0)
        b = df[df["Region"] == "ctx-lh-regB"].iloc[0]
        assert b["Value"] == pytest.approx(4.5)
        assert b["Median"] == pytest.approx(4.5)
        assert (b["Min"], b["Max"]) == (1.0, 8.0)
        hemi = df[df["Region"] == "ctx-lh-hemisphere"].iloc[0]
        assert hemi["Value"] == pytest.approx(4.0)
        assert (hemi["Min"], hemi["Max"]) == (0.0, 8.0)

    def test_table_metadata(self, annot):
        df, _, _ = cltmorph.compute_reg_val_fromannot(
            self.VALUES, annot, "lh", metric="thickness", add_bids_entities=False
        )
        assert set(df["Source"]) == {"vertices"}
        assert set(df["Metric"]) == {"thickness"}
        assert set(df["Units"]) == {"mm"}
        assert set(df["Supraregion"]) == {"ctx"}
        assert set(df["Hemisphere"]) == {"lh"}
        assert df.columns.tolist()[-5:] == ["Value", "Median", "Std", "Min", "Max"]

    def test_vertex_values_are_returned(self, annot):
        _, values, _ = cltmorph.compute_reg_val_fromannot(
            self.VALUES, annot, "lh", add_bids_entities=False
        )
        np.testing.assert_array_equal(values, self.VALUES)

    def test_column_vector_is_accepted(self, annot):
        df, values, _ = cltmorph.compute_reg_val_fromannot(
            self.VALUES.reshape(-1, 1), annot, "lh", add_bids_entities=False
        )
        assert values.shape == (9,)
        assert value_of(df, "ctx-lh-regA") == pytest.approx(3.0)

    def test_custom_statistics(self, annot):
        df, _, _ = cltmorph.compute_reg_val_fromannot(
            self.VALUES,
            annot,
            "lh",
            stats_list=["count", "sum"],
            add_bids_entities=False,
        )
        assert df.columns.tolist()[-2:] == ["Count", "Sum"]
        a = df[df["Region"] == "ctx-lh-regA"].iloc[0]
        assert (a["Count"], a["Sum"]) == (3.0, 9.0)

    def test_single_statistic_as_a_string(self, annot):
        df, _, _ = cltmorph.compute_reg_val_fromannot(
            self.VALUES, annot, "lh", stats_list="max", add_bids_entities=False
        )
        assert df.columns.tolist()[-1] == "Max"
        assert value_of(df, "ctx-lh-regA", "Max") == 6.0

    def test_zeros_are_kept_by_default(self, annot):
        df, _, _ = cltmorph.compute_reg_val_fromannot(
            self.VALUES, annot, "lh", stats_list=["count"], add_bids_entities=False
        )
        assert value_of(df, "ctx-lh-regA", "Count") == 3.0

    def test_zeros_can_be_ignored(self, annot):
        df, _, _ = cltmorph.compute_reg_val_fromannot(
            self.VALUES,
            annot,
            "lh",
            nonzeros_only=True,
            stats_list=["count", "mean", "min"],
            add_bids_entities=False,
        )
        # Vertex 0 (value 0) belongs to regA
        assert value_of(df, "ctx-lh-regA", "Count") == 2.0
        assert value_of(df, "ctx-lh-regA", "Mean") == pytest.approx(4.5)
        assert value_of(df, "ctx-lh-regA", "Min") == 3.0
        assert value_of(df, "ctx-lh-hemisphere", "Count") == 8.0

    def test_region_table(self, annot):
        df, _, _ = cltmorph.compute_reg_val_fromannot(
            self.VALUES, annot, "lh", table_type="region", add_bids_entities=False
        )
        assert df["Statistics"].tolist() == ["Value", "Median", "Std", "Min", "Max"]
        assert df["ctx-lh-regA"].tolist() == pytest.approx(
            [3.0, 3.0, np.sqrt(6.0), 0.0, 6.0]
        )

    def test_without_the_global_row(self, annot):
        df, _, _ = cltmorph.compute_reg_val_fromannot(
            self.VALUES, annot, "lh", include_global=False, add_bids_entities=False
        )
        assert df["Region"].tolist() == ["ctx-lh-regA", "ctx-lh-regB"]

    def test_medial_wall_is_excluded_by_default(self, annot_with_medialwall):
        df, _, _ = cltmorph.compute_reg_val_fromannot(
            self.VALUES, annot_with_medialwall, "lh", add_bids_entities=False
        )
        assert "ctx-lh-medialwall" not in df["Region"].tolist()
        # The hemisphere only has the vertices 0-5
        assert value_of(df, "ctx-lh-hemisphere") == pytest.approx(2.5)

    def test_medial_wall_can_be_included(self, annot_with_medialwall):
        df, _, _ = cltmorph.compute_reg_val_fromannot(
            self.VALUES,
            annot_with_medialwall,
            "lh",
            include_unknown=True,
            add_bids_entities=False,
        )
        assert value_of(df, "ctx-lh-medialwall") == pytest.approx(7.0)
        assert value_of(df, "ctx-lh-hemisphere") == pytest.approx(4.0)

    def test_vertices_with_a_code_missing_from_the_table_are_ignored(self):
        annot = make_annot(
            [10, 20, 20, 10, 99, 99, 10, 20, 20], ["regA", "regB"], [10, 20]
        )
        df, _, _ = cltmorph.compute_reg_val_fromannot(
            self.VALUES, annot, "lh", stats_list=["count"], add_bids_entities=False
        )
        assert value_of(df, "ctx-lh-regB", "Count") == 4.0
        assert value_of(df, "ctx-lh-hemisphere", "Count") == 7.0

    def test_annotation_object_is_not_modified(self, annot_with_medialwall):
        codes = annot_with_medialwall.codes.copy()
        cltmorph.compute_reg_val_fromannot(
            self.VALUES, annot_with_medialwall, "lh", add_bids_entities=False
        )
        np.testing.assert_array_equal(annot_with_medialwall.codes, codes)
        assert annot_with_medialwall.regnames == ["regA", "regB", "medialwall"]

    def test_input_values_are_not_modified(self, annot):
        values = self.VALUES.copy()
        cltmorph.compute_reg_val_fromannot(
            values, annot, "lh", metric="area", add_bids_entities=False
        )
        np.testing.assert_array_equal(values, self.VALUES)

    def test_units_come_from_the_configuration(self, annot):
        df, _, _ = cltmorph.compute_reg_val_fromannot(
            self.VALUES, annot, "lh", metric="curv", add_bids_entities=False
        )
        assert set(df["Units"]) == {"mm-1"}

    def test_unknown_metric_has_unknown_units(self, annot):
        df, _, _ = cltmorph.compute_reg_val_fromannot(
            self.VALUES, annot, "lh", add_bids_entities=False
        )
        assert set(df["Metric"]) == {"unknown"}
        assert set(df["Units"]) == {"unknown"}

    def test_units_can_be_given(self, annot):
        df, _, _ = cltmorph.compute_reg_val_fromannot(
            self.VALUES,
            annot,
            "lh",
            metric="myelin",
            units="a.u.",
            add_bids_entities=False,
        )
        assert set(df["Units"]) == {"a.u."}

    def test_area_is_converted_to_the_configured_units(self, cm_units, annot):
        df, values, _ = cltmorph.compute_reg_val_fromannot(
            self.VALUES * 100, annot, "lh", metric="area", add_bids_entities=False
        )
        assert set(df["Units"]) == {"cm2"}
        np.testing.assert_allclose(values, self.VALUES)
        assert value_of(df, "ctx-lh-regA") == pytest.approx(3.0)

    def test_volume_is_converted_to_the_configured_units(self, cm_units, annot):
        df, values, _ = cltmorph.compute_reg_val_fromannot(
            self.VALUES * 1000, annot, "lh", metric="Volume", add_bids_entities=False
        )
        assert set(df["Units"]) == {"cm3"}
        np.testing.assert_allclose(values, self.VALUES)

    def test_units_of_an_area_are_ignored_with_a_warning(self, annot):
        with pytest.warns(UserWarning, match="units='cm2' ignored"):
            df, _, _ = cltmorph.compute_reg_val_fromannot(
                self.VALUES,
                annot,
                "lh",
                metric="area",
                units="cm2",
                add_bids_entities=False,
            )
        assert set(df["Units"]) == {"mm2"}

    def test_matching_units_of_an_area_do_not_warn(self, annot):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cltmorph.compute_reg_val_fromannot(
                self.VALUES,
                annot,
                "lh",
                metric="area",
                units="mm2",
                add_bids_entities=False,
            )

    def test_metric_file(self, tmp_path, annot):
        path = tmp_path / "sub-01_ses-M00_hemi-L_thickness.curv"
        nib.freesurfer.io.write_morph_data(str(path), self.VALUES.astype(np.float32))
        df, values, _ = cltmorph.compute_reg_val_fromannot(
            path, annot, "lh", metric="thickness"
        )
        np.testing.assert_allclose(values, self.VALUES)
        assert set(df["MetricFile"]) == {str(path)}
        assert df["Participant"].unique().tolist() == ["01"]
        assert df["Session"].unique().tolist() == ["M00"]
        assert value_of(df, "ctx-lh-regB") == pytest.approx(4.5)

    def test_gifti_metric_file(self, tmp_path, annot):
        path = tmp_path / "sub-01_ses-M00_hemi-L_thickness.shape.gii"
        gii = nib.gifti.GiftiImage()
        gii.add_gifti_data_array(
            nib.gifti.GiftiDataArray(
                self.VALUES.astype(np.float32), intent="NIFTI_INTENT_SHAPE"
            )
        )
        nib.save(gii, str(path))
        df, values, _ = cltmorph.compute_reg_val_fromannot(
            path, annot, "lh", metric="thickness"
        )
        np.testing.assert_allclose(values, self.VALUES)
        assert set(df["MetricFile"]) == {str(path)}
        assert df["Participant"].unique().tolist() == ["01"]
        assert value_of(df, "ctx-lh-regB") == pytest.approx(4.5)

    def test_gifti_metric_file_with_several_arrays(self, tmp_path, annot):
        path = tmp_path / "two_maps.shape.gii"
        gii = nib.gifti.GiftiImage()
        for _ in range(2):
            gii.add_gifti_data_array(
                nib.gifti.GiftiDataArray(
                    self.VALUES.astype(np.float32), intent="NIFTI_INTENT_SHAPE"
                )
            )
        nib.save(gii, str(path))
        with pytest.raises(ValueError, match="data arrays"):
            cltmorph.compute_reg_val_fromannot(path, annot, "lh")

    def test_without_bids_entities(self, tmp_path, annot):
        path = tmp_path / "sub-01_ses-M00_thickness.curv"
        nib.freesurfer.io.write_morph_data(str(path), self.VALUES.astype(np.float32))
        df, _, _ = cltmorph.compute_reg_val_fromannot(
            path, annot, "lh", add_bids_entities=False
        )
        assert "Participant" not in df.columns
        assert "Session" not in df.columns

    def test_annotation_file(self, tmp_path):
        sphere = cltsurf.Surface.simulate_surface(seed=6, shape="sphere", maps=None)
        annot = cltfree.AnnotParcellation.simulate_annotation(sphere, n_regions=3)
        annot_path = tmp_path / "lh.aparc.annot"
        annot.save_annotation(str(annot_path))
        values = np.random.default_rng(0).random(sphere.mesh.n_points)

        from_file, _, _ = cltmorph.compute_reg_val_fromannot(
            values, annot_path, "lh", add_bids_entities=False
        )
        from_object, _, _ = cltmorph.compute_reg_val_fromannot(
            values, annot, "lh", add_bids_entities=False
        )
        pd.testing.assert_frame_equal(from_file, from_object)

    def test_table_is_saved(self, tmp_path, annot):
        out = tmp_path / "thickness.tsv"
        df, _, path = cltmorph.compute_reg_val_fromannot(
            self.VALUES, annot, "lh", output_table=out, add_bids_entities=False
        )
        assert path == str(out)
        saved = pd.read_csv(out, sep="\t")
        assert saved.columns.tolist() == df.columns.tolist()
        np.testing.assert_allclose(saved["Value"], df["Value"])

    def test_number_of_values_must_match_the_annotation(self, annot):
        with pytest.raises(ValueError, match="annotation has 9 vertices"):
            cltmorph.compute_reg_val_fromannot(np.arange(5.0), annot, "lh")

    def test_missing_metric_file(self, tmp_path, annot):
        with pytest.raises(FileNotFoundError, match="Metric file not found"):
            cltmorph.compute_reg_val_fromannot(tmp_path / "lh.thickness", annot, "lh")

    def test_invalid_metric_type(self, annot):
        with pytest.raises(TypeError, match="metric_file must be"):
            cltmorph.compute_reg_val_fromannot([1, 2, 3], annot, "lh")

    def test_invalid_annotation_type(self):
        with pytest.raises(TypeError, match="parc_file must be"):
            cltmorph.compute_reg_val_fromannot(self.VALUES, 5, "lh")

    def test_unsupported_statistic(self, annot):
        with pytest.raises(ValueError, match="Unsupported statistics"):
            cltmorph.compute_reg_val_fromannot(
                self.VALUES, annot, "lh", stats_list=["variance"]
            )

    def test_invalid_table_type(self, annot):
        with pytest.raises(ValueError, match="Invalid table_type"):
            cltmorph.compute_reg_val_fromannot(self.VALUES, annot, "lh", table_type="x")

    def test_output_directory_must_exist(self, tmp_path, annot):
        with pytest.raises(FileNotFoundError, match="Directory does not exist"):
            cltmorph.compute_reg_val_fromannot(
                self.VALUES, annot, "lh", output_table=tmp_path / "no" / "t.tsv"
            )

    def test_repeated_region_names_are_rejected(self):
        annot = make_annot(
            [10, 20, 20, 10, 20, 20, 10, 20, 20], ["same", "same"], [10, 20]
        )
        with pytest.raises(ValueError, match="share the same name"):
            cltmorph.compute_reg_val_fromannot(self.VALUES, annot, "lh")


####################################################################################################
def parcellation_array():
    """Region 1 has 2 voxels, region 2 has 16 voxels and region 7 has 1 voxel."""
    arr = np.zeros((4, 4, 4), dtype=int)
    arr[0, 0, :2] = 1
    arr[1, :, :] = 2
    arr[3, 3, 3] = 7
    return arr


@pytest.fixture
def parc_arr():
    return parcellation_array()


@pytest.fixture
def metric_vol():
    """Metric equal to the voxel number: 0..63."""
    return np.arange(64, dtype=float).reshape(4, 4, 4)


@pytest.fixture
def named_parc(parc_arr):
    """Parcellation with names for the labels 1, 2 and 7."""
    colortable = {
        "index": [1, 2, 7],
        "name": ["left", "right", "other"],
        "color": ["#ff0000", "#00ff00", "#0000ff"],
    }
    return cltparc.Parcellation(parc_file=parc_arr, color_table=colortable)


@pytest.fixture
def parc_files(tmp_path, parc_arr, metric_vol):
    """BIDS-named parcellation and metric files on a 2 mm grid."""
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    parc_path = tmp_path / "sub-01_ses-M00_desc-x_dseg.nii.gz"
    metric_path = tmp_path / "sub-01_ses-M00_FA.nii.gz"
    nib.save(nib.Nifti1Image(parc_arr.astype(np.int16), affine), parc_path)
    nib.save(nib.Nifti1Image(metric_vol.astype(np.float32), affine), metric_path)
    return metric_path, parc_path


class TestComputeRegValFromParcellation:
    def test_statistics_of_the_regions(self, metric_vol, parc_arr):
        df, values, path = cltmorph.compute_reg_val_fromparcellation(
            metric_vol, parc_arr, metric="fa", add_bids_entities=False
        )
        assert path is None
        np.testing.assert_array_equal(values, metric_vol)
        # Region 1: voxels with values 0 and 1. Region 2: the values 16..31
        r1 = df[df["Region"] == "auto-roi-000001"].iloc[0]
        assert (r1["Value"], r1["Median"], r1["Min"], r1["Max"]) == (0.5, 0.5, 0.0, 1.0)
        assert r1["Std"] == pytest.approx(0.5)
        r2 = df[df["Region"] == "auto-roi-000002"].iloc[0]
        assert (r2["Value"], r2["Median"], r2["Min"], r2["Max"]) == (23.5, 23.5, 16, 31)
        assert r2["Std"] == pytest.approx(np.sqrt((16**2 - 1) / 12))
        assert value_of(df, "auto-roi-000007") == 63.0

    def test_global_row_uses_all_the_labeled_voxels(self, metric_vol, parc_arr):
        df, _, _ = cltmorph.compute_reg_val_fromparcellation(
            metric_vol,
            parc_arr,
            stats_list=["count", "sum"],
            add_bids_entities=False,
        )
        labeled = metric_vol[parc_arr != 0]
        assert value_of(df, GLOBAL, "Count") == labeled.size == 19
        assert value_of(df, GLOBAL, "Sum") == labeled.sum()

    def test_table_metadata(self, metric_vol, parc_arr):
        df, _, _ = cltmorph.compute_reg_val_fromparcellation(
            metric_vol, parc_arr, metric="fa", add_bids_entities=False
        )
        assert set(df["Source"]) == {"volume"}
        assert set(df["Metric"]) == {"fa"}
        assert set(df["Units"]) == {"au"}
        assert df.columns.tolist()[-5:] == ["Value", "Median", "Std", "Min", "Max"]
        assert df["Region"].tolist()[0] == GLOBAL

    def test_region_names_come_from_the_color_table(self, metric_vol, named_parc):
        df, _, _ = cltmorph.compute_reg_val_fromparcellation(
            metric_vol, named_parc, add_bids_entities=False
        )
        assert df["Region"].tolist() == [GLOBAL, "left", "right", "other"]
        assert value_of(df, "left") == 0.5

    def test_labels_missing_from_the_color_table_get_generated_names(
        self, metric_vol, parc_arr
    ):
        colortable = {"index": [1], "name": ["left"], "color": ["#ff0000"]}
        parc = cltparc.Parcellation(parc_file=parc_arr, color_table=colortable)
        df, _, _ = cltmorph.compute_reg_val_fromparcellation(
            metric_vol, parc, add_bids_entities=False, region_prefix="roi-x"
        )
        assert df["Region"].tolist()[:2] == [GLOBAL, "left"]
        assert df["Region"].tolist()[2].startswith("roi-x-")

    def test_exclude_by_code(self, metric_vol, named_parc):
        df, _, _ = cltmorph.compute_reg_val_fromparcellation(
            metric_vol,
            named_parc,
            exclude_by_code=[1],
            stats_list=["count"],
            add_bids_entities=False,
        )
        assert df["Region"].tolist() == [GLOBAL, "right", "other"]
        assert value_of(df, GLOBAL, "Count") == 17

    def test_exclude_by_name(self, metric_vol, named_parc):
        df, _, _ = cltmorph.compute_reg_val_fromparcellation(
            metric_vol, named_parc, exclude_by_name="left", add_bids_entities=False
        )
        assert df["Region"].tolist() == [GLOBAL, "right", "other"]

    def test_include_by_code(self, metric_vol, named_parc):
        df, _, _ = cltmorph.compute_reg_val_fromparcellation(
            metric_vol,
            named_parc,
            include_by_code=[2],
            stats_list=["count"],
            add_bids_entities=False,
        )
        assert df["Region"].tolist() == [GLOBAL, "right"]
        assert value_of(df, GLOBAL, "Count") == 16

    def test_include_by_name(self, metric_vol, named_parc):
        df, _, _ = cltmorph.compute_reg_val_fromparcellation(
            metric_vol,
            named_parc,
            include_by_name=["right", "other"],
            add_bids_entities=False,
        )
        assert df["Region"].tolist() == [GLOBAL, "right", "other"]

    def test_exclusions_are_applied_before_inclusions(self, metric_vol, named_parc):
        df, _, _ = cltmorph.compute_reg_val_fromparcellation(
            metric_vol,
            named_parc,
            exclude_by_code=[2],
            include_by_code=[1, 2],
            add_bids_entities=False,
        )
        assert df["Region"].tolist() == [GLOBAL, "left"]

    def test_filters_do_not_modify_the_parcellation(self, metric_vol, named_parc):
        data = named_parc.data.copy()
        cltmorph.compute_reg_val_fromparcellation(
            metric_vol, named_parc, exclude_by_code=[1], add_bids_entities=False
        )
        np.testing.assert_array_equal(named_parc.data, data)
        assert named_parc.index == [1, 2, 7]

    def test_nothing_left_after_filtering(self, metric_vol, named_parc):
        with pytest.raises(ValueError, match="No valid regions"):
            cltmorph.compute_reg_val_fromparcellation(
                metric_vol, named_parc, exclude_by_code=[1, 2, 7]
            )

    def test_parcellation_without_labels(self, metric_vol):
        with pytest.raises(ValueError, match="No valid regions"):
            cltmorph.compute_reg_val_fromparcellation(
                metric_vol, np.zeros((4, 4, 4), dtype=int)
            )

    def test_without_the_global_row(self, metric_vol, parc_arr):
        df, _, _ = cltmorph.compute_reg_val_fromparcellation(
            metric_vol, parc_arr, include_global=False, add_bids_entities=False
        )
        assert GLOBAL not in df["Region"].tolist()
        assert len(df) == 3

    def test_zeros_can_be_ignored(self, metric_vol, parc_arr):
        df, _, _ = cltmorph.compute_reg_val_fromparcellation(
            metric_vol,
            parc_arr,
            nonzeros_only=True,
            stats_list=["count", "mean"],
            add_bids_entities=False,
        )
        # Region 1 has the values 0 and 1: only the 1 is kept
        assert value_of(df, "auto-roi-000001", "Count") == 1.0
        assert value_of(df, "auto-roi-000001", "Mean") == 1.0

    def test_region_table(self, metric_vol, parc_arr):
        df, _, _ = cltmorph.compute_reg_val_fromparcellation(
            metric_vol,
            parc_arr,
            table_type="region",
            stats_list=["mean", "max"],
            add_bids_entities=False,
        )
        assert df["Statistics"].tolist() == ["Mean", "Max"]
        assert df["auto-roi-000002"].tolist() == [23.5, 31.0]

    def test_units_can_be_given(self, metric_vol, parc_arr):
        df, _, _ = cltmorph.compute_reg_val_fromparcellation(
            metric_vol, parc_arr, metric="fa", units="a.u.", add_bids_entities=False
        )
        assert set(df["Units"]) == {"a.u."}

    def test_trailing_singleton_dimension_is_dropped(self, metric_vol, parc_arr):
        df, values, _ = cltmorph.compute_reg_val_fromparcellation(
            metric_vol[..., np.newaxis], parc_arr, add_bids_entities=False
        )
        assert values.shape == (4, 4, 4)
        assert value_of(df, "auto-roi-000001") == 0.5

    def test_input_arrays_are_not_modified(self, metric_vol, parc_arr):
        metric_before, parc_before = metric_vol.copy(), parc_arr.copy()
        cltmorph.compute_reg_val_fromparcellation(
            metric_vol, parc_arr, exclude_by_code=[1], add_bids_entities=False
        )
        np.testing.assert_array_equal(metric_vol, metric_before)
        np.testing.assert_array_equal(parc_arr, parc_before)

    def test_files(self, parc_files):
        metric_path, parc_path = parc_files
        df, values, _ = cltmorph.compute_reg_val_fromparcellation(
            metric_path, parc_path, metric="fa"
        )
        assert values.shape == (4, 4, 4)
        assert set(df["MetricFile"]) == {str(metric_path)}
        assert df["Participant"].unique().tolist() == ["01"]
        assert df["Session"].unique().tolist() == ["M00"]
        assert value_of(df, "auto-roi-000002") == 23.5

    def test_metric_file_with_another_grid_is_resampled(self, tmp_path, parc_files):
        _, parc_path = parc_files
        # Constant map on a 1 mm grid that covers the 2 mm parcellation grid
        finer = tmp_path / "sub-01_ses-M00_MD.nii.gz"
        nib.save(
            nib.Nifti1Image(np.full((8, 8, 8), 5.0, dtype=np.float32), np.eye(4)), finer
        )
        with pytest.warns(UserWarning, match="Resampling the metric"):
            df, values, _ = cltmorph.compute_reg_val_fromparcellation(finer, parc_path)
        assert values.shape == (4, 4, 4)
        np.testing.assert_allclose(values, 5.0, atol=1e-4)
        assert value_of(df, GLOBAL) == pytest.approx(5.0, abs=1e-4)

    def test_metric_file_with_the_same_shape_but_another_affine_is_resampled(
        self, tmp_path, parc_files
    ):
        _, parc_path = parc_files
        shifted = tmp_path / "sub-01_ses-M00_MD.nii.gz"
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        affine[:3, 3] = 2.0  # one voxel away from the parcellation
        data = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
        nib.save(nib.Nifti1Image(data, affine), shifted)
        with pytest.warns(UserWarning, match="Resampling the metric"):
            _, values, _ = cltmorph.compute_reg_val_fromparcellation(
                shifted, parc_path, interp_method="nearest", add_bids_entities=False
            )
        # The parcellation voxel (i, j, k) sees the metric voxel (i-1, j-1, k-1)
        assert values[1, 1, 1] == data[0, 0, 0]
        assert values[2, 3, 1] == data[1, 2, 0]

    def test_interpolation_method_of_the_resampling(self, tmp_path, parc_arr):
        # The centers of the 2 mm voxels are 0.5 mm away from those of the 1 mm voxels,
        # and the step of the map lies between the first and second voxels
        parc_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        parc_affine[:3, 3] = 0.5
        parc_path = tmp_path / "sub-01_dseg.nii.gz"
        nib.save(nib.Nifti1Image(parc_arr.astype(np.int16), parc_affine), parc_path)
        data = np.full((8, 8, 8), 9.0, dtype=np.float32)
        data[:3] = 3.0
        finer = tmp_path / "sub-01_MD.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), finer)

        _, nearest, _ = cltmorph.compute_reg_val_fromparcellation(
            finer, parc_path, interp_method="nearest", add_bids_entities=False
        )
        _, linear, _ = cltmorph.compute_reg_val_fromparcellation(
            finer, parc_path, interp_method="linear", add_bids_entities=False
        )
        assert set(np.unique(nearest)) <= {3.0, 9.0}
        assert linear[1, 0, 0] == pytest.approx(6.0)
        assert linear[0, 0, 0] == pytest.approx(3.0)

    def test_parcellation_object_with_a_file_resamples_the_metric(self, tmp_path):
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        parc = cltparc.Parcellation(parc_file=parcellation_array(), affine=affine)
        finer = tmp_path / "FA.nii.gz"
        nib.save(
            nib.Nifti1Image(np.full((8, 8, 8), 2.0, dtype=np.float32), np.eye(4)), finer
        )
        _, values, _ = cltmorph.compute_reg_val_fromparcellation(
            finer, parc, add_bids_entities=False
        )
        assert values.shape == (4, 4, 4)

    def test_metric_file_must_be_3d(self, tmp_path, parc_files):
        _, parc_path = parc_files
        path = tmp_path / "multi.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((4, 4, 4, 2), dtype=np.float32), np.eye(4)), path
        )
        with pytest.raises(ValueError, match="must be 3D"):
            cltmorph.compute_reg_val_fromparcellation(path, parc_path)

    def test_metric_file_with_a_singleton_fourth_dimension(self, tmp_path, parc_files):
        _, parc_path = parc_files
        path = tmp_path / "sub-01_ses-M00_4d.nii.gz"
        nib.save(
            nib.Nifti1Image(
                np.ones((4, 4, 4, 1), dtype=np.float32), np.diag([2, 2, 2, 1])
            ),
            path,
        )
        df, _, _ = cltmorph.compute_reg_val_fromparcellation(path, parc_path)
        assert value_of(df, GLOBAL) == 1.0

    def test_array_with_another_shape_is_rejected(self, parc_arr):
        with pytest.raises(ValueError, match="does not match parcellation shape"):
            cltmorph.compute_reg_val_fromparcellation(np.zeros((5, 5, 5)), parc_arr)

    def test_invalid_interpolation(self, metric_vol, parc_arr):
        with pytest.raises(ValueError, match="Invalid interp_method"):
            cltmorph.compute_reg_val_fromparcellation(
                metric_vol, parc_arr, interp_method="spline"
            )

    def test_missing_files(self, tmp_path, metric_vol, parc_arr, parc_files):
        with pytest.raises(FileNotFoundError, match="Parcellation file not found"):
            cltmorph.compute_reg_val_fromparcellation(metric_vol, tmp_path / "p.nii.gz")
        with pytest.raises(FileNotFoundError, match="Metric file not found"):
            cltmorph.compute_reg_val_fromparcellation(tmp_path / "m.nii.gz", parc_arr)

    def test_invalid_input_types(self, metric_vol, parc_arr):
        with pytest.raises(TypeError, match="metric_file must be"):
            cltmorph.compute_reg_val_fromparcellation([1, 2], parc_arr)
        with pytest.raises(TypeError, match="parc_file must be"):
            cltmorph.compute_reg_val_fromparcellation(metric_vol, 5)

    def test_repeated_region_names_are_rejected(self, metric_vol, parc_arr):
        colortable = {
            "index": [1, 2, 7],
            "name": ["same", "same", "other"],
            "color": ["#ff0000", "#00ff00", "#0000ff"],
        }
        parc = cltparc.Parcellation(parc_file=parc_arr, color_table=colortable)
        with pytest.raises(ValueError, match="share the same name"):
            cltmorph.compute_reg_val_fromparcellation(metric_vol, parc)

    def test_unsupported_statistic(self, metric_vol, parc_arr):
        with pytest.raises(ValueError, match="Unsupported statistics"):
            cltmorph.compute_reg_val_fromparcellation(
                metric_vol, parc_arr, stats_list=["mode"]
            )

    def test_table_is_saved(self, tmp_path, metric_vol, parc_arr):
        out = tmp_path / "fa.tsv"
        df, _, path = cltmorph.compute_reg_val_fromparcellation(
            metric_vol, parc_arr, output_table=out, add_bids_entities=False
        )
        assert path == str(out)
        saved = pd.read_csv(out, sep="\t")
        assert saved["Region"].tolist() == df["Region"].tolist()
        np.testing.assert_allclose(saved["Value"], df["Value"])


####################################################################################################
class TestComputeRegVolumeFromParcellation:
    def test_volumes_of_a_2mm_grid(self, parc_files):
        _, parc_path = parc_files
        df, path = cltmorph.compute_reg_volume_fromparcellation(parc_path)
        assert path is None
        # 2 x 2 x 2 mm voxels have 8 mm3
        assert value_of(df, "auto-roi-000001") == pytest.approx(16.0)
        assert value_of(df, "auto-roi-000002") == pytest.approx(128.0)
        assert value_of(df, "auto-roi-000007") == pytest.approx(8.0)
        assert value_of(df, GLOBAL) == pytest.approx(152.0)
        assert set(df["Source"]) == {"parcellation"}
        assert set(df["Metric"]) == {"volume"}
        assert set(df["Units"]) == {"mm3"}
        assert df["Participant"].unique().tolist() == ["01"]

    def test_arrays_have_one_millimeter_voxels(self, parc_arr):
        df, _ = cltmorph.compute_reg_volume_fromparcellation(parc_arr)
        assert value_of(df, "auto-roi-000001") == 2.0
        assert value_of(df, "auto-roi-000002") == 16.0
        assert value_of(df, GLOBAL) == 19.0
        assert set(df["MetricFile"]) == {"3darray"}

    def test_anisotropic_voxels(self, parc_arr):
        parc = cltparc.Parcellation(
            parc_file=parc_arr, affine=np.diag([1.0, 2.0, 3.0, 1.0])
        )
        df, _ = cltmorph.compute_reg_volume_fromparcellation(parc)
        assert value_of(df, "auto-roi-000001") == 12.0  # 2 voxels of 6 mm3

    def test_rotated_voxel_grid(self, parc_arr):
        angle = np.pi / 5
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        affine = np.eye(4)
        affine[:3, :3] = rotation * 2.0
        parc = cltparc.Parcellation(parc_file=parc_arr, affine=affine)
        df, _ = cltmorph.compute_reg_volume_fromparcellation(parc)
        assert value_of(df, "auto-roi-000001") == pytest.approx(16.0)

    def test_flipped_axis_keeps_the_volume_positive(self, parc_arr):
        parc = cltparc.Parcellation(
            parc_file=parc_arr, affine=np.diag([-2.0, 2.0, 2.0, 1.0])
        )
        df, _ = cltmorph.compute_reg_volume_fromparcellation(parc)
        assert value_of(df, "auto-roi-000001") == pytest.approx(16.0)

    def test_translation_does_not_change_the_volume(self, parc_arr):
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        affine[:3, 3] = [-50, 20, 7]
        parc = cltparc.Parcellation(parc_file=parc_arr, affine=affine)
        df, _ = cltmorph.compute_reg_volume_fromparcellation(parc)
        assert value_of(df, "auto-roi-000001") == pytest.approx(16.0)

    def test_volumes_are_given_in_the_configured_units(self, cm_units, parc_files):
        _, parc_path = parc_files
        df, _ = cltmorph.compute_reg_volume_fromparcellation(parc_path)
        assert set(df["Units"]) == {"cm3"}
        assert value_of(df, "auto-roi-000002") == pytest.approx(0.128)
        assert value_of(df, GLOBAL) == pytest.approx(0.152)

    def test_names_come_from_the_color_table(self, named_parc):
        df, _ = cltmorph.compute_reg_volume_fromparcellation(named_parc)
        assert df["Region"].tolist() == [GLOBAL, "left", "right", "other"]
        assert df["Value"].tolist() == [19.0, 2.0, 16.0, 1.0]

    def test_filters(self, named_parc):
        df, _ = cltmorph.compute_reg_volume_fromparcellation(
            named_parc, exclude_by_code=[2]
        )
        assert df["Region"].tolist() == [GLOBAL, "left", "other"]
        assert value_of(df, GLOBAL) == 3.0

        df, _ = cltmorph.compute_reg_volume_fromparcellation(
            named_parc, include_by_name=["right"]
        )
        assert df["Region"].tolist() == [GLOBAL, "right"]

        df, _ = cltmorph.compute_reg_volume_fromparcellation(
            named_parc, exclude_by_name=["right"], include_by_code=[1, 2]
        )
        assert df["Region"].tolist() == [GLOBAL, "left"]

    def test_filters_do_not_modify_the_parcellation(self, named_parc):
        data = named_parc.data.copy()
        cltmorph.compute_reg_volume_fromparcellation(named_parc, exclude_by_code=[1])
        np.testing.assert_array_equal(named_parc.data, data)

    def test_without_the_global_row(self, parc_arr):
        df, _ = cltmorph.compute_reg_volume_fromparcellation(
            parc_arr, include_global=False
        )
        assert GLOBAL not in df["Region"].tolist()

    def test_region_table(self, parc_arr):
        df, _ = cltmorph.compute_reg_volume_fromparcellation(
            parc_arr, table_type="region"
        )
        assert df["Statistics"].tolist() == ["Value"]
        assert df["auto-roi-000002"].tolist() == [16.0]

    def test_custom_prefix_for_labels_missing_from_the_color_table(self, parc_arr):
        colortable = {"index": [1], "name": ["left"], "color": ["#ff0000"]}
        parc = cltparc.Parcellation(parc_file=parc_arr, color_table=colortable)
        df, _ = cltmorph.compute_reg_volume_fromparcellation(
            parc, region_prefix="roi-x"
        )
        regions = df["Region"].tolist()
        assert regions[:2] == [GLOBAL, "left"]
        assert all(r.startswith("roi-x-") for r in regions[2:])
        assert len(regions) == 4

    def test_labels_do_not_need_to_be_contiguous(self):
        arr = np.zeros((3, 3, 3), dtype=int)
        arr[0, 0, 0] = 5
        arr[1, 1, 1] = 900
        arr[2, 2, 2] = 900
        df, _ = cltmorph.compute_reg_volume_fromparcellation(arr)
        assert df["Value"].tolist() == [3.0, 1.0, 2.0]

    def test_empty_parcellation(self):
        with pytest.raises(ValueError, match="No valid regions"):
            cltmorph.compute_reg_volume_fromparcellation(np.zeros((3, 3, 3), dtype=int))

    def test_nothing_left_after_filtering(self, named_parc):
        with pytest.raises(ValueError, match="No valid regions"):
            cltmorph.compute_reg_volume_fromparcellation(
                named_parc, exclude_by_code=[1, 2, 7]
            )

    def test_repeated_region_names_are_rejected(self, parc_arr):
        colortable = {
            "index": [1, 2, 7],
            "name": ["same", "same", "other"],
            "color": ["#ff0000", "#00ff00", "#0000ff"],
        }
        parc = cltparc.Parcellation(parc_file=parc_arr, color_table=colortable)
        with pytest.raises(ValueError, match="share the same name"):
            cltmorph.compute_reg_volume_fromparcellation(parc)

    def test_table_is_saved(self, tmp_path, parc_arr):
        out = tmp_path / "vol.tsv"
        df, path = cltmorph.compute_reg_volume_fromparcellation(
            parc_arr, output_table=out
        )
        assert path == str(out)
        saved = pd.read_csv(out, sep="\t")
        np.testing.assert_allclose(saved["Value"], df["Value"])

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Parcellation file not found"):
            cltmorph.compute_reg_volume_fromparcellation(tmp_path / "p.nii.gz")

    def test_invalid_input_type(self):
        with pytest.raises(TypeError, match="parc_file must be"):
            cltmorph.compute_reg_volume_fromparcellation([1, 2, 3])

    def test_invalid_table_type(self, parc_arr):
        with pytest.raises(ValueError, match="Invalid table_type"):
            cltmorph.compute_reg_volume_fromparcellation(parc_arr, table_type="x")


####################################################################################################
ASEG_STATS = """# Title Segmentation Statistics
#
# generating_program mri_segstats
# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1100000.000000, mm^3
# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, 1500000.000000, mm^3
# Measure TotalGray, TotalGrayVol, Total gray matter volume, 600000.000000, mm^3
# Measure Cortex, SurfaceHoles, Total number of defect holes, 12, unitless
# Measure BrainSegVol-to-eTIV, BrainSegVol-to-eTIV, Brain Segmentation Volume Ratio to Estimated Total Intracranial Volume, 0.7333, unitless
# Measure WhiteSurfArea, WhiteSurfArea, White Surface Total Area, 80000.0, mm^2
# NTableCols 10
# ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
  1  10   7000   7500.0  Left-Thalamus-Proper  90.0 5.0 40.0 110.0 70.0
  2  11   3000   3300.5  Left-Caudate  80.0 5.0 40.0 110.0 70.0
  3  49   7100   7600.0  Right-Thalamus  90.0 5.0 40.0 110.0 70.0
  4  4    2500   2750.0  Left-Lateral-Ventricle  30.0 5.0 10.0 50.0 40.0
"""

CUSTOM_GLOBAL_CONFIG = {
    "global": {
        "etiv": {"key": "Estimated Total Intracranial Volume"},
        "holes": {"key": "SurfaceHoles"},
        "ratio": {"key": "BrainSegVol-to-eTIV"},
        "white-area": {"key": "WhiteSurfArea"},
    }
}


@pytest.fixture
def aseg_file(tmp_path):
    path = tmp_path / "sub-01_ses-M00_aseg.stats"
    path.write_text(ASEG_STATS)
    return path


@pytest.fixture
def custom_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(CUSTOM_GLOBAL_CONFIG))
    return path


class TestParseFreesurferGlobalFromAseg:
    def test_default_measures(self, aseg_file):
        df, path = cltmorph.parse_freesurfer_global_fromaseg(
            aseg_file, add_bids_entities=False
        )
        assert path is None
        assert value_of(df, "brain-brain-intracraneal") == 1500000.0
        assert value_of(df, GLOBAL) == 1100000.0
        assert value_of(df, "gm-brain-graymatter") == 600000.0
        assert set(df["Source"]) == {"statsfile"}
        assert set(df["Metric"]) == {"volume"}
        assert set(df["Units"]) == {"mm3"}
        assert set(df["MetricFile"]) == {str(aseg_file)}

    def test_regions_are_split_into_supraregion_and_hemisphere(self, aseg_file):
        df, _ = cltmorph.parse_freesurfer_global_fromaseg(
            aseg_file, add_bids_entities=False
        )
        row = df[df["Region"] == "vent-lh-lateral"].iloc[0]
        assert (row["Supraregion"], row["Hemisphere"]) == ("vent", "lh")

    def test_measure_found_by_key_alternate_key_or_description(self, tmp_path):
        text = (
            "# Measure Whatever, eTIV, Something else, 100.0, mm^3\n"
            "# Measure BrainSeg, Other, Brain Segmentation Volume, 200.0, mm^3\n"
            "# Measure TotalGray, TotalGray, Another name, 300.0, mm^3\n"
        )
        path = tmp_path / "aseg.stats"
        path.write_text(text)
        df, _ = cltmorph.parse_freesurfer_global_fromaseg(
            path, include_missing=False, add_bids_entities=False
        )
        assert value_of(df, "brain-brain-intracraneal") == 100.0
        assert value_of(df, GLOBAL) == 200.0
        assert value_of(df, "gm-brain-graymatter") == 300.0

    def test_volumes_are_converted_to_the_configured_units(self, cm_units, aseg_file):
        df, _ = cltmorph.parse_freesurfer_global_fromaseg(
            aseg_file, add_bids_entities=False
        )
        assert set(df["Units"]) == {"cm3"}
        assert value_of(df, "brain-brain-intracraneal") == pytest.approx(1500.0)
        assert value_of(df, GLOBAL) == pytest.approx(1100.0)

    def test_missing_measures_are_reported_as_zero_with_a_warning(self, aseg_file):
        with pytest.warns(UserWarning, match="Total cerebral white matter volume"):
            df, _ = cltmorph.parse_freesurfer_global_fromaseg(
                aseg_file, add_bids_entities=False
            )
        assert value_of(df, "wm-brain-whitematter") == 0.0
        assert value_of(df, "vent-rh-lateral") == 0.0

    def test_missing_measures_can_be_left_out(self, aseg_file):
        with pytest.warns(UserWarning, match="not found"):
            df, _ = cltmorph.parse_freesurfer_global_fromaseg(
                aseg_file, include_missing=False, add_bids_entities=False
            )
        assert "wm-brain-whitematter" not in df["Region"].tolist()
        assert "vent-rh-lateral" not in df["Region"].tolist()
        assert GLOBAL in df["Region"].tolist()

    def test_ventricles_are_found_in_the_table(self, aseg_file):
        df, _ = cltmorph.parse_freesurfer_global_fromaseg(
            aseg_file, include_missing=False, add_bids_entities=False
        )
        assert value_of(df, "vent-lh-lateral") == 2750.0

    def test_custom_configuration_with_unitless_and_area_measures(
        self, custom_config, aseg_file
    ):
        df, _ = cltmorph.parse_freesurfer_global_fromaseg(
            aseg_file, config_json=custom_config, add_bids_entities=False
        )
        rows = {r["Region"].split("-")[-1]: r for _, r in df.iterrows()}
        assert df["Value"].tolist() == pytest.approx([1500000.0, 12.0, 0.7333, 80000.0])
        assert df["Metric"].tolist() == ["volume", "measure", "measure", "area"]
        assert df["Units"].tolist() == ["mm3", "au", "au", "mm2"]
        assert len(rows) == 4

    def test_area_measures_are_converted(self, cm_units, custom_config, aseg_file):
        df, _ = cltmorph.parse_freesurfer_global_fromaseg(
            aseg_file, config_json=custom_config, add_bids_entities=False
        )
        assert df["Units"].tolist() == ["cm3", "au", "au", "cm2"]
        assert df["Value"].iloc[3] == pytest.approx(800.0)
        assert df["Value"].iloc[1] == 12.0  # unitless: unchanged

    def test_region_table_with_mixed_units(self, custom_config, aseg_file):
        df, _ = cltmorph.parse_freesurfer_global_fromaseg(
            aseg_file,
            config_json=custom_config,
            table_type="region",
            add_bids_entities=False,
        )
        assert df["Metric"].tolist() == ["mixed"]
        assert df["Units"].tolist() == ["mixed"]
        assert df["Statistics"].tolist() == ["Value"]

    def test_region_table_with_a_single_unit(self, aseg_file):
        with pytest.warns(UserWarning):
            df, _ = cltmorph.parse_freesurfer_global_fromaseg(
                aseg_file,
                table_type="region",
                include_missing=False,
                add_bids_entities=False,
            )
        assert df["Metric"].tolist() == ["volume"]
        assert df["Units"].tolist() == ["mm3"]
        assert df[GLOBAL].tolist() == [1100000.0]

    def test_missing_configuration_falls_back_to_the_default(self, tmp_path, aseg_file):
        with pytest.warns(UserWarning, match="Using the default configuration"):
            df, _ = cltmorph.parse_freesurfer_global_fromaseg(
                aseg_file,
                config_json=tmp_path / "missing.json",
                include_missing=False,
                add_bids_entities=False,
            )
        assert GLOBAL in df["Region"].tolist()

    def test_invalid_configuration_falls_back_to_the_default(self, tmp_path, aseg_file):
        broken = tmp_path / "broken.json"
        broken.write_text("{nope")
        with pytest.warns(UserWarning, match="Error loading config file"):
            df, _ = cltmorph.parse_freesurfer_global_fromaseg(
                aseg_file,
                config_json=broken,
                include_missing=False,
                add_bids_entities=False,
            )
        assert GLOBAL in df["Region"].tolist()

    def test_configuration_without_level_key_is_used_as_it_is(
        self, tmp_path, aseg_file
    ):
        config = tmp_path / "flat.json"
        config.write_text(json.dumps({"etiv": {"key": "eTIV"}}))
        df, _ = cltmorph.parse_freesurfer_global_fromaseg(
            aseg_file, config_json=config, add_bids_entities=False
        )
        assert df["Value"].tolist() == [1500000.0]

    def test_nothing_found(self, tmp_path):
        path = tmp_path / "aseg.stats"
        path.write_text("# Nothing here\n")
        with pytest.warns(UserWarning):
            with pytest.raises(ValueError, match="No volume measurements"):
                cltmorph.parse_freesurfer_global_fromaseg(path, include_missing=False)

    def test_file_without_column_headers(self, tmp_path):
        text = (
            "# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1.0, mm^3\n"
            "  1  4   2500   2750.0  Left-Lateral-Ventricle  30.0\n"
        )
        path = tmp_path / "aseg.stats"
        path.write_text(text)
        df, _ = cltmorph.parse_freesurfer_global_fromaseg(
            path, include_missing=False, add_bids_entities=False
        )
        assert value_of(df, "vent-lh-lateral") == 2750.0

    def test_file_with_tablecol_headers(self, tmp_path):
        text = (
            "# TableCol 1 ColHeader Index\n"
            "# TableCol 2 ColHeader SegId\n"
            "# TableCol 3 ColHeader NVoxels\n"
            "# TableCol 4 ColHeader Volume_mm3\n"
            "# TableCol 5 ColHeader StructName\n"
            "  1  4   2500   2750.0  Left-Lateral-Ventricle\n"
        )
        path = tmp_path / "aseg.stats"
        path.write_text(text)
        df, _ = cltmorph.parse_freesurfer_global_fromaseg(
            path, include_missing=False, add_bids_entities=False
        )
        assert value_of(df, "vent-lh-lateral") == 2750.0

    def test_malformed_lines_are_skipped_with_a_warning(self, tmp_path):
        text = (
            "# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, abc, mm^3\n"
            "# Measure TotalGray, TotalGrayVol, Total gray matter volume, 5.0, mm^3\n"
        )
        path = tmp_path / "aseg.stats"
        path.write_text(text)
        with pytest.warns(UserWarning, match="Error parsing measure line"):
            df, _ = cltmorph.parse_freesurfer_global_fromaseg(
                path, include_missing=False, add_bids_entities=False
            )
        assert df["Region"].tolist() == ["gm-brain-graymatter"]

    def test_bids_entities(self, aseg_file):
        df, _ = cltmorph.parse_freesurfer_global_fromaseg(aseg_file)
        assert df["Participant"].unique().tolist() == ["01"]
        assert df["Session"].unique().tolist() == ["M00"]

    def test_table_is_saved(self, tmp_path, aseg_file):
        out = tmp_path / "global.tsv"
        df, path = cltmorph.parse_freesurfer_global_fromaseg(
            aseg_file, output_table=out, include_missing=False, add_bids_entities=False
        )
        assert path == str(out)
        saved = pd.read_csv(out, sep="\t")
        assert saved["Region"].tolist() == df["Region"].tolist()

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Stats file not found"):
            cltmorph.parse_freesurfer_global_fromaseg(tmp_path / "aseg.stats")

    def test_invalid_table_type(self, aseg_file):
        with pytest.raises(ValueError, match="Invalid table_type"):
            cltmorph.parse_freesurfer_global_fromaseg(aseg_file, table_type="x")

    def test_path_and_string_inputs(self, aseg_file):
        first, _ = cltmorph.parse_freesurfer_global_fromaseg(
            aseg_file, include_missing=False, add_bids_entities=False
        )
        second, _ = cltmorph.parse_freesurfer_global_fromaseg(
            str(aseg_file), include_missing=False, add_bids_entities=False
        )
        pd.testing.assert_frame_equal(first, second)


class TestParseFreesurferStatsFromAseg:
    def test_regions_found_by_name(self, aseg_file):
        with pytest.warns(UserWarning):
            df, path = cltmorph.parse_freesurfer_stats_fromaseg(
                aseg_file, add_bids_entities=False
            )
        assert path is None
        assert value_of(df, "thal-lh-thalamus") == 7500.0
        assert value_of(df, "subc-lh-caudate") == 3300.5
        assert set(df["Source"]) == {"statsfile"}
        assert set(df["Metric"]) == {"volume"}
        assert set(df["Units"]) == {"mm3"}

    def test_region_found_by_alternate_name(self, aseg_file):
        # The file has Right-Thalamus, the alternate key of Right-Thalamus-Proper
        with pytest.warns(UserWarning):
            df, _ = cltmorph.parse_freesurfer_stats_fromaseg(
                aseg_file, add_bids_entities=False
            )
        assert value_of(df, "thal-rh-thalamus") == 7600.0

    def test_alternate_name_is_used_when_the_id_does_not_match(self, tmp_path):
        text = (
            "# ColHeaders Index SegId NVoxels Volume_mm3 StructName\n"
            "  1  999  7100  7600.0  Right-Thalamus\n"
        )
        path = tmp_path / "aseg.stats"
        path.write_text(text)
        df, _ = cltmorph.parse_freesurfer_stats_fromaseg(
            path, include_missing=False, add_bids_entities=False
        )
        assert df["Region"].tolist() == ["thal-rh-thalamus"]
        assert df["Value"].tolist() == [7600.0]

    def test_region_found_by_segmentation_id(self, tmp_path):
        text = (
            "# ColHeaders Index SegId NVoxels Volume_mm3 StructName\n"
            "  1  11  3000  3300.5  Renamed-Caudate\n"
        )
        path = tmp_path / "aseg.stats"
        path.write_text(text)
        df, _ = cltmorph.parse_freesurfer_stats_fromaseg(
            path, include_missing=False, add_bids_entities=False
        )
        assert df["Region"].tolist() == ["subc-lh-caudate"]
        assert df["Value"].tolist() == [3300.5]

    def test_missing_regions_are_zero_with_a_warning(self, aseg_file):
        with pytest.warns(UserWarning, match="Left-Putamen"):
            df, _ = cltmorph.parse_freesurfer_stats_fromaseg(
                aseg_file, add_bids_entities=False
            )
        assert value_of(df, "subc-lh-putamen") == 0.0
        assert len(df) == 12

    def test_missing_regions_can_be_left_out(self, aseg_file):
        with pytest.warns(UserWarning):
            df, _ = cltmorph.parse_freesurfer_stats_fromaseg(
                aseg_file, include_missing=False, add_bids_entities=False
            )
        assert sorted(df["Region"]) == [
            "subc-lh-caudate",
            "thal-lh-thalamus",
            "thal-rh-thalamus",
        ]

    def test_volumes_are_converted_to_the_configured_units(self, cm_units, aseg_file):
        df, _ = cltmorph.parse_freesurfer_stats_fromaseg(
            aseg_file, include_missing=False, add_bids_entities=False
        )
        assert set(df["Units"]) == {"cm3"}
        assert value_of(df, "thal-lh-thalamus") == pytest.approx(7.5)

    def test_region_table(self, aseg_file):
        df, _ = cltmorph.parse_freesurfer_stats_fromaseg(
            aseg_file,
            include_missing=False,
            table_type="region",
            add_bids_entities=False,
        )
        assert df["Statistics"].tolist() == ["Value"]
        assert df["thal-lh-thalamus"].tolist() == [7500.0]

    def test_custom_configuration(self, tmp_path, aseg_file):
        config = tmp_path / "aseg_config.json"
        config.write_text(
            json.dumps({"aseg": {"ventricle": {"key": "Left-Lateral-Ventricle"}}})
        )
        df, _ = cltmorph.parse_freesurfer_stats_fromaseg(
            aseg_file, config_json=config, add_bids_entities=False
        )
        assert df["Value"].tolist() == [2750.0]

    def test_nothing_found(self, tmp_path):
        path = tmp_path / "aseg.stats"
        path.write_text("# ColHeaders Index SegId NVoxels Volume_mm3 StructName\n")
        with pytest.warns(UserWarning):
            with pytest.raises(ValueError, match="No region measurements"):
                cltmorph.parse_freesurfer_stats_fromaseg(path, include_missing=False)

    def test_bids_entities_and_output(self, tmp_path, aseg_file):
        out = tmp_path / "aseg.tsv"
        df, path = cltmorph.parse_freesurfer_stats_fromaseg(
            aseg_file, output_table=out, include_missing=False
        )
        assert path == str(out)
        assert df["Participant"].unique().tolist() == ["01"]
        assert out.exists()

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Stats file not found"):
            cltmorph.parse_freesurfer_stats_fromaseg(tmp_path / "aseg.stats")

    def test_invalid_table_type(self, aseg_file):
        with pytest.raises(ValueError, match="Invalid table_type"):
            cltmorph.parse_freesurfer_stats_fromaseg(aseg_file, table_type="x")


APARC_STATS = """# Table of FreeSurfer cortical parcellation anatomical statistics
#
# hemi lh
# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
bankssts                                   1000    700   1800  2.500 0.500  0.100  0.020    10   1.0
caudalmiddlefrontal                        2000   1400   3600  2.800 0.400  0.090  0.030    20   2.0
"""


@pytest.fixture
def aparc_file(tmp_path):
    path = tmp_path / "sub-01_ses-M00_lh.aparc.stats"
    path.write_text(APARC_STATS)
    return path


class TestParseFreesurferCortexStats:
    def test_metrics_of_every_region(self, aparc_file):
        df, path = cltmorph.parse_freesurfer_cortex_stats(
            aparc_file, add_bids_entities=False
        )
        assert path is None
        assert set(df["Metric"]) == {"area", "volume", "thickness", "curv"}
        assert set(df["Region"]) == {"ctx-lh-bankssts", "ctx-lh-caudalmiddlefrontal"}

        def get(region, metric, column="Value"):
            rows = df[(df["Region"] == region) & (df["Metric"] == metric)]
            assert len(rows) == 1
            return rows[column].iloc[0]

        assert get("ctx-lh-bankssts", "area") == 700.0
        assert get("ctx-lh-bankssts", "volume") == 1800.0
        assert get("ctx-lh-bankssts", "thickness") == 2.5
        assert get("ctx-lh-bankssts", "curv") == 0.1
        assert get("ctx-lh-caudalmiddlefrontal", "thickness") == 2.8

    def test_thickness_has_a_standard_deviation(self, aparc_file):
        df, _ = cltmorph.parse_freesurfer_cortex_stats(
            aparc_file, add_bids_entities=False
        )
        thickness = df[df["Metric"] == "thickness"].set_index("Region")["Std"]
        assert thickness["ctx-lh-bankssts"] == 0.5
        assert thickness["ctx-lh-caudalmiddlefrontal"] == 0.4
        assert df[df["Metric"] == "area"]["Std"].isna().all()

    def test_table_metadata(self, aparc_file):
        df, _ = cltmorph.parse_freesurfer_cortex_stats(
            aparc_file, add_bids_entities=False
        )
        assert set(df["Source"]) == {"statsfile"}
        assert set(df["Supraregion"]) == {"ctx"}
        assert set(df["Hemisphere"]) == {"lh"}
        assert set(df["MetricFile"]) == {str(aparc_file)}
        units = df.groupby("Metric")["Units"].first().to_dict()
        assert units["thickness"] == "mm"
        assert units["area"] == "mm2"
        assert units["volume"] == "mm3"

    def test_area_and_volume_are_converted_to_the_configured_units(
        self, cm_units, aparc_file
    ):
        df, _ = cltmorph.parse_freesurfer_cortex_stats(
            aparc_file, add_bids_entities=False
        )
        area = df[(df["Metric"] == "area") & (df["Region"] == "ctx-lh-bankssts")]
        volume = df[(df["Metric"] == "volume") & (df["Region"] == "ctx-lh-bankssts")]
        assert area["Value"].iloc[0] == pytest.approx(7.0)
        assert area["Units"].iloc[0] == "cm2"
        assert volume["Value"].iloc[0] == pytest.approx(1.8)
        assert volume["Units"].iloc[0] == "cm3"
        thickness = df[df["Metric"] == "thickness"]
        assert thickness["Value"].tolist() == [2.5, 2.8]  # never converted

    def test_metrics_can_be_selected(self, aparc_file):
        df, _ = cltmorph.parse_freesurfer_cortex_stats(
            aparc_file, include_metrics=["Thickness", "area"], add_bids_entities=False
        )
        assert set(df["Metric"]) == {"thickness", "area"}

    def test_unknown_selected_metrics_are_rejected(self, aparc_file):
        with pytest.raises(ValueError, match="None of the requested metrics"):
            cltmorph.parse_freesurfer_cortex_stats(
                aparc_file, include_metrics=["gyrification"]
            )

    def test_region_table(self, aparc_file):
        df, _ = cltmorph.parse_freesurfer_cortex_stats(
            aparc_file,
            include_metrics=["thickness", "area"],
            table_type="region",
            add_bids_entities=False,
        )
        assert df["Statistics"].tolist() == ["area", "thickness"]
        assert df["Value_ctx-lh-bankssts"].tolist() == [700.0, 2.5]
        assert df["Value_ctx-lh-caudalmiddlefrontal"].tolist() == [1400.0, 2.8]
        assert df["Std_ctx-lh-bankssts"].isna().tolist() == [True, False]
        assert df["Std_ctx-lh-bankssts"].iloc[1] == 0.5

    @pytest.mark.parametrize(
        "name, hemi", [("lh.aparc.stats", "lh"), ("rh.aparc.stats", "rh")]
    )
    def test_hemisphere_from_the_file_name(self, tmp_path, name, hemi):
        path = tmp_path / name
        path.write_text(APARC_STATS.replace("# hemi lh\n", ""))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)  # no fallback to the default
            df, _ = cltmorph.parse_freesurfer_cortex_stats(
                path, add_bids_entities=False
            )
        assert set(df["Hemisphere"]) == {hemi}
        assert all(r.startswith(f"ctx-{hemi}-") for r in df["Region"])

    def test_hemisphere_from_the_file_content(self, tmp_path):
        path = tmp_path / "aparc.stats"
        path.write_text(APARC_STATS.replace("# hemi lh", "# hemi rh"))
        df, _ = cltmorph.parse_freesurfer_cortex_stats(path, add_bids_entities=False)
        assert set(df["Hemisphere"]) == {"rh"}

    def test_hemisphere_can_be_given(self, tmp_path):
        path = tmp_path / "aparc.stats"
        path.write_text(APARC_STATS)
        df, _ = cltmorph.parse_freesurfer_cortex_stats(
            path, hemi="rh", add_bids_entities=False
        )
        assert set(df["Hemisphere"]) == {"rh"}

    def test_undetected_hemisphere_defaults_to_lh_with_a_warning(self, tmp_path):
        path = tmp_path / "aparc.stats"
        path.write_text(APARC_STATS.replace("# hemi lh\n", ""))
        with pytest.warns(UserWarning, match="Could not determine hemisphere"):
            df, _ = cltmorph.parse_freesurfer_cortex_stats(
                path, add_bids_entities=False
            )
        assert set(df["Hemisphere"]) == {"lh"}

    def test_file_without_column_headers_uses_the_standard_layout(self, tmp_path):
        lines = [
            line for line in APARC_STATS.splitlines() if not line.startswith("# ColH")
        ]
        path = tmp_path / "lh.aparc.stats"
        path.write_text("\n".join(lines) + "\n")
        with pytest.warns(UserWarning, match="standard aparc.stats layout"):
            df, _ = cltmorph.parse_freesurfer_cortex_stats(
                path, add_bids_entities=False
            )
        row = df[(df["Region"] == "ctx-lh-bankssts") & (df["Metric"] == "thickness")]
        assert row["Value"].iloc[0] == 2.5

    def test_file_with_tablecol_headers(self, tmp_path):
        headers = (
            "StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv".split()
        )
        lines = [f"# TableCol {i + 1} ColHeader {h}" for i, h in enumerate(headers)]
        lines += ["bankssts 1000 700 1800 2.5 0.5 0.1"]
        path = tmp_path / "lh.aparc.stats"
        path.write_text("\n".join(lines) + "\n")
        df, _ = cltmorph.parse_freesurfer_cortex_stats(path, add_bids_entities=False)
        row = df[df["Metric"] == "thickness"]
        assert row["Value"].tolist() == [2.5]
        assert row["Std"].tolist() == [0.5]

    def test_short_lines_are_skipped_with_a_warning(self, tmp_path):
        path = tmp_path / "lh.aparc.stats"
        path.write_text(APARC_STATS + "broken 1 2 3\n")
        with pytest.warns(UserWarning, match="Skipping line"):
            df, _ = cltmorph.parse_freesurfer_cortex_stats(
                path, add_bids_entities=False
            )
        assert "ctx-lh-broken" not in set(df["Region"])

    def test_values_that_are_not_numbers_are_skipped_with_a_warning(self, tmp_path):
        text = APARC_STATS.replace("2.500", "n/a  ")
        path = tmp_path / "lh.aparc.stats"
        path.write_text(text)
        with pytest.warns(UserWarning, match="Could not parse thickness"):
            df, _ = cltmorph.parse_freesurfer_cortex_stats(
                path, add_bids_entities=False
            )
        thickness = df[df["Metric"] == "thickness"]
        assert thickness["Region"].tolist() == ["ctx-lh-caudalmiddlefrontal"]

    def test_custom_configuration(self, tmp_path, aparc_file):
        config = tmp_path / "cortex.json"
        config.write_text(
            json.dumps(
                {
                    "cortex": {
                        "numvert": {
                            "column": "NumVert",
                            "unit": "au",
                            "std_index": None,
                        }
                    }
                }
            )
        )
        df, _ = cltmorph.parse_freesurfer_cortex_stats(
            aparc_file, config_json=config, add_bids_entities=False
        )
        assert df["Metric"].unique().tolist() == ["numvert"]
        assert df["Value"].tolist() == [1000.0, 2000.0]
        assert df["Units"].unique().tolist() == ["au"]

    def test_empty_configuration(self, tmp_path, aparc_file):
        config = tmp_path / "cortex.json"
        config.write_text(json.dumps({"cortex": {}}))
        with pytest.raises(ValueError, match="No valid metrics"):
            cltmorph.parse_freesurfer_cortex_stats(aparc_file, config_json=config)

    def test_file_without_data(self, tmp_path):
        path = tmp_path / "lh.aparc.stats"
        path.write_text("# hemi lh\n# ColHeaders StructName NumVert SurfArea\n")
        with pytest.raises(ValueError, match="No cortical parcellation data"):
            cltmorph.parse_freesurfer_cortex_stats(path)

    def test_file_without_headers_or_data(self, tmp_path):
        path = tmp_path / "lh.aparc.stats"
        path.write_text("# only comments\n")
        with pytest.raises(ValueError, match="Could not find column headers"):
            cltmorph.parse_freesurfer_cortex_stats(path)

    def test_bids_entities(self, aparc_file):
        df, _ = cltmorph.parse_freesurfer_cortex_stats(aparc_file)
        assert df["Participant"].unique().tolist() == ["01"]
        assert df["Session"].unique().tolist() == ["M00"]

    def test_table_is_saved(self, tmp_path, aparc_file):
        out = tmp_path / "cortex.tsv"
        df, path = cltmorph.parse_freesurfer_cortex_stats(
            aparc_file, output_table=out, add_bids_entities=False
        )
        assert path == str(out)
        saved = pd.read_csv(out, sep="\t")
        assert len(saved) == len(df) == 8

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Stats file not found"):
            cltmorph.parse_freesurfer_cortex_stats(tmp_path / "lh.aparc.stats")

    def test_invalid_table_type(self, aparc_file):
        with pytest.raises(ValueError, match="Invalid table_type"):
            cltmorph.parse_freesurfer_cortex_stats(aparc_file, table_type="x")


class TestStatsFileEdgeCases:
    """Unusual lines and configurations in the FreeSurfer stats files."""

    @staticmethod
    def write_config(tmp_path, level, entries):
        path = tmp_path / f"{level}.json"
        path.write_text(json.dumps({level: entries}))
        return path

    def test_measure_with_another_unit_is_kept_as_a_measure(self, tmp_path):
        stats = tmp_path / "aseg.stats"
        stats.write_text("# Measure Cortex, MeanThick, Mean thickness, 2.5, mm\n")
        config = self.write_config(tmp_path, "global", {"thick": {"key": "MeanThick"}})
        df, _ = cltmorph.parse_freesurfer_global_fromaseg(
            stats, config_json=config, add_bids_entities=False
        )
        assert df["Value"].tolist() == [2.5]
        assert df["Metric"].tolist() == ["measure"]
        assert df["Units"].tolist() == ["mm"]

    def test_measure_without_a_unit(self, tmp_path):
        stats = tmp_path / "aseg.stats"
        stats.write_text("# Measure Cortex, Holes, Number of holes, 3\n")
        config = self.write_config(tmp_path, "global", {"holes": {"key": "Holes"}})
        df, _ = cltmorph.parse_freesurfer_global_fromaseg(
            stats, config_json=config, add_bids_entities=False
        )
        assert df["Value"].tolist() == [3.0]
        assert df["Units"].tolist() == ["au"]

    def test_measure_lines_with_too_few_fields_are_ignored(self, tmp_path):
        stats = tmp_path / "aseg.stats"
        stats.write_text(
            "# Measure Cortex, Holes\n# Measure Cortex, Holes, Number of holes, 3, unitless\n"
        )
        config = self.write_config(tmp_path, "global", {"holes": {"key": "Holes"}})
        df, _ = cltmorph.parse_freesurfer_global_fromaseg(
            stats, config_json=config, add_bids_entities=False
        )
        assert df["Value"].tolist() == [3.0]

    def test_structure_name_does_not_override_the_short_name(self, tmp_path):
        stats = tmp_path / "aseg.stats"
        stats.write_text(
            "# Measure Cortex, First, First measure, 1.0, unitless\n"
            "# Measure Cortex, Second, Second measure, 2.0, unitless\n"
        )
        config = self.write_config(
            tmp_path, "global", {"first": {"key": "First"}, "cortex": {"key": "Cortex"}}
        )
        df, _ = cltmorph.parse_freesurfer_global_fromaseg(
            stats, config_json=config, add_bids_entities=False
        )
        # "Cortex" is ambiguous: the first line that names it is used
        assert df["Value"].tolist() == [1.0, 1.0]

    def test_legacy_lookup_by_position(self, tmp_path):
        stats = tmp_path / "aseg.stats"
        stats.write_text("Strange-Region some text 42.5 more\n")
        config = self.write_config(
            tmp_path, "global", {"strange": {"key": "Strange-Region", "index": 3}}
        )
        df, _ = cltmorph.parse_freesurfer_global_fromaseg(
            stats, config_json=config, add_bids_entities=False
        )
        assert df["Value"].tolist() == [42.5]
        assert df["Metric"].tolist() == ["volume"]

    def test_legacy_lookup_with_a_negative_position(self, tmp_path):
        stats = tmp_path / "aseg.stats"
        stats.write_text("Strange-Region some text 42.5 77.0\n")
        config = self.write_config(
            tmp_path, "global", {"strange": {"key": "Strange-Region", "index": -1}}
        )
        df, _ = cltmorph.parse_freesurfer_global_fromaseg(
            stats, config_json=config, add_bids_entities=False
        )
        assert df["Value"].tolist() == [77.0]

    def test_legacy_lookup_assumes_mm3_outside_measure_lines(self, tmp_path):
        stats = tmp_path / "aseg.stats"
        stats.write_text("# Strange-Region, text, 42.5 , mm^2\n")
        config = self.write_config(
            tmp_path, "global", {"strange": {"key": "Strange-Region", "index": 3}}
        )
        # The line is not a "# Measure" line, so its unit is not read
        df, _ = cltmorph.parse_freesurfer_global_fromaseg(
            stats, config_json=config, add_bids_entities=False
        )
        assert df["Metric"].tolist() == ["volume"]

    @pytest.mark.parametrize("index", [10, -10])
    def test_legacy_lookup_outside_the_line_is_missing(self, tmp_path, index):
        stats = tmp_path / "aseg.stats"
        stats.write_text("Strange-Region some text\nStrange-Region 1.0\n")
        config = self.write_config(
            tmp_path,
            "global",
            {
                "strange": {"key": "Strange-Region", "index": index},
                "x": {"key": "Strange"},
            },
        )
        with pytest.warns(UserWarning, match="not found"):
            df, _ = cltmorph.parse_freesurfer_global_fromaseg(
                stats, config_json=config, add_bids_entities=False
            )
        assert df["Value"].tolist() == [0.0, 0.0]

    def test_legacy_lookup_of_a_value_that_is_not_a_number(self, tmp_path):
        stats = tmp_path / "aseg.stats"
        stats.write_text("Strange-Region some text abc\n")
        config = self.write_config(
            tmp_path, "global", {"strange": {"key": "Strange-Region", "index": 3}}
        )
        with pytest.warns(UserWarning, match="not found"):
            df, _ = cltmorph.parse_freesurfer_global_fromaseg(
                stats, config_json=config, add_bids_entities=False
            )
        assert df["Value"].tolist() == [0.0]

    def test_malformed_tablecol_lines_are_ignored(self, tmp_path):
        stats = tmp_path / "aseg.stats"
        stats.write_text(
            "# TableCol x ColHeader Index\n"
            "# TableCol 1 ColHeader Index\n"
            "# TableCol 2 ColHeader SegId\n"
            "# TableCol 3 ColHeader NVoxels\n"
            "# TableCol 4 ColHeader Volume_mm3\n"
            "# TableCol 5 ColHeader StructName\n"
            "  1  4   2500   2750.0  Left-Lateral-Ventricle\n"
        )
        df, _ = cltmorph.parse_freesurfer_global_fromaseg(
            stats, include_missing=False, add_bids_entities=False
        )
        assert value_of(df, "vent-lh-lateral") == 2750.0

    def test_table_rows_with_too_few_columns_are_skipped(self, tmp_path):
        stats = tmp_path / "aseg.stats"
        stats.write_text(
            "# ColHeaders Index SegId NVoxels Volume_mm3 StructName\n"
            "  1  4   2500\n"
            "  2  11  3000  3300.5  Left-Caudate\n"
        )
        df, _ = cltmorph.parse_freesurfer_stats_fromaseg(
            stats, include_missing=False, add_bids_entities=False
        )
        assert df["Region"].tolist() == ["subc-lh-caudate"]

    def test_table_rows_with_bad_numbers_warn(self, tmp_path):
        stats = tmp_path / "aseg.stats"
        stats.write_text(
            "# ColHeaders Index SegId NVoxels Volume_mm3 StructName\n"
            "  1  4   2500   abc  Left-Lateral-Ventricle\n"
            "  2  11  3000  3300.5  Left-Caudate\n"
        )
        with pytest.warns(UserWarning, match="Error parsing table line"):
            df, _ = cltmorph.parse_freesurfer_stats_fromaseg(
                stats, include_missing=False, add_bids_entities=False
            )
        assert df["Region"].tolist() == ["subc-lh-caudate"]

    def test_failure_to_add_the_bids_entities_is_reported(self, monkeypatch, aseg_file):
        def broken(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(cltmorph.cltbids, "entities_to_table", broken)
        with pytest.warns(UserWarning, match="Could not add BIDS entities: boom"):
            df, _ = cltmorph.parse_freesurfer_global_fromaseg(
                aseg_file, include_missing=False
            )
        assert "Participant" not in df.columns
        assert value_of(df, GLOBAL) == 1100000.0

    # Cortex files
    def cortex(self, tmp_path, metrics, text=APARC_STATS, **kwargs):
        stats = tmp_path / "lh.aparc.stats"
        stats.write_text(text)
        config = self.write_config(tmp_path, "cortex", metrics)
        return cltmorph.parse_freesurfer_cortex_stats(
            stats, config_json=config, add_bids_entities=False, **kwargs
        )

    def test_missing_column_without_an_index_warns_once(self, tmp_path):
        with pytest.warns(
            UserWarning, match="Column Missing for metric ghost"
        ) as caught:
            df, _ = self.cortex(
                tmp_path,
                {
                    "ghost": {"column": "Missing"},
                    "thickness": {"column": "ThickAvg"},
                },
            )
        assert len([w for w in caught if "ghost" in str(w.message)]) == 1
        assert df["Metric"].unique().tolist() == ["thickness"]

    def test_missing_column_falls_back_to_the_index(self, tmp_path):
        df, _ = self.cortex(
            tmp_path, {"vertices": {"column": "Missing", "index": 1, "unit": "au"}}
        )
        assert df["Value"].tolist() == [1000.0, 2000.0]

    def test_index_beyond_the_row_is_skipped(self, tmp_path):
        df, _ = self.cortex(
            tmp_path,
            {
                "far": {"column": "Missing", "index": 50},
                "thickness": {"column": "ThickAvg"},
            },
        )
        assert df["Metric"].unique().tolist() == ["thickness"]

    def test_standard_deviation_by_column_name(self, tmp_path):
        df, _ = self.cortex(
            tmp_path, {"curv": {"column": "MeanCurv", "std_index": "GausCurv"}}
        )
        assert df["Std"].tolist() == [0.02, 0.03]

    def test_standard_deviation_by_position_as_a_string(self, tmp_path):
        df, _ = self.cortex(
            tmp_path, {"curv": {"column": "MeanCurv", "std_index": "7"}}
        )
        assert df["Std"].tolist() == [0.02, 0.03]

    def test_standard_deviation_of_the_thickness_falls_back_to_thickstd(self, tmp_path):
        df, _ = self.cortex(
            tmp_path, {"thickness": {"column": "ThickAvg", "std_index": "Missing"}}
        )
        assert df["Std"].tolist() == [0.5, 0.4]

    @pytest.mark.parametrize("std_index", ["Missing", 99, None, "null", "None", ""])
    def test_unavailable_standard_deviation_is_left_out(self, tmp_path, std_index):
        df, _ = self.cortex(
            tmp_path, {"curv": {"column": "MeanCurv", "std_index": std_index}}
        )
        assert "Std" not in df.columns

    def test_standard_deviation_that_is_not_a_number_is_left_out(self, tmp_path):
        df, _ = self.cortex(
            tmp_path, {"curv": {"column": "MeanCurv", "std_index": "StructName"}}
        )
        assert "Std" not in df.columns

    def test_malformed_tablecol_lines_in_cortex_files_are_ignored(self, tmp_path):
        headers = "StructName NumVert SurfArea GrayVol ThickAvg ThickStd".split()
        lines = ["# TableCol x ColHeader Broken"]
        lines += [f"# TableCol {i + 1} ColHeader {h}" for i, h in enumerate(headers)]
        lines += ["bankssts 1000 700 1800 2.5 0.5"]
        stats = tmp_path / "lh.aparc.stats"
        stats.write_text("\n".join(lines) + "\n")
        df, _ = cltmorph.parse_freesurfer_cortex_stats(stats, add_bids_entities=False)
        assert df[df["Metric"] == "thickness"]["Value"].tolist() == [2.5]


class TestConfigurationFiles:
    """The package configuration is read from config/config.json."""

    @staticmethod
    def use_config(tmp_path, text):
        package = tmp_path / "package"
        (package / "config").mkdir(parents=True)
        if text is not None:
            (package / "config" / "config.json").write_text(text)
        return package

    @pytest.fixture
    def fake_package(self, tmp_path):
        """Point the module to a package folder whose config.json the test writes."""
        original = cltmorph.__file__

        def install(text):
            package = self.use_config(tmp_path, text)
            cltmorph.__file__ = str(package / "morphometrytools.py")
            cltmorph._default_units_lookup.cache_clear()
            cltmorph._default_network_metrics.cache_clear()

        yield install
        cltmorph.__file__ = original
        cltmorph._default_units_lookup.cache_clear()
        cltmorph._default_network_metrics.cache_clear()

    def test_units_are_read_from_the_configuration(self, fake_package):
        fake_package(json.dumps({"metrics_units": {"Thickness": "um"}}))
        assert cltmorph.get_units("thickness") == ["um"]

    def test_units_with_a_missing_configuration(self, fake_package):
        fake_package(None)
        with pytest.raises(ValueError, match="Error loading default configuration"):
            cltmorph.get_units("thickness")

    def test_units_with_an_invalid_configuration(self, fake_package):
        fake_package("{nope")
        with pytest.raises(ValueError, match="Error loading default configuration"):
            cltmorph.get_units("thickness")

    def test_default_metrics_are_lowercase_and_unique(self, fake_package):
        fake_package(
            json.dumps(
                {
                    "network_metrics": {
                        "binary": ["Degree", " degree ", "x"],
                        "weighted": ["y"],
                    }
                }
            )
        )
        assert cltmorph._default_network_metrics() == {
            "binary": ("degree", "x"),
            "weighted": ("y",),
        }

    def test_default_metrics_with_a_missing_configuration(self, fake_package):
        fake_package(None)
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            cltmorph._default_network_metrics()

    def test_default_metrics_with_an_invalid_configuration(self, fake_package):
        fake_package("{nope")
        with pytest.raises(ValueError, match="Invalid JSON"):
            cltmorph._default_network_metrics()

    def test_default_metrics_without_the_network_metrics_entry(self, fake_package):
        fake_package(json.dumps({"metrics_units": {}}))
        with pytest.raises(ValueError, match="'network_metrics' dictionary not found"):
            cltmorph._default_network_metrics()

    @pytest.mark.parametrize(
        "entry",
        [
            {"binary": ["degree"]},
            {"binary": ["degree"], "weighted": []},
            {"binary": ["degree"], "weighted": "strength"},
        ],
    )
    def test_default_metrics_need_a_list_for_each_weighting(self, fake_package, entry):
        fake_package(json.dumps({"network_metrics": entry}))
        with pytest.raises(ValueError, match="needs a non-empty 'weighted' list"):
            cltmorph._default_network_metrics()

    def test_default_metrics_cannot_be_modified_by_the_caller(self):
        defaults = cltmorph._default_network_metrics()
        assert isinstance(defaults["binary"], tuple)
        assert isinstance(defaults["weighted"], tuple)

    def test_unknown_default_metrics_are_rejected(self, fake_package):
        pytest.importorskip("bct")
        fake_package(
            json.dumps(
                {
                    "network_metrics": {
                        "binary": ["degree", "bogus"],
                        "weighted": ["strength"],
                    }
                }
            )
        )
        with pytest.raises(ValueError, match="Unknown metrics in config.json.*bogus"):
            cltmorph.network_metrics_to_table(path_graph())

    def test_custom_default_metrics_are_used_in_their_order(self, fake_package):
        pytest.importorskip("bct")
        fake_package(
            json.dumps(
                {
                    "network_metrics": {
                        "binary": ["diameter", "degree"],
                        "weighted": ["strength"],
                    }
                }
            )
        )
        df, _ = cltmorph.network_metrics_to_table(path_graph(), add_bids_entities=False)
        assert df["Metric"].drop_duplicates().tolist() == ["diameter", "degree"]
        df, _ = cltmorph.network_metrics_to_table(
            WEIGHTED_TRIANGLE, add_bids_entities=False
        )
        assert df["Metric"].drop_duplicates().tolist() == ["strength"]


class TestGetStatsDictionary:
    @pytest.mark.parametrize("level", ["global", "aseg", "cortex"])
    def test_levels(self, level):
        config = cltmorph.get_stats_dictionary(level)
        assert isinstance(config, dict)
        assert len(config) > 0

    def test_global_is_the_default(self):
        assert cltmorph.get_stats_dictionary() == cltmorph.get_stats_dictionary(
            "global"
        )

    def test_global_entries_have_a_key(self):
        for name, info in cltmorph.get_stats_dictionary("global").items():
            assert "key" in info, name

    def test_cortex_entries_have_a_column(self):
        config = cltmorph.get_stats_dictionary("cortex")
        assert {"area", "volume", "thickness", "curv"} <= set(config)
        assert config["thickness"]["column"] == "ThickAvg"

    def test_unknown_level(self):
        with pytest.raises(KeyError, match="not found in stats_mapping.json"):
            cltmorph.get_stats_dictionary("subcortex")


####################################################################################################
# Network metrics
####################################################################################################
def path_graph(n=4):
    """Binary path 0 - 1 - ... - (n-1)."""
    matrix = np.zeros((n, n))
    for i in range(n - 1):
        matrix[i, i + 1] = matrix[i + 1, i] = 1
    return matrix


COMPLETE_4 = 1.0 - np.eye(4)

# Triangle with the edge weights 0-1: 1, 0-2: 2 and 1-2: 3
WEIGHTED_TRIANGLE = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)


def metric_values(df, metric):
    """Values of one metric, in the order of the regions."""
    return df.loc[df["Metric"] == metric, "Value"].to_numpy()


@pytest.fixture
def bct():
    return pytest.importorskip("bct")


@pytest.fixture
def default_metrics():
    """Default metrics of each weighting, read directly from config.json."""
    config = cltmorph.os.path.join(
        cltmorph.os.path.dirname(cltmorph.__file__), "config", "config.json"
    )
    with open(config, encoding="utf-8") as f:
        return json.load(f)["network_metrics"]


@pytest.mark.usefixtures("bct")
class TestNetworkMetricsToTable:
    def test_complete_graph(self):
        df, _ = cltmorph.network_metrics_to_table(
            COMPLETE_4, seed=1, add_bids_entities=False
        )
        for metric in (
            "glob_efficiency",
            "transitivity",
            "density_coeff",
            "char_path_length",
            "radius",
            "diameter",
        ):
            assert metric_values(df, metric).tolist() == [1.0], metric
        assert metric_values(df, "degree").tolist() == [3.0] * 4
        assert metric_values(df, "clustering_coeff").tolist() == [1.0] * 4
        assert metric_values(df, "eccentricity").tolist() == [1.0] * 4
        assert metric_values(df, "betw_centrality").tolist() == [0.0] * 4
        assert metric_values(df, "kcoreness_centrality").tolist() == [3.0] * 4
        assert metric_values(df, "modularity").tolist() == [0.0]

    def test_path_graph_global_metrics(self):
        df, _ = cltmorph.network_metrics_to_table(
            path_graph(), seed=1, add_bids_entities=False
        )
        # Distances: three pairs at 1, two at 2 and one at 3
        assert metric_values(df, "char_path_length") == pytest.approx([10 / 6])
        assert metric_values(df, "glob_efficiency") == pytest.approx(
            [(3 + 2 / 2 + 1 / 3) / 6]
        )
        assert metric_values(df, "density_coeff") == pytest.approx([0.5])
        assert metric_values(df, "radius").tolist() == [2.0]
        assert metric_values(df, "diameter").tolist() == [3.0]
        assert metric_values(df, "transitivity").tolist() == [0.0]

    def test_path_graph_nodal_metrics(self):
        df, _ = cltmorph.network_metrics_to_table(
            path_graph(), seed=1, add_bids_entities=False
        )
        assert metric_values(df, "degree").tolist() == [1, 2, 2, 1]
        assert metric_values(df, "eccentricity").tolist() == [3, 2, 2, 3]
        assert metric_values(df, "betw_centrality").tolist() == [0, 4, 4, 0]
        assert metric_values(df, "clustering_coeff").tolist() == [0] * 4
        assert metric_values(df, "kcoreness_centrality").tolist() == [1] * 4
        # The ends of the path are symmetric, and so are the inner nodes
        for metric in (
            "eigenvector_centrality",
            "pagerank_centrality",
            "subgraph_centrality",
        ):
            values = metric_values(df, metric)
            assert values[0] == pytest.approx(values[3]), metric
            assert values[1] == pytest.approx(values[2]), metric
            assert values[1] > values[0], metric
        assert metric_values(df, "pagerank_centrality").sum() == pytest.approx(1.0)

    def test_weighted_triangle_global_metrics(self):
        df, _ = cltmorph.network_metrics_to_table(
            WEIGHTED_TRIANGLE, seed=1, add_bids_entities=False
        )
        # Lengths 1/w: 1, 1/2 and 1/3. The shortest path 0-1 goes through 2 (5/6)
        assert metric_values(df, "char_path_length_wei") == pytest.approx([5 / 9])
        assert metric_values(df, "radius_wei") == pytest.approx([0.5])
        assert metric_values(df, "diameter_wei") == pytest.approx([5 / 6])
        # Weights normalized by the maximum: distances 2.5, 1.5 and 1
        assert metric_values(df, "glob_efficiency_wei") == pytest.approx(
            [(1 / 2.5 + 1 / 1.5 + 1) / 3]
        )
        assert metric_values(df, "density_coeff").tolist() == [1.0]

    def test_weighted_triangle_nodal_metrics(self):
        df, _ = cltmorph.network_metrics_to_table(
            WEIGHTED_TRIANGLE, seed=1, add_bids_entities=False
        )
        assert metric_values(df, "degree").tolist() == [2, 2, 2]
        assert metric_values(df, "strength").tolist() == [3, 4, 5]
        assert metric_values(df, "eccentricity_wei") == pytest.approx(
            [5 / 6, 5 / 6, 0.5]
        )
        assert metric_values(df, "betw_centrality_wei").tolist() == [0, 0, 2]
        # Geometric mean of the normalized weights 1/3, 2/3 and 1
        expected = (2 / 9) ** (1 / 3)
        assert metric_values(df, "clustering_coeff_wei") == pytest.approx(
            [expected] * 3
        )
        assert metric_values(df, "transitivity_wei") == pytest.approx([expected])
        assert metric_values(df, "pagerank_centrality").sum() == pytest.approx(1.0)

    def test_nodal_metrics_are_reported_per_region(self):
        df, _ = cltmorph.network_metrics_to_table(
            path_graph(), metrics="degree", add_bids_entities=False
        )
        assert df["Region"].tolist() == [
            "supra-side-000001",
            "supra-side-000002",
            "supra-side-000003",
            "supra-side-000004",
        ]
        assert set(df["Supraregion"]) == {"supra"}
        assert set(df["Hemisphere"]) == {"side"}

    def test_global_metrics_are_reported_for_the_whole_brain(self):
        df, _ = cltmorph.network_metrics_to_table(
            path_graph(), metrics="diameter", add_bids_entities=False
        )
        assert df["Region"].tolist() == [GLOBAL]

    def test_default_metrics_of_a_binary_graph(self, default_metrics):
        df, _ = cltmorph.network_metrics_to_table(
            path_graph(), seed=1, add_bids_entities=False
        )
        assert df["Metric"].drop_duplicates().tolist() == default_metrics["binary"]
        assert set(df["Source"]) == {"conn_matrix_binary"}
        assert set(df["Units"]) == {"au"}

    def test_default_metrics_of_a_weighted_graph(self, default_metrics):
        df, _ = cltmorph.network_metrics_to_table(
            WEIGHTED_TRIANGLE, seed=1, add_bids_entities=False
        )
        assert df["Metric"].drop_duplicates().tolist() == default_metrics["weighted"]
        assert set(df["Source"]) == {"conn_matrix_weighted"}

    def test_weighting_is_detected_from_the_matrix(self):
        # All the edges share one value: binary, whatever that value is
        df, _ = cltmorph.network_metrics_to_table(
            path_graph() * 3.5, metrics="degree", add_bids_entities=False
        )
        assert set(df["Source"]) == {"conn_matrix_binary"}

    def test_weighting_can_be_forced(self, default_metrics):
        df, _ = cltmorph.network_metrics_to_table(
            path_graph(), weighting="weighted", seed=1, add_bids_entities=False
        )
        assert set(df["Source"]) == {"conn_matrix_weighted"}
        assert df["Metric"].drop_duplicates().tolist() == default_metrics["weighted"]

    def test_invalid_weighting(self):
        with pytest.raises(ValueError, match="Invalid weighting"):
            cltmorph.network_metrics_to_table(path_graph(), weighting="signed")

    def test_selected_metrics_keep_their_order(self):
        df, _ = cltmorph.network_metrics_to_table(
            path_graph(),
            metrics=["strength", "degree", "diameter"],
            add_bids_entities=False,
        )
        assert df["Metric"].drop_duplicates().tolist() == [
            "strength",
            "degree",
            "diameter",
        ]

    def test_metric_names_are_case_insensitive_and_unique(self):
        df, _ = cltmorph.network_metrics_to_table(
            path_graph(),
            metrics=[" Degree ", "DEGREE", "diameter"],
            add_bids_entities=False,
        )
        assert df["Metric"].drop_duplicates().tolist() == ["degree", "diameter"]
        assert (df["Metric"] == "degree").sum() == 4

    def test_single_metric_as_a_string(self):
        df, _ = cltmorph.network_metrics_to_table(
            path_graph(), metrics="degree", add_bids_entities=False
        )
        assert set(df["Metric"]) == {"degree"}

    def test_unknown_metric(self):
        with pytest.raises(ValueError, match="Unknown metrics: nonsense"):
            cltmorph.network_metrics_to_table(path_graph(), metrics=["nonsense"])

    def test_invalid_metrics_type(self):
        with pytest.raises(TypeError, match="metrics must be"):
            cltmorph.network_metrics_to_table(path_graph(), metrics=5)

    def test_metrics_of_another_weighting_warn_but_are_computed(self):
        with pytest.warns(UserWarning, match="not among the default binary metrics"):
            df, _ = cltmorph.network_metrics_to_table(
                path_graph(), metrics=["strength"], add_bids_entities=False
            )
        assert metric_values(df, "strength").tolist() == [1, 2, 2, 1]

    def test_default_metrics_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            cltmorph.network_metrics_to_table(
                path_graph(), seed=1, add_bids_entities=False
            )

    def test_without_the_global_metrics(self):
        df, _ = cltmorph.network_metrics_to_table(
            path_graph(), include_global=False, add_bids_entities=False
        )
        assert GLOBAL not in df["Region"].tolist()
        assert "diameter" not in set(df["Metric"])
        assert {"degree", "eccentricity"} <= set(df["Metric"])

    def test_global_metrics_requested_without_globals_warn(self):
        with pytest.warns(UserWarning, match="skipping global metrics"):
            df, _ = cltmorph.network_metrics_to_table(
                path_graph(),
                metrics=["degree", "diameter"],
                include_global=False,
                add_bids_entities=False,
            )
        assert set(df["Metric"]) == {"degree"}

    def test_nothing_left_to_compute(self):
        with pytest.warns(UserWarning):
            with pytest.raises(ValueError, match="No metrics left"):
                cltmorph.network_metrics_to_table(
                    path_graph(), metrics=["diameter"], include_global=False
                )

    def test_region_table(self):
        df, _ = cltmorph.network_metrics_to_table(
            path_graph(),
            metrics=["degree", "diameter"],
            table_type="region",
            add_bids_entities=False,
        )
        assert df["Metric"].tolist() == ["degree", "diameter"]
        assert df["Statistics"].tolist() == ["Value", "Value"]
        assert df["supra-side-000002"].tolist()[0] == 2.0
        assert np.isnan(df["supra-side-000002"].tolist()[1])
        assert df[GLOBAL].tolist()[1] == 3.0
        assert np.isnan(df[GLOBAL].tolist()[0])

    def test_source_label(self):
        df, _ = cltmorph.network_metrics_to_table(
            path_graph(),
            metrics="degree",
            cmat_met="fiber-count",
            add_bids_entities=False,
        )
        assert set(df["Source"]) == {"conn_matrix_fiber-count"}

    def test_source_label_from_a_list(self):
        df, _ = cltmorph.network_metrics_to_table(
            path_graph(),
            metrics="degree",
            cmat_met=["sift2", "fa"],
            add_bids_entities=False,
        )
        assert set(df["Source"]) == {"conn_matrix_sift2-fa"}

    def test_region_names_from_a_dictionary(self):
        lut = {"index": [1, 2, 3, 4], "name": ["a", "b", "c", "d"]}
        df, _ = cltmorph.network_metrics_to_table(
            path_graph(), lut_file=lut, metrics="degree", add_bids_entities=False
        )
        assert df["Region"].tolist() == ["a", "b", "c", "d"]
        assert df["Value"].tolist() == [1, 2, 2, 1]

    def test_region_names_from_a_file(self, tmp_path):
        path = tmp_path / "regions.tsv"
        path.write_text(
            "index\tname\tcolor\n1\ta\t#ff0000\n2\tb\t#00ff00\n3\tc\t#0000ff\n4\td\t#ffffff\n"
        )
        df, _ = cltmorph.network_metrics_to_table(
            path_graph(), lut_file=path, metrics="degree", add_bids_entities=False
        )
        assert df["Region"].tolist() == ["a", "b", "c", "d"]

    def test_region_names_of_a_connectome(self):
        connectome = cltconn.Connectome(
            matrix=path_graph(), region_names=["a", "b", "c", "d"]
        )
        df, _ = cltmorph.network_metrics_to_table(
            connectome, metrics="degree", add_bids_entities=False
        )
        assert df["Region"].tolist() == ["a", "b", "c", "d"]
        assert df["Value"].tolist() == [1, 2, 2, 1]

    def test_lookup_table_wins_over_the_connectome_names(self):
        connectome = cltconn.Connectome(
            matrix=path_graph(), region_names=["a", "b", "c", "d"]
        )
        lut = {"index": [1, 2, 3, 4], "name": ["w", "x", "y", "z"]}
        df, _ = cltmorph.network_metrics_to_table(
            connectome, lut_file=lut, metrics="degree", add_bids_entities=False
        )
        assert df["Region"].tolist() == ["w", "x", "y", "z"]

    def test_custom_prefix_for_generated_names(self):
        df, _ = cltmorph.network_metrics_to_table(
            path_graph(),
            metrics="degree",
            region_prefix="node-x",
            add_bids_entities=False,
        )
        assert all(r.startswith("node-x-") for r in df["Region"])

    def test_too_few_region_names(self):
        lut = {"index": [1, 2], "name": ["a", "b"]}
        with pytest.raises(ValueError, match="names 2 regions but the matrix has 4"):
            cltmorph.network_metrics_to_table(path_graph(), lut_file=lut)

    def test_extra_region_names_are_dropped_with_a_warning(self):
        lut = {"index": [1, 2, 3, 4, 5], "name": ["a", "b", "c", "d", "e"]}
        with pytest.warns(UserWarning, match="Only the first 4 names are used"):
            df, _ = cltmorph.network_metrics_to_table(
                path_graph(), lut_file=lut, metrics="degree", add_bids_entities=False
            )
        assert df["Region"].tolist() == ["a", "b", "c", "d"]

    def test_repeated_region_names_are_rejected(self):
        lut = {"index": [1, 2, 3, 4], "name": ["a", "a", "c", "d"]}
        with pytest.raises(ValueError, match="share the same name"):
            cltmorph.network_metrics_to_table(path_graph(), lut_file=lut)

    def test_missing_lookup_table_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Lookup table not found"):
            cltmorph.network_metrics_to_table(path_graph(), lut_file=tmp_path / "x.tsv")

    def test_invalid_lookup_table_type(self):
        with pytest.raises(TypeError, match="lut_file must be"):
            cltmorph.network_metrics_to_table(path_graph(), lut_file=["a", "b"])

    def test_matrix_must_be_square(self):
        with pytest.raises(ValueError, match="must be a square matrix"):
            cltmorph.network_metrics_to_table(np.ones((3, 4)))

    def test_asymmetric_matrix_warns(self):
        matrix = path_graph()
        matrix[0, 1] = 5
        with pytest.warns(UserWarning, match="not symmetric"):
            cltmorph.network_metrics_to_table(
                matrix, metrics="degree", add_bids_entities=False
            )

    def test_self_connections_are_removed_with_a_warning(self):
        matrix = path_graph()
        np.fill_diagonal(matrix, 1)
        with pytest.warns(UserWarning, match="self-connections"):
            df, _ = cltmorph.network_metrics_to_table(
                matrix, metrics=["degree", "strength"], add_bids_entities=False
            )
        assert metric_values(df, "degree").tolist() == [1, 2, 2, 1]
        assert metric_values(df, "strength").tolist() == [1, 2, 2, 1]

    def test_input_matrix_is_not_modified(self):
        matrix = path_graph()
        np.fill_diagonal(matrix, 7)
        before = matrix.copy()
        cltmorph.network_metrics_to_table(
            matrix, metrics="degree", add_bids_entities=False
        )
        np.testing.assert_array_equal(matrix, before)

    def test_integer_and_boolean_matrices(self):
        for matrix in (path_graph().astype(int), path_graph().astype(bool)):
            df, _ = cltmorph.network_metrics_to_table(
                matrix, metrics="degree", add_bids_entities=False
            )
            assert df["Value"].tolist() == [1, 2, 2, 1]

    def test_connectome_object_matrix_is_used(self):
        connectome = cltconn.Connectome(matrix=WEIGHTED_TRIANGLE)
        df, _ = cltmorph.network_metrics_to_table(
            connectome, metrics="strength", add_bids_entities=False
        )
        assert df["Value"].tolist() == [3, 4, 5]

    def test_seed_makes_the_community_metrics_reproducible(self):
        rng = np.random.default_rng(0)
        upper = np.triu(rng.random((20, 20)) < 0.3, 1).astype(float)
        matrix = upper + upper.T
        metrics = ["modularity", "participation_coeff", "within_module_zscore"]
        first, _ = cltmorph.network_metrics_to_table(
            matrix, metrics=metrics, seed=7, add_bids_entities=False
        )
        second, _ = cltmorph.network_metrics_to_table(
            matrix, metrics=metrics, seed=7, add_bids_entities=False
        )
        pd.testing.assert_frame_equal(first, second)

    def test_two_communities_are_found(self):
        # Two triangles joined by one edge
        matrix = np.zeros((6, 6))
        for a, b in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)]:
            matrix[a, b] = matrix[b, a] = 1
        df, _ = cltmorph.network_metrics_to_table(
            matrix, metrics="modularity", seed=1, add_bids_entities=False
        )
        # Q = 2 * (3/7 - (7/14)^2) for the two triangles
        assert df["Value"].tolist() == pytest.approx([2 * (3 / 7 - 0.25)])

    def test_bids_entities_come_from_the_source_file(self, tmp_path):
        source = tmp_path / "sub-02_ses-M06_desc-fc_connectivity.tsv"
        df, _ = cltmorph.network_metrics_to_table(
            path_graph(), metrics="degree", source_file=source
        )
        assert df["Participant"].unique().tolist() == ["02"]
        assert df["Session"].unique().tolist() == ["M06"]
        assert set(df["MetricFile"]) == {str(source)}

    def test_table_is_saved(self, tmp_path):
        out = tmp_path / "network.tsv"
        df, path = cltmorph.network_metrics_to_table(
            path_graph(),
            metrics=["degree", "diameter"],
            output_table=out,
            add_bids_entities=False,
        )
        assert path == str(out)
        saved = pd.read_csv(out, sep="\t")
        assert saved["Metric"].tolist() == df["Metric"].tolist()
        np.testing.assert_allclose(saved["Value"], df["Value"])

    def test_output_directory_must_exist(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Directory does not exist"):
            cltmorph.network_metrics_to_table(
                path_graph(), output_table=tmp_path / "no" / "t.tsv"
            )

    def test_invalid_table_type(self):
        with pytest.raises(ValueError, match="Invalid table_type"):
            cltmorph.network_metrics_to_table(path_graph(), table_type="x")

    def test_disconnected_graph_excludes_the_unreachable_pairs(self):
        matrix = np.zeros((5, 5))
        matrix[0, 1] = matrix[1, 0] = matrix[2, 3] = matrix[3, 2] = 1  # node 4 is alone
        df, _ = cltmorph.network_metrics_to_table(
            matrix,
            metrics=["char_path_length", "radius", "glob_efficiency", "density_coeff"],
            add_bids_entities=False,
        )
        assert metric_values(df, "char_path_length").tolist() == [1.0]
        assert metric_values(df, "radius").tolist() == [1.0]
        assert metric_values(df, "density_coeff") == pytest.approx([0.2])
        assert metric_values(df, "glob_efficiency") == pytest.approx([0.2])

    def test_isolated_node_has_no_eccentricity(self):
        matrix = np.zeros((5, 5))
        matrix[0, 1] = matrix[1, 0] = matrix[2, 3] = matrix[3, 2] = 1  # node 4 is alone
        df, _ = cltmorph.network_metrics_to_table(
            matrix,
            metrics=["diameter", "radius", "eccentricity"],
            add_bids_entities=False,
        )
        assert metric_values(df, "diameter").tolist() == [1.0]
        assert metric_values(df, "radius").tolist() == [1.0]
        eccentricity = metric_values(df, "eccentricity")
        assert eccentricity[:4].tolist() == [1.0] * 4
        assert np.isnan(eccentricity[4])

    def test_isolated_node_does_not_change_the_other_eccentricities(self):
        matrix = np.zeros((5, 5))
        matrix[:4, :4] = path_graph()
        df, _ = cltmorph.network_metrics_to_table(
            matrix,
            metrics=["diameter", "radius", "eccentricity", "char_path_length"],
            add_bids_entities=False,
        )
        assert metric_values(df, "eccentricity")[:4].tolist() == [3, 2, 2, 3]
        assert np.isnan(metric_values(df, "eccentricity")[4])
        assert metric_values(df, "diameter").tolist() == [3.0]
        assert metric_values(df, "radius").tolist() == [2.0]
        assert metric_values(df, "char_path_length") == pytest.approx([10 / 6])

    def test_isolated_node_in_a_weighted_graph(self):
        matrix = np.zeros((4, 4))
        matrix[:3, :3] = WEIGHTED_TRIANGLE
        df, _ = cltmorph.network_metrics_to_table(
            matrix,
            metrics=["diameter_wei", "radius_wei", "eccentricity_wei"],
            add_bids_entities=False,
        )
        assert metric_values(df, "diameter_wei") == pytest.approx([5 / 6])
        assert metric_values(df, "radius_wei") == pytest.approx([0.5])
        eccentricity = metric_values(df, "eccentricity_wei")
        assert eccentricity[:3] == pytest.approx([5 / 6, 5 / 6, 0.5])
        assert np.isnan(eccentricity[3])

    def test_graph_without_edges(self):
        df, _ = cltmorph.network_metrics_to_table(
            np.zeros((3, 3)),
            metrics=["diameter", "radius", "eccentricity", "degree"],
            add_bids_entities=False,
        )
        assert np.isnan(metric_values(df, "diameter")).all()
        assert np.isnan(metric_values(df, "radius")).all()
        assert np.isnan(metric_values(df, "eccentricity")).all()
        assert metric_values(df, "degree").tolist() == [0, 0, 0]

    def test_bctpy_is_required(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "bct", None)
        with pytest.raises(ImportError, match="bctpy is required"):
            cltmorph.network_metrics_to_table(path_graph())
