"""Tests for clabtoolkit.tracttools (the plotting method is not covered)."""

import contextlib
import io
import warnings

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from nibabel.streamlines import ArraySequence

import clabtoolkit.tracttools as clttract
from clabtoolkit.parcellationtools import Parcellation
from clabtoolkit.tracttools import Tractogram

# Reference space of the simulated tractograms: 60 x 60 x 50 voxels of 2 mm
SHAPE = (60, 60, 50)
AFFINE = np.diag([2.0, 2.0, 2.0, 1.0])
AFFINE[:3, 3] = [-60.0, -60.0, -40.0]
# Extent of the reference image: the outer faces of its first and last voxels
FOV_MIN = AFFINE[:3, 3] - 1.0
FOV_MAX = AFFINE[:3, 3] + (np.array(SHAPE) - 1) * 2.0 + 1.0

# Value that encodes the default color '#BFBDBD' in data_per_streamline['default']
DEFAULT_COLOR_VALUE = 12434879


####################################################################################################
# Fixtures and helpers
####################################################################################################
@pytest.fixture(autouse=True)
def _quiet():
    """Silence the warnings and the progress messages of the functions."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture
def ref_image():
    return nib.Nifti1Image(np.zeros(SHAPE, dtype=np.float32), AFFINE)


@pytest.fixture
def ref_file(tmp_path, ref_image):
    path = tmp_path / "ref.nii.gz"
    nib.save(ref_image, path)
    return path


@pytest.fixture
def sim(ref_image):
    """Simulated tractogram: 3 bundles of 10 streamlines with 30 points each."""
    return Tractogram.simulate_tractogram(
        ref_image, n_bundles=3, n_streamlines=10, n_points=30, seed=0
    )


def line(x0, x1, y=0.0, z=0.0, n=11):
    """Straight streamline along x, with equally spaced points."""
    return np.c_[np.linspace(x0, x1, n), np.full(n, y), np.full(n, z)].astype(
        np.float32
    )


def hand_made(streamlines, header=None, name="hand"):
    """Tractogram built from components, on an identity grid (1 voxel = 1 mm)."""
    header = {"nb_streamlines": len(streamlines)} if header is None else header
    return Tractogram(
        tracts=ArraySequence(streamlines), affine=np.eye(4), header=header, name=name
    )


@pytest.fixture
def lengths_tract():
    """Five straight streamlines of 2, 4, 6, 8 and 10 mm, with one map of each kind."""
    tract = hand_made(
        [line(0, length, y=i) for i, length in enumerate((2, 4, 6, 8, 10))]
    )
    tract.data_per_streamline["score"] = np.array([[10], [20], [30], [40], [50]])
    tract.data_per_point["ramp"] = ArraySequence(
        [
            np.linspace(0, 1, len(s)).reshape(-1, 1) * (i + 1)
            for i, s in enumerate(tract.tracts)
        ]
    )
    return tract


@pytest.fixture
def clusters_tract():
    """Three well separated groups (100 mm apart) of 5 almost identical streamlines."""
    streamlines = [
        line(0, 50, y=100.0 * group + 0.5 * i, z=0.0, n=12)
        for group in range(3)
        for i in range(5)
    ]
    return hand_made(streamlines, name="clusters")


@pytest.fixture
def mask_tract():
    """
    Four streamlines on an identity grid: S0 crosses x=4, S1 stays at x<4, S2 stays
    at x>=4 and S3 lies outside of the 10 x 10 x 10 voxel volume of the labels.
    """
    return hand_made(
        [
            line(1, 8, 2, 2, n=8),
            line(1, 3, 2, 2, n=8),
            line(5, 8, 2, 2, n=8),
            line(1, 8, 20, 20, n=8),
        ]
    )


@pytest.fixture
def labels():
    """Parcellation of 10 x 10 x 10 voxels: label 1 for x < 4 and label 2 for x >= 4."""
    data = np.zeros((10, 10, 10), dtype=np.int32)
    data[:4] = 1
    data[4:] = 2
    return Parcellation(
        data,
        color_table={
            "index": [1, 2],
            "name": ["front", "back"],
            "color": ["#ff0000", "#0000ff"],
        },
        affine=np.eye(4),
    )


def points_of(tract):
    return np.concatenate(list(tract.tracts))


def quiet_call(function, *args, **kwargs):
    """Call a function that prints progress messages and return what it returns."""
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


####################################################################################################
# simulate_tractogram
####################################################################################################
class TestSimulateTractogram:
    def test_structure(self, sim):
        assert isinstance(sim, Tractogram) and isinstance(sim.tracts, ArraySequence)
        assert len(sim.tracts) == 30 and all(s.shape == (30, 3) for s in sim.tracts)
        assert sim.tracts[0].dtype == np.float32
        assert sim.name == "simulated" and sim.tract_file == "from_simulation"
        assert np.array_equal(
            sim.affine, np.eye(4)
        )  # Streamlines are in world coordinates
        assert sim.data_per_point == {}

    def test_header_describes_the_reference_grid(self, sim):
        assert sim.header["dimensions"].tolist() == list(SHAPE)
        assert sim.header["voxel_sizes"].tolist() == [2.0, 2.0, 2.0]
        assert sim.header["voxel_order"] == "RAS"
        assert np.allclose(sim.header["voxel_to_rasmm"], AFFINE)
        assert sim.header["nb_streamlines"] == 30

    def test_bundle_map_and_colortable(self, sim):
        ids = sim.data_per_streamline["bundle_id"].ravel()
        assert ids.tolist() == [1] * 10 + [2] * 10 + [3] * 10
        table = sim.colortables["bundle_id"]
        assert table["names"] == ["bundle_00", "bundle_01", "bundle_02"]
        assert table["color_table"][:, 4].tolist() == [1, 2, 3]
        assert {"default", "length", "bundle_id"} == set(sim.data_per_streamline)

    def test_lengths_are_stored(self, sim):
        lengths = sim.data_per_streamline["length"]
        expected = [
            np.linalg.norm(np.diff(s, axis=0), axis=1).sum() for s in sim.tracts
        ]
        assert lengths.shape == (30, 1) and np.allclose(lengths.ravel(), expected)

    def test_streamlines_stay_inside_the_field_of_view(self, sim):
        points = points_of(sim)
        assert (points >= FOV_MIN - 1e-3).all() and (points <= FOV_MAX + 1e-3).all()

    def test_noisy_streamlines_also_stay_inside_the_field_of_view(self, ref_image):
        tract = Tractogram.simulate_tractogram(
            ref_image,
            n_bundles=3,
            n_streamlines=5,
            n_points=20,
            noise=25.0,
            n_noise_streamlines=5,
            seed=0,
        )
        points = points_of(tract)
        assert (points >= FOV_MIN - 1e-3).all() and (points <= FOV_MAX + 1e-3).all()
        assert (points == FOV_MIN).any() or (
            points == FOV_MAX
        ).any()  # Some were clipped

    def test_seed(self, ref_image):
        kwargs = {"n_bundles": 2, "n_streamlines": 5, "n_points": 20}
        first = Tractogram.simulate_tractogram(ref_image, seed=3, **kwargs)
        again = Tractogram.simulate_tractogram(ref_image, seed=3, **kwargs)
        other = Tractogram.simulate_tractogram(ref_image, seed=4, **kwargs)
        assert np.array_equal(points_of(first), points_of(again))
        assert np.array_equal(
            first.colortables["bundle_id"]["color_table"],
            again.colortables["bundle_id"]["color_table"],
        )
        assert not np.array_equal(points_of(first), points_of(other))

    def test_number_of_streamlines_per_bundle(self, ref_image):
        tract = Tractogram.simulate_tractogram(
            ref_image, n_bundles=3, n_streamlines=[5, 10, 15], n_points=20, seed=0
        )
        ids, counts = np.unique(
            tract.data_per_streamline["bundle_id"], return_counts=True
        )
        assert ids.tolist() == [1, 2, 3] and counts.tolist() == [5, 10, 15]

    def test_names_colors_map_name_and_tractogram_name(self, ref_image):
        tract = Tractogram.simulate_tractogram(
            ref_image,
            n_bundles=2,
            n_streamlines=3,
            n_points=10,
            bundle_names=["left", "right"],
            colors=["#ff0000", "#00ff00"],
            map_name="group",
            name="mine",
            seed=0,
        )
        assert tract.name == "mine" and "bundle_id" not in tract.data_per_streamline
        table = tract.colortables["group"]
        assert table["names"] == ["left", "right"]
        assert table["color_table"][:, :3].tolist() == [[255, 0, 0], [0, 255, 0]]

    def test_fixed_bundle_length(self, ref_image):
        tract = Tractogram.simulate_tractogram(
            ref_image,
            n_bundles=1,
            n_streamlines=1,
            n_points=50,
            bundle_length=40.0,
            length_variability=0.0,
            spread=0.0,
            fanning=0.0,
            curvature=0.0,
            bundle_radius=0.5,
            seed=0,
        )
        assert tract.data_per_streamline["length"][0, 0] == pytest.approx(
            40.0, abs=0.05
        )

    def test_prescribed_direction(self, ref_image):
        for seed in range(5):
            tract = Tractogram.simulate_tractogram(
                ref_image,
                n_bundles=1,
                n_streamlines=1,
                n_points=30,
                directions=[[1, 0, 0]],
                curvature=0.0,
                spread=0.0,
                fanning=0.0,
                length_variability=0.0,
                seed=seed,
            )
            extent = np.abs(tract.tracts[0][-1] - tract.tracts[0][0])
            assert extent.argmax() == 0 and extent[0] > 5 * max(
                extent[1], extent[2], 1e-9
            )

    def test_mask_constrains_the_extremities(self, ref_image):
        mask = np.zeros(SHAPE, dtype=np.uint8)
        mask[20:40, 20:40, 10:30] = 1
        low, high = np.array([20, 20, 10]) - 1.0, np.array([39, 39, 29]) + 1.0
        inverse = np.linalg.inv(AFFINE)
        for seed in range(5):
            tract = Tractogram.simulate_tractogram(
                ref_image,
                n_bundles=2,
                n_streamlines=1,
                n_points=30,
                mask=mask,
                bundle_length=(10, 20),
                spread=0.0,
                fanning=0.0,
                length_variability=0.0,
                bundle_radius=0.5,
                curvature=0.0,
                seed=seed,
            )
            for streamline in tract.tracts:
                for end in (streamline[0], streamline[-1]):
                    voxel = nib.affines.apply_affine(inverse, end)
                    assert (voxel >= low).all() and (voxel <= high).all()

    def test_noise_displaces_the_points(self, ref_image):
        kwargs = {"n_bundles": 1, "n_streamlines": 5, "n_points": 10, "seed": 0}
        clean = Tractogram.simulate_tractogram(ref_image, **kwargs)
        noisy = Tractogram.simulate_tractogram(ref_image, noise=1.0, **kwargs)
        assert not np.allclose(points_of(clean), points_of(noisy))

    def test_noise_can_leave_the_endpoints_untouched(self, ref_image):
        kwargs = {"n_bundles": 2, "n_streamlines": 4, "n_points": 20, "seed": 1}
        clean = Tractogram.simulate_tractogram(ref_image, **kwargs)
        noisy = Tractogram.simulate_tractogram(
            ref_image, noise=2.0, noise_fix_endpoints=True, **kwargs
        )
        assert not np.allclose(points_of(clean), points_of(noisy))
        for a, b in zip(clean.tracts, noisy.tracts, strict=True):
            assert np.allclose(a[0], b[0], atol=1e-4) and np.allclose(
                a[-1], b[-1], atol=1e-4
            )

    def test_spurious_streamlines(self, ref_image):
        tract = Tractogram.simulate_tractogram(
            ref_image,
            n_bundles=2,
            n_streamlines=3,
            n_points=10,
            n_noise_streamlines=5,
            seed=0,
        )
        ids = tract.data_per_streamline["bundle_id"].ravel()
        assert ids.tolist() == [1] * 3 + [2] * 3 + [0] * 5
        assert tract.colortables["bundle_id"]["names"] == [
            "noise",
            "bundle_00",
            "bundle_01",
        ]
        assert tract.colortables["bundle_id"]["color_table"][:, 4].tolist() == [0, 1, 2]

    @pytest.mark.parametrize("kind", ["image", "path", "str"])
    def test_reference_image_can_be_given_in_several_ways(
        self, ref_image, ref_file, kind
    ):
        ref = {"image": ref_image, "path": ref_file, "str": str(ref_file)}[kind]
        tract = Tractogram.simulate_tractogram(
            ref, n_bundles=1, n_streamlines=3, n_points=10, seed=0
        )
        assert len(tract.tracts) == 3 and tract.header["dimensions"].tolist() == list(
            SHAPE
        )

    def test_module_function_is_a_wrapper(self, ref_image):
        tract = clttract.simulate_tractogram(
            ref_image, n_bundles=1, n_streamlines=3, n_points=10, seed=0
        )
        same = Tractogram.simulate_tractogram(
            ref_image, n_bundles=1, n_streamlines=3, n_points=10, seed=0
        )
        assert np.array_equal(points_of(tract), points_of(same))

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"n_bundles": 0}, "n_bundles"),
            ({"n_points": 1}, "n_points"),
            ({"n_streamlines": [1, 2, 3, 4]}, "n_streamlines"),
            ({"n_streamlines": 0}, "at least one streamline"),
            ({"margin": 0.6}, "margin"),
            ({"curvature": -1}, "curvature"),
            ({"length_variability": 0.6}, "length_variability"),
            ({"noise": -1}, "noise cannot be negative"),
            ({"noise_smoothness": 2}, "noise_smoothness"),
            ({"n_noise_streamlines": -1}, "n_noise_streamlines"),
            ({"bundle_length": -5}, "Invalid bundle_length"),
            ({"bundle_radius": 0}, "Invalid bundle_radius"),
            ({"directions": [[1, 0]]}, "3-element"),
            ({"bundle_names": ["only-one"]}, "bundle_names"),
        ],
    )
    def test_invalid_arguments(self, ref_image, kwargs, message):
        arguments = {"n_bundles": 2, "n_streamlines": 2, "n_points": 10, **kwargs}
        with pytest.raises(ValueError, match=message):
            Tractogram.simulate_tractogram(ref_image, **arguments)

    def test_missing_reference_and_empty_mask(self, tmp_path, ref_image):
        with pytest.raises(FileNotFoundError):
            Tractogram.simulate_tractogram(tmp_path / "missing.nii.gz")
        with pytest.raises(ValueError, match="non-zero voxel"):
            Tractogram.simulate_tractogram(
                ref_image,
                mask=np.zeros(SHAPE),
                n_bundles=1,
                n_streamlines=2,
                n_points=10,
            )


####################################################################################################
# Creation and loading
####################################################################################################
class TestCreation:
    def test_empty(self):
        tract = Tractogram()
        assert tract.tracts is None and tract.affine is None and tract.header is None
        assert tract.tract_file is None and tract.name == "default"
        assert tract.data_per_point == {} and tract.data_per_streamline == {}
        assert tract.colortables == {}

    def test_from_components(self):
        streamlines = [line(0, 4), line(0, 8)]
        tract = Tractogram(
            tracts=ArraySequence(streamlines),
            affine=np.eye(4),
            header={},
            name="parts",
            color="red",
            alpha=0.5,
        )
        assert tract.tract_file == "from_components" and tract.name == "parts"
        assert tract.data_per_streamline["length"].ravel() == pytest.approx([4, 8])
        assert tract.data_per_point == {}
        # The color travels in three places that must agree
        value = 255  # 'red' encoded as R + 256 G + 65536 B
        assert tract.colortables["default"]["color_table"].tolist() == [
            [255, 0, 0, 0.5, value]
        ]
        assert tract.data_per_streamline["default"].ravel().tolist() == [value, value]

    def test_default_color(self):
        tract = hand_made([line(0, 4)])
        assert tract.colortables["default"]["color_table"][0, 4] == DEFAULT_COLOR_VALUE
        assert tract.data_per_streamline["default"].tolist() == [[DEFAULT_COLOR_VALUE]]

    @pytest.mark.parametrize("kind", ["str", "path"])
    def test_from_trk_file(self, tmp_path, sim, kind):
        sim.save_tractogram(str(tmp_path / "bundles.trk"))
        file = tmp_path / "bundles.trk"
        loaded = Tractogram(str(file) if kind == "str" else file)
        assert loaded.tract_file == str(file) and len(loaded.tracts) == 30
        assert np.allclose(points_of(loaded), points_of(sim), atol=1e-4)
        assert loaded.header["dimensions"].tolist() == list(SHAPE)
        assert np.array_equal(loaded.affine, np.eye(4))
        assert set(loaded.data_per_streamline) == {"bundle_id", "default", "length"}
        assert (
            loaded.data_per_streamline["bundle_id"].ravel().tolist()
            == [1] * 10 + [2] * 10 + [3] * 10
        )
        assert list(loaded.colortables) == [
            "default"
        ]  # Color tables are not stored in files

    def test_from_tck_file(self, tmp_path, sim):
        sim.save_tractogram(str(tmp_path / "bundles.tck"), file_type="tck")
        loaded = Tractogram(tmp_path / "bundles.tck")
        assert len(loaded.tracts) == 30
        assert np.allclose(points_of(loaded), points_of(sim), atol=1e-4)
        assert "length" in loaded.data_per_streamline

    def test_from_nibabel_object(self, tmp_path, sim):
        sim.save_tractogram(str(tmp_path / "bundles.trk"))
        nb_tractogram = nib.streamlines.load(str(tmp_path / "bundles.trk")).tractogram
        tract = Tractogram(nb_tractogram)
        assert tract.tract_file == "from_object" and len(tract.tracts) == 30
        assert {"bundle_id", "default", "length"} == set(tract.data_per_streamline)

    def test_load_tractogram_from_object_needs_a_nibabel_object(self):
        with pytest.raises(TypeError, match="nibabel Tractogram"):
            Tractogram().load_tractogram_from_object([1, 2, 3])

    def test_data_per_point_is_loaded(self, tmp_path, lengths_tract):
        lengths_tract.save_tractogram(str(tmp_path / "ramp.trk"))
        loaded = Tractogram(tmp_path / "ramp.trk")
        assert list(loaded.data_per_point) == ["ramp"]
        assert np.allclose(
            loaded.data_per_point["ramp"][2].ravel(),
            lengths_tract.data_per_point["ramp"][2].ravel(),
        )

    def test_errors(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Tractogram(tmp_path / "missing.trk")
        (tmp_path / "streamlines.txt").write_text("not a tractogram")
        with pytest.raises(ValueError, match="Must be TRK or TCK"):
            Tractogram(tmp_path / "streamlines.txt")
        with pytest.raises(FileNotFoundError):
            Tractogram().load_tractogram_from_file(str(tmp_path / "missing.trk"))
        with pytest.raises(ValueError, match="Alpha"):
            Tractogram(alpha=2)

    def test_inconsistent_arguments(self, tmp_path):
        streamlines = ArraySequence([line(0, 4)])
        with pytest.raises(ValueError, match="affine and header must also"):
            Tractogram(tracts=streamlines)
        with pytest.raises(ValueError, match="affine and header must also"):
            Tractogram(tracts=streamlines, affine=np.eye(4))
        with pytest.raises(ValueError, match="tracts and header must also"):
            Tractogram(affine=np.eye(4))
        with pytest.raises(ValueError, match="tracts and affine must also"):
            Tractogram(header={})
        with pytest.raises(ValueError, match="Cannot specify both"):
            Tractogram(
                tmp_path / "x.trk", tracts=streamlines, affine=np.eye(4), header={}
            )


####################################################################################################
# Saving and converting
####################################################################################################
class TestSaveAndConvert:
    def test_trk_round_trip(self, tmp_path, sim):
        out = tmp_path / "out.trk"
        sim.save_tractogram(str(out))
        assert nib.streamlines.detect_format(str(out)) is nib.streamlines.TrkFile
        loaded = nib.streamlines.load(str(out))
        assert len(loaded.streamlines) == 30
        assert loaded.header["dimensions"].tolist() == list(SHAPE)  # The grid is kept
        assert np.allclose(
            np.concatenate(list(loaded.streamlines)), points_of(sim), atol=1e-4
        )

    def test_tck_round_trip(self, tmp_path, sim):
        out = tmp_path / "out.tck"
        sim.save_tractogram(str(out), file_type="tck")
        assert nib.streamlines.detect_format(str(out)) is nib.streamlines.TckFile
        assert len(nib.streamlines.load(str(out)).streamlines) == 30

    def test_save_accepts_path_objects(self, tmp_path, sim):
        sim.save_tractogram(tmp_path / "from_path.trk")
        sim.save_tractogram(tmp_path / "from_path_no_ext", file_type="tck")
        assert len(Tractogram(tmp_path / "from_path.trk").tracts) == 30
        assert (tmp_path / "from_path_no_ext.tck").is_file()

    def test_extension_follows_the_file_type(self, tmp_path, sim):
        sim.save_tractogram(str(tmp_path / "no_extension"), file_type="trk")
        sim.save_tractogram(str(tmp_path / "wrong.trk"), file_type="tck")
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "no_extension.trk",
            "wrong.tck",
        ]

    def test_overwrite(self, tmp_path, sim):
        out = str(tmp_path / "out.trk")
        sim.save_tractogram(out)
        with pytest.raises(FileExistsError, match="overwrite=True"):
            sim.save_tractogram(out)
        sim.save_tractogram(out, overwrite=True)  # No error

    def test_overwriting_replaces_a_file_of_another_format(self, tmp_path, sim):
        out = str(tmp_path / "out.trk")
        sim.save_tractogram(str(tmp_path / "out.tck"), file_type="tck")
        (tmp_path / "out.trk").write_bytes(
            (tmp_path / "out.tck").read_bytes()
        )  # Stale content
        sim.save_tractogram(out, overwrite=True)
        assert nib.streamlines.detect_format(out) is nib.streamlines.TrkFile

    def test_unsupported_file_type(self, tmp_path, sim):
        with pytest.raises(ValueError, match="Unsupported file type"):
            sim.save_tractogram(str(tmp_path / "out.xyz"), file_type="xyz")

    def test_data_per_point_survives_the_round_trip(self, tmp_path, lengths_tract):
        lengths_tract.save_tractogram(str(tmp_path / "ramp.trk"))
        loaded = nib.streamlines.load(str(tmp_path / "ramp.trk"))
        assert list(loaded.tractogram.data_per_point) == ["ramp"]
        assert loaded.tractogram.data_per_point["ramp"][4].shape == (11, 1)

    def test_data_per_streamline_survives_the_round_trip(self, tmp_path, lengths_tract):
        lengths_tract.save_tractogram(str(tmp_path / "score.trk"))
        loaded = Tractogram(tmp_path / "score.trk")
        assert loaded.data_per_streamline["score"].ravel().tolist() == [
            10,
            20,
            30,
            40,
            50,
        ]

    def test_trk2tck(self, tmp_path, sim):
        sim.save_tractogram(str(tmp_path / "in.trk"))
        result = clttract.trk2tck(tmp_path / "in.trk", tmp_path / "out.tck")
        assert isinstance(result, Tractogram)
        converted = nib.streamlines.load(str(tmp_path / "out.tck"))
        assert (
            nib.streamlines.detect_format(str(tmp_path / "out.tck"))
            is nib.streamlines.TckFile
        )
        assert np.allclose(
            np.concatenate(list(converted.streamlines)), points_of(sim), atol=1e-4
        )

    def test_trk2tck_from_an_object_and_with_extension_fixed(self, tmp_path, sim):
        clttract.trk2tck(sim, tmp_path / "from_object")
        assert (tmp_path / "from_object.tck").is_file()

    def test_trk2tck_overwrite(self, tmp_path, sim):
        sim.save_tractogram(str(tmp_path / "in.trk"))
        clttract.trk2tck(tmp_path / "in.trk", tmp_path / "out.tck")
        with pytest.raises(FileExistsError):
            clttract.trk2tck(tmp_path / "in.trk", tmp_path / "out.tck")
        clttract.trk2tck(tmp_path / "in.trk", tmp_path / "out.tck", overwrite=True)

    def test_trk2tck_errors(self, tmp_path, sim):
        with pytest.raises(FileNotFoundError):
            clttract.trk2tck(tmp_path / "missing.trk", tmp_path / "out.tck")
        sim.save_tractogram(str(tmp_path / "in.tck"), file_type="tck")
        with pytest.raises(ValueError, match="not a valid TRK"):
            clttract.trk2tck(tmp_path / "in.tck", tmp_path / "out.tck")

    def test_tck2trk_recovers_the_grid_from_the_reference_image(
        self, tmp_path, sim, ref_file
    ):
        sim.save_tractogram(str(tmp_path / "in.tck"), file_type="tck")
        result = clttract.tck2trk(tmp_path / "in.tck", ref_file, tmp_path / "out.trk")
        assert isinstance(result, Tractogram) and len(result.tracts) == 30
        assert result.header["dimensions"].tolist() == list(SHAPE)
        assert result.header["voxel_sizes"].tolist() == [2.0, 2.0, 2.0]
        order = result.header["voxel_order"]
        assert (order.decode() if isinstance(order, bytes) else order) == "RAS"
        assert np.allclose(points_of(result), points_of(sim), atol=1e-4)

    def test_tck2trk_overwrite_and_extension(self, tmp_path, sim, ref_file):
        sim.save_tractogram(str(tmp_path / "in.tck"), file_type="tck")
        clttract.tck2trk(tmp_path / "in.tck", ref_file, tmp_path / "out")
        assert (tmp_path / "out.trk").is_file()
        with pytest.raises(FileExistsError):
            clttract.tck2trk(tmp_path / "in.tck", ref_file, tmp_path / "out.trk")
        clttract.tck2trk(
            tmp_path / "in.tck", ref_file, tmp_path / "out.trk", overwrite=True
        )

    def test_tck2trk_errors(self, tmp_path, sim, ref_file):
        sim.save_tractogram(str(tmp_path / "in.tck"), file_type="tck")
        sim.save_tractogram(str(tmp_path / "in.trk"))
        with pytest.raises(FileNotFoundError, match="TCK"):
            clttract.tck2trk(tmp_path / "missing.tck", ref_file, tmp_path / "out.trk")
        with pytest.raises(FileNotFoundError, match="Reference image"):
            clttract.tck2trk(
                tmp_path / "in.tck", tmp_path / "missing.nii.gz", tmp_path / "out.trk"
            )
        with pytest.raises(ValueError, match="not a valid TCK"):
            clttract.tck2trk(tmp_path / "in.trk", ref_file, tmp_path / "out.trk")


####################################################################################################
# Maps, lengths and information
####################################################################################################
class TestMapsAndInformation:
    def test_lengths(self, lengths_tract):
        lengths = lengths_tract.compute_streamline_lengths()
        assert lengths.shape == (5, 1)
        assert lengths.ravel() == pytest.approx([2, 4, 6, 8, 10])
        assert np.array_equal(lengths_tract.data_per_streamline["length"], lengths)

    def test_lengths_of_a_curved_streamline(self):
        quarter_turn = np.c_[
            np.cos(np.linspace(0, np.pi / 2, 2001)),
            np.sin(np.linspace(0, np.pi / 2, 2001)),
            np.zeros(2001),
        ]
        tract = hand_made([quarter_turn.astype(np.float32)])
        assert tract.data_per_streamline["length"][0, 0] == pytest.approx(
            np.pi / 2, abs=1e-4
        )

    def test_lengths_need_streamlines(self):
        with pytest.raises(ValueError, match="no streamlines"):
            Tractogram().compute_streamline_lengths()

    def test_list_maps(self, lengths_tract):
        maps = lengths_tract.list_maps()
        assert maps["maps_per_point"] == ["ramp"]
        assert set(maps["maps_per_streamline"]) == {"default", "length", "score"}
        assert Tractogram().list_maps() == {
            "maps_per_point": [],
            "maps_per_streamline": [],
        }

    def test_remove_maps_and_its_colortable(self, sim):
        assert "bundle_id" in sim.colortables
        sim.remove_maps("bundle_id")
        assert "bundle_id" not in sim.data_per_streamline
        assert "bundle_id" not in sim.colortables
        assert "default" in sim.colortables

    def test_remove_maps_stored_per_point(self, lengths_tract):
        lengths_tract.remove_maps("ramp")
        assert lengths_tract.list_maps()["maps_per_point"] == []
        assert "score" in lengths_tract.data_per_streamline

    def test_remove_maps_stored_in_both(self, lengths_tract):
        lengths_tract.data_per_point["score"] = lengths_tract.data_per_point["ramp"]
        lengths_tract.remove_maps("score")
        maps = lengths_tract.list_maps()
        assert "score" not in maps["maps_per_point"]
        assert "score" not in maps["maps_per_streamline"]

    def test_remove_several_maps(self, lengths_tract):
        lengths_tract.remove_maps(["ramp", "score"])
        maps = lengths_tract.list_maps()
        assert maps["maps_per_point"] == []
        assert set(maps["maps_per_streamline"]) == {"default", "length"}

    def test_remove_several_maps_is_all_or_nothing(self, lengths_tract):
        with pytest.raises(ValueError, match="nope"):
            lengths_tract.remove_maps(["ramp", "nope"])
        with pytest.raises(ValueError, match="default"):
            lengths_tract.remove_maps(["ramp", "default"])
        assert lengths_tract.list_maps()["maps_per_point"] == ["ramp"]

    def test_remove_maps_errors(self, sim):
        with pytest.raises(ValueError, match="default"):
            sim.remove_maps("default")
        with pytest.raises(ValueError, match="not found"):
            sim.remove_maps("nope")

    def test_maps_info(self, lengths_tract):
        info = lengths_tract.get_maps_info().set_index("map_name")
        assert isinstance(lengths_tract.get_maps_info(), pd.DataFrame)
        assert info.loc["score", "storage_mode"] == "data_per_streamline"
        assert info.loc["score", ["min_value", "max_value", "mean_value"]].tolist() == [
            10,
            50,
            30,
        ]
        assert info.loc["score", "number_of_streamlines"] == 5
        assert info.loc["ramp", "storage_mode"] == "data_per_point"
        assert info.loc["ramp", "num_values"] == 55  # 5 streamlines of 11 points
        assert info.loc["ramp", "max_value"] == pytest.approx(5.0)

    def test_maps_info_of_an_empty_tractogram(self):
        assert Tractogram().get_maps_info().empty

    def test_tractogram_info(self, lengths_tract):
        info = lengths_tract.get_tractogram_info()
        assert info["number_of_streamlines"] == 5
        assert info["mean_streamline_length"] == pytest.approx(6.0)
        assert np.array_equal(info["affine"], np.eye(4))
        assert info["header"] == {"nb_streamlines": 5}

    def test_tractogram_info_of_an_empty_tractogram(self):
        assert Tractogram().get_tractogram_info() == {
            "number_of_streamlines": 0,
            "mean_streamline_length": None,
            "affine": None,
            "header": None,
        }

    def test_streamline_values_are_copied_to_every_point(self, sim):
        points = sim.streamline_to_points("bundle_id")
        assert isinstance(points, ArraySequence) and len(points) == 30
        assert all(p.shape == (30, 1) for p in points)
        assert [np.unique(p).tolist() for p in points[::10]] == [[1], [2], [3]]
        assert sim.data_per_point["bundle_id"] is not None
        # The color table of the map is propagated to the new map
        sim.streamline_to_points("bundle_id", "bundle_pt")
        assert np.array_equal(
            sim.colortables["bundle_pt"]["color_table"],
            sim.colortables["bundle_id"]["color_table"],
        )

    def test_streamline_to_points_with_another_name(self, lengths_tract):
        lengths_tract.streamline_to_points("score", "score_per_point")
        values = lengths_tract.data_per_point["score_per_point"]
        assert [np.unique(v).tolist() for v in values] == [[10], [20], [30], [40], [50]]

    def test_streamline_to_points_errors(self, lengths_tract):
        with pytest.raises(ValueError, match="not found in data_per_streamline"):
            lengths_tract.streamline_to_points("nope")
        lengths_tract.data_per_streamline["pair"] = np.ones((5, 2))
        with pytest.raises(ValueError, match="single value per streamline"):
            lengths_tract.streamline_to_points("pair")

    @pytest.mark.parametrize(
        "metric, expected",
        [("mean", 0.5), ("median", 0.5), ("max", 1.0), ("min", 0.0), ("anything", 0.5)],
    )
    def test_point_values_are_summarized_per_streamline(self, sim, metric, expected):
        sim.data_per_point["flat"] = ArraySequence(
            [np.linspace(0, 1, len(s)) for s in sim.tracts]
        )
        values = sim.points_to_streamline("flat", "summary", metric=metric)
        assert values.shape == (30, 1) and values.dtype == np.float64
        assert np.allclose(values, expected)
        assert sim.data_per_streamline["summary"] is values

    def test_points_to_streamline_uses_the_map_name_by_default(self, sim):
        sim.data_per_point["flat"] = ArraySequence(
            [np.ones(len(s)) for s in sim.tracts]
        )
        sim.points_to_streamline("flat")
        assert "flat" in sim.data_per_streamline

    def test_points_to_streamline_follows_each_streamline(self, lengths_tract):
        lengths_tract.data_per_point["flat"] = ArraySequence(
            [np.full(len(s), float(i)) for i, s in enumerate(lengths_tract.tracts)]
        )
        values = lengths_tract.points_to_streamline("flat", metric="max")
        assert values.ravel().tolist() == [0, 1, 2, 3, 4]

    def test_points_to_streamline_unknown_map(self, sim):
        with pytest.raises(ValueError, match="not found in data_per_point"):
            sim.points_to_streamline("nope")

    def test_point_maps_with_several_features(self, sim):
        sim.data_per_point["two"] = ArraySequence(
            [
                np.c_[np.linspace(0, 1, len(s)), np.linspace(0, 10, len(s))]
                for s in sim.tracts
            ]
        )
        mean = sim.points_to_streamline("two", "two_mean")
        assert mean.shape == (30, 2) and np.allclose(mean, [0.5, 5.0])
        maximum = sim.points_to_streamline("two", "two_max", metric="max")
        assert np.allclose(maximum, [1.0, 10.0])

    def test_column_shaped_point_maps_give_a_numeric_result(self, sim):
        sim.streamline_to_points("bundle_id", "bundle_pt")
        values = sim.points_to_streamline("bundle_pt", "bundle_back")
        assert values.dtype != object
        assert (
            values.astype(float).ravel().tolist()
            == [1.0] * 10 + [2.0] * 10 + [3.0] * 10
        )


####################################################################################################
# Geometry
####################################################################################################
class TestGeometry:
    @staticmethod
    def roughness(streamline):
        return float(np.sum(np.diff(streamline, n=2, axis=0) ** 2))

    @pytest.fixture
    def zigzag(self):
        rng = np.random.default_rng(0)
        x = np.linspace(0, 30, 40)
        points = np.c_[x, rng.normal(0, 1, 40), np.zeros(40)].astype(np.float32)
        return hand_made([points])

    def test_smoothing_reduces_the_roughness(self, zigzag):
        before = self.roughness(zigzag.tracts[0])
        zigzag.smooth_streamlines(iterations=3, sigma=2.0)
        assert self.roughness(zigzag.tracts[0]) < 0.05 * before
        assert zigzag.tracts[0].shape == (40, 3)

    def test_stronger_smoothing_is_smoother(self, zigzag):
        weak, strong = hand_made([zigzag.tracts[0].copy()]), hand_made(
            [zigzag.tracts[0].copy()]
        )
        weak.smooth_streamlines(iterations=1, sigma=2.0)
        strong.smooth_streamlines(iterations=3, sigma=2.0)
        assert self.roughness(strong.tracts[0]) < self.roughness(weak.tracts[0])
        wider = hand_made([zigzag.tracts[0].copy()])
        wider.smooth_streamlines(iterations=3, sigma=4.0)
        assert self.roughness(wider.tracts[0]) < self.roughness(strong.tracts[0])

    def test_smoothing_leaves_a_straight_line_alone(self):
        tract = hand_made([line(0, 30, 5, 5, n=20)])
        tract.smooth_streamlines(iterations=2, sigma=2.0)
        assert np.allclose(
            tract.tracts[0][5:-5], line(0, 30, 5, 5, n=20)[5:-5], atol=1e-3
        )
        assert np.allclose(tract.tracts[0][:, 1:], 5, atol=1e-3)

    def test_smoothing_keeps_the_container_type(self):
        tract = Tractogram(
            tracts=[line(0, 30).astype(float)], affine=np.eye(4), header={}
        )
        tract.smooth_streamlines()
        assert isinstance(tract.tracts, list)

    def test_smoothing_needs_streamlines(self):
        with pytest.raises(ValueError, match="No streamlines loaded"):
            Tractogram().smooth_streamlines()

    def test_resampling(self, sim):
        before = sim.tracts[0].copy()
        resampled = sim.resample_streamlines(nb_points=11)
        assert isinstance(resampled, ArraySequence)
        assert {len(s) for s in sim.tracts} == {11} and len(resampled) == 30
        assert np.allclose(sim.tracts[0][0], before[0], atol=1e-4)
        assert np.allclose(sim.tracts[0][-1], before[-1], atol=1e-4)
        assert sim.header["nb_streamlines"] == 30

    def test_resampled_points_are_equally_spaced(self):
        tract = hand_made([line(0, 10, n=4), line(0, 20, n=7)])
        tract.resample_streamlines(nb_points=5)
        for streamline, length in zip(tract.tracts, (10, 20), strict=True):
            steps = np.linalg.norm(np.diff(streamline, axis=0), axis=1)
            assert steps == pytest.approx([length / 4] * 4, abs=1e-4)

    def test_resampling_interpolates_the_point_maps(self, lengths_tract):
        lengths_tract.data_per_point["ramp"] = ArraySequence(
            [np.linspace(0, 1, len(s)).reshape(-1, 1) for s in lengths_tract.tracts]
        )
        lengths_tract.resample_streamlines(nb_points=5)
        for values in lengths_tract.data_per_point["ramp"]:
            assert values.shape == (5, 1)
            assert values.ravel() == pytest.approx([0, 0.25, 0.5, 0.75, 1.0], abs=1e-5)

    def test_resampling_keeps_the_container_type(self):
        tract = Tractogram(
            tracts=[line(0, 10), line(0, 20)], affine=np.eye(4), header={}
        )
        resampled = tract.resample_streamlines(nb_points=6)
        assert isinstance(resampled, list) and isinstance(tract.tracts, list)
        assert all(len(s) == 6 for s in resampled)

    @pytest.mark.parametrize("nb_points", [0, -3, 5.5])
    def test_invalid_number_of_points(self, sim, nb_points):
        with pytest.raises(ValueError, match="positive integer"):
            sim.resample_streamlines(nb_points)

    def test_resampling_needs_streamlines(self):
        with pytest.raises(ValueError, match="No streamlines loaded"):
            Tractogram().resample_streamlines(5)

    def test_resample_function_with_arrays_and_lists(self):
        as_list = clttract.resample_streamlines(
            [line(0, 10, n=4), line(0, 20, n=9)], nb_points=6
        )
        assert isinstance(as_list, list) and [s.shape for s in as_list] == [
            (6, 3),
            (6, 3),
        ]
        as_sequence = clttract.resample_streamlines(
            ArraySequence([line(0, 10, n=4)]), nb_points=6
        )
        assert isinstance(as_sequence, ArraySequence) and as_sequence[0].shape == (6, 3)
        assert as_sequence[0][:, 0] == pytest.approx(np.linspace(0, 10, 6), abs=1e-4)

    @pytest.mark.parametrize(
        "streamlines, kwargs, message",
        [
            ([], {}, "empty"),
            ("not streamlines", {}, "list of numpy arrays"),
            ([line(0, 1)], {"nb_points": 0}, "positive integer"),
            ([np.zeros((1, 3))], {}, "at least 2 points"),
            ([np.zeros((5, 2))], {}, "shape"),
            ([[1, 2, 3]], {}, "not a valid numpy array"),
        ],
    )
    def test_resample_function_errors(self, streamlines, kwargs, message):
        with pytest.raises(ValueError, match=message):
            clttract.resample_streamlines(streamlines, **kwargs)

    def test_interpolating_values_along_a_streamline(self):
        old = np.c_[np.linspace(0, 10, 11), np.zeros(11), np.zeros(11)]
        new = np.c_[np.linspace(0, 10, 21), np.zeros(21), np.zeros(21)]
        values = clttract.interpolate_streamline_values(old, np.arange(11.0), new)
        assert values == pytest.approx(np.linspace(0, 10, 21))
        nearest = clttract.interpolate_streamline_values(
            old, np.arange(11.0), new, method="nearest"
        )
        assert set(nearest) <= set(range(11))

    def test_interpolating_several_features_at_once(self):
        old = np.c_[np.linspace(0, 10, 11), np.zeros(11), np.zeros(11)]
        new = np.c_[np.linspace(0, 10, 6), np.zeros(6), np.zeros(6)]
        features = np.c_[np.arange(11.0), 2 * np.arange(11.0)]
        values = clttract.interpolate_streamline_values(old, features, new)
        assert values.shape == (6, 2)
        assert values[:, 0] == pytest.approx(np.linspace(0, 10, 6))
        assert values[:, 1] == pytest.approx(np.linspace(0, 20, 6))

    def test_applying_an_affine(self, sim):
        affine = np.array([[0, -1, 0, 5], [1, 0, 0, -3], [0, 0, 2, 7], [0, 0, 0, 1.0]])
        before = points_of(sim).copy()
        assert sim.apply_affine(affine) is None
        assert isinstance(sim.tracts, ArraySequence)
        assert np.allclose(
            points_of(sim), nib.affines.apply_affine(affine, before), atol=1e-4
        )
        assert [len(s) for s in sim.tracts] == [
            30
        ] * 30  # Streamlines keep their points

    def test_applying_the_inverse_undoes_the_affine(self, sim):
        affine = np.array([[0, -1, 0, 5], [1, 0, 0, -3], [0, 0, 2, 7], [0, 0, 0, 1.0]])
        before = points_of(sim).copy()
        sim.apply_affine(affine)
        sim.apply_affine(affine, inverse=True)
        assert np.allclose(points_of(sim), before, atol=1e-3)

    def test_applying_an_affine_without_modifying_the_tractogram(self, sim):
        shift = np.eye(4)
        shift[:3, 3] = [10, 0, 0]
        before = points_of(sim).copy()
        moved = sim.apply_affine(shift, inplace=False)
        assert isinstance(moved, ArraySequence)
        assert np.array_equal(points_of(sim), before)
        assert np.allclose(np.concatenate(list(moved)), before + [10, 0, 0], atol=1e-4)

    def test_applying_an_affine_needs_a_4x4_matrix(self, sim):
        with pytest.raises(ValueError, match="4x4"):
            sim.apply_affine(np.eye(3))

    @pytest.mark.parametrize(
        "percentage, expected", [(100, 30), (50, 15), (34, 10), (25, 8), (1, 1)]
    )
    def test_reducing_the_number_of_streamlines(self, sim, percentage, expected):
        sim.reduce_streamlines(percentage)
        assert len(sim.tracts) == expected
        assert sim.header["nb_streamlines"] == expected
        assert sim.data_per_streamline["bundle_id"].shape == (expected, 1)

    def test_reducing_keeps_every_nth_streamline_and_its_maps(self, sim):
        original = [s.copy() for s in sim.tracts]
        sim.data_per_point["ramp"] = ArraySequence(
            [np.full((len(s), 1), float(i)) for i, s in enumerate(sim.tracts)]
        )
        sim.reduce_streamlines(50)  # Every 2nd streamline
        assert np.array_equal(sim.tracts[1], original[2])
        assert [v[0, 0] for v in sim.data_per_point["ramp"]][:3] == [0.0, 2.0, 4.0]

    @pytest.mark.parametrize("percentage", [0, -5, 101])
    def test_invalid_percentage(self, sim, percentage):
        with pytest.raises(ValueError, match="between 0 and 100"):
            sim.reduce_streamlines(percentage)


####################################################################################################
# Filtering by condition
####################################################################################################
class TestFilter:
    def test_by_length(self, lengths_tract):
        kept = quiet_call(lengths_tract.filter, "length > 5")
        assert len(kept) == 3 and len(lengths_tract.tracts) == 3
        assert lengths_tract.data_per_streamline["score"].ravel().tolist() == [
            30,
            40,
            50,
        ]
        assert lengths_tract.data_per_streamline["length"].ravel() == pytest.approx(
            [6, 8, 10]
        )
        assert len(lengths_tract.data_per_point["ramp"]) == 3
        assert lengths_tract.header["nb_streamlines"] == 3

    @pytest.mark.parametrize(
        "condition, expected",
        [
            ("length > 6", [8, 10]),
            ("length >= 6", [6, 8, 10]),
            ("length < 6", [2, 4]),
            ("length <= 6", [2, 4, 6]),
            ("length == 6", [6]),
            ("length != 6", [2, 4, 8, 10]),
            ("3 < length < 9", [4, 6, 8]),
        ],
    )
    def test_conditions(self, lengths_tract, condition, expected):
        quiet_call(lengths_tract.filter, condition)
        assert lengths_tract.data_per_streamline["length"].ravel() == pytest.approx(
            expected
        )

    def test_maps_can_be_stored_in_any_container(self, lengths_tract):
        lengths_tract.data_per_streamline["as_list"] = [[1], [2], [3], [4], [5]]
        lengths_tract.data_per_streamline["as_sequence"] = ArraySequence(
            [np.array([10 * i]) for i in range(1, 6)]
        )
        quiet_call(lengths_tract.filter, "length > 5")  # Keeps the last three
        for name, expected in (("as_list", [3, 4, 5]), ("as_sequence", [30, 40, 50])):
            values = lengths_tract.data_per_streamline[name]
            assert np.asarray(values).ravel().tolist() == expected, name

    def test_color_table_drops_entries_without_streamlines(self, sim):
        sim.data_per_streamline["bundle_id"][
            sim.data_per_streamline["bundle_id"] == 3
        ] = 1
        quiet_call(sim.filter, "bundle_id >= 1")  # Nobody is left in bundle 3
        assert np.unique(sim.data_per_streamline["bundle_id"]).tolist() == [1, 2]
        assert sim.colortables["bundle_id"]["color_table"][:, 4].tolist() == [1, 2]
        assert sim.colortables["bundle_id"]["names"] == ["bundle_00", "bundle_01"]

    def test_color_table_of_a_point_map(self, sim):
        sim.data_per_point["roi"] = ArraySequence(
            [np.full(len(s), float(i % 3 + 1)) for i, s in enumerate(sim.tracts)]
        )
        sim.colortables["roi"] = {
            "names": ["a", "b", "c"],
            "color_table": np.array(
                [[255, 0, 0, 1, 1], [0, 255, 0, 1, 2], [0, 0, 255, 1, 3]], dtype=float
            ),
            "lookup_table": None,
        }
        quiet_call(sim.filter, "roi >= 2")
        assert len(sim.tracts) == 20 and len(sim.data_per_point["roi"]) == 20
        assert sim.colortables["roi"]["color_table"][:, 4].tolist() == [2, 3]
        assert sim.colortables["roi"]["names"] == ["b", "c"]

    def test_tractograms_made_by_merging(self, ref_image):
        first = Tractogram.simulate_tractogram(
            ref_image, n_bundles=3, n_streamlines=6, n_points=20, seed=0
        )
        second = Tractogram.simulate_tractogram(
            ref_image, n_bundles=2, n_streamlines=6, n_points=20, seed=1
        )
        # add_tractogram numbers the tractograms from 0 and merge_tractograms from 1
        cases = [
            (first.add_tractogram(second), "tract_id >= 1", 1),
            (clttract.merge_tractograms([first, second]), "tract_id >= 2", 2),
        ]
        for merged, condition, kept_id in cases:
            kept = quiet_call(merged.filter, condition)
            assert len(kept) == 12 and len(merged.tracts) == 12  # The second tractogram
            assert all(len(v) == 12 for v in merged.data_per_streamline.values())
            assert merged.colortables["tract_id"]["color_table"][:, 4].tolist() == [
                kept_id
            ]

    def test_by_another_map(self, lengths_tract):
        kept = quiet_call(lengths_tract.filter, "score >= 40")
        assert len(kept) == 2
        assert lengths_tract.data_per_streamline["score"].ravel().tolist() == [40, 50]

    def test_by_a_point_map_uses_its_mean(self, lengths_tract):
        # The mean of the ramp of streamline i is (i + 1) / 2
        quiet_call(lengths_tract.filter, "ramp > 1.2")
        assert len(lengths_tract.tracts) == 3
        assert len(lengths_tract.data_per_point["ramp"]) == 3

    def test_returns_the_kept_streamlines(self, lengths_tract):
        kept = quiet_call(lengths_tract.filter, "length > 8")
        assert len(kept) == 1 and np.array_equal(kept[0], lengths_tract.tracts[0])

    def test_nothing_matches(self, lengths_tract):
        assert quiet_call(lengths_tract.filter, "length > 100") is None
        assert len(lengths_tract.tracts) == 5  # Left untouched

    def test_unknown_map(self, lengths_tract):
        with pytest.raises(ValueError, match="not found"):
            lengths_tract.filter("nope > 3")

    def test_color_table_keeps_the_entries_of_the_kept_bundles(self, sim):
        quiet_call(sim.filter, "bundle_id >= 2")
        assert np.unique(sim.data_per_streamline["bundle_id"]).tolist() == [2, 3]
        assert sim.colortables["bundle_id"]["color_table"][:, 4].tolist() == [2, 3]
        assert sim.colortables["bundle_id"]["names"] == ["bundle_01", "bundle_02"]

    @pytest.mark.parametrize(
        "condition, kept",
        [
            ("bundle_id == 1", [1]),
            ("bundle_id < 3", [1, 2]),
            ("2 <= bundle_id <= 3", [2, 3]),
        ],
    )
    def test_color_table_follows_other_conditions(self, sim, condition, kept):
        quiet_call(sim.filter, condition)
        assert np.unique(sim.data_per_streamline["bundle_id"]).tolist() == kept
        assert sim.colortables["bundle_id"]["color_table"][:, 4].tolist() == kept
        # The colors of the surviving bundles can still be resolved
        colors = sim.get_pointwise_colors("bundle_id")
        assert len(colors) == len(sim.tracts)


####################################################################################################
# Interpolating images on the streamlines
####################################################################################################
class TestInterpolateOnTractogram:
    @pytest.fixture
    def x_map(self, tmp_path):
        """Image whose value is the x coordinate (mm) of each voxel."""
        i = np.indices(SHAPE)[0]
        path = tmp_path / "x.nii.gz"
        nib.save(nib.Nifti1Image((i * 2.0 - 60.0).astype(np.float32), AFFINE), path)
        return path

    def test_values_per_point(self, sim, x_map):
        sim.interpolate_on_tractogram(x_map, map_name="x")
        values = sim.data_per_point["x"]
        assert len(values) == 30 and all(v.shape == (30,) for v in values)
        for value, streamline in zip(values, sim.tracts, strict=True):
            assert value == pytest.approx(
                streamline[:, 0], abs=1e-3
            )  # Interpolation is exact

    @pytest.mark.parametrize(
        "reduction, function",
        [("mean", np.mean), ("median", np.median), ("min", np.min), ("max", np.max)],
    )
    def test_values_per_streamline(self, sim, x_map, reduction, function):
        sim.interpolate_on_tractogram(
            x_map, storage_mode="data_per_streamline", map_name="x", reduction=reduction
        )
        expected = [function(s[:, 0]) for s in sim.tracts]
        assert sim.data_per_streamline[f"x_{reduction}"].shape == (30, 1)
        assert sim.data_per_streamline[f"x_{reduction}"].ravel() == pytest.approx(
            expected, abs=1e-3
        )
        assert "x" not in sim.data_per_point  # Only the requested storage is filled

    def test_path_argument_types(self, sim, x_map):
        sim.interpolate_on_tractogram(str(x_map), map_name="from_str")
        sim.interpolate_on_tractogram(x_map, map_name="from_path")
        assert np.allclose(
            np.concatenate(sim.data_per_point["from_str"]),
            np.concatenate(sim.data_per_point["from_path"]),
        )

    def test_nearest_neighbour(self, sim, x_map):
        sim.interpolate_on_tractogram(x_map, map_name="linear")
        sim.interpolate_on_tractogram(
            x_map, interp_method="nearest", map_name="nearest"
        )
        nearest = np.concatenate(sim.data_per_point["nearest"])
        assert not np.allclose(nearest, np.concatenate(sim.data_per_point["linear"]))
        # Voxel centers lie every 2 mm, starting at -60 mm
        assert np.allclose((nearest + 60) % 2, 0)

    def test_points_outside_of_the_image_are_nan(self, sim, x_map):
        sim.apply_affine(
            np.array([[1, 0, 0, 1000], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        )
        sim.interpolate_on_tractogram(x_map, map_name="x")
        assert np.isnan(np.concatenate(sim.data_per_point["x"])).all()
        sim.interpolate_on_tractogram(
            x_map, storage_mode="data_per_streamline", map_name="x"
        )
        assert np.isnan(sim.data_per_streamline["x_mean"]).all()

    def test_parcellation_labels_and_colors(self, sim):
        voxels = nib.affines.apply_affine(np.linalg.inv(AFFINE), points_of(sim))
        split = int(np.median(voxels[:, 0]))
        data = np.full(SHAPE, 2, dtype=np.int32)
        data[:split] = 1
        parcellation = Parcellation(
            data,
            color_table={
                "index": [1, 2],
                "name": ["low", "high"],
                "color": ["#ff0000", "#0000ff"],
            },
            affine=AFFINE,
        )
        sim.interpolate_on_tractogram(parcellation, map_name="roi")
        labels = np.concatenate(sim.data_per_point["roi"])
        assert set(np.unique(labels).tolist()) == {1.0, 2.0}
        assert np.array_equal(labels, np.where(np.rint(voxels[:, 0]) < split, 1.0, 2.0))
        assert sim.colortables["roi"]["names"] == ["low", "high"]
        assert sim.colortables["roi"]["color_table"][:, 4].tolist() == [1, 2]

    def test_errors(self, tmp_path, sim, x_map):
        with pytest.raises(FileNotFoundError):
            sim.interpolate_on_tractogram(tmp_path / "missing.nii.gz")
        with pytest.raises(ValueError, match="interpolation method"):
            sim.interpolate_on_tractogram(x_map, interp_method="cubic")
        with pytest.raises(ValueError, match="storage mode"):
            sim.interpolate_on_tractogram(x_map, storage_mode="elsewhere")
        with pytest.raises(ValueError, match="reduction"):
            sim.interpolate_on_tractogram(x_map, reduction="sum")
        (tmp_path / "broken.nii.gz").write_text("this is not an image")
        with pytest.raises(OSError, match="Failed to load"):
            sim.interpolate_on_tractogram(tmp_path / "broken.nii.gz")


####################################################################################################
# Colors
####################################################################################################
class TestColors:
    @pytest.mark.parametrize(
        "color, rgb",
        [
            ("#d62728", [214, 39, 40]),
            ("red", [255, 0, 0]),
            ((214, 39, 40), [214, 39, 40]),
            (np.array([214, 39, 40]), [214, 39, 40]),
            ([[214, 39, 40]], [214, 39, 40]),
        ],
    )
    def test_set_color(self, sim, color, rgb):
        sim.set_color(color)
        row = sim.colortables["default"]["color_table"][0]
        assert row[:3].tolist() == rgb and row[3] == 1.0
        encoded = rgb[0] + 256 * rgb[1] + 65536 * rgb[2]
        assert row[4] == encoded
        assert sim.data_per_streamline["default"].ravel().tolist() == [encoded] * 30
        assert sim.color == [f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"]

    def test_set_color_from_a_flat_list(self, sim):
        sim.set_color([214, 39, 40])
        assert sim.colortables["default"]["color_table"][0, :3].tolist() == [
            214,
            39,
            40,
        ]

    def test_set_color_alpha(self, sim):
        sim.set_color("#0000ff", alpha=0.25)
        assert sim.alpha == 0.25
        assert sim.colortables["default"]["color_table"][0, 3] == 0.25
        sim.set_color("red")  # The opacity is kept when none is given
        assert sim.colortables["default"]["color_table"][0, 3] == 0.25

    def test_set_color_leaves_the_other_maps_alone(self, sim):
        bundles = sim.colortables["bundle_id"]["color_table"].copy()
        points = points_of(sim).copy()
        sim.set_color("red")
        assert np.array_equal(sim.colortables["bundle_id"]["color_table"], bundles)
        assert np.array_equal(points_of(sim), points)

    def test_set_color_errors(self, sim):
        with pytest.raises(ValueError, match="Alpha"):
            sim.set_color("red", alpha=2)

    @pytest.mark.parametrize("color", [[214.0, 39.0, 40.0], (214.0, 39, 40)])
    def test_set_color_from_a_flat_list_of_numbers(self, sim, color):
        sim.set_color(color)
        assert sim.colortables["default"]["color_table"][
            0, :3
        ].tolist() == pytest.approx(np.asarray(color, dtype=float), abs=1.0)

    def test_set_color_rejects_several_colors(self, sim):
        with pytest.raises(ValueError):
            sim.set_color(["red", "blue"])

    COLORTABLE = {
        "index": [1, 2, 3],
        "name": ["a", "b", "c"],
        "color": ["#ff0000", "#00ff00", "#0000ff"],
    }

    def test_load_colortable_for_an_existing_map(self, sim):
        sim.load_colortable(self.COLORTABLE, map_name="bundle_id")
        table = sim.colortables["bundle_id"]
        assert table["names"] == ["a", "b", "c"]
        assert table["color_table"][:, :3].tolist() == [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ]
        assert table["color_table"][:, 4].tolist() == [1, 2, 3]  # The values of the map

    def test_load_colortable_opacity(self, sim):
        sim.load_colortable(self.COLORTABLE, map_name="bundle_id", opacity=0.5)
        assert sim.colortables["bundle_id"]["color_table"][:, 3].tolist() == [0.5] * 3
        sim.load_colortable(
            self.COLORTABLE, map_name="bundle_id", opacity=[0.1, 0.2, 0.3]
        )
        assert sim.colortables["bundle_id"]["color_table"][:, 3].tolist() == [
            0.1,
            0.2,
            0.3,
        ]

    def test_load_colortable_for_a_new_map(self, sim):
        sim.load_colortable(self.COLORTABLE, map_name="new_map")
        assert sim.colortables["new_map"]["names"] == ["a", "b", "c"]

    @pytest.mark.parametrize("as_str", [True, False])
    def test_load_colortable_from_a_lut_file(self, tmp_path, sim, as_str):
        lut = tmp_path / "bundles.lut"
        lut.write_text("1 a 255 0 0 0\n2 b 0 255 0 0\n3 c 0 0 255 0\n")
        sim.load_colortable(str(lut) if as_str else lut, map_name="bundle_id")
        assert sim.colortables["bundle_id"]["names"] == ["a", "b", "c"]

    def test_load_colortable_errors(self, tmp_path, sim):
        with pytest.raises(ValueError, match="does not cover all IDs"):
            sim.load_colortable(
                {"index": [1, 2], "name": ["a", "b"], "color": ["#ff0000", "#00ff00"]},
                map_name="bundle_id",
            )
        with pytest.raises(ValueError, match="does not exist"):
            sim.load_colortable(tmp_path / "missing.lut")

    def test_load_colortable_needs_a_path_or_a_dictionary(self, sim):
        with pytest.raises(TypeError, match="path or a dictionary"):
            sim.load_colortable(5)

    def test_colors_of_a_categorical_map(self, sim):
        colors = sim.get_pointwise_colors("bundle_id")
        table = sim.colortables["bundle_id"]["color_table"]
        assert isinstance(colors, ArraySequence) and len(colors) == 30
        assert all(c.shape[0] == 30 for c in colors)  # One color per point
        for i, bundle in enumerate([0] * 10 + [1] * 10 + [2] * 10):
            assert (
                colors[i][:, :3] == table[bundle, :3]
            ).all()  # Same color along a streamline

    def test_colors_of_a_scalar_map_per_streamline(self, sim):
        colors = sim.get_pointwise_colors("length", colormap="viridis")
        lengths = sim.data_per_streamline["length"].ravel()
        assert all((c == c[0]).all() for c in colors)
        assert colors[int(lengths.argmin())][0].tolist() == [68, 1, 84]  # Lowest value
        assert colors[int(lengths.argmax())][0] == pytest.approx([253, 231, 36], abs=1)

    def test_colors_of_a_per_point_map(self, sim):
        sim.data_per_point["ramp"] = ArraySequence(
            [np.linspace(0, 1, len(s)) for s in sim.tracts]
        )
        colors = sim.get_pointwise_colors("ramp", colormap="viridis")
        assert all(c.shape == (30, 3) for c in colors)
        assert colors[0][0].tolist() == [68, 1, 84]
        assert colors[0][-1] == pytest.approx([253, 231, 36], abs=1)
        assert (
            len({tuple(row) for row in colors[0]}) > 10
        )  # It varies along the streamline

    def test_colors_outside_of_a_range_use_the_range_color(self, sim):
        sim.data_per_point["ramp"] = ArraySequence(
            [np.linspace(0, 1, len(s)) for s in sim.tracts]
        )
        colors = sim.get_pointwise_colors(
            "ramp", range_min=0.5, range_max=0.9, range_color=(1, 2, 3, 255)
        )
        values = np.linspace(0, 1, 30)
        outside = (values < 0.5) | (values > 0.9)
        assert (colors[0][outside] == [1, 2, 3]).all()
        assert not (colors[0][~outside] == [1, 2, 3]).all(axis=1).any()

    def test_color_limits(self, sim):
        sim.data_per_point["ramp"] = ArraySequence(
            [np.linspace(0, 1, len(s)) for s in sim.tracts]
        )
        saturated = sim.get_pointwise_colors("ramp", vmin=2.0, vmax=3.0)
        assert (
            np.concatenate(list(saturated)) == [68, 1, 84]
        ).all()  # Everything is below vmin

    def test_range_with_column_shaped_point_maps(self, sim):
        sim.data_per_point["column"] = ArraySequence(
            [np.linspace(0, 1, len(s)).reshape(-1, 1) for s in sim.tracts]
        )
        sim.data_per_point["flat"] = ArraySequence(
            [np.linspace(0, 1, len(s)) for s in sim.tracts]
        )
        kwargs = {"range_min": 0.5, "range_max": 0.9, "range_color": (1, 2, 3, 255)}
        column = sim.get_pointwise_colors("column", **kwargs)
        flat = sim.get_pointwise_colors("flat", **kwargs)
        assert all(np.array_equal(a, b) for a, b in zip(column, flat, strict=True))
        assert (column[0][0] == [1, 2, 3]).all()  # Below the range

    def test_colors_of_an_unknown_map(self, sim):
        with pytest.raises(ValueError, match="not found"):
            sim.get_pointwise_colors("nope")


####################################################################################################
# Merging tractograms
####################################################################################################
class TestMerge:
    @pytest.fixture
    def pair(self, ref_image):
        first = Tractogram.simulate_tractogram(
            ref_image, n_bundles=3, n_streamlines=10, n_points=30, seed=0
        )
        second = Tractogram.simulate_tractogram(
            ref_image, n_bundles=2, n_streamlines=5, n_points=30, seed=1
        )
        return first, second

    def test_add_tractogram(self, pair):
        first, second = pair
        merged = first.add_tractogram(second)
        assert isinstance(merged, Tractogram) and merged is not first
        assert merged.name == "merged_tractogram" and len(merged.tracts) == 40
        assert np.array_equal(merged.tracts[0], first.tracts[0])
        assert np.array_equal(
            merged.tracts[30], second.tracts[0]
        )  # The new ones come last
        assert (len(first.tracts), len(second.tracts)) == (
            30,
            10,
        )  # Inputs are untouched

    def test_add_tractogram_labels_the_source(self, pair):
        merged = pair[0].add_tractogram(pair[1])
        ids = np.concatenate(
            [np.ravel(v) for v in merged.data_per_streamline["tract_id"]]
        )
        assert ids.tolist() == [0] * 30 + [1] * 10
        table = merged.colortables["tract_id"]
        assert table["names"] == ["Tractogram_0", "Tractogram_1"]
        assert table["color_table"][:, 4].tolist() == [0, 1]

    def test_merged_maps_are_arrays_like_those_of_any_tractogram(self, pair):
        merged = pair[0].add_tractogram(pair[1])
        for name, values in merged.data_per_streamline.items():
            assert isinstance(values, np.ndarray) and values.shape == (40, 1), name
        assert merged.data_per_streamline["bundle_id"].ravel().tolist()[:31:30] == [
            1,
            1,
        ]
        info = merged.get_maps_info().set_index("map_name")  # Summaries work on them
        assert info.loc["tract_id", ["min_value", "max_value"]].tolist() == [0, 1]
        assert info.loc["bundle_id", "num_values"] == 40

    def test_add_tractogram_keeps_the_colors_of_the_sources(self, pair):
        first, second = pair
        first.set_color("#ff0000")
        second.set_color("#0000ff")
        merged = first.add_tractogram(second)
        assert merged.colortables["tract_id"]["color_table"][:, :3].tolist() == [
            [255, 0, 0],
            [0, 0, 255],
        ]

    def test_add_tractogram_keeps_only_the_maps_in_common(self, pair):
        first, second = pair
        del second.data_per_streamline["bundle_id"]
        merged = first.add_tractogram(second)
        assert set(merged.data_per_streamline) == {"default", "length", "tract_id"}
        full = pair[0].add_tractogram(pair[0])
        assert "bundle_id" in full.data_per_streamline

    def test_add_tractogram_from_files_and_lists(self, tmp_path, pair):
        first, second = pair
        second.save_tractogram(str(tmp_path / "second.trk"))
        assert len(first.add_tractogram(str(tmp_path / "second.trk")).tracts) == 40
        assert len(first.add_tractogram(tmp_path / "second.trk").tracts) == 40
        assert len(first.add_tractogram([second, tmp_path / "second.trk"]).tracts) == 50

    def test_add_tractogram_errors(self, tmp_path, pair):
        first, _ = pair
        with pytest.raises(ValueError, match="cannot be empty"):
            first.add_tractogram([])
        with pytest.raises(TypeError, match="must be a Tractogram"):
            first.add_tractogram(5)
        with pytest.raises(TypeError, match="not a Tractogram"):
            first.add_tractogram([5])
        with pytest.raises(FileNotFoundError):
            first.add_tractogram(tmp_path / "missing.trk")

    def test_merge_tractograms(self, pair):
        merged = clttract.merge_tractograms(list(pair))
        assert len(merged.tracts) == 40
        ids = merged.data_per_streamline["tract_id"].ravel()
        assert ids.tolist() == [1] * 30 + [2] * 10
        table = merged.colortables["tract_id"]
        assert table["names"] == ["tract_0", "tract_1"]
        assert table["color_table"][:, 4].tolist() == [1, 2]

    def test_merge_three_tractograms_with_another_map_name(self, pair):
        merged = clttract.merge_tractograms(
            [pair[0], pair[1], pair[0]], map_name="source"
        )
        counts = np.unique(merged.data_per_streamline["source"], return_counts=True)
        assert counts[0].tolist() == [1, 2, 3] and counts[1].tolist() == [30, 10, 30]
        assert "source" in merged.colortables

    def test_merge_with_a_color_table(self, pair):
        table = {
            "names": ["left", "right"],
            "color_table": np.array(
                [[255, 0, 0, 1, 10], [0, 255, 0, 1, 20]], dtype=float
            ),
        }
        merged = clttract.merge_tractograms(list(pair), color_table=table)
        assert np.unique(merged.data_per_streamline["tract_id"]).tolist() == [10, 20]
        assert merged.colortables["tract_id"]["names"] == ["left", "right"]

    def test_merge_files(self, tmp_path, pair):
        pair[0].save_tractogram(str(tmp_path / "a.trk"))
        pair[1].save_tractogram(str(tmp_path / "b.trk"))
        merged = clttract.merge_tractograms(
            [str(tmp_path / "a.trk"), tmp_path / "b.trk"]
        )
        assert len(merged.tracts) == 40

    def test_merge_a_single_file(self, tmp_path, pair):
        pair[1].save_tractogram(str(tmp_path / "b.trk"))
        assert len(clttract.merge_tractograms([tmp_path / "b.trk"]).tracts) == 10

    def test_merge_nothing(self):
        assert clttract.merge_tractograms([]) is None

    def test_merge_a_single_tractogram_object(self, pair):
        merged = clttract.merge_tractograms([pair[0]])
        assert merged is pair[0] and len(merged.tracts) == 30

    def test_merge_errors(self, pair):
        first, second = pair
        with pytest.raises(TypeError, match="must be a list"):
            clttract.merge_tractograms(first)
        with pytest.raises(TypeError, match="Tractogran objects"):
            clttract.merge_tractograms([first, 3])
        with pytest.raises(TypeError, match="dictionary"):
            clttract.merge_tractograms([first, second], color_table=[1, 2])
        with pytest.raises(ValueError, match="must contain the keys"):
            clttract.merge_tractograms(
                [first, second], color_table={"names": ["a", "b"]}
            )
        wrong = {"names": ["only-one"], "color_table": np.zeros((1, 5))}
        with pytest.raises(ValueError, match="must match number of tractograms"):
            clttract.merge_tractograms([first, second], color_table=wrong)


####################################################################################################
# Clusters and centroids
####################################################################################################
class TestClusters:
    def test_quickbundles_finds_the_three_groups(self, clusters_tract):
        centroids, indexes = clusters_tract.compute_centroids(
            method="qb", thresholds=[10]
        )
        assert isinstance(centroids, ArraySequence) and len(centroids) == 3
        assert sorted(sorted(i) for i in indexes) == [
            list(range(5)),
            list(range(5, 10)),
            list(range(10, 15)),
        ]
        assert (
            clusters_tract.centroids is centroids
            and clusters_tract.centroids_indexes is indexes
        )

    def test_centroids_lie_in_the_middle_of_their_group(self, clusters_tract):
        centroids, indexes = clusters_tract.compute_centroids()  # Default threshold
        assert len(centroids) == 3
        for centroid, members in zip(centroids, indexes, strict=True):
            group = np.mean([clusters_tract.tracts[i][:, 1].mean() for i in members])
            assert centroid[:, 1].mean() == pytest.approx(group, abs=1.0)
            assert centroid[0, 0] == pytest.approx(0, abs=1.0) and centroid[
                -1, 0
            ] == pytest.approx(50, abs=1.0)

    def test_a_large_threshold_merges_the_groups(self, clusters_tract):
        centroids, _ = clusters_tract.compute_centroids(method="qb", thresholds=[1000])
        assert len(centroids) == 1

    def test_quickbundles_x(self, clusters_tract):
        centroids, indexes = clusters_tract.compute_centroids(
            method="qbx", thresholds=[40, 10]
        )
        assert len(centroids) == len(indexes) >= 3
        assert set().union(*map(set, indexes)) == set(
            range(15)
        )  # Every streamline is clustered

    def test_labels(self, clusters_tract):
        clusters_tract.compute_centroids(method="qb", thresholds=[10])
        ids = clusters_tract.label_streamlines_by_clusters()
        assert ids.shape == (15,) and sorted(np.unique(ids).tolist()) == [0, 1, 2]
        for (
            members
        ) in clusters_tract.centroids_indexes:  # Members of a cluster share a label
            assert len({ids[i] for i in members}) == 1
        assert clusters_tract.data_per_streamline["cluster_id"].shape == (15, 1)
        table = clusters_tract.colortables["cluster_id"]
        assert table["names"] == ["cluster_0", "cluster_1", "cluster_2"]
        assert table["color_table"][:, 4].tolist() == [0, 1, 2]

    def test_cluster_streamlines(self, clusters_tract):
        clusters_tract.compute_centroids(method="qb", thresholds=[10])
        members = clusters_tract.get_cluster_streamlines(1)
        expected = clusters_tract.centroids_indexes[1]
        assert isinstance(members, list) and len(members) == 5
        for streamline, index in zip(members, expected, strict=True):
            assert np.array_equal(streamline, clusters_tract.tracts[index])

    def test_centroids_as_a_tractogram(self, clusters_tract):
        clusters_tract.compute_centroids(method="qb", thresholds=[10])
        centroid_tract = clusters_tract.centroids_to_tractogram()
        assert (
            isinstance(centroid_tract, Tractogram) and len(centroid_tract.tracts) == 3
        )
        assert set(centroid_tract.data_per_streamline) == {"length", "cluster_id"}
        assert centroid_tract.data_per_streamline["cluster_id"].ravel().tolist() == [
            0,
            1,
            2,
        ]
        assert centroid_tract.header["nb_streamlines"] == 3
        assert centroid_tract.data_per_streamline["length"].ravel() == pytest.approx(
            [50] * 3, abs=1.0
        )
        assert len(clusters_tract.tracts) == 15  # The original is untouched

    def test_errors(self, clusters_tract):
        with pytest.raises(ValueError, match="Unknown clustering method"):
            clusters_tract.compute_centroids(method="nope")
        for thresholds in ([], 10):
            with pytest.raises(ValueError, match="non-empty list"):
                clusters_tract.compute_centroids(method="qbx", thresholds=thresholds)
        with pytest.raises(ValueError, match="compute_centroids"):
            clusters_tract.label_streamlines_by_clusters()
        with pytest.raises(ValueError, match="compute_centroids"):
            clusters_tract.get_cluster_streamlines(0)
        with pytest.raises(ValueError, match="compute_centroids"):
            clusters_tract.centroids_to_tractogram()
        clusters_tract.compute_centroids(method="qb", thresholds=[10])
        for cluster_id in (3, 99, -1):
            with pytest.raises(
                IndexError, match=f"Cluster ID {cluster_id} is out of range"
            ):
                clusters_tract.get_cluster_streamlines(cluster_id)


####################################################################################################
# Selecting streamlines with masks and splitting
####################################################################################################
class TestFilterByMask:
    def test_every_streamline_that_touches_the_mask(self, mask_tract, labels):
        selected = quiet_call(mask_tract.filter_by_mask, labels)
        assert isinstance(selected, Tractogram) and selected is not mask_tract
        assert len(selected.tracts) == 3  # S3 lies outside of the volume
        for kept, original in zip(
            selected.tracts, list(mask_tract.tracts)[:3], strict=True
        ):
            assert np.array_equal(kept, original)
        assert selected.data_per_streamline["tract_id"].tolist() == [1, 1, 1]
        assert selected.colortables["tract_id"]["names"] == ["front-back"]
        assert len(mask_tract.tracts) == 4  # The source keeps all its streamlines

    @pytest.mark.parametrize(
        "pairs, expected, name",
        [
            ({1: 2}, [0], "front_to_back"),  # Only S0 touches both regions
            ({1: None}, [0, 1], "front"),
            ({2: None}, [0, 2], "back"),
            ({(1, 2): None}, [0, 1, 2], "front-back"),
            ({1: [1, 2]}, [0, 1], "front_to_front-back"),  # The target includes 1
        ],
    )
    def test_region_pairs(self, mask_tract, labels, pairs, expected, name):
        selected = quiet_call(mask_tract.filter_by_mask, labels, pairs=pairs)
        assert len(selected.tracts) == len(expected)
        for kept, index in zip(selected.tracts, expected, strict=True):
            assert np.array_equal(kept, mask_tract.tracts[index])
        assert selected.colortables["tract_id"]["names"] == [name]

    def test_each_streamline_gets_the_first_pair_it_matches(self, mask_tract, labels):
        selected = quiet_call(
            mask_tract.filter_by_mask, labels, pairs={1: None, 2: None}
        )
        assert len(selected.tracts) == 3
        assert selected.data_per_streamline["tract_id"].tolist() == [
            1,
            1,
            2,
        ]  # S0 matches both
        table = selected.colortables["tract_id"]
        assert table["names"] == ["front", "back"] and table["color_table"][
            :, 4
        ].tolist() == [1, 2]

    def test_splitting_into_bundles(self, mask_tract, labels):
        selected, bundles, registry = quiet_call(
            mask_tract.filter_by_mask,
            labels,
            pairs={1: None, 2: None},
            split_bundles=True,
        )
        assert len(selected.tracts) == 3 and [len(b.tracts) for b in bundles] == [2, 2]
        assert registry == {
            1: {"source": [1], "target": None, "name": "front", "n_streamlines": 2},
            2: {"source": [2], "target": None, "name": "back", "n_streamlines": 2},
        }
        assert [b.colortables["tract_id"]["names"] for b in bundles] == [
            ["front"],
            ["back"],
        ]
        assert [b.data_per_streamline["tract_id"].tolist() for b in bundles] == [
            [1, 1],
            [2, 2],
        ]
        # A streamline that matches several pairs belongs to every one of those bundles
        assert np.array_equal(bundles[0].tracts[0], bundles[1].tracts[0])

    def test_small_bundles_are_dropped(self, mask_tract, labels):
        _, bundles, registry = quiet_call(
            mask_tract.filter_by_mask,
            labels,
            pairs={1: None, 2: None},
            split_bundles=True,
            min_streamlines=3,
        )
        assert bundles == [] and registry == {}

    def test_nothing_matches(self, mask_tract, labels):
        selected = quiet_call(mask_tract.filter_by_mask, labels, pairs={1: 99})
        assert len(selected.tracts) == 0 and len(mask_tract.tracts) == 4
        empty, bundles, registry = quiet_call(
            mask_tract.filter_by_mask, labels, pairs={1: 99}, split_bundles=True
        )
        assert len(empty.tracts) == 0 and bundles == [] and registry == {}

    def test_in_place(self, mask_tract, labels):
        result = quiet_call(mask_tract.filter_by_mask, labels, inplace=True)
        assert result is mask_tract and len(mask_tract.tracts) == 3
        assert mask_tract.header["nb_streamlines"] == 3
        assert len(mask_tract.data_per_point["interp_mask"]) == 3

    def test_custom_map_name(self, mask_tract, labels):
        selected = quiet_call(mask_tract.filter_by_mask, labels, map_name="pairs")
        assert (
            "pairs" in selected.data_per_streamline and "pairs" in selected.colortables
        )
        assert "tract_id" not in selected.data_per_streamline

    @pytest.mark.parametrize("as_str", [True, False])
    def test_mask_from_a_file(self, tmp_path, mask_tract, labels, as_str):
        labels.save_parcellation(tmp_path / "mask_dseg.nii.gz")
        mask = tmp_path / "mask_dseg.nii.gz"
        selected = quiet_call(mask_tract.filter_by_mask, str(mask) if as_str else mask)
        assert len(selected.tracts) == 3
        assert selected.colortables["tract_id"]["names"] == ["front-back"]

    def test_array_mask_follows_the_grid_of_the_header(self, labels):
        """A tractogram whose voxel grid is shifted by 10 mm along x."""
        shift = np.eye(4)
        shift[0, 3] = 10.0
        streamlines = [
            line(1 + 10, 8 + 10, 2, 2, n=8),
            line(1 + 10, 3 + 10, 2, 2, n=8),
            line(5 + 10, 8 + 10, 2, 2, n=8),
        ]
        tract = hand_made(
            streamlines, header={"nb_streamlines": 3, "voxel_to_rasmm": shift}
        )
        selected = quiet_call(
            tract.filter_by_mask, np.asarray(labels.get_data()), pairs={1: 2}
        )
        assert len(selected.tracts) == 1 and np.array_equal(
            selected.tracts[0], tract.tracts[0]
        )

    def test_unsupported_mask(self, mask_tract):
        with pytest.raises(TypeError, match="Unsupported mask type"):
            mask_tract.filter_by_mask(5)

    def test_array_mask_is_in_the_space_of_the_tractogram(self, mask_tract):
        everything = np.ones(
            (10, 10, 10), dtype=np.int32
        )  # Covers the whole identity grid
        selected = quiet_call(mask_tract.filter_by_mask, everything)
        assert len(selected.tracts) == 3


class TestSplit:
    def test_one_tractogram_per_value(self, sim):
        parts = sim.split("bundle_id")
        assert len(parts) == 3 and all(isinstance(p, Tractogram) for p in parts)
        assert [len(p.tracts) for p in parts] == [10, 10, 10]
        assert [
            np.unique(p.data_per_streamline["bundle_id"]).tolist() for p in parts
        ] == [[1], [2], [3]]
        assert [p.header["nb_streamlines"] for p in parts] == [10, 10, 10]
        assert len(sim.tracts) == 30  # The source is untouched

    def test_streamlines_follow_their_value(self, sim):
        parts = sim.split("bundle_id")
        for i, part in enumerate(parts):
            for streamline, original in zip(
                part.tracts, list(sim.tracts)[10 * i : 10 * (i + 1)], strict=True
            ):
                assert np.array_equal(streamline, original)

    def test_each_part_gets_its_color_table_entry(self, sim):
        parts = sim.split("bundle_id")
        assert [p.colortables["bundle_id"]["names"] for p in parts] == [
            ["bundle_00"],
            ["bundle_01"],
            ["bundle_02"],
        ]
        assert [
            p.colortables["bundle_id"]["color_table"][:, 4].tolist() for p in parts
        ] == [[1], [2], [3]]

    def test_values_are_split_in_ascending_order(self, sim):
        sim.data_per_streamline["group"] = np.array([[3], [1], [2]] * 10)
        parts = sim.split("group")
        assert [np.unique(p.data_per_streamline["group"]).item() for p in parts] == [
            1,
            2,
            3,
        ]

    def test_unknown_map(self, sim):
        with pytest.raises(KeyError, match="not found in data_per_streamline"):
            sim.split("nope")
        with pytest.raises(KeyError):
            sim.split()  # The default map, 'tract_id', does not exist here


####################################################################################################
# Information display
####################################################################################################
class TestGetInfo:
    def test_overview(self, lengths_tract, capsys):
        lengths_tract.get_info()
        out = capsys.readouterr().out
        lines = out.splitlines()
        assert "TRACTOGRAM EXPLORATION" in out
        assert len({len(line) for line in lines}) == 1  # The box is aligned
        for text in (
            "Name: hand",
            "Streamlines: 5",
            "AFFINE TRANSFORMATION MATRIX",
            "SCALAR DATA PER POINT (1 map)",
            "SCALAR DATA PER STREAMLINE (3 maps)",
            "COLORTABLES (1 available)",
        ):
            assert text in out
        assert "ramp" in out and "score" in out and "length" in out

    def test_statistics_of_the_maps(self, lengths_tract, capsys):
        lengths_tract.get_info()
        out = capsys.readouterr().out
        score = next(line for line in out.splitlines() if "score" in line)
        assert "10.00" in score and "50.00" in score

    def test_simulated_tractogram(self, sim, capsys):
        sim.get_info()
        out = capsys.readouterr().out
        assert "Name: simulated" in out and "Streamlines: 30" in out
        assert "COLORTABLES (2 available)" in out and "bundle_id" in out

    def test_empty_tractogram(self, capsys):
        Tractogram().get_info()
        out = capsys.readouterr().out
        assert "Streamlines: 0" in out
        assert "Not available" in out and "No colortables available" in out
        assert "No scalar data available" in out
