"""Tests for clabtoolkit.freesurfertools.

Nothing here needs FreeSurfer: the annotations are simulated or built by hand, the
FreeSurfer subjects are small synthetic folders, and the FreeSurfer commands are
replaced by fakes that record the command line they receive. A few tests that read
the real fsaverage subject only run when FreeSurfer is installed.
"""

import json
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

import clabtoolkit.freesurfertools as cltfree
import clabtoolkit.surfacetools as cltsurf

AnnotParcellation = cltfree.AnnotParcellation
FreeSurferSubject = cltfree.FreeSurferSubject

# Packed RGB codes (R + 256 G + 65536 B) of red, green and blue
RED, GREEN, BLUE = 255, 255 * 256, 255 * 65536

FREESURFER_HOME = os.environ.get("FREESURFER_HOME")
FSAVERAGE = Path(FREESURFER_HOME or "/nonexistent") / "subjects" / "fsaverage"
needs_fsaverage = pytest.mark.skipif(
    not (FSAVERAGE / "label" / "lh.aparc.annot").is_file(),
    reason="FreeSurfer's fsaverage subject is not installed",
)


####################################################################################################
# Helpers and fixtures
####################################################################################################
def grid_mesh(n):
    """n x n grid of vertices on z=0, two triangles per cell: vertex = row * n + col."""
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    coords = np.c_[xs.ravel(), ys.ravel(), np.zeros(n * n)].astype(np.float32)
    faces = []
    for row in range(n - 1):
        for col in range(n - 1):
            a = row * n + col
            faces += [[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]
    return coords, np.array(faces, dtype=np.int32)


def write_grid_surface(path, n):
    coords, faces = grid_mesh(n)
    nib.freesurfer.write_geometry(str(path), coords, faces)
    return path


def write_cortex_label(path, vertices):
    with open(path, "w") as f:
        f.write("#!ascii label\n")
        f.write(f"{len(vertices)}\n")
        for v in vertices:
            f.write(f"{int(v)} 0.000000 0.000000 0.000000 0.500000\n")
    return path


def hand_annot(codes, names=("regA", "regB", "regC"), hemi="lh"):
    """Annotation of red, green and blue regions with the given vertex codes."""
    annot = AnnotParcellation()
    regtable = np.array(
        [
            [255, 0, 0, 0, RED],
            [0, 255, 0, 0, GREEN],
            [0, 0, 255, 0, BLUE],
        ],
        dtype=np.int32,
    )
    annot.create_from_data(
        codes=np.asarray(codes, dtype=np.int32),
        regtable=regtable,
        regnames=list(names),
        annot_id="hand",
        hemi=hemi,
    )
    return annot


@pytest.fixture(autouse=True)
def _quiet():
    """Silence the warnings and keep the environment of each test separate."""
    saved = {k: os.environ.get(k) for k in ("SUBJECTS_DIR",)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def sphere():
    return cltsurf.Surface.simulate_surface(seed=1, shape="sphere", maps=None)


@pytest.fixture
def annot(sphere):
    return AnnotParcellation.simulate_annotation(sphere, n_regions=5, seed=1)


@pytest.fixture
def annot_file(tmp_path, annot):
    path = tmp_path / "lh.simulated.annot"
    annot.save_annotation(str(path))
    return path


####################################################################################################
class TestSimulateAnnotation:
    def test_default_surface_and_regions(self):
        annot = AnnotParcellation.simulate_annotation()
        assert annot.is_loaded()
        assert len(annot.regnames) == 10
        assert annot.hemi == "lh"
        assert annot.id == "simulated"

    def test_every_vertex_gets_a_region(self, sphere, annot):
        assert annot.codes.shape == (sphere.mesh.n_points,)
        assert set(np.unique(annot.codes)) == set(annot.regtable[:, 4])
        assert len(annot.regnames) == annot.regtable.shape[0] == 5

    def test_codes_are_packed_rgb_values(self, annot):
        ok, errors = cltfree.verify_packed_rgb_values(annot.regtable)
        assert ok, errors

    def test_regions_are_connected(self, sphere, annot):
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        edges = sphere.get_edges()
        n = sphere.mesh.n_points
        for code in annot.regtable[:, 4]:
            inside = annot.codes == code
            keep = inside[edges[:, 0]] & inside[edges[:, 1]]
            graph = coo_matrix(
                (np.ones(keep.sum()), (edges[keep, 0], edges[keep, 1])), shape=(n, n)
            )
            n_components, labels = connected_components(graph, directed=False)
            assert len(set(labels[inside])) == 1, code

    def test_seed_makes_it_reproducible(self, sphere):
        first = AnnotParcellation.simulate_annotation(sphere, n_regions=6, seed=3)
        second = AnnotParcellation.simulate_annotation(sphere, n_regions=6, seed=3)
        np.testing.assert_array_equal(first.codes, second.codes)
        np.testing.assert_array_equal(first.regtable, second.regtable)

    def test_different_seeds_give_different_layouts(self, sphere):
        first = AnnotParcellation.simulate_annotation(sphere, n_regions=6, seed=3)
        second = AnnotParcellation.simulate_annotation(sphere, n_regions=6, seed=4)
        assert not np.array_equal(first.codes, second.codes)

    def test_region_names_and_colors(self, sphere):
        annot = AnnotParcellation.simulate_annotation(
            sphere,
            n_regions=3,
            region_names=["frontal", "parietal", "occipital"],
            colors=["#e41a1c", "#377eb8", "#4daf4a"],
            seed=0,
        )
        assert annot.regnames == ["frontal", "parietal", "occipital"]
        np.testing.assert_array_equal(
            annot.regtable[:, :3], [[228, 26, 28], [55, 126, 184], [77, 175, 74]]
        )

    def test_rgb_array_of_colors(self, sphere):
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        annot = AnnotParcellation.simulate_annotation(
            sphere, n_regions=3, colors=colors
        )
        np.testing.assert_array_equal(annot.regtable[:, 4], [RED, GREEN, BLUE])

    def test_default_names(self, sphere):
        annot = AnnotParcellation.simulate_annotation(sphere, n_regions=3)
        assert annot.regnames == ["auto-roi-001", "auto-roi-002", "auto-roi-003"]

    def test_names_of_the_wrong_length_are_replaced_with_a_warning(self, sphere):
        with pytest.warns(UserWarning, match="2 region names were supplied"):
            annot = AnnotParcellation.simulate_annotation(
                sphere, n_regions=3, region_names=["a", "b"]
            )
        assert annot.regnames == ["auto-roi-001", "auto-roi-002", "auto-roi-003"]

    def test_colors_of_the_wrong_length_are_replaced_with_a_warning(self, sphere):
        with pytest.warns(UserWarning, match="2 colors were supplied"):
            annot = AnnotParcellation.simulate_annotation(
                sphere, n_regions=3, colors=["#ff0000", "#00ff00"]
            )
        assert annot.regtable.shape[0] == 3

    def test_repeated_colors_are_rejected(self, sphere):
        with pytest.raises(ValueError, match="must all be different"):
            AnnotParcellation.simulate_annotation(
                sphere, n_regions=2, colors=["#ff0000", "#ff0000"]
            )

    @pytest.mark.parametrize("n_regions", [0, -1, 10**6])
    def test_number_of_regions_is_limited(self, sphere, n_regions):
        with pytest.raises(ValueError, match="n_regions must be between 1"):
            AnnotParcellation.simulate_annotation(sphere, n_regions=n_regions)

    def test_single_region(self, sphere):
        annot = AnnotParcellation.simulate_annotation(sphere, n_regions=1)
        assert len(np.unique(annot.codes)) == 1

    def test_hemisphere_option(self, sphere):
        assert AnnotParcellation.simulate_annotation(sphere, hemi="rh").hemi == "rh"

    def test_hemisphere_of_the_surface(self):
        surface = cltsurf.Surface.simulate_surface(seed=1, maps=None, hemi="rh")
        assert AnnotParcellation.simulate_annotation(surface).hemi == "rh"

    def test_annotation_id(self, sphere):
        annot = AnnotParcellation.simulate_annotation(sphere, annot_id="mine")
        assert annot.id == "mine"

    def test_surface_file(self, tmp_path, sphere):
        path = tmp_path / "lh.white"
        sphere.save_surface(str(path))
        annot = AnnotParcellation.simulate_annotation(path, n_regions=4, seed=1)
        assert annot.codes.shape == (sphere.mesh.n_points,)

    def test_surface_is_not_modified(self, sphere):
        points = np.asarray(sphere.mesh.points).copy()
        AnnotParcellation.simulate_annotation(sphere, n_regions=4)
        np.testing.assert_array_equal(sphere.mesh.points, points)

    def test_invalid_surface(self):
        with pytest.raises(ValueError, match="surface must be"):
            AnnotParcellation.simulate_annotation(surface=12)

    def test_surface_without_geometry(self):
        with pytest.raises(ValueError, match="does not contain any geometry"):
            AnnotParcellation.simulate_annotation(surface=cltsurf.Surface())


####################################################################################################
class TestConstruction:
    def test_empty_object(self):
        annot = AnnotParcellation(annot_id="test")
        assert not annot.is_loaded()
        assert annot.id == "test"
        for attribute in ("filename", "path", "name", "hemi", "codes", "regtable"):
            assert getattr(annot, attribute) is None, attribute

    def test_from_file(self, annot, annot_file):
        loaded = AnnotParcellation(parc_file=str(annot_file))
        assert loaded.is_loaded()
        assert loaded.filename == str(annot_file)
        assert loaded.path == str(annot_file.parent)
        assert loaded.name == "lh.simulated.annot"
        assert loaded.hemi == "lh"
        np.testing.assert_array_equal(loaded.codes, annot.codes)
        np.testing.assert_array_equal(loaded.regtable, annot.regtable)
        assert loaded.regnames == annot.regnames

    def test_from_path_object(self, annot_file):
        loaded = AnnotParcellation(parc_file=annot_file)
        assert loaded.filename == str(annot_file)

    def test_annotation_id_defaults_to_the_file_name(self, annot_file):
        assert AnnotParcellation(annot_file).id == "lh.simulated"

    def test_annotation_id_can_be_given(self, annot_file):
        assert AnnotParcellation(annot_file, annot_id="Sim").id == "Sim"

    @pytest.mark.parametrize("prefix, hemi", [("lh", "lh"), ("rh", "rh")])
    def test_hemisphere_comes_from_the_file_name(self, tmp_path, annot, prefix, hemi):
        path = tmp_path / f"{prefix}.aparc.annot"
        annot.save_annotation(str(path))
        assert AnnotParcellation(path).hemi == hemi

    def test_hemisphere_from_a_bids_name(self, tmp_path, annot):
        path = tmp_path / "sub-01_hemi-R_desc-x_dseg.annot"
        annot.save_annotation(str(path))
        assert AnnotParcellation(path).hemi == "rh"

    def test_file_without_hemisphere(self, tmp_path, annot):
        path = tmp_path / "atlas.annot"
        annot.save_annotation(str(path))
        with pytest.warns(UserWarning, match="hemisphere could not be extracted"):
            loaded = AnnotParcellation(path)
        assert loaded.hemi is None

    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            AnnotParcellation(tmp_path / "lh.missing.annot")

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "lh.atlas.txt"
        path.write_text("not an annotation")
        with pytest.raises(ValueError, match="Unsupported"):
            AnnotParcellation(path)

    def test_load_into_an_existing_object(self, annot, annot_file):
        loaded = AnnotParcellation()
        loaded.load_from_file(annot_file, annot_id="again")
        assert loaded.id == "again"
        np.testing.assert_array_equal(loaded.codes, annot.codes)

    def test_table_rows_without_vertices_are_dropped(self, tmp_path):
        codes = np.array([0, 0, 1, 1, 0])  # indices into the table
        regtable = np.array(
            [
                [255, 0, 0, 0, RED],
                [0, 255, 0, 0, GREEN],
                [0, 0, 255, 0, BLUE],
            ],
            dtype=np.int32,
        )
        path = tmp_path / "lh.partial.annot"
        nib.freesurfer.write_annot(
            str(path), codes, regtable, [b"regA", b"regB", b"regC"]
        )
        loaded = AnnotParcellation(path)
        assert loaded.regnames == ["regA", "regB"]
        assert loaded.regtable.shape == (2, 5)
        assert set(loaded.regtable[:, 4]) == {RED, GREEN}

    def test_gifti_files_are_converted(self, tmp_path, annot):
        gii = tmp_path / "lh.simulated.label.gii"
        AnnotParcellation.annot2gii(annot, gii_file=str(gii))
        loaded = AnnotParcellation(gii)
        assert loaded.hemi == "lh"
        assert loaded.regnames == annot.regnames
        np.testing.assert_array_equal(loaded.regtable[:, :3], annot.regtable[:, :3])
        np.testing.assert_array_equal(loaded.codes, annot.codes)

    def test_loading_a_gifti_file_keeps_the_annotation_next_to_it(
        self, tmp_path, annot
    ):
        gii = tmp_path / "lh.same.gii"
        AnnotParcellation.annot2gii(annot, gii_file=str(gii))
        precious = tmp_path / "lh.same.annot"
        precious.write_text("an annotation that is not ours")
        AnnotParcellation(gii)
        assert precious.read_text() == "an annotation that is not ours"
        assert not (tmp_path / "lh.same.annot.tmp").exists()

    def test_gcs_files_are_converted(self, tmp_path, annot, monkeypatch):
        gcs = tmp_path / "lh.classifier.gcs"
        gcs.write_text("classifier")
        precious = tmp_path / "lh.classifier.annot"
        precious.write_text("an annotation that is not ours")

        def fake_gcs2annot(gcs_file, annot_file=None, **kwargs):
            annot.save_annotation(str(annot_file))
            return annot_file

        monkeypatch.setattr(
            AnnotParcellation, "gcs2annot", staticmethod(fake_gcs2annot)
        )
        loaded = AnnotParcellation(gcs)
        np.testing.assert_array_equal(loaded.codes, annot.codes)
        assert precious.read_text() == "an annotation that is not ours"


####################################################################################################
class TestCreateFromData:
    def test_basic(self):
        annot = hand_annot([RED, RED, GREEN, GREEN, BLUE])
        assert annot.is_loaded()
        assert annot.id == "hand"
        assert annot.hemi == "lh"
        assert annot.regnames == ["regA", "regB", "regC"]
        assert annot.filename is None

    def test_rows_without_vertices_are_dropped(self):
        annot = hand_annot([RED, RED, BLUE])
        assert annot.regnames == ["regA", "regC"]
        assert annot.regtable[:, 4].tolist() == [RED, BLUE]

    def test_colors_between_0_and_1_are_scaled(self):
        annot = AnnotParcellation()
        regtable = np.array([[1.0, 0.0, 0.0, 0, RED], [0.0, 1.0, 0.0, 0, GREEN]])
        annot.create_from_data(
            np.array([RED, GREEN]), regtable, ["a", "b"], annot_id="x"
        )
        np.testing.assert_array_equal(annot.regtable[:, :3], [[255, 0, 0], [0, 255, 0]])

    def test_filename_sets_the_path_and_the_name(self):
        annot = AnnotParcellation()
        annot.create_from_data(
            np.array([RED]),
            np.array([[255, 0, 0, 0, RED]]),
            ["a"],
            annot_id="x",
            hemi="rh",
            filename="/some/dir/rh.x.annot",
        )
        assert annot.hemi == "rh"
        assert annot.path == "/some/dir"
        assert annot.name == "rh.x.annot"


####################################################################################################
class TestGetInfo:
    def test_info_of_a_simulated_annotation(self, annot):
        info = annot.get_info(verbose=False)
        assert info["id"] == "simulated"
        assert info["hemi"] == "lh"
        assert info["n_vertices"] == annot.codes.size
        assert info["n_regions_table"] == 5
        assert info["n_unique_labels"] == 5
        assert info["labels_not_in_table"] == []
        assert info["regions_not_in_vertices"] == []
        assert info["name"] is None
        assert info["filename"] is None

    def test_info_of_a_loaded_annotation(self, annot_file):
        info = AnnotParcellation(annot_file).get_info(verbose=False)
        assert info["name"] == "lh.simulated.annot"
        assert info["filename"] == str(annot_file)

    def test_labels_missing_from_the_table(self):
        annot = hand_annot([RED, GREEN, 12345, 0])
        info = annot.get_info(verbose=False)
        assert info["labels_not_in_table"] == [0, 12345]
        assert info["n_unique_labels"] == 4

    def test_regions_missing_from_the_vertices(self):
        annot = hand_annot([RED, GREEN, GREEN])
        annot.regtable = np.vstack([annot.regtable, [0, 0, 1, 0, 256]])
        annot.regnames = annot.regnames + ["ghost"]
        info = annot.get_info(verbose=False)
        assert info["regions_not_in_vertices"] == ["ghost"]

    def test_empty_object(self):
        info = AnnotParcellation().get_info(verbose=False)
        assert info["n_vertices"] is None
        assert info["n_regions_table"] is None

    def test_nothing_is_printed_when_not_verbose(self, annot, capsys):
        annot.get_info(verbose=False)
        assert capsys.readouterr().out == ""

    def test_report_is_printed_by_default(self, annot_file, capsys):
        AnnotParcellation(annot_file).get_info()
        out = capsys.readouterr().out
        assert "ANNOTATION PARCELLATION INFO" in out
        assert "lh.simulated.annot" in out
        assert "Total vertices" in out
        assert "Labels not in table     :    0" in out

    def test_report_lists_the_inconsistencies(self, capsys):
        hand_annot([RED, 12345]).get_info()
        out = capsys.readouterr().out
        assert "Labels not in table     :    1" in out
        assert "12345" in out

    def test_report_of_an_empty_object(self, capsys):
        AnnotParcellation().get_info()
        out = capsys.readouterr().out
        assert "No vertex data loaded" in out
        assert "No region table loaded" in out

    def test_long_paths_are_truncated(self, tmp_path, annot, capsys):
        deep = tmp_path / ("a" * 40) / ("b" * 40)
        deep.mkdir(parents=True)
        path = deep / "lh.deep.annot"
        annot.save_annotation(str(path))
        AnnotParcellation(path).get_info()
        out = capsys.readouterr().out
        assert "..." in out
        for line in out.splitlines():
            assert len(line) == 66, line


####################################################################################################
class TestSaveAnnotation:
    def test_round_trip(self, tmp_path, annot):
        path = tmp_path / "lh.copy.annot"
        annot.save_annotation(str(path))
        loaded = AnnotParcellation(path)
        np.testing.assert_array_equal(loaded.codes, annot.codes)
        np.testing.assert_array_equal(loaded.regtable, annot.regtable)
        assert loaded.regnames == annot.regnames

    def test_round_trip_of_a_hand_made_annotation(self, tmp_path):
        annot = hand_annot([RED, RED, GREEN, BLUE, GREEN])
        path = tmp_path / "lh.hand.annot"
        annot.save_annotation(str(path))
        codes, regtable, names = nib.freesurfer.read_annot(str(path), orig_ids=True)
        assert codes.tolist() == [RED, RED, GREEN, BLUE, GREEN]
        assert regtable[:, 4].tolist() == [RED, GREEN, BLUE]
        assert names == [b"regA", b"regB", b"regC"]

    def test_missing_folders_are_created(self, tmp_path, annot):
        path = tmp_path / "new" / "folder" / "lh.copy.annot"
        annot.save_annotation(str(path))
        assert path.is_file()

    def test_existing_file_is_replaced(self, tmp_path, annot):
        path = tmp_path / "lh.copy.annot"
        path.write_text("old")
        annot.save_annotation(str(path), overwrite=True)
        assert AnnotParcellation(path).is_loaded()

    def test_existing_file_is_kept_without_overwrite(self, tmp_path, annot):
        path = tmp_path / "lh.copy.annot"
        path.write_text("old")
        with pytest.raises(ValueError, match="already exists"):
            annot.save_annotation(str(path), overwrite=False)
        assert path.read_text() == "old"

    def test_default_output_is_the_loaded_file(self, annot_file):
        loaded = AnnotParcellation(annot_file)
        loaded.codes[:3] = loaded.regtable[1, 4]
        loaded.save_annotation()
        again = AnnotParcellation(annot_file)
        np.testing.assert_array_equal(again.codes, loaded.codes)

    def test_file_in_the_current_folder(self, tmp_path, annot, monkeypatch):
        monkeypatch.chdir(tmp_path)
        annot.save_annotation("lh.here.annot")
        assert (tmp_path / "lh.here.annot").is_file()

    def test_no_output_and_no_loaded_file(self, annot):
        with pytest.raises(ValueError, match="out_file"):
            annot.save_annotation()

    def test_unlabeled_vertices_are_saved_as_minus_one(self, tmp_path):
        annot = hand_annot([RED, 0, GREEN])
        path = tmp_path / "lh.gap.annot"
        annot.save_annotation(str(path))
        codes, _, _ = nib.freesurfer.read_annot(str(path))
        assert codes.tolist() == [0, -1, 1]


####################################################################################################
class TestRegionInformation:
    @pytest.fixture
    def named(self):
        return hand_annot([RED, RED, GREEN, BLUE, BLUE, BLUE])

    def test_info_from_name(self, named):
        info = named.get_info_from_name("regB")
        assert list(info) == ["regB"]
        assert info["regB"]["id"] == GREEN
        assert info["regB"]["n_vertices"] == 1
        np.testing.assert_array_equal(info["regB"]["color"], [0, 255, 0])

    def test_info_from_id(self, named):
        info = named.get_info_from_id(BLUE)
        assert info["regC"]["n_vertices"] == 3
        assert info["regC"]["id"] == BLUE

    def test_info_from_id_accepts_numpy_integers(self, named):
        assert "regA" in named.get_info_from_id(np.int64(RED))

    def test_unknown_name_and_id(self, named, capsys):
        assert named.get_info_from_name("nothing") is None
        assert named.get_info_from_id(12345) is None
        out = capsys.readouterr().out
        assert "'nothing' not found" in out
        assert "'12345' not found" in out

    def test_wrong_types(self, named):
        with pytest.raises(TypeError, match="region_name must be a str"):
            named.get_info_from_name(5)
        with pytest.raises(TypeError, match="region_id must be an int"):
            named.get_info_from_id("regA")

    def test_regions_by_name(self, named):
        infos = named.get_regions_info(["regA", "regC"])
        assert [list(i)[0] for i in infos] == ["regA", "regC"]
        assert infos[0]["regA"]["n_vertices"] == 2

    def test_regions_by_id(self, named):
        infos = named.get_regions_info([GREEN, BLUE])
        assert [list(i)[0] for i in infos] == ["regB", "regC"]

    def test_single_name_and_single_id(self, named):
        assert named.get_regions_info("regA")[0]["regA"]["id"] == RED
        assert named.get_regions_info(GREEN)[0]["regB"]["id"] == GREEN

    def test_arrays(self, named):
        assert len(named.get_regions_info(np.array([RED, BLUE]))) == 2
        assert len(named.get_regions_info(np.array(["regA", "regB", "regC"]))) == 3

    def test_all_regions_by_default(self, named):
        infos = named.get_regions_info()
        assert [list(i)[0] for i in infos] == ["regA", "regB", "regC"]

    def test_unknown_regions_are_skipped(self, named):
        infos = named.get_regions_info(["regA", "nothing"])
        assert len(infos) == 1
        assert named.get_regions_info([12345]) == []

    def test_unsupported_items_are_skipped(self, named, capsys):
        infos = named.get_regions_info(["regA", 2.5])
        assert len(infos) == 1
        assert "unsupported type" in capsys.readouterr().out

    def test_data_frame(self, named):
        df = named.get_regions_info(return_df=True)
        assert df.columns.tolist() == ["name", "id", "color", "n_vertices"]
        assert df["name"].tolist() == ["regA", "regB", "regC"]
        assert df["n_vertices"].tolist() == [2, 1, 3]
        assert df["id"].tolist() == [RED, GREEN, BLUE]
        assert df["color"].tolist() == ["#ff0000", "#00ff00", "#0000ff"]

    def test_empty_data_frame(self, named):
        df = named.get_regions_info(["nothing"], return_df=True)
        assert df.empty
        assert df.columns.tolist() == ["name", "id", "color", "n_vertices"]

    def test_wrong_types_of_the_arguments(self, named):
        with pytest.raises(TypeError, match="Unsupported type for region_list"):
            named.get_regions_info(2.5)
        with pytest.raises(TypeError, match="return_df must be a bool"):
            named.get_regions_info(return_df="yes")

    def test_object_without_regions(self, capsys):
        assert AnnotParcellation().get_regions_info() is None
        assert "No regions available" in capsys.readouterr().out


####################################################################################################
class TestFillParcellation:
    @pytest.fixture
    def setup(self, tmp_path):
        """4 x 4 grid: columns 0-1 are red and columns 2-3 are green; vertices 5 and
        10 are unlabeled (their neighbours are mostly red and mostly green)."""
        surf = write_grid_surface(tmp_path / "lh.white", 4)
        codes = np.array([RED, RED, GREEN, GREEN] * 4, dtype=np.int32)
        codes[5] = 0
        codes[10] = 0
        annot = hand_annot(codes, names=("left", "right", "unused"))
        cortex = write_cortex_label(tmp_path / "lh.cortex.label", range(16))
        return annot, str(surf), str(cortex), tmp_path

    def test_unlabeled_vertices_get_the_most_frequent_neighbour_label(self, setup):
        annot, surf, cortex, tmp_path = setup
        out = tmp_path / "lh.filled.annot"
        path, labels, table, names = annot.fill_parcellation(cortex, surf, str(out))
        assert path == str(out)
        assert labels[5] == RED  # neighbours 1, 2, 4, 6, 8, 9: four red, two green
        assert labels[10] == GREEN  # neighbours 6, 7, 9, 11, 13, 14: four green
        assert (labels != 0).all()

    def test_filled_annotation_is_saved(self, setup):
        annot, surf, cortex, tmp_path = setup
        out = tmp_path / "filled" / "lh.filled.annot"
        annot.fill_parcellation(cortex, surf, str(out))
        loaded = AnnotParcellation(out)
        assert loaded.codes[5] == RED
        assert loaded.codes[10] == GREEN

    def test_vertices_outside_the_cortex_label_stay_unlabeled(self, setup):
        annot, surf, _, tmp_path = setup
        cortex = write_cortex_label(
            tmp_path / "small.label", [v for v in range(16) if v != 5]
        )
        _, labels, _, _ = annot.fill_parcellation(cortex, surf)
        assert labels[5] == 0
        assert labels[10] == GREEN

    def test_without_output_file_the_annotation_name_is_returned(self, setup):
        annot, surf, cortex, tmp_path = setup
        annot.filename = str(tmp_path / "lh.hand.annot")
        path, _, _, _ = annot.fill_parcellation(cortex, surf)
        assert path == str(tmp_path / "lh.hand.annot")

    def test_labeled_vertices_do_not_change(self, setup):
        annot, surf, cortex, _ = setup
        before = annot.codes.copy()
        _, labels, _, _ = annot.fill_parcellation(cortex, surf)
        keep = before != 0
        np.testing.assert_array_equal(labels[keep], before[keep])

    def test_table_and_names_are_returned(self, setup):
        annot, surf, cortex, _ = setup
        _, _, table, names = annot.fill_parcellation(cortex, surf)
        assert names == ["left", "right"]
        assert table.shape == (2, 5)

    def test_missing_files(self, setup):
        annot, surf, cortex, tmp_path = setup
        with pytest.raises(ValueError, match="Surface file not found"):
            annot.fill_parcellation(cortex, str(tmp_path / "missing"))
        with pytest.raises(ValueError, match="Cortex label file not found"):
            annot.fill_parcellation(str(tmp_path / "missing.label"), surf)


####################################################################################################
class TestCorrectVertexwiseParcellation:
    N = 6

    @pytest.fixture
    def setup(self, tmp_path):
        """6 x 6 grid. Green: the two first rows (12 vertices) plus a pair (21, 22) and
        a lone vertex (32) inside the red region."""
        surf = write_grid_surface(tmp_path / "lh.white", self.N)
        codes = np.full(self.N * self.N, RED, dtype=np.int32)
        codes[:12] = GREEN
        codes[[21, 22]] = GREEN
        codes[32] = GREEN
        annot = hand_annot(codes, names=("red", "green", "blue"))
        cortex = write_cortex_label(
            tmp_path / "lh.cortex.label", range(self.N * self.N)
        )
        return annot, str(surf), str(cortex)

    def test_lone_vertices_are_always_removed(self, setup):
        annot, surf, _ = setup
        corrected = annot.correct_vertexwise_parcellation(surf, min_cluster_size=2)
        assert corrected[32] == 0
        assert corrected[21] == GREEN and corrected[22] == GREEN

    def test_small_clusters_are_removed(self, setup):
        annot, surf, _ = setup
        corrected = annot.correct_vertexwise_parcellation(surf, min_cluster_size=5)
        assert (corrected[[21, 22, 32]] == 0).all()
        assert (corrected[:12] == GREEN).all()

    def test_without_a_size_every_non_largest_cluster_is_removed(self, setup):
        annot, surf, _ = setup
        corrected = annot.correct_vertexwise_parcellation(surf)
        assert (corrected[[21, 22, 32]] == 0).all()
        assert (corrected[:12] == GREEN).all()

    def test_the_remaining_vertices_do_not_change(self, setup):
        annot, surf, _ = setup
        before = annot.codes.copy()
        corrected = annot.correct_vertexwise_parcellation(surf, min_cluster_size=5)
        removed = np.zeros(before.size, dtype=bool)
        removed[[21, 22, 32]] = True
        np.testing.assert_array_equal(corrected[~removed], before[~removed])

    def test_regions_below_the_size_are_removed(self, setup):
        annot, surf, _ = setup
        corrected = annot.correct_vertexwise_parcellation(surf, min_cluster_size=13)
        assert (corrected[:12] == 0).all()  # the green region has only 12 vertices

    def test_the_cortex_label_refills_the_removed_vertices(self, setup):
        annot, surf, cortex = setup
        corrected = annot.correct_vertexwise_parcellation(
            surf, cortex_file=cortex, min_cluster_size=5
        )
        assert (corrected != 0).all()
        assert (corrected[[21, 22, 32]] == RED).all()

    def test_the_annotation_is_updated(self, setup):
        annot, surf, _ = setup
        corrected = annot.correct_vertexwise_parcellation(surf, min_cluster_size=5)
        np.testing.assert_array_equal(annot.codes, corrected)

    def test_surface_objects_are_accepted(self, setup):
        annot, surf, _ = setup
        surface = cltsurf.Surface(surf)
        corrected = annot.correct_vertexwise_parcellation(surface, min_cluster_size=5)
        assert corrected[32] == 0

    def test_path_objects_are_accepted(self, setup):
        annot, surf, _ = setup
        corrected = annot.correct_vertexwise_parcellation(
            Path(surf), min_cluster_size=5
        )
        assert corrected[32] == 0


####################################################################################################
class TestExportToTsv:
    @pytest.fixture
    def named(self):
        return hand_annot([RED, GREEN, BLUE], names=("alpha", "beta", "gamma"))

    def test_full_table(self, named):
        df, path = named.export_to_tsv()
        assert path is None
        assert df.columns.tolist() == [
            "index",
            "hemi",
            "annotid",
            "parcid",
            "name",
            "color",
        ]
        assert df["index"].tolist() == [1, 2, 3]  # 0 is not a region
        assert df["hemi"].tolist() == ["L", "L", "L"]
        assert df["annotid"].tolist() == [RED, GREEN, BLUE]
        assert df["parcid"].tolist() == [1001, 1002, 1003]
        assert df["name"].tolist() == ["alpha", "beta", "gamma"]
        assert df["color"].tolist() == ["#ff0000", "#00ff00", "#0000ff"]

    def test_prefix_offset_and_hemisphere(self, named):
        df, _ = named.export_to_tsv(prefix2add="rh_", reg_offset=2000, hemi="R")
        assert df["name"].tolist() == ["rh_alpha", "rh_beta", "rh_gamma"]
        assert df["parcid"].tolist() == [2001, 2002, 2003]
        assert set(df["hemi"]) == {"R"}

    @pytest.mark.parametrize("position", [0, 1, 2])
    def test_unknown_region_has_the_index_0(self, position):
        names = ["alpha", "beta", "gamma"]
        names[position] = "unknown"
        annot = hand_annot([RED, GREEN, BLUE], names=tuple(names))
        df, _ = annot.export_to_tsv()
        assert df["index"][position] == 0
        assert df["parcid"][position] == 1000
        # The other regions are numbered from 1 in the order of the table
        assert df["index"].drop(position).tolist() == [1, 2]
        assert df["parcid"].drop(position).tolist() == [1001, 1002]

    def test_region_with_the_code_0_has_the_index_0(self):
        annot = AnnotParcellation()
        regtable = np.array(
            [[255, 0, 0, 0, RED], [0, 255, 0, 0, GREEN], [0, 0, 0, 0, 0]],
            dtype=np.int32,
        )
        annot.create_from_data(
            codes=np.array([RED, GREEN, 0], dtype=np.int32),
            regtable=regtable,
            regnames=["alpha", "beta", "gamma"],
            annot_id="hand",
            hemi="lh",
        )
        df, _ = annot.export_to_tsv()
        assert df["name"].tolist() == ["alpha", "beta", "gamma"]
        assert df["index"].tolist() == [1, 2, 0]

    def test_templateflow_format(self, named):
        df, _ = named.export_to_tsv(
            prefix2add="ctx-lh-", hemi="L", output_format="templateflow"
        )
        assert df.columns.tolist() == ["index", "hemi", "name", "color"]
        assert df["index"].tolist() == [RED, GREEN, BLUE]
        assert df["name"].tolist() == ["ctx-lh-alpha", "ctx-lh-beta", "ctx-lh-gamma"]

    def test_file_is_saved(self, tmp_path, named):
        out = tmp_path / "tables" / "lh.tsv"
        df, path = named.export_to_tsv(tsv_file=str(out))
        assert path == str(out)
        saved = pd.read_csv(out, sep="\t")
        assert saved["name"].tolist() == df["name"].tolist()
        assert saved["parcid"].tolist() == df["parcid"].tolist()

    def test_path_objects_are_accepted(self, tmp_path, named):
        out = tmp_path / "lh.tsv"
        _, path = named.export_to_tsv(tsv_file=out)
        assert path == str(out)
        assert out.is_file()

    def test_file_in_the_current_folder(self, tmp_path, named, monkeypatch):
        monkeypatch.chdir(tmp_path)
        named.export_to_tsv(tsv_file="here.tsv")
        assert (tmp_path / "here.tsv").is_file()

    def test_invalid_hemisphere(self, named):
        with pytest.raises(ValueError, match="Invalid value for hemi"):
            named.export_to_tsv(hemi="lh")

    def test_simulated_annotation(self, annot):
        df, _ = annot.export_to_tsv()
        assert len(df) == 5
        assert df["color"].str.match(r"^#[0-9a-f]{6}$").all()


####################################################################################################
class TestMapValues:
    @pytest.fixture
    def named(self):
        return hand_annot([RED, RED, GREEN, BLUE, BLUE, 0])

    def test_one_dimensional_array(self, named):
        result = named.map_values(np.array([1.5, 2.5, 3.5]))
        assert result.shape == (6, 1)
        assert result[:, 0].tolist() == [1.5, 1.5, 2.5, 3.5, 3.5, 0.0]

    def test_two_dimensional_array(self, named):
        values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        result = named.map_values(values)
        assert result.shape == (6, 2)
        np.testing.assert_array_equal(result[2], [2.0, 20.0])
        np.testing.assert_array_equal(result[5], [0.0, 0.0])

    def test_data_frame_keeps_the_column_names(self, named):
        df = pd.DataFrame({"thickness": [1.0, 2.0, 3.0], "area": [10.0, 20.0, 30.0]})
        result = named.map_values(df)
        assert isinstance(result, pd.DataFrame)
        assert result.columns.tolist() == ["thickness", "area"]
        assert result["area"].tolist() == [10.0, 10.0, 20.0, 30.0, 30.0, 0.0]

    def test_text_file(self, tmp_path, named):
        path = tmp_path / "values.txt"
        np.savetxt(path, [1.0, 2.0, 3.0])
        result = named.map_values(str(path))
        assert isinstance(result, np.ndarray)
        assert result[:, 0].tolist() == [1.0, 1.0, 2.0, 3.0, 3.0, 0.0]

    def test_path_object(self, tmp_path, named):
        path = tmp_path / "values.txt"
        np.savetxt(path, [1.0, 2.0, 3.0])
        result = named.map_values(path)
        assert result[:, 0].tolist() == [1.0, 1.0, 2.0, 3.0, 3.0, 0.0]

    def test_csv_file_with_headers(self, tmp_path, named):
        path = tmp_path / "values.csv"
        pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}).to_csv(
            path, index=False
        )
        result = named.map_values(str(path), is_dataframe=True)
        assert result.columns.tolist() == ["a", "b"]
        assert result["b"].tolist() == [4.0, 4.0, 5.0, 6.0, 6.0, 0.0]

    def test_values_have_a_single_precision(self, named):
        assert named.map_values(np.array([1.0, 2.0, 3.0])).dtype == np.float32

    def test_wrong_number_of_regions(self, tmp_path, named):
        with pytest.raises(ValueError, match="number of rows"):
            named.map_values(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="number of rows"):
            named.map_values(pd.DataFrame({"a": [1.0, 2.0]}))
        path = tmp_path / "short.txt"
        np.savetxt(path, [1.0, 2.0])
        with pytest.raises(ValueError, match="number of rows"):
            named.map_values(str(path))

    def test_data_frame_must_be_numeric(self, named):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        with pytest.raises(ValueError, match="should be numeric"):
            named.map_values(df)

    def test_missing_file(self, tmp_path, named):
        with pytest.raises(ValueError, match="does not exist"):
            named.map_values(str(tmp_path / "missing.txt"))

    def test_unsupported_type(self, named):
        with pytest.raises(ValueError, match="should be a pandas dataframe"):
            named.map_values([1.0, 2.0, 3.0])


####################################################################################################
class TestCompleteTable:
    def test_complete_annotation_is_not_changed(self, annot):
        before_codes, before_table = annot.codes.copy(), annot.regtable.copy()
        result = annot.complete_table()
        assert result is annot
        np.testing.assert_array_equal(annot.codes, before_codes)
        np.testing.assert_array_equal(annot.regtable, before_table)

    def test_one_missing_code_gets_the_light_gray(self):
        annot = hand_annot([RED, GREEN, 0, 0])
        annot.complete_table()
        assert annot.regtable.shape == (3, 5)
        np.testing.assert_array_equal(annot.regtable[-1], [250, 250, 250, 0, 16448250])
        assert annot.regnames[-1] == "unknown_1"
        assert annot.codes.tolist() == [RED, GREEN, 16448250, 16448250]

    def test_several_missing_codes_get_different_colors(self):
        annot = hand_annot([RED, 0, -1, 77])
        annot.complete_table(seed=0)
        assert annot.regtable.shape == (4, 5)
        new_codes = annot.regtable[1:, 4]
        assert len(set(new_codes.tolist())) == len(new_codes) == 3
        assert not set(new_codes.tolist()) & {RED, GREEN, BLUE}
        # Vertices keep the grouping of the codes they had
        assert annot.codes[1] != annot.codes[2] != annot.codes[3]
        ok, errors = cltfree.verify_packed_rgb_values(annot.regtable)
        assert ok, errors

    def test_the_table_is_consistent_afterwards(self):
        annot = hand_annot([RED, 0, -1, 77, RED])
        annot.complete_table()
        info = annot.get_info(verbose=False)
        assert info["labels_not_in_table"] == []

    def test_names_have_a_prefix(self):
        annot = hand_annot([RED, 0, -1])
        annot.complete_table(prefix="ctx-lh", seed=1)
        assert annot.regnames[1:] == ["ctx-lh-unknown_1", "ctx-lh-unknown_2"]

    def test_names_do_not_clash_with_existing_ones(self):
        annot = hand_annot([RED, 0], names=("unknown_1", "b", "c"))
        annot.complete_table()
        assert annot.regnames[-1] == "unknown_2"

    def test_color_clashes_are_avoided(self):
        # The gray (250, 250, 250) is already in the table
        annot = hand_annot([RED, 0])
        annot.regtable = np.vstack([annot.regtable, [250, 250, 250, 0, 16448250]])
        annot.regnames = annot.regnames + ["gray"]
        annot.codes = np.array([RED, 0, 16448250], dtype=np.int32)
        annot.complete_table(seed=0)
        new_code = annot.regtable[-1, 4]
        assert new_code not in (RED, GREEN, BLUE, 16448250)
        assert annot.codes[1] == new_code
        assert annot.codes[2] == 16448250

    def test_not_in_place(self):
        annot = hand_annot([RED, 0])
        completed = annot.complete_table(inplace=False)
        assert completed is not annot
        assert annot.codes.tolist() == [RED, 0]
        assert completed.regtable.shape[0] == 2

    def test_seed_makes_it_reproducible(self):
        first = hand_annot([RED, 0, -1, 77])
        second = hand_annot([RED, 0, -1, 77])
        first.complete_table(seed=5)
        second.complete_table(seed=5)
        np.testing.assert_array_equal(first.regtable, second.regtable)

    def test_verbose_reports_the_added_codes(self, capsys):
        annot = hand_annot([RED, -1])
        annot.complete_table(prefix="x", verbose=True)
        out = capsys.readouterr().out
        assert "Added 1 missing code(s)" in out
        assert "x-unknown_1" in out

    def test_verbose_reports_a_complete_table(self, annot, capsys):
        annot.complete_table(verbose=True)
        assert "already in the color table" in capsys.readouterr().out

    def test_object_without_data(self):
        with pytest.raises(ValueError, match="does not contain any data"):
            AnnotParcellation().complete_table()


####################################################################################################
class TestGiftiConversion:
    def test_round_trip_keeps_everything(self, tmp_path, annot):
        gii = AnnotParcellation.annot2gii(
            annot, gii_file=str(tmp_path / "lh.x.label.gii")
        )
        back = AnnotParcellation.gii2annot(
            gii, annot_file=str(tmp_path / "lh.back.annot")
        )
        codes, regtable, names = nib.freesurfer.read_annot(back, orig_ids=True)
        np.testing.assert_array_equal(codes, annot.codes)
        np.testing.assert_array_equal(regtable, annot.regtable)
        assert [n.decode() for n in names] == annot.regnames

    def test_label_table_of_the_gifti_file(self, tmp_path):
        annot = hand_annot([RED, GREEN, BLUE, RED], names=("alpha", "beta", "gamma"))
        gii = AnnotParcellation.annot2gii(
            annot, gii_file=str(tmp_path / "lh.x.label.gii")
        )
        img = nib.load(gii)
        labels = img.labeltable.labels
        assert [lbl.label for lbl in labels] == ["alpha", "beta", "gamma"]
        assert [lbl.key for lbl in labels] == [0, 1, 2]
        assert (labels[0].red, labels[0].green, labels[0].blue) == (1.0, 0.0, 0.0)
        assert labels[0].alpha == 1.0  # GIFTI stores opacity, FreeSurfer transparency
        assert img.darrays[0].data.tolist() == [0, 1, 2, 0]

    def test_transparency_becomes_opacity_and_back(self, tmp_path):
        annot = hand_annot([RED, GREEN])
        annot.regtable[0, 3] = 51  # 20 % transparent
        gii = AnnotParcellation.annot2gii(
            annot, gii_file=str(tmp_path / "lh.x.label.gii")
        )
        assert nib.load(gii).labeltable.labels[0].alpha == pytest.approx(0.8)
        back = AnnotParcellation.gii2annot(gii, annot_file=str(tmp_path / "lh.x.annot"))
        _, regtable, _ = nib.freesurfer.read_annot(back, orig_ids=True)
        assert regtable[:, 3].tolist() == [51, 0]

    def test_hemisphere_is_tagged(self, tmp_path):
        for hemi, structure in (("lh", "CortexLeft"), ("rh", "CortexRight")):
            annot = hand_annot([RED, GREEN], hemi=hemi)
            gii = AnnotParcellation.annot2gii(
                annot, gii_file=str(tmp_path / f"{hemi}.x.label.gii")
            )
            meta = nib.load(gii).darrays[0].meta
            assert meta["AnatomicalStructurePrimary"] == structure

    def test_unknown_hemisphere_is_not_tagged(self, tmp_path):
        annot = hand_annot([RED, GREEN], hemi=None)
        gii = AnnotParcellation.annot2gii(annot, gii_file=str(tmp_path / "x.label.gii"))
        assert "AnatomicalStructurePrimary" not in nib.load(gii).darrays[0].meta

    def test_input_can_be_a_file(self, tmp_path, annot_file):
        gii = AnnotParcellation.annot2gii(str(annot_file))
        assert gii == str(tmp_path / "lh.simulated.label.gii")
        assert Path(gii).is_file()

    def test_default_output_is_next_to_the_annotation(self, tmp_path, annot_file):
        gii = AnnotParcellation.annot2gii(AnnotParcellation(annot_file))
        assert gii == str(annot_file.with_suffix("")) + ".label.gii"

    def test_object_without_a_file_needs_an_output(self, annot):
        with pytest.raises(ValueError, match="Please provide gii_file"):
            AnnotParcellation.annot2gii(annot)

    def test_vertices_outside_the_table_become_unknown(self, tmp_path):
        annot = hand_annot([RED, 0, GREEN, -1])
        with pytest.warns(UserWarning, match="2 vertices have codes"):
            gii = AnnotParcellation.annot2gii(
                annot, gii_file=str(tmp_path / "x.label.gii")
            )
        img = nib.load(gii)
        assert [lbl.label for lbl in img.labeltable.labels][-1] == "unknown"
        assert img.darrays[0].data.tolist() == [0, 2, 1, 2]

    def test_existing_unknown_region_is_reused(self, tmp_path):
        annot = hand_annot([RED, 0, GREEN], names=("unknown", "b", "c"))
        with pytest.warns(UserWarning):
            gii = AnnotParcellation.annot2gii(
                annot, gii_file=str(tmp_path / "x.label.gii")
            )
        img = nib.load(gii)
        assert len(img.labeltable.labels) == 2
        assert img.darrays[0].data.tolist() == [0, 0, 1]

    def test_completing_the_table_gives_the_vertices_a_color(self, tmp_path):
        annot = hand_annot([RED, 0, GREEN])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gii = AnnotParcellation.annot2gii(
                annot, gii_file=str(tmp_path / "x.label.gii"), complete_table=True
            )
        assert len(nib.load(gii).labeltable.labels) == 3
        assert annot.codes.tolist() == [RED, 0, GREEN]  # the object is not changed

    def test_existing_file(self, tmp_path, annot):
        gii = tmp_path / "x.label.gii"
        gii.write_text("old")
        with pytest.raises(FileExistsError):
            AnnotParcellation.annot2gii(annot, gii_file=str(gii), overwrite=False)
        AnnotParcellation.annot2gii(annot, gii_file=str(gii), overwrite=True)
        assert nib.load(str(gii)).darrays

    def test_output_folders_are_created(self, tmp_path, annot):
        gii = tmp_path / "a" / "b" / "x.label.gii"
        AnnotParcellation.annot2gii(annot, gii_file=gii)
        assert gii.is_file()

    def test_annot2gii_errors(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            AnnotParcellation.annot2gii(tmp_path / "missing.annot")
        with pytest.raises(TypeError, match="annot_file must be"):
            AnnotParcellation.annot2gii(12)
        with pytest.raises(ValueError, match="does not contain any data"):
            AnnotParcellation.annot2gii(AnnotParcellation())

    def test_default_output_of_gii2annot(self, tmp_path, annot):
        gii = AnnotParcellation.annot2gii(
            annot, gii_file=str(tmp_path / "lh.x.label.gii")
        )
        assert AnnotParcellation.gii2annot(gii) == str(tmp_path / "lh.x.annot")
        plain = tmp_path / "lh.plain.gii"
        shutil.copy(gii, plain)
        assert AnnotParcellation.gii2annot(str(plain)) == str(
            tmp_path / "lh.plain.annot"
        )

    def test_gii2annot_errors(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            AnnotParcellation.gii2annot(str(tmp_path / "missing.gii"))

        no_arrays = tmp_path / "empty.gii"
        nib.save(nib.gifti.GiftiImage(), str(no_arrays))
        with pytest.raises(ValueError, match="no data arrays"):
            AnnotParcellation.gii2annot(str(no_arrays))

        no_labels = tmp_path / "nolabels.gii"
        darray = nib.gifti.GiftiDataArray(np.zeros(3, dtype=np.int32), datatype="int32")
        nib.save(nib.gifti.GiftiImage(darrays=[darray]), str(no_labels))
        with pytest.raises(ValueError, match="LabelTable is empty"):
            AnnotParcellation.gii2annot(str(no_labels))


####################################################################################################
class FakeRun:
    """Stands for subprocess.run: records the commands and can create files."""

    def __init__(self, returncode=0, stdout="", creates=None):
        self.calls = []
        self.returncode = returncode
        self.stdout = stdout
        self.creates = creates

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append((list(cmd), kwargs))
        if self.creates is not None and self.returncode == 0:
            for path in self.creates(cmd):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text("created")
        return subprocess.CompletedProcess(
            cmd, self.returncode, stdout=self.stdout, stderr=""
        )

    @property
    def commands(self):
        return [call[0] for call in self.calls]


@pytest.fixture
def fake_run(monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(cltfree.subprocess, "run", run)
    return run


class TestGcsConversion:
    def test_gcs2annot_command(self, tmp_path, fake_run):
        gcs = tmp_path / "lh.aparc.gcs"
        gcs.write_text("classifier")
        subjects = tmp_path / "subjects"
        out = tmp_path / "out" / "lh.aparc.annot"
        fake_run.creates = lambda cmd: [cmd[-1]]
        result = AnnotParcellation.gcs2annot(
            gcs, annot_file=out, freesurfer_dir=subjects, ref_id="fsaverage"
        )
        assert result == str(out)
        assert fake_run.commands == [
            [
                "mris_ca_label",
                "-l",
                str(subjects / "fsaverage" / "label" / "lh.cortex.label"),
                "-aseg",
                str(subjects / "fsaverage" / "mri" / "aseg.mgz"),
                "fsaverage",
                "lh",
                str(subjects / "fsaverage" / "surf" / "lh.sphere.reg"),
                str(gcs),
                str(out),
            ]
        ]
        assert os.environ["SUBJECTS_DIR"] == str(subjects)

    def test_default_output_is_next_to_the_classifier(self, tmp_path, fake_run):
        gcs = tmp_path / "rh.atlas.gcs"
        gcs.write_text("classifier")
        fake_run.creates = lambda cmd: [cmd[-1]]
        result = AnnotParcellation.gcs2annot(gcs, freesurfer_dir=tmp_path / "subjects")
        assert result == str(tmp_path / "rh.atlas.annot")
        assert fake_run.commands[0][6] == "rh"

    def test_subjects_directory_from_the_environment(
        self, tmp_path, fake_run, monkeypatch
    ):
        gcs = tmp_path / "lh.atlas.gcs"
        gcs.write_text("classifier")
        monkeypatch.setenv("SUBJECTS_DIR", str(tmp_path / "from_env"))
        fake_run.creates = lambda cmd: [cmd[-1]]
        AnnotParcellation.gcs2annot(gcs)
        assert (tmp_path / "from_env").is_dir()
        assert str(tmp_path / "from_env" / "fsaverage") in fake_run.commands[0][2]

    def test_container_command(self, tmp_path, fake_run, monkeypatch):
        def fake_container(args, tech, image):
            return ["docker", "run", image, *args]

        monkeypatch.setattr(
            cltfree.cltmisc, "generate_container_command", fake_container
        )
        gcs = tmp_path / "lh.atlas.gcs"
        gcs.write_text("classifier")
        fake_run.creates = lambda cmd: [cmd[-1]]
        AnnotParcellation.gcs2annot(
            gcs,
            freesurfer_dir=tmp_path / "subjects",
            cont_tech="docker",
            cont_image="freesurfer/freesurfer:7.2.0",
        )
        assert fake_run.commands[0][0] == "docker"
        assert "freesurfer/freesurfer:7.2.0" in fake_run.commands[0]

    def test_failure_is_reported(self, tmp_path, fake_run):
        gcs = tmp_path / "lh.atlas.gcs"
        gcs.write_text("classifier")
        fake_run.returncode = 1
        with pytest.raises(RuntimeError, match="mris_ca_label failed"):
            AnnotParcellation.gcs2annot(gcs, freesurfer_dir=tmp_path / "subjects")

    def test_missing_classifier(self, tmp_path):
        with pytest.raises(ValueError, match="gcs file does not exist"):
            AnnotParcellation.gcs2annot(tmp_path / "lh.missing.gcs")

    def test_no_subjects_directory(self, tmp_path, monkeypatch):
        gcs = tmp_path / "lh.atlas.gcs"
        gcs.write_text("classifier")
        monkeypatch.delenv("SUBJECTS_DIR", raising=False)
        with pytest.raises(ValueError, match="FreeSurfer directory must be set"):
            AnnotParcellation.gcs2annot(gcs)


class TestAnnot2Gcs:
    @pytest.fixture
    def trained(self, tmp_path, annot):
        """Annotation saved to disk and a subject with a sphere.reg file."""
        path = tmp_path / "lh.manual.annot"
        annot.save_annotation(str(path))
        loaded = AnnotParcellation(path)
        subjects = tmp_path / "subjects"
        (subjects / "sub-01" / "surf").mkdir(parents=True)
        (subjects / "sub-01" / "surf" / "lh.sphere.reg").write_text("sphere")
        return loaded, subjects

    def test_command_and_color_table(self, tmp_path, trained, fake_run):
        loaded, subjects = trained
        out = tmp_path / "atlases" / "lh.manual.gcs"
        tables = []

        def create(cmd):
            ctab = cmd[cmd.index("-t") + 1]
            tables.append(Path(ctab).read_text())
            return [cmd[-1]]

        fake_run.creates = create
        result = loaded.annot2gcs(
            gcs_file=out, freesurfer_dir=subjects, fssubj_id="sub-01"
        )
        assert result == str(out)
        cmd = fake_run.commands[0]
        assert cmd[:3] == ["mris_ca_train", "-n", "2"]
        assert (
            cmd[-6:]
            == [
                "lh",
                "sphere.reg",
                str(loaded.filename),
                "sub-01",
                str(out),
            ]
            or cmd[-5:]
            == ["lh", "sphere.reg", str(loaded.filename), "sub-01", str(out)][-5:]
        )
        lines = tables[0].strip().splitlines()
        assert len(lines) == 5
        assert lines[0].split()[:2] == ["1", loaded.regnames[0]]
        assert fake_run.calls[0][1]["env"]["SUBJECTS_DIR"] == str(subjects)

    def test_session_environment_is_not_changed(
        self, tmp_path, trained, fake_run, monkeypatch
    ):
        loaded, subjects = trained
        monkeypatch.delenv("SUBJECTS_DIR", raising=False)
        fake_run.creates = lambda cmd: [cmd[-1]]
        loaded.annot2gcs(freesurfer_dir=subjects, fssubj_id="sub-01")
        assert "SUBJECTS_DIR" not in os.environ

    def test_color_table_is_removed(self, tmp_path, trained, fake_run):
        loaded, subjects = trained
        fake_run.creates = lambda cmd: [cmd[-1]]
        loaded.annot2gcs(freesurfer_dir=subjects, fssubj_id="sub-01")
        assert not list(tmp_path.glob("*.ctab"))

    def test_default_output_is_next_to_the_annotation(
        self, tmp_path, trained, fake_run
    ):
        loaded, subjects = trained
        fake_run.creates = lambda cmd: [cmd[-1]]
        result = loaded.annot2gcs(freesurfer_dir=subjects, fssubj_id="sub-01")
        assert result == str(tmp_path / "lh.manual.gcs")

    def test_failure_is_reported(self, trained, fake_run):
        loaded, subjects = trained
        fake_run.returncode = 1
        fake_run.stdout = "mris_ca_train: boom"
        with pytest.raises(RuntimeError, match="(?s)mris_ca_train failed.*boom"):
            loaded.annot2gcs(freesurfer_dir=subjects, fssubj_id="sub-01")

    def test_missing_output_is_reported(self, trained, fake_run):
        loaded, subjects = trained  # the command succeeds but writes nothing
        with pytest.raises(RuntimeError, match="mris_ca_train failed"):
            loaded.annot2gcs(freesurfer_dir=subjects, fssubj_id="sub-01")

    def test_validation(self, tmp_path, trained, monkeypatch):
        loaded, subjects = trained
        monkeypatch.delenv("SUBJECTS_DIR", raising=False)
        with pytest.raises(ValueError, match="subjects directory must be passed"):
            loaded.annot2gcs(fssubj_id="sub-01")
        with pytest.raises(ValueError, match="valid subject ID"):
            loaded.annot2gcs(freesurfer_dir=subjects)
        with pytest.raises(ValueError, match="subject directory does not exist"):
            loaded.annot2gcs(freesurfer_dir=subjects, fssubj_id="sub-99")
        with pytest.raises(ValueError, match="sphere.reg file does not exist"):
            loaded.annot2gcs(freesurfer_dir=subjects, fssubj_id="sub-01", hemi="rh")
        loaded.hemi = None
        with pytest.raises(ValueError, match="hemisphere could not be determined"):
            loaded.annot2gcs(freesurfer_dir=subjects, fssubj_id="sub-01")

    def test_annotation_must_be_on_disk(self, trained):
        _, subjects = trained
        memory = hand_annot([RED, GREEN])
        with pytest.raises(ValueError, match="must exist on disk"):
            memory.annot2gcs(freesurfer_dir=subjects, fssubj_id="sub-01")


####################################################################################################
DESIKAN = [
    "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus",
    "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal",
    "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal", "lingual",
    "medialorbitofrontal", "middletemporal", "parahippocampal", "paracentral",
    "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine",
    "postcentral", "posteriorcingulate", "precentral", "precuneus",
    "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal",
    "superiorparietal", "superiortemporal", "supramarginal", "frontalpole",
    "temporalpole", "transversetemporal", "insula",
]  # fmt: skip


@pytest.fixture
def desikan_annot(sphere):
    return AnnotParcellation.simulate_annotation(
        sphere, n_regions=34, region_names=DESIKAN, seed=3, annot_id="aparc"
    )


class TestGroupIntoLobes:
    FRONTAL = {
        "caudalmiddlefrontal", "lateralorbitofrontal", "medialorbitofrontal",
        "paracentral", "parsopercularis", "parsorbitalis", "parstriangularis",
        "precentral", "rostralmiddlefrontal", "superiorfrontal", "frontalpole",
        "caudalanteriorcingulate", "rostralanteriorcingulate",
    }  # fmt: skip

    def test_lobes_of_the_default_scheme(self, desikan_annot):
        lobar = desikan_annot.group_into_lobes()
        assert lobar.regnames == [
            "frontal",
            "parietal",
            "temporal",
            "occipital",
            "insula",
        ]
        assert lobar.hemi == "lh"
        assert lobar.id == "aparc_lobes"

    def test_every_vertex_goes_to_its_lobe(self, desikan_annot):
        lobar = desikan_annot.group_into_lobes()
        frontal_code = lobar.regtable[lobar.regnames.index("frontal"), 4]
        regions = desikan_annot.regnames
        in_frontal = np.isin(
            desikan_annot.codes,
            [
                desikan_annot.regtable[i, 4]
                for i, n in enumerate(regions)
                if n in self.FRONTAL
            ],
        )
        np.testing.assert_array_equal(lobar.codes == frontal_code, in_frontal)
        assert (lobar.codes != 0).all()

    def test_lobe_colors_come_from_the_scheme(self, desikan_annot):
        lobar = desikan_annot.group_into_lobes()
        assert lobar.regtable[0, :3].tolist() == [12, 73, 153]  # #0c4999
        ok, errors = cltfree.verify_packed_rgb_values(lobar.regtable)
        assert ok, errors

    def test_the_original_is_not_changed_by_default(self, desikan_annot):
        names = list(desikan_annot.regnames)
        codes = desikan_annot.codes.copy()
        desikan_annot.group_into_lobes()
        assert desikan_annot.regnames == names
        np.testing.assert_array_equal(desikan_annot.codes, codes)

    def test_in_place(self, desikan_annot):
        result = desikan_annot.group_into_lobes(inplace=True)
        assert result is desikan_annot
        assert len(desikan_annot.regnames) == 5

    def test_other_grouping(self, desikan_annot):
        lobar = desikan_annot.group_into_lobes(grouping="desikan_w_cingulate")
        assert "cingulate" in lobar.regnames
        assert len(lobar.regnames) == 6

    def test_prefix_of_the_names(self, desikan_annot):
        lobar = desikan_annot.group_into_lobes(ctxprefix="ctx-lh-")
        assert lobar.regnames[0] == "ctx-lh-frontal"

    def test_unlabeled_vertices_get_the_unknown_color(self, desikan_annot):
        desikan_annot.regnames = [
            n if n != "insula" else "not-a-region" for n in desikan_annot.regnames
        ]
        lobar = desikan_annot.group_into_lobes()
        assert lobar.regnames[0] == "unknown"
        assert lobar.regtable[0, :3].tolist() == [250, 250, 250]
        assert "insula" not in lobar.regnames
        insula_vertices = desikan_annot.codes == desikan_annot.regtable[33, 4]
        assert (lobar.codes[insula_vertices] == lobar.regtable[0, 4]).all()

    def test_custom_unknown_color(self, desikan_annot):
        desikan_annot.regnames[33] = "not-a-region"
        lobar = desikan_annot.group_into_lobes(unknown_color="#101010")
        assert lobar.regtable[0, :3].tolist() == [16, 16, 16]

    def test_saved_file(self, tmp_path, desikan_annot):
        out = tmp_path / "lobes" / "lh.lobes.annot"
        lobar = desikan_annot.group_into_lobes(out_annot=out)
        assert out.is_file()
        assert lobar.filename == str(out)
        loaded = AnnotParcellation(out)
        assert loaded.regnames == lobar.regnames
        np.testing.assert_array_equal(loaded.codes, lobar.codes)

    def test_existing_file_needs_overwrite(self, tmp_path, desikan_annot):
        out = tmp_path / "lh.lobes.annot"
        out.write_text("old")
        with pytest.raises(FileExistsError):
            desikan_annot.group_into_lobes(out_annot=out)
        assert out.read_text() == "old"
        desikan_annot.group_into_lobes(out_annot=out, overwrite=True)
        assert AnnotParcellation(out).is_loaded()

    def test_unsaved_result_has_no_file(self, desikan_annot):
        lobar = desikan_annot.group_into_lobes()
        assert lobar.filename is None
        assert lobar.path is None
        assert lobar.name is None

    def test_custom_json(self, tmp_path, desikan_annot):
        config = tmp_path / "lobes.json"
        config.write_text(
            json.dumps(
                {
                    "lobes": {
                        "front": ["precentral", "postcentral"],
                        "back": ["cuneus"],
                    },
                    "colors": {"front": "#ff0000", "back": "#0000ff"},
                }
            )
        )
        lobar = desikan_annot.group_into_lobes(lobes_json=config)
        assert lobar.regnames == ["unknown", "front", "back"]

    def test_missing_colors_are_generated(self, tmp_path, desikan_annot):
        config = tmp_path / "lobes.json"
        config.write_text(json.dumps({"lobes": {"a": ["precentral"], "b": ["cuneus"]}}))
        lobar = desikan_annot.group_into_lobes(lobes_json=config)
        assert len(set(lobar.regtable[:, 4].tolist())) == len(lobar.regnames)

    def test_region_in_two_lobes(self, tmp_path, desikan_annot):
        config = tmp_path / "lobes.json"
        config.write_text(
            json.dumps(
                {
                    "lobes": {"first": ["precentral"], "second": ["precentral"]},
                    "colors": {"first": "#ff0000", "second": "#00ff00"},
                }
            )
        )
        with pytest.warns(UserWarning, match="assigned to both 'first' and 'second'"):
            lobar = desikan_annot.group_into_lobes(lobes_json=config)
        assert lobar.regnames == ["unknown", "second"] or "second" in lobar.regnames
        assert "first" not in lobar.regnames

    def test_repeated_lobe_colors(self, tmp_path, desikan_annot):
        config = tmp_path / "lobes.json"
        config.write_text(
            json.dumps(
                {
                    "lobes": {"a": ["precentral"], "b": ["cuneus"]},
                    "colors": {"a": "#ff0000", "b": "#ff0000"},
                }
            )
        )
        with pytest.raises(ValueError, match="already used"):
            desikan_annot.group_into_lobes(lobes_json=config)

    def test_unknown_grouping(self, desikan_annot):
        with pytest.raises(ValueError, match="Grouping 'nothing' not found"):
            desikan_annot.group_into_lobes(grouping="nothing")

    @needs_fsaverage
    def test_fsaverage(self):
        parc = AnnotParcellation(FSAVERAGE / "label" / "lh.aparc.annot")
        lobar = parc.group_into_lobes()
        assert lobar.regnames == [
            "unknown", "frontal", "parietal", "temporal", "occipital", "insula",
        ]  # fmt: skip
        assert lobar.codes.size == 163842


@needs_fsaverage
class TestFsaverage:
    def test_desikan_annotation(self):
        parc = AnnotParcellation(FSAVERAGE / "label" / "lh.aparc.annot")
        assert parc.hemi == "lh"
        assert parc.codes.size == 163842
        assert "precentral" in parc.regnames
        info = parc.get_info(verbose=False)
        assert info["labels_not_in_table"] == [0]  # the medial wall is unlabeled

    def test_regions_of_the_annotation(self):
        parc = AnnotParcellation(FSAVERAGE / "label" / "lh.aparc.annot")
        info = parc.get_info_from_name("bankssts")
        assert info["bankssts"]["id"] == 2647065
        assert info["bankssts"]["color"].tolist() == [25, 100, 40]

    def test_gifti_round_trip(self, tmp_path):
        parc = AnnotParcellation(FSAVERAGE / "label" / "lh.aparc.annot")
        gii = AnnotParcellation.annot2gii(
            parc, gii_file=str(tmp_path / "lh.aparc.label.gii")
        )
        back = AnnotParcellation(gii)
        # Unlabeled vertices (code 0) go to the region called 'unknown'
        unknown = parc.regtable[parc.regnames.index("unknown"), 4]
        expected = np.where(parc.codes == 0, unknown, parc.codes)
        np.testing.assert_array_equal(back.codes, expected)
        assert back.regnames == parc.regnames


####################################################################################################
# FreeSurfer subjects
####################################################################################################
TALAIRACH_LTA = """# transform file transforms/talairach.lta
# created by someone on Mon Jun 12 00:30:01 2023

type      = 0 # LINEAR_VOX_TO_VOX
nxforms   = 1
mean      = 127.0000 128.0000 115.0000
sigma     = 10000.0000
1 4 4
1.103049755096436e+00 3.382610157132149e-02 5.719165317714214e-03 -1.870643997192383e+01
-3.551609069108963e-02 1.016282677650452e+00 1.718285232782364e-01 -1.649753761291504e+01
0.000000000000000e+00 -2.052353024482727e-01 8.770194649696350e-01 3.713496017456055e+01
0.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00 1.000000000000000e+00
src volume info
valid = 1  # volume info valid
filename = nu.mgz
volume = 256 256 256
voxelsize = 1.000000000000000e+00 1.000000000000000e+00 1.000000000000000e+00
xras   = -1.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00
yras   = 0.000000000000000e+00 0.000000000000000e+00 -1.000000000000000e+00
zras   = 0.000000000000000e+00 1.000000000000000e+00 0.000000000000000e+00
cras   = 5.399719238281250e+00 1.800000000000000e+01 0.000000000000000e+00
dst volume info
valid = 1  # volume info valid
filename = /usr/local/freesurfer/dev/average/RB_all_2020-01-02.gca
volume = 256 256 256
voxelsize = 1.000000000000000e+00 1.000000000000000e+00 1.000000000000000e+00
xras   = -1.000000000000000e+00 0.000000000000000e+00 0.000000000000000e+00
yras   = 0.000000000000000e+00 0.000000000000000e+00 -1.000000000000000e+00
zras   = 0.000000000000000e+00 1.000000000000000e+00 0.000000000000000e+00
cras   = 1.000000000000000e+00 2.000000000000000e+00 3.000000000000000e+00
"""

ASEG_STATS = """# Title Segmentation Statistics
# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1100000.000000, mm^3
# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, 1500000.000000, mm^3
# Measure TotalGray, TotalGrayVol, Total gray matter volume, 600000.000000, mm^3
# ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
  1  10   7000   7500.0  Left-Thalamus  90.0 5.0 40.0 110.0 70.0
  2  49   7100   7600.0  Right-Thalamus  90.0 5.0 40.0 110.0 70.0
"""


def aparc_stats_text(hemi):
    lines = [
        f"# hemi {hemi}",
        "# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd",
    ]
    for i, name in enumerate(DESIKAN):
        lines.append(f"{name} {1000 + i} {700 + i} {1800 + i} 2.5 0.5 0.1 0.02 10 1.0")
    return "\n".join(lines) + "\n"


def touch(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    return path


# What each processing level adds to the subject folder
LEVEL1 = ["mri/orig.mgz", "mri/brainmask.mgz", "mri/T1.mgz"]
LEVEL2 = ["mri/transforms/talairach.lta", "mri/wm.mgz"] + [
    f"surf/{h}.{n}" for h in ("lh", "rh") for n in ("pial", "white", "inflated", "curv", "sulc")
] + [f"stats/{h}.curv.stats" for h in ("lh", "rh")]  # fmt: skip
LEVEL3 = [
    "mri/aseg.mgz", "mri/aparc+aseg.mgz", "mri/aparc.a2009s+aseg.mgz",
    "mri/aparc.DKTatlas+aseg.mgz", "mri/wmparc.mgz", "mri/ribbon.mgz",
] + [
    f"surf/{h}.{n}" for h in ("lh", "rh") for n in ("sphere", "thickness", "area", "volume")
] + [
    f"label/{h}.{a}.annot" for h in ("lh", "rh") for a in ("aparc", "aparc.a2009s", "aparc.DKTatlas")
] + [
    f"stats/{h}.{a}.stats" for h in ("lh", "rh") for a in ("aparc", "aparc.a2009s", "aparc.DKTatlas")
]  # fmt: skip


def make_touched_subject(root, subj_id, level):
    """Subject folder whose files exist (empty) up to the given processing level."""
    subj = Path(root) / subj_id
    subj.mkdir(parents=True, exist_ok=True)
    for level_files in (LEVEL1, LEVEL2, LEVEL3)[:level]:
        for rel in level_files:
            touch(subj / rel)
    return subj


@pytest.fixture
def subjects(tmp_path):
    folder = tmp_path / "subjects"
    folder.mkdir()
    return folder


def build_real_subject(subjects, sphere, desikan_annot):
    """Subject with real surfaces, maps, a Desikan annotation and stats (no volumes)."""
    subj = subjects / "sub-01_ses-M00"
    coords = np.asarray(sphere.mesh.points, dtype=np.float32)
    faces = sphere.get_faces().astype(np.int32)
    n = coords.shape[0]
    rng = np.random.default_rng(0)
    for hemi in ("lh", "rh"):
        (subj / "surf").mkdir(parents=True, exist_ok=True)
        (subj / "label").mkdir(exist_ok=True)
        (subj / "stats").mkdir(exist_ok=True)
        for name, scale in (("pial", 1.0), ("white", 0.9), ("inflated", 1.1)):
            nib.freesurfer.write_geometry(
                str(subj / "surf" / f"{hemi}.{name}"), coords * scale, faces
            )
        for name in ("curv", "sulc", "thickness", "area", "volume"):
            values = rng.random(n).astype(np.float32) + 1.0
            nib.freesurfer.write_morph_data(
                str(subj / "surf" / f"{hemi}.{name}"), values
            )
        desikan_annot.save_annotation(str(subj / "label" / f"{hemi}.aparc.annot"))
        write_cortex_label(subj / "label" / f"{hemi}.cortex.label", range(n))
        (subj / "stats" / f"{hemi}.aparc.stats").write_text(aparc_stats_text(hemi))
        touch(subj / "stats" / f"{hemi}.curv.stats")
    (subj / "stats" / "aseg.stats").write_text(ASEG_STATS)
    (subj / "mri" / "transforms").mkdir(parents=True)
    (subj / "mri" / "transforms" / "talairach.lta").write_text(TALAIRACH_LTA)
    return subj


@pytest.fixture
def real_subject(subjects, sphere, desikan_annot):
    return build_real_subject(subjects, sphere, desikan_annot)


class TestSubjectConstruction:
    def test_attributes(self, subjects):
        make_touched_subject(subjects, "bert", 3)
        subject = FreeSurferSubject("bert", subjects)
        assert subject.subj_id == "bert"
        assert subject.subjs_dir == str(subjects)
        assert set(subject.fs_files) == {"mri", "surf", "stats"}

    def test_files_of_a_complete_subject(self, subjects):
        subj = make_touched_subject(subjects, "bert", 3)
        files = FreeSurferSubject("bert", str(subjects)).fs_files
        assert files["mri"]["T1"] == str(subj / "mri" / "T1.mgz")
        assert files["mri"]["talairach"] == str(
            subj / "mri" / "transforms" / "talairach.lta"
        )
        assert files["mri"]["parc"]["desikan+aseg"] == str(
            subj / "mri" / "aparc+aseg.mgz"
        )
        assert files["surf"]["lh"]["mesh"]["pial"] == str(subj / "surf" / "lh.pial")
        assert files["surf"]["rh"]["map"]["thickness"] == str(
            subj / "surf" / "rh.thickness"
        )
        assert files["surf"]["lh"]["parc"]["desikan"] == str(
            subj / "label" / "lh.aparc.annot"
        )
        assert files["surf"]["rh"]["parc"]["dkt"] == str(
            subj / "label" / "rh.aparc.DKTatlas.annot"
        )
        assert files["stats"]["lh"]["destrieux"] == str(
            subj / "stats" / "lh.aparc.a2009s.stats"
        )

    def test_missing_files_are_empty_strings(self, subjects):
        make_touched_subject(subjects, "bert", 1)
        files = FreeSurferSubject("bert", subjects).fs_files
        assert files["mri"]["T1"] != ""
        assert files["mri"]["talairach"] == ""
        assert files["surf"]["lh"]["mesh"]["pial"] == ""
        assert files["surf"]["lh"]["map"]["lgi"] == ""
        assert files["stats"]["global"]["aseg"] == ""

    def test_file_counts(self, subjects):
        make_touched_subject(subjects, "full", 3)
        counts = FreeSurferSubject("full", subjects).fs_files_count
        assert counts == {
            "mri": 4,
            "vparc": 7,
            "surf": 8,
            "map": 10,
            "sparc": 6,
            "stats": 8,
        }

    def test_file_counts_of_partial_subjects(self, subjects):
        make_touched_subject(subjects, "one", 1)
        make_touched_subject(subjects, "two", 2)
        assert FreeSurferSubject("one", subjects).fs_files_count == {
            "mri": 3, "vparc": 0, "surf": 0, "map": 0, "sparc": 0, "stats": 0,
        }  # fmt: skip
        assert FreeSurferSubject("two", subjects).fs_files_count == {
            "mri": 4, "vparc": 1, "surf": 6, "map": 4, "sparc": 0, "stats": 2,
        }  # fmt: skip

    def test_global_stats_are_counted(self, subjects):
        subj = make_touched_subject(subjects, "full", 3)
        touch(subj / "stats" / "aseg.stats")
        touch(subj / "stats" / "wmparc.stats")
        touch(subj / "stats" / "brainvol.stats")
        subject = FreeSurferSubject("full", subjects)
        assert subject.fs_files_count["stats"] == 11
        assert subject.fs_files["stats"]["global"]["brainvol"].endswith(
            "brainvol.stats"
        )

    def test_lgi_map(self, subjects):
        subj = make_touched_subject(subjects, "full", 3)
        touch(subj / "surf" / "lh.pial_lgi")
        subject = FreeSurferSubject("full", subjects)
        assert subject.fs_files["surf"]["lh"]["map"]["lgi"].endswith("lh.pial_lgi")
        assert subject.fs_files_count["map"] == 11

    def test_unknown_subject(self, subjects):
        subject = FreeSurferSubject("ghost", subjects)
        assert sum(subject.fs_files_count.values()) == 0

    def test_subjects_directory_from_the_environment(self, subjects, monkeypatch):
        make_touched_subject(subjects, "bert", 1)
        monkeypatch.setenv("SUBJECTS_DIR", str(subjects))
        subject = FreeSurferSubject("bert")
        assert subject.subjs_dir == str(subjects)
        assert subject.fs_files_count["mri"] == 3

    def test_missing_subjects_directory_is_created(self, tmp_path):
        folder = tmp_path / "new" / "subjects"
        subject = FreeSurferSubject("bert", folder)
        assert folder.is_dir()
        assert subject.subjs_dir == str(folder)

    def test_no_subjects_directory(self, monkeypatch):
        monkeypatch.delenv("SUBJECTS_DIR", raising=False)
        with pytest.raises(ValueError, match="SUBJECTS_DIR"):
            FreeSurferSubject("bert")


class TestHemiDicts:
    def test_dictionaries_and_counts(self, subjects):
        subj = make_touched_subject(subjects, "full", 3)
        subject = FreeSurferSubject("full", subjects)
        s, m, p, t, s_cnt, m_cnt, p_cnt, t_cnt = subject.get_hemi_dicts(str(subj), "lh")
        assert set(s) == {"pial", "white", "inflated", "sphere"}
        assert set(m) == {"curv", "sulc", "thickness", "area", "volume", "lgi"}
        assert set(p) == {"desikan", "destrieux", "dkt"}
        assert set(t) == {"desikan", "destrieux", "dkt", "curv"}
        assert (s_cnt, m_cnt, p_cnt, t_cnt) == (4, 5, 3, 4)
        assert s["pial"] == str(subj / "surf" / "lh.pial")
        assert m["lgi"] == ""

    def test_right_hemisphere(self, subjects):
        subj = make_touched_subject(subjects, "full", 3)
        subject = FreeSurferSubject("full", subjects)
        s, m, p, t, *_ = subject.get_hemi_dicts(str(subj), "rh")
        assert s["white"] == str(subj / "surf" / "rh.white")
        assert m["sulc"] == str(subj / "surf" / "rh.sulc")
        assert p["dkt"] == str(subj / "label" / "rh.aparc.DKTatlas.annot")
        assert t["curv"] == str(subj / "stats" / "rh.curv.stats")

    def test_only_one_hemisphere_present(self, subjects):
        subj = make_touched_subject(subjects, "half", 2)
        (subj / "surf" / "rh.pial").unlink()
        subject = FreeSurferSubject("half", subjects)
        assert subject.get_hemi_dicts(str(subj), "lh")[4] == 3
        assert subject.get_hemi_dicts(str(subj), "rh")[4] == 2


class TestProcessingStatus:
    @pytest.mark.parametrize(
        "level, status",
        [(0, "unprocessed"), (1, "autorecon1"), (2, "autorecon2"), (3, "processed")],
    )
    def test_status_by_level(self, subjects, level, status):
        make_touched_subject(subjects, "bert", level)
        subject = FreeSurferSubject("bert", subjects)
        subject.get_proc_status()
        assert subject.pstatus == status

    def test_subject_without_folder(self, subjects):
        subject = FreeSurferSubject("ghost", subjects)
        subject.get_proc_status()
        assert subject.pstatus == "unprocessed"

    def test_one_missing_file_lowers_the_status(self, subjects):
        subj = make_touched_subject(subjects, "bert", 3)
        (subj / "stats" / "rh.aparc.DKTatlas.stats").unlink()
        subject = FreeSurferSubject("bert", subjects)
        subject.get_proc_status()
        assert subject.pstatus == "autorecon2"

    def test_missing_early_file(self, subjects):
        subj = make_touched_subject(subjects, "bert", 3)
        (subj / "mri" / "T1.mgz").unlink()
        subject = FreeSurferSubject("bert", subjects)
        subject.get_proc_status()
        assert subject.pstatus == "unprocessed"

    def test_inconsistent_subject(self, subjects):
        subj = make_touched_subject(subjects, "bert", 3)
        (
            subj / "mri" / "wm.mgz"
        ).unlink()  # autorecon2 incomplete but autorecon3 present
        subject = FreeSurferSubject("bert", subjects)
        subject.get_proc_status()
        assert subject.pstatus == "unprocessed"

    def test_pial_files_are_restored_from_the_t1_versions(self, subjects):
        subj = make_touched_subject(subjects, "bert", 3)
        for hemi in ("lh", "rh"):
            (subj / "surf" / f"{hemi}.pial").write_text("pial")
            (subj / "surf" / f"{hemi}.pial").rename(subj / "surf" / f"{hemi}.pial.T1")
        subject = FreeSurferSubject("bert", subjects)
        subject.get_proc_status()
        assert (subj / "surf" / "lh.pial").read_text() == "pial"
        assert (subj / "surf" / "rh.pial").read_text() == "pial"
        assert subject.pstatus == "processed"
        assert subject.fs_files["surf"]["lh"]["mesh"]["pial"] == str(
            subj / "surf" / "lh.pial"
        )

    def test_existing_pial_files_are_not_replaced(self, subjects):
        subj = make_touched_subject(subjects, "bert", 3)
        (subj / "surf" / "lh.pial").write_text("final")
        (subj / "surf" / "lh.pial.T1").write_text("t1")
        FreeSurferSubject("bert", subjects).get_proc_status()
        assert (subj / "surf" / "lh.pial").read_text() == "final"


class TestCras:
    def test_cras_of_the_talairach_transform(self, real_subject, subjects):
        subject = FreeSurferSubject(real_subject.name, subjects)
        result = subject.get_cras()
        assert subject.cras == pytest.approx((5.39971923828125, 18.0, 0.0))
        assert result == subject.cras

    def test_missing_transform(self, subjects):
        make_touched_subject(subjects, "bert", 1)
        subject = FreeSurferSubject("bert", subjects)
        with pytest.raises(ValueError, match="Talairach transform file does not exist"):
            subject.get_cras()


class TestFileGetters:
    @pytest.fixture
    def subject(self, subjects):
        make_touched_subject(subjects, "full", 3)
        return FreeSurferSubject("full", subjects)

    @pytest.mark.parametrize("hemi", ["lh", "rh"])
    @pytest.mark.parametrize("surf", ["pial", "white", "inflated", "sphere"])
    def test_surfaces(self, subject, hemi, surf):
        assert subject.get_surface(hemi, surf).endswith(
            f"surf/{hemi}.{surf}"
        ) or subject.get_surface(hemi, surf).endswith(f"surf{os.sep}{hemi}.{surf}")

    @pytest.mark.parametrize(
        "map_type", ["curv", "sulc", "thickness", "area", "volume"]
    )
    def test_maps(self, subject, map_type):
        assert Path(subject.get_map("rh", map_type)).name == f"rh.{map_type}"

    @pytest.mark.parametrize(
        "annot_type, name",
        [
            ("desikan", "aparc"),
            ("destrieux", "aparc.a2009s"),
            ("dkt", "aparc.DKTatlas"),
        ],
    )
    def test_annotations(self, subject, annot_type, name):
        assert Path(subject.get_annotation("lh", annot_type)).name == f"lh.{name}.annot"

    def test_invalid_arguments(self, subject):
        with pytest.raises(ValueError, match="hemisphere must be lh or rh"):
            subject.get_surface("bh", "pial")
        with pytest.raises(ValueError, match="surface type must be"):
            subject.get_surface("lh", "midthickness")
        with pytest.raises(ValueError, match="hemisphere must be lh or rh"):
            subject.get_map("both", "curv")
        with pytest.raises(ValueError, match="map type must be"):
            subject.get_map("lh", "nonexistent")
        with pytest.raises(ValueError, match="hemisphere must be lh or rh"):
            subject.get_annotation("x", "desikan")
        with pytest.raises(ValueError, match="annotation type must be"):
            subject.get_annotation("lh", "custom")

    def test_missing_files_are_empty_strings(self, subjects):
        make_touched_subject(subjects, "early", 1)
        subject = FreeSurferSubject("early", subjects)
        assert subject.get_surface("lh", "pial") == ""
        assert subject.get_map("lh", "thickness") == ""
        assert subject.get_annotation("lh", "desikan") == ""


class TestSetFreesurferDirectory:
    def test_explicit_directory(self, tmp_path):
        target = tmp_path / "a" / "subjects"
        FreeSurferSubject.set_freesurfer_directory(target)
        assert target.is_dir()
        assert os.environ["SUBJECTS_DIR"] == str(target)

    def test_directory_from_the_environment(self, tmp_path, monkeypatch):
        target = tmp_path / "from_env"
        monkeypatch.setenv("SUBJECTS_DIR", str(target))
        FreeSurferSubject.set_freesurfer_directory()
        assert target.is_dir()
        assert os.environ["SUBJECTS_DIR"] == str(target)

    def test_no_directory(self, monkeypatch):
        monkeypatch.delenv("SUBJECTS_DIR", raising=False)
        with pytest.raises(
            ValueError, match="must be set in the environment variables"
        ):
            FreeSurferSubject.set_freesurfer_directory()


####################################################################################################
class TestSurfaceHemiMorpho:
    LOBES = ["frontal", "parietal", "temporal", "occipital", "insula"]

    def rows(self, df, atlas, metric, source=None):
        keep = (df["Atlas"] == atlas) & (df["Metric"] == metric)
        if source is not None:
            keep &= df["Source"] == source
        return df[keep]

    @pytest.fixture
    def morpho(self, real_subject, subjects):
        subject = FreeSurferSubject(real_subject.name, subjects)
        return subject, subject.surface_hemi_morpho("lh")

    def test_columns_and_atlases(self, morpho):
        _, df = morpho
        assert df.columns.tolist() == [
            "Source", "Metric", "Units", "MetricFile", "Atlas",
            "Supraregion", "Hemisphere", "Region", "Value", "Median", "Std", "Min", "Max",
        ]  # fmt: skip
        assert set(df["Atlas"]) == {"", "desikan", "lobes_desikan"}
        assert set(df["Hemisphere"]) == {"lh"}

    def test_the_lobes_are_grouped(self, morpho):
        _, df = morpho
        regions = self.rows(df, "lobes_desikan", "thickness", "vertices")[
            "Region"
        ].tolist()
        assert regions == ["ctx-lh-hemisphere"] + [
            f"ctx-lh-{lobe}" for lobe in self.LOBES
        ]

    def test_lobe_thickness_is_the_mean_over_its_vertices(
        self, morpho, real_subject, desikan_annot
    ):
        _, df = morpho
        thickness = nib.freesurfer.read_morph_data(
            str(real_subject / "surf" / "lh.thickness")
        )
        lobar = desikan_annot.group_into_lobes()
        for name, code in zip(lobar.regnames, lobar.regtable[:, 4], strict=True):
            row = self.rows(df, "lobes_desikan", "thickness", "vertices")
            value = row[row["Region"] == f"ctx-lh-{name}"]["Value"].iloc[0]
            assert value == pytest.approx(
                thickness[lobar.codes == code].mean(), rel=1e-5
            )

    def test_regional_thickness(self, morpho, real_subject, desikan_annot):
        _, df = morpho
        thickness = nib.freesurfer.read_morph_data(
            str(real_subject / "surf" / "lh.thickness")
        )
        rows = self.rows(df, "desikan", "thickness", "vertices")
        assert len(rows) == 34  # no hemisphere row
        code = desikan_annot.regtable[desikan_annot.regnames.index("insula"), 4]
        value = rows[rows["Region"] == "ctx-lh-insula"]["Value"].iloc[0]
        assert value == pytest.approx(
            thickness[desikan_annot.codes == code].mean(), rel=1e-5
        )

    def test_surface_areas_of_the_lobes_add_up_to_the_surface(self, morpho, sphere):
        from clabtoolkit import morphometrytools as morpho_tools

        _, df = morpho
        total = morpho_tools.area_from_mesh(sphere.mesh.points, sphere.get_faces())[0]
        pial = self.rows(df, "lobes_desikan", "area", "pial")
        hemisphere = pial[pial["Region"] == "ctx-lh-hemisphere"]["Value"].iloc[0]
        lobes = pial[pial["Region"] != "ctx-lh-hemisphere"]["Value"]
        assert len(lobes) == 5
        assert lobes.sum() == pytest.approx(total, rel=1e-4)
        assert hemisphere == pytest.approx(total, rel=1e-4)

    def test_white_surface_is_smaller_than_the_pial(self, morpho):
        _, df = morpho
        pial = self.rows(df, "desikan", "area", "pial")["Value"].sum()
        white = self.rows(df, "desikan", "area", "white")["Value"].sum()
        assert white == pytest.approx(0.81 * pial, rel=1e-4)  # scaled by 0.9

    def test_euler_characteristic_of_each_surface(self, morpho):
        _, df = morpho
        euler = df[df["Metric"] == "euler"]
        assert sorted(euler["Source"]) == ["pial", "white"]
        assert euler["Value"].tolist() == [2, 2]

    def test_values_from_the_stats_file(self, morpho):
        _, df = morpho
        rows = self.rows(df, "desikan", "thickness", "statsfile")
        assert len(rows) == 34
        assert set(rows["Value"]) == {2.5}
        assert set(rows["Std"]) == {0.5}

    def test_each_row_appears_once(self, morpho):
        _, df = morpho
        keys = ["Source", "Metric", "Atlas", "Region"]
        assert not df.duplicated(keys).any()

    def test_several_parcellations_do_not_repeat_the_lobes_or_the_euler_numbers(
        self, real_subject, subjects, desikan_annot
    ):
        single = FreeSurferSubject(real_subject.name, subjects).surface_hemi_morpho(
            "lh"
        )
        # A second parcellation with its own stats file
        desikan_annot.save_annotation(
            str(real_subject / "label" / "lh.aparc.DKTatlas.annot")
        )
        (real_subject / "stats" / "lh.aparc.DKTatlas.stats").write_text(
            aparc_stats_text("lh")
        )
        both = FreeSurferSubject(real_subject.name, subjects).surface_hemi_morpho("lh")
        assert set(both["Atlas"]) == {"", "desikan", "dkt", "lobes_desikan"}
        assert len(both[both["Metric"] == "euler"]) == 2
        lobes = both[both["Atlas"] == "lobes_desikan"]
        assert len(lobes) == len(single[single["Atlas"] == "lobes_desikan"])
        assert len(both[both["Atlas"] == "dkt"]) == len(
            single[single["Atlas"] == "desikan"]
        )

    def test_without_the_desikan_parcellation(
        self, real_subject, subjects, desikan_annot
    ):
        desikan_annot.save_annotation(
            str(real_subject / "label" / "lh.aparc.DKTatlas.annot")
        )
        (real_subject / "stats" / "lh.aparc.DKTatlas.stats").write_text(
            aparc_stats_text("lh")
        )
        (real_subject / "label" / "lh.aparc.annot").unlink()
        df = FreeSurferSubject(real_subject.name, subjects).surface_hemi_morpho("lh")
        assert set(df["Atlas"]) == {"", "dkt"}
        assert "lobes_desikan" not in set(df["Atlas"])

    def test_without_the_stats_file(self, real_subject, subjects):
        (real_subject / "stats" / "lh.aparc.stats").unlink()
        df = FreeSurferSubject(real_subject.name, subjects).surface_hemi_morpho("lh")
        assert "statsfile" not in set(df["Source"])
        assert len(self.rows(df, "desikan", "thickness", "vertices")) == 34

    def test_right_hemisphere(self, real_subject, subjects):
        df = FreeSurferSubject(real_subject.name, subjects).surface_hemi_morpho("rh")
        assert set(df["Hemisphere"]) == {"rh"}
        assert all(r.startswith("ctx-rh-") for r in df["Region"])

    def test_other_lobe_scheme(self, real_subject, subjects):
        df = FreeSurferSubject(real_subject.name, subjects).surface_hemi_morpho(
            "lh", lobes_grouping="desikan_w_cingulate"
        )
        assert "lobes_desikan_w_cingulate" in set(df["Atlas"])
        regions = self.rows(df, "lobes_desikan_w_cingulate", "thickness", "vertices")[
            "Region"
        ]
        assert "ctx-lh-cingulate" in regions.tolist()

    def test_verbose_reports_the_steps(self, real_subject, subjects, capsys):
        FreeSurferSubject(real_subject.name, subjects).surface_hemi_morpho(
            "lh", verbose=True
        )
        out = capsys.readouterr().out
        assert "Step 0: Grouping into lobes" in out
        assert "Step 1: Computing morphometric metrics for the LH hemisphere" in out
        assert "Euler Characteristic" in out

    def test_subject_without_surfaces(self, subjects):
        make_touched_subject(subjects, "early", 1)
        df = FreeSurferSubject("early", subjects).surface_hemi_morpho("lh")
        assert df.empty


class TestStatsTables:
    @pytest.fixture
    def subject(self, real_subject, subjects):
        return FreeSurferSubject(real_subject.name, subjects)

    def test_volume_table_without_volumes(self, subject):
        assert subject.volume_morpho().empty

    def test_volume_table_ignores_unknown_parcellations(self, subject):
        assert subject.volume_morpho(parcellations=["nothing", "wm"]).empty

    def test_stats_table(self, subject):
        df = subject.create_stats_table()
        assert df.columns.tolist() == [
            "Participant", "Source", "Metric", "Units", "MetricFile", "Atlas",
            "Supraregion", "Hemisphere", "Region", "Value", "Median", "Std", "Min", "Max",
        ]  # fmt: skip
        assert set(df["Participant"]) == {"sub-01_ses-M00"}
        assert {"lh", "rh", "brain"} <= set(df["Hemisphere"])
        assert subject.stats_table is df

    def test_global_and_aseg_volumes(self, subject):
        df = subject.create_stats_table()
        whole = df[df["Region"] == "brain-brain-intracraneal"]
        assert whole["Value"].tolist() == [1500000.0]
        thalamus = df[df["Region"] == "thal-lh-thalamus"]
        assert thalamus["Value"].tolist() == [7500.0]

    def test_both_hemispheres_are_included(self, subject):
        df = subject.create_stats_table()
        thickness = df[(df["Metric"] == "thickness") & (df["Atlas"] == "desikan")]
        assert {"ctx-lh-insula", "ctx-rh-insula"} <= set(thickness["Region"])

    def test_bids_entities(self, subject):
        df = subject.create_stats_table(add_bids_entities=True)
        assert df["Participant"].unique().tolist() == ["01"]
        assert df["Session"].unique().tolist() == ["M00"]

    def test_table_is_saved(self, tmp_path, subject):
        out = tmp_path / "stats.csv"
        df = subject.create_stats_table(output_file=out)
        saved = pd.read_csv(out)
        assert len(saved) == len(df)
        assert saved.columns.tolist() == df.columns.tolist()

    def test_subject_without_global_stats(self, real_subject, subjects):
        (real_subject / "stats" / "aseg.stats").unlink()
        with pytest.raises(FileNotFoundError, match="Stats file not found"):
            FreeSurferSubject(real_subject.name, subjects).create_stats_table()


####################################################################################################
@pytest.mark.skipif(os.name == "nt", reason="FreeSurfer does not run on Windows")
class TestLaunchFreesurfer:
    @pytest.fixture
    def version(self, monkeypatch):
        """FreeSurfer version reported by recon-all."""
        state = {"value": "7.4.1"}
        monkeypatch.setattr(cltfree, "get_version", lambda **kwargs: state["value"])
        return state

    @pytest.fixture
    def t1w(self, tmp_path):
        path = tmp_path / "sub-01_T1w.nii.gz"
        path.write_text("image")
        return str(path)

    def launch(self, subjects, subj_id, **kwargs):
        subject = FreeSurferSubject(subj_id, subjects)
        return subject, subject.launch_freesurfer(**kwargs)

    def test_new_subject_runs_the_whole_pipeline(
        self, subjects, fake_run, version, t1w
    ):
        self.launch(subjects, "new", t1w_img=t1w)
        assert fake_run.commands == [["recon-all", "-i", t1w, "-subjid", "new", "-all"]]

    def test_path_object_for_the_image(self, subjects, fake_run, version, t1w):
        self.launch(subjects, "new", t1w_img=Path(t1w))
        assert fake_run.commands[0][2] == t1w

    def test_the_subjects_directory_is_set_for_freesurfer(
        self, subjects, fake_run, version, t1w, monkeypatch
    ):
        monkeypatch.delenv("SUBJECTS_DIR", raising=False)
        self.launch(subjects, "new", t1w_img=t1w)
        assert os.environ["SUBJECTS_DIR"] == str(subjects)

    def test_unprocessed_subject_with_an_orig_volume(self, subjects, fake_run, version):
        subj = subjects / "partial"
        touch(subj / "mri" / "orig.mgz")
        self.launch(subjects, "partial")
        assert fake_run.commands == [["recon-all", "-subjid", "partial", "-all"]]

    def test_new_subject_needs_an_image(self, subjects, fake_run, version):
        with pytest.raises(ValueError, match="t1w_img"):
            self.launch(subjects, "new")
        assert fake_run.commands == []

    def test_missing_image(self, subjects, fake_run, version, tmp_path):
        with pytest.raises(ValueError, match="T1w image does not exist"):
            self.launch(subjects, "new", t1w_img=tmp_path / "missing.nii.gz")
        assert fake_run.commands == []

    def test_autorecon1_subject_continues_with_the_next_stages(
        self, subjects, fake_run, version
    ):
        make_touched_subject(subjects, "bert", 1)
        self.launch(subjects, "bert")
        assert fake_run.commands == [
            ["recon-all", "-subjid", "bert", "-autorecon2"],
            ["recon-all", "-subjid", "bert", "-autorecon3"],
        ]

    def test_autorecon2_subject_runs_the_last_stage(self, subjects, fake_run, version):
        make_touched_subject(subjects, "bert", 2)
        self.launch(subjects, "bert")
        assert fake_run.commands == [["recon-all", "-subjid", "bert", "-autorecon3"]]

    def test_processed_subject_is_left_alone(self, subjects, fake_run, version):
        make_touched_subject(subjects, "bert", 3)
        _, status = self.launch(subjects, "bert")
        assert fake_run.commands == []
        assert status == "processed"

    def test_requested_stage(self, subjects, fake_run, version):
        make_touched_subject(subjects, "bert", 1)
        self.launch(subjects, "bert", proc_stage="autorecon2")
        assert fake_run.commands == [["recon-all", "-subjid", "bert", "-autorecon2"]]

    def test_requested_stage_that_is_already_done(self, subjects, fake_run, version):
        make_touched_subject(subjects, "bert", 2)
        self.launch(subjects, "bert", proc_stage="autorecon2")
        assert fake_run.commands == []

    def test_requested_stage_of_a_new_subject(self, subjects, fake_run, version, t1w):
        self.launch(subjects, "new", t1w_img=t1w, proc_stage=["AutoRecon1"])
        assert fake_run.commands == [
            ["recon-all", "-i", t1w, "-subjid", "new", "-autorecon1"]
        ]

    def test_stage_names_are_validated_first(self, subjects, fake_run, version, t1w):
        with pytest.raises(ValueError, match="Stage autorecon9 is not valid"):
            self.launch(subjects, "new", t1w_img=t1w, proc_stage="autorecon9")
        with pytest.raises(ValueError, match="Stage cerebellum is not valid"):
            self.launch(subjects, "new", t1w_img=t1w, extra_proc=["lgi", "cerebellum"])
        assert fake_run.commands == []

    def test_overwrite_with_an_image(self, subjects, fake_run, version, t1w):
        make_touched_subject(subjects, "bert", 3)
        self.launch(
            subjects,
            "bert",
            t1w_img=t1w,
            proc_stage=["autorecon1", "autorecon2"],
            overwrite=True,
        )
        assert fake_run.commands == [
            ["recon-all", "-subjid", "bert", "-i", t1w, "-autorecon1"],
            ["recon-all", "-subjid", "bert", "-i", t1w, "-autorecon2"],
        ]

    def test_overwrite_without_an_image_uses_the_orig_volume(
        self, subjects, fake_run, version
    ):
        make_touched_subject(subjects, "bert", 3)
        self.launch(subjects, "bert", proc_stage="all", overwrite=True)
        assert fake_run.commands == [["recon-all", "-subjid", "bert", "-all"]]

    def test_overwrite_without_an_image_nor_an_orig_volume(
        self, subjects, fake_run, version
    ):
        self.launch(subjects, "ghost", overwrite=True)
        assert fake_run.commands == []

    def test_all_replaces_the_other_stages(self, subjects, fake_run, version):
        make_touched_subject(subjects, "bert", 3)
        self.launch(subjects, "bert", proc_stage=["autorecon1", "all"], overwrite=True)
        assert fake_run.commands == [["recon-all", "-subjid", "bert", "-all"]]

    def test_lgi(self, subjects, fake_run, version):
        make_touched_subject(subjects, "bert", 3)
        self.launch(subjects, "bert", extra_proc="lgi")
        assert fake_run.commands == [["recon-all", "-subjid", "bert", "-lgi"]]

    def test_lgi_is_skipped_when_it_exists(self, subjects, fake_run, version):
        subj = make_touched_subject(subjects, "bert", 3)
        touch(subj / "surf" / "lh.pial_lgi")
        self.launch(subjects, "bert", extra_proc="lgi")
        assert fake_run.commands == []

    def test_lgi_is_computed_again_when_overwriting(self, subjects, fake_run, version):
        subj = make_touched_subject(subjects, "bert", 3)
        touch(subj / "surf" / "lh.pial_lgi")
        self.launch(
            subjects,
            "bert",
            extra_proc=["lgi"],
            overwrite=True,
            proc_stage="autorecon3",
        )
        assert ["recon-all", "-subjid", "bert", "-lgi"] in fake_run.commands

    def test_subregion_segmentations_of_recent_versions(
        self, subjects, fake_run, version
    ):
        make_touched_subject(subjects, "bert", 3)
        self.launch(
            subjects, "bert", extra_proc=["thalamus", "brainstem", "hippocampus"]
        )
        assert fake_run.commands == [
            ["segment_subregions", "thalamus", "--cross", "bert"],
            ["segment_subregions", "brainstem", "--cross", "bert"],
            ["segment_subregions", "hippo-amygdala", "--cross", "bert"],
        ]

    def test_subregion_segmentations_of_old_versions(self, subjects, fake_run, version):
        version["value"] = "7.2.0"
        make_touched_subject(subjects, "bert", 3)
        self.launch(subjects, "bert", extra_proc=["thalamus", "brainstem", "amygdala"])
        assert fake_run.commands == [
            ["segmentThalamicNuclei.sh", "bert", str(subjects)],
            ["segmentBS.sh", "bert", str(subjects)],
            ["segmentHA_T1.sh", "bert", str(subjects)],
        ]

    def test_version_7_3_uses_the_new_commands(self, subjects, fake_run, version):
        version["value"] = "7.3.0"
        make_touched_subject(subjects, "bert", 3)
        self.launch(subjects, "bert", extra_proc="thalamus")
        assert fake_run.commands == [
            ["segment_subregions", "thalamus", "--cross", "bert"]
        ]

    def test_hippocampus_and_amygdala_run_once(self, subjects, fake_run, version):
        make_touched_subject(subjects, "bert", 3)
        self.launch(subjects, "bert", extra_proc=["Hippocampus", "amygdala"])
        assert fake_run.commands == [
            ["segment_subregions", "hippo-amygdala", "--cross", "bert"]
        ]

    def test_hypothalamus(self, subjects, fake_run, version):
        make_touched_subject(subjects, "bert", 3)
        self.launch(subjects, "bert", extra_proc="hypothalamus")
        assert fake_run.commands == [
            [
                "mri_segment_hypothalamic_subunits", "--s", "bert", "--sd", str(subjects),
                "--write_posteriors",
            ]
        ]  # fmt: skip

    def test_existing_segmentations_are_skipped(self, subjects, fake_run, version):
        subj = make_touched_subject(subjects, "bert", 3)
        for i in range(3):
            touch(subj / "mri" / f"ThalamicNuclei.v13.{i}.mgz")
            touch(subj / "mri" / f"brainstemSsLabels.{i}.mgz")
            touch(subj / "mri" / f"hypothalamic_subunits_{i}.mgz")
        for i in range(16):
            touch(subj / "mri" / f"lh.hippoAmygLabels.{i}.mgz")
        self.launch(
            subjects,
            "bert",
            extra_proc=["thalamus", "brainstem", "hippocampus", "hypothalamus"],
        )
        assert fake_run.commands == []

    def test_segmentations_write_the_posteriors(self, subjects, fake_run, version):
        make_touched_subject(subjects, "bert", 3)
        self.launch(subjects, "bert", extra_proc="brainstem")
        assert fake_run.calls[-1][1]["env"]["WRITE_POSTERIORS"] == "1"

    def test_the_extra_modules_come_after_the_main_stages(
        self, subjects, fake_run, version
    ):
        make_touched_subject(subjects, "bert", 2)
        self.launch(subjects, "bert", extra_proc="lgi")
        assert fake_run.commands == [
            ["recon-all", "-subjid", "bert", "-autorecon3"],
            ["recon-all", "-subjid", "bert", "-lgi"],
        ]

    def test_unknown_version_uses_the_recent_commands(
        self, subjects, fake_run, version
    ):
        version["value"] = None
        make_touched_subject(subjects, "bert", 3)
        self.launch(subjects, "bert", extra_proc="thalamus")
        assert fake_run.commands == [
            ["segment_subregions", "thalamus", "--cross", "bert"]
        ]

    def test_containers_mount_the_subjects_directory_and_the_license(
        self, subjects, fake_run, version, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            cltfree.cltmisc,
            "generate_container_command",
            lambda args, tech, image: [tech, "run", *args],
        )
        fake_run.stdout = "/opt/value\n"
        make_touched_subject(subjects, "bert", 3)
        license_file = tmp_path / "license.txt"
        license_file.write_text("license")
        subject = FreeSurferSubject("bert", subjects)
        subject.launch_freesurfer(
            extra_proc="lgi",
            cont_tech="singularity",
            cont_image="fs.sif",
            fs_license=license_file,
        )
        cmd = fake_run.commands[-1]
        assert cmd[:6] == [
            "singularity", "run", "--bind", f"{subjects}:/opt/value",
            "--bind", f"{license_file}:/opt/value/license.txt",
        ]  # fmt: skip
        assert cmd[-3:] == ["-subjid", "bert", "-lgi"]

    def test_files_created_by_freesurfer_update_the_status(
        self, subjects, fake_run, version
    ):
        subj = make_touched_subject(subjects, "bert", 2)

        def create_the_last_stage(cmd):
            for rel in LEVEL3:
                touch(subj / rel)
            return []

        fake_run.creates = create_the_last_stage
        assert FreeSurferSubject("bert", subjects).launch_freesurfer() == "processed"

    def test_the_status_is_refreshed(self, subjects, fake_run, version):
        make_touched_subject(subjects, "bert", 2)
        subject = FreeSurferSubject("bert", subjects)
        subject.pstatus = "unprocessed"  # stale value from an earlier check
        assert subject.launch_freesurfer() == "autorecon2"
        assert fake_run.commands == [["recon-all", "-subjid", "bert", "-autorecon3"]]


####################################################################################################
class TestProjectionToIndividualSpace:
    @pytest.fixture
    def subject(self, real_subject, subjects, desikan_annot):
        """Subject with a reference subject that has the annotation to project."""
        ref_label = subjects / "fsaverage" / "label"
        ref_label.mkdir(parents=True)
        desikan_annot.save_annotation(str(ref_label / "lh.aparc.annot"))
        return FreeSurferSubject(real_subject.name, subjects)

    @staticmethod
    def creates_gappy_annotation(sphere, n_gaps=30):
        """A fake mri_surf2surf that writes an annotation with unlabeled vertices."""

        def create(cmd):
            target = cmd[cmd.index("--tval") + 1] if "--tval" in cmd else cmd[-1]
            annot = AnnotParcellation.simulate_annotation(sphere, n_regions=6, seed=2)
            annot.codes[:n_gaps] = 0
            annot.save_annotation(target)
            return []

        return create

    def test_annotation_projected_by_name(self, tmp_path, subject, fake_run, sphere):
        fake_run.creates = self.creates_gappy_annotation(sphere)
        out = tmp_path / "out" / "lh.aparc.individual.annot"
        result = subject.annot2ind("fsaverage", "lh", "aparc", out)
        assert result == str(out)
        assert fake_run.commands == [
            [
                "mri_surf2surf", "--srcsubject", "fsaverage", "--trgsubject", subject.subj_id,
                "--hemi", "lh", "--sval-annot", "aparc", "--tval", str(out),
            ]
        ]  # fmt: skip
        assert os.environ["SUBJECTS_DIR"] == subject.subjs_dir

    def test_unlabeled_vertices_are_filled(self, tmp_path, subject, fake_run, sphere):
        fake_run.creates = self.creates_gappy_annotation(sphere)
        out = tmp_path / "lh.individual.annot"
        subject.annot2ind("fsaverage", "lh", "aparc", out)
        filled = AnnotParcellation(out)
        assert (filled.codes != 0).all()

    def test_annotation_projected_from_a_file(
        self, tmp_path, subject, fake_run, sphere
    ):
        source = tmp_path / "custom.annot"
        AnnotParcellation.simulate_annotation(sphere, n_regions=3).save_annotation(
            str(source)
        )
        fake_run.creates = self.creates_gappy_annotation(sphere)
        subject.annot2ind("fsaverage", "lh", source, tmp_path / "lh.out.annot")
        assert fake_run.commands[0][
            fake_run.commands[0].index("--sval-annot") + 1
        ] == str(source)

    def test_gifti_sources_are_converted(self, tmp_path, subject, fake_run, sphere):
        source = tmp_path / "lh.custom.label.gii"
        AnnotParcellation.annot2gii(
            AnnotParcellation.simulate_annotation(sphere, n_regions=3),
            gii_file=str(source),
        )
        create_target = self.creates_gappy_annotation(sphere)
        converted = []

        def create(cmd):
            sval = cmd[cmd.index("--sval-annot") + 1]
            converted.append((sval, AnnotParcellation(sval).regnames))
            return create_target(cmd)

        fake_run.creates = create
        subject.annot2ind("fsaverage", "lh", source, tmp_path / "lh.out.annot")
        sval, names = converted[0]
        assert sval.endswith(".annot")
        assert (
            len(names) == 3
        )  # the converted annotation existed when mri_surf2surf ran
        assert not os.path.exists(sval)  # and was removed afterwards

    def test_existing_output_is_kept(self, tmp_path, subject, fake_run, capsys):
        out = tmp_path / "lh.out.annot"
        out.write_text("kept")
        assert subject.annot2ind("fsaverage", "lh", "aparc", out, verbose=True) == str(
            out
        )
        assert fake_run.commands == []
        assert out.read_text() == "kept"
        assert "already exists" in capsys.readouterr().out

    def test_existing_output_is_replaced_when_overwriting(
        self, tmp_path, subject, fake_run, sphere
    ):
        out = tmp_path / "lh.out.annot"
        out.write_text("old")
        fake_run.creates = self.creates_gappy_annotation(sphere)
        subject.annot2ind("fsaverage", "lh", "aparc", out, overwrite=True)
        assert len(fake_run.commands) == 1
        assert AnnotParcellation(out).is_loaded()

    def test_missing_source_annotation(self, tmp_path, subject, fake_run):
        with pytest.raises(FileNotFoundError, match="do not exist"):
            subject.annot2ind("fsaverage", "lh", "nothing", tmp_path / "lh.out.annot")
        assert fake_run.commands == []

    def test_failed_projection(self, tmp_path, subject, fake_run):
        with pytest.raises(RuntimeError, match="mri_surf2surf did not create"):
            subject.annot2ind("fsaverage", "lh", "aparc", tmp_path / "lh.out.annot")

    def test_gcs_classifier_applied_to_the_subject(
        self, tmp_path, subject, fake_run, sphere
    ):
        gcs = tmp_path / "lh.atlas.gcs"
        gcs.write_text("classifier")
        fake_run.creates = self.creates_gappy_annotation(sphere)
        out = tmp_path / "lh.gcs.annot"
        assert subject.gcs2ind(gcs, out, "lh") == str(out)
        sdir = Path(subject.subjs_dir) / subject.subj_id
        assert fake_run.commands == [
            [
                "mris_ca_label", "-l", str(sdir / "label" / "lh.cortex.label"), subject.subj_id,
                "lh", str(sdir / "surf" / "lh.sphere.reg"), str(gcs), str(out),
            ]
        ]  # fmt: skip
        assert (AnnotParcellation(out).codes != 0).all()

    def test_gcs_existing_output_is_kept(self, tmp_path, subject, fake_run, capsys):
        out = tmp_path / "lh.gcs.annot"
        out.write_text("kept")
        subject.gcs2ind(tmp_path / "lh.atlas.gcs", out, "lh", verbose=True)
        assert fake_run.commands == []
        assert "already exists" in capsys.readouterr().out

    def test_gcs_overwrite(self, tmp_path, subject, fake_run, sphere):
        out = tmp_path / "lh.gcs.annot"
        out.write_text("old")
        fake_run.creates = self.creates_gappy_annotation(sphere)
        subject.gcs2ind(tmp_path / "lh.atlas.gcs", out, "lh", overwrite=True)
        assert len(fake_run.commands) == 1

    def test_gcs_failure(self, tmp_path, subject, fake_run):
        with pytest.raises(RuntimeError, match="mris_ca_label did not create"):
            subject.gcs2ind(tmp_path / "lh.atlas.gcs", tmp_path / "lh.gcs.annot", "lh")


####################################################################################################
class TestSurfaceToVolume:
    @pytest.fixture
    def fs_home(self, tmp_path, monkeypatch):
        """Minimal FreeSurfer installation with a color lookup table."""
        home = tmp_path / "freesurfer"
        home.mkdir()
        (home / "FreeSurferColorLUT.txt").write_text(
            "#No. Label Name:                            R   G   B   A\n"
            "0   Unknown                                 0   0   0   0\n"
            "10  Left-Thalamus                           0 118  14   0\n"
            "49  Right-Thalamus                          0 118  14   0\n"
            "1001 ctx-lh-bankssts                       25 100  40   0\n"
        )
        monkeypatch.setenv("FREESURFER_HOME", str(home))
        return home

    @staticmethod
    def creates_volume(values=(0, 10, 49, 1001, 2001)):
        """A fake mri_aparc2aseg that writes the volume given with --o."""

        def create(cmd):
            if "--o" not in cmd:  # not the conversion itself
                return []
            out = cmd[cmd.index("--o") + 1]
            data = np.array(values, dtype=np.int32).reshape(1, 1, -1)
            if out.endswith(".mgz"):
                img = nib.MGHImage(data, np.eye(4))
            else:
                img = nib.Nifti1Image(data, np.eye(4))
            nib.save(img, out)
            return []

        return create

    @pytest.fixture
    def subject(self, real_subject, subjects):
        return FreeSurferSubject(real_subject.name, subjects)

    def test_command_without_growing(self, tmp_path, subject, fake_run):
        fake_run.creates = self.creates_volume()
        out = tmp_path / "vol" / "aparc.nii.gz"
        assert subject.surf2vol("aparc", out) == str(out)
        assert fake_run.commands == [
            [
                "mri_aparc2aseg", "--s", subject.subj_id, "--annot", "aparc",
                "--hypo-as-wm", "--new-ribbon", "--o", str(out),
            ]
        ]  # fmt: skip

    @pytest.mark.parametrize(
        "grow, extra", [("wm", ["--labelwm"]), (2, ["--wmparc-dmax", "2", "--labelwm"])]
    )
    def test_command_with_growing(self, tmp_path, subject, fake_run, grow, extra):
        fake_run.creates = self.creates_volume()
        out = tmp_path / "aparc.nii.gz"
        subject.surf2vol("aparc", out, gm_grow=grow)
        cmd = fake_run.commands[0]
        for item in extra:
            assert item in cmd
        assert cmd[-3:] == ["--new-ribbon", "--o", str(out)]

    def test_existing_volume_is_kept(self, tmp_path, subject, fake_run, capsys):
        out = tmp_path / "aparc.nii.gz"
        out.write_text("kept")
        subject.surf2vol("aparc", out, verbose=True)
        assert fake_run.commands == []
        assert out.read_text() == "kept"
        assert "already exists" in capsys.readouterr().out

    def test_existing_volume_is_replaced_when_overwriting(
        self, tmp_path, subject, fake_run
    ):
        out = tmp_path / "aparc.nii.gz"
        out.write_text("old")
        fake_run.creates = self.creates_volume()
        subject.surf2vol("aparc", out, overwrite=True)
        assert len(fake_run.commands) == 1

    def test_color_tables_next_to_the_volume(
        self, tmp_path, subject, fake_run, fs_home
    ):
        fake_run.creates = self.creates_volume()
        out = tmp_path / "aparc.nii.gz"
        subject.surf2vol("aparc", out, color_table=["tsv", "lut"])
        tsv = pd.read_csv(tmp_path / "aparc.tsv", sep="\t")
        assert (tmp_path / "aparc.lut").is_file()
        names = tsv["name"].tolist()
        assert "Left-Thalamus" in names and "Right-Thalamus" in names
        assert "ctx-lh-insula" in names and "ctx-rh-insula" in names
        # Cortical regions come from the subject's annotations, offset by 1000 and 2000
        index = dict(zip(tsv["name"], tsv["index"], strict=True))
        assert index["ctx-lh-bankssts"] == 1000
        assert index["ctx-rh-bankssts"] == 2000
        assert not any(n.startswith("wm-") for n in names)

    def test_color_tables_of_a_volume_with_another_extension(
        self, tmp_path, subject, fake_run, fs_home
    ):
        fake_run.creates = self.creates_volume()
        out = tmp_path / "aparc.mgz"
        subject.surf2vol("aparc", out, color_table="tsv")
        assert (tmp_path / "aparc.tsv").is_file()
        # The volume itself is not replaced by the table
        assert nib.load(str(out)).get_fdata().ravel().tolist() == [
            0,
            10,
            49,
            1001,
            2001,
        ]

    def test_white_matter_regions_when_growing(
        self, tmp_path, subject, fake_run, fs_home
    ):
        fake_run.creates = self.creates_volume()
        subject.surf2vol(
            "aparc", tmp_path / "aparc.nii.gz", gm_grow=2, color_table="tsv"
        )
        tsv = pd.read_csv(tmp_path / "aparc.tsv", sep="\t")
        index = dict(zip(tsv["name"], tsv["index"], strict=True))
        assert index["wm-lh-bankssts"] == 3000
        assert index["wm-rh-bankssts"] == 4000

    def test_color_table_formats_are_validated(
        self, tmp_path, subject, fake_run, fs_home
    ):
        fake_run.creates = self.creates_volume()
        with pytest.raises(ValueError, match="color_table must be a list or a string"):
            subject.surf2vol("aparc", tmp_path / "a.nii.gz", color_table=5)
        subject.surf2vol("aparc", tmp_path / "b.nii.gz", color_table=["csv"])
        assert (tmp_path / "b.lut").is_file()  # unknown formats fall back to lut

    def test_native_space_output(self, tmp_path, subject, fake_run, monkeypatch):
        fake_run.creates = self.creates_volume()
        calls = []
        monkeypatch.setattr(
            FreeSurferSubject,
            "conform2native",
            lambda self, **kwargs: calls.append(kwargs),
        )
        out = tmp_path / "aparc.nii.gz"
        subject.surf2vol("aparc", out, bool_native=True, overwrite=True)
        assert len(calls) == 1
        assert calls[0]["mgz_conform"] == str(out)
        assert calls[0]["nii_native"] == str(out)
        assert calls[0]["overwrite"] is True

    def test_white_matter_labels_are_mixed_with_the_gray_matter_ones(
        self, tmp_path, subject, fake_run
    ):
        fake_run.creates = self.creates_volume(values=(0, 10, 1001, 3001, 4002))
        out = tmp_path / "aparc.nii.gz"
        subject.surf2vol("aparc", out, gm_grow=2, bool_mixwm=True)
        assert nib.load(str(out)).get_fdata().ravel().tolist() == [
            0,
            10,
            1001,
            1001,
            2002,
        ]

    def test_containers_get_the_subjects_directory(
        self, tmp_path, subject, fake_run, monkeypatch
    ):
        monkeypatch.setattr(
            cltfree.cltmisc,
            "generate_container_command",
            lambda args, tech, image: ["docker", "run", *args],
        )
        fake_run.stdout = "/opt/subjects\n"
        fake_run.creates = self.creates_volume()
        subject.surf2vol(
            "aparc", tmp_path / "aparc.nii.gz", cont_tech="docker", cont_image="fs"
        )
        cmd = fake_run.commands[-1]
        assert cmd[:4] == ["docker", "run", "-v", f"{subject.subjs_dir}:/opt/subjects"]
        assert "mri_aparc2aseg" in cmd

    def test_failed_conversion(self, tmp_path, subject, fake_run):
        with pytest.raises(RuntimeError, match="mri_aparc2aseg did not create"):
            subject.surf2vol("aparc", tmp_path / "aparc.nii.gz")


class TestConformToNative:
    @pytest.fixture
    def subject(self, subjects):
        subj = make_touched_subject(subjects, "bert", 1)
        (subj / "tmp").mkdir()
        touch(subj / "mri" / "rawavg.mgz")
        return FreeSurferSubject("bert", subjects)

    @staticmethod
    def fake_commands(shape=(4, 4, 4)):
        def create(cmd):
            if cmd[0] == "mri_convert":
                out = cmd[cmd.index("-o") + 1]
            else:
                out = cmd[cmd.index("--o") + 1]
            nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.int16), np.eye(4)), out)
            return []

        return create

    def test_command(self, tmp_path, subject, fake_run):
        fake_run.creates = self.fake_commands()
        mgz = tmp_path / "T1.mgz"
        mgz.write_text("conform")
        out = tmp_path / "T1_native.nii.gz"
        subject.conform2native(mgz, out, interp_method="trilinear")
        raw = Path(subject.subjs_dir) / "bert" / "mri" / "rawavg.mgz"
        assert fake_run.commands[-1] == [
            "mri_vol2vol", "--mov", str(mgz), "--targ", str(raw), "--regheader",
            "--o", str(out), "--no-save-reg", "--interp", "trilinear",
        ]  # fmt: skip
        assert out.is_file()

    def test_temporary_volume_is_removed(self, tmp_path, subject, fake_run):
        fake_run.creates = self.fake_commands()
        mgz = tmp_path / "T1.mgz"
        mgz.write_text("conform")
        subject.conform2native(mgz, tmp_path / "native.nii.gz")
        assert not (Path(subject.subjs_dir) / "bert" / "tmp" / "rawavg.nii.gz").exists()

    def test_nearest_neighbour_is_the_default(self, tmp_path, subject, fake_run):
        fake_run.creates = self.fake_commands()
        mgz = tmp_path / "T1.mgz"
        mgz.write_text("conform")
        subject.conform2native(mgz, tmp_path / "native.nii.gz")
        assert fake_run.commands[-1][-2:] == ["--interp", "nearest"]

    def test_missing_temporary_folder_is_created(self, tmp_path, subject, fake_run):
        shutil.rmtree(Path(subject.subjs_dir) / "bert" / "tmp")
        fake_run.creates = self.fake_commands()
        mgz = tmp_path / "T1.mgz"
        mgz.write_text("conform")
        subject.conform2native(mgz, tmp_path / "native.nii.gz")
        assert (tmp_path / "native.nii.gz").is_file()

    def test_existing_output_with_the_native_size_is_kept(
        self, tmp_path, subject, fake_run, capsys
    ):
        fake_run.creates = self.fake_commands()
        mgz = tmp_path / "T1.mgz"
        mgz.write_text("conform")
        out = tmp_path / "native.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), np.eye(4)), str(out)
        )
        subject.conform2native(mgz, out)
        assert [c[0] for c in fake_run.commands] == [
            "mri_convert"
        ]  # only the size check
        assert "already exists and has the same dimensions" in capsys.readouterr().out

    def test_existing_output_with_another_size_is_replaced(
        self, tmp_path, subject, fake_run
    ):
        fake_run.creates = self.fake_commands()
        mgz = tmp_path / "T1.mgz"
        mgz.write_text("conform")
        out = tmp_path / "native.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.int16), np.eye(4)), str(out)
        )
        subject.conform2native(mgz, out)
        assert fake_run.commands[-1][0] == "mri_vol2vol"
        assert nib.load(str(out)).shape == (4, 4, 4)

    def test_overwrite(self, tmp_path, subject, fake_run):
        fake_run.creates = self.fake_commands()
        mgz = tmp_path / "T1.mgz"
        mgz.write_text("conform")
        out = tmp_path / "native.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), np.eye(4)), str(out)
        )
        subject.conform2native(mgz, out, overwrite=True)
        assert fake_run.commands[-1][0] == "mri_vol2vol"

    def test_missing_files(self, tmp_path, subject, fake_run):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            subject.conform2native(tmp_path / "missing.mgz", tmp_path / "native.nii.gz")
        (Path(subject.subjs_dir) / "bert" / "mri" / "rawavg.mgz").unlink()
        mgz = tmp_path / "T1.mgz"
        mgz.write_text("conform")
        with pytest.raises(FileNotFoundError, match="rawavg.mgz does not exist"):
            subject.conform2native(mgz, tmp_path / "native.nii.gz")


####################################################################################################
# Functions of the module
####################################################################################################
class TestCreateSubjectStatsTable:
    def test_table_of_a_subject(self, real_subject, subjects):
        df = cltfree.create_subject_stats_table(real_subject.name, subjects)
        assert isinstance(df, pd.DataFrame)
        assert set(df["Participant"]) == {"sub-01_ses-M00"}
        assert {"lh", "rh"} <= set(df["Hemisphere"])

    def test_table_is_saved_and_the_folders_created(
        self, tmp_path, real_subject, subjects
    ):
        out = tmp_path / "tables" / "sub-01" / "stats.csv"
        df = cltfree.create_subject_stats_table(
            real_subject.name, subjects, out_tab_file=out, add_bids_entities=True
        )
        saved = pd.read_csv(out)
        assert len(saved) == len(df)
        assert saved["Participant"].astype(str).str.zfill(2).unique().tolist() == ["01"]
        assert saved["Session"].unique().tolist() == ["M00"]

    def test_existing_table_is_loaded_without_computing_it(self, tmp_path):
        out = tmp_path / "table.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(out, index=False)
        df = cltfree.create_subject_stats_table(
            "ghost", tmp_path / "nowhere", out_tab_file=out
        )
        assert df.columns.tolist() == ["a", "b"]
        assert df["b"].tolist() == [3, 4]

    def test_existing_table_with_another_separator(self, tmp_path):
        out = tmp_path / "table.tsv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(out, sep="\t", index=False)
        df = cltfree.create_subject_stats_table("ghost", tmp_path, out_tab_file=out)
        assert df["a"].tolist() == [1, 2]

    def test_existing_table_is_replaced_when_overwriting(
        self, tmp_path, real_subject, subjects
    ):
        out = tmp_path / "table.csv"
        out.write_text("a,b\n1,2\n")
        df = cltfree.create_subject_stats_table(
            real_subject.name, subjects, out_tab_file=out, overwrite=True
        )
        assert "Region" in pd.read_csv(out).columns
        assert len(df) > 100

    def test_subjects_directory_from_the_environment(
        self, real_subject, subjects, monkeypatch
    ):
        monkeypatch.setenv("SUBJECTS_DIR", str(subjects))
        df = cltfree.create_subject_stats_table(real_subject.name)
        assert not df.empty

    def test_invalid_arguments(self, tmp_path, subjects, monkeypatch):
        monkeypatch.delenv("SUBJECTS_DIR", raising=False)
        with pytest.raises(ValueError, match="Pass subjs_dir or set"):
            cltfree.create_subject_stats_table("bert")
        with pytest.raises(ValueError, match="not a valid directory"):
            cltfree.create_subject_stats_table("bert", tmp_path / "missing")
        with pytest.raises(ValueError, match="subj_id should be a string"):
            cltfree.create_subject_stats_table(5, subjects)
        with pytest.raises(ValueError, match="does not exist in the directory"):
            cltfree.create_subject_stats_table("ghost", subjects)
        (subjects / "empty").mkdir()
        with pytest.raises(ValueError, match="not a valid FreeSurfer directory"):
            cltfree.create_subject_stats_table("empty", subjects)
        with pytest.raises(ValueError, match="out_tab_file should be"):
            cltfree.create_subject_stats_table("bert", subjects, out_tab_file=5)


class TestProcessSubject:
    @pytest.fixture
    def recorder(self, monkeypatch):
        calls = []

        def fake(
            subj_id,
            subjs_dir,
            out_tab_file=None,
            add_bids_entities=False,
            overwrite=False,
        ):
            calls.append(
                (subj_id, subjs_dir, out_tab_file, add_bids_entities, overwrite)
            )
            Path(out_tab_file).parent.mkdir(parents=True, exist_ok=True)
            Path(out_tab_file).write_text("table")

        monkeypatch.setattr(cltfree, "create_subject_stats_table", fake)
        return calls

    def test_subject_with_a_session(self, tmp_path, recorder):
        result = cltfree.process_subject("sub-01_ses-M00", "/fs", tmp_path)
        expected = (
            tmp_path
            / "sub-01"
            / "ses-M00"
            / "stats"
            / "sub-01_ses-M00_desc-statstable_morphometry.csv"
        )
        assert result == ("sub-01_ses-M00", True, None)
        assert recorder == [("sub-01_ses-M00", "/fs", str(expected), True, False)]
        assert expected.is_file()

    def test_subject_without_a_session(self, tmp_path, recorder):
        cltfree.process_subject("sub-02", "/fs", tmp_path)
        expected = (
            tmp_path / "sub-02" / "stats" / "sub-02_desc-statstable_morphometry.csv"
        )
        assert recorder[0][2] == str(expected)
        assert recorder[0][3] is True

    def test_subject_that_is_not_bids(self, tmp_path, recorder):
        result = cltfree.process_subject("bert", "/fs", tmp_path)
        expected = tmp_path / "bert" / "stats" / "bert_desc-statstable_morphometry.csv"
        assert result == ("bert", True, None)
        assert recorder[0][2] == str(expected)
        assert recorder[0][3] is False

    def test_existing_table_is_skipped(self, tmp_path, recorder):
        cltfree.process_subject("bert", "/fs", tmp_path)
        recorder.clear()
        result = cltfree.process_subject("bert", "/fs", tmp_path)
        assert result == ("bert", True, "File already exists - skipped")
        assert recorder == []

    def test_existing_table_is_replaced_when_overwriting(self, tmp_path, recorder):
        cltfree.process_subject("bert", "/fs", tmp_path)
        result = cltfree.process_subject("bert", "/fs", tmp_path, overwrite=True)
        assert result == ("bert", True, "Overwritten")
        assert recorder[-1][4] is True

    def test_errors_are_returned(self, tmp_path, monkeypatch):
        def broken(*args, **kwargs):
            raise ValueError("no such subject")

        monkeypatch.setattr(cltfree, "create_subject_stats_table", broken)
        assert cltfree.process_subject("bert", "/fs", tmp_path) == (
            "bert", False, "ValueError: no such subject",
        )  # fmt: skip

    def test_real_subject(self, tmp_path, real_subject, subjects):
        out = tmp_path / "out"
        result = cltfree.process_subject(real_subject.name, subjects, out)
        assert result == (real_subject.name, True, None)
        table = (
            out
            / "sub-01"
            / "ses-M00"
            / "stats"
            / f"{real_subject.name}_desc-statstable_morphometry.csv"
        )
        assert table.is_file()


class TestCreateCohortStatsTable:
    @pytest.fixture
    def processed(self, monkeypatch):
        """Replace the per-subject work by a recorder with chosen outcomes."""
        state = types_namespace(calls=[], outcomes={})

        def fake(fs_fullid, fs_subject_dir, out_folder, overwrite=False):
            state.calls.append((fs_fullid, fs_subject_dir, out_folder, overwrite))
            outcome = state.outcomes.get(fs_fullid, (fs_fullid, True, None))
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(cltfree, "process_subject", fake)
        return state

    def test_list_of_subjects(self, tmp_path, subjects, processed, capsys):
        cltfree.create_cohort_stats_table(
            tmp_path / "out", ["sub-01", "sub-02"], subjects
        )
        assert sorted(call[0] for call in processed.calls) == ["sub-01", "sub-02"]
        assert all(call[2] == str(tmp_path / "out") for call in processed.calls)
        out = capsys.readouterr().out
        assert "Total subjects to process: 2" in out
        assert "Created: 2" in out

    def test_file_of_subjects(self, tmp_path, subjects, processed):
        ids = tmp_path / "ids.txt"
        ids.write_text("sub-01\n\nbert\n")
        cltfree.create_cohort_stats_table(tmp_path / "out", str(ids), subjects)
        assert sorted(call[0] for call in processed.calls) == ["bert", "sub-01"]

    def test_all_the_bids_folders_by_default(self, tmp_path, subjects, processed):
        for name in ("sub-02", "sub-01", "bert"):
            (subjects / name).mkdir()
        (subjects / "sub-03.html").write_text("report")
        cltfree.create_cohort_stats_table(tmp_path / "out", fs_subject_dir=subjects)
        assert sorted(call[0] for call in processed.calls) == ["sub-01", "sub-02"]

    def test_subjects_directory_from_the_environment(
        self, tmp_path, subjects, processed, monkeypatch
    ):
        monkeypatch.setenv("SUBJECTS_DIR", str(subjects))
        cltfree.create_cohort_stats_table(tmp_path / "out", ["bert"])
        assert processed.calls[0][1] == str(subjects)

    def test_overwrite_option(self, tmp_path, subjects, processed):
        cltfree.create_cohort_stats_table(
            tmp_path / "out", ["bert"], subjects, overwrite=True
        )
        assert processed.calls[0][3] is True

    def test_summary_of_every_outcome(self, tmp_path, subjects, processed, capsys):
        processed.outcomes = {
            "a": ("a", True, None),
            "b": ("b", True, "Overwritten"),
            "c": ("c", True, "File already exists - skipped"),
            "d": ("d", False, "ValueError: broken"),
            "e": RuntimeError("worker crashed"),
        }
        cltfree.create_cohort_stats_table(
            tmp_path / "out", ["a", "b", "c", "d", "e"], subjects, max_workers=2
        )
        out = capsys.readouterr().out
        assert "Created: 1" in out
        assert "Overwritten: 1" in out
        assert "Skipped (already existed): 1" in out
        assert "Failed: 2" in out
        assert "d: ValueError: broken" in out
        assert "e: worker crashed" in out

    def test_missing_subjects_directory(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SUBJECTS_DIR", raising=False)
        with pytest.raises(ValueError, match="Pass fs_subject_dir or set"):
            cltfree.create_cohort_stats_table(tmp_path, ["bert"])
        with pytest.raises(ValueError, match="does not exist"):
            cltfree.create_cohort_stats_table(tmp_path, ["bert"], tmp_path / "missing")

    def test_real_subjects(self, tmp_path, real_subject, subjects):
        out = tmp_path / "out"
        cltfree.create_cohort_stats_table(
            out, [real_subject.name], subjects, max_workers=2
        )
        tables = list(out.rglob("*_desc-statstable_morphometry.csv"))
        assert len(tables) == 1
        assert not pd.read_csv(tables[0]).empty


def types_namespace(**kwargs):
    import types

    return types.SimpleNamespace(**kwargs)


@pytest.mark.skipif(os.name == "nt", reason="Symbolic links need privileges on Windows")
class TestFsaverageLinks:
    @pytest.fixture
    def template(self, tmp_path):
        folder = tmp_path / "templates" / "fsaverage"
        folder.mkdir(parents=True)
        (folder / "marker.txt").write_text("fsaverage")
        return folder

    def test_link_to_a_template(self, subjects, template):
        link = cltfree.create_fsaverage_links(
            subjects, fsavg_dir=template, refsubj_name="fsaverage"
        )
        assert link == str(subjects / "fsaverage")
        assert os.path.islink(link)
        assert os.path.realpath(link) == str(template.resolve())
        assert (Path(link) / "marker.txt").read_text() == "fsaverage"

    def test_default_name(self, subjects, template):
        link = cltfree.create_fsaverage_links(subjects, fsavg_dir=template)
        assert link == str(subjects / "fsaverage")
        assert os.path.islink(link)

    def test_template_folder_without_its_own_name(self, tmp_path, subjects):
        parent = tmp_path / "templates"
        (parent / "fsaverage6").mkdir(parents=True)
        link = cltfree.create_fsaverage_links(
            subjects, fsavg_dir=str(parent) + os.sep, refsubj_name="fsaverage6"
        )
        assert os.path.realpath(link) == str((parent / "fsaverage6").resolve())

    def test_template_from_the_freesurfer_installation(
        self, tmp_path, subjects, monkeypatch
    ):
        home = tmp_path / "freesurfer"
        (home / "subjects" / "fsaverage6").mkdir(parents=True)
        monkeypatch.setenv("FREESURFER_HOME", str(home))
        link = cltfree.create_fsaverage_links(subjects, refsubj_name="fsaverage6")
        assert os.path.realpath(link) == str(
            (home / "subjects" / "fsaverage6").resolve()
        )

    def test_existing_link_is_returned(self, subjects, template):
        first = cltfree.create_fsaverage_links(subjects, fsavg_dir=template)
        second = cltfree.create_fsaverage_links(subjects, fsavg_dir=template)
        assert first == second

    def test_existing_folder_is_returned(self, subjects, template):
        (subjects / "fsaverage").mkdir()
        assert cltfree.create_fsaverage_links(subjects, fsavg_dir=template) == str(
            subjects / "fsaverage"
        )
        assert not os.path.islink(subjects / "fsaverage")

    def test_existing_file(self, subjects, template):
        (subjects / "fsaverage").write_text("not a folder")
        with pytest.raises(ValueError, match="already exists and is not a directory"):
            cltfree.create_fsaverage_links(subjects, fsavg_dir=template)

    def test_missing_folders(self, tmp_path, subjects, template):
        with pytest.raises(ValueError, match="FreeSurfer directory does not exist"):
            cltfree.create_fsaverage_links(tmp_path / "missing", fsavg_dir=template)
        with pytest.raises(ValueError, match="fsaverage directory does not exist"):
            cltfree.create_fsaverage_links(
                subjects, fsavg_dir=tmp_path / "nothing" / "fsaverage"
            )

    def test_removing_a_link(self, subjects, template):
        link = cltfree.create_fsaverage_links(subjects, fsavg_dir=template)
        cltfree.remove_fsaverage_links(link)
        assert not os.path.lexists(link)
        assert (template / "marker.txt").is_file()

    def test_removing_a_link_without_a_freesurfer_installation(
        self, subjects, template, monkeypatch
    ):
        monkeypatch.delenv("FREESURFER_HOME", raising=False)
        link = cltfree.create_fsaverage_links(subjects, fsavg_dir=template)
        cltfree.remove_fsaverage_links(Path(link))
        assert not os.path.lexists(link)

    def test_the_installed_fsaverage_is_never_removed(
        self, tmp_path, subjects, monkeypatch
    ):
        home = tmp_path / "freesurfer"
        (home / "subjects" / "fsaverage").mkdir(parents=True)
        monkeypatch.setenv("FREESURFER_HOME", str(home))
        link = cltfree.create_fsaverage_links(subjects)
        cltfree.remove_fsaverage_links(link)
        assert os.path.islink(link)

    def test_the_installed_fsaverage_is_found_through_symbolic_links(
        self, tmp_path, subjects, monkeypatch
    ):
        home = tmp_path / "freesurfer"
        (home / "subjects" / "fsaverage").mkdir(parents=True)
        home_link = tmp_path / "freesurfer_link"
        os.symlink(home, home_link)
        monkeypatch.setenv("FREESURFER_HOME", str(home_link))
        link = cltfree.create_fsaverage_links(subjects)
        cltfree.remove_fsaverage_links(link)
        assert os.path.islink(link)

    def test_real_folders_are_never_removed(self, subjects, template):
        (subjects / "fsaverage").mkdir()
        cltfree.remove_fsaverage_links(subjects / "fsaverage")
        assert (subjects / "fsaverage").is_dir()

    def test_missing_path(self, tmp_path):
        cltfree.remove_fsaverage_links(tmp_path / "nothing")


class TestVertexwiseValues:
    TABLE = np.array(
        [[255, 0, 0, 0, RED], [0, 255, 0, 0, GREEN], [0, 0, 255, 0, BLUE]],
        dtype=np.int32,
    )
    LABELS = np.array([RED, GREEN, BLUE, GREEN, 0, RED])

    def test_one_dimensional_values(self):
        result = cltfree.region_to_vertexwise(
            np.array([1.0, 2.0, 3.0]), self.LABELS, self.TABLE
        )
        assert result.shape == (6, 1)
        assert result[:, 0].tolist() == [1.0, 2.0, 3.0, 2.0, 0.0, 1.0]

    def test_two_dimensional_values(self):
        values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        result = cltfree.region_to_vertexwise(values, self.LABELS, self.TABLE)
        assert result.shape == (6, 2)
        np.testing.assert_array_equal(result[3], [2.0, 20.0])
        np.testing.assert_array_equal(result[4], [0.0, 0.0])

    def test_single_precision_output(self):
        result = cltfree.region_to_vertexwise(
            np.array([1, 2, 3]), self.LABELS, self.TABLE
        )
        assert result.dtype == np.float32

    def test_regions_without_vertices(self):
        result = cltfree.region_to_vertexwise(
            np.array([1.0, 2.0, 3.0]), np.array([RED, RED]), self.TABLE
        )
        assert result[:, 0].tolist() == [1.0, 1.0]

    def test_input_is_not_modified(self):
        values = np.array([1.0, 2.0, 3.0])
        cltfree.region_to_vertexwise(values, self.LABELS, self.TABLE)
        assert values.shape == (3,)

    def test_invalid_shapes(self):
        with pytest.raises(ValueError, match="number of rows in reg_values"):
            cltfree.region_to_vertexwise(np.array([1.0, 2.0]), self.LABELS, self.TABLE)
        with pytest.raises(ValueError, match="1D or 2D"):
            cltfree.region_to_vertexwise(np.zeros((3, 2, 2)), self.LABELS, self.TABLE)

    def test_vertex_colors(self):
        colors = cltfree.create_vertex_colors(self.LABELS, self.TABLE)
        assert colors.shape == (6, 4)
        np.testing.assert_array_equal(
            colors[:3, :3], [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        )
        np.testing.assert_array_equal(
            colors[4, :3], [240, 240, 240]
        )  # unlabeled vertex
        np.testing.assert_array_equal(colors[5, :3], [255, 0, 0])

    def test_vertex_colors_need_a_complete_table(self):
        with pytest.raises(ValueError, match="5 columns"):
            cltfree.create_vertex_colors(self.LABELS, self.TABLE[:, :4])

    def test_vertex_colors_of_an_annotation(self, annot):
        colors = cltfree.create_vertex_colors(annot.codes, annot.regtable)
        assert colors.shape == (annot.codes.size, 4)
        first = np.flatnonzero(annot.codes == annot.regtable[0, 4])[0]
        np.testing.assert_array_equal(colors[first, :3], annot.regtable[0, :3])


class TestColorTables:
    def test_hex_colors(self):
        table = cltfree.colors2colortable(["#FF0000", "#00FF00", "#0000FF"])
        assert table.dtype == np.uint32
        assert table.tolist() == [
            [255, 0, 0, 0, RED],
            [0, 255, 0, 0, GREEN],
            [0, 0, 255, 0, BLUE],
        ]

    def test_rgb_array(self):
        table = cltfree.colors2colortable(np.array([[255, 0, 0], [0, 255, 0]]))
        assert table.shape == (2, 5)
        assert table[:, 4].tolist() == [RED, GREEN]

    def test_colors_between_0_and_1(self):
        table = cltfree.colors2colortable(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        assert table[:, 4].tolist() == [RED, GREEN]

    @pytest.mark.parametrize("bad", ["abc", 5, None])
    def test_invalid_colors(self, bad):
        with pytest.raises(ValueError, match="list or a numpy array"):
            cltfree.colors2colortable(bad)

    def test_verification_of_a_correct_table(self):
        table = cltfree.colors2colortable(["#123456", "#abcdef"])
        assert cltfree.verify_packed_rgb_values(table) == (True, [])

    def test_verification_reports_the_wrong_entries(self):
        table = cltfree.colors2colortable(["#123456", "#abcdef", "#ff0000"]).astype(
            np.int64
        )
        table[1, 4] = 7
        ok, errors = cltfree.verify_packed_rgb_values(table)
        assert ok is False
        assert errors == [
            {
                "index": 1,
                "rgb": [171, 205, 239],
                "expected_packed": 171 + 205 * 256 + 239 * 65536,
                "actual_packed": 7,
            }
        ]

    def test_verification_of_an_empty_table(self):
        assert cltfree.verify_packed_rgb_values(np.zeros((0, 5))) == (True, [])


class TestColorTableDuplicates:
    @staticmethod
    def packed(rgb):
        return rgb[0] + rgb[1] * 256 + rgb[2] * 65536

    @pytest.fixture
    def table(self):
        rows = [(255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0)]
        return np.array([[*rgb, 0, self.packed(rgb)] for rgb in rows], dtype=float)

    def test_duplicates_are_resolved(self, table):
        fixed, log, mapping = cltfree.resolve_colortable_duplicates(table)
        colors = {tuple(row[:3].astype(int)) for row in fixed}
        assert len(colors) == 5
        assert fixed[:2].tolist() == table[:2].tolist()  # first occurrences are kept
        assert fixed[3].tolist() == table[3].tolist()

    def test_blue_is_changed_first(self, table):
        fixed, log, _ = cltfree.resolve_colortable_duplicates(table)
        assert fixed[2, :3].tolist() == [255, 0, 1]
        assert fixed[4, :3].tolist() == [0, 255, 1]
        assert [m["channel_modified"] for m in log["modifications_made"]] == ["B", "B"]
        assert [m["increment_applied"] for m in log["modifications_made"]] == [1, 1]

    def test_packed_values_follow_the_colors(self, table):
        fixed, _, _ = cltfree.resolve_colortable_duplicates(table)
        assert cltfree.verify_packed_rgb_values(fixed) == (True, [])

    def test_log(self, table):
        _, log, _ = cltfree.resolve_colortable_duplicates(table)
        assert log["original_duplicates"] == 2
        assert log["final_unique_colors"] == 5
        assert [m["index"] for m in log["modifications_made"]] == [2, 4]
        assert log["modifications_made"][0]["original_rgb"] == [255, 0, 0]
        assert log["modifications_made"][0]["new_rgb"] == [255, 0, 1]
        assert log["modifications_made"][0]["new_packed_value"] == 255 + 65536

    def test_mapping_of_the_packed_values(self, table):
        _, _, mapping = cltfree.resolve_colortable_duplicates(table)
        assert mapping["old_values"].tolist() == [RED, GREEN]
        assert mapping["new_values"].tolist() == [RED + 65536, GREEN + 65536]

    def test_the_input_is_not_modified(self, table):
        before = table.copy()
        cltfree.resolve_colortable_duplicates(table)
        np.testing.assert_array_equal(table, before)

    def test_table_without_duplicates(self):
        table = cltfree.colors2colortable(["#ff0000", "#00ff00"]).astype(np.int64)
        fixed, log, mapping = cltfree.resolve_colortable_duplicates(table)
        np.testing.assert_array_equal(fixed, table)
        assert log == {
            "original_duplicates": 0,
            "modifications_made": [],
            "final_unique_colors": 2,
        }
        assert mapping["old_values"].size == 0

    def test_three_copies_of_a_color(self):
        rows = [(10, 20, 30)] * 3
        table = np.array([[*rgb, 0, self.packed(rgb)] for rgb in rows])
        fixed, log, _ = cltfree.resolve_colortable_duplicates(table)
        assert fixed[:, :3].tolist() == [[10, 20, 30], [10, 20, 31], [10, 20, 32]]
        assert log["final_unique_colors"] == 3

    def test_a_blue_channel_at_the_maximum_is_decreased(self):
        rows = [(1, 2, 255), (1, 2, 255)]
        table = np.array([[*rgb, 0, self.packed(rgb)] for rgb in rows])
        fixed, _, _ = cltfree.resolve_colortable_duplicates(table)
        assert fixed[1, :3].tolist() == [1, 2, 254]

    def test_unique_color_search(self):
        used = {(10, 20, 30): [0], (10, 20, 31): [1]}
        assert cltfree.find_unique_color(np.array([10, 20, 30]), used).tolist() == [
            10,
            20,
            32,
        ]

    def test_unique_color_search_skips_taken_colors(self):
        used = {(10, 20, 255): [0], (10, 20, 254): [1], (10, 20, 253): [2]}
        assert cltfree.find_unique_color(np.array([10, 20, 255]), used).tolist() == [
            10,
            20,
            252,
        ]

    def test_unique_color_search_changes_green_when_blue_is_exhausted(self):
        used = {(10, 20, b): [b] for b in range(256)}
        assert cltfree.find_unique_color(np.array([10, 20, 30]), used).tolist() == [
            10,
            21,
            30,
        ]


class TestDetectHemi:
    @pytest.mark.parametrize(
        "name, hemi",
        [
            ("lh.aparc.annot", "lh"),
            ("rh.pial", "rh"),
            ("lh.thickness", "lh"),
            ("rh.aparc.a2009s.annot", "rh"),
            ("sub-01_hemi-L_pial.surf.gii", "lh"),
            ("sub-01_hemi-R_thickness.shape.gii", "rh"),
            ("sub-01_hemi-left_T1w.nii.gz", "lh"),
            ("sub-01_hemi-right_T1w.nii.gz", "rh"),
            ("sub-01_hemi-lh_pial.gii", "lh"),
        ],
    )
    def test_names(self, name, hemi):
        assert cltfree.detect_hemi(name) == hemi

    def test_paths(self):
        assert cltfree.detect_hemi("/data/rh.subjects/sub-01/surf/lh.pial") == "lh"
        assert cltfree.detect_hemi(Path("/data/subjects/rh.white")) == "rh"

    @pytest.mark.parametrize(
        "name", ["sub-01_T1w.nii.gz", "atlas.annot", "sub-01_hemi-X_pial.gii"]
    )
    def test_names_without_a_hemisphere(self, name):
        with pytest.warns(UserWarning, match="hemisphere could not be extracted"):
            assert cltfree.detect_hemi(name) is None

    def test_no_warning_when_the_hemisphere_is_found(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cltfree.detect_hemi("lh.aparc.annot")


class TestLtaFiles:
    @pytest.fixture
    def lta(self, tmp_path):
        path = tmp_path / "talairach.lta"
        path.write_text(TALAIRACH_LTA)
        return path

    def test_parsed_content(self, lta):
        parsed = cltfree.parse_freesurfer_lta(lta)
        assert set(parsed) == {
            "src_cras",
            "dst_cras",
            "transform_matrix",
            "src_volume_info",
            "dst_volume_info",
        }
        assert parsed["src_cras"] == pytest.approx((5.39971923828125, 18.0, 0.0))
        assert parsed["dst_cras"] == (1.0, 2.0, 3.0)
        assert parsed["src_volume_info"] == {
            "filename": "nu.mgz",
            "volume": [256, 256, 256],
        }
        assert parsed["dst_volume_info"]["volume"] == [256, 256, 256]
        assert parsed["dst_volume_info"]["filename"].endswith("RB_all_2020-01-02.gca")

    def test_transform_matrix(self, lta):
        matrix = cltfree.parse_freesurfer_lta(str(lta))["transform_matrix"]
        assert matrix.shape == (4, 4)
        assert matrix[0, 0] == pytest.approx(1.103049755096436)
        assert matrix[1, 3] == pytest.approx(-16.49753761291504)
        assert matrix[3].tolist() == [0.0, 0.0, 0.0, 1.0]

    def test_file_without_the_sections(self, tmp_path):
        path = tmp_path / "empty.lta"
        path.write_text("# nothing here\n")
        parsed = cltfree.parse_freesurfer_lta(path)
        assert parsed["src_cras"] is None
        assert parsed["dst_cras"] is None
        assert parsed["transform_matrix"] is None
        assert parsed["src_volume_info"] == {}

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltfree.parse_freesurfer_lta(tmp_path / "missing.lta")

    def test_cras_coordinates(self, lta):
        assert cltfree.get_cras_coordinates(lta) == pytest.approx(
            (5.39971923828125, 18.0, 0.0)
        )
        assert cltfree.get_cras_coordinates(
            lta, source=True
        ) == cltfree.get_cras_coordinates(lta)
        assert cltfree.get_cras_coordinates(lta, source=False) == (1.0, 2.0, 3.0)


class TestLobesJson:
    def test_default_definitions(self):
        lobes = cltfree.load_lobes_json()
        assert {"desikan", "desikan_w_cingulate"} <= set(lobes)
        assert list(lobes["desikan"]["lobes"]) == [
            "frontal",
            "parietal",
            "temporal",
            "occipital",
            "insula",
        ]
        assert "precentral" in lobes["desikan"]["lobes"]["frontal"]
        assert set(lobes["desikan"]["colors"]) == set(lobes["desikan"]["lobes"])

    def test_every_desikan_region_is_in_one_lobe(self):
        lobes = cltfree.load_lobes_json()["desikan"]["lobes"]
        members = [region for regions in lobes.values() for region in regions]
        assert sorted(members) == sorted(DESIKAN)

    def test_custom_file(self, tmp_path):
        path = tmp_path / "custom.json"
        path.write_text(json.dumps({"lobes": {"a": ["x"]}}))
        assert cltfree.load_lobes_json(path) == {"lobes": {"a": ["x"]}}
        assert cltfree.load_lobes_json(str(path)) == {"lobes": {"a": ["x"]}}

    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="valid JSON file"):
            cltfree.load_lobes_json(tmp_path / "missing.json")


class TestGetVersion:
    @pytest.mark.parametrize(
        "output, version",
        [
            ("freesurfer-linux-ubuntu22_x86_64-7.4.1-20230613-7eb8460\n", "7.4.1"),
            ("freesurfer-darwin-macOS_x86_64-7.2.0-20210721-aa8f76b\n", "7.2.0"),
            ("freesurfer-linux-centos7_x86_64-7.1.1-20200723-8b40551", "7.1.1"),
            ("freesurfer-linux-ubuntu20_x86_64-v8.0.0-20250101-abcdef", "8.0.0"),
        ],
    )
    def test_versions(self, fake_run, output, version):
        fake_run.stdout = output
        assert cltfree.get_version() == version
        assert fake_run.commands == [["recon-all", "-version"]]

    def test_dots_without_numbers_are_not_a_version(self, fake_run):
        fake_run.stdout = "freesurfer-linux.build-dev"
        assert cltfree.get_version() is None

    def test_unknown_version(self, fake_run):
        fake_run.stdout = "freesurfer-linux-centos7_x86_64-dev-20240101-abcdef"
        assert cltfree.get_version() is None

    def test_no_output(self, fake_run):
        fake_run.stdout = ""
        assert cltfree.get_version() is None

    def test_container(self, fake_run, monkeypatch):
        monkeypatch.setattr(
            cltfree.cltmisc,
            "generate_container_command",
            lambda args, tech, image: ["docker", "run", image, *args],
        )
        fake_run.stdout = "freesurfer-linux-ubuntu22_x86_64-7.4.1-20230613-7eb8460"
        assert cltfree.get_version("docker", "fs:7.4.1") == "7.4.1"
        assert fake_run.commands == [
            ["docker", "run", "fs:7.4.1", "recon-all", "-version"]
        ]


@needs_fsaverage
class TestInstalledFreesurfer:
    def test_version(self):
        version = cltfree.get_version()
        assert version is None or all(part.isdigit() for part in version.split("."))

    def test_installed_fsaverage_subject(self):
        subject = FreeSurferSubject("fsaverage", FSAVERAGE.parent)
        assert subject.fs_files["surf"]["lh"]["parc"]["desikan"].endswith(
            "lh.aparc.annot"
        )
        assert subject.fs_files_count["sparc"] >= 2
