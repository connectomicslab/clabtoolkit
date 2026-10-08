"""Tests for clabtoolkit.parcellationtools."""

import copy
import warnings
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import pytest

import clabtoolkit
from clabtoolkit.parcellationtools import Parcellation, RegionTimeSeries

DATA_DIR = Path(clabtoolkit.__file__).parent / "data" / "parcellationtools"
REAL_PARC = (
    DATA_DIR
    / "sub-test2_ses-01_atlas-chimeraLFIIHISIFN_scale-1_desc-grow1mm_dseg.nii.gz"
)


####################################################################################################
# Fixtures
####################################################################################################
@pytest.fixture(scope="session")
def real_parc_session():
    """The Chimera parcellation shipped with the package (loaded once, ~7 s)."""
    return Parcellation(REAL_PARC)


@pytest.fixture
def real_parc(real_parc_session):
    """A copy of the real parcellation that a test can modify."""
    return copy.deepcopy(real_parc_session)


@pytest.fixture
def sim():
    """Small simulated parcellation with 5 regions (labels 1-5, no background)."""
    return Parcellation.simulate_parcellation(
        n_regions=5, dimensions=(12, 12, 10), seed=1
    )


@pytest.fixture
def named():
    """
    Hand-made 10 x 10 x 10 parcellation with named regions and background:
    1 ctx-lh-a, 2 ctx-lh-b, 3 ctx-rh-a (cubes of 3 x 3 x 3 voxels), 4 wm-lh-a (2 voxels).
    """
    data = np.zeros((10, 10, 10), dtype=np.int32)
    data[0:3, 0:3, 0:3] = 1
    data[3:6, 0:3, 0:3] = 2  # Touches region 1
    data[7:10, 7:10, 7:10] = 3
    data[0, 4, 0] = data[0, 5, 0] = 4
    ctab = {
        "index": [1, 2, 3, 4],
        "name": ["ctx-lh-a", "ctx-lh-b", "ctx-rh-a", "wm-lh-a"],
        "color": ["#ff0000", "#00ff00", "#0000ff", "#ffff00"],
        "opacity": [1.0, 0.5, 1.0, 1.0],
    }
    return Parcellation(data, color_table=ctab, affine=np.diag([2.0, 2.0, 2.0, 1.0]))


####################################################################################################
# Creation and attributes
####################################################################################################
class TestCreation:
    def test_from_array(self, named):
        assert named.index == [1, 2, 3, 4]
        assert named.name[0] == "ctx-lh-a" and named.color[1] == "#00ff00"
        assert named.opacity == [1.0, 0.5, 1.0, 1.0]
        assert named.parc_file == "numpy_array" and named.id == "numpy_array"
        assert named.dim == (10, 10, 10) and named.voxel_volume == pytest.approx(8.0)
        assert (named.minlab, named.maxlab) == (1, 4)

    def test_from_array_defaults(self):
        data = np.zeros((4, 4, 4), dtype=np.int16)
        data[0, 0, 0], data[1, 1, 1] = 3, 7
        parc = Parcellation(data, parc_id="custom", space_id="native")
        assert parc.index == [3, 7] and parc.name == [
            "auto-roi-000003",
            "auto-roi-000007",
        ]
        assert len(parc.color) == 2 and parc.opacity == [1.0, 1.0]
        assert parc.id == "custom" and parc.space == "native"
        assert np.allclose(parc.affine[:3, 3], [-2, -2, -2])  # Centered identity affine

    def test_colortable_dict_without_colors(self):
        data = np.array([[[1, 2]]], dtype=np.int32)
        parc = Parcellation(data, color_table={"index": [1, 2], "name": ["a", "b"]})
        assert parc.name == ["a", "b"] and len(parc.color) == 2

    def test_colortable_entries_not_in_data_are_dropped(self):
        data = np.array([[[1, 2]]], dtype=np.int32)
        ctab = {"index": [1, 2, 9], "name": ["a", "b", "z"], "color": ["#ff0000"] * 3}
        assert Parcellation(data, color_table=ctab).index == [1, 2]

    def test_from_file(self, real_parc_session):
        parc = real_parc_session
        assert parc.dim == (256, 392, 416)
        assert len(parc.index) == 277 and parc.index == sorted(parc.index)
        assert parc.name[0] == "ctx-rh-lateralorbitofrontal"
        assert parc.id == "atlas-chimeraLFIIHISIFN_scale-1_desc-grow1mm"
        assert parc.space == "unknown"  # No space entity in the file name
        assert parc.lut_file.endswith(".lut")  # Sidecar LUT found automatically
        assert all(c.startswith("#") for c in parc.color)

    def test_space_from_file_name_and_argument(self, tmp_path, sim):
        out = tmp_path / "sub-01_space-MNI_atlas-sim_dseg.nii.gz"
        sim.save_parcellation(out)
        assert Parcellation(out).space == "MNI"
        assert Parcellation(out, space_id="native").space == "native"

    def test_explicit_color_table_file(self, tmp_path, named):
        lut = tmp_path / "custom.tsv"
        named.export_colortable(str(lut), lut_type="tsv")
        named.save_parcellation(tmp_path / "parc.nii.gz", lut_type=None)
        parc = Parcellation(tmp_path / "parc.nii.gz", color_table=lut)
        assert parc.name == ["ctx-lh-a", "ctx-lh-b", "ctx-rh-a", "wm-lh-a"]

    def test_errors(self, tmp_path, named):
        with pytest.raises(ValueError):
            Parcellation(None)
        with pytest.raises(ValueError, match="does not exist"):
            Parcellation(tmp_path / "missing.nii.gz")
        with pytest.raises(TypeError):
            Parcellation(5)
        named.save_parcellation(tmp_path / "parc.nii.gz", lut_type=None)
        with pytest.raises(FileNotFoundError):
            Parcellation(tmp_path / "parc.nii.gz", color_table=tmp_path / "missing.lut")
        with pytest.raises(ValueError):
            Parcellation(named.data, color_table={"index": [1]})  # No names

    def test_simulate_parcellation(self):
        parc = Parcellation.simulate_parcellation(
            n_regions=6, dimensions=8, voxel_size=2.0, seed=3
        )
        assert parc.dim == (8, 8, 8) and parc.index == [1, 2, 3, 4, 5, 6]
        assert parc.space == "simulated" and parc.voxel_volume == pytest.approx(8.0)
        again = Parcellation.simulate_parcellation(n_regions=6, dimensions=8, seed=3)
        assert np.array_equal(parc.data, again.data) and parc.color == again.color
        with pytest.warns(UserWarning, match="voxel_size"):
            Parcellation.simulate_parcellation(
                2, dimensions=4, voxel_size=2.0, affine=np.eye(4)
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_regions": 0},
            {"n_regions": 2, "dimensions": (4, 4)},
            {"n_regions": 2, "voxel_size": -1.0},
            {"n_regions": 2, "affine": np.eye(3)},
        ],
    )
    def test_simulate_parcellation_errors(self, kwargs):
        with pytest.raises(ValueError):
            Parcellation.simulate_parcellation(**kwargs)


class TestAccessors:
    def test_getters(self, named):
        assert named.get_data() is named.data
        assert named.get_affine() is named.affine
        assert named.get_index() == [1, 2, 3, 4]
        assert named.get_names()[3] == "wm-lh-a"
        assert named.get_colors() == ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]

    def test_setters(self, named):
        named.set_names(["a", "b", "c", "d"])
        named.set_colors([[255, 0, 0], "#00FF00", (0, 0, 255), "#ffff00"])
        named.set_index([1, 2, 3, 4])
        named.set_affine(np.eye(4))
        named.set_data(named.data.copy())
        assert named.name[0] == "a" and named.color[0] == "#ff0000"
        assert named.color[2] == "#0000ff"

    def test_setter_errors(self, named):
        with pytest.raises(TypeError):
            named.set_data([1, 2])
        with pytest.raises(TypeError):
            named.set_affine([[1]])
        with pytest.raises(ValueError):
            named.set_affine(np.eye(3))
        with pytest.raises(TypeError):
            named.set_index([1.5])
        with pytest.raises(TypeError):
            named.set_names([1])
        with pytest.raises(TypeError):
            named.set_colors("#ff0000")

    def test_ids(self, tmp_path, sim):
        out = (
            tmp_path
            / "sub-01_ses-01_space-T1w_atlas-xxx_seg-yyy_scale-1_desc-test_dseg.nii.gz"
        )
        sim.save_parcellation(out)
        parc = Parcellation(out)
        assert parc.get_parcellation_id() == "atlas-xxx_seg-yyy_scale-1_desc-test"
        assert parc.get_space_id() == "T1w"
        assert parc.set_space_id("MNI") == "MNI" and parc.space == "MNI"
        plain = tmp_path / "custom_parcellation.nii.gz"
        sim.save_parcellation(plain)
        assert Parcellation(plain).id == "custom_parcellation"

    def test_get_info(self, named, capsys):
        named.index = named.index + [9]
        named.name = named.name + ["unused"]
        named.color = named.color + ["#000000"]
        named.opacity = named.opacity + [1.0]
        info = named.get_info()
        out = capsys.readouterr().out
        assert "PARCELLATION INFO" in out and "unused" in out
        assert info["n_regions"] == 5 and info["regions_not_in_data"] == ["unused"]
        assert info["labels_not_in_table"] == [] and info["opacity_range"] == (0.5, 1.0)
        assert named.get_info(verbose=False)["dim"] == (10, 10, 10)
        assert capsys.readouterr().out == ""

    def test_get_regions_info(self, named):
        df = named.get_regions_info()
        assert df["nvoxels"].tolist() == [27, 27, 27, 2]
        assert df["volume"].tolist() == [216.0, 216.0, 216.0, 16.0]
        info = named.get_regions_info(region_labels=[1, 99], output_format="dict")
        assert list(info) == [1] and info[1]["name"] == "ctx-lh-a"
        assert named.get_regions_info(region_names="lh")["index"].tolist() == [1, 2, 4]
        with pytest.raises(ValueError):
            named.get_regions_info(region_labels=[99])
        with pytest.raises(ValueError):
            named.get_regions_info(region_names="nothing")
        with pytest.raises(ValueError):
            named.get_regions_info(output_format="list")

    def test_print_properties(self, named, capsys):
        named.print_properties()
        out = capsys.readouterr().out
        assert "Attributes:" in out and "keep_by_code" in out and "index" in out


####################################################################################################
# Selecting and removing regions
####################################################################################################
class TestSelection:
    def test_keep_by_code(self, named):
        named.keep_by_code([1, 3])
        assert named.index == [1, 3] and set(np.unique(named.data)) == {0, 1, 3}
        assert named.opacity == [1.0, 1.0] and named.maxlab == 3

    def test_keep_by_code_rearrange_and_ranges(self, named):
        named.keep_by_code("2-3", rearrange=True)
        assert named.index == [1, 2] and named.name == ["ctx-lh-b", "ctx-rh-a"]

    def test_keep_by_name(self, named, capsys):
        named.keep_by_name("ctx-lh")
        assert named.name == ["ctx-lh-a", "ctx-lh-b"]
        named.keep_by_name("nothing")
        assert "not found" in capsys.readouterr().out and named.index == [1, 2]

    def test_remove_by_code_and_name(self, named, capsys):
        named.remove_by_code([2])
        assert named.index == [1, 3, 4]
        named.remove_by_name("wm", rearrange=True)
        assert named.index == [1, 2] and named.name == ["ctx-lh-a", "ctx-rh-a"]
        named.remove_by_name("nothing")
        assert "No regions found" in capsys.readouterr().out

    def test_names_and_labels(self, named):
        assert named.names_to_labels("lh") == [1, 2, 4]
        assert named.names_to_labels(["rh-a"]) == [3]
        assert named.labels_to_names([1, 3]) == ["ctx-lh-a", "ctx-rh-a"]
        assert named.labels_to_names("1-2") == ["ctx-lh-a", "ctx-lh-b"]
        with pytest.raises(ValueError, match="not present"):
            named.labels_to_names(9)

    def test_labels_to_names_keeps_input_order(self, named):
        assert named.labels_to_names([3, 1]) == ["ctx-rh-a", "ctx-lh-a"]

    def test_labels_to_names_accepts_names(self, named):
        # Documented: strings are converted to labels with the name mapping
        assert named.labels_to_names("ctx-rh-a") == ["ctx-rh-a"]
        assert named.labels_to_names(["wm-lh-a", 1]) == ["wm-lh-a", "ctx-lh-a"]

    def test_get_voxels(self, named):
        assert len(named.get_voxels_by_code([1, 4])) == 29
        coords = named.get_voxels_by_code([4, 99], all_voxels=False)
        assert list(coords) == [4] and coords[4].tolist() == [[0, 4, 0], [0, 5, 0]]
        assert len(named.get_voxels_by_name("lh")) == 56
        by_name = named.get_voxels_by_name(["wm", "nothing"], all_voxels=False)
        assert list(by_name) == ["wm-lh-a"]

    def test_real_parcellation_selection(self, real_parc):
        real_parc.keep_by_name(["hipp-", "amygd-"])
        assert len(real_parc.index) == 24
        assert all(n.startswith(("hipp-", "amygd-")) for n in real_parc.name)
        assert set(np.unique(real_parc.data)) == {0, *real_parc.index}


####################################################################################################
# Masking
####################################################################################################
class TestMasking:
    def test_apply_mask_array_and_invert(self, named):
        mask = np.zeros(named.dim)
        mask[0:6, 0:3, 0:3] = 1
        keep = copy.deepcopy(named)
        keep.apply_mask(mask)
        assert keep.index == [1, 2]
        named.apply_mask(mask, invert=True)
        assert named.index == [3, 4]

    def test_apply_mask_codes_file_and_parcellation(self, tmp_path, named):
        labels = np.zeros(named.dim, dtype=np.int16)
        labels[0:3, 0:3, 0:3], labels[7:10, 7:10, 7:10] = 5, 6
        path = tmp_path / "mask.nii.gz"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), path)
        from_file = copy.deepcopy(named)
        from_file.apply_mask(path, mask_codes=[6])
        assert from_file.index == [3]
        from_parc = copy.deepcopy(named)
        from_parc.apply_mask(Parcellation(labels), mask_codes="5")
        assert from_parc.index == [1]

    def test_apply_mask_fill(self, named):
        named.data[1, 1, 1] = 0  # A hole inside region 1
        mask = np.zeros(named.dim)
        mask[0:3, 0:3, 0:3] = 1
        named.apply_mask(mask, fill=True)
        assert named.data[1, 1, 1] == 1

    def test_apply_mask_errors(self, tmp_path, named):
        with pytest.raises(ValueError):
            named.apply_mask(np.ones((2, 2, 2)))
        with pytest.raises(ValueError):
            named.apply_mask(tmp_path / "missing.nii.gz")
        with pytest.raises(ValueError):
            named.apply_mask([1, 2])

    def test_mask_image_array(self, named):
        img = np.ones(named.dim)
        masked = named.mask_image(img, region_labels=[1])
        assert masked.sum() == 27 and img.sum() == 1000  # The input is not modified
        assert named.mask_image(img, region_names="lh").sum() == 56
        assert named.mask_image(img, region_labels=[1], invert=True).sum() == 973
        ts = np.ones((*named.dim, 3))
        assert named.mask_image(ts).sum() == 83 * 3  # Every volume of a 4D image

    def test_mask_image_files(self, tmp_path, named):
        paths, outs = [], []
        for i in range(2):
            paths.append(tmp_path / f"img{i}.nii.gz")
            outs.append(tmp_path / f"masked{i}.nii.gz")
            nib.save(nib.Nifti1Image(np.ones(named.dim), np.eye(4)), paths[-1])
        result = named.mask_image(paths, outs, region_labels=[3])
        assert result == [str(o) for o in outs]
        assert nib.load(outs[1]).get_fdata().sum() == 27

    def test_mask_image_errors(self, tmp_path, named):
        img = tmp_path / "img.nii.gz"
        nib.save(nib.Nifti1Image(np.ones(named.dim), np.eye(4)), img)
        with pytest.raises(ValueError):
            named.mask_image([])
        with pytest.raises(ValueError):
            named.mask_image(img)  # Output path required
        with pytest.raises(ValueError):
            named.mask_image([img, img], [tmp_path / "a.nii.gz"])
        with pytest.raises(ValueError):
            named.mask_image(np.ones(named.dim), region_labels=[99])
        with pytest.raises(ValueError):
            named.mask_image(np.ones(named.dim), region_names="nothing")
        with pytest.raises(ValueError):
            named.mask_image(np.ones((2, 2, 2)))


####################################################################################################
# Relabeling, renaming and grouping
####################################################################################################
class TestRelabeling:
    def test_rearrange(self, named):
        named.remove_by_code([2])
        named.rearrange()
        assert named.index == [1, 2, 3] and named.name == [
            "ctx-lh-a",
            "ctx-rh-a",
            "wm-lh-a",
        ]
        named.rearrange(offset=99)
        assert named.index == [100, 101, 102] and named.minlab == 100

    def test_relabel_regions(self, named):
        with pytest.warns(UserWarning, match="both an old_label and a new_label"):
            named.relabel_regions({1: 2, 2: 1})  # Swap
        assert named.data[0, 0, 0] == 2 and named.data[3, 0, 0] == 1
        assert named.relabel_regions({3: 30}, rearrange=True).index == [1, 2, 3, 4]
        with pytest.raises(TypeError):
            named.relabel_regions([1, 2])

    def test_relabel_regions_merging_two_regions(self, named):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            named.relabel_regions({1: 2})
        assert named.index == [2, 3, 4] and len(named.name) == 3
        assert (named.data == 2).sum() == 54

    def test_rename_regions(self, named):
        named.rename_regions({"ctx-lh-a": "left-a", "nothing": "x"})
        assert named.name[0] == "left-a" and named.name[1] == "ctx-lh-b"
        with pytest.raises(TypeError):
            named.rename_regions(["a"])

    def test_replace_labels(self, named):
        named.replace_labels([[1, 2]], [10])  # Merge 1 and 2
        assert named.index == [3, 4, 10] and named.name[2] == "ctx-lh-a"
        named.replace_labels({3: 4, 4: 3})  # Swap without cascading
        assert named.data[8, 8, 8] == 4 and named.data[0, 4, 0] == 3

    def test_replace_labels_forms(self, named):
        assert named.replace_labels("1-2", 20).index == [3, 4, 20]
        assert named.replace_labels(np.array([3, 4]), [30, 40]).index == [20, 30, 40]
        with pytest.warns(UserWarning, match="is present in the data"):
            named.replace_labels({99: 5})

    def test_replace_labels_errors(self, named):
        with pytest.raises(ValueError):
            named.replace_labels([1, 2])  # new_codes missing
        with pytest.raises(ValueError):
            named.replace_labels([1, 2], [5])  # Lengths differ
        with pytest.raises(ValueError):
            named.replace_labels([[1], [1]], [5, 6])  # Code to two targets
        with pytest.raises(TypeError):
            named.replace_labels(1.5, 2)

    def test_replace_names(self, named):
        named.replace_names({"ctx-lh-a": "L_A"})
        assert named.name[0] == "L_A"
        named.replace_names(["ctx-lh-b", "ctx-rh-a"], ["L_B", "R_A"])
        assert named.name[:3] == ["L_A", "L_B", "R_A"]
        named.replace_names("wm", "white", match="contains")
        assert named.name[3] == "white"
        named.replace_names({"L_": "R_", "R_": "L_"}, match="substring")  # Swap
        assert named.name[:3] == ["R_A", "R_B", "L_A"]

    def test_replace_names_case_and_errors(self, named):
        named.replace_names({"CTX-LH-A": "x"}, bool_case=False)
        assert named.name[0] == "x"
        with pytest.warns(UserWarning, match="duplicate"):
            named.replace_names([["ctx-lh-b", "ctx-rh-a"]], ["same"])
        with pytest.raises(ValueError):
            named.replace_names({"a": "b"}, match="regex")
        with pytest.raises(ValueError):
            named.replace_names(["x"])
        with pytest.raises(ValueError):
            named.replace_names({"": "b"})

    def test_group_by_codes(self, named):
        _, ctab = named.group_by_codes(
            {10: {"index": [1, 2], "name": "left", "color": "#123456"}},
            keep_ungrouped=True,
        )
        assert named.index == [3, 4, 10] and named.name[2] == "left"
        assert ctab["index"] == [10, 3, 4] and (named.data == 10).sum() == 54

    def test_group_by_codes_drop_ungrouped_and_errors(self, named):
        named.group_by_codes({7: {"index": "1-2"}})
        assert named.index == [7] and named.name == ["group-000001"]
        with pytest.raises(ValueError):
            named.group_by_codes({7: {"index": [7]}, 8: {"index": [7]}})

    def test_group_by_names(self, named):
        named.group_by_names(
            {"Left": {"names": "ctx-lh", "index": 20}, "Right": {"names": ["ctx-rh"]}}
        )
        assert named.index == [2, 4, 20] and named.name == ["Right", "wm-lh-a", "Left"]

    def test_group_by_names_errors(self, named):
        with pytest.raises(ValueError):
            named.group_by_names({})
        with pytest.raises(ValueError):
            named.group_by_names({"a": {"index": 1}})
        with pytest.raises(ValueError):
            named.group_by_names({"a": {"names": "lh"}, "b": {"names": "ctx"}})
        with pytest.warns(UserWarning), pytest.raises(ValueError):
            named.group_by_names({"a": {"names": "nothing"}})

    def test_real_parcellation_grouping(self, real_parc):
        real_parc.group_by_names(
            {
                "Cortex": {"names": ["ctx-lh", "ctx-rh"], "index": 1},
                "Thalamus": {"names": "thal-", "index": 2},
            },
            keep_ungrouped=False,
        )
        assert real_parc.index == [1, 2] and real_parc.name == ["Cortex", "Thalamus"]
        assert set(np.unique(real_parc.data)) == {0, 1, 2}

    def test_sort_and_harmonize(self, named):
        named.index, named.name = [4, 3, 2, 1], named.name[::-1]
        named.color, named.opacity = named.color[::-1], named.opacity[::-1]
        named.sort_index()
        assert named.index == [1, 2, 3, 4] and named.name[0] == "ctx-lh-a"
        named.data[named.data == 4] = 0
        named.opacity = np.array(named.opacity)
        named.harmonize()
        assert named.index == [1, 2, 3] and named.opacity == [1.0, 0.5, 1.0]

    def test_add_parcellation(self, named, sim):
        other = Parcellation(np.where(named.data == 3, 9, 0), parc_id="other")
        base = copy.deepcopy(named)
        base.remove_by_code([3])
        base.add_parcellation(other)
        assert base.index == [1, 2, 4, 9]
        appended = copy.deepcopy(named)
        extra = np.zeros(named.dim, dtype=np.int32)
        extra[9, 0, 9] = 1  # Background voxel in the base parcellation
        appended.add_parcellation(Parcellation(extra), append=True)
        assert appended.index == [1, 2, 3, 4, 5] and appended.data[9, 0, 9] == 5
        with pytest.raises(TypeError):
            named.add_parcellation("x")
        with pytest.raises(ValueError):
            named.add_parcellation([])


####################################################################################################
# Saving and color tables
####################################################################################################
class TestSaving:
    def test_save_and_reload(self, tmp_path, named):
        out = tmp_path / "parc.nii.gz"
        named.save_parcellation(out, lut_type=["lut", "tsv"])
        assert (tmp_path / "parc.lut").is_file() and (tmp_path / "parc.tsv").is_file()
        reloaded = Parcellation(out)  # The TSV sidecar takes priority
        assert np.array_equal(reloaded.data, named.data)
        assert reloaded.name == named.name and reloaded.color == named.color
        assert reloaded.opacity == named.opacity
        assert np.allclose(reloaded.affine, named.affine)

    def test_save_lut_round_trip(self, tmp_path, named):
        named.save_parcellation(tmp_path / "parc.nii.gz", lut_type="lut")
        reloaded = Parcellation(tmp_path / "parc.nii.gz")
        assert reloaded.name == named.name and reloaded.color == named.color
        np.testing.assert_allclose(reloaded.opacity, named.opacity, atol=1 / 255)

    def test_save_with_header_line_string(self, tmp_path, named):
        named.save_parcellation(tmp_path / "parc.nii.gz", headerlines="# My atlas")
        assert "# My atlas" in (tmp_path / "parc.lut").read_text()

    def test_save_options_and_errors(self, tmp_path, named):
        named.save_parcellation(
            tmp_path / "p.nii.gz",
            lut_type=["fsl", "nilearn"],
            lut_file=[tmp_path / "a.txt", tmp_path / "b.tsv"],
        )
        assert (tmp_path / "a.txt").is_file() and (tmp_path / "b.tsv").is_file()
        named.save_parcellation(tmp_path / "nolut.nii.gz", lut_type=None)
        assert not (tmp_path / "nolut.lut").exists()
        with pytest.raises(FileExistsError):
            named.save_parcellation(tmp_path / "p.nii.gz", overwrite=False)
        with pytest.raises(ValueError):
            named.save_parcellation(tmp_path / "x.nii.gz", lut_type="csv")
        with pytest.raises(ValueError):
            named.save_parcellation(
                tmp_path / "x.nii.gz",
                lut_type=["lut", "tsv"],
                lut_file=tmp_path / "x.lut",
            )
        with pytest.raises(ValueError):
            named.save_parcellation(
                tmp_path / "x.nii.gz",
                lut_type=["lut"],
                lut_file=[tmp_path / "a", tmp_path / "b"],
            )

    def test_export_colortable_keeps_attributes(self, tmp_path, named):
        named.export_colortable(str(tmp_path / "c.lut"))
        assert isinstance(named.index, list) and named.index == [1, 2, 3, 4]
        assert named.name == ["ctx-lh-a", "ctx-lh-b", "ctx-rh-a", "wm-lh-a"]

    def test_export_and_load_colortable(self, tmp_path, named):
        named.export_colortable(str(tmp_path / "c.tsv"), lut_type="tsv")
        df = pd.read_csv(tmp_path / "c.tsv", sep="\t")
        assert df["name"].tolist() == named.name
        other = Parcellation(named.data)
        other.load_colortable(str(tmp_path / "c.tsv"))
        assert other.name == named.name and other.lut_file == str(tmp_path / "c.tsv")
        other.load_colortable({"index": [1, 2, 3, 4], "name": list("abcd")})
        assert other.name == list("abcd") and other.lut_file is None
        with pytest.raises(ValueError):
            other.load_colortable(str(tmp_path / "missing.lut"))
        with pytest.raises(ValueError):
            other.load_colortable({"index": [1]})

    def test_load_colortable_without_freesurfer(self, named, monkeypatch):
        monkeypatch.delenv("FREESURFER_HOME", raising=False)
        with pytest.raises(ValueError, match="FREESURFER_HOME"):
            named.load_colortable()

    def test_export_summary_to_hdf5(self, tmp_path, named):
        out = tmp_path / "summary.h5"
        named.export_summary_to_hdf5(str(out))
        with h5py.File(out) as f:
            group = f["parcellation_numpy_array/space-unknown"]
            assert list(group["regions_indices"][()]) == [1, 2, 3, 4]
            assert group["header/num_regions"][()] == 4
        with pytest.raises(ValueError):
            named.export_summary_to_hdf5(str(out))  # Exists, overwrite=False
        with pytest.raises(ValueError):
            named.export_summary_to_hdf5(str(tmp_path / "missing" / "s.h5"))

    def test_export_summary_to_hdf5_with_morphometry(self, tmp_path, named):
        named.compute_morphometry_table()
        out = tmp_path / "summary.h5"
        named.export_summary_to_hdf5(str(out))
        assert out.is_file()


####################################################################################################
# Measures
####################################################################################################
class TestMeasures:
    def test_parc_range(self, named):
        assert named.parc_range() == (1, 4)
        named.data[:] = 0
        assert named.parc_range() == (0, 0)

    def test_compute_centroids(self, tmp_path, named):
        df = named.compute_centroids(gaussian_smooth=False, closing_iterations=0)
        assert df["index"].tolist() == [1, 2, 3, 4]
        assert df.loc[0, ["x_vox", "y_vox", "z_vox"]].tolist() == [1, 1, 1]
        assert df.loc[0, ["x_mm", "y_mm", "z_mm"]].tolist() == [2, 2, 2]
        assert df["nvoxels"].tolist() == [27, 27, 27, 2] and df.loc[0, "volume"] == 216
        assert named.centroids.shape == (4, 3)
        out = tmp_path / "centroids.tsv"
        sub = named.compute_centroids(region_names="ctx-lh", centroid_table=out)
        assert sub["index"].tolist() == [1, 2] and out.is_file()
        pc = named.compute_centroids(region_labels=[3], output_format="pointcloud")
        assert pc.coords.shape == (1, 3)
        with pytest.raises(ValueError):
            named.compute_centroids(region_labels=[1], region_names="a")

    def test_compute_region_adjacency(self, named):
        adjacency, source, target = named.compute_region_adjacency()
        assert adjacency.matrix.shape == (4, 4)
        assert list(zip(source["codes"], target["codes"], strict=True)) == [(1, 2)]
        assert adjacency.matrix[0, 1] == adjacency.matrix[1, 0] == 1
        weighted, source_w, _ = named.compute_region_adjacency(weighted=True)
        assert weighted.matrix[0, 1] == source_w["weights"][0] > 1
        sub, src, _ = named.compute_region_adjacency(region_names="ctx", rearrange=True)
        assert sub.matrix.shape == (3, 3) and src["names"] == ["ctx-lh-a"]
        with pytest.raises(ValueError):
            named.compute_region_adjacency(region_labels=[1], region_names="a")

    def test_volume_and_morphometry_tables(self, tmp_path, named):
        volumes = named.compute_volume_table()
        table = volumes[0] if isinstance(volumes, tuple) else volumes
        assert isinstance(table, pd.DataFrame) and len(table) > 0
        out = tmp_path / "morpho.csv"
        morpho = named.compute_morphometry_table(output_table=out)
        assert out.is_file() and named.morphometry is morpho
        with pytest.raises(FileNotFoundError):
            named.compute_morphometry_table(output_table=tmp_path / "missing" / "m.csv")
        with pytest.raises(ValueError):
            named.compute_morphometry_table(map_files=["a", "b"], map_ids="x")

    def test_morphometry_with_a_map(self, tmp_path, named):
        values = tmp_path / "fa.nii.gz"
        nib.save(nib.Nifti1Image(np.full(named.dim, 0.5), named.affine), values)
        table = named.compute_morphometry_table(
            map_files=values, map_ids="fa", units="none"
        )
        assert "fa" in table.to_string()
        with pytest.warns(UserWarning, match="not found"):
            named.compute_morphometry_table(map_files=tmp_path / "missing.nii.gz")


####################################################################################################
# Time series and functional connectivity
####################################################################################################
class TestTimeSeries:
    def test_regionwise_timeseries(self, named):
        rng = np.random.default_rng(0)
        ts = rng.normal(size=(*named.dim, 6))
        rts = named.get_regionwise_timeseries(ts, method="clabtoolkit")
        assert isinstance(rts, RegionTimeSeries) and rts.data.shape == (4, 6)
        expected = ts[named.data == 1].mean(axis=0)
        np.testing.assert_allclose(rts.data[0], expected)
        assert rts.region_names == named.name
        sub = named.get_regionwise_timeseries(
            ts, method="clabtoolkit", region_labels=[3], vols_to_delete=[0, 1]
        )
        assert sub.data.shape == (1, 4)

    def test_regionwise_timeseries_from_file(self, tmp_path, named):
        path = tmp_path / "bold.nii.gz"
        ts = np.random.default_rng(1).normal(size=(*named.dim, 5))
        nib.save(nib.Nifti1Image(ts, named.affine), path)
        rts = named.get_regionwise_timeseries(
            str(path), method="clabtoolkit", metric="median"
        )
        np.testing.assert_allclose(rts.data[2], np.median(ts[named.data == 3], axis=0))

    def test_regionwise_timeseries_errors(self, named):
        with pytest.raises(ValueError):
            named.get_regionwise_timeseries(np.ones(named.dim), method="clabtoolkit")
        with pytest.raises(ValueError):
            named.get_regionwise_timeseries(np.ones((2, 2, 2, 3)), method="clabtoolkit")

    def test_compute_fc_matrix(self, named):
        rng = np.random.default_rng(2)
        data = rng.normal(size=(4, 50))
        fc = named.compute_fc_matrix(data)
        assert fc.matrix.shape == (4, 4) and np.allclose(np.diag(fc.matrix), 1)
        np.testing.assert_allclose(fc.matrix, np.corrcoef(data))
        for method in ["spearman", "kendall", "partial", "mutual_info"]:
            assert named.compute_fc_matrix(data, method=method).matrix.shape == (4, 4)
        z = named.compute_fc_matrix(
            data, z_transform=True, absolute=True, threshold=0.1
        )
        assert (z.matrix >= 0).all()
        ts4d = rng.normal(size=(*named.dim, 20))
        assert named.compute_fc_matrix(ts4d, ts_method="clabtoolkit").matrix.shape == (
            4,
            4,
        )

    def test_compute_fc_matrix_errors(self, tmp_path, named):
        data = np.random.default_rng(3).normal(size=(4, 10))
        with pytest.raises(ValueError):
            named.compute_fc_matrix(data, method="cosine")
        with pytest.raises(ValueError):
            named.compute_fc_matrix(data[:3])
        with pytest.raises(ValueError):
            named.compute_fc_matrix(
                data[:, :4], method="partial"
            )  # 4 time points, 4 rois
        with pytest.raises(ValueError):
            named.compute_fc_matrix(np.ones(3))
        with pytest.raises(ValueError):
            named.compute_fc_matrix(str(tmp_path / "missing.nii.gz"))

    def test_compute_fc_matrix_from_region_timeseries(self, named):
        rts = RegionTimeSeries(np.random.default_rng(4).normal(size=(4, 30)))
        fc = named.compute_fc_matrix(rts)
        np.testing.assert_allclose(fc.matrix, np.corrcoef(rts.data))
        named_rts = RegionTimeSeries(rts.data, region_names=named.name)
        sub = named.compute_fc_matrix(
            named_rts, region_labels=[1, 3], vols_to_delete="0-4"
        )
        np.testing.assert_allclose(sub.matrix, np.corrcoef(rts.data[[0, 2], 5:]))

    def test_region_timeseries(self, capsys):
        data = np.random.default_rng(5).normal(size=(3, 40))
        rts = RegionTimeSeries(data, region_names=["a", "b", "c"])
        assert len(rts.region_colors) == 3
        fc = rts.compute_fc_matrix(region_names=["a", "c"], vols_to_delete="0-4")
        np.testing.assert_allclose(fc.matrix, np.corrcoef(data[[0, 2], 5:]))
        rts.get_info()
        assert "REGION TIME SERIES" in capsys.readouterr().out
        with pytest.raises(ValueError):
            RegionTimeSeries(data, region_names=["a"])
        with pytest.raises(ValueError):
            rts.compute_fc_matrix(vols_to_delete=[100])

    def test_region_timeseries_empty_vols_to_delete(self):
        data = np.random.default_rng(6).normal(size=(2, 20))
        fc = RegionTimeSeries(data).compute_fc_matrix(vols_to_delete=[])
        np.testing.assert_allclose(fc.matrix, np.corrcoef(data))


####################################################################################################
# Chimera-specific methods (real parcellation)
####################################################################################################
class TestChimera:
    def test_merge_ctx_wm(self, real_parc):
        lh_code = real_parc.names_to_labels("ctx-lh-precentral")[0]
        n_before = (real_parc.data == lh_code).sum() + (
            real_parc.data == lh_code + 3000
        ).sum()
        real_parc.merge_ctx_wm()
        assert (real_parc.data == lh_code).sum() == n_before
        assert not any(n.startswith(("wm-lh", "wm-rh")) for n in real_parc.name)
        assert "wm-brain-whitematter" in real_parc.name

    def test_merge_ctx_wm_without_wm(self, named):
        named.remove_by_name("wm")
        with pytest.warns(UserWarning, match="No WM parcel"):
            named.merge_ctx_wm()

    def test_prepare_for_connectomics(self, tmp_path, real_parc):
        out = tmp_path / "conn.nii.gz"
        real_parc.prepare_for_connectomics(output_file=out)
        assert real_parc.maxlab < 3000 and out.is_file()
        assert not any(n.startswith("wm-") for n in real_parc.name)

    def test_create_5tt(self, tmp_path, real_parc):
        out = tmp_path / "5tt.nii.gz"
        five_tt = real_parc.create_5tt(output_file=out)
        assert five_tt.shape == (*real_parc.dim, 5) and out.is_file()
        assert five_tt.sum(axis=-1).max() == 1  # Tissues do not overlap
        labeled = real_parc.data > 0
        assert (
            five_tt[labeled].sum() == labeled.sum()
        )  # Every labeled voxel is assigned
        merged = real_parc.create_5tt(mergectx=True)
        assert merged[..., 0].sum() > five_tt[..., 0].sum()
        assert len(real_parc.name) == 277  # mergectx works on a copy

    def test_create_5tt_requires_chimera_names(self, sim):
        with pytest.warns(UserWarning), pytest.raises(ValueError, match="Chimera"):
            sim.create_5tt()


####################################################################################################
# Surfaces
####################################################################################################
class TestSurfaces:
    def test_surface_extraction(self, tmp_path, named):
        surfaces = named.surface_extraction(region_labels=[1, 3], merge_surfaces=False)
        assert len(surfaces) == 2 and surfaces[0].mesh.n_points > 0
        merged = named.surface_extraction(region_labels=[1, 3])
        assert merged.mesh.n_cells == sum(s.mesh.n_cells for s in surfaces)
        assert sorted(np.unique(merged.mesh.point_data["surf_id"])) == [1, 2]
        out = tmp_path / "regions.surf"
        named.surface_extraction(region_names="ctx-rh", out_filename=str(out))
        assert out.is_file() and (tmp_path / "regions.annot").is_file()
        with pytest.raises(FileExistsError):
            named.surface_extraction(region_names="ctx-rh", out_filename=str(out))
        with pytest.raises(FileNotFoundError):
            named.surface_extraction(out_filename=str(tmp_path / "missing" / "s.surf"))
        with pytest.raises(ValueError):
            named.surface_extraction(region_labels=[1], region_names="a")
