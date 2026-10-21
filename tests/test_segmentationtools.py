"""Tests for the segmentationtools module.

tissue_seg_table is tested directly. abased_parcellation needs ANTs, so subprocess.run is
replaced by a fake that records every command and creates the files ANTs would write.
"""

import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

import clabtoolkit.segmentationtools as cltseg


######################################################################################################
class TestTissueSegTable:
    def test_table_content(self):
        df = cltseg.tissue_seg_table()
        assert list(df.columns) == ["index", "name", "abbreviation", "color"]
        assert df["index"].tolist() == [1, 2, 3]
        assert df["name"].tolist() == [
            "cerebro_spinal_fluid",
            "gray_matter",
            "white_matter",
        ]
        assert df["abbreviation"].tolist() == ["CSF", "GM", "WM"]

    def test_colors_are_hexadecimal(self):
        colors = cltseg.tissue_seg_table()["color"]
        assert colors.str.fullmatch(r"#[0-9a-fA-F]{6}").all()
        assert colors.str.lower().tolist() == ["#ac0000", "#00994c", "#0066cc"]

    @pytest.mark.parametrize("as_path", [False, True])
    def test_save_tsv(self, tmp_path, as_path):
        tsv = tmp_path / "tissues.tsv"
        df = cltseg.tissue_seg_table(tsv if as_path else str(tsv))
        assert tsv.is_file()
        assert pd.read_csv(tsv, sep="\t").equals(df)

    def test_no_file_without_filename(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cltseg.tissue_seg_table()
        assert not list(tmp_path.iterdir())

    def test_overwrite_true_replaces_existing_file(self, tmp_path):
        tsv = tmp_path / "tissues.tsv"
        tsv.write_text("old")
        cltseg.tissue_seg_table(tsv, overwrite=True)
        assert tsv.read_text().startswith("index\tname")

    def test_overwrite_false_keeps_existing_file(self, tmp_path):
        tsv = tmp_path / "tissues.tsv"
        tsv.write_text("old")
        df = cltseg.tissue_seg_table(tsv, overwrite=False)
        assert tsv.read_text() == "old"
        assert len(df) == 3

    def test_overwrite_false_still_creates_missing_file(self, tmp_path):
        tsv = tmp_path / "tissues.tsv"
        cltseg.tissue_seg_table(tsv, overwrite=False)
        assert tsv.is_file()

    def test_missing_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltseg.tissue_seg_table(tmp_path / "missing_dir" / "tissues.tsv")
        assert not (tmp_path / "missing_dir").exists()


######################################################################################################
class FakeAnts:
    """Replacement of subprocess.run that records commands and mimics the ANTs outputs."""

    def __init__(self, label_data=None):
        self.calls = []
        self.label_data = label_data

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append(list(cmd))
        tool = cmd[0]
        if tool == "antsRegistrationSyN.sh":
            prefix = cmd[cmd.index("-o") + 1]
            for suffix in (
                "0GenericAffine.mat",
                "1Warp.nii.gz",
                "1InverseWarp.nii.gz",
                "Warped.nii.gz",
                "InverseWarped.nii.gz",
            ):
                Path(prefix + suffix).write_text(suffix)
        elif tool == "mv":
            shutil.move(cmd[1], cmd[2])
        elif tool == "antsApplyTransforms":
            out = cmd[cmd.index("-o") + 1]
            data = (
                self.label_data if self.label_data is not None else np.zeros((4, 4, 4))
            )
            nib.save(nib.Nifti1Image(data, np.eye(4)), out)

    def tools(self):
        return [c[0] for c in self.calls]

    def call(self, tool):
        return next(c for c in self.calls if c[0] == tool)


@pytest.fixture
def ants(monkeypatch):
    fake = FakeAnts(
        label_data=np.array([0, 1, 2, 3] * 16, dtype=np.int16).reshape(4, 4, 4)
    )
    monkeypatch.setattr(cltseg.subprocess, "run", fake)
    return fake


@pytest.fixture
def paths(tmp_path):
    return {
        "t1": str(tmp_path / "sub-01_T1w.nii.gz"),
        "temp": str(tmp_path / "MNI152_T1_1mm.nii.gz"),
        "atlas": str(tmp_path / "atlas_dseg.nii.gz"),
        "out": str(tmp_path / "derivatives" / "anat" / "sub-01_atlas-sim_dseg.nii.gz"),
        "xfm": str(tmp_path / "derivatives" / "xfm" / "sub-01_from-T1w_to-MNI152"),
        "xfm_dir": tmp_path / "derivatives" / "xfm",
    }


def run(paths, **kwargs):
    return cltseg.abased_parcellation(
        t1=paths["t1"],
        t1_temp=paths["temp"],
        atlas=paths["atlas"],
        out_parc=paths["out"],
        xfm_output=paths["xfm"],
        **kwargs,
    )


class TestAbasedParcellation:
    def test_returns_output_path_and_creates_folders(self, ants, paths):
        assert run(paths) == paths["out"]
        assert Path(paths["out"]).is_file()
        assert paths["xfm_dir"].is_dir()

    def test_registration_command(self, ants, paths):
        run(paths)
        cmd = ants.call("antsRegistrationSyN.sh")
        assert cmd[cmd.index("-f") + 1] == paths["temp"]  # template is the fixed image
        assert cmd[cmd.index("-m") + 1] == paths["t1"]  # subject is the moving image
        assert cmd[cmd.index("-t") + 1] == "s"
        assert cmd[cmd.index("-o") + 1] == str(paths["xfm_dir"] / "temp_")

    def test_transforms_get_bids_names(self, ants, paths):
        run(paths)
        base = "sub-01_from-T1w_to-MNI152"
        assert sorted(p.name for p in paths["xfm_dir"].iterdir()) == [
            f"{base}_desc-affine_xfm.mat",
            f"{base}_desc-iwarp_xfm.nii.gz",
            f"{base}_desc-warp_xfm.nii.gz",
        ]

    def test_warped_images_are_removed(self, ants, paths):
        run(paths)
        assert not list(paths["xfm_dir"].glob("*Warped*"))
        assert not list(paths["xfm_dir"].glob("temp*"))

    def test_spam_application(self, ants, paths):
        run(paths, atlas_type="spam", interp="BSpline")
        cmd = ants.call("antsApplyTransforms")
        assert (
            cmd[cmd.index("-e") + 1] == "3"
        )  # 4D images are transformed volume by volume
        assert cmd[cmd.index("-n") + 1] == "BSpline"
        assert cmd[cmd.index("-i") + 1] == paths["atlas"]
        assert cmd[cmd.index("-r") + 1] == paths["t1"]  # subject T1 is the reference
        transforms = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-t"]
        assert transforms[0].endswith("_desc-iwarp_xfm.nii.gz")
        assert transforms[1].startswith("[") and transforms[1].endswith(
            "_desc-affine_xfm.mat,1]"
        )

    def test_spam_does_not_write_lookup_tables(self, ants, paths):
        run(paths, atlas_type="spam")
        assert sorted(p.name for p in Path(paths["out"]).parent.iterdir()) == [
            Path(paths["out"]).name
        ]

    def test_maxprob_uses_nearest_neighbor_and_saves_lookup_tables(self, ants, paths):
        run(paths, atlas_type="maxprob", interp="Linear")
        cmd = ants.call("antsApplyTransforms")
        assert cmd[cmd.index("-n") + 1] == "NearestNeighbor"
        assert "-e" not in cmd
        out_dir = Path(paths["out"]).parent
        stem = "sub-01_atlas-sim_dseg"
        assert (out_dir / f"{stem}.lut").is_file()
        assert (out_dir / f"{stem}.tsv").is_file()
        labels = np.unique(nib.load(paths["out"]).get_fdata())
        assert set(labels) <= {0, 1, 2, 3}

    def test_existing_inverse_warp_skips_registration(self, ants, paths):
        run(paths)
        out = Path(paths["out"])
        out.unlink()
        ants.calls.clear()

        run(paths)
        assert "antsRegistrationSyN.sh" not in ants.tools()
        assert "antsApplyTransforms" in ants.tools()

    def test_overwrite_repeats_registration(self, ants, paths):
        run(paths)
        ants.calls.clear()
        run(paths, overwrite=True)
        assert ants.tools().count("antsRegistrationSyN.sh") == 1

    def test_existing_parcellation_is_not_recomputed(self, ants, paths):
        run(paths)
        ants.calls.clear()
        run(paths)
        assert "antsApplyTransforms" not in ants.tools()

    @pytest.mark.parametrize(
        "xfm_name",
        [
            "sub-01_from-T1w_to-MNI152",
            "sub-01_from-T1w_to-MNI152.nii.gz",
            "sub-01_from-T1w_to-MNI152.mat",
            "sub-01_from-T1w_to-MNI152_xfm",
        ],
    )
    def test_extensions_and_suffix_are_stripped_from_the_transform_name(
        self, ants, paths, xfm_name
    ):
        paths["xfm"] = str(paths["xfm_dir"] / xfm_name)
        run(paths)
        assert (
            paths["xfm_dir"] / "sub-01_from-T1w_to-MNI152_desc-affine_xfm.mat"
        ).is_file()

    def test_existing_desc_entity_is_replaced(self, ants, paths):
        paths["xfm"] = str(paths["xfm_dir"] / "sub-01_from-T1w_to-MNI152_desc-reg")
        run(paths)
        names = sorted(p.name for p in paths["xfm_dir"].iterdir())
        assert names == [
            "sub-01_from-T1w_to-MNI152_desc-affine_xfm.mat",
            "sub-01_from-T1w_to-MNI152_desc-iwarp_xfm.nii.gz",
            "sub-01_from-T1w_to-MNI152_desc-warp_xfm.nii.gz",
        ]

    def test_container_requires_an_image(self, ants, paths):
        with pytest.raises(ValueError):
            run(paths, cont_tech="docker")

    def test_container_image_must_exist(self, ants, paths, tmp_path):
        with pytest.raises(ValueError):
            run(
                paths, cont_tech="singularity", cont_image=str(tmp_path / "missing.sif")
            )

    def test_container_command_wraps_ants(self, ants, paths, tmp_path):
        image = tmp_path / "ants.sif"
        image.write_text("")
        run(paths, cont_tech="singularity", cont_image=str(image))
        assert ants.calls[0][0] == "singularity"
        assert "antsRegistrationSyN.sh" in ants.calls[0]
