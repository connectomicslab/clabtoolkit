"""Tests for clabtoolkit.qcqatools.

The figures are drawn on the Agg backend, which does not need a display. To check what
is drawn, the tests record the figure at the moment it is saved.
"""

import json
import os
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pytest

import clabtoolkit.qcqatools as cltqcqa

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


####################################################################################################
# Helpers and fixtures
####################################################################################################
def make_image(
    path, shape=(20, 24, 16), zooms=(1.0, 1.0, 1.0), seed=0, origin=(0, 0, 0)
):
    """Image with a box of signal (values between 1 and 2) surrounded by zeros."""
    rng = np.random.default_rng(seed)
    data = np.zeros(shape, dtype=np.float32)
    inner = tuple(slice(4, s - 4) for s in shape[:3])
    inner_shape = tuple(s - 8 for s in shape[:3]) + tuple(shape[3:])
    data[inner] = rng.random(inner_shape) + 1
    affine = np.diag([*zooms, 1.0])
    affine[:3, 3] = origin
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return data


def read_png(path):
    """Pixels of a PNG file (rows, columns, channels)."""
    return plt.imread(str(path))


@pytest.fixture(autouse=True)
def _agg_backend(monkeypatch):
    """Draw on the Agg backend, without waiting between the files."""
    previous = matplotlib.get_backend()
    plt.switch_backend("Agg")
    monkeypatch.setattr(cltqcqa.time, "sleep", lambda seconds: None)

    # The images are saved at 300 dpi by default: draw them smaller to save time
    original_savefig = plt.savefig

    def small_savefig(fname, *args, **kwargs):
        kwargs["dpi"] = min(kwargs.get("dpi", 300), 100)
        return original_savefig(fname, *args, **kwargs)

    monkeypatch.setattr(plt, "savefig", small_savefig)
    yield
    plt.close("all")
    plt.switch_backend(previous)


@pytest.fixture
def figures(monkeypatch):
    """Record what is on the figure each time it is saved."""
    saved = []
    original = plt.savefig

    def recording_savefig(fname, *args, **kwargs):
        fig = plt.gcf()
        panels = [ax for ax in fig.axes if ax.get_title()]
        saved.append(
            {
                "figure": fig,
                "panels": panels,
                "titles": [ax.get_title() for ax in panels],
                "arrays": [np.asarray(ax.images[0].get_array()) for ax in panels],
                "clims": [ax.images[0].get_clim() for ax in panels],
                "cmaps": [ax.images[0].get_cmap().name for ax in panels],
                "extents": [ax.images[0].get_extent() for ax in panels],
                "n_axes": len(fig.axes),
                "size": tuple(fig.get_size_inches()),
                "facecolor": fig.patch.get_facecolor(),
                "title_colors": [ax.title.get_color() for ax in panels],
                "kwargs": kwargs,
            }
        )
        return original(fname, *args, **kwargs)

    monkeypatch.setattr(plt, "savefig", recording_savefig)
    return saved


@pytest.fixture
def t1(tmp_path):
    path = tmp_path / "sub-01_T1w.nii.gz"
    data = make_image(path)
    return path, data


####################################################################################################
class TestGetValidSlices:
    def test_slices_inside_the_signal(self):
        data = np.zeros((100, 100, 100))
        data[20:61, 40:81, 10:51] = 1
        sagittal, coronal, axial = cltqcqa.get_valid_slices(data)
        # 30 %, 50 % and 70 % of the extent of the signal (not of the volume)
        assert sagittal == [32, 40, 48]
        assert coronal == [52, 60, 68]
        assert axial == [22, 30, 38]

    def test_background_value_and_slice_positions(self):
        data = np.full((100, 100, 100), -1.0)
        data[20:61, 40:81, 10:51] = 0
        sagittal, coronal, axial = cltqcqa.get_valid_slices(
            data, ignore_value=-1, slice_positions=[0.25, 0.5, 0.75]
        )
        assert sagittal == [30, 40, 50]
        assert coronal == [50, 60, 70]
        assert axial == [20, 30, 40]

    def test_zero_is_not_the_background_when_another_value_is_given(self):
        data = np.full((10, 10, 10), -1.0)
        data[2:5, 3:7, 4:9] = 0
        with_zero = cltqcqa.get_valid_slices(data)
        with_minus_one = cltqcqa.get_valid_slices(data, ignore_value=-1)
        assert with_zero[0] == [0, 4, 6] or with_zero != with_minus_one
        assert with_minus_one[0][0] >= 2 and with_minus_one[0][-1] <= 4

    def test_positions_at_the_ends_of_the_signal(self):
        data = np.zeros((30, 30, 30))
        data[5:16, 6:21, 7:27] = 3
        sagittal, coronal, axial = cltqcqa.get_valid_slices(
            data, slice_positions=[0, 1]
        )
        assert sagittal == [5, 15]
        assert coronal == [6, 20]
        assert axial == [7, 26]

    def test_the_number_of_slices_follows_the_positions(self):
        data = np.ones((10, 10, 10))
        for positions in ([0.5], [0.1, 0.9], [0.1, 0.2, 0.3, 0.4]):
            slices = cltqcqa.get_valid_slices(data, slice_positions=positions)
            assert [len(s) for s in slices] == [len(positions)] * 3

    def test_fractions_are_truncated(self):
        data = np.zeros((10, 10, 10))
        data[0:6, 0:6, 0:6] = 1  # extent 0..5
        sagittal, _, _ = cltqcqa.get_valid_slices(data, slice_positions=[0.5, 0.9])
        assert sagittal == [2, 4]

    def test_whole_volume_without_signal(self):
        data = np.zeros((11, 21, 31))
        sagittal, coronal, axial = cltqcqa.get_valid_slices(
            data, slice_positions=[0, 0.5, 1]
        )
        assert sagittal == [0, 5, 10]
        assert coronal == [0, 10, 20]
        assert axial == [0, 15, 30]

    def test_volume_that_is_all_signal(self):
        data = np.ones((11, 21, 31))
        sagittal, coronal, axial = cltqcqa.get_valid_slices(
            data, slice_positions=[0, 0.5, 1]
        )
        assert (sagittal, coronal, axial) == ([0, 5, 10], [0, 10, 20], [0, 15, 30])

    def test_single_voxel(self):
        data = np.zeros((10, 10, 10))
        data[3, 4, 5] = 7
        assert cltqcqa.get_valid_slices(data) == ([3] * 3, [4] * 3, [5] * 3)

    def test_negative_values_are_signal(self):
        data = np.zeros((10, 10, 10))
        data[2:5, 2:5, 2:5] = -3
        assert cltqcqa.get_valid_slices(data, slice_positions=[0, 1])[0] == [2, 4]

    def test_nan_counts_as_signal(self):
        data = np.zeros((10, 10, 10))
        data[6, 6, 6] = np.nan
        assert cltqcqa.get_valid_slices(data, slice_positions=[0.5])[0] == [6]

    def test_results_are_python_integers(self):
        data = np.ones((10, 10, 10), dtype=np.float32)
        for slices in cltqcqa.get_valid_slices(data):
            assert all(type(s) is int for s in slices)

    def test_input_is_not_modified(self):
        data = np.zeros((10, 10, 10))
        data[2:5, 2:5, 2:5] = 1
        before = data.copy()
        cltqcqa.get_valid_slices(data, ignore_value=5)
        np.testing.assert_array_equal(data, before)

    def test_tuple_of_positions(self):
        data = np.ones((10, 10, 10))
        assert cltqcqa.get_valid_slices(data, slice_positions=(0, 1))[0] == [0, 9]


####################################################################################################
class TestGenerateSlices:
    def test_output_next_to_the_image(self, t1):
        path, _ = t1
        result = cltqcqa.generate_slices(path, dpi=40)
        assert result == str(path).replace(".nii.gz", ".png")
        assert Path(result).read_bytes()[:8] == PNG_MAGIC

    @pytest.mark.parametrize("name", ["T1w.nii", "T1w.mgz"])
    def test_other_extensions(self, tmp_path, name):
        path = tmp_path / name
        make_image(
            path.with_suffix(".nii")
            if name.endswith(".nii")
            else path.with_suffix(".nii")
        )
        if name.endswith(".mgz"):
            img = nib.load(str(path.with_suffix(".nii")))
            nib.save(nib.MGHImage(np.asarray(img.dataobj), img.affine), str(path))
        else:
            path = path.with_suffix(".nii")
        result = cltqcqa.generate_slices(path, dpi=40)
        assert result == str(tmp_path / "T1w.png")

    def test_output_path_and_missing_folders(self, tmp_path, t1):
        out = tmp_path / "figures" / "deeper" / "T1.png"
        result = cltqcqa.generate_slices(t1[0], output_path=out, dpi=40)
        assert result == str(out)
        assert out.is_file()

    def test_string_paths(self, tmp_path, t1):
        out = tmp_path / "T1.png"
        result = cltqcqa.generate_slices(str(t1[0]), output_path=str(out), dpi=40)
        assert result == str(out)

    def test_existing_png_is_kept_without_overwrite(self, tmp_path, t1, capsys):
        out = tmp_path / "T1.png"
        out.write_bytes(b"old")
        result = cltqcqa.generate_slices(t1[0], output_path=out)
        assert result == str(out)
        assert out.read_bytes() == b"old"
        assert "already exists" in capsys.readouterr().out

    def test_existing_png_is_replaced_with_overwrite(self, tmp_path, t1):
        out = tmp_path / "T1.png"
        out.write_bytes(b"old")
        cltqcqa.generate_slices(t1[0], output_path=out, overwrite=True, dpi=40)
        assert out.read_bytes()[:8] == PNG_MAGIC

    def test_missing_image(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            cltqcqa.generate_slices(tmp_path / "missing.nii.gz")

    def test_unreadable_image(self, tmp_path):
        path = tmp_path / "broken.nii.gz"
        path.write_text("this is not an image")
        with pytest.raises(ValueError, match="Failed to load"):
            cltqcqa.generate_slices(path)

    def test_nine_slices_by_default(self, tmp_path, t1, figures):
        cltqcqa.generate_slices(t1[0], output_path=tmp_path / "a.png", dpi=40)
        record = figures[0]
        assert len(record["panels"]) == 9
        assert record["n_axes"] == 10  # and the colorbar
        assert [t[0] for t in record["titles"]] == ["X"] * 3 + ["Y"] * 3 + ["Z"] * 3

    def test_the_slices_are_the_ones_of_the_image(self, tmp_path, t1, figures):
        path, data = t1
        cltqcqa.generate_slices(path, output_path=tmp_path / "a.png", dpi=40)
        sagittal, coronal, axial = cltqcqa.get_valid_slices(data.astype(float))
        expected = (
            [data[s, :, :] for s in sagittal]
            + [data[:, s, :] for s in coronal]
            + [data[:, :, s] for s in axial]
        )
        for shown, wanted in zip(figures[0]["arrays"], expected, strict=True):
            np.testing.assert_allclose(shown, np.rot90(wanted))

    def test_slice_positions(self, tmp_path, t1, figures):
        cltqcqa.generate_slices(
            t1[0], output_path=tmp_path / "a.png", slice_positions=[0.5], dpi=40
        )
        assert len(figures[0]["panels"]) == 3
        assert figures[0]["titles"][0].startswith("X = ")

    def test_titles_give_the_position_in_millimeters(self, tmp_path, figures):
        path = tmp_path / "scaled.nii.gz"
        make_image(path, shape=(20, 24, 16), zooms=(2.0, 3.0, 4.0))
        cltqcqa.generate_slices(
            path, output_path=tmp_path / "a.png", slice_positions=[0, 1], dpi=40
        )
        # The signal spans the voxels 4..15, 4..19 and 4..11
        assert figures[0]["titles"] == [
            "X = 8.0 mm", "X = 30.0 mm",
            "Y = 12.0 mm", "Y = 57.0 mm",
            "Z = 16.0 mm", "Z = 44.0 mm",
        ]  # fmt: skip

    def test_extents_follow_the_affine(self, tmp_path, figures):
        path = tmp_path / "moved.nii.gz"
        make_image(
            path, shape=(20, 24, 16), zooms=(2.0, 2.0, 2.0), origin=(-10, -12, -8)
        )
        cltqcqa.generate_slices(
            path, output_path=tmp_path / "a.png", slice_positions=[0.5], dpi=40
        )
        sagittal, coronal, axial = (list(e) for e in figures[0]["extents"])
        assert sagittal == [-12, -12 + 24 * 2, -8, -8 + 16 * 2]  # y and z
        assert coronal == [-10, -10 + 20 * 2, -8, -8 + 16 * 2]  # x and z
        assert axial == [-10, -10 + 20 * 2, -12, -12 + 24 * 2]  # x and y

    def test_intensity_is_the_same_in_all_the_slices(self, tmp_path, t1, figures):
        path, data = t1
        cltqcqa.generate_slices(path, output_path=tmp_path / "a.png", dpi=40)
        signal = data[data != 0].astype(float)
        wanted = (np.percentile(signal, 1), np.percentile(signal, 99))
        for clim in figures[0]["clims"]:
            assert clim == pytest.approx(wanted)

    def test_intensity_percentiles(self, tmp_path, t1, figures):
        path, data = t1
        cltqcqa.generate_slices(
            path, output_path=tmp_path / "a.png", intensity_percentiles=(10, 90), dpi=40
        )
        signal = data[data != 0].astype(float)
        assert figures[0]["clims"][0] == pytest.approx(
            (np.percentile(signal, 10), np.percentile(signal, 90))
        )

    def test_background_value(self, tmp_path, figures):
        path = tmp_path / "bg.nii.gz"
        data = np.full((20, 24, 16), -1.0, dtype=np.float32)
        data[5:15, 6:18, 4:12] = 5.0
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
        cltqcqa.generate_slices(
            path, output_path=tmp_path / "a.png", ignore_value=-1, dpi=40
        )
        assert figures[0]["clims"][0] == pytest.approx((5.0, 5.0))

    def test_image_without_signal_uses_its_whole_range(self, tmp_path, figures):
        path = tmp_path / "empty.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.eye(4)), str(path)
        )
        result = cltqcqa.generate_slices(path, output_path=tmp_path / "a.png", dpi=40)
        assert Path(result).is_file()
        assert figures[0]["clims"][0] == (0.0, 0.0)

    def test_nan_voxels_are_set_to_zero(self, tmp_path, figures):
        path = tmp_path / "nan.nii.gz"
        data = np.ones((8, 8, 8), dtype=np.float32)
        data[0, 0, 0] = np.nan
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
        cltqcqa.generate_slices(
            path, output_path=tmp_path / "a.png", slice_positions=[0, 1], dpi=40
        )
        assert not any(np.isnan(a).any() for a in figures[0]["arrays"])
        # The first sagittal slice holds the NaN voxel
        assert np.rot90(figures[0]["arrays"][0], -1)[0, 0] == 0

    def test_four_dimensional_images_show_the_first_volume(self, tmp_path, figures):
        path = tmp_path / "bold.nii.gz"
        first = make_image(path.with_name("first.nii.gz"), shape=(16, 16, 12), seed=1)
        second = make_image(path.with_name("second.nii.gz"), shape=(16, 16, 12), seed=2)
        nib.save(
            nib.Nifti1Image(np.stack([first, second], axis=-1), np.eye(4)), str(path)
        )
        cltqcqa.generate_slices(
            path, output_path=tmp_path / "a.png", slice_positions=[0.5], dpi=40
        )
        sagittal = figures[0]["arrays"][0]
        slice_index = cltqcqa.get_valid_slices(
            first.astype(float), slice_positions=[0.5]
        )[0][0]
        np.testing.assert_allclose(sagittal, np.rot90(first[slice_index]))

    def test_colormap_and_colors(self, tmp_path, t1, figures):
        cltqcqa.generate_slices(
            t1[0],
            output_path=tmp_path / "a.png",
            cmap="viridis",
            bg_color="navy",
            text_color="yellow",
            dpi=40,
        )
        record = figures[0]
        assert set(record["cmaps"]) == {"viridis"}
        assert record["facecolor"] == matplotlib.colors.to_rgba("navy")
        assert set(record["title_colors"]) == {"yellow"}
        assert record["kwargs"]["facecolor"] == "navy"

    def test_without_the_colorbar(self, tmp_path, t1, figures):
        cltqcqa.generate_slices(
            t1[0], output_path=tmp_path / "a.png", show_colorbar=False, dpi=40
        )
        assert figures[0]["n_axes"] == 9

    def test_colorbar_ticks(self, tmp_path, t1, figures):
        cltqcqa.generate_slices(t1[0], output_path=tmp_path / "a.png", dpi=40)
        colorbar = figures[0]["figure"].axes[-1]
        low, high = figures[0]["clims"][0]
        ticks = colorbar.get_yticks()
        assert ticks == pytest.approx([low, (low + high) / 2, high])

    def test_figure_size(self, tmp_path, t1, figures):
        cltqcqa.generate_slices(t1[0], output_path=tmp_path / "a.png", dpi=40)
        cltqcqa.generate_slices(
            t1[0],
            output_path=tmp_path / "b.png",
            fig_size=(6, 2),
            slice_positions=[0.5],
            dpi=40,
        )
        assert figures[0]["size"] == (18.0, 6.0)  # 2 inches per slice, 9 slices
        assert figures[1]["size"] == (6.0, 2.0)

    def test_resolution(self, tmp_path, t1):
        low = cltqcqa.generate_slices(t1[0], output_path=tmp_path / "low.png", dpi=30)
        high = cltqcqa.generate_slices(t1[0], output_path=tmp_path / "high.png", dpi=60)
        assert read_png(high).shape[1] == pytest.approx(
            2 * read_png(low).shape[1], abs=4
        )

    def test_figures_are_closed(self, tmp_path, t1):
        cltqcqa.generate_slices(t1[0], output_path=tmp_path / "a.png", dpi=40)
        assert plt.get_fignums() == []

    @pytest.mark.parametrize(
        "shape",
        [(8, 8), (8, 8, 1), (1, 8, 8), (4, 4, 4, 2, 2)],
        ids=["2d", "flat-z", "flat-x", "5d"],
    )
    def test_images_that_cannot_be_shown(self, tmp_path, shape, capsys):
        path = tmp_path / "bad.nii.gz"
        nib.save(
            nib.Nifti1Image(np.ones(shape, dtype=np.float32), np.eye(4)), str(path)
        )
        result = cltqcqa.generate_slices(path, remove_invalid=False)
        assert result is None
        assert path.exists()
        assert "Skipping" in capsys.readouterr().out
        assert not path.with_suffix("").with_suffix(".png").exists()

    @pytest.mark.parametrize("shape", [(8, 8), (8, 8, 1), (4, 4, 4, 2, 2)])
    def test_invalid_images_are_removed_with_their_sidecar(self, tmp_path, shape):
        path = tmp_path / "bad.nii.gz"
        nib.save(
            nib.Nifti1Image(np.ones(shape, dtype=np.float32), np.eye(4)), str(path)
        )
        sidecar = tmp_path / "bad.json"
        sidecar.write_text("{}")
        assert cltqcqa.generate_slices(path, remove_invalid=True) is None
        assert not path.exists()
        assert not sidecar.exists()

    def test_invalid_images_are_kept_by_default(self, tmp_path):
        path = tmp_path / "bad.nii.gz"
        nib.save(
            nib.Nifti1Image(np.ones((8, 8, 1), dtype=np.float32), np.eye(4)), str(path)
        )
        sidecar = tmp_path / "bad.json"
        sidecar.write_text("{}")
        assert cltqcqa.generate_slices(path) is None  # kept by default
        assert path.exists()
        assert sidecar.exists()
        assert cltqcqa.generate_slices(path, remove_invalid=False) is None
        assert path.exists()

    def test_the_output_name_only_changes_the_extension(self, tmp_path):
        folder = tmp_path / "study.nii"
        path = folder / "T1w.nii"
        make_image(path)
        result = cltqcqa.generate_slices(path, dpi=40)
        assert result == str(folder / "T1w.png")


####################################################################################################
class TestProcessSingleFile:
    def args(self, file_path, input_folder, output_folder=None, overwrite=False):
        return (file_path, input_folder, output_folder, [0.5], None, overwrite)

    def test_output_next_to_the_image(self, tmp_path):
        make_image(tmp_path / "sub-01" / "T1w.nii.gz")
        success, result = cltqcqa._process_single_file(
            self.args(str(tmp_path / "sub-01" / "T1w.nii.gz"), tmp_path)
        )
        assert success is True
        assert result == str(tmp_path / "sub-01" / "T1w.png")

    def test_output_folder_keeps_the_structure(self, tmp_path):
        make_image(tmp_path / "data" / "sub-01" / "ses-1" / "T1w.nii.gz")
        out = tmp_path / "figures"
        success, result = cltqcqa._process_single_file(
            self.args(
                str(tmp_path / "data" / "sub-01" / "ses-1" / "T1w.nii.gz"),
                tmp_path / "data",
                out,
            )
        )
        assert success is True
        assert result == str(out / "sub-01" / "ses-1" / "T1w.png")
        assert Path(result).is_file()

    def test_image_at_the_top_of_the_input_folder(self, tmp_path):
        make_image(tmp_path / "data" / "T1w.nii")
        out = tmp_path / "figures"
        out.mkdir()
        success, result = cltqcqa._process_single_file(
            self.args(str(tmp_path / "data" / "T1w.nii"), tmp_path / "data", out)
        )
        assert result == str(out / "T1w.png")

    def test_other_extensions(self, tmp_path):
        data = make_image(tmp_path / "tmp.nii")
        mgz = tmp_path / "data" / "aseg.mgz"
        mgz.parent.mkdir()
        nib.save(nib.MGHImage(data, np.eye(4)), str(mgz))
        out = tmp_path / "figures"
        out.mkdir()
        _, result = cltqcqa._process_single_file(
            self.args(str(mgz), tmp_path / "data", out)
        )
        assert result == str(out / "aseg.png")

    def test_invalid_files_are_reported(self, tmp_path):
        path = tmp_path / "broken.nii.gz"
        path.write_text("nothing")
        success, message = cltqcqa._process_single_file(self.args(str(path), tmp_path))
        assert success is False
        assert message.startswith(f"Invalid file {path}")

    def test_images_that_cannot_be_shown(self, tmp_path):
        path = tmp_path / "flat.nii.gz"
        nib.save(
            nib.Nifti1Image(np.ones((6, 6, 1), dtype=np.float32), np.eye(4)), str(path)
        )
        assert cltqcqa._process_single_file(self.args(str(path), tmp_path)) == (
            True,
            None,
        )

    def test_existing_png_is_kept(self, tmp_path):
        path = tmp_path / "T1w.nii.gz"
        make_image(path)
        png = tmp_path / "T1w.png"
        png.write_bytes(b"old")
        success, result = cltqcqa._process_single_file(self.args(str(path), tmp_path))
        assert (success, result) == (True, str(png))
        assert png.read_bytes() == b"old"

    def test_overwrite(self, tmp_path):
        path = tmp_path / "T1w.nii.gz"
        make_image(path)
        png = tmp_path / "T1w.png"
        png.write_bytes(b"old")
        cltqcqa._process_single_file(self.args(str(path), tmp_path, overwrite=True))
        assert png.read_bytes()[:8] == PNG_MAGIC


####################################################################################################
@pytest.fixture
def dataset(tmp_path):
    """Small BIDS-like dataset with anatomical and functional images."""
    root = tmp_path / "bids"
    for sub in ("sub-01", "sub-02"):
        make_image(root / sub / "anat" / f"{sub}_T1w.nii.gz")
        make_image(root / sub / "anat" / f"{sub}_T2w.nii.gz", seed=3)
        make_image(
            root / sub / "func" / f"{sub}_task-rest_bold.nii.gz", shape=(16, 16, 12, 3)
        )
        (root / sub / "anat" / f"{sub}_T1w.json").write_text(
            json.dumps({"SeriesDescription": "t1"})
        )
    (root / "README.txt").write_text("a dataset")
    return root


def names(paths):
    return sorted(Path(p).name for p in paths)


class TestRecursivelyGenerateSlices:
    def test_one_png_per_image_next_to_the_images(self, dataset):
        result = cltqcqa.recursively_generate_slices(dataset, verbose=False)
        assert names(result) == [
            "sub-01_T1w.png", "sub-01_T2w.png", "sub-01_task-rest_bold.png",
            "sub-02_T1w.png", "sub-02_T2w.png", "sub-02_task-rest_bold.png",
        ]  # fmt: skip
        assert all(isinstance(p, str) for p in result)
        assert (dataset / "sub-01" / "anat" / "sub-01_T1w.png").is_file()
        assert (dataset / "sub-02" / "func" / "sub-02_task-rest_bold.png").is_file()

    def test_other_files_are_ignored(self, dataset):
        cltqcqa.recursively_generate_slices(dataset, verbose=False)
        assert not (dataset / "README.png").exists()
        assert len(list(dataset.rglob("*.png"))) == 6

    def test_output_folder_keeps_the_structure(self, tmp_path, dataset):
        out = tmp_path / "qc"
        result = cltqcqa.recursively_generate_slices(
            dataset, output_folder=out, verbose=False
        )
        assert sorted(
            str(Path(p).relative_to(out)).replace("\\", "/") for p in result
        ) == [
            "sub-01/anat/sub-01_T1w.png",
            "sub-01/anat/sub-01_T2w.png",
            "sub-01/func/sub-01_task-rest_bold.png",
            "sub-02/anat/sub-02_T1w.png",
            "sub-02/anat/sub-02_T2w.png",
            "sub-02/func/sub-02_task-rest_bold.png",
        ]
        assert not list(dataset.rglob("*.png"))

    def test_and_filter(self, dataset):
        result = cltqcqa.recursively_generate_slices(
            dataset, and_filter="T1w", verbose=False
        )
        assert names(result) == ["sub-01_T1w.png", "sub-02_T1w.png"]

    def test_and_filter_with_several_words(self, dataset):
        result = cltqcqa.recursively_generate_slices(
            dataset, and_filter=["sub-02", "T2w"], verbose=False
        )
        assert names(result) == ["sub-02_T2w.png"]

    def test_or_filter(self, dataset):
        result = cltqcqa.recursively_generate_slices(
            dataset, or_filter=["T1w", "bold"], verbose=False
        )
        assert len(result) == 4

    def test_file_extensions(self, dataset):
        data = make_image(dataset / "tmp.nii")
        (dataset / "tmp.nii").unlink()
        nib.save(nib.MGHImage(data, np.eye(4)), str(dataset / "aseg.mgz"))
        mgz_only = cltqcqa.recursively_generate_slices(
            dataset, file_extensions=[".mgz"], verbose=False
        )
        assert names(mgz_only) == ["aseg.png"]

    def test_freesurfer_volumes_are_included_by_default(self, tmp_path):
        data = make_image(tmp_path / "tmp.nii")
        (tmp_path / "tmp.nii").unlink()
        nib.save(nib.MGHImage(data, np.eye(4)), str(tmp_path / "T1.mgz"))
        assert names(cltqcqa.recursively_generate_slices(tmp_path, verbose=False)) == [
            "T1.png"
        ]

    def test_the_number_of_jobs_selects_the_processes(self, dataset, monkeypatch):
        from concurrent.futures import ThreadPoolExecutor

        workers = []

        class RecordingExecutor(ThreadPoolExecutor):
            def __init__(self, max_workers=None, **kwargs):
                workers.append(max_workers)
                super().__init__(max_workers=max_workers)

        monkeypatch.setattr(cltqcqa, "ProcessPoolExecutor", RecordingExecutor)
        cltqcqa.recursively_generate_slices(
            dataset, and_filter="T1w", n_jobs=1, verbose=False
        )
        assert workers == []
        result = cltqcqa.recursively_generate_slices(
            dataset, and_filter="T1w", n_jobs=3, overwrite=True, verbose=False
        )
        assert workers == [3]
        assert names(result) == ["sub-01_T1w.png", "sub-02_T1w.png"]

    def test_extensions_are_case_insensitive(self, tmp_path):
        make_image(tmp_path / "T1W.NII.GZ")
        result = cltqcqa.recursively_generate_slices(tmp_path, verbose=False)
        assert len(result) == 1

    def test_slice_positions_are_used(self, dataset, figures):
        cltqcqa.recursively_generate_slices(
            dataset, and_filter="sub-01_T1w", slice_positions=[0.5], verbose=False
        )
        assert len(figures[0]["panels"]) == 3

    def test_existing_pngs_are_kept_unless_overwriting(self, dataset):
        cltqcqa.recursively_generate_slices(
            dataset, and_filter="sub-01_T1w", verbose=False
        )
        png = dataset / "sub-01" / "anat" / "sub-01_T1w.png"
        png.write_bytes(b"old")
        again = cltqcqa.recursively_generate_slices(
            dataset, and_filter="sub-01_T1w", verbose=False
        )
        assert again == [str(png)]
        assert png.read_bytes() == b"old"
        cltqcqa.recursively_generate_slices(
            dataset, and_filter="sub-01_T1w", overwrite=True, verbose=False
        )
        assert png.read_bytes()[:8] == PNG_MAGIC

    def test_parallel_processing(self, dataset):
        result = cltqcqa.recursively_generate_slices(
            dataset, and_filter="T1w", n_jobs=2, verbose=False
        )
        assert names(result) == ["sub-01_T1w.png", "sub-02_T1w.png"]
        assert all(Path(p).read_bytes()[:8] == PNG_MAGIC for p in result)

    def test_corrupt_images_do_not_stop_the_others(self, dataset):
        (dataset / "sub-01" / "anat" / "sub-01_T1w.nii.gz").write_text("corrupt")
        result = cltqcqa.recursively_generate_slices(
            dataset, and_filter="anat", verbose=False
        )
        assert "sub-01_T1w.png" not in names(result)
        assert len(result) == 3

    def test_images_that_cannot_be_shown_are_not_deleted(self, dataset):
        flat = dataset / "sub-01" / "anat" / "scout.nii.gz"
        nib.save(
            nib.Nifti1Image(np.ones((6, 6, 1), dtype=np.float32), np.eye(4)), str(flat)
        )
        cltqcqa.recursively_generate_slices(dataset, and_filter="scout", verbose=False)
        assert flat.exists()

    def test_images_that_cannot_be_shown_are_left_out(self, dataset):
        nib.save(
            nib.Nifti1Image(np.ones((6, 6, 1), dtype=np.float32), np.eye(4)),
            str(dataset / "sub-01" / "anat" / "sub-01_T1w.nii.gz"),
        )
        result = cltqcqa.recursively_generate_slices(
            dataset, and_filter="sub-01_T1w", verbose=False
        )
        assert result == []

    def test_folder_without_images(self, tmp_path):
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "README.txt").write_text("no images here")
        assert (
            cltqcqa.recursively_generate_slices(tmp_path / "docs", verbose=False) == []
        )

    def test_empty_folder(self, tmp_path):
        (tmp_path / "empty").mkdir()
        assert (
            cltqcqa.recursively_generate_slices(tmp_path / "empty", verbose=False) == []
        )

    def test_string_paths(self, tmp_path, dataset):
        result = cltqcqa.recursively_generate_slices(
            str(dataset),
            output_folder=str(tmp_path / "qc"),
            and_filter="sub-02_T1w",
            verbose=False,
        )
        assert names(result) == ["sub-02_T1w.png"]

    def test_missing_input_folder(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Input folder not found"):
            cltqcqa.recursively_generate_slices(tmp_path / "missing")

    def test_output_folder_is_created_even_without_images(self, tmp_path):
        (tmp_path / "in").mkdir()
        (tmp_path / "in" / "README.txt").write_text("no images here")
        cltqcqa.recursively_generate_slices(
            tmp_path / "in", output_folder=tmp_path / "new" / "out", verbose=False
        )
        assert (tmp_path / "new" / "out").is_dir()

    def test_only_the_top_folder_when_not_recursive(self, tmp_path, dataset):
        make_image(dataset / "top_T1w.nii.gz")
        result = cltqcqa.recursively_generate_slices(
            dataset, recursive=False, verbose=False
        )
        assert names(result) == ["top_T1w.png"]


####################################################################################################
def make_pngs(root, relative_paths):
    for rel in relative_paths:
        path = Path(root) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(PNG_MAGIC + b"data")


def checkboxes(html):
    return re.findall(r'<input type="checkbox" id="([^"]*)" value="([^"]*)">', html)


class TestGenerateImageSelectionWebpage:
    def test_default_page(self, tmp_path):
        make_pngs(
            tmp_path, ["sub-01/anat/a.png", "sub-01/func/b.png", "sub-02/anat/c.png"]
        )
        result = cltqcqa.generate_image_selection_webpage(tmp_path)
        assert result == str(tmp_path / "png_selection.html")
        html = Path(result).read_text(encoding="utf-8")
        assert html.startswith("<!DOCTYPE html>")
        assert "<title>PNG Image Selection</title>" in html
        assert "<h1>PNG Image Selection</h1>" in html
        assert str(tmp_path) in html
        assert [c[0] for c in checkboxes(html)] == ["a", "b", "c"]

    def test_images_are_linked_by_their_relative_path(self, tmp_path):
        make_pngs(tmp_path, ["sub-01/anat/a.png"])
        html = Path(cltqcqa.generate_image_selection_webpage(tmp_path)).read_text(
            encoding="utf-8"
        )
        assert re.findall(r'<img src="([^"]*)"', html) == ["sub-01/anat/a.png"]

    def test_folders_become_nested_headings(self, tmp_path):
        make_pngs(tmp_path, ["sub-01/ses-1/anat/a.png"])
        html = Path(cltqcqa.generate_image_selection_webpage(tmp_path)).read_text(
            encoding="utf-8"
        )
        positions = [
            html.index("<h2>📂 sub-01</h2>"),
            html.index("<h3>📂 ses-1</h3>"),
            html.index("<h4>📂 anat</h4>"),
            html.index('id="a"'),
        ]
        assert positions == sorted(positions)
        assert "📁 sub-01 / ses-1 / anat" in html

    def test_headings_stop_at_the_fifth_level(self, tmp_path):
        make_pngs(tmp_path, ["a/b/c/d/e/f/img.png"])
        html = Path(cltqcqa.generate_image_selection_webpage(tmp_path)).read_text(
            encoding="utf-8"
        )
        assert "<h5>📂 e</h5>" in html
        assert "<h5>📂 f</h5>" in html
        assert "<h6" not in html

    def test_images_in_the_root_folder(self, tmp_path):
        make_pngs(tmp_path, ["top.png"])
        html = Path(cltqcqa.generate_image_selection_webpage(tmp_path)).read_text(
            encoding="utf-8"
        )
        assert [c[0] for c in checkboxes(html)] == ["top"]
        assert "📁 Root" in html

    def test_values_are_the_original_images_when_they_are_found(self, tmp_path):
        make_pngs(tmp_path, ["sub-01/a.png", "sub-01/b.png", "sub-01/c.png"])
        (tmp_path / "sub-01" / "a.nii.gz").write_text("image")
        (tmp_path / "sub-01" / "b.nii").write_text("image")
        html = Path(cltqcqa.generate_image_selection_webpage(tmp_path)).read_text(
            encoding="utf-8"
        )
        values = dict(checkboxes(html))
        assert values["a"] == str(tmp_path / "sub-01" / "a.nii.gz")
        assert values["b"] == str(tmp_path / "sub-01" / "b.nii")
        assert values["c"] == str(tmp_path / "sub-01" / "c.png")

    def test_descriptions_come_from_the_sidecars(self, tmp_path):
        make_pngs(tmp_path, ["a.png", "b.png", "c.png"])
        (tmp_path / "a.json").write_text(json.dumps({"SeriesDescription": "mprage"}))
        (tmp_path / "b.json").write_text(json.dumps({"Description": "a mask"}))
        (tmp_path / "c.json").write_text(
            json.dumps({"SeriesDescription": "ignored when invalid"})[:-3]
        )
        html = Path(cltqcqa.generate_image_selection_webpage(tmp_path)).read_text(
            encoding="utf-8"
        )
        descriptions = re.findall(r'<div class="description">📝 ([^<]*)</div>', html)
        assert sorted(descriptions) == ["a mask", "mprage"]

    def test_series_description_wins_over_description(self, tmp_path):
        make_pngs(tmp_path, ["a.png"])
        (tmp_path / "a.json").write_text(
            json.dumps({"SeriesDescription": "first", "Description": "second"})
        )
        html = Path(cltqcqa.generate_image_selection_webpage(tmp_path)).read_text(
            encoding="utf-8"
        )
        assert "📝 first" in html and "second" not in html

    def test_descriptions_can_be_hidden(self, tmp_path):
        make_pngs(tmp_path, ["a.png"])
        (tmp_path / "a.json").write_text(json.dumps({"SeriesDescription": "mprage"}))
        html = Path(
            cltqcqa.generate_image_selection_webpage(tmp_path, show_descriptions=False)
        ).read_text(encoding="utf-8")
        assert "mprage" not in html
        assert 'class="description"' not in html.split("</style>")[1]

    def test_title_and_image_width(self, tmp_path):
        make_pngs(tmp_path, ["a.png"])
        html = Path(
            cltqcqa.generate_image_selection_webpage(
                tmp_path, title="Quality control", image_width="800px"
            )
        ).read_text(encoding="utf-8")
        assert "<title>Quality control</title>" in html
        assert "width: 800px;" in html

    def test_output_file(self, tmp_path):
        make_pngs(tmp_path, ["a.png"])
        out = tmp_path / "reports" / "qc.html"
        out.parent.mkdir()
        result = cltqcqa.generate_image_selection_webpage(
            tmp_path, output_html=str(out)
        )
        assert result == str(out)
        assert out.is_file()
        assert not (tmp_path / "png_selection.html").exists()

    def test_existing_page_is_kept_without_overwrite(self, tmp_path, capsys):
        make_pngs(tmp_path, ["a.png"])
        page = tmp_path / "png_selection.html"
        page.write_text("old")
        assert cltqcqa.generate_image_selection_webpage(tmp_path) == str(page)
        assert page.read_text() == "old"
        assert "already exists" in capsys.readouterr().out

    def test_existing_page_is_replaced_with_overwrite(self, tmp_path):
        make_pngs(tmp_path, ["a.png"])
        page = tmp_path / "png_selection.html"
        page.write_text("old")
        cltqcqa.generate_image_selection_webpage(tmp_path, overwrite=True)
        assert page.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    def test_pattern(self, tmp_path):
        make_pngs(tmp_path, ["sub-01/a.png", "sub-02/b.png"])
        html = Path(
            cltqcqa.generate_image_selection_webpage(
                tmp_path, png_pattern="sub-01/*.png"
            )
        ).read_text(encoding="utf-8")
        assert [c[0] for c in checkboxes(html)] == ["a"]

    def test_not_recursive(self, tmp_path):
        make_pngs(tmp_path, ["top.png", "sub-01/deep.png"])
        html = Path(
            cltqcqa.generate_image_selection_webpage(tmp_path, recursive=False)
        ).read_text(encoding="utf-8")
        assert [c[0] for c in checkboxes(html)] == ["top"]

    def test_controls_and_scripts(self, tmp_path):
        make_pngs(tmp_path, ["a.png"])
        html = Path(cltqcqa.generate_image_selection_webpage(tmp_path)).read_text(
            encoding="utf-8"
        )
        assert html.count('onclick="selectAll()"') == 2
        assert html.count('onclick="selectNone()"') == 2
        assert html.count('onclick="downloadChecked()"') == 2
        for function in ("downloadChecked", "selectAll", "selectNone", "updateStats"):
            assert f"function {function}()" in html
        assert 'a.download = "selected_images.txt"' in html
        assert html.rstrip().endswith("</html>")

    def test_no_images(self, tmp_path, capsys):
        result = cltqcqa.generate_image_selection_webpage(tmp_path)
        assert result == str(tmp_path / "png_selection.html")
        assert "No PNG files found" in capsys.readouterr().out
        assert not Path(result).exists()

    def test_unicode_titles(self, tmp_path):
        make_pngs(tmp_path, ["a.png"])
        html = Path(
            cltqcqa.generate_image_selection_webpage(
                tmp_path, title="Contrôle qualité ✓"
            )
        ).read_text(encoding="utf-8")
        assert "Contrôle qualité ✓" in html

    def test_a_folder_called_files(self, tmp_path):
        make_pngs(tmp_path, ["files/a.png", "other/b.png"])
        html = Path(cltqcqa.generate_image_selection_webpage(tmp_path)).read_text(
            encoding="utf-8"
        )
        assert sorted(c[0] for c in checkboxes(html)) == ["a", "b"]
        assert "📂 files" in html

    def test_special_characters_are_escaped(self, tmp_path):
        make_pngs(tmp_path, ["a.png"])
        (tmp_path / "a.json").write_text(
            json.dumps({"SeriesDescription": "T1 <b>bold</b> & more"})
        )
        html = Path(
            cltqcqa.generate_image_selection_webpage(tmp_path, title="A & B <i>")
        ).read_text(encoding="utf-8")
        assert "<b>bold</b>" not in html
        assert "T1 &lt;b&gt;bold&lt;/b&gt; &amp; more" in html
        assert "<title>A &amp; B &lt;i&gt;</title>" in html

    @pytest.mark.skipif(
        os.name == "nt", reason="Windows file names cannot hold these characters"
    )
    def test_special_characters_of_the_paths_are_escaped(self, tmp_path):
        root = tmp_path / "r&d <lab>"
        make_pngs(root, ['sub <1> & co/my "scan" & more.png'])
        (root / "sub <1> & co" / 'my "scan" & more.nii.gz').write_text("image")
        out = cltqcqa.generate_image_selection_webpage(root, title="Review & <check>")
        html = Path(out).read_text(encoding="utf-8")
        body = html.split("</style>")[1]
        assert "<lab>" not in body and "r&amp;d &lt;lab&gt;" in body
        assert "<h1>Review &amp; &lt;check&gt;</h1>" in body
        assert "<h2>📂 sub &lt;1&gt; &amp; co</h2>" in body
        assert "📁 sub &lt;1&gt; &amp; co" in body
        assert "<strong>my &quot;scan&quot; &amp; more</strong>" in body
        # The value of the checkbox and the id are safe inside their attributes
        ((image_id, value),) = checkboxes(body)
        assert "&amp;" in value and '"' not in value
        assert re.fullmatch(r"[A-Za-z0-9_.:-]+", image_id)
        assert f"document.getElementById('{image_id}')" in body

    def test_images_with_the_same_name_get_different_ids(self, tmp_path):
        make_pngs(tmp_path, ["sub-01/slices.png", "sub-02/slices.png"])
        html = Path(cltqcqa.generate_image_selection_webpage(tmp_path)).read_text(
            encoding="utf-8"
        )
        ids = [c[0] for c in checkboxes(html)]
        assert len(ids) == len(set(ids)) == 2
        # Clicking an image toggles its own checkbox
        for image_id in ids:
            assert f"document.getElementById('{image_id}')" in html

    def test_images_are_sorted_by_name(self, tmp_path, monkeypatch):
        make_pngs(tmp_path, ["a.png", "b.png", "c.png"])
        original = Path.glob
        monkeypatch.setattr(
            Path,
            "glob",
            lambda self, pattern: iter(sorted(original(self, pattern), reverse=True)),
        )
        html = Path(cltqcqa.generate_image_selection_webpage(tmp_path)).read_text(
            encoding="utf-8"
        )
        assert [c[0] for c in checkboxes(html)] == ["a", "b", "c"]


class TestCreatePngWebpageFromGeneratedSlices:
    def test_page_of_generated_slices(self, tmp_path, dataset):
        out = tmp_path / "qc"
        pngs = cltqcqa.recursively_generate_slices(
            dataset, output_folder=out, verbose=False
        )
        result = cltqcqa.create_png_webpage_from_generated_slices(out)
        assert result == str(out / "png_selection.html")
        html = Path(result).read_text(encoding="utf-8")
        assert "<title>Generated Brain Slice Images Selection</title>" in html
        assert len(checkboxes(html)) == len(pngs) == 6
        assert "width: 1200px;" in html

    def test_descriptions_of_the_images_next_to_the_slices(self, dataset):
        cltqcqa.recursively_generate_slices(dataset, verbose=False)
        html = Path(
            cltqcqa.create_png_webpage_from_generated_slices(dataset)
        ).read_text(encoding="utf-8")
        assert html.count("📝 t1") == 2
        values = dict(checkboxes(html))
        assert values["sub-01_T1w"] == str(
            dataset / "sub-01" / "anat" / "sub-01_T1w.nii.gz"
        )

    def test_output_file_and_overwrite(self, tmp_path):
        make_pngs(tmp_path, ["a.png"])
        out = tmp_path / "review.html"
        cltqcqa.create_png_webpage_from_generated_slices(tmp_path, output_html=out)
        out.write_text("old")
        assert cltqcqa.create_png_webpage_from_generated_slices(
            tmp_path, output_html=out
        ) == str(out)
        assert out.read_text() == "old"
        cltqcqa.create_png_webpage_from_generated_slices(
            tmp_path, output_html=out, overwrite=True
        )
        assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
