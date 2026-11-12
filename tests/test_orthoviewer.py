"""Tests for the orthogonal viewer of visualizationtools (rendered off-screen)."""

import nibabel as nib
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pv.OFF_SCREEN = True

import clabtoolkit.visualization_utils as visutils  # noqa: E402
from clabtoolkit.visualizationtools import OrthoViewer, plot_volume  # noqa: E402


@pytest.fixture
def volumes():
    x, y, z = np.mgrid[:20, :24, :16]
    img = (
        np.exp(-(((x - 10) / 6) ** 2 + ((y - 12) / 8) ** 2 + ((z - 8) / 5) ** 2)) * 1000
    )
    parc = np.zeros(img.shape, dtype=np.int32)
    parc[(x < 10) & (img > 300)] = 1
    parc[(x >= 10) & (img > 300)] = 2
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [-20, -24, -16]
    return img.astype(np.float32), parc, affine


def _viewer(**kw):
    return OrthoViewer(off_screen=True, window_size=(600, 450), **kw)


def test_requires_an_input():
    with pytest.raises(ValueError):
        OrthoViewer(off_screen=True)


def test_image_only(volumes):
    img, _, aff = volumes
    v = _viewer(image=nib.Nifti1Image(img, aff))
    assert v._parc is None and v._img is not None
    assert v.cursor == pytest.approx((0.0, 0.0, 0.0))
    v.close()


def test_parcellation_only_without_color_table(volumes):
    _, parc, aff = volumes
    v = _viewer(parcellation=parc, parcellation_affine=aff)
    assert v._img is None and v._rgba_table.shape[0] == 3  # transparent + 2 labels
    v.close()


def test_parcellation_with_color_table_dict(volumes):
    _, parc, aff = volumes
    ctab = {"index": [1, 2], "name": ["L", "R"], "color": ["#ff0000", "#00ff00"]}
    v = _viewer(parcellation=parc, parcellation_affine=aff, color_table=ctab)
    assert tuple(v._rgba_table[1][:3]) == (255, 0, 0)
    assert v._label_names[2] == "R"
    v.close()


def test_both_with_different_grids_are_resampled(volumes):
    img, parc, aff = volumes
    aff_parc = aff.copy()
    aff_parc[:3, :3] *= 2  # coarser parcellation grid covering the same field of view
    coarse = parc[::2, ::2, ::2]
    v = _viewer(
        image=nib.Nifti1Image(img, aff), parcellation=nib.Nifti1Image(coarse, aff_parc)
    )
    assert v._parc.shape == v._img.shape
    v.close()


def test_set_cursor_snaps_and_clips(volumes):
    img, parc, aff = volumes
    v = _viewer(image=img, parcellation=parc, image_affine=aff, parcellation_affine=aff)
    v.set_cursor((1000, -1000, 0.4))
    x, y, z = v.cursor
    assert (
        x == pytest.approx(-20 + 19 * 2)
        and y == pytest.approx(-24)
        and z == pytest.approx(0.0)
    )
    with pytest.raises(ValueError):
        v.set_cursor((1, 2))
    v.close()


def test_pick_moves_in_plane_axes_only(volumes):
    img, _, aff = volumes
    v = _viewer(image=img, image_affine=aff)
    before = v.cursor
    v.pick(2, 150, 340)  # axial view (top-left panel)
    after = v.cursor
    assert after[2] == before[2]
    v.close()


def test_toggles_and_screenshot(volumes, tmp_path):
    img, parc, aff = volumes
    v = _viewer(image=img, parcellation=parc, image_affine=aff, parcellation_affine=aff)
    v.toggle_labels()
    assert not v._lab_actors[0].GetVisibility()
    v.toggle_crosshair()
    out = tmp_path / "shot.png"
    v.screenshot(out)
    assert out.is_file()
    v.close()


def test_plot_volume_without_showing(volumes):
    img, _, aff = volumes
    v = plot_volume(img, image_affine=aff, show=False, off_screen=True)
    assert isinstance(v, OrthoViewer)
    v.close()


def test_colors_of_the_color_table():
    assert visutils.ortho_to_rgb01("#ff0000") == (1.0, 0.0, 0.0)
    assert visutils.ortho_to_rgb01([255, 0, 0]) == (1.0, 0.0, 0.0)
    assert visutils.ortho_to_rgb01((0.0, 0.5, 1.0)) == (0.0, 0.5, 1.0)
    with pytest.raises(ValueError):
        visutils.ortho_to_rgb01([1, 2])


def test_load_image(volumes, tmp_path):
    img, _, aff = volumes
    path = tmp_path / "img.nii.gz"
    nib.save(nib.Nifti1Image(img, aff), str(path))
    loaded = visutils.ortho_load_image(path, None)
    assert loaded.shape == img.shape and np.allclose(loaded.affine, aff)
    assert np.allclose(visutils.ortho_load_image(img, None).affine, np.eye(4))
    with pytest.warns(UserWarning, match="4D"):
        first = visutils.ortho_load_image(np.stack([img, img * 2], axis=-1), aff)
    assert np.array_equal(first.get_fdata(), img)
    with pytest.raises(FileNotFoundError):
        visutils.ortho_load_image(tmp_path / "none.nii.gz", None)
    with pytest.raises(ValueError):
        visutils.ortho_load_image(np.zeros((4, 4)), None)
    with pytest.raises(TypeError):
        visutils.ortho_load_image(5, None)


def test_canonical_reorients_to_ras(volumes):
    img, _, _ = volumes
    flipped = np.diag([-2.0, 2.0, 2.0, 1.0])
    data, affine = visutils.ortho_canonical(nib.Nifti1Image(img, flipped))
    assert affine[0, 0] > 0
    assert np.array_equal(data, img[::-1])


def test_default_clim():
    assert visutils.ortho_default_clim(np.zeros((4, 4, 4))) == (0.0, 1.0)
    lo, hi = visutils.ortho_default_clim(np.arange(1000.0).reshape(10, 10, 10))
    assert 0 <= lo < hi <= 999


def test_view_conventions():
    assert visutils.ortho_edge_labels(2) == ("L", "R", "A", "P")
    assert visutils.ortho_edge_labels(1, radiological=True) == ("R", "L", "S", "I")
    assert visutils.ortho_edge_labels(0) == ("A", "P", "S", "I")
    assert visutils.ortho_camera_side(2) == 1
    assert visutils.ortho_camera_side(2, radiological=True) == -1
    assert visutils.ortho_camera_side(1) == -1
    assert visutils.ortho_camera_side(0) == -1
    assert visutils.ortho_set_axis((1, 2, 3), 1, 9) == [1, 9.0, 3]


def test_color_lookup_with_labels_missing_from_the_table():
    parc = np.array([[[0, 1], [2, 7]]])
    table = {"index": [1, 2], "name": ["a", "b"], "color": ["#ff0000", "#00ff00"]}
    table["opacity"] = [1.0, 0.5]
    rows, rgba, names = visutils.ortho_color_lookup(parc, table)
    assert rows.tolist() == [[[0, 1], [2, 3]]]
    assert rgba.shape == (4, 4) and tuple(rgba[0]) == (0, 0, 0, 0)
    assert tuple(rgba[1]) == (255, 0, 0, 255) and rgba[2][3] == 128
    assert names == {1: "a", 2: "b", 7: "Region_7"}


def test_layout_without_the_3d_view(volumes, tmp_path):
    img, parc, aff = volumes
    assert visutils.ortho_view_layout() == ((2, 2), {2: (0, 0), 1: (0, 1), 0: (1, 0)})
    assert visutils.ortho_view_layout(False) == (
        (1, 3),
        {2: (0, 0), 1: (0, 1), 0: (0, 2)},
    )

    with_3d = OrthoViewer(image=img, image_affine=aff, off_screen=True)
    assert with_3d.plotter.shape == (2, 2) and 3 in with_3d._renderers
    assert tuple(with_3d.plotter.window_size) == (1200, 900)
    with_3d.close()

    v = OrthoViewer(
        image=img,
        parcellation=parc,
        image_affine=aff,
        parcellation_affine=aff,
        view3d=False,
        off_screen=True,
    )
    assert v.plotter.shape == (1, 3)
    assert sorted(v._renderers) == [0, 1, 2]
    assert tuple(v.plotter.window_size) == (1500, 520)
    # One image plane and one parcellation plane per view, none for the 3D view
    assert len(v._lab_actors) == 3
    v.set_cursor((0.0, 0.0, 0.0))
    assert "xyz (mm): 0.0, 0.0, 0.0" in v._info_text.GetInput()
    before = v.cursor
    v.pick(0, 1250, 260)  # sagittal view (right panel)
    assert v.cursor[0] == before[0]
    out = tmp_path / "row.png"
    v.screenshot(out)
    assert out.is_file()
    v.close()


def test_closed_viewer_keeps_its_cursor(volumes):
    img, _, aff = volumes
    v = _viewer(image=img, image_affine=aff)
    v.close()
    v.close()  # closing twice is harmless
    v.set_cursor((4.0, 4.0, 4.0))
    assert v.cursor == pytest.approx((4.0, 4.0, 4.0))
    v.toggle_crosshair()


def test_parcellation_given_as_the_image(volumes, tmp_path):
    from clabtoolkit.parcellationtools import Parcellation

    img, parc, aff = volumes
    ctab = {"index": [1, 2], "name": ["L", "R"], "color": ["#ff0000", "#00ff00"]}

    # A Parcellation object as the first input
    v = _viewer(image=Parcellation(parc, color_table=ctab, affine=aff))
    assert v._img is None and v._label_names == {1: "L", 2: "R"}
    assert v.parc_opacity == 1.0
    assert tuple(v._rgba_table[2][:3]) == (0, 255, 0)
    v.close()

    # A volume with a color table and no other volume
    path = tmp_path / "parc.nii.gz"
    nib.save(nib.Nifti1Image(parc, aff), str(path))
    v = plot_volume(path, color_table=ctab, show=False, off_screen=True)
    assert v._img is None and tuple(v._rgba_table[1][:3]) == (255, 0, 0)
    v.close()

    # Without a color table the volume is an intensity image
    v = _viewer(image=nib.Nifti1Image(parc.astype(np.float32), aff))
    assert v._img is not None and v._parc is None
    v.close()

    with pytest.raises(TypeError, match="Parcellation object"):
        _viewer(image=Parcellation(parc, affine=aff), parcellation=parc)
