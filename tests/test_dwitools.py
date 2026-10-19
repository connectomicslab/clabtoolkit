"""Tests for clabtoolkit.dwitools."""

import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import pyvista as pv

from clabtoolkit.dwitools import (
    DiffusionScheme,
    delete_dwi_volumes,
    get_b0s,
    maps_from_tensor_eigenvalues,
)

# Multi-shell acquisition: interleaved B0s, two shells and two trailing B0s
BVALS = np.array([0, 1000, 1000, 1000, 0, 2000, 2000, 2000, 1000, 2000, 0, 0])
AFFINE = np.array(
    [[1.5, 0, 0, 10.0], [0, 1.5, 0, 20.0], [0, 0, 1.5, 30.0], [0, 0, 0, 1.0]]
)
MULTISHELL = {1000: 32, 2000: 64, 3000: 96}
ALL_MAPS = ["AD", "CL", "CP", "CS", "FA", "GA", "MD", "RA", "RD", "VF"]
GLYPH_POINTS = pv.Sphere(theta_resolution=16, phi_resolution=16).n_points
BVAL_ARRAY = "b-value (s/mm²)"


####################################################################################################
# Fixtures and helpers
####################################################################################################
@pytest.fixture(autouse=True)
def _quiet():
    """Silence the warnings and the progress messages of the functions."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="session")
def can_render():
    """Skip the tests that need an off-screen rendering backend when there is none."""
    try:
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(pv.Sphere())
        plotter.screenshot()
        plotter.close()
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"Off-screen rendering is not available: {exc}")


@pytest.fixture(scope="module")
def multishell_scheme():
    """Simulated multi-shell scheme: 8 B0s and 32, 64 and 96 directions."""
    return DiffusionScheme.simulate_dwi_acq_scheme(
        "shelled", shells=MULTISHELL, n_b0s=8
    )


@pytest.fixture(scope="module")
def scheme():
    """Small simulated scheme: 2 B0s and 16 directions in each of two shells."""
    return DiffusionScheme.simulate_dwi_acq_scheme(
        "shelled", shells={1000: 16, 2000: 16}, n_b0s=2, n_iter=10
    )


def unit_bvecs(n, zero=None):
    """Unit gradient vectors (3 x n), with zero vectors where ``zero`` is True."""
    vecs = np.random.default_rng(0).normal(size=(3, n))
    vecs /= np.linalg.norm(vecs, axis=0)
    if zero is not None:
        vecs[:, zero] = 0
    return vecs


def make_dwi(
    folder,
    name="sub-01_dwi",
    bvals=BVALS,
    ext=".nii.gz",
    bvec=True,
    bval=True,
    bval_as_column=False,
    bvec_rows=False,
):
    """
    Write a DWI image whose volume i is filled with the value i (so the volumes that
    survive an operation can be identified), plus its bvec and bval files.
    """
    bvals = np.asarray(bvals)
    n = len(bvals)
    data = np.broadcast_to(np.arange(n, dtype=np.float32), (4, 4, 3, n)).copy()
    image = Path(folder) / f"{name}{ext}"
    nib.save(nib.Nifti1Image(data, AFFINE), image)
    if bvec:
        vecs = unit_bvecs(n, zero=bvals == 0)
        np.savetxt(
            Path(folder) / f"{name}.bvec", vecs.T if bvec_rows else vecs, fmt="%.6f"
        )
    if bval:
        np.savetxt(
            Path(folder) / f"{name}.bval",
            bvals[:, None] if bval_as_column else bvals[None, :],
            fmt="%d",
        )
    return image


def kept(image):
    """Original indexes of the volumes stored in a DWI file."""
    return nib.load(image).get_fdata()[0, 0, 0, :].astype(int).tolist()


def read_bvals(path):
    return np.loadtxt(path, dtype=int, ndmin=1).tolist()


def bmatrix(bvecs, bvals):
    """b-matrix rows [Bxx, Byy, Bzz, Bxy, Bxz, Byz] from bvecs (3 x N) and bvals (N,)."""
    gx, gy, gz = bvecs
    return np.c_[
        bvals * gx * gx,
        bvals * gy * gy,
        bvals * gz * gz,
        bvals * gx * gy,
        bvals * gx * gz,
        bvals * gy * gz,
    ]


def min_angle(directions):
    """Smallest angle (degrees) between two directions, opposite directions being equal."""
    cosines = np.abs(directions @ directions.T)
    np.fill_diagonal(cosines, 0)
    return np.degrees(np.arccos(cosines.max()))


####################################################################################################
# delete_dwi_volumes
####################################################################################################
class TestDeleteDwiVolumes:
    @pytest.fixture
    def dwi(self, tmp_path):
        return make_dwi(tmp_path)

    def run(self, dwi, tmp_path, name="out", **kwargs):
        return delete_dwi_volumes(dwi, out_image=tmp_path / f"{name}.nii.gz", **kwargs)

    def test_removes_the_trailing_b0s_by_default(self, dwi, tmp_path):
        out_image, out_bvec, out_bval, removed = self.run(dwi, tmp_path)
        assert removed.tolist() == [10, 11]
        assert kept(out_image) == list(range(10))
        assert read_bvals(out_bval) == BVALS[:10].tolist()
        assert np.loadtxt(out_bvec).shape == (3, 10)

    def test_output_files_are_named_after_the_image(self, dwi, tmp_path):
        out_image, out_bvec, out_bval, _ = self.run(dwi, tmp_path, name="cleaned")
        assert Path(out_image) == tmp_path / "cleaned.nii.gz"
        assert Path(out_bvec) == tmp_path / "cleaned.bvec"
        assert Path(out_bval) == tmp_path / "cleaned.bval"
        assert all(Path(p).is_file() for p in (out_image, out_bvec, out_bval))

    def test_gradients_follow_the_kept_volumes(self, dwi, tmp_path):
        original_bvecs = np.loadtxt(tmp_path / "sub-01_dwi.bvec")
        _, out_bvec, out_bval, removed = self.run(dwi, tmp_path, vols_to_delete=[1, 5])
        keep = np.setdiff1d(np.arange(12), removed)
        assert np.allclose(np.loadtxt(out_bvec), original_bvecs[:, keep], atol=1e-6)
        assert read_bvals(out_bval) == BVALS[keep].tolist()

    @pytest.mark.parametrize(
        "vols, expected",
        [
            (3, [3]),
            ([1, 5], [1, 5]),
            ([1, "5-6"], [1, 5, 6]),  # Ranges are inclusive
            (["0:2"], [0, 1, 2]),
            (["0-2", 9], [0, 1, 2, 9]),
        ],
    )
    def test_volumes_selected_by_index(self, dwi, tmp_path, vols, expected):
        out_image, _, out_bval, removed = self.run(dwi, tmp_path, vols_to_delete=vols)
        assert removed.tolist() == expected
        assert kept(out_image) == [i for i in range(12) if i not in expected]
        assert len(read_bvals(out_bval)) == 12 - len(expected)

    @pytest.mark.parametrize(
        "bvals_to_delete, expected",
        [
            ([2000], [5, 6, 7, 9]),
            (2000, [5, 6, 7, 9]),
            ([0], [0, 4, 10, 11]),
            ([1000, 2000], [1, 2, 3, 5, 6, 7, 8, 9]),
            (["bvals >= 2000"], [5, 6, 7, 9]),
            (["bvals > 1000"], [5, 6, 7, 9]),
            (["500 < bvals < 1500"], [1, 2, 3, 8]),
        ],
    )
    def test_volumes_selected_by_b_value(
        self, dwi, tmp_path, bvals_to_delete, expected
    ):
        out_image, _, out_bval, removed = self.run(
            dwi, tmp_path, bvals_to_delete=bvals_to_delete
        )
        assert removed.tolist() == expected
        assert read_bvals(out_bval) == np.delete(BVALS, expected).tolist()
        assert kept(out_image) == np.delete(np.arange(12), expected).tolist()

    def test_b_values_and_indexes_are_combined(self, dwi, tmp_path):
        _, _, _, removed = self.run(
            dwi, tmp_path, bvals_to_delete=[2000], vols_to_delete=[0]
        )
        assert removed.tolist() == [0, 5, 6, 7, 9]

    def test_affine_is_kept(self, dwi, tmp_path):
        out_image, *_ = self.run(dwi, tmp_path, vols_to_delete=[1])
        assert np.allclose(nib.load(out_image).affine, AFFINE)
        assert nib.load(out_image).shape == (4, 4, 3, 11)

    def test_paths_can_be_strings_or_path_objects(self, dwi, tmp_path):
        out_image, *_ = delete_dwi_volumes(
            str(dwi),
            bvec_file=str(tmp_path / "sub-01_dwi.bvec"),
            bval_file=tmp_path / "sub-01_dwi.bval",
            out_image=str(tmp_path / "str_out.nii.gz"),
            vols_to_delete=[1],
        )
        assert nib.load(out_image).shape[3] == 11

    def test_explicit_bvec_and_bval_files(self, tmp_path):
        dwi = make_dwi(tmp_path, name="image", bvec=False, bval=False)
        np.savetxt(tmp_path / "grads.bval", BVALS[None, :], fmt="%d")
        np.savetxt(tmp_path / "grads.bvec", unit_bvecs(12), fmt="%.6f")
        out_image, out_bvec, out_bval, removed = delete_dwi_volumes(
            dwi,
            bvec_file=tmp_path / "grads.bvec",
            bval_file=tmp_path / "grads.bval",
            out_image=tmp_path / "out.nii.gz",
            vols_to_delete=[0],
        )
        assert removed.tolist() == [0]
        assert read_bvals(out_bval) == BVALS[1:].tolist()

    def test_bval_stored_as_a_column(self, tmp_path):
        dwi = make_dwi(tmp_path, bval_as_column=True)
        _, _, out_bval, removed = self.run(dwi, tmp_path)
        assert removed.tolist() == [10, 11]
        assert read_bvals(out_bval) == BVALS[:10].tolist()

    def test_bvec_stored_with_one_row_per_volume(self, tmp_path):
        dwi = make_dwi(tmp_path, bvec_rows=True)
        _, out_bvec, _, _ = self.run(dwi, tmp_path, vols_to_delete=[1])
        assert np.loadtxt(out_bvec).shape == (11, 3)

    def test_overwrites_the_input_when_no_output_is_given(self, dwi, tmp_path):
        out_image, out_bvec, out_bval, removed = delete_dwi_volumes(
            dwi, vols_to_delete=[0]
        )
        assert Path(out_image) == dwi and removed.tolist() == [0]
        assert nib.load(dwi).shape[3] == 11
        assert read_bvals(tmp_path / "sub-01_dwi.bval") == BVALS[1:].tolist()
        assert np.loadtxt(tmp_path / "sub-01_dwi.bvec").shape == (3, 11)

    def test_uncompressed_input_is_saved_as_nii_gz(self, tmp_path):
        dwi = make_dwi(tmp_path, ext=".nii")
        out_image, out_bvec, out_bval, _ = delete_dwi_volumes(
            dwi, out_image=tmp_path / "out.nii", vols_to_delete=[1]
        )
        assert (Path(out_image).name, Path(out_bvec).name, Path(out_bval).name) == (
            "out.nii.gz",
            "out.bvec",
            "out.bval",
        )

    def test_without_gradient_files(self, tmp_path):
        dwi = make_dwi(tmp_path, bvec=False, bval=False)
        out_image, out_bvec, out_bval, removed = self.run(
            dwi, tmp_path, vols_to_delete=[0]
        )
        assert out_bvec is None and out_bval is None and removed.tolist() == [0]
        assert nib.load(out_image).shape[3] == 11

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"vols_to_delete": []},  # Nothing selected
            {"bvals_to_delete": [5000]},  # No volume has this b-value
            {"vols_to_delete": list(range(12))},  # Every volume would be removed
        ],
    )
    def test_nothing_is_removed(self, dwi, tmp_path, kwargs):
        out_image, out_bvec, out_bval, removed = self.run(dwi, tmp_path, **kwargs)
        assert Path(out_image) == dwi  # The inputs are returned untouched
        assert Path(out_bvec) == tmp_path / "sub-01_dwi.bvec"
        assert Path(out_bval) == tmp_path / "sub-01_dwi.bval"
        assert removed.size == 0
        assert not (tmp_path / "out.nii.gz").exists()
        assert nib.load(dwi).shape[3] == 12

    @pytest.mark.parametrize("bvals", [[0, 1000, 1000, 2000, 2000], [0, 0, 1000, 1000]])
    def test_no_trailing_b0s_to_remove(self, tmp_path, bvals):
        dwi = make_dwi(tmp_path, bvals=bvals)
        out_image, _, _, removed = self.run(dwi, tmp_path)
        assert Path(out_image) == dwi and removed.size == 0

    def test_b_values_below_10_count_as_b0s(self, tmp_path):
        dwi = make_dwi(tmp_path, bvals=[0, 1000, 1000, 5, 0])
        _, _, _, removed = self.run(dwi, tmp_path)
        assert removed.tolist() == [3, 4]

    def test_errors(self, dwi, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            delete_dwi_volumes(tmp_path / "missing.nii.gz")
        (tmp_path / "image.img").write_text("x")
        with pytest.raises(ValueError, match="recognized NIfTI extension"):
            delete_dwi_volumes(tmp_path / "image.img")
        with pytest.raises(FileNotFoundError, match="Output path"):
            delete_dwi_volumes(dwi, out_image=tmp_path / "no_dir" / "out.nii.gz")

    @pytest.mark.parametrize("vols", [[3, 40], [12], [-1]])
    def test_volumes_out_of_range(self, dwi, tmp_path, vols):
        with pytest.raises(ValueError, match="out of the range|out of range"):
            self.run(dwi, tmp_path, vols_to_delete=vols)

    def test_bval_count_must_match_the_volumes(self, tmp_path):
        dwi = make_dwi(tmp_path)
        np.savetxt(tmp_path / "sub-01_dwi.bval", BVALS[None, :8], fmt="%d")
        with pytest.raises(ValueError, match="has 8 values but the image has 12"):
            self.run(dwi, tmp_path)

    def test_3d_image(self, tmp_path):
        nib.save(
            nib.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4)), tmp_path / "vol.nii.gz"
        )
        with pytest.raises(ValueError, match="not a 4D image"):
            delete_dwi_volumes(tmp_path / "vol.nii.gz", vols_to_delete=[0])

    def test_bval_file_is_mandatory_when_it_is_needed(self, tmp_path):
        dwi = make_dwi(tmp_path, bvec=False, bval=False)
        with pytest.raises(FileNotFoundError, match="vols_to_delete"):
            self.run(dwi, tmp_path)  # Trailing B0s are found from the b-values
        with pytest.raises(FileNotFoundError, match="bvals_to_delete"):
            self.run(dwi, tmp_path, bvals_to_delete=[0])


####################################################################################################
# get_b0s
####################################################################################################
class TestGetB0s:
    @pytest.fixture
    def dwi(self, tmp_path):
        return make_dwi(tmp_path)

    def test_extracts_the_b0_volumes(self, dwi, tmp_path):
        out, b0_vols = get_b0s(dwi, tmp_path / "b0.nii.gz")
        assert Path(out) == tmp_path / "b0.nii.gz"
        assert b0_vols.tolist() == [0, 4, 10, 11]
        assert kept(out) == [0, 4, 10, 11]  # Volumes keep their order
        assert nib.load(out).shape == (4, 4, 3, 4)
        assert np.allclose(nib.load(out).affine, AFFINE)

    def test_default_output_name(self, dwi, tmp_path):
        out, _ = get_b0s(dwi)
        assert Path(out) == tmp_path / "sub-01_dwi_b0s.nii.gz"
        assert Path(out).is_file()

    def test_output_extension_is_kept(self, dwi, tmp_path):
        out, _ = get_b0s(str(dwi), str(tmp_path / "b0s.nii"))
        assert Path(out).name == "b0s.nii" and Path(out).is_file()

    def test_explicit_bval_file_stored_as_a_column(self, tmp_path):
        dwi = make_dwi(tmp_path, bval=False)
        np.savetxt(tmp_path / "other.bval", BVALS[:, None], fmt="%d")
        _, b0_vols = get_b0s(
            dwi, tmp_path / "b0.nii.gz", bval_file=tmp_path / "other.bval"
        )
        assert b0_vols.tolist() == [0, 4, 10, 11]

    def test_threshold_includes_the_boundary(self, tmp_path):
        dwi = make_dwi(tmp_path)
        np.savetxt(
            tmp_path / "low.bval", np.where(BVALS == 0, 5, BVALS)[None, :], fmt="%d"
        )
        low = tmp_path / "low.bval"
        for threshold in (5, 50):
            _, b0_vols = get_b0s(
                dwi, tmp_path / "b0.nii.gz", bval_file=low, bval_thresh=threshold
            )
            assert b0_vols.tolist() == [0, 4, 10, 11]
        with pytest.raises(ValueError, match="No B0 volumes found"):
            get_b0s(dwi, tmp_path / "b0.nii.gz", bval_file=low, bval_thresh=4)
        with pytest.raises(ValueError, match="No B0 volumes found"):
            get_b0s(dwi, tmp_path / "b0.nii.gz", bval_file=low)  # Default threshold: 0

    def test_a_higher_threshold_adds_low_b_volumes(self, dwi, tmp_path):
        _, b0_vols = get_b0s(dwi, tmp_path / "b0.nii.gz", bval_thresh=1000)
        assert b0_vols.tolist() == [0, 1, 2, 3, 4, 8, 10, 11]

    def test_single_b0_stays_a_4d_image(self, tmp_path):
        dwi = make_dwi(tmp_path, bvals=[0, 1000, 1000])
        out, b0_vols = get_b0s(dwi, tmp_path / "b0.nii.gz")
        assert b0_vols.tolist() == [0] and nib.load(out).shape == (4, 4, 3, 1)

    def test_errors(self, dwi, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            get_b0s(tmp_path / "missing.nii.gz")
        (tmp_path / "image.img").write_text("x")
        with pytest.raises(ValueError, match="recognized NIfTI extension"):
            get_b0s(tmp_path / "image.img")
        with pytest.raises(FileNotFoundError, match="Output path"):
            get_b0s(dwi, tmp_path / "no_dir" / "b0.nii.gz")
        no_bval = make_dwi(tmp_path, name="no_bval", bval=False)
        with pytest.raises(FileNotFoundError, match=r"no_bval\.bval"):
            get_b0s(no_bval)

    def test_bval_count_must_match_the_volumes(self, dwi, tmp_path):
        np.savetxt(tmp_path / "short.bval", BVALS[None, :8], fmt="%d")
        with pytest.raises(ValueError, match="has 8 values"):
            get_b0s(dwi, tmp_path / "b0.nii.gz", bval_file=tmp_path / "short.bval")

    def test_3d_image(self, tmp_path):
        nib.save(
            nib.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4)), tmp_path / "vol.nii.gz"
        )
        np.savetxt(tmp_path / "vol.bval", [0, 0, 0], fmt="%d")
        with pytest.raises(ValueError, match="has 3 values"):
            get_b0s(tmp_path / "vol.nii.gz")


####################################################################################################
# maps_from_tensor_eigenvalues
####################################################################################################
class TestTensorMaps:
    # One voxel per row: isotropic, prolate, oblate, a tensor with a single non-zero
    # eigenvalue (the most anisotropic one possible) and a background voxel
    EIGENVALUES = np.array(
        [
            [1.0e-3, 1.0e-3, 1.0e-3],
            [1.7e-3, 0.3e-3, 0.3e-3],
            [1.2e-3, 1.2e-3, 0.2e-3],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    @pytest.fixture
    def eig_file(self, tmp_path):
        data = self.EIGENVALUES.reshape(-1, 1, 1, 3)
        path = tmp_path / "eigvals.nii.gz"
        nib.save(nib.Nifti1Image(data, AFFINE), path)
        return path

    @pytest.fixture
    def maps(self, tmp_path, eig_file):
        (tmp_path / "out").mkdir()
        return maps_from_tensor_eigenvalues(eig_file, tmp_path / "out" / "sub")

    @staticmethod
    def values(path):
        return nib.load(path).get_fdata().ravel()

    def test_all_maps_are_computed_by_default(self, tmp_path, maps):
        assert sorted(maps) == ALL_MAPS
        for tag, path in maps.items():
            assert Path(path) == tmp_path / "out" / f"sub_{tag}.nii.gz"
            assert nib.load(path).shape == (5, 1, 1)
            assert np.allclose(nib.load(path).affine, AFFINE)

    def test_diffusivities(self, maps):
        l1, l2, l3 = self.EIGENVALUES.T
        assert np.allclose(self.values(maps["AD"]), l1)
        assert np.allclose(self.values(maps["RD"]), (l2 + l3) / 2)
        assert np.allclose(self.values(maps["MD"]), (l1 + l2 + l3) / 3)

    def test_fractional_anisotropy(self, maps):
        fa = self.values(maps["FA"])
        assert fa[0] == pytest.approx(0)  # Isotropic
        assert fa[3] == pytest.approx(1)  # A single non-zero eigenvalue
        for voxel in (1, 2):
            eigs = self.EIGENVALUES[voxel]
            expected = np.sqrt(
                1.5 * np.sum((eigs - eigs.mean()) ** 2) / np.sum(eigs**2)
            )
            assert fa[voxel] == pytest.approx(expected)
        assert fa[1] == pytest.approx(0.79902, abs=1e-4)
        assert fa[2] == pytest.approx(0.58521, abs=1e-4)

    def test_westin_coefficients(self, maps):
        cl, cp, cs = (self.values(maps[tag]) for tag in ("CL", "CP", "CS"))
        # Linear, planar and spherical coefficients of the prolate tensor
        assert cl[1] == pytest.approx(1.4 / 2.3)
        assert cp[1] == pytest.approx(0)
        assert cs[1] == pytest.approx(0.9 / 2.3)
        # And of the oblate one
        assert cl[2] == pytest.approx(0)
        assert cp[2] == pytest.approx(2.0 / 2.6)
        assert cs[2] == pytest.approx(0.6 / 2.6)
        # Isotropic and single-direction tensors
        assert (cl[0], cp[0], cs[0]) == pytest.approx((0, 0, 1))
        assert (cl[3], cp[3], cs[3]) == pytest.approx((1, 0, 0))

    def test_westin_coefficients_add_up_to_one(self, maps):
        total = sum(self.values(maps[tag]) for tag in ("CL", "CP", "CS"))
        assert np.allclose(total[:4], 1.0)  # Not defined for the background voxel

    def test_relative_anisotropy_and_volume_fraction(self, maps):
        ra, vf = self.values(maps["RA"]), self.values(maps["VF"])
        assert ra[0] == pytest.approx(0) and vf[0] == pytest.approx(0)
        assert ra[3] == pytest.approx(np.sqrt(2))  # Maximum for a single eigenvalue
        assert vf[3] == pytest.approx(1)
        eigs = self.EIGENVALUES[1]
        md = eigs.mean()
        assert ra[1] == pytest.approx(
            np.sqrt(np.sum((eigs - md) ** 2)) / (np.sqrt(3) * md)
        )
        assert vf[1] == pytest.approx(1 - np.prod(eigs) / md**3)

    def test_geodesic_anisotropy(self, maps):
        ga = self.values(maps["GA"])
        assert ga[0] == pytest.approx(0)
        for voxel in (1, 2):
            logs = np.log(self.EIGENVALUES[voxel])
            assert ga[voxel] == pytest.approx(
                np.sqrt(np.sum((logs - logs.mean()) ** 2))
            )
        # Only defined for positive eigenvalues
        assert ga[3] == 0 and ga[4] == 0

    def test_background_voxels_are_zero(self, maps):
        for tag in ALL_MAPS:
            assert self.values(maps[tag])[4] == 0, tag

    def test_all_values_are_finite(self, tmp_path):
        eig = np.zeros((4, 1, 1, 3))
        eig[1, 0, 0] = [1e-3, 1e-3, -1e-4]  # Not positive definite
        eig[2, 0, 0] = [np.nan] * 3
        eig[3, 0, 0] = [1, 1, 1]
        nib.save(nib.Nifti1Image(eig, np.eye(4)), tmp_path / "eig.nii.gz")
        maps = maps_from_tensor_eigenvalues(tmp_path / "eig.nii.gz", tmp_path / "bad")
        for tag, path in maps.items():
            data = nib.load(path).get_fdata().ravel()
            assert np.isfinite(data).all(), tag
            assert data[2] == 0, tag  # NaN voxels are filled with zeros
        assert nib.load(maps["GA"]).get_fdata().ravel()[1] == 0

    def test_three_separate_files_give_the_same_maps(self, tmp_path, eig_file, maps):
        files = []
        for i in range(3):
            path = tmp_path / f"l{i + 1}.nii.gz"
            nib.save(
                nib.Nifti1Image(self.EIGENVALUES.reshape(-1, 1, 1, 3)[..., i], AFFINE),
                path,
            )
            files.append(path)
        (tmp_path / "sep").mkdir()
        for eigvals in (files, tuple(str(f) for f in files)):
            separate = maps_from_tensor_eigenvalues(eigvals, tmp_path / "sep" / "sub")
            assert sorted(separate) == ALL_MAPS
            for tag in ALL_MAPS:
                assert np.allclose(
                    self.values(separate[tag]), self.values(maps[tag])
                ), tag
                assert np.allclose(nib.load(separate[tag]).affine, AFFINE)

    def test_four_dimensional_input_can_have_extra_volumes(self, tmp_path):
        data = np.concatenate(
            [self.EIGENVALUES.reshape(-1, 1, 1, 3), np.ones((5, 1, 1, 2))], axis=3
        )
        nib.save(nib.Nifti1Image(data, np.eye(4)), tmp_path / "five.nii.gz")
        maps = maps_from_tensor_eigenvalues(
            tmp_path / "five.nii.gz", tmp_path / "five", ["AD"]
        )
        assert np.allclose(self.values(maps["AD"]), self.EIGENVALUES[:, 0])

    def test_requested_maps_only(self, tmp_path, eig_file):
        (tmp_path / "some").mkdir()
        maps = maps_from_tensor_eigenvalues(
            eig_file, tmp_path / "some" / "sub", dtmaps=["FA", "md", "Ad"]
        )
        assert list(maps) == ["AD", "MD", "FA"]  # Tags are case-insensitive
        assert sorted(p.name for p in (tmp_path / "some").iterdir()) == [
            "sub_AD.nii.gz",
            "sub_FA.nii.gz",
            "sub_MD.nii.gz",
        ]

    def test_all_can_be_requested_explicitly(self, tmp_path, eig_file):
        maps = maps_from_tensor_eigenvalues(eig_file, tmp_path / "sub", dtmaps=["ALL"])
        assert sorted(maps) == ALL_MAPS

    def test_unknown_tags_are_ignored(self, tmp_path, eig_file):
        assert (
            maps_from_tensor_eigenvalues(eig_file, tmp_path / "sub", dtmaps=["nope"])
            == {}
        )
        assert not list(tmp_path.glob("sub_*"))

    def test_existing_maps_are_not_recomputed(self, tmp_path, eig_file):
        sentinel = tmp_path / "sub_FA.nii.gz"
        nib.save(nib.Nifti1Image(np.full((5, 1, 1), 7.0), AFFINE), sentinel)
        maps = maps_from_tensor_eigenvalues(eig_file, tmp_path / "sub", dtmaps=["FA"])
        assert Path(maps["FA"]) == sentinel
        assert np.allclose(self.values(sentinel), 7.0)  # Left untouched

    def test_overwrite(self, tmp_path, eig_file):
        nib.save(
            nib.Nifti1Image(np.full((5, 1, 1), 7.0), AFFINE), tmp_path / "sub_FA.nii.gz"
        )
        maps = maps_from_tensor_eigenvalues(
            eig_file, tmp_path / "sub", dtmaps=["FA"], overwrite=True
        )
        fa = self.values(maps["FA"])
        assert fa[0] == pytest.approx(0) and fa[3] == pytest.approx(1)

    def test_path_arguments(self, tmp_path, eig_file):
        maps = maps_from_tensor_eigenvalues(
            str(eig_file), str(tmp_path / "sub"), dtmaps=["MD"]
        )
        assert Path(maps["MD"]).is_file()

    def test_errors(self, tmp_path, eig_file):
        with pytest.raises(FileNotFoundError, match="Eigenvalue file not found"):
            maps_from_tensor_eigenvalues(tmp_path / "missing.nii.gz", tmp_path / "sub")
        with pytest.raises(FileNotFoundError, match="not found"):
            maps_from_tensor_eigenvalues(
                [eig_file, eig_file, tmp_path / "missing.nii.gz"], tmp_path / "sub"
            )
        with pytest.raises(ValueError, match="exactly 3 paths"):
            maps_from_tensor_eigenvalues([eig_file, eig_file], tmp_path / "sub")
        with pytest.raises(ValueError, match="at least 3 volumes"):
            nib.save(
                nib.Nifti1Image(np.zeros((2, 1, 1, 2)), np.eye(4)),
                tmp_path / "two.nii.gz",
            )
            maps_from_tensor_eigenvalues(tmp_path / "two.nii.gz", tmp_path / "sub")
        with pytest.raises(ValueError, match="at least 3 volumes"):
            nib.save(
                nib.Nifti1Image(np.zeros((2, 1, 1)), np.eye(4)), tmp_path / "vol.nii.gz"
            )
            maps_from_tensor_eigenvalues(tmp_path / "vol.nii.gz", tmp_path / "sub")
        with pytest.raises(ValueError, match="path string or a list"):
            maps_from_tensor_eigenvalues(5, tmp_path / "sub")
        with pytest.raises(FileNotFoundError, match="Output directory"):
            maps_from_tensor_eigenvalues(eig_file, tmp_path / "no_dir" / "sub")


####################################################################################################
# DiffusionScheme: loading
####################################################################################################
class TestLoadSchemes:
    @pytest.fixture
    def shelled(self):
        """bvecs (3 x N) and bvals (N,) of 6 B0s and 64 directions at b=1000."""
        return DiffusionScheme._shelled_scheme({1000: 64}, n_b0s=6, n_iter=20)

    def test_new_object_is_empty(self):
        scheme = DiffusionScheme()
        assert scheme.gradients is None and scheme.bvals is None
        assert scheme.scheme_type is None

    def test_from_arrays(self, shelled):
        bvecs, bvals = shelled
        scheme = DiffusionScheme.from_bvec_bval_arrays(bvecs, bvals)
        assert scheme.gradients.shape == (70, 3) and scheme.bvals.shape == (70,)
        assert np.allclose(scheme.gradients, bvecs.T)  # Gradients are stored as (N, 3)
        assert scheme.scheme_type == "shelled"

    def test_from_arrays_accepts_other_layouts(self, shelled):
        bvecs, bvals = shelled
        by_rows = DiffusionScheme.from_bvec_bval_arrays(bvecs.T, bvals)
        from_lists = DiffusionScheme.from_bvec_bval_arrays(
            bvecs.tolist(), bvals.tolist()
        )
        two_dimensional = DiffusionScheme.from_bvec_bval_arrays(bvecs, bvals[None, :])
        for scheme in (by_rows, from_lists, two_dimensional):
            assert np.allclose(scheme.gradients, bvecs.T)
            assert scheme.bvals.shape == (70,)

    def test_b0_only_acquisition(self):
        scheme = DiffusionScheme.from_bvec_bval_arrays(np.zeros((3, 5)), np.zeros(5))
        assert scheme.scheme_type == "b0_only" and len(scheme.bvals) == 5

    @pytest.mark.parametrize("as_str", [True, False])
    def test_from_files(self, tmp_path, shelled, as_str):
        bvecs, bvals = shelled
        np.savetxt(tmp_path / "a.bvec", bvecs, fmt="%.6f")
        np.savetxt(tmp_path / "a.bval", bvals[None, :], fmt="%d")
        files = (tmp_path / "a.bvec", tmp_path / "a.bval")
        scheme = DiffusionScheme.from_bvec_bval_files(
            *(map(str, files) if as_str else files)
        )
        assert scheme.gradients.shape == (70, 3) and scheme.scheme_type == "shelled"
        assert np.allclose(scheme.gradients, bvecs.T, atol=1e-6)
        assert np.array_equal(scheme.bvals, bvals)

    def test_from_files_with_other_layouts(self, tmp_path, shelled):
        bvecs, bvals = shelled
        np.savetxt(tmp_path / "rows.bvec", bvecs.T, fmt="%.6f")  # One gradient per row
        np.savetxt(tmp_path / "cols.bvec", bvecs, fmt="%.6f")
        np.savetxt(
            tmp_path / "col.bval", bvals[:, None], fmt="%d"
        )  # b-values in a column
        np.savetxt(tmp_path / "row.bval", bvals[None, :], fmt="%d")
        reference = DiffusionScheme.from_bvec_bval_files(
            tmp_path / "cols.bvec", tmp_path / "row.bval"
        )
        for bvec, bval in (("rows", "row"), ("cols", "col"), ("rows", "col")):
            scheme = DiffusionScheme.from_bvec_bval_files(
                tmp_path / f"{bvec}.bvec", tmp_path / f"{bval}.bval"
            )
            assert np.allclose(scheme.gradients, reference.gradients)
            assert np.array_equal(scheme.bvals, reference.bvals)

    def test_missing_files(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DiffusionScheme.from_bvec_bval_files(
                tmp_path / "x.bvec", tmp_path / "x.bval"
            )
        with pytest.raises(FileNotFoundError):
            DiffusionScheme.from_bmatrix_file(tmp_path / "x.bmat")


class TestBMatrix:
    DIRECTIONS = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0] / np.sqrt(2),
            [1, -1, 0] / np.sqrt(2),
        ]
    ).T

    def test_known_directions(self):
        scheme = DiffusionScheme.from_bmatrix_array(
            bmatrix(self.DIRECTIONS, np.full(5, 1000.0))
        )
        assert np.allclose(scheme.bvals, 1000)
        assert np.allclose(scheme.gradients, self.DIRECTIONS.T)
        assert scheme.scheme_type == "shelled"

    def test_zero_rows_are_b0_volumes(self):
        bmat = np.vstack(
            [np.zeros((2, 6)), bmatrix(self.DIRECTIONS, np.full(5, 2000.0))]
        )
        scheme = DiffusionScheme.from_bmatrix_array(bmat)
        assert np.allclose(scheme.bvals, [0, 0] + [2000] * 5)
        assert np.allclose(scheme.gradients[:2], 0)

    def test_gradients_are_recovered_up_to_their_sign(self):
        """A gradient and its opposite give the same b-matrix."""
        rng = np.random.default_rng(1)
        bvecs = rng.normal(size=(3, 50))
        bvecs /= np.linalg.norm(bvecs, axis=0)
        bvals = rng.choice([1000.0, 2000.0, 3000.0], 50)
        scheme = DiffusionScheme.from_bmatrix_array(bmatrix(bvecs, bvals))
        assert np.allclose(scheme.bvals, bvals)
        assert np.allclose(np.linalg.norm(scheme.gradients, axis=1), 1)
        assert np.allclose(np.abs(np.sum(scheme.gradients * bvecs.T, axis=1)), 1)

    def test_recovered_gradients_reproduce_the_b_matrix(self):
        rng = np.random.default_rng(2)
        bvecs = rng.normal(size=(3, 40))
        bvecs /= np.linalg.norm(bvecs, axis=0)
        bmat = bmatrix(bvecs, np.full(40, 1500.0))
        scheme = DiffusionScheme.from_bmatrix_array(bmat)
        assert np.allclose(bmatrix(scheme.gradients.T, scheme.bvals), bmat)

    def test_gradients_pointing_down_are_flipped_to_an_equivalent_direction(self):
        down = np.array([[0.0, 0, -1], [0.6, 0, -0.8], [0, -0.6, 0.8]]).T
        scheme = DiffusionScheme.from_bmatrix_array(bmatrix(down, np.full(3, 1000.0)))
        assert np.allclose(np.abs(np.sum(scheme.gradients * down.T, axis=1)), 1)

    def test_cartesian_scheme_is_detected(self):
        bvecs, bvals = DiffusionScheme._dsi_scheme()
        scheme = DiffusionScheme.from_bmatrix_array(bmatrix(bvecs, bvals))
        assert scheme.scheme_type == "cartesian" and len(scheme.bvals) == 153

    def test_lists_and_files(self, tmp_path):
        bmat = bmatrix(self.DIRECTIONS, np.full(5, 1000.0))
        np.savetxt(tmp_path / "scheme.bmat", bmat, fmt="%.6f")
        from_list = DiffusionScheme.from_bmatrix_array(bmat.tolist())
        for scheme in (
            from_list,
            DiffusionScheme.from_bmatrix_file(tmp_path / "scheme.bmat"),
            DiffusionScheme.from_bmatrix_file(str(tmp_path / "scheme.bmat")),
        ):
            assert np.allclose(scheme.bvals, 1000)
            assert np.allclose(scheme.gradients, from_list.gradients, atol=1e-5)

    def test_needs_six_columns(self):
        with pytest.raises(ValueError):
            DiffusionScheme.from_bmatrix_array(np.zeros((5, 5)))


####################################################################################################
# DiffusionScheme: simulation
####################################################################################################
class TestSimulateSchemes:
    def test_shelled_layout(self, multishell_scheme):
        shelled = multishell_scheme
        assert shelled.scheme_type == "shelled"
        assert shelled.gradients.shape == (8 + 32 + 64 + 96, 3)
        assert (shelled.bvals[:8] == 0).all()  # B0s come first
        assert np.allclose(shelled.gradients[:8], 0)
        for bval, n_dirs in MULTISHELL.items():
            assert (shelled.bvals == bval).sum() == n_dirs

    def test_shelled_directions(self, multishell_scheme):
        shelled = multishell_scheme
        for bval in MULTISHELL:
            directions = shelled.gradients[shelled.bvals == bval]
            assert np.allclose(np.linalg.norm(directions, axis=1), 1)
            assert (directions[:, 2] >= -1e-9).all()  # Upper half sphere
            assert min_angle(directions) > 12  # Well spread

    def test_repulsion_spreads_the_directions(self):
        initial = DiffusionScheme._sphere_dirs(32, n_iter=0)
        refined = DiffusionScheme._sphere_dirs(32, n_iter=200)
        assert min_angle(refined) > 20
        assert min_angle(refined) > min_angle(initial) + 5

    def test_defaults(self):
        scheme = DiffusionScheme.simulate_dwi_acq_scheme(n_iter=5)
        assert (scheme.bvals == 0).sum() == 6
        assert (scheme.bvals == 1000).sum() == 30 and (scheme.bvals == 2000).sum() == 60

    def test_is_reproducible(self):
        first = DiffusionScheme.simulate_dwi_acq_scheme(n_iter=20)
        second = DiffusionScheme.simulate_dwi_acq_scheme(n_iter=20)
        assert np.array_equal(first.gradients, second.gradients)

    def test_without_b0s(self):
        scheme = DiffusionScheme.simulate_dwi_acq_scheme(
            shells={1000: 10}, n_b0s=0, n_iter=5
        )
        assert len(scheme.bvals) == 10 and (scheme.bvals == 1000).all()

    @pytest.mark.parametrize(
        "alias", ["shelled", "shell", "hardi", "multishell", "SHELLED"]
    )
    def test_shelled_names(self, alias):
        scheme = DiffusionScheme.simulate_dwi_acq_scheme(
            alias, shells={1000: 12}, n_iter=5
        )
        assert scheme.scheme_type == "shelled"

    @pytest.mark.parametrize("alias", ["cartesian", "dsi", "grid", "DSI"])
    def test_cartesian_names(self, alias):
        assert DiffusionScheme.simulate_dwi_acq_scheme(alias).scheme_type == "cartesian"

    def test_the_scheme_type_is_not_left_to_detection(self):
        """9 shells of 3 directions look cartesian, but the simulated type is kept."""
        shells = {500 * (i + 1): 3 for i in range(9)}
        simulated = DiffusionScheme.simulate_dwi_acq_scheme(
            "shelled", shells=shells, n_iter=5
        )
        assert simulated.scheme_type == "shelled"
        detected = DiffusionScheme.from_bvec_bval_arrays(
            simulated.gradients, simulated.bvals
        )
        assert detected.scheme_type == "cartesian"

    @pytest.mark.parametrize("radius", [1, 2, 4, 5])
    def test_cartesian_grid(self, radius):
        grid = [
            (i, j, k)
            for i in range(-radius, radius + 1)
            for j in range(-radius, radius + 1)
            for k in range(radius + 1)
            if 0 < i * i + j * j + k * k <= radius**2
        ]
        scheme = DiffusionScheme.simulate_dwi_acq_scheme("dsi", radius=radius, n_b0s=0)
        assert len(scheme.bvals) == len(grid)
        q = np.linalg.norm(np.array(grid, dtype=float), axis=1)
        assert np.allclose(np.sort(scheme.bvals), np.sort(4000 * (q / radius) ** 2))
        assert scheme.bvals.max() == pytest.approx(4000)

    def test_cartesian_directions_and_b0s(self):
        scheme = DiffusionScheme.simulate_dwi_acq_scheme(
            "dsi", bmax=6000, radius=4, n_b0s=3
        )
        assert (scheme.bvals[:3] == 0).all() and np.allclose(scheme.gradients[:3], 0)
        q_space = scheme.gradients[3:]
        assert np.allclose(np.linalg.norm(q_space, axis=1), 1)
        assert (q_space[:, 2] >= 0).all()
        assert scheme.bvals.max() == pytest.approx(6000)

    def test_simulated_scheme_survives_saving_as_fsl_files(
        self, tmp_path, multishell_scheme
    ):
        shelled = multishell_scheme
        np.savetxt(tmp_path / "s.bvec", shelled.gradients.T, fmt="%.6f")
        np.savetxt(tmp_path / "s.bval", shelled.bvals[None, :], fmt="%d")
        loaded = DiffusionScheme.from_bvec_bval_files(
            tmp_path / "s.bvec", tmp_path / "s.bval"
        )
        assert np.allclose(loaded.gradients, shelled.gradients, atol=1e-5)
        assert np.array_equal(loaded.bvals, shelled.bvals)
        assert loaded.scheme_type == "shelled"

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"scheme_type": "spiral"}, "Unknown scheme_type"),
            ({"scheme_type": "shelled", "n_b0s": -1}, "n_b0s"),
            ({"scheme_type": "dsi", "n_b0s": -1}, "n_b0s"),
            ({"scheme_type": "dsi", "radius": 0}, "radius"),
            ({"scheme_type": "shelled", "shells": {}}, "non-empty dict"),
            ({"scheme_type": "shelled", "shells": [1000]}, "non-empty dict"),
        ],
    )
    def test_invalid_arguments(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            DiffusionScheme.simulate_dwi_acq_scheme(**kwargs)

    def test_sphere_directions_edge_cases(self):
        assert DiffusionScheme._sphere_dirs(0).shape == (0, 3)
        single = DiffusionScheme._sphere_dirs(1)
        assert single.shape == (1, 3) and np.linalg.norm(single) == pytest.approx(1)


####################################################################################################
# DiffusionScheme: detection
####################################################################################################
class TestDetectScheme:
    @staticmethod
    def detect(bvals):
        scheme = DiffusionScheme()
        scheme.bvals = np.asarray(bvals, dtype=float)
        scheme._detect_scheme()
        return scheme.scheme_type

    @pytest.mark.parametrize(
        "bvals, expected",
        [
            ([0, 0, 0], "b0_only"),
            ([49, 49, 49], "b0_only"),  # Up to b=50 counts as B0
            ([0, 51, 51, 51], "shelled"),
            ([0] * 3 + [1000] * 30, "shelled"),  # Single shell
            ([0] * 3 + [1000] * 30 + [2000] * 60, "shelled"),  # Multi-shell
            (
                [0] * 3 + [1000] * 20 + [1005] * 20,
                "shelled",
            ),  # Rounded to the same shell
            (
                [0] + [1000] * 4 + [2000] * 4 + [3000] * 4,
                "shelled",
            ),  # Few samples per shell
            ([0] + list(np.repeat(np.arange(1, 7) * 500, 3)), "shelled"),  # 6 shells
            ([0] + list(np.linspace(100, 4000, 100)), "cartesian"),  # Many b-values
            ([0] + list(np.repeat(np.arange(1, 10) * 400, 3)), "cartesian"),  # 9 shells
        ],
    )
    def test_detection(self, bvals, expected):
        assert self.detect(bvals) == expected

    def test_loaders_detect_the_scheme(self):
        bvecs, bvals = DiffusionScheme._dsi_scheme()
        assert (
            DiffusionScheme.from_bvec_bval_arrays(bvecs, bvals).scheme_type
            == "cartesian"
        )
        bvecs, bvals = DiffusionScheme._shelled_scheme({1000: 30, 2000: 60}, n_iter=5)
        assert (
            DiffusionScheme.from_bvec_bval_arrays(bvecs, bvals).scheme_type == "shelled"
        )


####################################################################################################
# DiffusionScheme: plot
####################################################################################################
class TestPlot:
    @pytest.fixture
    def plotter(self, can_render):
        created = []

        def make(scheme, **kwargs):
            created.append(scheme.plot(show=False, **kwargs))
            return created[-1]

        yield make
        for plotter in created:
            plotter.close()

    @staticmethod
    def spheres(plotter):
        """Mesh with the spheres of the gradient directions (the first actor)."""
        return next(iter(plotter.renderer.actors.values())).mapper.dataset

    @pytest.mark.parametrize(
        "name, signature",
        [
            ("scheme.png", b"\x89PNG"),
            ("scheme.jpg", b"\xff\xd8"),
            ("scheme.html", b"<"),
        ],
    )
    def test_saves_images_and_html(self, tmp_path, can_render, scheme, name, signature):
        assert scheme.plot(save_path=tmp_path / name) is None  # The plotter is closed
        content = (tmp_path / name).read_bytes()
        assert content.startswith(signature) and len(content) > 1000

    def test_saves_vector_graphics(self, tmp_path, can_render, scheme):
        scheme.plot(save_path=str(tmp_path / "scheme.svg"))
        assert "<svg" in (tmp_path / "scheme.svg").read_text(errors="ignore")[:500]

    def test_html_is_interactive(self, tmp_path, can_render, scheme):
        scheme.plot(save_path=tmp_path / "scheme.html")
        html = (tmp_path / "scheme.html").read_text(errors="ignore")
        assert "<html" in html.lower() and "vtk" in html.lower()

    def test_missing_folder_displays_the_figure_instead_of_saving(
        self, tmp_path, can_render, scheme
    ):
        plotter = scheme.plot(save_path=tmp_path / "no_dir" / "scheme.png", show=False)
        try:
            assert isinstance(plotter, pv.Plotter)
            assert not (tmp_path / "no_dir").exists()
        finally:
            plotter.close()

    def test_returns_a_plotter(self, plotter, scheme):
        assert isinstance(plotter(scheme), pv.Plotter)

    @staticmethod
    def expected_actors(n_shells, colorbar=True, axes=True):
        # Spheres of the directions, central sphere, shell toroids, axes and title
        return 1 + 1 + n_shells + 3 * axes + 1 + colorbar

    def test_scene_components(self, plotter, scheme):
        assert len(plotter(scheme).renderer.actors) == self.expected_actors(2)

    def test_axes_and_colorbar_can_be_hidden(self, plotter, scheme):
        assert len(
            plotter(scheme, show_axes=False).renderer.actors
        ) == self.expected_actors(2, axes=False)
        assert len(
            plotter(scheme, show_colorbar=False).renderer.actors
        ) == self.expected_actors(2, colorbar=False)

    def test_one_toroid_per_shell(self, plotter):
        scheme = DiffusionScheme.simulate_dwi_acq_scheme(
            "shelled", shells={1000: 8, 2000: 8, 3000: 8}, n_b0s=1, n_iter=5
        )
        assert len(plotter(scheme).renderer.actors) == self.expected_actors(3)

    def test_cartesian_scheme_has_a_toroid_per_b_value(self, plotter):
        scheme = DiffusionScheme.simulate_dwi_acq_scheme("dsi", radius=3, n_b0s=1)
        n_shells = len(np.unique(scheme.bvals[scheme.bvals > 10]))
        assert len(plotter(scheme).renderer.actors) == self.expected_actors(n_shells)

    def test_opposite_directions(self, plotter, scheme):
        n = len(scheme.bvals)
        assert self.spheres(plotter(scheme)).n_points == 2 * n * GLYPH_POINTS
        assert (
            self.spheres(plotter(scheme, show_opposite_dirs=False)).n_points
            == n * GLYPH_POINTS
        )

    def test_points_are_colored_by_b_value(self, plotter, scheme):
        spheres = self.spheres(plotter(scheme))
        assert set(np.unique(spheres[BVAL_ARRAY])) == {0.0, 1000.0, 2000.0}

    def test_shelled_points_are_placed_at_a_distance_equal_to_the_b_value(
        self, plotter, scheme
    ):
        spheres = self.spheres(plotter(scheme))
        radius = (
            30 * 0.0006 * 2000
        )  # Size of the spheres: relative to the extent of the plot
        for bval in (1000, 2000):
            distance = np.linalg.norm(
                spheres.points[spheres[BVAL_ARRAY] == bval], axis=1
            )
            assert (
                distance.min() >= bval - radius - 1
                and distance.max() <= bval + radius + 1
            )

    def test_cartesian_points_are_placed_at_sqrt_of_b_times_bmax(self, plotter):
        scheme = DiffusionScheme.simulate_dwi_acq_scheme(
            "dsi", bmax=4000, radius=4, n_b0s=1
        )
        spheres = self.spheres(plotter(scheme))
        radius = 30 * 0.0006 * 4000
        distance = np.linalg.norm(spheres.points[spheres[BVAL_ARRAY] == 1000], axis=1)
        expected = np.sqrt(1000 * 4000)  # Not 1000, as in a shelled scheme
        assert distance.min() >= expected - radius - 1
        assert distance.max() <= expected + radius + 1

    @staticmethod
    def toroid_extents(plotter, tube_radius):
        """Outer radius of each toroid (the actors that are exactly as thick as the tube)."""
        extents = []
        for actor in plotter.renderer.actors.values():
            dataset = getattr(getattr(actor, "mapper", None), "dataset", None)
            if dataset is None:
                continue
            xmin, xmax, ymin, ymax, zmin, zmax = dataset.bounds
            if abs((zmax - zmin) - 2 * tube_radius) < 0.02 * tube_radius and xmax > 0:
                extents.append(max(xmax, ymax))
        return sorted(extents)

    def test_shelled_toroids_lie_at_the_b_value(self, plotter, scheme):
        extents = self.toroid_extents(plotter(scheme, toroid_radius=1.0), 1.0)
        assert extents == pytest.approx([1000 + 1, 2000 + 1], abs=0.5)

    def test_cartesian_toroids_lie_at_the_radius_of_their_points(self, plotter):
        scheme = DiffusionScheme.simulate_dwi_acq_scheme(
            "dsi", bmax=4000, radius=2, n_b0s=1
        )
        extents = self.toroid_extents(plotter(scheme, toroid_radius=1.0), 1.0)
        b_values = np.unique(scheme.bvals[scheme.bvals > 10])
        assert len(extents) == len(b_values)
        assert extents == pytest.approx(np.sort(np.sqrt(b_values * 4000) + 1), abs=0.5)

    def test_toroid_radius_defaults_to_a_fraction_of_the_largest_b_value(
        self, plotter, scheme
    ):
        extents = self.toroid_extents(plotter(scheme), 0.005 * 2000)
        assert extents == pytest.approx([1000 + 10, 2000 + 10], abs=0.5)

    def test_window_size(self, plotter, scheme):
        assert list(plotter(scheme, window_size=(400, 300)).window_size) == [400, 300]

    def test_other_options_are_accepted(self, plotter, scheme):
        plotter(
            scheme,
            colormap="viridis",
            sphere_radius=5,
            toroid_alpha=0.9,
            toroid_radius=3.0,
        )

    def test_scheme_type_is_detected_when_missing(self, plotter, scheme):
        bare = DiffusionScheme()
        bare.gradients, bare.bvals = scheme.gradients, scheme.bvals
        plotter(bare)
        assert bare.scheme_type == "shelled"
