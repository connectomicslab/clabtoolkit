"""Tests for the imagetools and imagetools_utils modules.

The tests follow usage_examples/examples-imagetools-methods.ipynb. Everything uses small
synthetic arrays and images written to a temporary folder.
"""

import json
import shutil

import nibabel as nib
import numpy as np
import pytest

import clabtoolkit.imagetools as cltimg
import clabtoolkit.imagetools_utils as imgutils

AFFINE_MNI = np.array(
    [
        [-1.0, 0.0, 0.0, 90.0],
        [0.0, 1.0, 0.0, -126.0],
        [0.0, 0.0, 1.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _affine_dwi():
    affine = np.diag([2.0, 2.0, 2.5, 1.0])
    affine[:3, 3] = [-96.0, -96.0, -60.0]
    return affine


def _affine_oblique(theta_deg=15):
    theta = np.deg2rad(theta_deg)
    rot_z = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    affine = np.eye(4)
    affine[:3, :3] = rot_z @ np.diag([2.0, 2.0, 2.0])
    affine[:3, 3] = [-80.0, -100.0, -50.0]
    return affine


@pytest.fixture
def morph():
    return cltimg.MorphologicalOperations()


@pytest.fixture
def img2d():
    """Square with a 3 x 3 hole, a rectangle and two isolated noisy pixels."""
    img = np.zeros((40, 40), dtype=bool)
    img[5:20, 5:20] = True
    img[10:13, 10:13] = False
    img[25:35, 22:36] = True
    img[30, 5] = True
    img[3, 35] = True
    return img


@pytest.fixture
def sphere():
    zz, yy, xx = np.ogrid[:40, :40, :40]
    return (xx - 20) ** 2 + (yy - 20) ** 2 + (zz - 20) ** 2 <= 10**2


@pytest.fixture
def offcenter_sphere():
    zz, yy, xx = np.ogrid[:40, :40, :40]
    return ((zz - 12) ** 2 + (yy - 20) ** 2 + (xx - 28) ** 2 <= 8**2).astype(np.uint8)


######################################################################################################
class TestStructuringElement:
    def test_default_is_3x3x3_cube(self, morph):
        cube = morph.create_structuring_element()
        assert cube.shape == (3, 3, 3)
        assert cube.sum() == 27

    def test_ball_and_disk(self, morph):
        ball = morph.create_structuring_element(shape="ball", size=7, dimensions=3)
        disk = morph.create_structuring_element(shape="disk", size=5, dimensions=2)
        assert ball.shape == (7, 7, 7)
        assert disk.shape == (5, 5)
        assert ball[3, 3, 3] and disk[2, 2]
        assert not ball[0, 0, 0] and not disk[0, 0]

    def test_cross_has_face_connectivity_only(self, morph):
        assert morph.create_structuring_element("cross", dimensions=3).sum() == 7
        cross2d = morph.create_structuring_element("cross", dimensions=2)
        assert cross2d.astype(int).tolist() == [[0, 1, 0], [1, 1, 1], [0, 1, 0]]

    def test_unsupported_shape(self, morph):
        with pytest.raises(ValueError):
            morph.create_structuring_element(shape="pyramid")


class TestMorphologicalOperations:
    def test_erode_removes_noise(self, morph, img2d):
        eroded = morph.erode(img2d)
        assert eroded.sum() < img2d.sum()
        assert not eroded[30, 5] and not eroded[3, 35]

    def test_erode_iterations(self, morph, sphere):
        sizes = [morph.erode(sphere, iterations=i).sum() for i in (1, 3)]
        assert sphere.sum() > sizes[0] > sizes[1] > 0

    def test_erode_cross_removes_less_than_square(self, morph, img2d):
        cross = morph.create_structuring_element("cross", dimensions=2)
        assert morph.erode(img2d, structure=cross).sum() > morph.erode(img2d).sum()

    def test_dilate_grows_the_mask(self, morph, img2d, sphere):
        assert morph.dilate(img2d).sum() > img2d.sum()
        assert morph.dilate(sphere, iterations=2).sum() > morph.dilate(sphere).sum()

    def test_dilate_contains_original(self, morph, img2d):
        assert np.all(morph.dilate(img2d)[img2d])

    def test_opening_removes_isolated_pixels(self, morph, img2d):
        opened = morph.opening(img2d)
        assert not opened[30, 5] and not opened[3, 35]
        assert opened[5:10, 5:10].all()

    def test_opening_removes_spur(self, morph, sphere):
        spur = sphere.copy()
        spur[20, 20, 31:39] = True
        assert not morph.opening(spur)[20, 20, 31:39].any()

    def test_closing_iterations_fill_the_hole(self, morph, img2d):
        closed1 = morph.closing(img2d, iterations=1)
        closed2 = morph.closing(img2d, iterations=2)
        assert (~closed1[10:13, 10:13]).sum() > 0
        assert (~closed2[10:13, 10:13]).sum() == 0

    def test_closing_bridges_a_gap(self, morph):
        blocks = np.zeros((20, 20, 20), dtype=bool)
        blocks[5:15, 5:15, 2:9] = True
        blocks[5:15, 5:15, 10:17] = True
        assert not blocks[5:15, 5:15, 9].any()
        assert morph.closing(blocks)[5:15, 5:15, 9].all()

    def test_fill_holes(self, morph, img2d):
        filled = morph.fill_holes(img2d)
        assert filled.sum() - img2d.sum() == 9
        assert filled[10:13, 10:13].all()

    def test_fill_hollow_sphere(self, morph, sphere):
        shell = sphere & ~morph.erode(sphere, iterations=2)
        filled = morph.fill_holes(shell)
        assert filled.sum() > shell.sum()
        assert np.array_equal(filled, sphere)

    def test_hole_connected_to_background_is_not_filled(self, morph, img2d):
        ring = img2d.copy()
        ring[11, 5:10] = False  # channel from the hole to the outside
        assert morph.fill_holes(ring).sum() == ring.sum()

    def test_detect_communities(self, morph, img2d):
        labeled, n = morph.detect_communities(img2d)
        assert n == 4
        assert sorted(np.bincount(labeled.ravel())[1:].tolist()) == [1, 1, 140, 216]

    def test_connectivity_changes_the_number_of_components(self, morph):
        diag = np.zeros((5, 5), dtype=bool)
        diag[1, 1] = diag[2, 2] = True
        assert morph.detect_communities(diag)[1] == 1
        cross = morph.create_structuring_element("cross", dimensions=2)
        assert morph.detect_communities(diag, structure=cross)[1] == 2

    def test_detect_three_spheres(self, morph):
        three = np.zeros((60, 30, 30), dtype=bool)
        zz, yy, xx = np.ogrid[:60, :30, :30]
        for cx in (10, 30, 50):
            three |= (zz - cx) ** 2 + (yy - 15) ** 2 + (xx - 15) ** 2 <= 6**2
        labeled, n = morph.detect_communities(three)
        assert n == 3
        assert len(set(np.bincount(labeled.ravel())[1:].tolist())) == 1

    def test_remove_small_objects(self, morph, img2d):
        clean = morph.remove_small_objects(img2d, min_size=5)
        assert morph.detect_communities(clean)[1] == 2
        large = morph.remove_small_objects(img2d, min_size=150)
        assert large.sum() == 216  # only the square remains
        assert not large[25:35, 22:36].any()

    def test_gradient_extracts_boundaries(self, morph, img2d, sphere):
        edges = morph.gradient(img2d)
        cross = morph.create_structuring_element("cross", dimensions=2)
        assert 0 < morph.gradient(img2d, structure=cross).sum() < edges.sum()
        shell = morph.gradient(sphere)
        assert 0 < shell.sum() < sphere.sum()
        assert not shell[20, 20, 20]

    def test_tophat_extracts_thin_structures(self, morph, img2d):
        line = img2d.copy()
        line[30, 36:39] = True
        th = morph.tophat(line)
        assert th[30, 5] and th[3, 35]
        assert th[30, 36:39].any()
        assert not th[5:20, 5:20].any()

    def test_blackhat_extracts_small_dark_structures(self, morph, img2d):
        dark = img2d.copy()
        dark[25:35, 29] = False
        dark[16, 16] = False
        bh = morph.blackhat(dark)
        assert bh[16, 16] and bh[25:35, 29].all()
        disk = morph.create_structuring_element("disk", size=7, dimensions=2)
        assert morph.blackhat(dark, structure=disk)[10:13, 10:13].all()

    def test_non_binary_input_is_binarized(self, morph):
        arr = np.zeros((10, 10), dtype=np.int16)
        arr[3:7, 3:7] = 5
        assert morph.dilate(arr).dtype == bool


class TestQuickMorphology:
    def test_erode(self, morph, img2d):
        assert np.array_equal(
            cltimg.quick_morphology(img2d, "erode", iterations=2),
            morph.erode(img2d, iterations=2),
        )

    def test_chaining(self, img2d, morph):
        clean = cltimg.quick_morphology(img2d, "remove_small", min_size=5)
        clean = cltimg.quick_morphology(clean, "fill_holes")
        assert morph.detect_communities(clean)[1] == 2
        assert clean.sum() == 216 + 140 + 9

    def test_closing_with_structure(self, morph, sphere):
        cavity = sphere.copy()
        cavity[19:21, 19:21, 19:21] = False
        ball = morph.create_structuring_element("ball", size=5, dimensions=3)
        closed = cltimg.quick_morphology(cavity, "closing", structure=ball)
        assert closed.sum() > cavity.sum()

    def test_unsupported_operation(self, img2d):
        with pytest.raises(ValueError):
            cltimg.quick_morphology(img2d, "skeletonize")


class TestDilateMm:
    @pytest.fixture
    def seed(self):
        arr = np.zeros((21, 21, 21), dtype=np.uint8)
        arr[10, 10, 10] = 1
        return arr

    @staticmethod
    def extent(mask):
        idx = np.argwhere(mask)
        return tuple(int(v) for v in idx.max(axis=0) - idx.min(axis=0) + 1)

    @pytest.mark.parametrize("func", ["method", "function"])
    def test_cube_isotropic(self, func, seed, morph):
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        if func == "method":
            dil = morph.dilate_mm(seed, affine, shape="cube", dilation_mm=2)
        else:
            dil = cltimg.dilate_mm(seed, affine, shape="cube", dilation_mm=2)
        assert dil.dtype == bool
        assert self.extent(dil) == (5, 5, 5)
        assert dil.sum() == 125

    def test_voxel_size_changes_the_radius_in_voxels(self, seed):
        dil = cltimg.dilate_mm(seed, np.diag([1.5, 1.5, 1.5, 1.0]), shape="cube", dilation_mm=3)
        assert self.extent(dil) == (5, 5, 5)

    def test_anisotropic_sphere(self, seed):
        dil = cltimg.dilate_mm(seed, np.diag([1.0, 1.0, 3.0, 1.0]), shape="sphere", dilation_mm=3)
        assert self.extent(dil) == (7, 7, 3)

    def test_ball_is_smaller_than_cube(self, seed, morph):
        affine = np.eye(4)
        ball = morph.dilate_mm(seed, affine, shape="ball", dilation_mm=2)
        cube = morph.dilate_mm(seed, affine, shape="cube", dilation_mm=2)
        assert 0 < ball.sum() < cube.sum()

    def test_labels_are_binarized(self):
        labels = np.zeros((30, 30, 30), dtype=np.int16)
        labels[10:15, 10:15, 10:15] = 7
        labels[16:20, 16:20, 16:20] = 12
        dil = cltimg.dilate_mm(labels, np.eye(4), dilation_mm=2)
        assert dil.dtype == bool
        assert dil.sum() > np.count_nonzero(labels)


######################################################################################################
class TestAffineAttributes:
    def test_voxel_size(self):
        assert np.allclose(cltimg.get_voxel_size(AFFINE_MNI), [1, 1, 1])
        assert np.allclose(cltimg.get_voxel_size(_affine_dwi()), [2, 2, 2.5])

    def test_voxel_size_ignores_rotation(self):
        assert np.allclose(cltimg.get_voxel_size(_affine_oblique()), [2, 2, 2])

    def test_voxel_volume(self):
        assert cltimg.get_voxel_volume(AFFINE_MNI) == pytest.approx(1.0)
        assert cltimg.get_voxel_volume(_affine_dwi()) == pytest.approx(10.0)
        assert cltimg.get_voxel_volume(_affine_oblique()) == pytest.approx(8.0)

    def test_center_is_the_origin(self):
        assert np.allclose(cltimg.get_center(AFFINE_MNI), [90, -126, -72])
        assert np.allclose(
            cltimg.get_center(_affine_dwi()),
            cltimg.vox2mm(np.array([[0, 0, 0]]), _affine_dwi())[0],
        )

    def test_rotation_matrix_keeps_only_the_flip(self):
        assert np.allclose(cltimg.get_rotation_matrix(AFFINE_MNI), np.diag([-1, 1, 1]))
        assert np.allclose(cltimg.get_rotation_matrix(_affine_dwi()), np.eye(3))

    def test_rotation_matrix_of_oblique_acquisition(self):
        rot = cltimg.get_rotation_matrix(_affine_oblique(15))
        assert np.allclose(rot @ rot.T, np.eye(3))
        assert np.rad2deg(np.arctan2(rot[1, 0], rot[0, 0])) == pytest.approx(15)


class TestVoxMm:
    def test_vox2mm_mni_corners(self):
        vox = np.array([[0, 0, 0], [181, 217, 181], [90, 126, 72]])
        mm = cltimg.vox2mm(vox, AFFINE_MNI)
        assert np.allclose(mm, [[90, -126, -72], [-91, 91, 109], [0, 0, 0]])

    def test_vox2mm_accepts_3xN(self):
        vox = np.array([[0, 10, 20, 30]] * 3)
        mm = cltimg.vox2mm(vox, _affine_dwi())
        assert mm.shape == (4, 3)
        assert np.allclose(mm[1], [-76, -76, -35])

    def test_vox2mm_wrong_columns(self):
        with pytest.raises(ValueError):
            cltimg.vox2mm(np.array([[1, 2], [3, 4]]), AFFINE_MNI)

    def test_mm2vox_mni_origin(self):
        assert np.allclose(cltimg.mm2vox(np.array([[0.0, 0.0, 0.0]]), AFFINE_MNI), [[90, 126, 72]])

    def test_round_trip_in_oblique_image(self):
        vox = np.array([[3, 7, 11], [40, 2, 25]])
        affine = _affine_oblique()
        assert np.allclose(cltimg.mm2vox(cltimg.vox2mm(vox, affine), affine), vox)

    def test_mm2vox_wrong_columns(self):
        with pytest.raises(ValueError):
            cltimg.mm2vox(np.array([[1, 2], [3, 4]]), AFFINE_MNI)


class TestGetVoxNeighbors:
    center = np.array([10, 15, 20])

    @pytest.mark.parametrize("nb, n", [("6", 6), ("12", 12), ("18", 18), ("26", 26)])
    def test_number_of_neighbors_3d(self, nb, n):
        neigh = cltimg.get_vox_neighbors(self.center, neighborhood=nb, dims="3")
        assert len(neigh) == n
        assert not (neigh == self.center).all(axis=1).any()

    def test_6_neighbors_are_the_faces(self):
        neigh = cltimg.get_vox_neighbors(self.center, neighborhood="6")
        assert np.all(np.abs(neigh - self.center).sum(axis=1) == 1)

    def test_2d_neighbors(self):
        pix = np.array([5, 5])
        n4 = cltimg.get_vox_neighbors(pix, neighborhood="4", dims="2")
        assert sorted(n4.tolist()) == [[4, 5], [5, 4], [5, 6], [6, 5]]
        assert len(cltimg.get_vox_neighbors(pix, neighborhood="8", dims="2")) == 8

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError):
            cltimg.get_vox_neighbors(np.array([5, 5]), neighborhood="26", dims="3")


######################################################################################################
class TestSimulateImage:
    def test_from_shape(self, tmp_path):
        out = tmp_path / "sim.nii.gz"
        img = cltimg.simulate_image((16, 16, 10), str(out), loc=100, scale=15, random_seed=42)
        assert out.is_file()
        assert img.shape == (16, 16, 10)
        data = img.get_fdata()
        assert data.mean() == pytest.approx(100, abs=3)
        assert data.std() == pytest.approx(15, abs=3)

    def test_reproducible(self, tmp_path):
        a = cltimg.simulate_image((8, 8, 8), str(tmp_path / "a.nii.gz"), random_seed=1)
        b = cltimg.simulate_image((8, 8, 8), str(tmp_path / "b.nii.gz"), random_seed=1)
        assert np.array_equal(a.get_fdata(), b.get_fdata())

    def test_4d_uniform_with_affine(self, tmp_path):
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        img = cltimg.simulate_image(
            (10, 10, 8),
            str(tmp_path / "sim4d.nii.gz"),
            affine=affine,
            n_volumes=4,
            distribution="uniform",
            low=0,
            high=100,
            random_seed=1,
        )
        assert img.shape == (10, 10, 8, 4)
        assert np.allclose(img.affine, affine)
        assert 0 <= img.get_fdata().min() and img.get_fdata().max() <= 100

    def test_values_only_inside_a_mask_file(self, tmp_path):
        mask = np.zeros((20, 20, 12), dtype=np.uint8)
        mask[4:16, 4:16, 3:9] = 1
        mask_file = tmp_path / "mask.nii.gz"
        nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_file)
        img = cltimg.simulate_image(
            str(mask_file),
            str(tmp_path / "exp.nii.gz"),
            distribution="exponential",
            scale=2.0,
            random_seed=3,
        )
        data = img.get_fdata()
        assert np.all(data[mask == 0] == 0)
        assert data[mask == 1].mean() == pytest.approx(2.0, abs=0.3)

    def test_unsupported_distribution(self, tmp_path):
        with pytest.raises(ValueError):
            cltimg.simulate_image((5, 5, 5), str(tmp_path / "bad.nii.gz"), distribution="poisson")


class TestSimulateArray:
    def test_values_inside_the_mask_only(self, offcenter_sphere):
        sim = cltimg.simulate_array(offcenter_sphere, loc=10, scale=2, random_seed=42)
        assert sim.shape == offcenter_sphere.shape
        assert np.all(sim[offcenter_sphere == 0] == 0)
        assert sim[offcenter_sphere > 0].mean() == pytest.approx(10, abs=0.5)

    def test_4d_uniform(self, offcenter_sphere):
        sim = cltimg.simulate_array(
            offcenter_sphere, n_volumes=8, distribution="uniform", low=-1, high=1, random_seed=42
        )
        assert sim.shape == offcenter_sphere.shape + (8,)
        assert sim.min() >= -1 and sim.max() <= 1

    def test_reproducible(self, offcenter_sphere):
        a = cltimg.simulate_array(offcenter_sphere, distribution="exponential", scale=3, random_seed=7)
        b = cltimg.simulate_array(offcenter_sphere, distribution="exponential", scale=3, random_seed=7)
        assert np.array_equal(a, b)
        assert a[offcenter_sphere > 0].mean() == pytest.approx(3, abs=0.4)

    def test_empty_mask(self):
        with pytest.raises(RuntimeError):
            cltimg.simulate_array(np.zeros((5, 5, 5)))

    def test_invalid_uniform_limits(self, offcenter_sphere):
        with pytest.raises(ValueError):
            cltimg.simulate_array(offcenter_sphere, distribution="uniform", low=5, high=1)


######################################################################################################
class TestDeleteVolumes:
    @pytest.fixture
    def arr(self):
        return np.broadcast_to(np.arange(20, dtype=np.float32), (8, 8, 4, 20)).copy()

    def test_remove_individual_volumes(self, arr):
        out, removed = cltimg.delete_volumes_from_4D_array(arr, vols_to_delete=[0, 1, 2, 19])
        assert out.shape == (8, 8, 4, 16)
        assert out[0, 0, 0].astype(int).tolist() == list(range(3, 19))
        assert sorted(removed) == [0, 1, 2, 19]

    def test_tuples_strings_and_arrays(self, arr):
        out, removed = cltimg.delete_volumes_from_4D_array(
            arr, vols_to_delete=[(3, 5), "10:2:16", np.array([18])]
        )
        kept = out[0, 0, 0].astype(int).tolist()
        assert not {3, 4, 5, 10, 12, 14, 18} & set(kept)
        assert len(kept) == 20 - len(set(removed))

    def test_input_is_not_modified(self, arr):
        before = arr.copy()
        cltimg.delete_volumes_from_4D_array(arr, vols_to_delete=[0])
        assert np.array_equal(arr, before)

    def test_out_of_range(self, arr):
        with pytest.raises(ValueError):
            cltimg.delete_volumes_from_4D_array(arr, vols_to_delete=[30])

    def test_3d_array(self, arr):
        with pytest.raises(ValueError):
            cltimg.delete_volumes_from_4D_array(arr[..., 0], vols_to_delete=[0])


class TestDeleteVolumesFromImages:
    @pytest.fixture
    def in_file(self, tmp_path):
        data = np.broadcast_to(np.arange(20, dtype=np.float32), (16, 16, 10, 20)).copy()
        path = tmp_path / "sub-01_task-rest_bold.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), path)
        return str(path)

    @staticmethod
    def kept(file):
        return nib.load(file).get_fdata()[0, 0, 0, :].astype(int).tolist()

    def test_remove_dummy_scans(self, in_file, tmp_path):
        out_file, removed = cltimg.delete_volumes_from_4D_images(
            in_file, str(tmp_path / "clean.nii.gz"), vols_to_delete=[0, 1, 2, 19]
        )
        assert self.kept(out_file) == list(range(3, 19))

    def test_string_ranges(self, in_file, tmp_path):
        out_file, removed = cltimg.delete_volumes_from_4D_images(
            in_file,
            str(tmp_path / "subset.nii.gz"),
            vols_to_delete=[(0, 1), "5-7, 10:2:14"],
        )
        kept = self.kept(out_file)
        assert not {0, 1, 5, 6, 7, 10, 12, 14} & set(kept)

    def test_existing_output_requires_overwrite(self, in_file, tmp_path):
        out = str(tmp_path / "clean.nii.gz")
        cltimg.delete_volumes_from_4D_images(in_file, out, vols_to_delete=[0])
        with pytest.raises(FileExistsError):
            cltimg.delete_volumes_from_4D_images(in_file, out, vols_to_delete=[0])
        cltimg.delete_volumes_from_4D_images(in_file, out, vols_to_delete=[0, 1], overwrite=True)
        assert nib.load(out).shape[-1] == 18

    def test_out_of_range(self, in_file, tmp_path):
        with pytest.raises(ValueError):
            cltimg.delete_volumes_from_4D_images(
                in_file, str(tmp_path / "bad.nii.gz"), vols_to_delete=[5, 25], overwrite=True
            )


######################################################################################################
class TestMergeTo4d:
    affine = np.diag([2.0, 2.0, 2.0, 1.0])

    @pytest.fixture
    def runs_3d(self, tmp_path):
        files = []
        for i in range(3):
            f = tmp_path / f"sub-01_run-{i + 1}_T2w.nii.gz"
            cltimg.simulate_image((12, 12, 8), str(f), affine=self.affine, random_seed=i)
            files.append(str(f))
        return files

    @pytest.fixture
    def run_4d(self, tmp_path):
        f = tmp_path / "sub-01_run-4_T2w.nii.gz"
        cltimg.simulate_image((12, 12, 8), str(f), affine=self.affine, n_volumes=4, random_seed=3)
        return str(f)

    def test_merge_3d_in_memory(self, runs_3d):
        merged, meta = cltimg.merge_to_4d(runs_3d)
        assert merged.shape == (12, 12, 8, 3)
        assert meta is None  # no sidecars next to the images

    def test_merge_mixed_3d_and_4d_and_save(self, runs_3d, run_4d, tmp_path):
        out = tmp_path / "merged.nii.gz"
        merged, _ = cltimg.merge_to_4d(runs_3d + [run_4d], output_path=str(out))
        assert merged.shape[-1] == 7
        assert nib.load(out).shape == (12, 12, 8, 7)

    def test_volume_order_is_preserved(self, runs_3d):
        merged, _ = cltimg.merge_to_4d(runs_3d)
        for i, f in enumerate(runs_3d):
            assert np.allclose(np.asarray(merged.dataobj)[..., i], nib.load(f).get_fdata())

    def test_dwi_sidecars_are_concatenated(self, tmp_path):
        files = []
        for i, n_dirs in enumerate([6, 4]):
            stem = tmp_path / f"sub-01_run-{i + 1}_dwi"
            cltimg.simulate_image(
                (12, 12, 8), f"{stem}.nii.gz", affine=self.affine, n_volumes=n_dirs, random_seed=i
            )
            bvecs = np.random.default_rng(i).normal(size=(3, n_dirs))
            bvecs /= np.linalg.norm(bvecs, axis=0)
            np.savetxt(f"{stem}.bvec", bvecs, fmt="%.6f")
            np.savetxt(f"{stem}.bval", np.r_[0, np.full(n_dirs - 1, 1000 * (i + 1))][None, :], fmt="%d")
            with open(f"{stem}.json", "w") as fh:
                json.dump({"RepetitionTime": 3.2, "PhaseEncodingDirection": "j-" if i == 0 else "j"}, fh)
            files.append(f"{stem}.nii.gz")

        out = tmp_path / "sub-01_dwi.nii.gz"
        merged, meta = cltimg.merge_to_4d(files, output_path=str(out))
        assert merged.shape[-1] == 10
        assert meta["bvecs"].shape == (3, 10)
        assert meta["bvals"].astype(int).tolist() == [0, 1000, 1000, 1000, 1000, 1000, 0, 2000, 2000, 2000]
        assert meta["json"]["RepetitionTime"] == 3.2
        assert meta["json"]["PhaseEncodingDirection"] == "j-"  # first file wins
        assert (tmp_path / "sub-01_dwi.bval").is_file()
        assert (tmp_path / "sub-01_dwi.bvec").is_file()
        assert (tmp_path / "sub-01_dwi.json").is_file()

    def test_different_affines(self, runs_3d, tmp_path):
        other = tmp_path / "sub-01_run-5_T2w.nii.gz"
        cltimg.simulate_image((12, 12, 8), str(other), affine=np.eye(4), random_seed=5)
        with pytest.raises((ValueError, OSError)):
            cltimg.merge_to_4d([runs_3d[0], str(other)])


######################################################################################################
@pytest.fixture
def brain_images(tmp_path):
    """T1w, 4D BOLD and a spherical brain mask on the same grid."""
    shape = (32, 32, 24)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [-30, -30, -20]
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    brain = (zz - 16) ** 2 + (yy - 16) ** 2 + (xx - 12) ** 2 <= 8**2

    paths = {
        "mask": tmp_path / "mask.nii.gz",
        "t1": tmp_path / "T1w.nii.gz",
        "bold": tmp_path / "bold.nii.gz",
    }
    nib.save(nib.Nifti1Image(brain.astype(np.uint8), affine), paths["mask"])
    cltimg.simulate_image(
        shape, str(paths["t1"]), affine=affine, distribution="uniform", low=100, high=1000, random_seed=0
    )
    cltimg.simulate_image(shape, str(paths["bold"]), affine=affine, n_volumes=5, random_seed=0)
    paths = {k: str(v) for k, v in paths.items()}
    paths["brain"] = brain
    paths["affine"] = affine
    return paths


class TestCropImageFromMask:
    def test_crop_to_mask_bounding_box(self, brain_images, tmp_path):
        out = cltimg.crop_image_from_mask(
            brain_images["t1"], brain_images["mask"], str(tmp_path / "crop.nii.gz")
        )
        crop = nib.load(out)
        idx = np.argwhere(brain_images["brain"])
        expected = tuple(idx.max(axis=0) - idx.min(axis=0) + 1)
        assert crop.shape == expected

    def test_affine_keeps_the_mm_position(self, brain_images, tmp_path):
        out = cltimg.crop_image_from_mask(
            brain_images["t1"], brain_images["mask"], str(tmp_path / "crop.nii.gz")
        )
        crop = nib.load(out)
        first = np.argwhere(brain_images["brain"]).min(axis=0)
        assert np.allclose(
            crop.affine[:3, 3], cltimg.vox2mm(first[None, :], brain_images["affine"])[0]
        )

    def test_crop_with_array_and_structure_codes(self, brain_images, tmp_path):
        parc = np.zeros(brain_images["brain"].shape, dtype=np.int16)
        parc[5:10, 5:10, 5:10] = 3
        parc[15:20, 12:18, 8:12] = 7
        out = cltimg.crop_image_from_mask(
            brain_images["t1"], parc, str(tmp_path / "crop37.nii.gz"), st_codes=[3, 7]
        )
        assert nib.load(out).shape == (15, 13, 7)

    def test_crop_4d(self, brain_images, tmp_path):
        out = cltimg.crop_image_from_mask(
            brain_images["bold"], brain_images["mask"], str(tmp_path / "crop_bold.nii.gz")
        )
        shape = nib.load(out).shape
        assert len(shape) == 4 and shape[-1] == 5

    def test_mask_with_different_shape(self, brain_images, tmp_path):
        with pytest.raises(ValueError):
            cltimg.crop_image_from_mask(
                brain_images["t1"], np.ones((10, 10, 10)), str(tmp_path / "bad.nii.gz")
            )


class TestCroppedToNative:
    def test_restore_3d(self, brain_images, tmp_path):
        crop = cltimg.crop_image_from_mask(
            brain_images["t1"], brain_images["mask"], str(tmp_path / "crop.nii.gz")
        )
        native = cltimg.cropped_to_native(crop, brain_images["t1"], str(tmp_path / "restored.nii.gz"))
        restored = nib.load(native).get_fdata()
        original = nib.load(brain_images["t1"]).get_fdata()
        assert restored.shape == original.shape
        nz = restored != 0
        assert nz.any()
        assert np.allclose(restored[nz], original[nz])

    def test_restore_4d(self, brain_images, tmp_path):
        crop = cltimg.crop_image_from_mask(
            brain_images["bold"], brain_images["mask"], str(tmp_path / "crop_bold.nii.gz")
        )
        native = cltimg.cropped_to_native(
            crop, brain_images["bold"], str(tmp_path / "restored_bold.nii.gz")
        )
        assert nib.load(native).shape == nib.load(brain_images["bold"]).shape


@pytest.mark.skipif(shutil.which("antsApplyTransforms") is None, reason="ANTs is not installed")
class TestApplyMultiTransf:
    def test_unsupported_interpolation(self, brain_images, tmp_path):
        with pytest.raises(ValueError):
            cltimg.apply_multi_transf(
                brain_images["t1"],
                str(tmp_path / "out.nii.gz"),
                brain_images["t1"],
                str(tmp_path / "xfm"),
                interp_order=9,
            )

    def test_missing_transform(self, brain_images, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltimg.apply_multi_transf(
                brain_images["t1"],
                str(tmp_path / "out.nii.gz"),
                brain_images["t1"],
                str(tmp_path / "missing_xfm"),
                overwrite=True,
            )


######################################################################################################
class TestSpams:
    @pytest.fixture
    def parc_files(self, tmp_path):
        base = np.zeros((20, 20, 12), dtype=np.int16)
        base[2:10, 2:10, 2:10] = 1
        base[10:18, 2:10, 2:10] = 2
        base[2:10, 10:18, 2:10] = 3
        base[10:18, 10:18, 2:10] = 4
        rng = np.random.default_rng(0)
        files = []
        for subj in range(5):
            shifted = np.roll(base, shift=tuple(rng.integers(-1, 2, size=3)), axis=(0, 1, 2))
            f = tmp_path / f"sub-{subj + 1:02d}_atlas-sim_dseg.nii.gz"
            nib.save(nib.Nifti1Image(shifted, np.eye(4)), f)
            files.append(str(f))
        return files

    @pytest.fixture
    def lut_file(self, tmp_path):
        path = tmp_path / "atlas-sim.lut"
        colors = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200)]
        path.write_text(
            "".join(f"{i}  region-{i:02d}  {r} {g} {b} 0\n" for i, (r, g, b) in enumerate(colors, 1))
        )
        return str(path)

    def test_create_spams_from_lut_file(self, parc_files, lut_file, tmp_path):
        out = tmp_path / "spam.nii.gz"
        cltimg.create_spams(parc_files, str(out), lut_table=lut_file)
        spam = nib.load(out).get_fdata()
        assert spam.shape == (20, 20, 12, 4)
        assert spam.min() >= 0 and spam.max() <= 1

    def test_create_spams_without_lut(self, parc_files, tmp_path):
        out = tmp_path / "spam_nolut.nii.gz"
        cltimg.create_spams(parc_files, str(out), lut_table=None)
        assert nib.load(out).shape[:3] == (20, 20, 12)

    def test_create_spams_from_dictionary_with_colors(self, parc_files, tmp_path):
        lut = {
            "index": [1, 2, 3, 4],
            "name": [f"region-{i:02d}" for i in range(1, 5)],
            "color": ["#e6194b", "#3cb44b", "#ffe119", "#0082c8"],
        }
        out = tmp_path / "spam_dict.nii.gz"
        cltimg.create_spams(parc_files, str(out), lut_table=lut, save_color_spams=True)
        assert nib.load(out).shape == (20, 20, 12, 4)
        assert (tmp_path / "spam_dict_colored.nii.gz").is_file()

    def test_spams2maxprob_returns_array_and_saves(self, parc_files, lut_file, tmp_path):
        spam_file = tmp_path / "spam.nii.gz"
        cltimg.create_spams(parc_files, str(spam_file), lut_table=lut_file)
        maxprob = cltimg.spams2maxprob(str(spam_file))
        assert set(np.unique(maxprob)) <= {0, 1, 2, 3, 4}
        out = tmp_path / "maxprob70.nii.gz"
        cltimg.spams2maxprob(str(spam_file), prob_thresh=0.7, maxp_name=str(out))
        assert (nib.load(out).get_fdata() == 0).sum() >= (maxprob == 0).sum()

    def test_spams2maxprob_subset_of_volumes(self, parc_files, lut_file, tmp_path):
        spam_file = tmp_path / "spam.nii.gz"
        cltimg.create_spams(parc_files, str(spam_file), lut_table=lut_file)
        assert set(np.unique(cltimg.spams2maxprob(str(spam_file), vol_indexes=[0, 1]))) <= {0, 1, 2}


class TestSpamsFromVolume:
    @pytest.fixture
    def indiv(self):
        base = np.zeros((20, 20, 12), dtype=np.int32)
        base[2:10, 2:10, 2:10] = 1
        base[10:18, 2:10, 2:10] = 2
        base[2:10, 10:18, 2:10] = 5
        rng = np.random.default_rng(0)
        return np.stack(
            [np.roll(base, shift=tuple(rng.integers(-1, 2, size=3)), axis=(0, 1, 2)) for _ in range(5)],
            axis=-1,
        )

    def test_probabilities_sum_to_one(self, indiv):
        spams = cltimg.create_spams_from_volume(indiv, np.unique(indiv))
        assert spams.shape == indiv.shape[:3] + (4,)
        assert np.allclose(spams.sum(axis=3), 1)

    def test_selected_regions(self, indiv):
        spams = cltimg.create_spams_from_volume(indiv, [2, 5])
        assert spams.shape[-1] == 2
        expected = (indiv == 2).mean(axis=-1)
        assert np.allclose(spams[..., 0], expected)

    def test_3d_input(self, indiv):
        with pytest.raises(ValueError):
            cltimg.create_spams_from_volume(indiv[..., 0], [1, 2])

    def test_maxprob(self, indiv):
        spams = cltimg.create_spams_from_volume(indiv, np.unique(indiv))
        maxprob = cltimg.spams2maxprob_from_volume(spams)
        assert maxprob.shape == indiv.shape[:3]
        # labels are the 1-based position of the winning volume, one per region
        assert set(np.unique(maxprob)) == {1, 2, 3, 4}

    def test_maxprob_threshold_leaves_more_unlabeled(self, indiv):
        spams = cltimg.create_spams_from_volume(indiv, np.unique(indiv))
        base = cltimg.spams2maxprob_from_volume(spams)
        strict = cltimg.spams2maxprob_from_volume(spams, prob_thresh=0.9)
        assert (strict == 0).sum() >= (base == 0).sum()

    def test_maxprob_subset_does_not_modify_input(self, indiv):
        spams = cltimg.create_spams_from_volume(indiv, np.unique(indiv))
        before = spams.copy()
        sub = cltimg.spams2maxprob_from_volume(spams, vol_indexes=[0, 3])
        assert set(np.unique(sub)) <= {0, 1, 4}
        assert np.array_equal(spams, before)


######################################################################################################
class TestMeshAndCentroid:
    def test_mesh_in_voxel_coordinates(self, offcenter_sphere):
        mesh = cltimg.extract_mesh_from_volume(offcenter_sphere)
        assert mesh.n_points > 0 and mesh.n_cells > 0
        bounds = np.array(mesh.bounds)
        centre = (bounds[0::2] + bounds[1::2]) / 2
        assert len(centre) == 3
        assert sorted(np.round(centre).tolist()) == [12, 20, 28]

    def test_raw_mesh(self, offcenter_sphere):
        raw = cltimg.extract_mesh_from_volume(offcenter_sphere, gaussian_smooth=False)
        assert raw.n_points > 0

    def test_mesh_in_mm(self, offcenter_sphere):
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        affine[:3, 3] = [-40, -40, -40]
        mesh = cltimg.extract_mesh_from_volume(offcenter_sphere, affine=affine, smooth_iterations=30)
        # area of a 16 mm radius sphere, within the discretization error
        assert mesh.area == pytest.approx(4 * np.pi * 16**2, rel=0.15)

    def test_vertex_value(self, offcenter_sphere):
        mesh = cltimg.extract_mesh_from_volume(offcenter_sphere, vertex_value=4)
        assert np.unique(mesh.point_data["default"]).tolist() == [4]

    def test_mesh_2d_input(self):
        with pytest.raises(ValueError):
            cltimg.extract_mesh_from_volume(np.ones((10, 10)))

    def test_centroid_and_voxel_count(self, offcenter_sphere):
        centroid, nvox = cltimg.extract_centroid_from_volume(
            offcenter_sphere, gaussian_smooth=False, closing_iterations=0
        )
        assert np.allclose(centroid, [12, 20, 28], atol=0.01)
        assert nvox == offcenter_sphere.sum()

    def test_closing_recovers_the_centroid(self, offcenter_sphere):
        cavity = offcenter_sphere.copy()
        cavity[11:14, 19:22, 30:33] = 0
        c_open, nvox = cltimg.extract_centroid_from_volume(
            cavity, gaussian_smooth=False, closing_iterations=0
        )
        c_closed, _ = cltimg.extract_centroid_from_volume(
            cavity, gaussian_smooth=False, closing_iterations=2
        )
        target = np.array([12, 20, 28])
        assert np.linalg.norm(c_closed - target) < np.linalg.norm(c_open - target)
        assert nvox == cavity.sum()

    def test_smoothing_ignores_thin_spur(self, offcenter_sphere):
        spur = offcenter_sphere.copy()
        spur[12, 20, 37:40] = 1
        c_raw, n_raw = cltimg.extract_centroid_from_volume(spur, gaussian_smooth=False, closing_iterations=0)
        c_smooth, n_smooth = cltimg.extract_centroid_from_volume(spur, gaussian_smooth=True, sigma=1.0)
        target = np.array([12, 20, 28])
        assert np.linalg.norm(c_smooth - target) < np.linalg.norm(c_raw - target)
        assert n_raw == n_smooth == spur.sum()

    def test_empty_volume(self):
        centroid, nvox = cltimg.extract_centroid_from_volume(np.zeros((10, 10, 10)))
        assert np.isnan(centroid).all()
        assert nvox == 0


######################################################################################################
class TestStatisticsAtNonzeroVoxels:
    mask = np.array([[[0, 1], [0, 0]], [[1, 0], [0, 1]]])
    data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    def test_mean(self):
        assert cltimg.compute_statistics_at_nonzero_voxels(self.mask, self.data) == pytest.approx(5.0)

    @pytest.mark.parametrize(
        "metric, expected",
        [
            ("mean", 5.0),
            ("std", np.std([2, 5, 8])),
            ("var", np.var([2, 5, 8])),
            ("median", 5.0),
            ("sum", 15.0),
            ("max", 8.0),
            ("min", 2.0),
        ],
    )
    def test_metrics(self, metric, expected):
        value = cltimg.compute_statistics_at_nonzero_voxels(self.mask, self.data, metric=metric)
        assert value == pytest.approx(expected)

    def test_4d_gives_one_value_per_volume(self):
        data4d = np.stack([self.data, 2 * self.data, 3 * self.data], axis=-1)
        ts = cltimg.compute_statistics_at_nonzero_voxels(self.mask, data4d)
        assert np.allclose(ts, [5.0, 10.0, 15.0])

    def test_unsupported_metric(self):
        with pytest.raises(ValueError):
            cltimg.compute_statistics_at_nonzero_voxels(self.mask, self.data, metric="mode")


class TestInterpolate:
    @pytest.fixture
    def ramp(self):
        return np.broadcast_to(np.arange(20, dtype=float)[:, None, None], (20, 20, 20)).copy()

    points = np.array([[2.0, 5.0, 5.0], [7.25, 3.0, 9.0], [12.6, 10.5, 1.2]])

    def test_linear_recovers_the_x_coordinate(self, ramp):
        assert np.allclose(cltimg.interpolate(ramp, self.points), [2.0, 7.25, 12.6])

    def test_nearest(self, ramp):
        assert np.allclose(cltimg.interpolate(ramp, self.points, interp_method="nearest"), [2, 7, 13])

    def test_4d_and_points_outside(self, ramp):
        ramp4d = np.stack([ramp, 2 * ramp, 3 * ramp], axis=-1)
        pts = np.vstack([self.points, [[25.0, 5.0, 5.0]]])
        out = cltimg.interpolate(ramp4d, pts)
        assert out.shape == (4, 3)
        assert np.allclose(out[0], [2, 4, 6])
        assert np.allclose(out[3], 0)

    def test_wrong_coordinates_shape(self, ramp):
        with pytest.raises(ValueError):
            cltimg.interpolate(ramp, np.array([1.0, 2.0, 3.0]))


class TestRegionGrowing:
    @pytest.fixture
    def data(self):
        truth = np.zeros((20, 20, 20), dtype=np.int32)
        truth[:, :, :10] = 1
        truth[:, :, 10:] = 2
        zz, yy, xx = np.ogrid[:20, :20, :20]
        mask = (zz - 10) ** 2 + (yy - 10) ** 2 + (xx - 10) ** 2 <= 8**2
        truth = truth * mask
        rng = np.random.default_rng(0)
        gaps = mask & (rng.random(mask.shape) < 0.1)
        holed = truth.copy()
        holed[gaps] = 0
        return truth, holed, mask, gaps

    def test_gaps_are_filled(self, data):
        truth, holed, mask, gaps = data
        filled = cltimg.region_growing(holed, mask)
        assert ((filled == 0) & mask).sum() < gaps.sum()
        recovered = filled[gaps & (filled > 0)] == truth[gaps & (filled > 0)]
        assert recovered.mean() > 0.9

    def test_nothing_outside_the_mask_and_input_untouched(self, data):
        truth, holed, mask, gaps = data
        before = holed.copy()
        filled = cltimg.region_growing(holed, mask)
        assert not ((filled > 0) & ~mask).any()
        assert np.array_equal(holed, before)

    def test_conservative_neighborhood_fills_less_or_equal(self, data):
        truth, holed, mask, gaps = data
        default = cltimg.region_growing(holed, mask)
        strict = cltimg.region_growing(holed, mask, neighborhood="6", min_neighbors=4)
        assert (strict == 0).sum() >= (default == 0).sum()
        assert np.array_equal(strict[holed > 0], holed[holed > 0])


######################################################################################################
class TestImagetoolsUtils:
    def test_get_sidecars_files(self, tmp_path):
        nifti = tmp_path / "sub-01_dwi.nii.gz"
        nifti.write_text("")
        (tmp_path / "sub-01_dwi.bval").write_text("0 1000")
        (tmp_path / "sub-01_dwi.json").write_text("{}")
        sidecars = imgutils.get_sidecars_files(str(nifti))
        assert sidecars["bval"] == tmp_path / "sub-01_dwi.bval"
        assert sidecars["json"] == tmp_path / "sub-01_dwi.json"
        assert sidecars["bvec"] is None

    def test_get_sidecars_files_plain_nii(self, tmp_path):
        nifti = tmp_path / "dwi.nii"
        nifti.write_text("")
        (tmp_path / "dwi.bvec").write_text("0 0 0")
        sidecars = imgutils.get_sidecars_files(nifti)
        assert sidecars["bvec"] == tmp_path / "dwi.bvec"
        assert sidecars["bval"] is None and sidecars["json"] is None

    def test_merge_json_files_first_value_wins(self, tmp_path):
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        a.write_text(json.dumps({"RepetitionTime": 3.2, "PhaseEncodingDirection": "j-"}))
        b.write_text(json.dumps({"RepetitionTime": 3.2, "PhaseEncodingDirection": "j", "Extra": 1}))
        merged = imgutils.merge_json_files([a, str(b)])
        assert merged["RepetitionTime"] == 3.2
        assert merged["PhaseEncodingDirection"] == "j-"
        assert merged["Extra"] is None  # absent in the first file

    def test_merge_json_files_union_of_keys(self, tmp_path):
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        a.write_text(json.dumps({"A": 1}))
        b.write_text(json.dumps({"B": 2}))
        assert set(imgutils.merge_json_files([a, b])) == {"A", "B"}
