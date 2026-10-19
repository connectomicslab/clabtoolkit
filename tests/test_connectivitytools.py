"""Tests for clabtoolkit.connectivitytools."""

import warnings

import matplotlib

matplotlib.use("Agg")

import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import pyvista as pv  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse import random as sparse_random  # noqa: E402

import clabtoolkit.colorstools as cltcol  # noqa: E402
from clabtoolkit.connectivitytools import Connectome  # noqa: E402

# 4 x 4 matrix with known values. Off-diagonal values sorted by absolute value:
# 0.9 (x2), 0.8 (x2), 0.7 (x2, negative), 0.5 (x2), 0.2 (x2), 0.1 (x2)
KNOWN = np.array(
    [
        [0.0, 0.9, 0.1, 0.5],
        [0.9, 0.0, -0.7, 0.2],
        [0.1, -0.7, 0.0, 0.8],
        [0.5, 0.2, 0.8, 0.0],
    ]
)
WHOLE_BRAIN = "brain-brain-wholebrain"


####################################################################################################
# Fixtures
####################################################################################################
@pytest.fixture(autouse=True)
def _quiet():
    """Close the figures of every test and silence warnings and progress messages."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
    plt.close("all")


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


@pytest.fixture
def matrix():
    """Symmetric 6 x 6 matrix with a zero diagonal."""
    values = np.random.default_rng(0).random((6, 6))
    values = (values + values.T) / 2
    np.fill_diagonal(values, 0)
    return values


@pytest.fixture
def known():
    """Connectome holding the KNOWN matrix and complete, easily recognisable metadata."""
    return Connectome(
        KNOWN.copy(),
        name="known",
        region_names=["a", "b", "c", "d"],
        region_coords=np.arange(12, dtype=float).reshape(4, 3),
        region_colors=["#ff0000", "#00ff00", "#0000ff", "#ffff00"],
        region_index=[10, 20, 30, 40],
        modality="functional",
        affine=np.diag([2.0, 2.0, 2.0, 1.0]),
    )


@pytest.fixture
def synthetic():
    """Synthetic connectome with 8 regions (hex colors, spherical coordinates)."""
    return Connectome.generate_connectome(8, method="modular", seed=18)


@pytest.fixture
def h5_file(tmp_path, known):
    path = tmp_path / "known.h5"
    known.save_h5(path)
    return path


def hex_colors(colors):
    return cltcol.harmonize_colors(colors, "hex")


####################################################################################################
# Creation
####################################################################################################
class TestCreation:
    def test_empty(self):
        conn = Connectome(name="empty")
        assert conn.matrix is None and conn.n_regions == 0 and conn.name == "empty"
        assert conn.weighting is None and conn.modality == "unknown"
        assert conn.region_names is None and conn.region_colors is None
        assert conn.region_index is None and conn.region_coords is None

    def test_from_matrix_defaults(self, matrix):
        conn = Connectome(matrix, name="demo")
        assert conn.n_regions == 6 and conn.name == "demo"
        assert conn.matrix.dtype == np.float64 and np.array_equal(conn.matrix, matrix)
        assert conn.region_names == [f"auto-roi-{i:06d}" for i in range(1, 7)]
        assert conn.region_index == [0, 1, 2, 3, 4, 5]
        assert len(conn.region_colors) == 6 and conn.region_coords is None
        assert conn.modality == "unknown"
        assert np.array_equal(conn.affine, np.eye(4))

    def test_matrix_is_copied(self, matrix):
        conn = Connectome(matrix)
        conn.matrix[0, 1] = 99
        assert matrix[0, 1] != 99
        matrix[2, 3] = -5
        assert conn.matrix[2, 3] != -5

    def test_integer_matrix_becomes_float(self):
        conn = Connectome(np.array([[0, 1], [1, 0]]))
        assert conn.matrix.dtype == np.float64

    def test_custom_metadata(self, known):
        assert known.name == "known" and known.modality == "functional"
        assert known.region_names == ["a", "b", "c", "d"]
        assert known.region_colors == ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]
        assert known.region_index == [10, 20, 30, 40]
        assert known.region_coords.shape == (4, 3)
        assert known.affine[0, 0] == 2.0

    @pytest.mark.parametrize(
        "index", [[3, 4], (3, 4), np.array([3, 4]), np.array([3.0, 4.0])]
    )
    def test_region_index_is_a_list_of_ints(self, index):
        conn = Connectome(np.zeros((2, 2)), region_index=index)
        assert conn.region_index == [3, 4]
        assert all(type(i) is int for i in conn.region_index)

    def test_sparse_matrix_is_densified(self, matrix):
        conn = Connectome(csr_matrix(matrix))
        assert isinstance(conn.matrix, np.ndarray)
        assert np.allclose(conn.matrix, matrix)

    @pytest.mark.parametrize("as_str", [True, False])
    def test_from_h5_path(self, h5_file, as_str):
        conn = Connectome(str(h5_file) if as_str else h5_file)
        assert conn.name == "known" and conn.n_regions == 4  # Name from the file stem
        assert np.allclose(conn.matrix, KNOWN)
        assert Connectome(h5_file, name="mine").name == "mine"

    def test_errors(self, matrix):
        with pytest.raises(ValueError, match="square"):
            Connectome(np.zeros((3, 4)))
        with pytest.raises(ValueError, match="square"):
            Connectome(np.zeros(3))
        with pytest.raises(TypeError, match="must be np.ndarray"):
            Connectome([[0, 1], [1, 0]])
        with pytest.raises(ValueError, match="Affine"):
            Connectome(matrix, affine=np.eye(3))
        with pytest.raises(ValueError, match="Number of names"):
            Connectome(matrix, region_names=["a"])
        with pytest.raises(ValueError, match="Coordinates shape"):
            Connectome(matrix, region_coords=np.zeros((5, 3)))
        with pytest.raises(ValueError, match="Colors length"):
            Connectome(matrix, region_colors=["#ff0000"])
        with pytest.raises(ValueError, match="Region index length"):
            Connectome(matrix, region_index=[1, 2])

    def test_repr(self, matrix):
        assert repr(Connectome(name="e")) == "Connectome(name='e', no data loaded)"
        text = repr(Connectome(matrix, name="demo", modality="structural"))
        assert text == (
            "Connectome(name='demo', modality='structural', weighting='weighted', "
            "n_regions=6, density=1.000)"
        )


class TestWeighting:
    @pytest.mark.parametrize(
        "values, expected",
        [
            ([[0, 1], [1, 0]], "binary"),
            ([[0, 3], [3, 0]], "binary"),  # A single non-zero value is binary
            ([[0, 3], [1, 0]], "weighted"),
            ([[0, 0.3], [2.1, 0]], "weighted"),
            ([[0, 0], [0, 0]], "binary"),  # An empty network has no weights
        ],
    )
    def test_detection(self, values, expected):
        assert Connectome(np.array(values, dtype=float)).weighting == expected

    def test_is_none_without_matrix(self):
        assert Connectome().weighting is None
        assert Connectome._detect_weighting(None) is None

    def test_is_read_only(self, matrix):
        with pytest.raises(AttributeError):
            Connectome(matrix).weighting = "binary"

    def test_follows_the_matrix(self, matrix):
        conn = Connectome(matrix)
        assert conn.weighting == "weighted"
        conn.binarize()
        assert conn.weighting == "binary"


####################################################################################################
# CSV and sparse matrices
####################################################################################################
class TestCsv:
    @pytest.fixture
    def csv_file(self, tmp_path, matrix):
        path = tmp_path / "matrix.csv"
        np.savetxt(path, matrix, delimiter=",")
        return path

    def test_from_csv(self, csv_file, matrix):
        conn = Connectome.from_csv(csv_file, modality="structural")
        assert np.allclose(conn.matrix, matrix) and conn.n_regions == 6
        assert conn.name == "matrix" and conn.modality == "structural"
        assert conn.region_index == list(range(6))
        assert conn.region_names[0] == "auto-roi-000001"

    def test_from_csv_with_metadata(self, csv_file):
        conn = Connectome.from_csv(
            str(csv_file),
            name="mine",
            region_names=[f"r{i}" for i in range(6)],
            region_index=range(100, 106),
            region_coords=np.zeros((6, 3)),
            region_colors=["#ff0000"] * 6,
            affine=np.eye(4) * 2,
        )
        assert conn.name == "mine" and conn.region_names[:2] == ["r0", "r1"]
        assert conn.region_index == [100, 101, 102, 103, 104, 105]
        assert conn.region_coords.shape == (6, 3) and conn.affine[0, 0] == 2
        assert conn.region_colors == ["#ff0000"] * 6

    def test_load_csv_into_an_existing_object(self, csv_file, matrix):
        conn = Connectome()
        conn.load_csv(csv_file, name="loaded", modality="functional")
        assert conn.name == "loaded" and conn.modality == "functional"
        assert np.allclose(conn.matrix, matrix)

    def test_load_csv_without_modality_falls_back_to_unknown(self, csv_file):
        conn = Connectome()
        conn.load_csv(csv_file, modality=None)
        assert conn.modality == "unknown"

    def test_errors(self, tmp_path, csv_file):
        with pytest.raises(FileNotFoundError):
            Connectome.from_csv(tmp_path / "missing.csv")
        np.savetxt(tmp_path / "rect.csv", np.zeros((3, 4)), delimiter=",")
        with pytest.raises(ValueError, match="square"):
            Connectome.from_csv(tmp_path / "rect.csv")
        with pytest.raises(ValueError, match="Number of names"):
            Connectome.from_csv(csv_file, region_names=["a"])
        with pytest.raises(ValueError, match="Affine"):
            Connectome.from_csv(csv_file, affine=np.eye(3))


class TestCsr:
    @pytest.fixture
    def sparse(self):
        csr = sparse_random(30, 30, density=0.05, format="csr", random_state=0)
        return csr + csr.T

    def test_from_csr(self, sparse):
        conn = Connectome.from_csr(sparse, name="from_sparse")
        assert conn.name == "from_sparse" and isinstance(conn.matrix, np.ndarray)
        assert np.allclose(conn.matrix, sparse.toarray())

    def test_same_as_the_constructor(self, sparse):
        assert np.array_equal(
            Connectome.from_csr(sparse).matrix, Connectome(sparse).matrix
        )

    def test_metadata_is_passed_through(self, sparse):
        names = [f"ROI_{i:02d}" for i in range(30)]
        conn = Connectome.from_csr(
            sparse, region_names=names, modality="structural", region_index=range(30)
        )
        assert conn.region_names == names and conn.modality == "structural"
        assert conn.region_index == list(range(30))

    def test_other_sparse_formats(self, sparse):
        assert Connectome.from_csr(sparse.tocoo()).n_regions == 30

    def test_errors(self):
        with pytest.raises(TypeError, match="sparse"):
            Connectome.from_csr(np.eye(3))
        with pytest.raises(ValueError, match="square"):
            Connectome.from_csr(csr_matrix(np.zeros((2, 3))))

    def test_to_csr(self):
        conn = Connectome.generate_connectome(
            20, method="modular", sparsity=0.7, seed=31
        )
        csr = conn.to_csr()
        assert isinstance(csr, csr_matrix) and csr.shape == (20, 20)
        assert csr.nnz == np.count_nonzero(conn.matrix)
        assert np.allclose(csr.toarray(), conn.matrix)
        assert isinstance(conn.matrix, np.ndarray)  # The connectome is unchanged

    def test_to_csr_drop_diagonal(self):
        conn = Connectome(np.ones((4, 4)))
        assert conn.to_csr().diagonal().sum() == 4
        assert conn.to_csr(drop_diagonal=True).diagonal().sum() == 0
        assert conn.to_csr(drop_diagonal=True).nnz == 12
        assert np.diag(conn.matrix).sum() == 4  # The connectome keeps its diagonal

    def test_to_csr_has_no_explicit_zeros(self):
        csr = Connectome(KNOWN).threshold("value", 0.5).to_csr()
        assert csr.nnz == np.count_nonzero(csr.toarray())

    def test_to_csr_without_matrix(self):
        with pytest.raises(ValueError, match="No connectivity matrix"):
            Connectome().to_csr()

    def test_to_csr_feeds_networktools(self):
        import clabtoolkit.networktools as cltnet

        matrix = np.zeros((12, 12))
        matrix[:5, :5] = 1.0
        matrix[5:10, 5:10] = 1.0
        np.fill_diagonal(matrix, 0)
        n_components, labels, sizes = cltnet.connected_components(
            Connectome(matrix).to_csr(), verbose=False
        )
        assert n_components == 4  # Two modules of 5 regions and 2 isolated regions


####################################################################################################
# HDF5
####################################################################################################
class TestHdf5:
    def test_roundtrip_keeps_everything(self, known, h5_file):
        loaded = Connectome.from_h5(h5_file)
        assert loaded.name == "known"  # Taken from the file stem
        assert np.allclose(loaded.matrix, known.matrix)
        assert loaded.n_regions == 4 and loaded.modality == "functional"
        assert loaded.region_names == known.region_names
        assert loaded.region_colors == known.region_colors
        assert loaded.region_index == known.region_index
        assert np.allclose(loaded.region_coords, known.region_coords)
        assert np.allclose(loaded.affine, known.affine)
        assert loaded.weighting == known.weighting

    def test_roundtrip_of_default_colors(self, tmp_path, matrix):
        conn = Connectome(matrix)  # Default colors: integer RGB values
        conn.save_h5(tmp_path / "default.h5")
        loaded = Connectome.from_h5(tmp_path / "default.h5")
        assert loaded.region_colors == hex_colors(conn.region_colors)
        assert loaded.region_names == conn.region_names

    def test_from_h5_name(self, h5_file):
        assert Connectome.from_h5(h5_file, name="mine").name == "mine"

    def test_load_h5_into_an_existing_object(self, known, h5_file):
        conn = Connectome()
        conn.load_h5(str(h5_file))
        assert np.allclose(conn.matrix, known.matrix)
        assert conn.region_names == known.region_names

    def test_save_writes_the_metadata_attributes(self, h5_file):
        with h5py.File(h5_file, "r") as f:
            group = f["connmat"]
            assert group.attrs["modality"] == "functional"
            assert group.attrs["weighting"] == "weighted"
            assert group.attrs["n_regions"] == 4
            assert group.attrs["density"] == pytest.approx(1.0)
            assert {"matrix", "region_names", "region_colors", "region_index"} <= set(
                group
            )
            assert {"region_coords", "affine"} <= set(group)

    def test_compression_option(self, tmp_path, known):
        known.save_h5(tmp_path / "compressed.h5", compression=True)
        known.save_h5(tmp_path / "raw.h5", compression=False)
        with h5py.File(tmp_path / "compressed.h5") as f:
            assert f["connmat/matrix"].compression == "gzip"
        with h5py.File(tmp_path / "raw.h5") as f:
            assert f["connmat/matrix"].compression is None
        assert np.allclose(Connectome.from_h5(tmp_path / "raw.h5").matrix, KNOWN)

    @pytest.fixture
    def very_sparse(self):
        csr = sparse_random(300, 300, density=0.02, format="csr", random_state=1)
        return Connectome(csr + csr.T, name="very_sparse")

    @staticmethod
    def stored_format(path):
        with h5py.File(path, "r") as f:
            group = f["connmat"]
            keys = [k for k in group if k.startswith("matrix")]
            return group.attrs["matrix_format"], keys

    def test_auto_keeps_a_dense_matrix_dense(self, tmp_path, known):
        known.save_h5(tmp_path / "dense.h5")
        assert self.stored_format(tmp_path / "dense.h5") == ("dense", ["matrix"])

    def test_auto_switches_to_csr_for_a_very_sparse_matrix(self, tmp_path, very_sparse):
        very_sparse.save_h5(tmp_path / "auto.h5")
        assert self.stored_format(tmp_path / "auto.h5") == ("csr", ["matrix_csr"])

    def test_forcing_the_layout(self, tmp_path, very_sparse, known):
        very_sparse.save_h5(tmp_path / "dense.h5", sparse=False)
        assert self.stored_format(tmp_path / "dense.h5") == ("dense", ["matrix"])
        known.save_h5(tmp_path / "csr.h5", sparse=True)
        assert self.stored_format(tmp_path / "csr.h5") == ("csr", ["matrix_csr"])

    def test_density_threshold_for_auto(self, tmp_path, known):
        known.save_h5(
            tmp_path / "high.h5", sparse="auto", sparse_density_threshold=1.01
        )
        assert self.stored_format(tmp_path / "high.h5")[0] == "csr"
        known.save_h5(tmp_path / "zero.h5", sparse_density_threshold=0.0)
        assert self.stored_format(tmp_path / "zero.h5")[0] == "dense"

    def test_auto_uses_csr_only_below_the_density_threshold(self, tmp_path):
        matrix = np.zeros((4, 4))
        matrix[0, 1] = matrix[1, 0] = matrix[2, 3] = matrix[3, 2] = (
            1.0  # 4 / 16 nonzero
        )
        conn = Connectome(matrix)
        conn.save_h5(tmp_path / "equal.h5", sparse_density_threshold=0.25)
        assert (
            self.stored_format(tmp_path / "equal.h5")[0] == "dense"
        )  # Not strictly below
        conn.save_h5(tmp_path / "above.h5", sparse_density_threshold=0.26)
        assert self.stored_format(tmp_path / "above.h5")[0] == "csr"

    @pytest.mark.parametrize("sparse", [True, False, "auto"])
    def test_both_layouts_load_identically(self, tmp_path, very_sparse, sparse):
        very_sparse.save_h5(tmp_path / "m.h5", sparse=sparse)
        loaded = Connectome.from_h5(tmp_path / "m.h5")
        assert np.allclose(loaded.matrix, very_sparse.matrix)

    def test_csr_layout_is_smaller_for_a_sparse_matrix(self, tmp_path, very_sparse):
        very_sparse.save_h5(tmp_path / "csr.h5", sparse=True)
        very_sparse.save_h5(tmp_path / "dense.h5", sparse=False)
        assert (tmp_path / "csr.h5").stat().st_size < (
            tmp_path / "dense.h5"
        ).stat().st_size

    def test_missing_coordinates_warn(self, tmp_path, matrix):
        Connectome(matrix).save_h5(tmp_path / "nocoords.h5")
        with pytest.warns(UserWarning, match="No coordinates found"):
            loaded = Connectome.from_h5(tmp_path / "nocoords.h5")
        assert loaded.region_coords is None

    def test_legacy_keys(self, tmp_path, matrix):
        """Files written by older versions use other dataset names and numeric colors."""
        rng = np.random.default_rng(0)
        path = tmp_path / "legacy.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("matrix", data=matrix)
            f.create_dataset("gmcoords", data=rng.random((6, 3)))
            f.create_dataset("colors", data=rng.integers(0, 255, (6, 3)))
            f.create_dataset(
                "name", data=np.array([f"r{i}".encode() for i in range(6)])
            )
            f.create_dataset("gmindex", data=np.arange(6) + 100)
        loaded = Connectome.from_h5(path)
        assert loaded.region_names == [f"r{i}" for i in range(6)]
        assert loaded.region_index == [100, 101, 102, 103, 104, 105]
        assert loaded.region_coords.shape == (6, 3)
        assert loaded.region_colors.max() <= 1.0  # 0-255 values are rescaled
        assert loaded.modality == "unknown" and np.array_equal(loaded.affine, np.eye(4))

    def test_load_errors(self, tmp_path, matrix):
        with pytest.raises(FileNotFoundError):
            Connectome.from_h5(tmp_path / "missing.h5")
        (tmp_path / "corrupt.h5").write_text("not an hdf5 file")
        with pytest.raises(RuntimeError, match="Error loading HDF5 file"):
            Connectome.from_h5(tmp_path / "corrupt.h5")
        with h5py.File(tmp_path / "nomatrix.h5", "w") as f:
            f.create_dataset("other", data=[1])
        with pytest.raises(RuntimeError, match="No 'matrix' dataset"):
            Connectome.from_h5(tmp_path / "nomatrix.h5")
        with h5py.File(tmp_path / "badcoords.h5", "w") as f:
            f.create_dataset("matrix", data=matrix)
            f.create_dataset("coords", data=np.zeros((3, 3)))
        with pytest.raises(RuntimeError, match="coordinates"):
            Connectome.from_h5(tmp_path / "badcoords.h5")

    def test_incomplete_csr_group(self, tmp_path):
        with h5py.File(tmp_path / "bad.h5", "w") as f:
            group = f.create_group("connmat/matrix_csr")
            group.create_dataset("data", data=[1.0])
            group.create_dataset("indices", data=[0])
            group.attrs["shape"] = [2, 2]
        with pytest.raises(RuntimeError, match="indptr"):
            Connectome.from_h5(tmp_path / "bad.h5")

    def test_save_errors(self, tmp_path, known):
        with pytest.raises(ValueError, match="No connectivity matrix"):
            Connectome().save_h5(tmp_path / "empty.h5")
        with pytest.raises(ValueError, match="sparse must be"):
            known.save_h5(tmp_path / "x.h5", sparse="yes")


####################################################################################################
# Synthetic connectomes
####################################################################################################
class TestGenerateConnectome:
    @pytest.mark.parametrize("method", ["random", "modular", "distance"])
    def test_basic_properties(self, method):
        conn = Connectome.generate_connectome(20, method=method, seed=0)
        assert conn.matrix.shape == (20, 20) and conn.n_regions == 20
        assert np.allclose(conn.matrix, conn.matrix.T)
        assert (np.diag(conn.matrix) == 0).all()
        assert conn.matrix.min() >= 0 and conn.matrix.max() <= 1
        assert conn.name == f"synthetic_{method}_20"
        assert conn.region_coords.shape == (20, 3)
        assert len(conn.region_names) == len(conn.region_colors) == 20
        assert conn.region_index == list(range(20))

    @pytest.mark.parametrize("method", ["random", "modular", "distance"])
    def test_seed(self, method):
        a = Connectome.generate_connectome(15, method=method, seed=5)
        b = Connectome.generate_connectome(15, method=method, seed=5)
        c = Connectome.generate_connectome(15, method=method, seed=6)
        assert np.array_equal(a.matrix, b.matrix)
        assert np.array_equal(a.region_coords, b.region_coords)
        assert not np.array_equal(a.matrix, c.matrix)

    def test_directed(self):
        conn = Connectome.generate_connectome(20, symmetric=False, seed=0)
        assert not np.allclose(conn.matrix, conn.matrix.T)
        assert (np.diag(conn.matrix) == 0).all()

    def test_value_range(self):
        conn = Connectome.generate_connectome(30, value_range=(-1, 1), seed=0)
        assert conn.matrix.min() < -0.9 and conn.matrix.max() > 0.9
        assert conn.matrix.min() >= -1 and conn.matrix.max() <= 1

    @pytest.mark.parametrize("sparsity", [0.0, 0.25, 0.5, 0.8, 1.0])
    def test_sparsity(self, sparsity):
        conn = Connectome.generate_connectome(20, sparsity=sparsity, seed=0)
        assert conn.get_density() == pytest.approx(1 - sparsity, abs=1e-2)

    def test_sparsity_keeps_symmetry_unless_directed(self):
        sym = Connectome.generate_connectome(20, sparsity=0.5, seed=0)
        assert np.allclose(sym.matrix, sym.matrix.T)
        directed = Connectome.generate_connectome(
            20, sparsity=0.5, symmetric=False, seed=0
        )
        assert directed.get_density() == pytest.approx(0.5, abs=1e-2)

    def test_dense_by_default(self):
        assert Connectome.generate_connectome(20, seed=0).get_density() == 1.0

    def test_modular_structure(self):
        n, n_modules = 40, 4
        conn = Connectome.generate_connectome(
            n, method="modular", n_modules=n_modules, seed=1
        )
        modules = (np.arange(n) * n_modules) // n
        same = modules[:, None] == modules[None, :]
        off_diagonal = ~np.eye(n, dtype=bool)
        within, between = conn.matrix[same & off_diagonal], conn.matrix[~same]
        assert within.min() >= 0.6 and between.max() <= 0.3
        assert within.mean() > between.mean()

    def test_distance_structure(self):
        conn = Connectome.generate_connectome(30, method="distance", seed=2)
        coords = conn.region_coords
        distance = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
        upper = np.triu_indices(30, k=1)
        # Weights decay with the distance between regions
        assert np.corrcoef(distance[upper], conn.matrix[upper])[0, 1] < -0.5
        assert conn.matrix[upper].min() == pytest.approx(0.0)
        assert conn.matrix[upper].max() == pytest.approx(1.0)

    def test_distance_decay(self):
        coords = np.random.default_rng(0).random((20, 3)) * 100
        kwargs = {"method": "distance", "region_coords": coords}
        short = Connectome.generate_connectome(20, distance_decay=5, **kwargs)
        long = Connectome.generate_connectome(20, distance_decay=500, **kwargs)
        assert not np.allclose(short.matrix, long.matrix)

    @pytest.mark.parametrize(
        "alias, method", [("method1", "modular"), ("method2", "distance")]
    )
    def test_method_aliases(self, alias, method):
        a = Connectome.generate_connectome(20, method=alias, seed=3)
        b = Connectome.generate_connectome(20, method=method, seed=3)
        assert np.array_equal(a.matrix, b.matrix) and a.name == b.name

    def test_given_metadata(self):
        coords = np.random.default_rng(0).random((3, 3))
        conn = Connectome.generate_connectome(
            3,
            name="mine",
            modality="functional",
            region_names=["a", "b", "c"],
            region_colors=["#ff0000", "#00ff00", "#0000ff"],
            region_coords=coords,
            seed=0,
        )
        assert conn.name == "mine" and conn.modality == "functional"
        assert conn.region_names == ["a", "b", "c"]
        assert conn.region_colors == ["#ff0000", "#00ff00", "#0000ff"]
        assert np.allclose(conn.region_coords, coords)

    def test_single_region(self):
        conn = Connectome.generate_connectome(1, seed=0)
        assert conn.matrix.shape == (1, 1) and conn.matrix[0, 0] == 0

    def test_errors(self):
        generate = Connectome.generate_connectome
        with pytest.raises(ValueError, match="positive"):
            generate(0)
        with pytest.raises(ValueError, match="Unknown method"):
            generate(5, method="nope")
        with pytest.raises(ValueError, match="value_range"):
            generate(5, value_range=(1, 0))
        with pytest.raises(ValueError, match="sparsity"):
            generate(5, sparsity=1.5)
        with pytest.raises(ValueError, match="sparsity"):
            generate(5, sparsity=-0.1)
        with pytest.raises(ValueError, match="region_names"):
            generate(5, region_names=["a"])
        with pytest.raises(ValueError, match="region_colors"):
            generate(5, region_colors=["#ff0000"])
        with pytest.raises(ValueError, match="region_coords"):
            generate(5, region_coords=np.zeros((4, 3)))


####################################################################################################
# Region metadata
####################################################################################################
class TestRegionMetadata:
    def test_names(self, synthetic):
        assert synthetic.get_region_names() == synthetic.region_names
        synthetic.set_region_names([f"R{i}" for i in range(8)])
        assert synthetic.get_region_names()[:2] == ["R0", "R1"]

    def test_names_are_copied(self, synthetic):
        names = [f"R{i}" for i in range(8)]
        synthetic.set_region_names(names)
        names[0] = "changed"
        assert synthetic.get_region_names()[0] == "R0"

    def test_colors_are_converted_to_hex(self, synthetic):
        synthetic.set_region_colors(["#ff0000"] * 8)
        assert synthetic.get_region_colors() == ["#ff0000"] * 8
        synthetic.set_region_colors(np.array([[255, 0, 0]] * 8))  # 0-255 RGB
        assert synthetic.get_region_colors() == ["#ff0000"] * 8
        synthetic.set_region_colors(np.array([[0.0, 1.0, 0.0]] * 8))  # 0-1 RGB
        assert synthetic.get_region_colors() == ["#00ff00"] * 8

    def test_coordinates_are_copied(self, synthetic):
        coords = np.random.default_rng(7).random((8, 3)) * 100
        synthetic.set_region_coordinates(coords)
        coords[0, 0] = -1
        assert synthetic.get_region_coordinates()[0, 0] != -1
        assert np.allclose(synthetic.get_region_coordinates()[1:], coords[1:])

    def test_coordinates_default_to_none(self, matrix):
        assert Connectome(matrix).get_region_coordinates() is None

    def test_indices(self, synthetic):
        assert synthetic.get_region_indices() == list(range(8))
        synthetic.set_region_indices(np.arange(8) + 101)
        assert synthetic.get_region_indices() == list(range(101, 109))
        assert all(type(i) is int for i in synthetic.get_region_indices())
        synthetic.set_region_indices((8, 7, 6, 5, 4, 3, 2, 1))
        assert synthetic.get_region_indices() == [8, 7, 6, 5, 4, 3, 2, 1]

    def test_setters_validate_lengths(self, synthetic):
        with pytest.raises(ValueError, match="Number of names"):
            synthetic.set_region_names(["a"])
        with pytest.raises(ValueError, match="Colors length"):
            synthetic.set_region_colors(["#ff0000"])
        with pytest.raises(ValueError, match="Coordinates shape"):
            synthetic.set_region_coordinates(np.zeros((7, 3)))
        with pytest.raises(ValueError, match="Coordinates shape"):
            synthetic.set_region_coordinates(np.zeros((8, 2)))
        with pytest.raises(ValueError, match="Region index length"):
            synthetic.set_region_indices([1, 2])

    def test_setters_do_not_validate_without_a_matrix(self):
        conn = Connectome()
        conn.set_region_names(["a", "b"])
        conn.set_region_indices([1, 2])
        conn.set_region_coordinates(np.zeros((2, 3)))
        assert conn.region_names == ["a", "b"] and conn.region_index == [1, 2]

    def test_default_names_and_colors(self, synthetic):
        synthetic.set_region_names([f"R{i}" for i in range(8)])
        # Defaults are generated whatever is currently set
        assert synthetic.get_default_region_names() == [
            f"auto-roi-{i:06d}" for i in range(1, 9)
        ]
        colors = synthetic.get_default_region_colors()
        assert np.shape(colors) == (8, 3) and np.min(colors) >= 0

    def test_getters_fall_back_to_defaults(self, synthetic):
        synthetic.region_names = None
        synthetic.region_colors = None
        assert synthetic.get_region_names() == synthetic.get_default_region_names()
        assert np.shape(synthetic.get_region_colors()) == (8, 3)


class TestLoadColortable:
    CTAB = {
        "index": [1, 2, 3, 4],
        "name": ["Region_A", "Region_B", "Region_C", "Region_D"],
        "color": ["#ff0000", "#00ff00", "#0000ff", "#ffff00"],
    }

    @pytest.fixture
    def conn(self):
        return Connectome.generate_connectome(4, seed=10)

    def check(self, conn):
        assert conn.get_region_names() == self.CTAB["name"]
        assert conn.get_region_colors() == self.CTAB["color"]
        assert conn.get_region_indices() == self.CTAB["index"]

    def test_from_dict(self, conn):
        conn.load_colortable(self.CTAB)
        self.check(conn)

    def test_from_dict_without_index_keeps_the_index(self, conn):
        conn.load_colortable({k: v for k, v in self.CTAB.items() if k != "index"})
        assert conn.get_region_names() == self.CTAB["name"]
        assert conn.get_region_indices() == [0, 1, 2, 3]

    def test_from_loader(self, conn):
        conn.load_colortable(cltcol.ColorTableLoader(self.CTAB))
        self.check(conn)

    @pytest.mark.parametrize("as_str", [True, False])
    def test_from_lut_file(self, tmp_path, conn, as_str):
        lut = tmp_path / "regions.lut"
        lut.write_text(
            "# test table\n"
            "1 Region_A 255 0 0 0\n"
            "2 Region_B 0 255 0 0\n"
            "3 Region_C 0 0 255 0\n"
            "4 Region_D 255 255 0 0\n"
        )
        conn.load_colortable(str(lut) if as_str else lut)
        self.check(conn)

    def test_errors(self, tmp_path, conn):
        with pytest.raises(FileNotFoundError):
            conn.load_colortable(tmp_path / "missing.lut")
        with pytest.raises(ValueError, match="Colors length"):
            Connectome.generate_connectome(3, seed=1).load_colortable(self.CTAB)


####################################################################################################
# Analysis
####################################################################################################
class TestAnalysis:
    CHAIN = np.array([[0, 1, 0], [1, 0, 2], [0, 2, 0]], dtype=float)

    def test_density(self):
        assert Connectome(self.CHAIN).get_density() == pytest.approx(4 / 6)
        assert Connectome(np.ones((3, 3))).get_density() == 1.0
        assert Connectome(np.zeros((3, 3))).get_density() == 0.0

    def test_density_ignores_the_diagonal(self):
        assert Connectome(np.eye(3)).get_density() == 0.0

    def test_density_of_degenerate_connectomes(self):
        assert Connectome().get_density() == 0.0
        assert Connectome(np.zeros((1, 1))).get_density() == 0.0

    def test_density_of_a_directed_matrix(self):
        directed = np.array([[0, 1, 0], [0, 0, 0], [1, 1, 0]], dtype=float)
        assert Connectome(directed).get_density() == pytest.approx(3 / 6)

    def test_stats(self):
        stats = Connectome(self.CHAIN).get_connectivity_stats()
        assert stats["n_regions"] == 3 and stats["matrix_shape"] == (3, 3)
        assert stats["min_strength"] == 0 and stats["max_strength"] == 2
        assert stats["mean_strength"] == pytest.approx(6 / 9)
        assert stats["std_strength"] == pytest.approx(np.std(self.CHAIN))
        assert stats["density"] == pytest.approx(4 / 6)
        assert np.allclose(stats["node_strengths"], [1, 3, 2])
        assert "coord_ranges" not in stats

    def test_stats_node_strengths_use_absolute_values(self):
        assert np.allclose(
            Connectome(KNOWN).get_connectivity_stats()["node_strengths"],
            np.abs(KNOWN).sum(axis=1),
        )

    def test_stats_coordinate_ranges(self):
        conn = Connectome(
            self.CHAIN, region_coords=np.array([[0, 0, 0], [1, 2, 3], [-1, 5, 2.0]])
        )
        ranges = conn.get_connectivity_stats()["coord_ranges"]
        assert ranges == {"x": (-1.0, 1.0), "y": (0.0, 5.0), "z": (0.0, 3.0)}

    def test_stats_without_matrix(self):
        assert Connectome().get_connectivity_stats() == {}

    def test_set_diagonal_to_zero(self, matrix):
        conn = Connectome(matrix)
        conn.matrix[np.diag_indices(6)] = 99
        conn.set_diagonal_to_zero()
        assert (np.diag(conn.matrix) == 0).all()
        assert np.allclose(conn.matrix - np.diag(np.diag(conn.matrix)), matrix)

    def test_set_symmetric_averages_with_the_transpose(self):
        conn = Connectome(np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0.0]]))
        conn.set_symmetric()
        assert np.allclose(conn.matrix, [[0, 2, 3.5], [2, 0, 5], [3.5, 5, 0]])

    def test_binarize(self):
        conn = Connectome(KNOWN.copy())
        conn.binarize()
        assert set(np.unique(conn.matrix)) == {0.0, 1.0}
        assert np.array_equal(conn.matrix != 0, KNOWN != 0)

    @pytest.mark.parametrize(
        "method", ["binarize", "set_diagonal_to_zero", "set_symmetric", "threshold"]
    )
    def test_in_place_methods_need_a_matrix(self, method):
        with pytest.raises(ValueError, match="No connectivity matrix"):
            getattr(Connectome(), method)()

    def test_copy_is_independent(self, known):
        clone = known.copy()
        assert clone is not known and clone.name == "known"
        assert np.array_equal(clone.matrix, known.matrix)
        assert clone.region_names == known.region_names
        assert clone.region_index == known.region_index
        assert clone.modality == "functional" and clone.affine[0, 0] == 2
        clone.matrix[0, 1] = 5
        clone.region_names[0] = "changed"
        clone.region_index[0] = -1
        clone.region_coords[0, 0] = -9
        clone.affine[0, 0] = 7
        assert known.matrix[0, 1] == 0.9 and known.region_names[0] == "a"
        assert known.region_index[0] == 10 and known.region_coords[0, 0] == 0
        assert known.affine[0, 0] == 2

    def test_copy_of_an_empty_connectome(self):
        clone = Connectome(name="e").copy()
        assert clone.matrix is None and clone.name == "e"


class TestThreshold:
    def test_value_threshold_uses_absolute_values(self, known):
        result = known.threshold("value", 0.5)
        expected = KNOWN.copy()
        expected[np.abs(expected) < 0.5] = 0  # Keeps 0.9, 0.5, -0.7 and 0.8
        assert np.array_equal(result.matrix, expected)

    def test_value_threshold_with_raw_values(self, known):
        result = known.threshold("value", 0.5, absolute=False)
        expected = KNOWN.copy()
        expected[expected < 0.5] = 0  # The strong negative connection is removed
        assert np.array_equal(result.matrix, expected)

    def test_default_threshold_changes_nothing(self, known):
        assert np.array_equal(known.threshold().matrix, KNOWN)

    def test_binarize(self, known):
        result = known.threshold("value", 0.5, binarize=True)
        assert set(np.unique(result.matrix)) == {0.0, 1.0}
        assert np.array_equal(result.matrix != 0, np.abs(KNOWN) >= 0.5)
        assert result.weighting == "binary"

    def test_returns_a_copy_by_default(self, known):
        result = known.threshold("value", 0.5)
        assert result is not known and np.array_equal(known.matrix, KNOWN)

    def test_copy_keeps_the_metadata(self, known):
        result = known.threshold("value", 0.5)
        assert result.name == "known" and result.modality == "functional"
        assert result.region_names == known.region_names
        assert result.region_colors == known.region_colors
        assert result.region_index == known.region_index
        assert np.array_equal(result.region_coords, known.region_coords)
        assert np.array_equal(result.affine, known.affine)

    def test_in_place(self, known):
        assert known.threshold("value", 0.5, copy=False) is known
        assert known.matrix[0, 2] == 0 and known.matrix[0, 1] == 0.9

    def test_diagonal_is_always_zero(self):
        assert (
            np.diag(Connectome(np.ones((3, 3))).threshold("value", 0.5).matrix) == 0
        ).all()

    def test_sparsity_threshold_keeps_the_strongest_connections(self, known):
        result = known.threshold("sparsity", 0.5)  # Keeps 6 of the 12 connections
        expected = KNOWN.copy()
        expected[np.abs(expected) < 0.7] = 0
        assert np.array_equal(result.matrix, expected)
        assert result.get_density() == pytest.approx(0.5)

    def test_sparsity_threshold_with_raw_values(self, known):
        result = known.threshold("sparsity", 0.5, absolute=False)
        expected = KNOWN.copy()
        expected[expected < 0.5] = (
            0  # The ranking uses raw values, so -0.7 is the weakest
        )
        assert np.array_equal(result.matrix, expected)

    def test_sparsity_extremes(self, known):
        assert np.array_equal(known.threshold("sparsity", 1.0).matrix, KNOWN)
        assert not known.threshold("sparsity", 0.0).matrix.any()

    @pytest.mark.parametrize("target", [0.2, 0.5, 0.8])
    def test_sparsity_density_reaches_the_target(self, target):
        conn = Connectome.generate_connectome(10, seed=15)
        density = conn.threshold("sparsity", target).get_density()
        # Symmetric ties are kept together, so the density can exceed the target slightly
        assert target <= density <= target + 2 / 90 + 1e-9

    def test_errors(self, known):
        with pytest.raises(ValueError, match="Unknown method"):
            known.threshold("nope")
        with pytest.raises(ValueError, match="between 0 and 1"):
            known.threshold("sparsity", 1.5)
        with pytest.raises(ValueError, match="between 0 and 1"):
            known.threshold("sparsity", -0.1)


class TestSubnetwork:
    def test_selects_and_orders_regions(self, known):
        sub = known.get_subnetwork([3, 1])
        assert sub.n_regions == 2
        assert np.array_equal(sub.matrix, KNOWN[np.ix_([3, 1], [3, 1])])
        assert sub.region_names == ["d", "b"]
        assert sub.region_index == [40, 20]
        assert np.array_equal(sub.region_coords, known.region_coords[[3, 1]])
        assert sub.region_colors == ["#ffff00", "#00ff00"]
        assert sub.modality == "functional" and sub.affine[0, 0] == 2

    def test_is_a_copy_by_default(self, known):
        sub = known.get_subnetwork([0, 1])
        assert sub is not known and known.n_regions == 4
        sub.matrix[0, 1] = 5
        assert known.matrix[0, 1] == 0.9

    def test_name(self, known):
        assert known.get_subnetwork([0, 1]).name == "known_subnetwork"
        assert Connectome(KNOWN).get_subnetwork([0, 1]).name == "subnetwork"

    def test_in_place(self, known):
        assert known.get_subnetwork([0, 2], copy=False) is known
        assert known.n_regions == 2 and known.matrix.shape == (2, 2)
        assert known.region_names == ["a", "c"] and known.region_index == [10, 30]
        assert known.region_coords.shape == (2, 3)

    def test_array_of_indices_and_default_colors(self, matrix):
        conn = Connectome(matrix)
        sub = conn.get_subnetwork(np.array([1, 4, 5]))
        assert sub.n_regions == 3
        assert sub.region_names == [conn.region_names[i] for i in (1, 4, 5)]
        assert sub.region_colors == hex_colors(conn.region_colors[[1, 4, 5]])

    def test_without_coordinates(self, matrix):
        assert Connectome(matrix).get_subnetwork([0, 1]).region_coords is None

    def test_errors(self, known):
        with pytest.raises(IndexError):
            known.get_subnetwork([0, 9])
        with pytest.raises(ValueError, match="No connectivity matrix"):
            Connectome().get_subnetwork([0])


class TestGraphMetrics:
    @staticmethod
    def whole_brain(metrics):
        rows = metrics[metrics["Region"] == WHOLE_BRAIN]
        return dict(zip(rows["Metric"], rows["Value"], strict=True))

    def test_complete_graph(self):
        conn = Connectome(np.ones((5, 5)) - np.eye(5))
        values = self.whole_brain(conn.get_graph_metrics())
        for metric in (
            "glob_efficiency",
            "transitivity",
            "density_coeff",
            "char_path_length",
            "radius",
            "diameter",
        ):
            assert values[metric] == pytest.approx(1.0), metric

    def test_path_graph(self):
        path = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0.0]])
        values = self.whole_brain(Connectome(path).get_graph_metrics())
        assert values["glob_efficiency"] == pytest.approx(13 / 18)
        assert values["char_path_length"] == pytest.approx(5 / 3)
        assert values["radius"] == 2 and values["diameter"] == 3
        assert values["density_coeff"] == pytest.approx(0.5)
        assert values["transitivity"] == 0
        assert values["assortativity"] == pytest.approx(-0.5)

    def test_table_layout(self):
        conn = Connectome(np.ones((5, 5)) - np.eye(5), region_names=list("ABCDE"))
        metrics = conn.get_graph_metrics()
        assert isinstance(metrics, pd.DataFrame)
        assert list(metrics.columns) == [
            "Source",
            "Metric",
            "Units",
            "Supraregion",
            "Hemisphere",
            "Region",
            "Value",
        ]
        degrees = metrics[metrics["Metric"] == "degree"]
        assert list(degrees["Region"]) == list("ABCDE")
        assert (degrees["Value"] == 4).all()
        clustering = metrics[metrics["Metric"] == "clustering_coeff"]
        assert (clustering["Value"] == 1).all()

    def test_source_follows_the_weighting(self):
        binary = Connectome(np.ones((4, 4)) - np.eye(4)).get_graph_metrics()
        weighted = Connectome(
            Connectome.generate_connectome(8, seed=12).matrix
        ).get_graph_metrics()
        assert set(binary["Source"]) == {"conn_matrix_binary"}
        assert set(weighted["Source"]) == {"conn_matrix_weighted"}

    def test_binarized_connectome_uses_binary_metrics(self):
        conn = Connectome.generate_connectome(12, seed=34)
        conn.matrix[conn.matrix < 0.3] = 0
        conn.binarize()
        assert set(conn.get_graph_metrics()["Source"]) == {"conn_matrix_binary"}

    def test_table_is_saved(self, tmp_path):
        conn = Connectome.generate_connectome(8, seed=12)
        out = tmp_path / "metrics.csv"
        metrics = conn.get_graph_metrics(output_table=out)
        assert out.is_file() and len(pd.read_csv(out)) == len(metrics)


####################################################################################################
# Plots
####################################################################################################
class TestPlotMatrix:
    def test_heatmap(self, synthetic):
        synthetic.plot_matrix(figsize=(5, 4))
        fig = plt.gcf()
        ax = fig.axes[0]
        assert tuple(fig.get_size_inches()) == (5, 4)
        assert ax.get_title() == f"Connectivity Matrix - {synthetic.name}"
        assert ax.get_xlabel() == ax.get_ylabel() == "Brain Regions"
        assert np.allclose(ax.images[0].get_array(), synthetic.matrix)
        assert fig.axes[1].get_ylabel() == "Connection Strength"  # Colorbar

    def test_region_names_label_the_axes(self, synthetic):
        synthetic.plot_matrix()
        labels = [t.get_text() for t in plt.gca().get_xticklabels()]
        assert labels == synthetic.region_names

    def test_labels_can_be_hidden_and_are_skipped_for_large_connectomes(
        self, synthetic
    ):
        synthetic.plot_matrix(show_labels=False)
        assert synthetic.region_names[0] not in [
            t.get_text() for t in plt.gca().get_xticklabels()
        ]
        plt.close("all")
        large = Connectome.generate_connectome(60, seed=1)
        large.plot_matrix()
        assert large.region_names[0] not in [
            t.get_text() for t in plt.gca().get_xticklabels()
        ]

    def test_title_colormap_and_threshold(self, synthetic):
        synthetic.plot_matrix(title="Mine", cmap="viridis", threshold=0.5)
        ax = plt.gca()
        assert ax.get_title() == "Mine" and ax.images[0].get_cmap().name == "viridis"
        data = np.asarray(ax.images[0].get_array())
        assert ((data == 0) | (np.abs(data) >= 0.5)).all()
        # The connectome itself is not modified
        assert (
            synthetic.matrix[(synthetic.matrix > 0) & (synthetic.matrix < 0.5)]
        ).size

    def test_threshold_is_described_in_the_default_title(self, synthetic):
        synthetic.plot_matrix(threshold=0.3)
        assert plt.gca().get_title().endswith("(threshold: 0.3, mode: absolute)")
        plt.close("all")
        synthetic.plot_matrix(threshold=0.3, threshold_mode="raw")
        assert plt.gca().get_title().endswith("(threshold: 0.3, mode: raw)")

    def test_raw_threshold_keeps_negative_values_out(self):
        conn = Connectome(KNOWN, name="known")
        conn.plot_matrix(threshold=0.2, threshold_mode="raw")
        data = np.asarray(plt.gca().images[0].get_array())
        assert data.min() == 0  # The -0.7 connection is below the threshold
        plt.close("all")
        conn.plot_matrix(threshold=0.2, threshold_mode="absolute")
        assert np.asarray(plt.gca().images[0].get_array()).min() == -0.7

    def test_log_scale(self, synthetic):
        synthetic.plot_matrix(log_scale=True)
        data = np.asarray(plt.gca().images[0].get_array())
        assert np.allclose(data, np.log1p(synthetic.matrix))

    def test_log_scale_keeps_the_sign(self):
        Connectome(KNOWN).plot_matrix(log_scale=True)
        data = np.asarray(plt.gca().images[0].get_array())
        assert np.allclose(data, np.sign(KNOWN) * np.log1p(np.abs(KNOWN)))

    def test_without_matrix(self):
        with pytest.raises(ValueError, match="No connectivity matrix"):
            Connectome().plot_matrix()


class TestVisualize3d:
    @pytest.fixture
    def plotter(self, can_render):
        created = []

        def make(conn, **kwargs):
            kwargs.setdefault("off_screen", True)
            created.append(conn.visualize_3d(**kwargs))
            return created[-1]

        yield make
        for plotter in created:
            plotter.close()

    @staticmethod
    def actors(plotter, prefix):
        return {name for name in plotter.renderer.actors if name.startswith(prefix)}

    def test_one_sphere_per_region(self, plotter, synthetic):
        p = plotter(synthetic)
        assert isinstance(p, pv.Plotter)
        assert self.actors(p, "region_") == {f"region_{i}" for i in range(8)}

    def test_edges_above_the_threshold(self, plotter, synthetic):
        p = plotter(synthetic, connectivity_threshold=0.5)
        i, j = np.triu_indices(8, k=1)
        keep = np.abs(synthetic.matrix[i, j]) > 0.5
        expected = {f"edge_{a}_{b}" for a, b in zip(i[keep], j[keep], strict=True)}
        assert expected and self.actors(p, "edge_") == expected

    def test_edges_can_be_hidden_or_filtered_out(self, plotter, synthetic):
        assert not self.actors(plotter(synthetic, show_edges=False), "edge_")
        assert not self.actors(plotter(synthetic, connectivity_threshold=10), "edge_")

    def test_nodes_use_the_region_colors(self, plotter, known):
        coords = np.arange(12, dtype=float).reshape(4, 3)
        known.set_region_coordinates(coords)
        p = plotter(known, node_size_property="uniform")
        red = p.renderer.actors["region_0"].GetProperty().GetColor()
        assert np.allclose(red, (1, 0, 0))
        yellow = p.renderer.actors["region_3"].GetProperty().GetColor()
        assert np.allclose(yellow, (1, 1, 0))

    @pytest.mark.parametrize(
        "prop", ["strength", "degree", "uniform", "betweenness", "eigenvector"]
    )
    def test_node_size_properties(self, plotter, synthetic, prop):
        pytest.importorskip("networkx")
        p = plotter(synthetic, node_size_property=prop)
        assert len(self.actors(p, "region_")) == 8

    def test_default_colors_are_accepted(self, plotter, matrix):
        conn = Connectome(matrix, region_coords=np.random.default_rng(0).random((6, 3)))
        assert len(self.actors(plotter(conn), "region_")) == 6

    def test_errors(self, synthetic, matrix):
        with pytest.raises(ValueError, match="No connectivity matrix"):
            Connectome().visualize_3d()
        with pytest.raises(ValueError, match="No coordinates"):
            Connectome(matrix).visualize_3d()
        with pytest.raises(ValueError, match="Unknown node size property"):
            synthetic.visualize_3d(node_size_property="nope", off_screen=True)

    def test_save_visualization(self, tmp_path, can_render, synthetic):
        out = tmp_path / "connectome.png"
        synthetic.save_visualization(str(out), node_size_property="degree")
        assert out.read_bytes().startswith(b"\x89PNG")


class TestNodeSizes:
    def test_uniform(self, synthetic):
        sizes = synthetic._calculate_node_sizes("uniform", 0.1, 1.0, 0.5)
        assert np.array_equal(sizes, np.full(8, 0.5))

    def test_strength_scales_between_base_and_base_plus_ten_times_scale(self):
        conn = Connectome(KNOWN)
        sizes = conn._calculate_node_sizes("strength", 0.1, 2.0, 0.5)
        strengths = np.abs(KNOWN).sum(axis=1)
        assert np.allclose(sizes, strengths / strengths.max() * 20 + 0.5)

    def test_degree_counts_connections_above_the_threshold(self):
        conn = Connectome(KNOWN)
        # Connections equal to the threshold (the two 0.5 entries) are not counted
        sizes = conn._calculate_node_sizes("degree", 0.5, 1.0, 0.0)
        degrees = np.array(
            [1, 2, 2, 1]
        )  # Strictly above 0.5: 0.9, 0.7, 0.8 (twice each)
        assert np.allclose(sizes, degrees / degrees.max() * 10)

    def test_centralities(self, synthetic):
        pytest.importorskip("networkx")
        for prop in ("betweenness", "eigenvector"):
            sizes = synthetic._calculate_node_sizes(prop, 0.1, 1.0, 0.5)
            assert sizes.shape == (8,) and sizes.min() >= 0.5 and sizes.max() <= 10.5

    def test_unknown_property(self, synthetic):
        with pytest.raises(ValueError, match="Unknown node size property"):
            synthetic._calculate_node_sizes("nope", 0.1, 1.0, 0.5)


####################################################################################################
# Information
####################################################################################################
class TestGetInfo:
    def test_overview(self, known, capsys):
        known.get_info()
        out = capsys.readouterr().out
        lines = out.splitlines()
        assert "CONNECTOME OVERVIEW" in out
        assert len({len(line) for line in lines}) == 1  # The box is aligned
        for text in (
            "Name:      known",
            "Modality:  functional",
            "Weighting: weighted",
            "Regions:   4",
            "Shape:      4 × 4",
            "Range:      [-0.7000,  0.9000]",
            "Density:    1.0000",
            "Region names:   ✔  (4 entries)",
            "Region index:   ✔  (4 entries)",
        ):
            assert text in out
        assert "X:  [    0.00,     9.00]" in out

    def test_without_coordinates(self, matrix, capsys):
        Connectome(matrix).get_info()
        assert "Not available" in capsys.readouterr().out

    def test_nodes_without_coordinates(self, capsys):
        coords = np.array([[np.nan] * 3, [1, 2, 3.0], [4, 5, 6.0]])
        Connectome(np.ones((3, 3)), region_coords=coords).get_info()
        out = capsys.readouterr().out
        assert "X:  [    1.00,     4.00]" in out
        assert "Nodes without coordinates: 1" in out
        assert "Name:      N/A" in out

    def test_without_matrix(self, capsys):
        Connectome(name="empty").get_info()
        out = capsys.readouterr().out
        assert "Name:      empty" in out and "Weighting: n/a" in out
        assert "No connectivity matrix loaded." in out

    def test_missing_optional_data(self, known, capsys):
        known.region_names = None
        known.region_colors = None
        known.region_index = None
        known.get_info()
        out = capsys.readouterr().out
        assert out.count("✘") == 3
