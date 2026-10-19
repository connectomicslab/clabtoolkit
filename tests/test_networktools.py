"""Tests for clabtoolkit.networktools."""

import warnings

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

import clabtoolkit.networktools as cltnet

# Octahedron: 6 vertices, 12 edges and 8 triangular faces. Each vertex touches 4 others
# and the only pairs that are not connected are the opposite vertices (0-5, 1-3 and 2-4)
OCTAHEDRON = np.array(
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

# Graph with the components [2, 3, 4], [0, 1] and [5]
THREE_COMPONENTS = csr_matrix(
    (
        np.ones(8),
        (np.array([0, 1, 2, 2, 3, 3, 4, 4]), np.array([1, 0, 3, 4, 2, 4, 2, 3])),
    ),
    shape=(6, 6),
)

# Edges between non-contiguous vertices: [10, 11, 12], [50, 51] and [100, 101, 102, 103]
GLOBAL_EDGES = np.array(
    [[10, 11], [11, 12], [50, 51], [100, 101], [101, 102], [102, 103]]
)


####################################################################################################
# Helpers
####################################################################################################
@pytest.fixture(autouse=True)
def _quiet():
    """Silence the warnings of the functions (the tests of the warnings catch them)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def components_of(labels):
    """Set of frozensets with the vertices of each component, whatever their labels."""
    groups = {}
    for vertex, label in labels:
        groups.setdefault(int(label), set()).add(int(vertex))
    return {frozenset(group) for group in groups.values()}


def union_find_components(n_vertices, edges):
    """Reference implementation: set of frozensets with the vertices of each component."""
    parent = list(range(n_vertices))

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for a, b in edges:
        parent[find(int(a))] = find(int(b))

    groups = {}
    for vertex in range(n_vertices):
        groups.setdefault(find(vertex), set()).add(vertex)
    return {frozenset(group) for group in groups.values()}


####################################################################################################
# CSR graphs from different sources
####################################################################################################
class TestAdjacencyMatrixToCsr:
    def test_dense_matrix(self):
        adj = np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])
        graph = cltnet.adjacency_matrix_to_csr(adj)
        assert isinstance(graph, csr_matrix) and graph.shape == (4, 4)
        assert np.array_equal(graph.toarray(), adj)
        assert graph.nnz == 10  # Zeros are not stored

    def test_weights_are_kept(self):
        adj = np.array(
            [[0, 2.5, 1.0, 0], [2.5, 0, 0, 3.2], [1.0, 0, 0, 1.8], [0, 3.2, 1.8, 0]]
        )
        graph = cltnet.adjacency_matrix_to_csr(adj)
        assert graph.data.tolist() == [2.5, 1.0, 2.5, 3.2, 1.0, 1.8, 3.2, 1.8]
        assert np.array_equal(graph.toarray(), adj)

    @pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64, bool])
    def test_the_data_type_is_kept(self, dtype):
        adj = np.array([[0, 1], [1, 0]], dtype=dtype)
        assert cltnet.adjacency_matrix_to_csr(adj).dtype == dtype

    def test_directed_graphs_are_not_symmetrized(self):
        graph = cltnet.adjacency_matrix_to_csr(np.array([[0, 1], [0, 0]]))
        assert graph.toarray().tolist() == [[0, 1], [0, 0]]

    def test_a_graph_without_edges_and_an_empty_one(self):
        assert cltnet.adjacency_matrix_to_csr(np.zeros((3, 3))).nnz == 0
        assert cltnet.adjacency_matrix_to_csr(np.zeros((0, 0))).shape == (0, 0)

    @pytest.mark.parametrize(
        "bad, error, message",
        [
            ([[0, 1], [1, 0]], TypeError, "numpy array"),
            (csr_matrix(np.eye(2)), TypeError, "numpy array"),
            (np.zeros(3), ValueError, "2D"),
            (np.zeros((2, 2, 2)), ValueError, "2D"),
            (np.zeros((2, 3)), ValueError, "square"),
        ],
    )
    def test_errors(self, bad, error, message):
        with pytest.raises(error, match=message):
            cltnet.adjacency_matrix_to_csr(bad)


class TestTriangulatedMeshToCsr:
    def test_tetrahedron_is_a_complete_graph(self):
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        graph = cltnet.triangulated_mesh_to_csr(faces)
        assert graph.shape == (4, 4)
        assert graph.toarray().tolist() == (1 - np.eye(4, dtype=int)).tolist()

    def test_octahedron(self):
        graph = cltnet.triangulated_mesh_to_csr(OCTAHEDRON)
        dense = graph.toarray()
        assert graph.shape == (6, 6) and graph.nnz == 24  # 12 edges, both directions
        assert (dense == dense.T).all() and (np.diag(dense) == 0).all()
        assert dense.sum(axis=1).tolist() == [4] * 6
        for a, b in ((0, 5), (1, 3), (2, 4)):  # Opposite vertices
            assert dense[a, b] == 0
        # Euler's formula for a closed surface: V - E + F = 2
        assert 6 - graph.nnz // 2 + len(OCTAHEDRON) == 2

    def test_an_edge_shared_by_two_faces_counts_once(self):
        graph = cltnet.triangulated_mesh_to_csr(np.array([[0, 1, 2], [1, 2, 3]]))
        assert graph[1, 2] == 1 and graph[2, 1] == 1
        assert graph.data.max() == 1  # Binary graph
        assert np.asarray(graph.sum(axis=1)).ravel().tolist() == [2, 3, 3, 2]
        assert graph[0, 3] == 0  # The two outer vertices are not connected

    def test_the_vertex_count_is_inferred_from_the_faces(self):
        assert cltnet.triangulated_mesh_to_csr(np.array([[0, 1, 4]])).shape == (5, 5)

    def test_isolated_vertices_with_an_explicit_vertex_count(self):
        graph = cltnet.triangulated_mesh_to_csr(np.array([[0, 1, 2]]), n_vertices=5)
        assert graph.shape == (5, 5) and graph.nnz == 6
        assert np.asarray(graph.sum(axis=1)).ravel().tolist() == [2, 2, 2, 0, 0]

    def test_the_vertex_count_may_be_exactly_one_more_than_the_largest_index(self):
        assert cltnet.triangulated_mesh_to_csr(OCTAHEDRON, n_vertices=6).shape == (6, 6)

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint8, np.uint32])
    def test_integer_types(self, dtype):
        graph = cltnet.triangulated_mesh_to_csr(OCTAHEDRON.astype(dtype))
        assert graph.shape == (6, 6) and graph.nnz == 24

    def test_the_mesh_is_connected_unless_it_has_separate_parts(self):
        two_parts = np.vstack([OCTAHEDRON, OCTAHEDRON + 6])
        graph = cltnet.triangulated_mesh_to_csr(two_parts)
        n_components, _, sizes = cltnet.connected_components(graph, verbose=False)
        assert n_components == 2 and list(sizes.values()) == [6, 6]

    @pytest.mark.parametrize(
        "faces, kwargs, error, message",
        [
            ([[0, 1, 2]], {}, TypeError, "numpy array"),
            (np.array([0, 1, 2]), {}, ValueError, "2D"),
            (np.zeros((2, 4), dtype=int), {}, ValueError, "3 columns"),
            (np.array([[0.0, 1.0, 2.0]]), {}, TypeError, "integer"),
            (np.array([[0, -1, 2]]), {}, ValueError, "non-negative"),
            (np.array([[0, 1, 2]]), {"n_vertices": 2}, ValueError, "must be greater"),
            (np.array([[0, 1, 2]]), {"n_vertices": 1}, ValueError, "must be greater"),
        ],
    )
    def test_errors(self, faces, kwargs, error, message):
        with pytest.raises(error, match=message):
            cltnet.triangulated_mesh_to_csr(faces, **kwargs)


class TestEdgesToCsr:
    EDGES = np.array([[0, 1], [1, 2], [0, 2]])
    VALUES = np.array([2.5, 1.0, 3.2])

    def test_undirected_graph(self):
        graph = cltnet.edges_to_csr(self.EDGES, self.VALUES)
        assert isinstance(graph, csr_matrix) and graph.shape == (3, 3)
        assert graph.toarray().tolist() == [[0, 2.5, 3.2], [2.5, 0, 1.0], [3.2, 1.0, 0]]

    def test_directed_graph(self):
        graph = cltnet.edges_to_csr(
            np.array([[0, 1], [1, 2]]), np.array([0.8, 1.5]), symmetric=False
        )
        assert graph.toarray().tolist() == [[0, 0.8, 0], [0, 0, 1.5], [0, 0, 0]]

    def test_edges_without_values_give_a_binary_graph(self):
        graph = cltnet.edges_to_csr(np.array([[0, 1], [1, 2]]))
        assert graph.toarray().tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 0]]

    def test_duplicated_edges_are_summed(self):
        edges = np.array([[0, 1], [0, 1], [1, 0]])
        values = np.array([1.0, 2.0, 0.5])
        undirected = cltnet.edges_to_csr(edges, values)
        assert undirected.toarray().tolist() == [[0, 3.5], [3.5, 0]]
        directed = cltnet.edges_to_csr(edges, values, symmetric=False)
        assert directed.toarray().tolist() == [[0, 3.0], [0.5, 0]]

    def test_the_vertex_count(self):
        assert cltnet.edges_to_csr(np.array([[1, 3]])).shape == (4, 4)  # Inferred
        padded = cltnet.edges_to_csr(self.EDGES, self.VALUES, n_vertices=6)
        assert padded.shape == (6, 6) and padded.nnz == 6

    def test_the_data_type_of_the_values_is_kept(self):
        ints = cltnet.edges_to_csr(self.EDGES, np.array([1, 2, 3]))
        assert ints.dtype == np.int64
        assert ints.toarray().tolist() == [[0, 1, 3], [1, 0, 2], [3, 2, 0]]

    def test_negative_values_are_kept(self):
        graph = cltnet.edges_to_csr(np.array([[0, 1]]), np.array([-1.5]))
        assert graph.toarray().tolist() == [[0, -1.5], [-1.5, 0]]

    def test_symmetric_matrices_come_out_symmetric(self):
        graph = cltnet.edges_to_csr(self.EDGES, self.VALUES)
        assert (graph != graph.T).nnz == 0

    def test_an_empty_edge_list_warns_and_gives_an_empty_graph(self):
        empty = np.zeros((0, 2), dtype=int)
        with pytest.warns(UserWarning, match="Empty edge list"):
            assert cltnet.edges_to_csr(empty).shape == (0, 0)
        with pytest.warns(UserWarning, match="Empty edge list"):
            graph = cltnet.edges_to_csr(empty, n_vertices=4)
        assert graph.shape == (4, 4) and graph.nnz == 0

    @pytest.mark.parametrize(
        "edges, values, kwargs, error, message",
        [
            ([[0, 1]], None, {}, TypeError, "Edges must be a numpy array"),
            (EDGES, [1, 2, 3], {}, TypeError, "Edge values must be a numpy array"),
            (np.array([0, 1]), None, {}, ValueError, "Edges array must be 2D"),
            (np.zeros((2, 3), dtype=int), None, {}, ValueError, "exactly 2 columns"),
            (EDGES, np.ones((3, 1)), {}, ValueError, "1D array"),
            (EDGES, np.ones(2), {}, ValueError, "must match number of edge values"),
            (np.array([[0.0, 1.0]]), None, {}, TypeError, "integer"),
            (np.array([[True, False]]), None, {}, TypeError, "integer"),
            (EDGES, np.array(["a", "b", "c"]), {}, TypeError, "numeric"),
            (np.array([[0, -1]]), None, {}, ValueError, "non-negative"),
            (EDGES, None, {"n_vertices": 2}, ValueError, "must be greater"),
        ],
    )
    def test_errors(self, edges, values, kwargs, error, message):
        with pytest.raises(error, match=message):
            cltnet.edges_to_csr(edges, values, **kwargs)


####################################################################################################
# Connected components
####################################################################################################
class TestConnectedComponents:
    def test_components_labels_and_sizes(self):
        n_components, labels, sizes = cltnet.connected_components(
            THREE_COMPONENTS, verbose=False
        )
        assert n_components == 3
        assert sizes == {0: 3, 1: 2, 2: 1}
        assert labels.shape == (6, 2)
        assert labels[:, 0].tolist() == [0, 1, 2, 3, 4, 5]  # The vertex of each row
        assert labels[:, 1].tolist() == [1, 1, 0, 0, 0, 2]  # 0 is the largest component

    def test_returned_types(self):
        n_components, labels, sizes = cltnet.connected_components(
            THREE_COMPONENTS, verbose=False
        )
        assert type(n_components) is int
        assert all(
            type(label) is int and type(size) is int for label, size in sizes.items()
        )
        assert np.issubdtype(labels.dtype, np.integer)

    def test_components_are_labelled_by_decreasing_size(self):
        _, labels, sizes = cltnet.connected_components(THREE_COMPONENTS, verbose=False)
        assert list(sizes) == [0, 1, 2]
        assert list(sizes.values()) == sorted(sizes.values(), reverse=True)
        for label, size in sizes.items():
            assert (labels[:, 1] == label).sum() == size

    def test_the_summary_is_printed_by_default(self, capsys):
        cltnet.connected_components(THREE_COMPONENTS)
        assert capsys.readouterr().out == (
            "Components : 3\n"
            "  └─ Component   0 : 3 vertices\n"
            "  └─ Component   1 : 2 vertices\n"
            "  └─ Component   2 : 1 vertices\n"
        )

    def test_nothing_is_printed_when_not_verbose(self, capsys):
        cltnet.connected_components(THREE_COMPONENTS, verbose=False)
        assert capsys.readouterr().out == ""

    def test_one_component(self):
        ring = cltnet.edges_to_csr(np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))
        n_components, labels, sizes = cltnet.connected_components(ring, verbose=False)
        assert n_components == 1 and sizes == {0: 4} and (labels[:, 1] == 0).all()

    def test_isolated_vertices_are_components_of_one(self):
        n_components, _, sizes = cltnet.connected_components(
            csr_matrix((4, 4)), verbose=False
        )
        assert n_components == 4 and sizes == {0: 1, 1: 1, 2: 1, 3: 1}
        _, _, single = cltnet.connected_components(csr_matrix((1, 1)), verbose=False)
        assert single == {0: 1}

    def test_equal_sized_components_are_all_reported(self):
        graph = cltnet.edges_to_csr(np.array([[0, 1], [2, 3], [4, 5]]))
        n_components, labels, sizes = cltnet.connected_components(graph, verbose=False)
        assert n_components == 3 and sizes == {0: 2, 1: 2, 2: 2}
        assert components_of(labels) == {
            frozenset({0, 1}),
            frozenset({2, 3}),
            frozenset({4, 5}),
        }

    def test_edge_weights_are_ignored(self):
        light = cltnet.edges_to_csr(np.array([[0, 1], [1, 2]]), np.array([2.5, 3.0]))
        heavy = cltnet.edges_to_csr(np.array([[0, 1], [1, 2]]), np.array([1e9, 1e-9]))
        for graph in (light, heavy):
            assert cltnet.connected_components(graph, verbose=False)[0] == 1

    def test_negative_weights_still_connect(self):
        graph = cltnet.edges_to_csr(np.array([[0, 1], [2, 3]]), np.array([-1.0, 1.0]))
        _, labels, _ = cltnet.connected_components(graph, verbose=False)
        assert components_of(labels) == {frozenset({0, 1}), frozenset({2, 3})}

    def test_self_loops_are_ignored(self):
        loops = csr_matrix(np.eye(3))
        n_components, _, sizes = cltnet.connected_components(loops, verbose=False)
        assert n_components == 3 and sizes == {0: 1, 1: 1, 2: 1}
        with_loop = csr_matrix(np.array([[1.0, 0], [0, 0]]))
        assert cltnet.connected_components(with_loop, verbose=False)[0] == 2

    def test_directed_graphs_give_weakly_connected_components(self):
        directed = csr_matrix(
            (np.ones(2), (np.array([0, 1]), np.array([1, 2]))), shape=(4, 4)
        )  # 0 -> 1 -> 2 and an isolated vertex
        with pytest.warns(UserWarning, match="directed"):
            n_components, labels, sizes = cltnet.connected_components(
                directed, verbose=False
            )
        assert n_components == 2 and sizes == {0: 3, 1: 1}
        assert labels[:, 1].tolist() == [0, 0, 0, 1]

    def test_symmetric_graphs_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cltnet.connected_components(THREE_COMPONENTS, verbose=False)

    def test_the_symmetry_check_has_a_tolerance(self):
        almost = csr_matrix(np.array([[0, 1.0], [1.0 + 1e-12, 0]]))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cltnet.connected_components(almost, verbose=False)
        different = csr_matrix(np.array([[0, 1.0], [1.001, 0]]))
        with pytest.warns(UserWarning, match="directed"):
            cltnet.connected_components(different, verbose=False)

    def test_the_input_graph_is_not_modified(self):
        before = THREE_COMPONENTS.copy()
        cltnet.connected_components(THREE_COMPONENTS, verbose=False)
        assert (THREE_COMPONENTS != before).nnz == 0

    @pytest.mark.parametrize("seed", range(5))
    def test_random_graphs_against_a_union_find(self, seed):
        rng = np.random.default_rng(seed)
        n_vertices = 40
        edges = rng.integers(0, n_vertices, size=(30, 2))
        graph = cltnet.edges_to_csr(edges, n_vertices=n_vertices)
        n_components, labels, sizes = cltnet.connected_components(graph, verbose=False)
        expected = union_find_components(n_vertices, edges)
        assert components_of(labels) == expected
        assert n_components == len(expected) == len(sizes)
        assert sorted(sizes.values(), reverse=True) == sorted(
            map(len, expected), reverse=True
        )
        assert list(sizes.values()) == sorted(sizes.values(), reverse=True)

    @pytest.mark.parametrize(
        "graph, error, message",
        [
            (np.eye(3), TypeError, "csr_matrix"),
            (csc_matrix(np.eye(3)), TypeError, "csr_matrix"),
            (coo_matrix(np.eye(3)), TypeError, "csr_matrix"),
            (csr_matrix((2, 3)), ValueError, "square"),
            (csr_matrix((0, 0)), ValueError, "empty"),
        ],
    )
    def test_errors(self, graph, error, message):
        with pytest.raises(error, match=message):
            cltnet.connected_components(graph)


class TestEdgesToComponents:
    def test_non_contiguous_vertices(self):
        n_components, labels, sizes = cltnet.edges_to_components(
            GLOBAL_EDGES, verbose=False
        )
        assert n_components == 3 and sizes == {0: 4, 1: 3, 2: 2}
        assert labels.tolist() == [
            [10, 1],
            [11, 1],
            [12, 1],
            [50, 2],
            [51, 2],
            [100, 0],
            [101, 0],
            [102, 0],
            [103, 0],
        ]

    def test_rows_are_sorted_by_vertex_and_only_vertices_with_edges_appear(self):
        _, labels, _ = cltnet.edges_to_components(GLOBAL_EDGES, verbose=False)
        assert labels[:, 0].tolist() == sorted(np.unique(GLOBAL_EDGES).tolist())

    def test_vertices_of_the_largest_component(self):
        _, labels, _ = cltnet.edges_to_components(GLOBAL_EDGES, verbose=False)
        assert labels[labels[:, 1] == 0, 0].tolist() == [100, 101, 102, 103]

    def test_very_distant_vertex_numbers(self):
        edges = np.array([[1_000_000, 1_000_001], [7, 8], [8, 9]])
        n_components, labels, sizes = cltnet.edges_to_components(edges, verbose=False)
        assert n_components == 2 and sizes == {0: 3, 1: 2}
        assert components_of(labels) == {
            frozenset({7, 8, 9}),
            frozenset({1_000_000, 1_000_001}),
        }

    def test_the_summary_is_printed_by_default(self, capsys):
        cltnet.edges_to_components(GLOBAL_EDGES)
        assert capsys.readouterr().out == (
            "Components : 3\n"
            "  └─ Component   0 : 4 vertices\n"
            "  └─ Component   1 : 3 vertices\n"
            "  └─ Component   2 : 2 vertices\n"
        )
        cltnet.edges_to_components(GLOBAL_EDGES, verbose=False)
        assert capsys.readouterr().out == ""

    def test_repeated_reversed_and_looping_edges(self):
        edges = np.array([[0, 1], [1, 0], [0, 1], [1, 1]])
        n_components, labels, sizes = cltnet.edges_to_components(edges, verbose=False)
        assert n_components == 1 and sizes == {0: 2}
        assert labels.tolist() == [[0, 0], [1, 0]]

    def test_a_single_edge_and_a_single_self_loop(self):
        assert cltnet.edges_to_components(np.array([[5, 6]]), verbose=False)[
            1
        ].tolist() == [
            [5, 0],
            [6, 0],
        ]
        n_components, labels, sizes = cltnet.edges_to_components(
            np.array([[3, 3]]), verbose=False
        )
        assert n_components == 1 and labels.tolist() == [[3, 0]] and sizes == {0: 1}

    def test_same_result_as_connected_components_for_contiguous_vertices(self):
        edges = np.array([[0, 1], [1, 2], [3, 4], [5, 6], [6, 7], [7, 5]])
        from_edges = cltnet.edges_to_components(edges, verbose=False)
        from_graph = cltnet.connected_components(
            cltnet.edges_to_csr(edges), verbose=False
        )
        assert from_edges[0] == from_graph[0] and from_edges[2] == from_graph[2]
        assert np.array_equal(from_edges[1], from_graph[1])

    @pytest.mark.parametrize("seed", range(3))
    def test_random_edges_against_a_union_find(self, seed):
        rng = np.random.default_rng(seed)
        edges = (
            rng.integers(0, 10_000, size=(60, 2)) * 7
        )  # Sparse, non-contiguous numbers
        vertices = np.unique(edges)
        local = np.searchsorted(vertices, edges)
        expected = {
            frozenset(int(vertices[i]) for i in group)
            for group in union_find_components(len(vertices), local)
        }
        n_components, labels, sizes = cltnet.edges_to_components(edges, verbose=False)
        assert components_of(labels) == expected
        assert n_components == len(expected) == len(sizes)

    def test_lists_of_edges_are_accepted(self):
        n_components, labels, _ = cltnet.edges_to_components(
            [[0, 1], [1, 2]], verbose=False
        )
        assert n_components == 1 and labels[:, 0].tolist() == [0, 1, 2]

    def test_errors(self):
        with pytest.raises(ValueError, match="Edges array must be 2D"):
            cltnet.edges_to_components(np.array([0, 1]), verbose=False)
        with pytest.raises(ValueError, match="exactly 2 columns"):
            cltnet.edges_to_components(np.array([[0, 1, 2]]), verbose=False)

    def test_negative_vertex_numbers(self):
        edges = np.array([[-1, 0], [0, 1], [-5, -6]])
        n_components, labels, sizes = cltnet.edges_to_components(edges, verbose=False)
        assert n_components == 2 and sizes == {0: 3, 1: 2}
        assert labels.tolist() == [[-6, 1], [-5, 1], [-1, 0], [0, 0], [1, 0]]

    def test_shifting_all_the_vertex_numbers_does_not_change_the_components(self):
        edges = np.array([[0, 1], [1, 2], [5, 6], [8, 9], [9, 10], [10, 8]])
        _, base, sizes = cltnet.edges_to_components(edges, verbose=False)
        for shift in (-100, 1_000_000_000_000):
            _, moved, moved_sizes = cltnet.edges_to_components(
                edges + shift, verbose=False
            )
            assert moved_sizes == sizes
            assert np.array_equal(moved[:, 1], base[:, 1])
            assert np.array_equal(moved[:, 0], base[:, 0] + shift)

    def test_the_whole_range_of_64_bit_integers(self):
        top, bottom = np.iinfo(np.int64).max, np.iinfo(np.int64).min
        edges = np.array(
            [[top - 1, top], [bottom, bottom + 1], [bottom + 1, 0]], dtype=np.int64
        )
        n_components, labels, sizes = cltnet.edges_to_components(edges, verbose=False)
        assert n_components == 2 and sizes == {0: 3, 1: 2}
        assert labels[labels[:, 1] == 0, 0].tolist() == [bottom, bottom + 1, 0]
        assert labels[labels[:, 1] == 1, 0].tolist() == [top - 1, top]

    def test_vertex_numbers_beyond_the_signed_64_bit_range(self):
        edges = np.array([[2**63, 2**63 + 1]], dtype=np.uint64)
        with pytest.raises(ValueError, match="64-bit signed integer"):
            cltnet.edges_to_components(edges, verbose=False)
        within = np.array([[2**63 - 2, 2**63 - 1]], dtype=np.uint64)
        assert cltnet.edges_to_components(within, verbose=False)[1][:, 0].tolist() == [
            2**63 - 2,
            2**63 - 1,
        ]

    @pytest.mark.parametrize(
        "dtype", [np.int8, np.int32, np.int64, np.uint16, np.uint32]
    )
    def test_integer_types(self, dtype):
        edges = np.array([[1, 2], [2, 3], [10, 11]], dtype=dtype)
        n_components, labels, sizes = cltnet.edges_to_components(edges, verbose=False)
        assert n_components == 2 and sizes == {0: 3, 1: 2}
        assert labels[:, 0].tolist() == [1, 2, 3, 10, 11]

    def test_the_input_is_not_modified(self):
        edges = GLOBAL_EDGES.copy()
        cltnet.edges_to_components(edges, verbose=False)
        assert np.array_equal(edges, GLOBAL_EDGES)

    def test_floating_point_vertex_numbers_are_rejected(self):
        with pytest.raises(TypeError, match="integer vertex indices"):
            cltnet.edges_to_components(np.array([[0.0, 1.0]]), verbose=False)

    def test_no_edges(self):
        with pytest.raises(ValueError, match="Graph cannot be empty"):
            cltnet.edges_to_components(np.zeros((0, 2), dtype=int), verbose=False)
