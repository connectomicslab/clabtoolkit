import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple, Union, Optional
import warnings


####################################################################################################
####################################################################################################
############                                                                            ############
############                                                                            ############
############  Section 1: Methods dedicated create CSR graphs from different sources     ############
############  CSR (Compressed Sparse Row) format is efficient for graph representation  ############
############  and is widely used in scientific computing and machine learning.          ############
############                                                                            ############
############                                                                            ############
####################################################################################################
####################################################################################################
def adjacency_matrix_to_csr(adj_matrix: np.ndarray) -> csr_matrix:
    """
    Convert a dense adjacency matrix to CSR (Compressed Sparse Row) format.

    This method takes a square adjacency matrix where non-zero entries represent
    connections between vertices and converts it to an efficient sparse representation.

    Parameters
    ----------
    adj_matrix : np.ndarray
        A square 2D numpy array representing the adjacency matrix.
        Shape should be (n_vertices, n_vertices) where n_vertices is the number
        of vertices in the graph. Non-zero values represent edge weights.

    Returns
    -------
    csr_matrix
        A scipy sparse CSR matrix representing the same graph connectivity.

    Raises
    ------
    ValueError
        If the input matrix is not 2D or not square.
    TypeError
        If the input is not a numpy array.

    Examples
    --------
    >>> import numpy as np
    >>> # Create a simple 4-vertex graph
    >>> adj = np.array([[0, 1, 1, 0],
    ...                 [1, 0, 1, 1],
    ...                 [1, 1, 0, 1],
    ...                 [0, 1, 1, 0]])
    >>> csr_graph = adjacency_matrix_to_csr(adj)
    >>> print(csr_graph.toarray())
    [[0 1 1 0]
    [1 0 1 1]
    [1 1 0 1]
    [0 1 1 0]]

    >>> # With weighted edges
    >>> adj_weighted = np.array([[0, 2.5, 1.0, 0],
    ...                          [2.5, 0, 0, 3.2],
    ...                          [1.0, 0, 0, 1.8],
    ...                          [0, 3.2, 1.8, 0]])
    >>> csr_weighted = adjacency_matrix_to_csr(adj_weighted)
    >>> print(f"Non-zero values: {csr_weighted.data}")
    Non-zero values: [2.5 1.  2.5 3.2 1.  1.8 3.2 1.8]
    """
    if not isinstance(adj_matrix, np.ndarray):
        raise TypeError("Input must be a numpy array")

    if adj_matrix.ndim != 2:
        raise ValueError("Input must be a 2D array")

    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square")

    return csr_matrix(adj_matrix)


####################################################################################################
def triangulated_mesh_to_csr(
    faces: np.ndarray, n_vertices: Optional[int] = None
) -> csr_matrix:
    """
    Convert triangulated mesh faces to a CSR graph representation.

    This method constructs a graph where vertices are connected if they share
    an edge in any triangle face. All edge weights are set to 1. The resulting
    graph represents the 1-ring neighborhood connectivity of the mesh.

    Parameters
    ----------
    faces : np.ndarray
        A 2D numpy array of shape (n_faces, 3) where each row contains the
        indices of three vertices forming a triangle. Vertex indices should
        be non-negative integers.

    n_vertices : int, optional
        Total number of vertices in the mesh. If None, it will be inferred
        as the maximum vertex index + 1. Providing this parameter is recommended
        for meshes with isolated vertices.

    Returns
    -------
    csr_matrix
        A scipy sparse CSR matrix of shape (n_vertices, n_vertices) where
        entry (i,j) is 1 if vertices i and j are connected by an edge in the mesh,
        and 0 otherwise. The matrix is symmetric for undirected graphs.

    Raises
    ------
    ValueError
        If faces array is not 2D, doesn't have 3 columns, contains negative
        indices, or if n_vertices is less than the maximum vertex index.

    TypeError
        If faces is not a numpy array or contains non-integer values.

    Examples
    --------
    >>> import numpy as np
    >>> # Define a simple tetrahedron (4 faces, 4 vertices)
    >>> faces = np.array([[0, 1, 2],
    ...                   [0, 1, 3],
    ...                   [0, 2, 3],
    ...                   [1, 2, 3]])
    >>> csr_graph = triangulated_mesh_to_csr(faces)
    >>> print("Adjacency matrix:")
    >>> print(csr_graph.toarray())
    Adjacency matrix:
    [[0 1 1 1]
    [1 0 1 1]
    [1 1 0 1]
    [1 1 1 0]]

    >>> # Triangle mesh with explicit vertex count
    >>> faces_triangle = np.array([[0, 1, 2]])
    >>> csr_triangle = triangulated_mesh_to_csr(faces_triangle, n_vertices=5)
    >>> print(f"Shape: {csr_triangle.shape}")
    >>> print("Connections for triangle [0,1,2]:")
    >>> print(csr_triangle.toarray())
    Shape: (5, 5)
    Connections for triangle [0,1,2]:
    [[0 1 1 0 0]
    [1 0 1 0 0]
    [1 1 0 0 0]
    [0 0 0 0 0]
    [0 0 0 0 0]]
    """
    if not isinstance(faces, np.ndarray):
        raise TypeError("Faces must be a numpy array")

    if faces.ndim != 2:
        raise ValueError("Faces array must be 2D")

    if faces.shape[1] != 3:
        raise ValueError(
            "Faces array must have exactly 3 columns for triangulated mesh"
        )

    if not np.issubdtype(faces.dtype, np.integer):
        raise TypeError("Faces array must contain integer vertex indices")

    if np.any(faces < 0):
        raise ValueError("Vertex indices must be non-negative")

    max_vertex_idx = np.max(faces)

    if n_vertices is None:
        n_vertices = max_vertex_idx + 1
    elif n_vertices <= max_vertex_idx:
        raise ValueError(
            f"n_vertices ({n_vertices}) must be greater than maximum vertex index ({max_vertex_idx})"
        )

    # Extract all edges from triangular faces
    # Each triangle (v0, v1, v2) generates edges: (v0,v1), (v1,v2), (v0,v2)
    edges = []
    for face in faces:
        v0, v1, v2 = face
        edges.extend([(v0, v1), (v1, v2), (v0, v2)])

    edges = np.array(edges)

    # Create symmetric edges (undirected graph)
    edges_symmetric = np.vstack([edges, edges[:, [1, 0]]])

    # Remove duplicate edges and create CSR matrix
    row_indices = edges_symmetric[:, 0]
    col_indices = edges_symmetric[:, 1]
    data = np.ones(len(edges_symmetric), dtype=int)

    # Create sparse matrix and eliminate duplicates by summing
    csr_graph = csr_matrix(
        (data, (row_indices, col_indices)), shape=(n_vertices, n_vertices)
    )

    # Convert to binary (in case of duplicate edges)
    csr_graph.data = (csr_graph.data > 0).astype(int)

    return csr_graph


####################################################################################################
def edges_to_csr(
    edges: np.ndarray,
    edge_values: np.ndarray,
    n_vertices: Optional[int] = None,
    symmetric: bool = True,
) -> csr_matrix:
    """
    Convert an edge list with values to CSR graph representation.

    This method constructs a graph from a list of edges and their corresponding
    weights/values. Useful for creating graphs from pre-computed edge lists.

    Parameters
    ----------
    edges : np.ndarray
        A 2D numpy array of shape (n_edges, 2) where each row contains the
        indices of two connected vertices. Vertex indices should be non-negative integers.

    edge_values : np.ndarray
        A 1D numpy array of length n_edges containing the weight/value for each edge.
        Values can be any numeric type (int, float, etc.).

    n_vertices : int, optional
        Total number of vertices in the graph. If None, it will be inferred
        as the maximum vertex index + 1. Providing this parameter is recommended
        for graphs with isolated vertices.

    symmetric : bool, default=True
        If True, creates an undirected graph by adding reverse edges with the same values.
        If False, creates a directed graph using only the provided edges.

    Returns
    -------
    csr_matrix
        A scipy sparse CSR matrix of shape (n_vertices, n_vertices) where
        entry (i,j) contains the weight of the edge from vertex i to vertex j.
        For undirected graphs (symmetric=True), the matrix is symmetric.

    Raises
    ------
    ValueError
        If edges array is not 2D, doesn't have 2 columns, contains negative indices,
        edge_values length doesn't match number of edges, or if n_vertices is less
        than the maximum vertex index.

    TypeError
        If edges contains non-integer values or edge_values is not numeric.

    Examples
    --------
    >>> import numpy as np
    >>> # Create a simple weighted graph
    >>> edges = np.array([[0, 1],
    ...                   [1, 2],
    ...                   [0, 2]])
    >>> values = np.array([2.5, 1.0, 3.2])
    >>> csr_graph = edges_to_csr(edges, values)
    >>> print("Symmetric weighted graph:")
    >>> print(csr_graph.toarray())
    Symmetric weighted graph:
    [[0.  2.5 3.2]
    [2.5 0.  1. ]
    [3.2 1.  0. ]]

    >>> # Directed graph example
    >>> edges_directed = np.array([[0, 1], [1, 2]])
    >>> values_directed = np.array([0.8, 1.5])
    >>> csr_directed = edges_to_csr(edges_directed, values_directed, symmetric=False)
    >>> print("Directed graph:")
    >>> print(csr_directed.toarray())
    Directed graph:
    [[0.  0.8 0. ]
    [0.  0.  1.5]
    [0.  0.  0. ]]

    >>> # Handle duplicate edges (values are summed)
    >>> edges_dup = np.array([[0, 1], [0, 1], [1, 0]])
    >>> values_dup = np.array([1.0, 2.0, 0.5])
    >>> csr_dup = edges_to_csr(edges_dup, values_dup)
    >>> print("Duplicate edges (summed):")
    >>> print(csr_dup.toarray())
    Duplicate edges (summed):
    [[0.  3.5]
    [3.5 0. ]]
    """
    if not isinstance(edges, np.ndarray):
        raise TypeError("Edges must be a numpy array")

    if not isinstance(edge_values, np.ndarray):
        raise TypeError("Edge values must be a numpy array")

    if edges.ndim != 2:
        raise ValueError("Edges array must be 2D")

    if edges.shape[1] != 2:
        raise ValueError("Edges array must have exactly 2 columns")

    if edge_values.ndim != 1:
        raise ValueError("Edge values must be a 1D array")

    if len(edges) != len(edge_values):
        raise ValueError(
            f"Number of edges ({len(edges)}) must match number of edge values ({len(edge_values)})"
        )

    if not np.issubdtype(edges.dtype, np.integer):
        raise TypeError("Edges array must contain integer vertex indices")

    if not np.issubdtype(edge_values.dtype, np.number):
        raise TypeError("Edge values must be numeric")

    if np.any(edges < 0):
        raise ValueError("Vertex indices must be non-negative")

    if len(edges) == 0:
        warnings.warn("Empty edge list provided", UserWarning)
        if n_vertices is None:
            n_vertices = 0
        return csr_matrix((n_vertices, n_vertices))

    max_vertex_idx = np.max(edges)

    if n_vertices is None:
        n_vertices = max_vertex_idx + 1
    elif n_vertices <= max_vertex_idx:
        raise ValueError(
            f"n_vertices ({n_vertices}) must be greater than maximum vertex index ({max_vertex_idx})"
        )

    # Prepare edge data
    if symmetric:
        # Add reverse edges for undirected graph
        all_edges = np.vstack([edges, edges[:, [1, 0]]])
        all_values = np.concatenate([edge_values, edge_values])
    else:
        all_edges = edges
        all_values = edge_values

    row_indices = all_edges[:, 0]
    col_indices = all_edges[:, 1]

    # Create CSR matrix (duplicate edges will be summed automatically)
    csr_graph = csr_matrix(
        (all_values, (row_indices, col_indices)), shape=(n_vertices, n_vertices)
    )

    return csr_graph
