networktools module
===================

.. automodule:: clabtoolkit.networktools
   :members:
   :undoc-members:
   :show-inheritance:

The networktools module provides sparse graph tools for brain connectivity matrices and surface meshes. All the functions work with ``scipy.sparse`` CSR matrices, which keeps large graphs light in memory.

Key Features
------------
- Convert dense adjacency matrices, edge lists and triangulated meshes to CSR graphs
- Find the connected components of a graph, or directly of an edge list
- Support for symmetric (undirected) graphs and weighted edges

Main Functions
--------------

Graph Conversion
~~~~~~~~~~~~~~~~
- ``adjacency_matrix_to_csr(adj_matrix)``: dense adjacency matrix to CSR
- ``triangulated_mesh_to_csr(faces, n_vertices=None)``: mesh faces to a vertex adjacency graph
- ``edges_to_csr(edges, edge_values=None, n_vertices=None, symmetric=True)``: edge list (with optional values) to CSR

Connected Components
~~~~~~~~~~~~~~~~~~~~
- ``connected_components(csr_graph)``: components of a CSR graph
- ``edges_to_components(edges)``: components of an edge array with arbitrary vertex indices

Both functions return the number of components, the component label of each vertex and a dictionary with the size of each component.

Common Usage Examples
---------------------

Converting a mesh to a graph::

    import numpy as np
    from clabtoolkit.networktools import triangulated_mesh_to_csr

    # Two triangles sharing an edge (faces only: the vertex coordinates are not needed)
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    graph = triangulated_mesh_to_csr(faces)

    print(graph.shape)  # (4, 4)
    print(graph.nnz)    # 10 non-zero entries

    # With a FreeSurfer surface
    # import nibabel as nib
    # vertices, faces = nib.freesurfer.read_geometry("lh.pial")
    # graph = triangulated_mesh_to_csr(faces, n_vertices=len(vertices))

Working with connectivity matrices::

    from clabtoolkit.networktools import adjacency_matrix_to_csr

    conn = np.random.rand(100, 100)
    conn = (conn + conn.T) / 2  # make it symmetric
    sparse_conn = adjacency_matrix_to_csr(conn)

Edge lists and connected components::

    from clabtoolkit.networktools import edges_to_csr, connected_components

    # A triangle (0, 1, 2) and a separate edge (3, 4)
    edges = np.array([[0, 1], [1, 2], [2, 0], [3, 4]])
    graph = edges_to_csr(edges, n_vertices=5)

    n_components, labels, sizes = connected_components(graph, verbose=False)
    print(n_components)  # 2
    print(sizes)         # {0: 3, 1: 2}
