networktools module
==================

.. automodule:: clabtoolkit.networktools
   :members:
   :undoc-members:
   :show-inheritance:

The networktools module provides graph analysis and network theory tools specifically designed for brain connectivity and mesh-based analysis.

Key Features
------------
- Graph representation creation from brain meshes
- Sparse matrix operations for large-scale networks  
- Connectivity analysis utilities
- Network topology analysis
- Efficient storage and manipulation of large graphs
- Integration with scipy.sparse for performance

Main Functions
--------------

Graph Conversion
~~~~~~~~~~~~~~~~
- ``triangulated_mesh_to_csr()``: Convert triangulated meshes to compressed sparse row format
- ``adjacency_matrix_to_csr()``: Convert adjacency matrices to sparse format
- ``edges_to_csr()``: Convert edge lists to compressed sparse row format

Common Usage Examples
---------------------

Converting mesh to graph::

    from clabtoolkit.networktools import triangulated_mesh_to_csr
    import nibabel as nib
    
    # Load surface mesh
    vertices, faces = nib.freesurfer.read_geometry("lh.pial")
    
    # Convert to sparse graph representation
    graph_csr = triangulated_mesh_to_csr(vertices, faces)
    
    print(f"Graph shape: {graph_csr.shape}")
    print(f"Number of edges: {graph_csr.nnz}")

Working with connectivity matrices::

    from clabtoolkit.networktools import adjacency_matrix_to_csr
    import numpy as np
    
    # Convert dense connectivity matrix to sparse format
    connectivity_matrix = np.random.rand(100, 100)
    connectivity_matrix = (connectivity_matrix + connectivity_matrix.T) / 2  # Make symmetric
    
    # Convert to sparse format for efficient processing
    sparse_conn = adjacency_matrix_to_csr(connectivity_matrix, threshold=0.1)
    
    # Analyze network properties
    print(f"Network density: {sparse_conn.nnz / (sparse_conn.shape[0]**2):.3f}")

Edge list conversion::

    # Convert edge list to sparse matrix
    edges = [[0, 1], [1, 2], [2, 0]]  # Triangle connectivity
    sparse_matrix = edges_to_csr(edges, n_nodes=3)
    
    print(f"Graph shape: {sparse_matrix.shape}")
    print(f"Number of edges: {sparse_matrix.nnz}")
    print(f"Network density: {sparse_matrix.nnz / (sparse_matrix.shape[0]**2):.3f}")