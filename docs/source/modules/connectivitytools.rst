connectivitytools module
========================

The connectivitytools module provides the ``Connectome`` class, a container for brain connectivity matrices (structural or functional) together with the information of their regions (names, colors, indices and 3D coordinates).

Key Features
------------
- Create a connectome from a NumPy array, a SciPy sparse matrix, a CSV file or an HDF5 file
- Region metadata: names, colors, indices, coordinates and color tables
- Matrix operations: threshold (by value or sparsity), binarize, symmetrize, remove the diagonal and extract subnetworks
- Basic statistics: density, connectivity statistics and graph metrics
- Save to and load from HDF5 (dense or sparse storage) and CSV
- Plot the matrix as a heatmap or visualize the network in 3D with PyVista
- Generate synthetic connectomes for testing

API reference
-------------

.. automodule:: clabtoolkit.connectivitytools
   :members:
   :undoc-members:
   :show-inheritance:

Main Class
----------

Connectome
~~~~~~~~~~
Key methods:

- ``from_csv``, ``from_h5``, ``from_csr``, ``generate_connectome``: alternative constructors
- ``save_h5``, ``load_h5``, ``load_csv``, ``to_csr``: input and output
- ``get_region_names``, ``set_region_names``, ``get_region_colors``, ``set_region_colors``, ``load_colortable``: region metadata
- ``threshold``, ``binarize``, ``set_symmetric``, ``set_diagonal_to_zero``, ``get_subnetwork``: matrix operations
- ``get_density``, ``get_connectivity_stats``, ``get_graph_metrics``: metrics
- ``plot_matrix``, ``visualize_3d``, ``save_visualization``: visualization
- ``get_info``: summary of the object

Common Usage Examples
---------------------

Generating and inspecting a connectome::

    import clabtoolkit.connectivitytools as cltconn

    # Simulate a connectome with 20 regions organized in modules
    conn = cltconn.Connectome.generate_connectome(20, method="modular", seed=18)
    conn.get_info()

    print("Density:", conn.get_density())
    stats = conn.get_connectivity_stats()

Cleaning and thresholding::

    # Operations that modify the object in place
    conn.set_symmetric()
    conn.set_diagonal_to_zero()

    # With copy=True (default) a new connectome is returned
    strong = conn.threshold(method="value", threshold=0.3)
    sparse = conn.threshold(method="sparsity", threshold=0.2, binarize=True)

    # Keep only some regions
    sub = conn.get_subnetwork([0, 1, 2, 3])

Plotting::

    conn.plot_matrix(figsize=(5, 4))
    conn.plot_matrix(threshold=0.3, log_scale=True, cmap="viridis", figsize=(5, 4))

Saving and loading::

    conn.save_h5("connectome.h5")
    conn2 = cltconn.Connectome.from_h5("connectome.h5")

    # From a CSV file with the matrix
    conn3 = cltconn.Connectome.from_csv("matrix.csv", name="my_connectome")
