Quick Start
===========

The examples below are self-contained: they use simulated data, so you can run them as they are.

Filter a parcellation by region name
------------------------------------

.. code-block:: python

    import clabtoolkit.parcellationtools as cltparc

    # Simulate a parcellation with 10 regions and name them
    parc = cltparc.Parcellation.simulate_parcellation(n_regions=10, seed=42)
    parc.set_names(['ctx-rh-precentral', 'ctx-lh-precentral', 'ctx-rh-postcentral', 'ctx-lh-postcentral',
                    'ctx-rh-superiorfrontal', 'ctx-lh-superiorfrontal', 'ctx-rh-supramarginal',
                    'ctx-lh-supramarginal', 'subc-rh-amygdala', 'subc-lh-amygdala'])

    # Keep only the precentral regions (the object is modified in place)
    parc.keep_by_name(['ctx-rh-precentral', 'ctx-lh-precentral'])
    print(parc.name)  # ['ctx-rh-precentral', 'ctx-lh-precentral']

To load your own parcellation, use ``cltparc.Parcellation("parc.nii.gz", color_table="lut.txt")``.

Sample an image along the streamlines of a tractogram
-----------------------------------------------------

.. code-block:: python

    import clabtoolkit.imagetools as cltimg
    import clabtoolkit.tracttools as clttract

    # Simulate a reference image and a tractogram with 5 bundles
    cltimg.simulate_image(reference=(100, 100, 70), simulated_image="image.nii.gz")
    tract = clttract.Tractogram.simulate_tractogram(ref_image="image.nii.gz")

    # Sample the image at every point of every streamline
    tract.interpolate_on_tractogram("image.nii.gz", map_name="fa",
                                    storage_mode="data_per_point", interp_method="linear")

    print(tract.list_maps()["maps_per_point"])  # ['fa']
    print(tract.data_per_point["fa"][0][:5])    # values along the first streamline

Use ``storage_mode="data_per_streamline"`` with ``reduction="mean"`` to store a single value per streamline.

Plot a surface with a scalar map
--------------------------------

.. code-block:: python

    import clabtoolkit.surfacetools as cltsurf
    import clabtoolkit.visualizationtools as cltvis

    # Simulate a surface and a thickness map
    surf = cltsurf.Surface.simulate_surface(seed=42, shape="sphere")
    surf.simulate_map(map_name="thickness")

    # Plot two views of the map and save the figure
    plotter = cltvis.BrainPlotter()
    plotter.plot(objs2plot=surf, views=["lateral", "medial"], map_names="thickness",
                 colormaps="viridis", colorbar=True, save_path="thickness.png")

To plot your own data, load a surface with ``cltsurf.Surface("lh.pial")`` and its map with
``surf.load_scalar_maps("lh.thickness", maps_names="thickness")``.

Next steps
----------

See :doc:`examples` for the notebooks of every module and :doc:`api` for the full reference.
