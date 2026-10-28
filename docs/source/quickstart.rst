Quick Start
===========

The examples below use the sample data of clabtoolkit, which is retrieved from its Zenodo repository the first time they
run (this needs an internet connection), so you can run them as they are.

Filter a parcellation by region name
------------------------------------

.. code-block:: python

    import clabtoolkit.misctools as cltmisc
    import clabtoolkit.parcellationtools as cltparc

    # Load the sample parcellation of clabtoolkit (a Chimera parcellation of the Colin27 average brain).
    # The color table that is next to the file is found automatically
    data_dir = cltmisc.get_sample_data_dir("parcellations")
    parc = cltparc.Parcellation(data_dir / "sub-colin27_ses-01_space-tallin_atlas-chimeraDFMFHISIFN_seg-aparc_desc-grow1mm_dseg.nii.gz")

    # Keep only the precentral regions (the object is modified in place)
    parc.keep_by_name(['ctx-rh-precentral', 'ctx-lh-precentral'])
    print(parc.name)  # ['ctx-rh-precentral', 'ctx-lh-precentral']

To load your own parcellation, use ``cltparc.Parcellation("parc.nii.gz", color_table="lut.txt")``.

Sample an image along the streamlines of a tractogram
-----------------------------------------------------

.. code-block:: python

    import clabtoolkit.misctools as cltmisc
    import clabtoolkit.tracttools as clttract

    # The sample tractogram is retrieved from Zenodo the first time. The sample parcellation is the image
    tract_file = cltmisc.get_sample_data_dir("tractograms") / "sub-colin27_ses-01_space-tallin_label-cc_tractogram.trk"
    parc_file = cltmisc.get_sample_data_dir("parcellations") / "sub-colin27_ses-01_space-tallin_atlas-chimeraDFMFHISIFN_seg-aparc_desc-grow1mm_dseg.nii.gz"
    tract = clttract.Tractogram(tract_file)

    # Sample the parcellation at every point of every streamline. The values are labels,
    # so the nearest neighbor is used
    tract.interpolate_on_tractogram(parc_file, map_name="label",
                                    storage_mode="data_per_point", interp_method="nearest")

    print(tract.list_maps()["maps_per_point"])  # [..., 'label']
    print(tract.data_per_point["label"][0][:5])  # labels along the first streamline

Use ``interp_method="linear"`` for scalar images, such as FA, and ``storage_mode="data_per_streamline"`` with
``reduction="mean"`` to store a single value per streamline.

Plot a surface with a scalar map
--------------------------------

.. code-block:: python

    import clabtoolkit.misctools as cltmisc
    import clabtoolkit.surfacetools as cltsurf
    import clabtoolkit.visualizationtools as cltvis

    # Load the sample surface and thickness map of clabtoolkit (retrieved from Zenodo the first time)
    surf = cltsurf.Surface(cltmisc.get_sample_data_dir("surfaces") / "sub-colin27_ses-01_hemi-L_space-tallin_pial.surf.gii")
    surf.load_scalar_maps(cltmisc.get_sample_data_dir("corticalmaps") / "sub-colin27_ses-01_hemi-L_space-tallin_thickness.shape.gii",
                          maps_names="thickness")

    # Plot two views of the map and save the figure
    plotter = cltvis.BrainPlotter()
    plotter.plot(objs2plot=surf, views=["lateral", "medial"], map_names="thickness",
                 colormaps="viridis", colorbar=True, save_path="/tmp/thickness.png")

To plot your own data, load a surface with ``cltsurf.Surface("lh.pial")`` and its map with
``surf.load_scalar_maps("lh.thickness", maps_names="thickness")``.

Next steps
----------

See :doc:`examples` for the notebooks of every module and :doc:`api` for the full reference.
