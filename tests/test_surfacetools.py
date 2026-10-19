"""Tests for clabtoolkit.surfacetools."""

import copy
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import pyvista as pv

import clabtoolkit.freesurfertools as cltfree
from clabtoolkit.surfacetools import Surface, create_surface_colortable, merge_surfaces

# Sphere simulated by pyvista: 842 vertices, 1680 faces and 2520 edges
N_VERTS, N_FACES, N_EDGES = 842, 1680, 2520


####################################################################################################
# Fixtures
####################################################################################################
@pytest.fixture(autouse=True)
def _quiet():
    """Silence the progress messages and warnings that the surface methods emit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture
def sphere():
    """Simulated sphere with the default thickness, curv and sulc maps."""
    return Surface.simulate_surface(seed=42, shape="sphere")


@pytest.fixture
def bare():
    """Simulated sphere without any map."""
    return Surface.simulate_surface(seed=42, shape="sphere", maps=None)


@pytest.fixture
def plane():
    """Open mesh (a plane) that has boundary edges."""
    return Surface.simulate_surface(shape="plane", maps=None)


@pytest.fixture
def annot(sphere):
    """Simulated annotation with 10 regions that fits the sphere."""
    return cltfree.AnnotParcellation.simulate_annotation(sphere, n_regions=10)


@pytest.fixture
def parc(sphere, annot):
    """The sphere with the simulated annotation loaded as the overlay 'Sim'."""
    sphere.load_annotation(annot, parc_name="Sim")
    return sphere


@pytest.fixture
def surf_file(tmp_path, bare):
    """FreeSurfer-format file holding the bare sphere (hemisphere detected: lh)."""
    path = tmp_path / "lh.pial"
    bare.save_surface(str(path))
    return path


@pytest.fixture
def two_surfs():
    """A sphere (842 vertices) and a subdivided cube (98 vertices), both with thickness."""
    first = Surface.simulate_surface(seed=1, shape="sphere", maps=["thickness"])
    second = Surface.simulate_surface(
        seed=2, shape="cube", subdivide=2, maps=["thickness"], hemi="rh"
    )
    return first, second


####################################################################################################
# Simulated surfaces
####################################################################################################
class TestSimulateSurface:
    @pytest.mark.parametrize("shape", sorted(Surface.SIMULATED_SHAPES))
    def test_all_shapes_are_triangular_meshes(self, shape):
        surf = Surface.simulate_surface(shape=shape, maps=None, seed=0)
        faces = surf.get_faces()
        assert surf.is_loaded() and surf.name == f"simulated_{shape}"
        assert faces.shape[0] > 0 and faces.shape[1] == 3
        assert faces.min() >= 0 and faces.max() < surf.mesh.n_points

    def test_defaults(self, sphere):
        assert (sphere.mesh.n_points, sphere.mesh.n_cells) == (N_VERTS, N_FACES)
        assert sphere.hemi == "lh" and sphere.name == "simulated_sphere"
        assert sphere.surf is None
        assert sphere.list_overlays() == {
            "default": "scalar_with_colortable",
            "thickness": "scalar",
            "curv": "scalar",
            "sulc": "scalar",
        }

    @pytest.mark.parametrize("maps", [None, []])
    def test_without_maps(self, maps):
        surf = Surface.simulate_surface(seed=0, maps=maps)
        assert list(surf.list_overlays()) == ["default"]

    def test_maps_as_string(self):
        surf = Surface.simulate_surface(maps="myelin", seed=0)
        assert list(surf.list_overlays()) == ["default", "myelin"]

    def test_seed_makes_maps_reproducible(self):
        a = Surface.simulate_surface(seed=5)
        b = Surface.simulate_surface(seed=5)
        c = Surface.simulate_surface(seed=6)
        for name in ("thickness", "curv", "sulc"):
            assert np.array_equal(a.mesh.point_data[name], b.mesh.point_data[name])
            assert not np.array_equal(a.mesh.point_data[name], c.mesh.point_data[name])

    def test_maps_of_one_surface_differ(self, sphere):
        data = sphere.mesh.point_data
        assert not np.allclose(data["thickness"], data["curv"])
        assert not np.allclose(data["curv"], data["sulc"])

    def test_subdivide_densifies_mesh(self):
        n_points = [
            Surface.simulate_surface("cube", subdivide=n, maps=None).mesh.n_points
            for n in (0, 1, 2)
        ]
        assert n_points == [8, 26, 98]

    def test_shape_kwargs_are_forwarded(self):
        surf = Surface.simulate_surface("sphere", radius=50, maps=None)
        assert np.allclose(
            surf.get_bounding_box(), [[-50, -50, -50], [50, 50, 50]], atol=0.5
        )

    def test_shape_from_mesh(self):
        mesh = pv.Superquadric()
        n_before = mesh.n_points
        surf = Surface.simulate_surface(mesh, maps="area", seed=1)
        assert surf.name == "simulated_mesh" and "area" in surf.list_overlays()
        assert surf.get_faces().shape[1] == 3
        assert mesh.n_points == n_before  # The input mesh is not modified

    def test_hemi_name_color_and_alpha(self):
        surf = Surface.simulate_surface(
            hemi="rh", name="mine", color="#3b2f89", alpha=0.5, maps=None
        )
        assert (surf.hemi, surf.name) == ("rh", "mine")
        row = surf.colortables["default"]["color_table"][0]
        assert np.allclose(row[:4], [0x3B / 255, 0x2F / 255, 0x89 / 255, 0.5])

    def test_errors(self):
        with pytest.raises(ValueError, match="Unknown shape"):
            Surface.simulate_surface("not-a-shape")
        with pytest.raises(ValueError, match="shape must be"):
            Surface.simulate_surface(5)
        with pytest.raises(ValueError, match="non-negative"):
            Surface.simulate_surface("cube", subdivide=-1)
        with pytest.raises(ValueError, match="Alpha"):
            Surface.simulate_surface(alpha=2)


####################################################################################################
# Creation and loading
####################################################################################################
class TestCreation:
    def test_empty(self):
        surf = Surface()
        assert not surf.is_loaded() and surf.mesh is None
        assert surf.name == "unknown" and surf.hemi is None and surf.surf is None
        assert list(surf.colortables) == ["default"]

    def test_from_arrays(self, sphere):
        surf = Surface(
            vertices=sphere.get_vertices(),
            faces=sphere.get_faces(),
            color="red",
            alpha=0.8,
            hemi="rh",
            name="arrays",
        )
        assert surf.is_loaded() and (surf.hemi, surf.name) == ("rh", "arrays")
        assert np.allclose(surf.get_vertices(), sphere.get_vertices())
        assert np.array_equal(surf.get_faces(), sphere.get_faces())
        assert np.allclose(
            surf.colortables["default"]["color_table"][0, :4], [1, 0, 0, 0.8]
        )

    def test_from_arrays_defaults_to_lh(self, sphere):
        surf = Surface(vertices=sphere.get_vertices(), faces=sphere.get_faces())
        assert surf.hemi == "lh"
        assert list(surf.list_overlays()) == ["default"]

    def test_from_mesh_copies_the_mesh(self):
        mesh = pv.Sphere()
        surf = Surface(mesh, hemi="rh")
        assert surf.hemi == "rh" and surf.mesh.n_points == mesh.n_points
        assert surf.mesh is not mesh
        surf.mesh.points = surf.mesh.points + 100
        assert np.abs(mesh.points).max() < 1  # The original mesh was not shifted

    def test_load_from_mesh_method(self):
        surf = Surface()
        surf.load_from_mesh(pv.Sphere(), color="#26D217", alpha=0.2)
        assert surf.is_loaded() and surf.hemi == "lh" and surf.surf is None
        assert surf.colortables["default"]["color_table"][0, 3] == pytest.approx(0.2)

    def test_load_from_arrays_method(self, sphere):
        surf = Surface()
        surf.load_from_arrays(
            sphere.get_vertices(), sphere.get_faces(), hemi="rh", surface_file="x.surf"
        )
        assert surf.hemi == "rh" and surf.surf == "x.surf"
        assert surf.mesh.n_points == N_VERTS

    @pytest.mark.parametrize("as_path", [True, False])
    def test_from_file(self, surf_file, bare, as_path):
        surf = Surface(
            surf_file if as_path else str(surf_file), color="#3b2f89", alpha=0.5
        )
        assert surf.surf == str(surf_file) and surf.hemi == "lh"
        assert np.allclose(surf.get_vertices(), bare.get_vertices(), atol=1e-4)
        assert np.array_equal(surf.get_faces(), bare.get_faces())
        assert surf.colortables["default"]["color_table"][0, 3] == pytest.approx(0.5)

    def test_load_from_file_method(self, surf_file):
        surf = Surface()
        surf.load_from_file(surf_file)
        assert surf.is_loaded() and surf.mesh.n_points == N_VERTS

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("lh.pial", "lh"),
            ("rh.pial", "rh"),
            ("sub-01_hemi-R_pial.surf", "rh"),
            ("sub-01_hemi-L_pial.surf", "lh"),
            ("something.pial", "lh"),  # Default when nothing identifies the hemisphere
        ],
    )
    def test_hemisphere_detected_from_file_name(self, tmp_path, bare, name, expected):
        bare.save_surface(str(tmp_path / name))
        assert Surface(tmp_path / name).hemi == expected

    def test_explicit_hemisphere_wins(self, tmp_path, bare):
        bare.save_surface(str(tmp_path / "rh.pial"))
        assert Surface(tmp_path / "rh.pial", hemi="lh").hemi == "lh"

    def test_name_defaults_to_unknown(self, surf_file):
        assert Surface(surf_file).name == "unknown"
        assert Surface(surf_file, name="mine").name == "mine"
        assert Surface(surf_file, name=None).name == "unknown"

    def test_errors(self, tmp_path, sphere):
        verts, faces = sphere.get_vertices(), sphere.get_faces()
        with pytest.raises(FileNotFoundError):
            Surface(tmp_path / "missing.pial")
        with pytest.raises(FileNotFoundError):
            Surface().load_from_file(tmp_path / "missing.pial")
        with pytest.raises(ValueError, match="Cannot specify both"):
            Surface(tmp_path / "x", vertices=verts, faces=faces)
        with pytest.raises(ValueError, match="faces must also be provided"):
            Surface(vertices=verts)
        with pytest.raises(ValueError, match="vertices must also be provided"):
            Surface(faces=faces)
        with pytest.raises(ValueError, match="Faces array must have shape"):
            Surface(vertices=verts, faces=np.zeros((3, 4), dtype=int))
        with pytest.raises(ValueError, match="Vertices array must have shape"):
            Surface(vertices=np.zeros((4, 2)), faces=faces)
        with pytest.raises(ValueError, match="valid indices"):
            Surface(vertices=verts[:4], faces=np.array([[0, 1, 9]]))

    @pytest.mark.parametrize("alpha", [-0.1, 1.5, 2])
    def test_alpha_out_of_range(self, sphere, alpha):
        verts, faces = sphere.get_vertices(), sphere.get_faces()
        with pytest.raises(ValueError, match="Alpha"):
            Surface(vertices=verts, faces=faces, alpha=alpha)
        with pytest.raises(ValueError, match="Alpha"):
            Surface().load_from_arrays(verts, faces, alpha=alpha)
        with pytest.raises(ValueError, match="Alpha"):
            Surface().load_from_mesh(pv.Sphere(), alpha=alpha)

    def test_corrupted_file(self, tmp_path):
        bad = tmp_path / "lh.corrupted"
        bad.write_text("this is not a surface")
        with pytest.raises(ValueError, match="Failed to load"):
            Surface(bad)


####################################################################################################
# Geometry
####################################################################################################
class TestGeometry:
    def test_vertices_and_faces(self, bare):
        verts, faces = bare.get_vertices(), bare.get_faces()
        assert verts.shape == (N_VERTS, 3) and faces.shape == (N_FACES, 3)
        assert np.allclose(verts, bare.mesh.points)

    def test_edges_of_a_closed_mesh(self, bare):
        edges = bare.get_edges()
        assert edges.shape == (N_EDGES, 2)
        assert (edges[:, 0] < edges[:, 1]).all()  # Each edge appears once, ordered
        assert len({tuple(e) for e in edges}) == N_EDGES
        # Euler characteristic of a sphere
        assert N_VERTS - len(edges) + N_FACES == 2

    def test_edge_counts_of_a_closed_mesh(self, bare):
        edges, counts = bare.get_edges(return_counts=True)
        assert edges.shape == (N_EDGES, 2) and counts.shape == (N_EDGES,)
        assert (counts == 2).all()  # Every edge is shared by two faces
        assert bare.get_boundary_edges().shape == (0, 2)
        assert np.array_equal(bare.get_manifold_edges(), edges)

    def test_boundary_and_manifold_edges_of_an_open_mesh(self, plane):
        edges, counts = plane.get_edges(return_counts=True)
        boundary, manifold = plane.get_boundary_edges(), plane.get_manifold_edges()
        assert len(edges) == 320
        assert set(np.unique(counts)) == {1, 2}
        assert len(boundary) == 40 and len(manifold) == 280
        assert np.array_equal(boundary, edges[counts == 1])
        assert np.array_equal(manifold, edges[counts == 2])
        assert len(np.unique(boundary)) == 40  # The boundary is a single closed loop

    def test_boundary_edges_of_a_mesh_with_a_hole(self):
        theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
        outer = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
        hole = np.column_stack([0.3 * np.cos(theta), 0.3 * np.sin(theta), np.zeros(30)])
        mesh = pv.PolyData(np.vstack([outer, hole])).delaunay_2d()
        # The hole is filled by the triangulation, so only the square is a boundary
        assert len(Surface(mesh).get_boundary_edges()) == 4

    def test_normals(self, bare):
        assert bare.get_normals() is None
        bare.compute_normals()
        normals = bare.get_normals()
        assert normals.shape == (N_VERTS, 3)
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
        assert bare.list_overlays()["Normals"] == "normals"
        # The vertices of a sphere centred at the origin point outwards
        assert (np.sum(normals * bare.get_vertices(), axis=1) > 0).all()

    def test_normals_can_be_computed_twice(self, bare):
        bare.compute_normals()
        first = bare.get_normals().copy()
        bare.compute_normals()
        assert np.allclose(bare.get_normals(), first)

    def test_bounding_box(self):
        surf = Surface.simulate_surface("sphere", radius=10, maps=None)
        bbox = surf.get_bounding_box()
        assert bbox.shape == (2, 3)
        assert np.allclose(bbox[0], surf.get_vertices().min(axis=0))
        assert np.allclose(bbox[1], surf.get_vertices().max(axis=0))
        assert (bbox[0] < 0).all() and (bbox[1] > 0).all()

    def test_methods_fail_on_an_empty_surface(self):
        surf = Surface()
        for method in (
            surf.get_vertices,
            surf.get_faces,
            surf.get_edges,
            surf.compute_normals,
            surf.get_bounding_box,
        ):
            with pytest.raises(RuntimeError, match="No surface data loaded"):
                method()
        assert surf.get_normals() is None


####################################################################################################
# Information
####################################################################################################
class TestGetInfo:
    def test_fresh_surface(self, bare):
        info = bare.get_info(verbose=False)
        assert info["name"] == "simulated_sphere" and info["hemi"] == "lh"
        assert info["surf_file"] is None and info["is_loaded"] is True
        assert (info["n_vertices"], info["n_faces"]) == (N_VERTS, N_FACES)
        assert info["global_color"] == "#f0f0f0" and info["global_opacity"] == 1.0
        assert info["active_overlay"] == "default" and info["n_overlays"] == 1
        assert [o["name"] for o in info["overlays"]] == ["default"]
        assert info["overlays"][0]["has_colortable"] is True

    def test_after_adding_overlays(self, parc):
        parc.compute_normals()
        info = parc.get_info(verbose=False)
        overlays = {o["name"]: o for o in info["overlays"]}
        assert info["n_overlays"] == len(overlays) == 6
        assert overlays["Sim"]["has_colortable"] and overlays["Sim"]["n_regions"] == 10
        assert not overlays["thickness"]["has_colortable"]
        assert overlays["thickness"]["n_regions"] is None
        assert overlays["thickness"]["shape"] == (N_VERTS,)
        assert overlays["Normals"]["shape"] == (N_VERTS, 3)

    def test_file_name_color_and_opacity(self, surf_file):
        info = Surface(surf_file, color="#ff0000", alpha=0.5, name="subj").get_info(
            verbose=False
        )
        assert info["name"] == "subj" and info["surf_file"] == str(surf_file)
        assert info["global_color"] == "#ff0000" and info["global_opacity"] == 0.5

    def test_verbose_prints(self, sphere, capsys):
        sphere.get_info()
        out = capsys.readouterr().out
        assert "simulated_sphere" in out and str(N_VERTS) in out

    def test_non_verbose_is_silent(self, sphere, capsys):
        sphere.get_info(verbose=False)
        assert capsys.readouterr().out == ""

    def test_empty_surface(self):
        info = Surface().get_info(verbose=False)
        assert info["is_loaded"] is False and info["n_vertices"] is None
        assert info["overlays"] == []


####################################################################################################
# Overlays
####################################################################################################
class TestOverlays:
    def test_list_overlays_types(self, parc):
        parc.compute_normals()
        assert parc.list_overlays() == {
            "default": "scalar_with_colortable",
            "thickness": "scalar",
            "curv": "scalar",
            "sulc": "scalar",
            "Sim": "scalar_with_colortable",
            "Normals": "normals",
        }

    def test_set_active_overlay(self, parc):
        assert parc.mesh.active_scalars_name != "thickness"
        parc.set_active_overlay("thickness")
        assert parc.mesh.active_scalars_name == "thickness"
        parc.set_active_overlay("Sim")
        assert parc.mesh.active_scalars_name == "Sim"

    def test_set_active_overlay_unknown(self, sphere):
        with pytest.raises(ValueError, match="not found"):
            sphere.set_active_overlay("nope")

    def test_remove_scalar_overlay(self, sphere):
        sphere.remove_overlay("curv")
        assert list(sphere.list_overlays()) == ["default", "thickness", "sulc"]
        assert "curv" not in sphere.mesh.point_data

    def test_remove_parcellation_drops_its_colortable(self, parc):
        assert "Sim" in parc.colortables
        parc.remove_overlay("Sim")
        assert "Sim" not in parc.list_overlays() and "Sim" not in parc.colortables

    def test_remove_unknown_overlay(self, sphere):
        with pytest.raises(ValueError, match="not found"):
            sphere.remove_overlay("nope")

    def test_overlay_info_scalar(self, sphere):
        info = sphere.get_overlay_info("thickness")
        assert info["name"] == "thickness" and info["data_shape"] == (N_VERTS,)
        assert info["data_type"] == "float32" and info["has_colortable"] is False

    def test_overlay_info_parcellation(self, parc, annot):
        info = parc.get_overlay_info("Sim")
        assert info["has_colortable"] is True and info["num_regions"] == 10
        assert info["region_names"] == list(annot.regnames)
        assert info["data_shape"] == (N_VERTS,)

    def test_overlay_info_unknown_overlay_is_empty(self, sphere):
        info = sphere.get_overlay_info("nope")
        assert info["name"] == "nope" and info["data_shape"] is None
        assert info["data_type"] is None and info["has_colortable"] is False


####################################################################################################
# Annotations
####################################################################################################
class TestAnnotation:
    def test_load_annotation_object(self, sphere, annot):
        sphere.load_annotation(annot, parc_name="Sim")
        assert sphere.list_overlays()["Sim"] == "scalar_with_colortable"
        labels = sphere.mesh.point_data["Sim"]
        assert len(np.unique(labels)) == 10 and len(labels) == N_VERTS
        assert sphere.colortables["Sim"]["names"] == list(annot.regnames)

    @pytest.mark.parametrize("as_path", [True, False])
    def test_load_annotation_file(self, tmp_path, sphere, annot, as_path):
        file = tmp_path / "lh.sim.annot"
        annot.save_annotation(str(file), overwrite=True)
        sphere.load_annotation(file if as_path else str(file), parc_name="FromFile")
        assert sphere.get_overlay_info("FromFile")["num_regions"] == 10

    def test_annotation_without_name_still_adds_an_overlay(self, sphere, annot):
        before = set(sphere.list_overlays())
        sphere.load_annotation(annot)
        assert len(set(sphere.list_overlays()) - before) == 1

    def test_two_annotations_coexist(self, sphere, annot):
        other = cltfree.AnnotParcellation.simulate_annotation(sphere, n_regions=15)
        sphere.load_annotation(annot, parc_name="A")
        sphere.load_annotation(other, parc_name="B")
        assert sphere.get_overlay_info("A")["num_regions"] == 10
        assert sphere.get_overlay_info("B")["num_regions"] == 15

    def test_region_vertices_partition_the_surface(self, parc, annot):
        indices = [parc.get_region_vertices("Sim", name) for name in annot.regnames]
        all_indices = np.concatenate(indices)
        assert len(all_indices) == N_VERTS
        assert np.array_equal(np.sort(all_indices), np.arange(N_VERTS))
        assert all(len(idx) > 0 for idx in indices)

    def test_region_vertices_share_one_label(self, parc, annot):
        labels = parc.mesh.point_data["Sim"]
        idx = parc.get_region_vertices("Sim", annot.regnames[3])
        assert len(np.unique(labels[idx])) == 1

    def test_region_vertices_errors(self, parc, annot):
        with pytest.raises(ValueError, match="Parcellation 'nope' not found"):
            parc.get_region_vertices("nope", annot.regnames[0])
        with pytest.raises(ValueError, match="Region 'nope' not found"):
            parc.get_region_vertices("Sim", "nope")

    def test_load_annotation_errors(self, tmp_path, sphere):
        with pytest.raises(FileNotFoundError):
            sphere.load_annotation(str(tmp_path / "missing.annot"))
        with pytest.raises(ValueError):
            sphere.load_annotation(5)
        cube = Surface.simulate_surface("cube", maps=None)
        annot = cltfree.AnnotParcellation.simulate_annotation(sphere, n_regions=5)
        with pytest.raises(ValueError, match="does not match"):
            cube.load_annotation(annot)

    def test_export_annotation_roundtrip(self, tmp_path, parc, annot):
        file = tmp_path / "lh.exported.annot"
        parc.export_annotation(str(file), parc_name="Sim")
        loaded = cltfree.AnnotParcellation(str(file))
        assert list(loaded.regnames) == list(annot.regnames)

    def test_export_annotation_errors(self, tmp_path, parc):
        file = tmp_path / "lh.exported.annot"
        parc.export_annotation(str(file), parc_name="Sim")
        with pytest.raises(ValueError, match="already exists"):
            parc.export_annotation(str(file), parc_name="Sim")
        with pytest.raises(ValueError, match="not found"):
            parc.export_annotation(str(tmp_path / "x.annot"), parc_name="nope")
        with pytest.raises(ValueError, match="cannot be empty"):
            parc.export_annotation("", parc_name="Sim")
        with pytest.raises(FileNotFoundError):
            parc.export_annotation(str(tmp_path / "no" / "x.annot"), parc_name="Sim")


####################################################################################################
# Scalar maps
####################################################################################################
class TestScalarMaps:
    def test_load_from_freesurfer_map_file(self, tmp_path, sphere, surf_file):
        map_file = tmp_path / "lh.thickness"
        sphere.export_map("thickness", str(map_file))
        surf = Surface(surf_file)
        surf.load_scalar_maps(str(map_file), maps_names="Thick")
        assert surf.list_overlays()["Thick"] == "scalar"
        assert np.allclose(
            surf.mesh.point_data["Thick"],
            sphere.mesh.point_data["thickness"],
            atol=1e-5,
        )

    def test_map_name_defaults_to_the_file_extension(self, tmp_path, sphere, surf_file):
        map_file = tmp_path / "lh.thickness"
        sphere.export_map("thickness", str(map_file))
        surf = Surface(surf_file)
        surf.load_scalar_maps(map_file)  # Path object, no name
        assert "thickness" in surf.list_overlays()

    def test_several_map_files(self, tmp_path, sphere, surf_file):
        surf = Surface(surf_file)
        for name in ("thickness", "curv"):
            sphere.export_map(name, str(tmp_path / f"lh.{name}"))
            surf.load_scalar_maps(str(tmp_path / f"lh.{name}"), maps_names=name.title())
        assert {"Thickness", "Curv"} <= set(surf.list_overlays())

    def test_vertexwise_numpy_array(self, bare):
        values = np.linspace(0, 1, N_VERTS)
        bare.load_scalar_maps(values, maps_names=["ramp"])
        assert bare.list_overlays()["ramp"] == "scalar"
        assert np.allclose(bare.mesh.point_data["ramp"], values)

    def test_vertexwise_numpy_array_without_name(self, bare):
        bare.load_scalar_maps(np.random.default_rng(0).random(N_VERTS))
        assert len(bare.list_overlays()) == 2  # default + the new map

    def test_vertexwise_2d_numpy_array_with_names(self, bare):
        values = np.random.default_rng(0).random((N_VERTS, 2))
        bare.load_scalar_maps(values, maps_names=["a", "b"])
        assert np.allclose(bare.mesh.point_data["a"], values[:, 0])
        assert np.allclose(bare.mesh.point_data["b"], values[:, 1])

    def test_vertexwise_dataframe_uses_column_names(self, bare):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.random(N_VERTS), "b": rng.random(N_VERTS)})
        bare.load_scalar_maps(df)
        assert {"a", "b"} <= set(bare.list_overlays())
        assert np.allclose(bare.mesh.point_data["b"], df["b"])

    def test_vertexwise_csv(self, tmp_path, bare):
        df = pd.DataFrame(
            {"vertex_index": np.arange(N_VERTS), "val": np.arange(N_VERTS) * 0.5}
        )
        df.to_csv(tmp_path / "vertexwise.csv", index=False)
        bare.load_scalar_maps(str(tmp_path / "vertexwise.csv"))
        assert np.allclose(bare.mesh.point_data["val"], df["val"])

    @pytest.fixture
    def regionwise(self, annot):
        n = len(annot.regnames)
        return pd.DataFrame(
            {
                "region_index": np.arange(n),
                "ct": np.linspace(1, 2, n),
                "cv": np.linspace(-1, 1, n),
            }
        )

    def test_regionwise_dataframe(self, parc, annot, regionwise):
        parc.load_scalar_maps(regionwise, annotation=annot)
        assert {"ct", "cv"} <= set(parc.list_overlays())
        for i, name in enumerate(annot.regnames):
            idx = parc.get_region_vertices("Sim", name)
            assert np.allclose(parc.mesh.point_data["ct"][idx], regionwise["ct"][i])
            assert np.allclose(parc.mesh.point_data["cv"][idx], regionwise["cv"][i])

    def test_regionwise_csv_with_annotation_file(
        self, tmp_path, parc, annot, regionwise
    ):
        regionwise.to_csv(tmp_path / "regions.csv", index=False)
        annot.save_annotation(str(tmp_path / "lh.sim.annot"), overwrite=True)
        parc.load_scalar_maps(
            str(tmp_path / "regions.csv"), annotation=str(tmp_path / "lh.sim.annot")
        )
        idx = parc.get_region_vertices("Sim", annot.regnames[0])
        assert np.allclose(parc.mesh.point_data["ct"][idx], 1.0)

    def test_regionwise_numpy_array(self, parc, annot, regionwise):
        parc.load_scalar_maps(
            regionwise.to_numpy(), annotation=annot, maps_names=list(regionwise.columns)
        )
        idx = parc.get_region_vertices("Sim", annot.regnames[-1])
        assert np.allclose(parc.mesh.point_data["ct"][idx], 2.0)

    def test_errors(self, tmp_path, bare, regionwise):
        with pytest.raises(FileNotFoundError):
            bare.load_scalar_maps(str(tmp_path / "missing.thickness"))
        with pytest.raises(ValueError, match="annotation must be provided"):
            bare.load_scalar_maps(regionwise)  # Region-wise without annotation
        with pytest.raises(ValueError, match="annotation must be provided"):
            bare.load_scalar_maps(np.zeros(N_VERTS + 3), maps_names=["wrong"])
        with pytest.raises(ValueError, match="maps_names"):
            bare.load_scalar_maps(np.zeros((N_VERTS, 3)), maps_names=["only-one"])


class TestSimulateMap:
    def test_returns_and_stores_a_float32_map(self, bare):
        values = bare.simulate_map("thickness", seed=1)
        assert values.shape == (N_VERTS,) and values.dtype == np.float32
        assert bare.list_overlays()["thickness"] == "scalar"
        assert np.array_equal(bare.mesh.point_data["thickness"], values)

    def test_thickness_preset(self, bare):
        values = bare.simulate_map("thickness", seed=1)
        assert values.mean() == pytest.approx(2.5, abs=1e-3)
        assert values.min() >= 0.5 and values.max() <= 5.0

    def test_custom_mean_and_std(self, bare):
        values = bare.simulate_map("anything", mean=10, std=2, seed=0)
        assert values.mean() == pytest.approx(10, abs=1e-3)
        assert values.std() == pytest.approx(2, abs=1e-3)

    def test_value_range_clips(self, bare):
        values = bare.simulate_map(
            "anything", mean=10, std=5, value_range=(8, 12), seed=0
        )
        assert values.min() >= 8 and values.max() <= 12

    def test_seed(self, bare):
        a = bare.simulate_map("x", seed=1)
        b = bare.simulate_map("x", seed=1)
        c = bare.simulate_map("x", seed=2)
        assert np.array_equal(a, b) and not np.array_equal(a, c)

    def test_noise_makes_the_map_rougher(self, bare):
        smooth = bare.simulate_map("x", mean=0, std=1, seed=3, noise=0)
        noisy = bare.simulate_map("x", mean=0, std=1, seed=3, noise=1)
        assert not np.allclose(smooth, noisy)

    def test_set_active(self, bare):
        bare.simulate_map("first", seed=0)
        assert bare.mesh.active_scalars_name != "first"
        bare.simulate_map("second", seed=0, set_active=True)
        assert bare.mesh.active_scalars_name == "second"

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("thickness", "thickness"),
            ("lh.curv", "curv"),
            ("sulc", "sulc"),
            ("area", "area"),
            ("volume", "volume"),
            ("myelin", "myelin"),
            ("whatever", "unknown"),
        ],
    )
    def test_map_type_is_inferred_from_the_name(self, name, expected):
        assert Surface._infer_map_type(name) == expected

    def test_errors(self, bare):
        with pytest.raises(ValueError, match="map_name"):
            bare.simulate_map("")
        with pytest.raises(ValueError, match="std"):
            bare.simulate_map("x", std=-1)
        with pytest.raises(ValueError, match="smoothness"):
            bare.simulate_map("x", smoothness=0)
        with pytest.raises(ValueError, match="noise"):
            bare.simulate_map("x", noise=-1)
        with pytest.raises(ValueError, match="value_range"):
            bare.simulate_map("x", value_range=(5, 1))
        with pytest.raises(RuntimeError, match="No surface data loaded"):
            Surface().simulate_map("x")


####################################################################################################
# Colors
####################################################################################################
class TestColors:
    def test_colors_of_a_parcellation(self, parc, annot):
        colors = parc.get_vertexwise_colors("Sim")
        assert colors.shape == (N_VERTS, 4)
        regions = [parc.get_region_vertices("Sim", name) for name in annot.regnames]
        region_colors = []
        for idx in regions:  # One color per region
            assert len(np.unique(colors[idx], axis=0)) == 1
            region_colors.append(tuple(colors[idx[0]]))
        assert len(set(region_colors)) == len(regions)  # And different between regions

    def test_default_overlay_has_the_global_color(self, surf_file):
        surf = Surface(surf_file, color="#ff0000")
        colors = surf.get_vertexwise_colors()
        assert colors.shape[0] == N_VERTS
        assert np.allclose(colors[:, 0], 1.0) and np.allclose(colors[:, 1:3], 0.0)

    def test_colors_of_a_scalar_map(self, sphere):
        colors = sphere.get_vertexwise_colors("thickness")
        assert colors.shape == (N_VERTS, 3) and colors.dtype == np.uint8
        values = sphere.mesh.point_data["thickness"]
        # Lowest and highest values take the ends of the viridis colormap
        assert tuple(colors[values.argmin()]) == (68, 1, 84)
        assert tuple(colors[values.argmax()]) == (253, 231, 36)

    def test_colormap_is_used(self, sphere):
        viridis = sphere.get_vertexwise_colors("curv")
        bwr = sphere.get_vertexwise_colors("curv", colormap="bwr")
        assert not np.array_equal(viridis, bwr)

    def test_vmin_vmax_saturate_the_colormap(self, sphere):
        values = sphere.mesh.point_data["thickness"]
        colors = sphere.get_vertexwise_colors(
            "thickness", vmin=values.max() + 1, vmax=values.max() + 2
        )
        assert len(np.unique(colors, axis=0)) == 1  # Everything is below vmin

    def test_range_color_for_values_outside_the_range(self, sphere):
        values = sphere.mesh.point_data["thickness"]
        low, high = 2.4, 2.6
        colors = sphere.get_vertexwise_colors(
            "thickness", range_min=low, range_max=high, range_color="yellow"
        )
        outside = (values < low) | (values > high)
        assert outside.any() and (~outside).any()
        assert (colors[outside] == [255, 255, 0]).all()
        assert not (colors[~outside] == [255, 255, 0]).all(axis=1).any()

    def test_unknown_overlay(self, sphere):
        with pytest.raises(ValueError, match="not found"):
            sphere.get_vertexwise_colors("nope")

    def test_prepare_colors_of_a_parcellation(self, parc):
        parc.prepare_colors(overlay_name="Sim")
        assert parc.list_overlays()["rgba"] == "color_with_alpha"
        assert parc.mesh.point_data["rgba"].shape == (N_VERTS, 4)

    def test_prepare_colors_of_a_scalar_map(self, sphere):
        sphere.prepare_colors(overlay_name="curv", cmap="bwr")
        assert sphere.list_overlays()["rgba"] == "color"
        assert sphere.mesh.point_data["rgba"].shape[0] == N_VERTS

    def test_prepare_colors_unknown_overlay(self, sphere):
        with pytest.raises(ValueError, match="not found"):
            sphere.prepare_colors(overlay_name="nope")


####################################################################################################
# Merging surfaces
####################################################################################################
class TestMerge:
    def test_add_surface(self, two_surfs):
        first, second = two_surfs
        n1, n2 = first.mesh.n_points, second.mesh.n_points
        merged = first.add_surface(second)
        assert merged is not first and merged.name == "merged_surface"
        assert merged.mesh.n_points == n1 + n2
        assert merged.mesh.n_cells == first.mesh.n_cells + second.mesh.n_cells
        assert np.allclose(
            merged.get_vertices(),
            np.vstack([first.get_vertices(), second.get_vertices()]),
        )
        # The faces of the second surface are shifted by the vertices of the first one
        assert np.array_equal(
            merged.get_faces()[first.mesh.n_cells :], second.get_faces() + n1
        )
        assert np.allclose(
            merged.mesh.point_data["thickness"],
            np.r_[
                first.mesh.point_data["thickness"], second.mesh.point_data["thickness"]
            ],
        )

    def test_add_surface_does_not_modify_the_inputs(self, two_surfs):
        first, second = two_surfs
        before = (first.mesh.n_points, second.mesh.n_points)
        first.add_surface(second)
        assert (first.mesh.n_points, second.mesh.n_points) == before

    @pytest.mark.parametrize("as_path", [True, False])
    def test_add_surface_from_file(self, tmp_path, two_surfs, as_path):
        first, second = two_surfs
        second.save_surface(str(tmp_path / "rh.pial"))
        other = tmp_path / "rh.pial" if as_path else str(tmp_path / "rh.pial")
        assert first.add_surface(other).mesh.n_points == 842 + 98

    def test_add_surface_list(self, two_surfs):
        first, second = two_surfs
        third = copy.deepcopy(second)
        third.apply_cras((100, 0, 0))  # Moved away so that no point is shared
        assert first.add_surface([second, third]).mesh.n_points == 842 + 98 + 98

    def test_add_surface_merges_coincident_points(self, two_surfs):
        first, second = two_surfs
        assert first.add_surface([second, second]).mesh.n_points == 842 + 98

    def test_add_surface_errors(self, tmp_path, two_surfs):
        first, _ = two_surfs
        with pytest.raises(ValueError, match="cannot be empty"):
            first.add_surface([])
        with pytest.raises(TypeError, match="not a Surface"):
            first.add_surface([3])
        with pytest.raises(FileNotFoundError):
            first.add_surface(str(tmp_path / "missing.pial"))

    def test_merge_surfaces(self, two_surfs):
        first, second = two_surfs
        merged = merge_surfaces([first, second])
        assert merged.mesh.n_points == 842 + 98
        assert merged.list_overlays()["surf_id"] == "scalar_with_colortable"
        ids = merged.mesh.point_data["surf_id"]
        assert np.array_equal(np.unique(ids), [1, 2])
        assert (ids[:842] == 1).all() and (ids[842:] == 2).all()
        assert merged.get_overlay_info("surf_id")["num_regions"] == 2

    def test_merge_surfaces_custom_map_name_and_colortable(self, two_surfs):
        ctab = create_surface_colortable(["red", "blue"], struct_names=["L", "R"])
        merged = merge_surfaces(list(two_surfs), color_table=ctab, map_name="hemis")
        assert merged.get_overlay_info("hemis")["region_names"] == ["L", "R"]
        assert "surf_id" not in merged.list_overlays()

    def test_merge_surfaces_from_files(self, tmp_path, two_surfs):
        files = [tmp_path / "lh.pial", tmp_path / "rh.pial"]
        for surf, file in zip(two_surfs, files, strict=True):
            surf.save_surface(str(file))
        merged = merge_surfaces([files[0], str(files[1])])
        assert merged.mesh.n_points == 842 + 98

    def test_merge_surfaces_edge_cases(self, two_surfs):
        first, _ = two_surfs
        assert merge_surfaces([]) is None
        assert merge_surfaces([first]) is first

    def test_merge_surfaces_errors(self, two_surfs):
        first, second = two_surfs
        with pytest.raises(TypeError, match="must be a list"):
            merge_surfaces(first)
        with pytest.raises(TypeError, match="Surface objects"):
            merge_surfaces([first, 3])
        with pytest.raises(TypeError, match="dictionary"):
            merge_surfaces([first, second], color_table=[1, 2])
        with pytest.raises(ValueError, match="must contain the keys"):
            merge_surfaces([first, second], color_table={"names": ["a", "b"]})
        wrong = create_surface_colortable(["red", "green", "blue"])
        with pytest.raises(ValueError, match="must match number of surfaces"):
            merge_surfaces([first, second], color_table=wrong)

    def test_annotations_survive_the_merge(self):
        first = Surface.simulate_surface(seed=1, shape="sphere", maps=None)
        second = Surface.simulate_surface(seed=2, shape="icosphere", maps=None)
        for surf in (first, second):
            surf.load_annotation(
                cltfree.AnnotParcellation.simulate_annotation(surf, n_regions=5),
                parc_name="P",
            )
        merged = merge_surfaces([first, second])
        assert merged.mesh.point_data["P"].shape == (
            first.mesh.n_points + second.mesh.n_points,
        )
        assert merged.list_overlays()["P"] == "scalar_with_colortable"

    def test_separate_mesh_components(self, two_surfs):
        first, second = two_surfs
        parts = merge_surfaces([first, second]).separate_mesh_components()
        assert sorted(p.mesh.n_points for p in parts) == [98, 842]
        by_size = {p.mesh.n_points: p for p in parts}
        assert np.allclose(by_size[842].get_vertices(), first.get_vertices())
        assert np.allclose(by_size[98].get_vertices(), second.get_vertices())
        assert np.allclose(
            by_size[842].mesh.point_data["thickness"],
            first.mesh.point_data["thickness"],
        )
        # Faces are reindexed within every component
        assert by_size[98].get_faces().max() == 97
        for part in parts:
            assert len(np.unique(part.mesh.point_data["components"])) == 1

    def test_separate_a_single_component(self, sphere):
        parts = sphere.separate_mesh_components()
        assert len(parts) == 1 and parts[0].mesh.n_points == N_VERTS

    def test_separate_mesh_components_unknown_labels(self, sphere):
        with pytest.raises(KeyError):
            sphere.separate_mesh_components(component_labels="nope")


####################################################################################################
# Saving and exporting
####################################################################################################
class TestSaveAndExport:
    @pytest.mark.parametrize(
        "name, fmt",
        [
            ("lh.pial", "freesurfer"),
            ("surf.vtk", "vtk"),
            ("surf.ply", "ply"),
            ("surf.stl", "stl"),
            ("surf.obj", "obj"),
        ],
    )
    def test_geometry_roundtrip(self, tmp_path, bare, name, fmt):
        file = tmp_path / name
        bare.save_surface(str(file), format=fmt)
        assert file.is_file()
        loaded = Surface(file)
        assert loaded.mesh.n_points == N_VERTS and loaded.mesh.n_cells == N_FACES
        assert np.allclose(
            np.sort(loaded.get_vertices(), axis=0),
            np.sort(bare.get_vertices(), axis=0),
            atol=1e-4,
        )

    def test_default_format_is_freesurfer(self, tmp_path, bare):
        bare.save_surface(str(tmp_path / "lh.pial"))
        verts, faces = nib.freesurfer.read_geometry(str(tmp_path / "lh.pial"))
        assert np.allclose(verts, bare.get_vertices(), atol=1e-4)
        assert np.array_equal(faces, bare.get_faces())

    def test_extension_is_added_for_pyvista_formats(self, tmp_path, bare):
        bare.save_surface(str(tmp_path / "surface"), format="ply")
        assert (tmp_path / "surface.ply").is_file()

    def test_format_is_case_insensitive(self, tmp_path, bare):
        bare.save_surface(str(tmp_path / "lh.pial"), format="FreeSurfer")
        assert (tmp_path / "lh.pial").is_file()

    @pytest.mark.parametrize("name, fmt", [("p.vtk", "vtk"), ("p.pial", "freesurfer")])
    def test_annotation_saved_with_the_surface(self, tmp_path, parc, annot, name, fmt):
        parc.save_surface(
            str(tmp_path / name),
            format=fmt,
            map_name="Sim",
            save_annotation=str(tmp_path / "p.annot"),
        )
        loaded = Surface(tmp_path / name)
        loaded.load_annotation(str(tmp_path / "p.annot"), parc_name="Back")
        assert loaded.get_overlay_info("Back")["region_names"] == list(annot.regnames)
        assert len(np.unique(loaded.mesh.point_data["Back"])) == 10

    def test_obj_with_annotation(self, tmp_path, parc, annot):
        parc.export_to_obj(
            str(tmp_path / "p.obj"),
            map_name="Sim",
            save_annotation=str(tmp_path / "p.annot"),
        )
        loaded = Surface(tmp_path / "p.obj")
        loaded.load_annotation(str(tmp_path / "p.annot"), parc_name="Back")
        assert loaded.get_overlay_info("Back")["num_regions"] == 10

    def test_export_methods(self, tmp_path, parc):
        parc.export_to_pyvista(str(tmp_path / "a.vtk"))
        parc.export_to_freesurfer(str(tmp_path / "lh.a"))
        parc.export_to_obj(str(tmp_path / "a.obj"))
        for name in ("a.vtk", "lh.a", "a.obj"):
            assert (tmp_path / name).is_file()

    def test_save_errors(self, tmp_path, bare):
        file = tmp_path / "lh.pial"
        bare.save_surface(str(file))
        with pytest.raises(ValueError, match="already exists"):
            bare.save_surface(str(file))
        bare.save_surface(str(file), overwrite=True)  # No error
        with pytest.raises(ValueError, match="Unsupported file format"):
            bare.save_surface(str(tmp_path / "x"), format="zzz")
        with pytest.raises(FileNotFoundError):
            bare.save_surface(str(tmp_path / "no" / "x.pial"))
        with pytest.raises(ValueError, match="cannot be empty"):
            bare.save_surface("")
        with pytest.raises(ValueError, match="must be a string"):
            bare.save_surface(5)

    def test_export_errors(self, tmp_path, bare):
        for method in (
            bare.export_to_pyvista,
            bare.export_to_obj,
            bare.export_to_freesurfer,
        ):
            with pytest.raises(ValueError, match="cannot be empty"):
                method("")
            with pytest.raises(FileNotFoundError):
                method(str(tmp_path / "no" / "x.surf"))
        with pytest.raises(ValueError, match="not found"):
            bare.export_to_pyvista(str(tmp_path / "x.vtk"), map_name="nope")

    def test_export_scalar_map(self, tmp_path, sphere):
        file = sphere.export_map("thickness", str(tmp_path / "lh.thickness"))
        assert Path(file).is_file()
        values = nib.freesurfer.read_morph_data(file)
        assert np.allclose(values, sphere.mesh.point_data["thickness"], atol=1e-5)

    def test_export_parcellation_as_annot(self, tmp_path, parc, annot):
        file = parc.export_map("Sim", str(tmp_path / "lh.sim.annot"))
        assert list(cltfree.AnnotParcellation(file).regnames) == list(annot.regnames)

    def test_export_map_errors(self, tmp_path, sphere):
        file = str(tmp_path / "lh.thickness")
        sphere.export_map("thickness", file)
        with pytest.raises(ValueError, match="already exists"):
            sphere.export_map("thickness", file)
        sphere.export_map("thickness", file, overwrite=True)  # No error
        with pytest.raises(ValueError, match="not found"):
            sphere.export_map("nope", str(tmp_path / "x"))
        with pytest.raises(ValueError, match="overlay_name"):
            sphere.export_map("", str(tmp_path / "x"))
        with pytest.raises(FileNotFoundError):
            sphere.export_map("thickness", str(tmp_path / "no" / "x"))
        with pytest.raises(RuntimeError, match="No surface data loaded"):
            Surface().export_map("thickness", str(tmp_path / "x"))

    def test_roundtrip_of_a_whole_surface(self, tmp_path, sphere, annot):
        """Save geometry, map and annotation, then rebuild the surface from the files."""
        sphere.load_annotation(annot, parc_name="Sim")
        sphere.save_surface(str(tmp_path / "lh.pial"))
        sphere.export_map("thickness", str(tmp_path / "lh.thickness"))
        sphere.export_map("Sim", str(tmp_path / "lh.sim.annot"))

        rebuilt = Surface(tmp_path / "lh.pial")
        rebuilt.load_scalar_maps(str(tmp_path / "lh.thickness"), maps_names="thickness")
        rebuilt.load_annotation(str(tmp_path / "lh.sim.annot"), parc_name="Sim")
        assert np.allclose(
            rebuilt.mesh.point_data["thickness"],
            sphere.mesh.point_data["thickness"],
            atol=1e-5,
        )
        assert np.array_equal(
            rebuilt.mesh.point_data["Sim"], sphere.mesh.point_data["Sim"]
        )


####################################################################################################
# Transformations
####################################################################################################
class TestTransformations:
    AFFINE = np.array(
        [[-1, 0, 0, 133.0], [0, 1, 0, -110.0], [0, 0, 1, -128.0], [0, 0, 0, 1]]
    )

    def test_apply_affine(self, bare):
        original = bare.get_vertices().copy()
        assert bare.apply_affine(self.AFFINE) is bare  # Returns the surface itself
        expected = original * [-1, 1, 1] + [133, -110, -128]
        assert np.allclose(bare.get_vertices(), expected)

    def test_apply_affine_inverse(self, bare):
        original = bare.get_vertices().copy()
        bare.apply_affine(self.AFFINE)
        bare.apply_affine(self.AFFINE, inverse=True)
        assert np.allclose(bare.get_vertices(), original)

    def test_apply_affine_not_in_place_returns_the_coordinates(self, bare):
        original = bare.get_vertices().copy()
        moved = bare.apply_affine(self.AFFINE, inplace=False)
        assert isinstance(moved, np.ndarray) and moved.shape == (N_VERTS, 3)
        assert np.allclose(moved, original * [-1, 1, 1] + [133, -110, -128])
        assert np.array_equal(bare.get_vertices(), original)  # Surface untouched
        back = bare.apply_affine(self.AFFINE, inverse=True, inplace=False)
        assert np.allclose(back, (original - [133, -110, -128]) * [-1, 1, 1])

    def test_apply_affine_keeps_the_topology(self, bare):
        faces = bare.get_faces().copy()
        bare.apply_affine(self.AFFINE)
        assert np.array_equal(bare.get_faces(), faces)

    def test_apply_affine_errors(self, bare):
        with pytest.raises(ValueError, match="4x4"):
            bare.apply_affine(np.eye(3))
        with pytest.raises(ValueError, match="empty"):
            Surface().apply_affine(np.eye(4))

    @pytest.fixture
    def lta(self, tmp_path):
        path = tmp_path / "talairach.lta"
        path.write_text(
            "type      = 1 # LINEAR_RAS_TO_RAS\n"
            "nxforms   = 1\n"
            "mean      = 0.0000 0.0000 0.0000\n"
            "sigma     = 10000.0000\n"
            "1 4 4\n"
            "1.0 0.0 0.0 0.0\n"
            "0.0 1.0 0.0 0.0\n"
            "0.0 0.0 1.0 0.0\n"
            "0.0 0.0 0.0 1.0\n"
            "src volume info\n"
            "valid = 1  # volume info valid\n"
            "filename = orig.mgz\n"
            "volume = 256 256 256\n"
            "voxelsize = 1.0 1.0 1.0\n"
            "xras   = -1 0 0\n"
            "yras   = 0 0 -1\n"
            "zras   = 0 1 0\n"
            "cras   = 5.3 18.2 -7.1\n"
            "\n"
            "dst volume info\n"
            "valid = 1  # volume info valid\n"
            "filename = mni.nii\n"
            "volume = 182 218 182\n"
            "voxelsize = 1.0 1.0 1.0\n"
            "xras   = -1 0 0\n"
            "yras   = 0 0 -1\n"
            "zras   = 0 1 0\n"
            "cras   = 1.0 2.0 3.0\n"
        )
        return path

    def test_apply_cras_from_lta_file(self, bare, lta):
        original = bare.get_vertices().copy()
        assert bare.apply_cras(lta) is bare
        assert np.allclose(bare.get_vertices(), original + [5.3, 18.2, -7.1])

    def test_apply_cras_destination(self, bare, lta):
        original = bare.get_vertices().copy()
        bare.apply_cras(str(lta), source=False)
        assert np.allclose(bare.get_vertices(), original + [1, 2, 3])

    def test_apply_cras_inverse_undoes_it(self, bare, lta):
        original = bare.get_vertices().copy()
        bare.apply_cras(lta)
        bare.apply_cras(lta, inverse=True)
        assert np.allclose(bare.get_vertices(), original)

    @pytest.mark.parametrize("cras", [(1, 2, 3), [1, 2, 3], np.array([1, 2, 3])])
    def test_apply_cras_from_coordinates(self, bare, cras):
        original = bare.get_vertices().copy()
        bare.apply_cras(cras)
        assert np.allclose(bare.get_vertices(), original + [1, 2, 3])

    def test_apply_cras_from_a_4x4_matrix(self, bare):
        original = bare.get_vertices().copy()
        bare.apply_cras(self.AFFINE)
        assert np.allclose(
            bare.get_vertices(), original * [-1, 1, 1] + [133, -110, -128]
        )

    def test_apply_cras_errors(self, tmp_path, bare):
        with pytest.raises(ValueError, match="not found"):
            bare.apply_cras(tmp_path / "missing.lta")
        with pytest.raises(ValueError, match="Invalid CRAS"):
            bare.apply_cras([1, 2])
        with pytest.raises(ValueError, match="empty"):
            Surface().apply_cras((1, 2, 3))


class TestSmooth:
    def test_smooth_in_place(self):
        surf = Surface.simulate_surface("cube", subdivide=3, maps=None)
        verts, faces = surf.get_vertices().copy(), surf.get_faces().copy()
        assert surf.smooth(smooth_iterations=20) is None
        assert surf.get_vertices().shape == verts.shape
        assert not np.allclose(surf.get_vertices(), verts)
        assert np.isfinite(surf.get_vertices()).all()
        assert np.array_equal(surf.get_faces(), faces)  # Same topology

    def test_smooth_copy_leaves_the_original_untouched(self):
        surf = Surface.simulate_surface("cube", subdivide=3, maps=None)
        verts = surf.get_vertices().copy()
        smoothed = surf.smooth(smooth_iterations=20, inplace=False)
        assert smoothed is not surf
        assert np.array_equal(surf.get_vertices(), verts)
        assert not np.allclose(smoothed.get_vertices(), verts)

    def test_more_iterations_smooth_more(self):
        surf = Surface.simulate_surface("cube", subdivide=3, maps=None)
        few = surf.smooth(smooth_iterations=2, inplace=False)
        many = surf.smooth(smooth_iterations=40, inplace=False)

        def shift(other):
            return np.abs(other.get_vertices() - surf.get_vertices()).max()

        assert shift(many) > shift(few) > 0

    @pytest.mark.parametrize("n_iter", [0, -3])
    def test_invalid_iterations(self, bare, n_iter):
        with pytest.raises(ValueError, match="positive integer"):
            bare.smooth(smooth_iterations=n_iter)


####################################################################################################
# Volumes
####################################################################################################
class TestVolumes:
    def test_simulate_image(self, tmp_path, bare):
        out = tmp_path / "image.nii.gz"
        img = bare.simulate_image(str(out), voxel_size=0.1, random_seed=0)
        assert out.is_file() and isinstance(img, nib.Nifti1Image)
        assert np.allclose(img.header.get_zooms()[:3], 0.1)
        # The grid contains the whole surface (bounding box + padding)
        bbox = bare.get_bounding_box()
        corner_low = nib.affines.apply_affine(img.affine, [0, 0, 0])
        corner_high = nib.affines.apply_affine(img.affine, np.array(img.shape[:3]) - 1)
        assert (corner_low < bbox[0]).all() and (corner_high > bbox[1]).all()

    def test_simulate_image_volumes(self, tmp_path, bare):
        img = bare.simulate_image(
            str(tmp_path / "f.nii.gz"), voxel_size=0.1, n_volumes=3
        )
        assert img.shape[3] == 3 and len(img.shape) == 4

    def test_simulate_image_masks(self, tmp_path, bare):
        inside = bare.simulate_image(
            str(tmp_path / "in.nii.gz"), voxel_size=0.1, mask="interior", random_seed=0
        )
        box = bare.simulate_image(
            str(tmp_path / "box.nii.gz"), voxel_size=0.1, mask="box", random_seed=0
        )

        def filled(img):
            return np.count_nonzero(np.asarray(img.dataobj))

        assert 0 < filled(inside) < filled(box)
        assert filled(box) == np.prod(box.shape[:3])

    def test_simulate_image_errors(self, tmp_path, bare):
        out = str(tmp_path / "x.nii.gz")
        with pytest.raises(ValueError, match="mask"):
            bare.simulate_image(out, mask="nope")
        with pytest.raises(ValueError, match="padding"):
            bare.simulate_image(out, padding=-1)
        with pytest.raises(ValueError, match="voxel_size"):
            bare.simulate_image(out, voxel_size=-1)
        with pytest.raises(RuntimeError, match="No surface data loaded"):
            Surface().simulate_image(out)

    @pytest.fixture
    def ramp(self):
        """
        Sphere of radius 10 and a volume holding, in each of its 3 frames, the x, y
        and z world coordinate of every voxel. Linear interpolation of the volume at
        the vertices must therefore return the coordinates of the vertices.
        """
        surf = Surface.simulate_surface("sphere", radius=10, maps=None)
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        affine[:3, 3] = -15
        ijk = np.indices((31, 31, 31)).reshape(3, -1).T
        xyz = nib.affines.apply_affine(affine, ijk).reshape(31, 31, 31, 3)
        return surf, nib.Nifti1Image(xyz.astype(np.float32), affine)

    @pytest.mark.parametrize("input_type", ["image", "str", "path"])
    def test_map_volume_to_surface(self, tmp_path, ramp, input_type):
        surf, img = ramp
        file = tmp_path / "ramp.nii.gz"
        nib.save(img, file)
        image = {"image": img, "str": str(file), "path": file}[input_type]
        values = surf.map_volume_to_surface(image, method="clabtoolkit")
        assert values.shape == (surf.mesh.n_points, 3)
        assert np.allclose(values, surf.get_vertices(), atol=1e-3)

    def test_map_numpy_array_uses_vertices_as_voxel_coordinates(self, ramp):
        surf, _ = ramp
        voxels = np.indices((31, 31, 31)).transpose(1, 2, 3, 0).astype(np.float32)
        surf.apply_cras((15, 15, 15))  # Moves the sphere inside the voxel grid
        values = surf.map_volume_to_surface(voxels, method="clabtoolkit")
        assert values.shape == (surf.mesh.n_points, 3)
        assert np.allclose(values, surf.get_vertices(), atol=1e-3)

    def test_map_volume_respects_the_affine_orientation(self, ramp):
        surf, img = ramp
        flipped_affine = img.affine.copy()
        flipped_affine[0, 0], flipped_affine[0, 3] = -1.0, 15.0  # x axis reversed
        flipped = nib.Nifti1Image(np.asarray(img.dataobj)[::-1], flipped_affine)
        values = surf.map_volume_to_surface(flipped, method="clabtoolkit")
        assert np.allclose(values, surf.get_vertices(), atol=1e-3)

    def test_map_volume_stores_the_overlay(self, ramp):
        surf, img = ramp
        values = surf.map_volume_to_surface(
            img, method="clabtoolkit", overlay_name="ramp"
        )
        assert np.array_equal(surf.mesh.point_data["ramp"], values)

    def test_map_3d_volume(self, ramp):
        surf, img = ramp
        volume = nib.Nifti1Image(np.asarray(img.dataobj)[..., 0], img.affine)
        values = surf.map_volume_to_surface(volume, method="clabtoolkit")
        assert values.shape == (surf.mesh.n_points,)
        assert np.allclose(values, surf.get_vertices()[:, 0], atol=1e-3)

    @pytest.mark.parametrize("interp", ["linear", "nearest", "slinear"])
    def test_map_volume_interpolations(self, ramp, interp):
        surf, img = ramp
        values = surf.map_volume_to_surface(
            img, method="clabtoolkit", interp_method=interp
        )
        assert values.shape == (surf.mesh.n_points, 3) and np.isfinite(values).all()
        # Nearest neighbour can be off by half a voxel at most per axis
        assert np.abs(values - surf.get_vertices()).max() <= 0.51 + 1e-3

    def test_map_constant_volume(self, ramp):
        surf, img = ramp
        const = nib.Nifti1Image(
            np.full(img.shape[:3], 7.0, dtype=np.float32), img.affine
        )
        values = surf.map_volume_to_surface(const, method="clabtoolkit")
        assert np.allclose(values, 7.0)

    def test_map_volume_with_nilearn(self, ramp):
        pytest.importorskip("nilearn")
        surf, img = ramp
        values = surf.map_volume_to_surface(img, method="nilearn")
        assert values.shape == (surf.mesh.n_points, 3)
        assert np.isfinite(values).all()

    def test_map_volume_errors(self, tmp_path, ramp):
        surf, img = ramp
        with pytest.raises(FileNotFoundError):
            surf.map_volume_to_surface(str(tmp_path / "missing.nii.gz"))
        with pytest.raises(ValueError, match="Unknown projection method"):
            surf.map_volume_to_surface(img, method="nope")
        with pytest.raises(ValueError, match="interp_method"):
            surf.map_volume_to_surface(img, method="clabtoolkit", interp_method="nope")
        with pytest.raises(ValueError, match="3D or 4D"):
            surf.map_volume_to_surface(np.zeros((3, 3)), method="clabtoolkit")


####################################################################################################
# Colortables
####################################################################################################
class TestCreateSurfaceColortable:
    def test_from_rgba_array(self):
        colors = np.array([[255, 0, 0, 204], [0, 255, 0, 204], [0, 0, 255, 204]])
        ctab = create_surface_colortable(
            colors, struct_names=["A", "B", "C"], alpha=0.8
        )
        assert ctab["names"] == ["A", "B", "C"] and ctab["lookup_table"] is None
        table = ctab["color_table"]
        assert table.shape == (3, 5)
        assert np.allclose(table[:, :3], np.eye(3))  # RGB scaled to [0, 1]
        assert np.allclose(table[:, 3], 0.8)
        # The last column packs the RGB values: R + 256 * G + 65536 * B
        assert np.allclose(table[:, 4], [255, 255 * 256, 255 * 65536])

    def test_from_color_names_and_hex(self):
        ctab = create_surface_colortable(["red", "#00ff00"], struct_names=["a", "b"])
        assert np.allclose(ctab["color_table"][:, :3], [[1, 0, 0], [0, 1, 0]])
        assert np.allclose(ctab["color_table"][:, 3], 1.0)

    def test_default_names(self):
        ctab = create_surface_colortable(["red", "blue"])
        assert ctab["names"] == ["region_1", "region_2"]

    def test_errors(self):
        with pytest.raises(ValueError, match="Alpha"):
            create_surface_colortable("red", alpha=3)
        with pytest.raises(ValueError, match="Length of struct_names"):
            create_surface_colortable(["red", "green"], struct_names=["a"])
        with pytest.raises(ValueError, match="Length of struct_names"):
            create_surface_colortable("red", struct_names=["a", "b"])
