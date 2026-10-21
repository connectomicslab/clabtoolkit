"""Tests for the visualizationtools module.

Only the parts that do not need a display or a plotting backend are tested: the
configuration, themes, view and layout lookup, titles, colors and the dispatch of the
objects to the plotter. The PyVista plotter is replaced by a fake that records the calls.
plot, plot_hemispheres, plot_scene, the colorbar rendering and create_carpet_plot are
not covered here.
"""

import json

import nibabel as nib
import numpy as np
import pytest

import clabtoolkit.pointstools as cltpts
import clabtoolkit.surfacetools as cltsurf
import clabtoolkit.tracttools as clttract
import clabtoolkit.visualizationtools as cltvis

ALL_LAYOUTS = [
    "8_views",
    "8_views_8x1",
    "8_views_1x8",
    "6_views",
    "6_views_6x1",
    "6_views_1x6",
    "4_views",
    "4_views_4x1",
    "4_views_1x4",
    "2_views",
]
SINGLE_VIEWS = ["lateral", "medial", "dorsal", "ventral", "rostral", "caudal"]


class FakeCamera:
    def __init__(self):
        self.azimuth = 0
        self.elevation = 0
        self.zoom_factor = None

    def zoom(self, factor):
        self.zoom_factor = factor


class FakePlotter:
    """Records what the BrainPlotter sends to a pyvista.Plotter."""

    def __init__(self):
        self.calls = []
        self.camera = FakeCamera()
        self.camera_position = None

    def add_mesh(self, *args, **kwargs):
        self.calls.append(("mesh", args, kwargs))

    def add_points(self, *args, **kwargs):
        self.calls.append(("points", args, kwargs))

    def subplot(self, *args):
        self.calls.append(("subplot", args, {}))


@pytest.fixture
def plotter():
    return cltvis.BrainPlotter()


@pytest.fixture
def fake():
    return FakePlotter()


@pytest.fixture
def saved_plotter(tmp_path):
    """BrainPlotter whose configuration lives in a temporary file."""
    config = tmp_path / "viz_config.json"
    cltvis.BrainPlotter().save_config(config)
    return cltvis.BrainPlotter(config), config


@pytest.fixture
def surface():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    surf = cltsurf.Surface(vertices=vertices, faces=faces, color="#ff0000", name="t")
    surf.mesh.point_data["rgba"] = np.tile([1.0, 0.0, 0.0, 1.0], (4, 1))
    return surf


######################################################################################################
class TestInit:
    def test_default_configuration(self, plotter):
        for section in (
            "figure_conf",
            "views_conf",
            "layouts_conf",
            "objs_conf",
            "themes_conf",
        ):
            assert isinstance(getattr(plotter, section), dict)
        assert plotter.config_file.endswith("viz_views.json")

    def test_configuration_from_dict_overrides_defaults(self):
        plot = cltvis.BrainPlotter({"figure_conf": {"title_font_size": 20}})
        assert plot.figure_conf["title_font_size"] == 20
        assert "background_color" in plot.figure_conf  # the other defaults are kept

    def test_input_dict_is_not_modified(self):
        conf = {"figure_conf": {"title_font_size": 20}}
        cltvis.BrainPlotter(conf)
        assert conf == {"figure_conf": {"title_font_size": 20}}

    def test_configuration_from_file(self, saved_plotter):
        plot, config = saved_plotter
        assert str(plot.config_file) == str(config)

    def test_invalid_configuration_is_reported_and_defaults_are_used(self, capsys):
        plot = cltvis.BrainPlotter(5)
        assert "Error loading configuration file" in capsys.readouterr().out
        assert plot.figure_conf["title_font_size"] == 11

    def test_missing_configuration_file_is_reported(self, tmp_path, capsys):
        cltvis.BrainPlotter(tmp_path / "missing.json")
        assert "Error loading configuration file" in capsys.readouterr().out

    def test_instances_do_not_share_configuration(self):
        a, b = cltvis.BrainPlotter(), cltvis.BrainPlotter()
        a.update_figure_config(title_font_size=20)
        assert b.figure_conf["title_font_size"] == 11

    def test_update_configs_merges_known_sections(self, plotter, capsys):
        plotter._update_configs(
            {"figure_conf": {"title_font_size": 7}, "unknown_section": {}}
        )
        assert plotter.figure_conf["title_font_size"] == 7
        assert "unknown_section" in capsys.readouterr().out


######################################################################################################
class TestViewsAndLayouts:
    def test_view_names(self, plotter, capsys):
        assert plotter.list_available_view_names() == [
            "Lateral",
            "Medial",
            "Dorsal",
            "Ventral",
            "Rostral",
            "Caudal",
        ]
        assert "Lateral" in capsys.readouterr().out

    def test_layouts(self, plotter, capsys):
        layouts = plotter.list_available_layouts()
        assert list(layouts) == ALL_LAYOUTS
        assert layouts["8_views"]["shape"] == [2, 4]
        assert layouts["8_views"]["num_views"] == 8
        assert len(layouts["8_views"]["views"]) == 8
        assert "Available Brain Visualization Layouts" in capsys.readouterr().out

    def test_layout_details(self, plotter, capsys):
        details = plotter.get_layout_details("4_views")
        assert details["shape"] == [2, 2]
        assert details["views"] == [
            "lh-lateral",
            "rh-lateral",
            "lh-medial",
            "rh-medial",
        ]

    def test_unknown_layout_details(self, plotter, capsys):
        assert plotter.get_layout_details("nope") is None

    def test_multiview_layouts(self, plotter):
        assert plotter._list_multiviews_layouts() == ALL_LAYOUTS

    def test_single_views(self, plotter):
        assert plotter._list_single_views() == SINGLE_VIEWS

    def test_all_views_and_layouts(self, plotter):
        assert plotter._list_all_views_and_layouts() == ALL_LAYOUTS + SINGLE_VIEWS

    def test_valid_views(self, plotter):
        assert plotter._get_valid_views("8_views") == ["8_views"]
        assert plotter._get_valid_views(["lateral", "medial"]) == ["lateral", "medial"]

    def test_invalid_views(self, plotter):
        with pytest.raises(ValueError, match="No valid views"):
            plotter._get_valid_views("bogus")

    def test_views_configuration(self, plotter):
        assert sorted(plotter.views_conf) == sorted(
            f"{hemi}-{view}" for hemi in ("lh", "rh") for view in SINGLE_VIEWS
        )
        for camera in plotter.views_conf.values():
            assert {"title", "view", "azimuth", "elevation", "zoom"} <= set(camera)

    def test_every_layout_view_has_a_camera(self, plotter):
        for layout in plotter.layouts_conf.values():
            for view_id in layout["views"]:
                assert view_id.replace("merg", "lh") in plotter.views_conf


class TestRemoveDuplicatedViews:
    def test_dorsal_views_share_the_camera(self, plotter):
        views = ["lh-dorsal", "rh-dorsal", "lh-lateral", "rh-lateral"]
        assert plotter._remove_duplicated_views(views) == [
            "lh-dorsal",
            "lh-lateral",
            "rh-lateral",
        ]

    def test_merged_views_use_the_left_camera(self, plotter):
        assert plotter._remove_duplicated_views(["merg-dorsal", "lh-dorsal"]) == [
            "merg-dorsal"
        ]

    def test_order_is_kept(self, plotter):
        views = ["rh-medial", "lh-lateral", "merg-caudal"]
        assert plotter._remove_duplicated_views(views) == views

    def test_empty(self, plotter):
        assert plotter._remove_duplicated_views([]) == []


class TestTitles:
    @pytest.mark.parametrize(
        "view_id, title",
        [
            ("lh-lateral", "Left hemisphere: Lateral view"),
            ("rh-medial", "Right hemisphere: Medial view"),
            ("merg-dorsal", "Dorsal view"),
            ("dorsal", "Dorsal"),
        ],
    )
    def test_subplot_title(self, view_id, title):
        assert cltvis.BrainPlotter._build_subplot_title(view_id) == title

    def test_scene_titles_mention_the_side_only_when_repeated(self):
        titles = cltvis.BrainPlotter._build_scene_titles(
            ["lh-lateral", "rh-lateral", "lh-dorsal", "lh-medial", "rh-medial"]
        )
        assert titles == [
            "Left lateral view",
            "Right lateral view",
            "Dorsal view",
            "Left medial view",
            "Right medial view",
        ]

    def test_scene_titles_without_hemisphere(self):
        assert cltvis.BrainPlotter._build_scene_titles(["dorsal", "merg-ventral"]) == [
            "Dorsal view",
            "Ventral view",
        ]

    def test_scene_titles_empty(self):
        assert cltvis.BrainPlotter._build_scene_titles([]) == []


######################################################################################################
class TestFigureConfiguration:
    def test_get_figure_config(self, plotter):
        conf = plotter.get_figure_config()
        assert conf["background_color"] == "white"
        assert conf["title_font_size"] == 11

    def test_update_figure_config(self, plotter, capsys):
        plotter.update_figure_config(background_color="black", title_font_size=14)
        assert plotter.figure_conf["background_color"] == "black"
        assert plotter.figure_conf["title_font_size"] == 14
        assert "title_font_size: 11 → 14" in capsys.readouterr().out

    def test_update_does_not_save_by_default(self, saved_plotter):
        plot, config = saved_plotter
        plot.update_figure_config(title_font_size=9)
        assert json.loads(config.read_text())["figure_conf"]["title_font_size"] == 11

    def test_update_with_auto_save(self, saved_plotter):
        plot, config = saved_plotter
        plot.update_figure_config(auto_save=True, title_font_size=9)
        assert json.loads(config.read_text())["figure_conf"]["title_font_size"] == 9

    def test_unknown_parameter(self, plotter):
        with pytest.raises(ValueError, match="Invalid parameter 'bogus_param'"):
            plotter.update_figure_config(bogus_param=1)

    @pytest.mark.parametrize(
        "kwargs",
        [{"title_font_size": 1000}, {"title_font_size": 2}, {"mesh_ambient": 5.0}],
    )
    def test_values_out_of_range(self, plotter, kwargs):
        with pytest.raises(ValueError, match="must be between"):
            plotter.update_figure_config(**kwargs)

    @pytest.mark.parametrize(
        "kwargs", [{"title_font_size": "big"}, {"title_shadow": "yes"}]
    )
    def test_values_of_wrong_type(self, plotter, kwargs):
        with pytest.raises(TypeError, match="must be of type"):
            plotter.update_figure_config(**kwargs)

    @pytest.mark.parametrize(
        "kwargs, error",
        [
            ({"background_color": "black", "title_font_size": 1000}, ValueError),
            ({"background_color": "black", "title_font_size": "big"}, TypeError),
            ({"background_color": "black", "bogus_param": 1}, ValueError),
            ({"title_font_size": 1000, "background_color": "black"}, ValueError),
        ],
    )
    def test_failed_update_leaves_the_configuration_unchanged(
        self, plotter, kwargs, error
    ):
        before = dict(plotter.figure_conf)
        with pytest.raises(error):
            plotter.update_figure_config(**kwargs)
        assert plotter.figure_conf == before

    def test_list_options(self, plotter, capsys):
        plotter.list_figure_config_options()
        assert "title_font_size" in capsys.readouterr().out


class TestThemes:
    def test_available_themes_listing(self, plotter, capsys):
        plotter.list_available_themes()
        out = capsys.readouterr().out
        for theme in ("dark", "light", "high_contrast", "minimal", "publication"):
            assert theme in out

    def test_theme_configuration(self, plotter):
        assert list(plotter.themes_conf) == [
            "default",
            "dark",
            "light",
            "high_contrast",
            "minimal",
            "publication",
            "colorful",
        ]

    def test_apply_theme(self, plotter):
        assert plotter.figure_conf["background_color"] == "white"
        plotter.apply_theme("dark")
        for key, value in plotter.themes_conf["dark"].items():
            if key != "description":
                assert plotter.figure_conf[key] == value
        assert plotter.figure_conf["background_color"] != "white"

    def test_apply_theme_does_not_save_by_default(self, saved_plotter):
        plot, config = saved_plotter
        plot.apply_theme("dark")
        saved = json.loads(config.read_text())["figure_conf"]["background_color"]
        assert saved == "white"

    def test_apply_theme_with_auto_save(self, saved_plotter):
        plot, config = saved_plotter
        plot.apply_theme("dark", auto_save=True)
        saved = json.loads(config.read_text())["figure_conf"]["background_color"]
        assert saved == plot.figure_conf["background_color"] != "white"

    def test_unknown_theme(self, plotter):
        with pytest.raises(ValueError, match="not recognized"):
            plotter.apply_theme("nope")

    def test_preview_does_not_apply(self, plotter, capsys):
        before = dict(plotter.figure_conf)
        plotter.preview_theme("dark")
        assert plotter.figure_conf == before
        assert "plotter.apply_theme('dark')" in capsys.readouterr().out

    def test_preview_unknown_theme(self, plotter):
        with pytest.raises(ValueError, match="not found"):
            plotter.preview_theme("nope")

    def test_reset_figure_config(self, plotter):
        plotter.apply_theme("dark")
        plotter.update_figure_config(title_font_size=20)
        plotter.reset_figure_config(auto_save=False)
        default = plotter.themes_conf["default"]
        for key, value in default.items():
            if key != "description":
                assert plotter.figure_conf[key] == value


class TestSaveConfig:
    def test_save_to_explicit_path(self, plotter, tmp_path):
        out = tmp_path / "custom.json"
        plotter.save_config(out)
        assert str(plotter.config_file) == str(out)
        assert sorted(json.loads(out.read_text())) == [
            "figure_conf",
            "layouts_conf",
            "objs_conf",
            "themes_conf",
            "views_conf",
        ]

    def test_round_trip(self, plotter, tmp_path):
        plotter.update_figure_config(title_font_size=18, background_color="gray")
        out = tmp_path / "custom.json"
        plotter.save_config(str(out))
        loaded = cltvis.BrainPlotter(out)
        assert loaded.figure_conf["title_font_size"] == 18
        assert loaded.figure_conf["background_color"] == "gray"
        assert loaded.views_conf == plotter.views_conf

    def test_save_without_path_reuses_the_configuration_file(self, saved_plotter):
        plot, config = saved_plotter
        plot.update_figure_config(title_font_size=12)
        plot.save_config()
        assert json.loads(config.read_text())["figure_conf"]["title_font_size"] == 12


######################################################################################################
class TestPrepareRgba:
    rgba = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.5]])

    def test_unchanged_by_default(self):
        out = cltvis.BrainPlotter._prepare_rgba(self.rgba)
        assert np.array_equal(out, self.rgba)

    def test_alpha_is_removed_without_opacity(self):
        out = cltvis.BrainPlotter._prepare_rgba(self.rgba, use_opacity=False)
        assert out.shape == (2, 3)

    def test_constant_opacity(self):
        out = cltvis.BrainPlotter._prepare_rgba(self.rgba, opacity=0.3)
        assert np.allclose(out[:, 3], 0.3)
        assert np.array_equal(out[:, :3], self.rgba[:, :3])

    def test_opacity_in_the_0_255_range(self):
        rgba = np.array([[255, 0, 0, 255]], dtype=np.uint8)
        out = cltvis.BrainPlotter._prepare_rgba(rgba, opacity=0.5)
        assert out.dtype == np.uint8
        assert out[0, 3] == 127

    def test_rgb_input_gets_an_alpha_channel(self):
        out = cltvis.BrainPlotter._prepare_rgba(
            np.array([[1.0, 0.0, 0.0]]), opacity=0.3
        )
        assert out.shape == (1, 4)
        assert out[0, 3] == pytest.approx(0.3)

    def test_input_is_not_modified(self):
        before = self.rgba.copy()
        cltvis.BrainPlotter._prepare_rgba(self.rgba, opacity=0.1)
        assert np.array_equal(self.rgba, before)


class TestSubplotCamera:
    def test_camera_parameters(self, plotter, fake):
        plotter._set_subplot_camera(fake, "lh-lateral")
        camera = plotter.views_conf["lh-lateral"]
        assert fake.camera_position == camera["view"]
        assert fake.camera.azimuth == camera["azimuth"]
        assert fake.camera.elevation == camera["elevation"]
        assert fake.camera.zoom_factor == camera["zoom"]

    def test_merged_views_use_the_left_hemisphere_camera(self, plotter, fake):
        plotter._set_subplot_camera(fake, "merg-dorsal")
        assert fake.camera_position == plotter.views_conf["lh-dorsal"]["view"]
        assert fake.camera.zoom_factor == plotter.views_conf["lh-dorsal"]["zoom"]

    def test_unknown_view(self, plotter, fake):
        with pytest.raises(KeyError):
            plotter._set_subplot_camera(fake, "lh-bogus")


class TestAddObjectToPlotter:
    def test_surface(self, plotter, fake, surface):
        plotter._add_object_to_plotter(fake, surface)
        kind, args, kwargs = fake.calls[0]
        assert kind == "mesh"
        assert kwargs["scalars"] == "rgba" and kwargs["rgb"] is True
        assert kwargs["show_scalar_bar"] is False
        assert kwargs["ambient"] == plotter.figure_conf["mesh_ambient"]
        assert kwargs["show_edges"] == plotter.objs_conf["surface"]["show_edges"]
        assert kwargs["edge_color"] == (0, 0, 0)

    def test_surface_opacity_does_not_modify_the_object(self, plotter, fake, surface):
        plotter._add_object_to_plotter(fake, surface, opacity=0.5)
        sent = fake.calls[0][1][0].point_data["rgba"]
        assert np.allclose(sent[:, 3], 0.5)
        assert np.allclose(surface.mesh.point_data["rgba"][:, 3], 1.0)

    def test_surface_without_opacity(self, plotter, fake, surface):
        plotter._add_object_to_plotter(fake, surface, use_opacity=False)
        assert fake.calls[0][1][0].point_data["rgba"].shape == (4, 3)

    def test_point_cloud(self, plotter, fake):
        cloud = cltpts.PointCloud(points=np.random.default_rng(0).random((5, 3)))
        cloud.point_data["rgba"] = np.tile([0.0, 1.0, 0.0, 1.0], (5, 1))
        plotter._add_object_to_plotter(fake, cloud, opacity=0.4)
        kind, args, kwargs = fake.calls[0]
        assert kind == "points"
        assert kwargs["point_size"] == plotter.objs_conf["points"]["spheres_radius"]
        assert kwargs["render_points_as_spheres"] is True
        assert np.allclose(kwargs["scalars"][:, 3], 0.4)

    @pytest.fixture
    def tractogram(self):
        rng = np.random.default_rng(0)
        tracts = nib.streamlines.ArraySequence([rng.random((5, 3)), rng.random((7, 3))])
        tract = clttract.Tractogram(
            tracts=tracts,
            affine=np.eye(4),
            header=nib.streamlines.TrkFile.create_empty_header(),
        )
        tract.data_per_point["rgba"] = nib.streamlines.ArraySequence(
            [
                np.tile([1.0, 0.0, 0.0, 1.0], (5, 1)),
                np.tile([1.0, 0.0, 0.0, 1.0], (7, 1)),
            ]
        )
        return tract

    def test_tractogram_as_lines(self, plotter, fake, tractogram):
        plotter.objs_conf["tracts"]["tubes"] = False
        plotter._add_object_to_plotter(fake, tractogram)
        kind, args, kwargs = fake.calls[0]
        poly = args[0]
        assert kind == "mesh" and kwargs["line_width"] == 2
        assert poly.n_points == 12
        # [n_points, idx...] for each streamline
        assert poly.lines.tolist() == [5, 0, 1, 2, 3, 4, 7, 5, 6, 7, 8, 9, 10, 11]

    def test_tractogram_as_tubes(self, plotter, fake, tractogram):
        plotter.objs_conf["tracts"]["tubes"] = True
        plotter._add_object_to_plotter(fake, tractogram)
        poly = fake.calls[0][1][0]
        assert poly.n_points > 12
        assert "rgba" in poly.point_data

    def test_unsupported_object(self, plotter, fake):
        with pytest.raises(TypeError, match="got str"):
            plotter._add_object_to_plotter(fake, "not an object")
        assert fake.calls == []


class TestAddColorbars:
    def test_nothing_to_add(self, plotter, fake):
        plotter._add_colorbars(fake, None)
        plotter._add_colorbars(fake, [])
        assert fake.calls == []

    def test_false_entries_are_skipped(self, plotter, fake):
        plotter._add_colorbars(fake, [False, False])
        assert fake.calls == []

    def test_categorical_maps_have_no_colorbar(self, plotter, fake):
        plotter._add_colorbars(fake, [{"position": (0, 1), "colormap": "colortable"}])
        assert [call[0] for call in fake.calls] == ["subplot"]
        assert fake.calls[0][1] == (0, 1)
