"""Tests for the helpers of clabtoolkit.visualization_utils.

None of them needs a display: the PyVista plotter is replaced by a recorder.
"""

import json

import nibabel as nib
import numpy as np
import pytest

import clabtoolkit.pointstools as cltpts
import clabtoolkit.surfacetools as cltsurf
import clabtoolkit.tracttools as clttract
import clabtoolkit.visualization_utils as vu
import clabtoolkit.visualizationtools as cltvis


def make_surface(with_labels=False):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    surf = cltsurf.Surface(vertices=vertices, faces=faces, color="#ff0000", name="t")
    surf.mesh.point_data["A"] = np.array([0.0, 1.0, 2.0, 3.0])
    surf.mesh.point_data["B"] = np.array([-3.0, -2.0, 1.0, 4.0])
    if with_labels:
        surf.mesh.point_data["Labels"] = np.array([1, 2, 1, 2])
        surf.colortables["Labels"] = {
            "color_table": cltsurf.create_surface_colortable(["red", "blue"])[
                "color_table"
            ]
        }
    return surf


def make_tractogram():
    rng = np.random.default_rng(0)
    tracts = nib.streamlines.ArraySequence([rng.random((5, 3)), rng.random((7, 3))])
    tract = clttract.Tractogram(
        tracts=tracts,
        affine=np.eye(4),
        header=nib.streamlines.TrkFile.create_empty_header(),
    )
    tract.data_per_point["P"] = nib.streamlines.ArraySequence(
        [np.linspace(0, 1, 5), np.linspace(2, 3, 7)]
    )
    tract.data_per_streamline["S"] = np.array([[10.0], [20.0]])
    return tract


def make_cloud():
    cloud = cltpts.PointCloud(points=np.random.default_rng(0).random((5, 3)))
    cloud.point_data["C"] = np.linspace(5.0, 6.0, 5)
    return cloud


class Recorder:
    """Plotter that records what finalize_plot asks for."""

    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail or []
        self.shape = (1, 3)
        self.renderers = [object(), object(), None]

    def _call(self, name, *args, **kwargs):
        self.calls.append(name)
        if name in self.fail:
            raise RuntimeError(f"{name} failed")
        return None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: self._call(name, *a, **k)


######################################################################################################
class TestFinalizePlot:
    @pytest.mark.parametrize(
        "extension,method",
        [("html", "export_html"), ("svg", "save_graphic"), ("png", "screenshot")],
    )
    def test_saving(self, extension, method, capsys):
        plotter = Recorder()
        vu.finalize_plot(plotter, True, f"/tmp/out.{extension}")
        assert method in plotter.calls and "close" in plotter.calls
        assert "Figure saved to" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "extension,method,message",
        [
            ("htm", "export_html", "Error saving HTML"),
            ("pdf", "save_graphic", "Error saving vector graphic"),
            ("png", "screenshot", "Error saving screenshot"),
        ],
    )
    def test_saving_errors_are_reported(self, extension, method, message, capsys):
        plotter = Recorder(fail=[method])
        vu.finalize_plot(plotter, True, f"/tmp/out.{extension}")
        assert message in capsys.readouterr().out
        assert "close" in plotter.calls  # the plotter is closed anyway

    def test_screenshot_is_retried_returning_the_image(self, capsys):
        class Once(Recorder):
            def screenshot(self, path, return_img=False):
                if not return_img:
                    raise RuntimeError("first attempt")
                return np.zeros((2, 2))

        vu.finalize_plot(Once(), True, "/tmp/out.png")
        assert "alternative method" in capsys.readouterr().out

    def test_screenshot_retry_that_fails_too(self, capsys):
        class Never(Recorder):
            def screenshot(self, path, return_img=False):
                raise RuntimeError("no")

        vu.finalize_plot(Never(), True, "/tmp/out.png")
        assert "Alternative screenshot method also failed" in capsys.readouterr().out

    def test_display(self):
        plotter = Recorder()
        vu.finalize_plot(plotter, False, None)
        assert plotter.calls == ["show"]

    def test_threaded_display(self, capsys):
        plotter = Recorder()
        vu.finalize_plot(plotter, False, None, use_threading=True)
        assert "separate window" in capsys.readouterr().out

    def test_threaded_display_reports_errors(self, capsys):
        import time

        plotter = Recorder(fail=["show"])
        vu.create_threaded_plot(plotter)
        time.sleep(0.2)
        assert "Error displaying plot in thread" in capsys.readouterr().out


class TestRenderMode:
    def test_saving_in_an_existing_folder(self, tmp_path):
        assert vu.determine_render_mode(str(tmp_path / "a.png"), False) == (
            True,
            True,
            False,
            False,
        )

    def test_saving_in_the_current_folder(self):
        assert vu.determine_render_mode("a.png", False)[0] is True

    def test_missing_folder_falls_back_to_display(self, tmp_path, capsys):
        result = vu.determine_render_mode(str(tmp_path / "no" / "a.png"), True, True)
        assert result == (False, False, True, False)
        assert "does not exist" in capsys.readouterr().out

    def test_display_modes(self, capsys):
        assert vu.determine_render_mode(None, False, True) == (
            False,
            False,
            False,
            True,
        )
        assert vu.determine_render_mode(None, True, True) == (False, False, True, False)
        assert "incompatible" in capsys.readouterr().out

    def test_window_size(self, monkeypatch):
        assert list(vu.resolve_window_size((10, 20), False)) == [10, 20]
        monkeypatch.setattr(vu.cltplot, "get_current_monitor_size", lambda: (800, 600))
        assert tuple(vu.resolve_window_size(None, False)) == (800, 600)
        assert len(vu.resolve_window_size(None, True)) == 2


class TestLinkCameras:
    def test_links_the_views_that_share_the_index(self, capsys):
        plotter = Recorder()
        positions = {(0, 0, 0): (0, 0), (0, 1, 0): (0, 1), (0, 0, 1): (0, 2)}
        vu.link_brain_subplot_cameras(plotter, positions)
        assert plotter.calls == ["link_views"]
        assert "Linked 2 views" in capsys.readouterr().out

    def test_invalid_positions_are_skipped_and_reported(self, capsys):
        plotter = Recorder()
        positions = {
            (0, 0, 0): (0, 0),
            (0, 1, 0): (5, 5),  # out of bounds
            (0, 2, 0): (0, 2),  # renderer is None
        }
        vu.link_brain_subplot_cameras(plotter, positions)
        out = capsys.readouterr().out
        assert "out of bounds" in out and "is None" in out
        assert "Not enough valid renderers" in out

    def test_missing_renderer_index(self, capsys):
        plotter = Recorder()
        plotter.shape = (2, 3)
        positions = {(0, 0, 0): (0, 0), (0, 1, 0): (1, 2)}
        vu.link_brain_subplot_cameras(plotter, positions)
        assert ">=" in capsys.readouterr().out

    def test_link_failure_is_reported(self, capsys):
        plotter = Recorder(fail=["link_views"])
        positions = {(0, 0, 0): (0, 0), (0, 1, 0): (0, 1)}
        vu.link_brain_subplot_cameras(plotter, positions)
        assert "Failed to link views" in capsys.readouterr().out


######################################################################################################
class TestLimitsAndData:
    def test_shared_limits_of_surfaces_and_tractograms(self):
        assert vu.get_shared_limits([make_surface()], "A", None, None) == (0.0, 3.0)
        assert vu.get_shared_limits([make_surface()], "A", -1, 9) == (-1, 9)
        tract = make_tractogram()
        assert vu.get_shared_limits([tract], "P", None, None) == (0.0, 3.0)
        assert vu.get_shared_limits([tract], "S", None, None) == (10.0, 20.0)
        with pytest.raises(KeyError):
            vu.get_shared_limits([tract], "none", None, None)

    def test_get_data_from_object(self):
        assert vu.get_data_from_object(make_surface(), "A").tolist() == [0, 1, 2, 3]
        assert len(vu.get_data_from_object(make_cloud(), "C")) == 5
        tract = make_tractogram()
        assert len(vu.get_data_from_object(tract, "P")) == 2
        assert vu.get_data_from_object(tract, "S").shape == (2, 1)
        with pytest.raises(KeyError):
            vu.get_data_from_object(tract, "none")
        with pytest.raises(TypeError):
            vu.get_data_from_object("not an object", "A")

    def test_map_limits(self):
        objs = [make_surface(), make_surface()]
        limits = vu.get_map_limits(objs, "A", (None, None))
        assert limits[0][:2] == (0.0, 3.0) and limits[0][2] == "A"
        limits = vu.get_map_limits(objs, "B", (-1.0, 1.0))
        assert limits[0][:2] == (-1.0, 1.0)
        assert vu.get_map_limits([make_cloud()], "C", (None, None))[0][:2] == (5.0, 6.0)
        assert vu.get_map_limits([make_tractogram()], "P", (None, 2.0))[0][1] == 2.0

    def test_v_limits_processing(self):
        assert vu.process_v_limits((0, 1), 2) == [(0, 1), (0, 1)]
        assert vu.process_v_limits([(0, 1), (1, 2)], 2) == [(0, 1), (1, 2)]

    def test_v_limits_that_are_automatic(self):
        assert vu.process_v_limits(None, 3) == [(None, None)] * 3
        assert vu.process_v_limits((None, None), 2) == [(None, None)] * 2
        assert vu.process_v_limits((None, 5), 2) == [(None, 5)] * 2
        assert vu.process_v_limits([(None, 1), (0, None)], 2) == [(None, 1), (0, None)]

    def test_v_limits_accept_numpy_numbers(self):
        limits = vu.process_v_limits((np.float32(0), np.int64(3)), 1)
        assert limits == [(0, 3)]

    @pytest.mark.parametrize(
        "bad,n_maps,error",
        [
            ([(0, 1)], 2, ValueError),
            ((1, 0), 1, ValueError),
            ((1, 1), 1, ValueError),
            ([(1, 0)], 1, ValueError),
            ([(0, "a")], 1, TypeError),
            (("a", 1), 1, TypeError),
            ((0, 1, 2), 1, TypeError),
            ([(0, 1), (0, 1, 2)], 2, TypeError),
            ([[0, 1]], 1, TypeError),
            ("x", 1, TypeError),
            (5, 1, TypeError),
        ],
    )
    def test_invalid_v_limits(self, bad, n_maps, error):
        with pytest.raises(error):
            vu.process_v_limits(bad, n_maps)


class TestMapNames:
    def test_colorbar_needed(self):
        assert vu.colorbar_needed(["A"], [])  # no objects
        assert vu.colorbar_needed(["A"], [make_surface()])
        assert not vu.colorbar_needed(["Labels"], [make_surface(with_labels=True)])
        assert vu.colorbar_needed(["Labels", "A"], [make_surface(with_labels=True)])

    def test_common_map_names(self):
        surf = make_surface(with_labels=True)
        no_ctab, names = vu.find_common_map_names([surf, make_surface()], ["A", "Zz"])
        assert names == ["A"] and no_ctab == ["A"]
        no_ctab, names = vu.find_common_map_names([make_tractogram()], ["P", "S"])
        assert names == ["P", "S"]
        no_ctab, names = vu.find_common_map_names([make_cloud()], ["C"])
        assert names == ["C"]

    def test_plotting_parameters_of_each_map(self):
        params = vu.prepare_map_plotting_params(
            ["A", "B"],
            colormaps=["viridis"],
            v_limits=[(0.0, 1.0), (None, None)],
            v_range=(0.2, 0.8),
            colorbar_titles="Title",
        )
        assert params["A"]["vmin"] == 0.0 and params["B"]["range_min"] == 0.2
        assert params["B"]["colormap"] == "viridis"
        assert params["A"]["colorbar_title"] == "Title"
        with pytest.raises(ValueError):
            vu.prepare_map_plotting_params(
                ["A"], "viridis", (None, None), (1.0, 0.0), colorbar_titles=None
            )
        with pytest.raises(ValueError):
            vu.prepare_map_plotting_params(
                ["A"], "viridis", (1.0, 0.0), (None, None), colorbar_titles=None
            )


class TestObjectAndMapConfig:
    def test_default_object_config(self):
        config = vu.create_default_object_config(
            [make_surface(), make_tractogram(), make_cloud()]
        )
        assert len(config) == 3
        assert {"A", "B"} <= set(config[0]) and {"P", "S"} <= set(config[1])
        assert config[2]["C"]["v_limits"] == (5.0, 6.0)

    def test_default_map_config(self):
        config = vu.create_default_map_config(
            [make_surface(with_labels=True), make_surface()]
        )
        assert {"A", "B", "Labels"} <= set(config)
        assert config["Labels"]["colormap"] == "colortable"
        assert config["B"]["colormap"] != config["A"]["colormap"]  # diverging vs not

    def test_final_object_config(self):
        objs = [make_surface(), make_surface(), make_cloud()]
        by_index = vu.create_final_object_config(
            objs,
            maps_config={
                0: {"map_name": "A", "opacity": 0.5, "colormap": "viridis"},
                1: {"map_name": "B", "v_range": (-1.0, 2.0)},
                2: {"map_name": "C", "v_limits": (5.0, 7.0)},
            },
        )
        assert by_index[0]["opacity"] == 0.5 and by_index[0]["colormap"] == "viridis"
        assert by_index[1]["v_range"][0] == -1.0
        assert by_index[2]["v_limits"] == (5.0, 7.0)
        assert vu.create_final_object_config([make_surface()], maps_config=None)

    def test_final_object_config_by_map_name(self):
        objs = [make_surface(), make_surface()]
        config = vu.create_final_object_config(
            objs, maps_config={"A": {"objects": [0, 1], "v_range": (0.5, 2.5)}}
        )
        assert config[0]["v_range"] == config[1]["v_range"] == (0.5, 2.5)

    def test_final_object_config_by_map_name_warnings(self):
        objs = [make_surface(with_labels=True)]
        with pytest.warns(UserWarning, match="invalid object indices"):
            vu.create_final_object_config(objs, maps_config={"A": {"objects": [0, 7]}})
        with pytest.warns(UserWarning, match="No valid object indices"):
            vu.create_final_object_config(objs, maps_config={"A": {"objects": [9]}})
        with pytest.warns(UserWarning, match="colortable"):
            vu.create_final_object_config(
                objs, maps_config={"Labels": {"objects": [0]}}
            )


class TestConfigurationHelpers:
    def test_list_themes_and_options(self, capsys):
        plotter = cltvis.BrainPlotter()
        vu.list_available_themes(plotter)
        plotter.list_available_themes()
        plotter.list_figure_config_options()
        assert capsys.readouterr().out

    def test_save_reload_and_reset(self, tmp_path, capsys):
        config = tmp_path / "conf.json"
        plotter = cltvis.BrainPlotter()
        plotter.update_figure_config(title_font_size=17)
        plotter.save_config(config)
        assert json.loads(config.read_text())["figure_conf"]["title_font_size"] == 17
        other = cltvis.BrainPlotter(config)
        assert other.figure_conf["title_font_size"] == 17
        data = json.loads(config.read_text())
        data["figure_conf"]["title_font_size"] = 9
        config.write_text(json.dumps(data))
        other.reload_config()
        assert other.figure_conf["title_font_size"] == 9
        other.reset_figure_config()
        assert other.figure_conf["title_font_size"] == 11  # the 'default' theme
        other.reset_figure_config(auto_save=True)
        saved = json.loads(config.read_text())
        assert saved["figure_conf"]["title_font_size"] == 11

    def test_reload_picks_up_every_section(self, tmp_path, capsys):
        config = tmp_path / "conf.json"
        plotter = cltvis.BrainPlotter()
        plotter.save_config(config)
        other = cltvis.BrainPlotter(config)
        data = json.loads(config.read_text())
        data["objs_conf"]["tracts"]["tube_radius"] = 0.77
        data["themes_conf"]["mine"] = {"description": "mine"}
        config.write_text(json.dumps(data))
        other.reload_config()
        assert other.objs_conf["tracts"]["tube_radius"] == 0.77
        assert "mine" in other.themes_conf
        assert "Reloading configuration" in capsys.readouterr().out

    def test_reload_of_a_file_without_objects_section(self, tmp_path):
        config = tmp_path / "conf.json"
        plotter = cltvis.BrainPlotter()
        plotter.save_config(config)
        other = cltvis.BrainPlotter(config)
        data = json.loads(config.read_text())
        del data["objs_conf"]
        config.write_text(json.dumps(data))
        before = other.objs_conf
        other.reload_config()
        assert other.objs_conf == before

    def test_reload_errors(self, tmp_path):
        config = tmp_path / "conf.json"
        plotter = cltvis.BrainPlotter()
        plotter.save_config(config)
        other = cltvis.BrainPlotter(config)
        config.write_text("{not json")
        with pytest.raises(ValueError, match="loading the JSON file"):
            other.reload_config()
        config.write_text("{}")
        with pytest.raises(KeyError):
            other.reload_config()
        config.unlink()
        with pytest.raises(FileNotFoundError):
            other.reload_config()

    def test_reset_needs_a_default_theme(self):
        plotter = cltvis.BrainPlotter()
        plotter.themes_conf = {}
        with pytest.raises(KeyError, match="default"):
            plotter.reset_figure_config()
