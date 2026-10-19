"""Tests for clabtoolkit.plottools.

None of the tests needs a graphical display: tkinter and screeninfo are replaced by
fakes, and the few figures that are created use the Agg backend.
"""

import sys
import types
import warnings
from collections import namedtuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import clabtoolkit.plottools as cltplot

FULL_HD = {"screen_width": 1920, "screen_height": 1080}
Monitor = namedtuple("Monitor", "x y width height is_primary")


####################################################################################################
# Helpers and fixtures
####################################################################################################
class NoDisplayError(Exception):
    """Stands for the tkinter.TclError of a machine without a display."""


@pytest.fixture(autouse=True)
def _quiet():
    """Silence the warnings that matplotlib emits when it lays out the figures."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture
def agg_backend():
    """Draw on the Agg backend, which does not need a display, and close the figures."""
    previous = matplotlib.get_backend()
    plt.switch_backend("Agg")
    yield
    plt.close("all")
    plt.switch_backend(previous)


def install_fake_tkinter(monkeypatch, size=(1920, 1080), pointer=(0, 0), error=None):
    """Replace tkinter by a module whose Tk window reports the given screen."""
    state = types.SimpleNamespace(roots=[])

    class Tk:
        def __init__(self):
            if error is not None:
                raise error
            self.withdrawn = False
            self.destroyed = False
            state.roots.append(self)

        def withdraw(self):
            self.withdrawn = True

        def winfo_screenwidth(self):
            return size[0]

        def winfo_screenheight(self):
            return size[1]

        def winfo_pointerx(self):
            return pointer[0]

        def winfo_pointery(self):
            return pointer[1]

        def destroy(self):
            self.destroyed = True

    module = types.ModuleType("tkinter")
    module.Tk = Tk
    monkeypatch.setitem(sys.modules, "tkinter", module)
    return state


def install_fake_screeninfo(monkeypatch, monitors):
    module = types.ModuleType("screeninfo")
    module.get_monitors = lambda: list(monitors)
    monkeypatch.setitem(sys.modules, "screeninfo", module)


@pytest.fixture
def no_display(monkeypatch):
    """Screen detection fails, as on a machine without a graphical display."""

    def fail(*args, **kwargs):
        raise NoDisplayError("no display name and no $DISPLAY environment variable")

    monkeypatch.setattr(cltplot, "get_screen_size", fail)
    monkeypatch.setattr(cltplot, "get_current_monitor_size", fail)


def spy_on_detection(monkeypatch, size=(1920, 1080)):
    """Count how many times the screen is detected."""
    calls = []

    def detect():
        calls.append(1)
        return size

    monkeypatch.setattr(cltplot, "get_screen_size", detect)
    monkeypatch.setattr(cltplot, "get_current_monitor_size", detect)
    return calls


####################################################################################################
class TestGetScreenSize:
    def test_size_of_the_screen(self, monkeypatch):
        install_fake_tkinter(monkeypatch, size=(2560, 1440))
        assert cltplot.get_screen_size() == (2560, 1440)

    def test_window_is_hidden_and_destroyed(self, monkeypatch):
        state = install_fake_tkinter(monkeypatch)
        cltplot.get_screen_size()
        assert len(state.roots) == 1
        assert state.roots[0].withdrawn
        assert state.roots[0].destroyed

    def test_window_is_destroyed_when_the_query_fails(self, monkeypatch):
        state = install_fake_tkinter(monkeypatch)
        monkeypatch.setattr(
            sys.modules["tkinter"].Tk,
            "winfo_screenwidth",
            lambda self: (_ for _ in ()).throw(NoDisplayError("boom")),
        )
        with pytest.raises(NoDisplayError):
            cltplot.get_screen_size()
        assert state.roots[0].destroyed

    def test_no_display_is_reported(self, monkeypatch):
        install_fake_tkinter(monkeypatch, error=NoDisplayError("no display"))
        with pytest.raises(NoDisplayError):
            cltplot.get_screen_size()


####################################################################################################
class TestGetCurrentMonitorSize:
    LEFT = Monitor(0, 0, 1920, 1080, True)
    RIGHT = Monitor(1920, 0, 2560, 1440, False)

    def test_monitor_with_the_mouse(self, monkeypatch):
        install_fake_tkinter(monkeypatch, pointer=(2500, 300))
        install_fake_screeninfo(monkeypatch, [self.LEFT, self.RIGHT])
        assert cltplot.get_current_monitor_size() == (2560, 1440)

    def test_primary_monitor_with_the_mouse(self, monkeypatch):
        install_fake_tkinter(monkeypatch, pointer=(100, 100))
        install_fake_screeninfo(monkeypatch, [self.RIGHT, self.LEFT])
        assert cltplot.get_current_monitor_size() == (1920, 1080)

    def test_monitor_edges_belong_to_the_next_monitor(self, monkeypatch):
        install_fake_screeninfo(monkeypatch, [self.LEFT, self.RIGHT])
        install_fake_tkinter(monkeypatch, pointer=(1919, 0))
        assert cltplot.get_current_monitor_size() == (1920, 1080)
        install_fake_tkinter(monkeypatch, pointer=(1920, 0))
        assert cltplot.get_current_monitor_size() == (2560, 1440)

    def test_monitor_to_the_left_of_the_primary(self, monkeypatch):
        left = Monitor(-1366, 0, 1366, 768, False)
        install_fake_tkinter(monkeypatch, pointer=(-10, 5))
        install_fake_screeninfo(monkeypatch, [self.LEFT, left])
        assert cltplot.get_current_monitor_size() == (1366, 768)

    def test_monitor_above_the_primary(self, monkeypatch):
        above = Monitor(0, -1080, 1920, 1080, False)
        small = Monitor(0, 0, 1280, 720, True)
        install_fake_tkinter(monkeypatch, pointer=(10, -5))
        install_fake_screeninfo(monkeypatch, [small, above])
        assert cltplot.get_current_monitor_size() == (1920, 1080)

    def test_mouse_outside_the_monitors_falls_back_to_the_primary(self, monkeypatch):
        install_fake_tkinter(monkeypatch, pointer=(9000, 9000))
        install_fake_screeninfo(monkeypatch, [self.RIGHT, self.LEFT])
        assert cltplot.get_current_monitor_size() == (1920, 1080)

    def test_without_a_primary_monitor_the_first_is_used(self, monkeypatch):
        first = Monitor(0, 0, 1280, 1024, False)
        second = Monitor(1280, 0, 1920, 1080, False)
        install_fake_tkinter(monkeypatch, pointer=(9000, 9000))
        install_fake_screeninfo(monkeypatch, [first, second])
        assert cltplot.get_current_monitor_size() == (1280, 1024)

    def test_window_is_destroyed(self, monkeypatch):
        state = install_fake_tkinter(monkeypatch)
        install_fake_screeninfo(monkeypatch, [self.LEFT])
        cltplot.get_current_monitor_size()
        assert state.roots[0].withdrawn
        assert state.roots[0].destroyed

    def test_screeninfo_is_required(self, monkeypatch):
        install_fake_tkinter(monkeypatch)
        monkeypatch.setitem(sys.modules, "screeninfo", None)
        with pytest.raises(ImportError):
            cltplot.get_current_monitor_size()


####################################################################################################
class TestEstimateMonitorDpi:
    @pytest.mark.parametrize(
        "resolution, dpi",
        [
            ((1920, 1080), 130),  # medium: median of 82, 92, 130, 147
            ((2560, 1440), 109),  # medium: median of 92, 109, 196
            ((3840, 2160), 103),  # more than 8M pixels: the lowest DPI
            ((1366, 768), 112),  # less than 1.5M pixels: the highest DPI
            ((1680, 1050), 90),
            ((2880, 1800), 220),
            ((3440, 1440), 110),
        ],
    )
    def test_known_resolutions(self, resolution, dpi):
        assert cltplot.estimate_monitor_dpi(*resolution) == dpi

    @pytest.mark.parametrize(
        "resolution, diagonal",
        [
            ((800, 600), 13),  # up to 1M pixels
            ((1280, 720), 13),
            ((1280, 1024), 21),  # up to 2M pixels
            ((1600, 1200), 21),
            ((2048, 1536), 24),  # up to 4M pixels
            ((2560, 1600), 27),  # up to 8M pixels
            ((5120, 2880), 32),  # more than 8M pixels
        ],
    )
    def test_unknown_resolutions_use_the_pixel_density(self, resolution, diagonal):
        expected = round(np.hypot(*resolution) / diagonal, 1)
        assert cltplot.estimate_monitor_dpi(*resolution) == expected

    def test_docstring_example(self):
        assert cltplot.estimate_monitor_dpi(1280, 1024) == 78.1

    def test_size_tiers_include_their_upper_limit(self):
        # 1000 x 1000 has exactly 1M pixels, and 1001 x 1000 has just over 1M
        assert cltplot.estimate_monitor_dpi(1000, 1000) == round(
            np.hypot(1000, 1000) / 13, 1
        )
        assert cltplot.estimate_monitor_dpi(1001, 1000) == round(
            np.hypot(1001, 1000) / 21, 1
        )

    def test_rotated_screen_has_the_same_dpi(self):
        assert cltplot.estimate_monitor_dpi(1080, 1920) == 130
        assert cltplot.estimate_monitor_dpi(1440, 2560) == 109
        assert cltplot.estimate_monitor_dpi(768, 1366) == 112

    def test_rotated_unknown_resolution_has_the_same_dpi(self):
        assert cltplot.estimate_monitor_dpi(1024, 1280) == cltplot.estimate_monitor_dpi(
            1280, 1024
        )

    def test_numpy_integers(self):
        assert cltplot.estimate_monitor_dpi(np.int64(1920), np.int64(1080)) == 130

    @pytest.mark.parametrize("size", [(0, 1080), (1920, 0), (-1920, 1080), (1920, -1)])
    def test_sizes_must_be_positive(self, size):
        with pytest.raises(ValueError, match="positive"):
            cltplot.estimate_monitor_dpi(*size)


####################################################################################################
class TestCalculateSubplotLayout:
    def test_docstring_examples(self):
        assert cltplot.calculate_subplot_layout(10, 1920, 1080) == (3, 4, 1920 / 1080)
        assert cltplot.calculate_subplot_layout(12, target_aspect_ratio=1.0) == (
            4,
            3,
            1.0,
        )

    def test_single_plot(self):
        assert cltplot.calculate_subplot_layout(1, 1920, 1080)[:2] == (1, 1)
        assert cltplot.calculate_subplot_layout(1, target_aspect_ratio=0.3)[:2] == (
            1,
            1,
        )

    def test_wide_screens_prefer_columns(self):
        assert cltplot.calculate_subplot_layout(2, 1920, 1080)[:2] == (1, 2)
        assert cltplot.calculate_subplot_layout(5, 1920, 1080)[:2] == (2, 3)
        assert cltplot.calculate_subplot_layout(7, 1920, 1080)[:2] == (2, 4)

    def test_tall_screens_prefer_rows(self):
        assert cltplot.calculate_subplot_layout(2, 1080, 1920)[:2] == (2, 1)
        assert cltplot.calculate_subplot_layout(5, 1080, 1920)[:2] == (3, 2)
        assert cltplot.calculate_subplot_layout(7, 1080, 1920)[:2] == (4, 2)

    def test_square_numbers_on_a_square_screen(self):
        for side in (2, 3, 4, 5):
            rows, cols, _ = cltplot.calculate_subplot_layout(
                side * side, target_aspect_ratio=1.0
            )
            assert (rows, cols) == (side, side)

    def test_aspect_ratio_is_returned(self):
        assert cltplot.calculate_subplot_layout(6, 3440, 1440)[2] == pytest.approx(
            3440 / 1440
        )
        assert cltplot.calculate_subplot_layout(6, target_aspect_ratio=2.5)[2] == 2.5

    def test_target_aspect_ratio_wins_over_the_screen(self):
        with_target = cltplot.calculate_subplot_layout(
            12, 1920, 1080, target_aspect_ratio=1.0
        )
        assert with_target == (4, 3, 1.0)

    @pytest.mark.parametrize("aspect", [0.3, 0.5625, 1.0, 1.7778, 2.4, 4.0])
    def test_layouts_always_fit_the_plots_with_few_empty_cells(self, aspect):
        for n_plots in range(1, 61):
            rows, cols, _ = cltplot.calculate_subplot_layout(
                n_plots, target_aspect_ratio=aspect
            )
            assert rows * cols >= n_plots, (n_plots, rows, cols)
            assert rows * cols - n_plots <= min(rows, cols), (n_plots, rows, cols)

    @pytest.mark.parametrize(
        "n_plots, aspect, layout",
        [(3, 0.69, (3, 1)), (4, 0.49, (4, 1)), (5, 1.2, (2, 3))],
    )
    def test_empty_cells_are_penalised(self, n_plots, aspect, layout):
        # Without the penalty, 3 plots at 0.69 would take the 2 x 2 grid (closer in shape)
        assert (
            cltplot.calculate_subplot_layout(n_plots, target_aspect_ratio=aspect)[:2]
            == layout
        )

    def test_wider_screens_never_give_fewer_columns(self):
        for n_plots in (6, 10, 12, 20, 30):
            columns = [
                cltplot.calculate_subplot_layout(n_plots, target_aspect_ratio=a)[1]
                for a in (0.5, 1.0, 1.78, 3.0)
            ]
            assert columns == sorted(columns), (n_plots, columns)

    def test_results_are_python_integers(self):
        rows, cols, _ = cltplot.calculate_subplot_layout(10, 1920, 1080)
        assert type(rows) is int
        assert type(cols) is int

    def test_numpy_integer_number_of_plots(self):
        assert cltplot.calculate_subplot_layout(np.int64(10), 1920, 1080)[:2] == (3, 4)

    def test_screen_is_detected_when_it_is_not_given(self, monkeypatch):
        calls = spy_on_detection(monkeypatch, size=(1080, 1920))
        assert cltplot.calculate_subplot_layout(5)[:2] == (3, 2)
        assert len(calls) == 1

    def test_screen_is_not_detected_when_it_is_given(self, monkeypatch):
        calls = spy_on_detection(monkeypatch)
        cltplot.calculate_subplot_layout(5, 1920, 1080)
        cltplot.calculate_subplot_layout(5, target_aspect_ratio=1.0)
        assert calls == []

    def test_layout_without_a_display_assumes_a_full_hd_screen(self, no_display):
        with pytest.warns(UserWarning, match="1920x1080"):
            rows, cols, aspect = cltplot.calculate_subplot_layout(10)
        assert (rows, cols) == (3, 4)
        assert aspect == pytest.approx(1920 / 1080)

    @pytest.mark.parametrize("n_plots", [0, -3])
    def test_number_of_plots_must_be_positive(self, n_plots):
        with pytest.raises(ValueError, match="positive"):
            cltplot.calculate_subplot_layout(n_plots, 1920, 1080)

    @pytest.mark.parametrize("n_plots", [2.5, "4", None])
    def test_number_of_plots_must_be_an_integer(self, n_plots):
        with pytest.raises(TypeError, match="integer"):
            cltplot.calculate_subplot_layout(n_plots, 1920, 1080)

    @pytest.mark.parametrize(
        "screen", [(1920, 0), (0, 1080), (-1920, 1080), (1920, -1080)]
    )
    def test_screen_size_must_be_positive(self, screen):
        with pytest.raises(ValueError, match="positive"):
            cltplot.calculate_subplot_layout(4, *screen)

    @pytest.mark.parametrize("aspect", [0, -1.0, float("nan"), float("inf")])
    def test_target_aspect_ratio_must_be_positive_and_finite(self, aspect):
        with pytest.raises(ValueError, match="aspect"):
            cltplot.calculate_subplot_layout(4, target_aspect_ratio=aspect)


####################################################################################################
class TestCalculateOptimalSubplotsGrid:
    FIXED = {
        1: [1, 1],
        2: [1, 2],
        3: [1, 3],
        4: [2, 2],
        5: [2, 3],
        6: [2, 3],
        7: [2, 4],
        8: [2, 4],
        16: [4, 4],
    }

    @pytest.mark.parametrize("n_views, grid", sorted(FIXED.items()))
    def test_fixed_grids(self, n_views, grid):
        grid_size, positions = cltplot.calculate_optimal_subplots_grid(n_views)
        assert grid_size == grid
        assert len(positions) == n_views

    @pytest.mark.parametrize("n_views", sorted(FIXED))
    def test_positions_fill_the_grid_in_row_major_order(self, n_views):
        grid_size, positions = cltplot.calculate_optimal_subplots_grid(n_views)
        rows, cols = grid_size
        assert positions == [(i // cols, i % cols) for i in range(n_views)]

    def test_docstring_examples(self):
        assert cltplot.calculate_optimal_subplots_grid(4) == (
            [2, 2],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        )
        assert cltplot.calculate_optimal_subplots_grid(1) == ([1, 1], [(0, 0)])
        assert cltplot.calculate_optimal_subplots_grid(6)[0] == [2, 3]

    def test_fixed_grids_do_not_need_a_display(self, monkeypatch):
        calls = spy_on_detection(monkeypatch)
        for n_views in self.FIXED:
            cltplot.calculate_optimal_subplots_grid(n_views)
        assert calls == []

    def test_other_numbers_follow_the_screen(self, monkeypatch):
        spy_on_detection(monkeypatch, size=(1920, 1080))
        grid_size, positions = cltplot.calculate_optimal_subplots_grid(10)
        assert grid_size == [3, 4]
        assert positions == [(i // 4, i % 4) for i in range(10)]

    def test_tall_screen_gives_more_rows(self, monkeypatch):
        spy_on_detection(monkeypatch, size=(1080, 1920))
        grid_size, _ = cltplot.calculate_optimal_subplots_grid(10)
        assert grid_size[0] > grid_size[1]

    @pytest.mark.parametrize("size", [(1920, 1080), (1080, 1920), (3440, 1440)])
    def test_every_view_gets_one_position_inside_the_grid(self, monkeypatch, size):
        spy_on_detection(monkeypatch, size=size)
        for n_views in range(1, 41):
            grid_size, positions = cltplot.calculate_optimal_subplots_grid(n_views)
            rows, cols = grid_size
            assert len(positions) == n_views, n_views
            assert len(set(positions)) == n_views, n_views
            assert all(0 <= r < rows and 0 <= c < cols for r, c in positions), n_views
            assert positions == sorted(positions), n_views
            assert rows * cols >= n_views, n_views

    def test_grid_without_a_display_assumes_a_full_hd_screen(self, no_display):
        with pytest.warns(UserWarning, match="1920x1080"):
            grid_size, positions = cltplot.calculate_optimal_subplots_grid(10)
        assert grid_size == [3, 4]
        assert len(positions) == 10

    def test_numpy_integer(self):
        assert cltplot.calculate_optimal_subplots_grid(np.int64(4))[0] == [2, 2]

    @pytest.mark.parametrize("n_views", [0, -1])
    def test_number_of_views_must_be_positive(self, n_views):
        with pytest.raises(ValueError, match="positive"):
            cltplot.calculate_optimal_subplots_grid(n_views)

    @pytest.mark.parametrize("n_views", [2.5, "4", None])
    def test_number_of_views_must_be_an_integer(self, n_views):
        with pytest.raises(TypeError, match="integer"):
            cltplot.calculate_optimal_subplots_grid(n_views)


####################################################################################################
@pytest.mark.usefixtures("agg_backend")
class TestCreateProportionalSubplots:
    def test_layout_of_seven_plots(self):
        fig, axes, info = cltplot.create_proportional_subplots(
            7, figsize_base=3, **FULL_HD
        )
        assert (info["rows"], info["cols"]) == (2, 4)
        assert info["used_subplots"] == 7
        assert info["total_subplots"] == 8
        assert info["aspect_ratio"] == pytest.approx(1920 / 1080)
        assert isinstance(fig, plt.Figure)
        assert len(axes) == 8

    def test_extra_axes_are_hidden(self):
        _, axes, _ = cltplot.create_proportional_subplots(7, **FULL_HD)
        assert [ax.get_visible() for ax in axes] == [True] * 7 + [False]

    def test_all_axes_are_used_when_the_grid_is_full(self):
        _, axes, info = cltplot.create_proportional_subplots(4, target_aspect_ratio=1.0)
        assert (info["rows"], info["cols"]) == (2, 2)
        assert all(ax.get_visible() for ax in axes)

    def test_single_plot(self):
        fig, axes, info = cltplot.create_proportional_subplots(1, **FULL_HD)
        assert len(axes) == 1
        assert axes[0].figure is fig
        assert info["total_subplots"] == 1

    @pytest.mark.parametrize("n_plots", [2, 3, 5, 9, 12])
    def test_axes_are_a_flat_sequence_of_the_cells(self, n_plots):
        fig, axes, info = cltplot.create_proportional_subplots(n_plots, **FULL_HD)
        assert len(axes) == info["rows"] * info["cols"]
        assert len(fig.axes) == len(axes)
        assert all(isinstance(ax, plt.Axes) for ax in axes)

    def test_single_column_on_a_tall_screen(self):
        _, axes, info = cltplot.create_proportional_subplots(
            3, screen_width=1080, screen_height=1920
        )
        assert (info["rows"], info["cols"]) == (3, 1)
        assert len(axes) == 3
        assert all(ax.get_visible() for ax in axes)

    def test_cells_beyond_the_plots_on_a_wide_screen(self):
        _, axes, info = cltplot.create_proportional_subplots(3, **FULL_HD)
        assert (info["rows"], info["cols"]) == (2, 2)
        assert [ax.get_visible() for ax in axes] == [True, True, True, False]

    def test_figure_size_follows_the_columns_and_the_aspect_ratio(self):
        fig, _, info = cltplot.create_proportional_subplots(
            7, figsize_base=3, **FULL_HD
        )
        width, height = fig.get_size_inches()
        assert width == pytest.approx(3 * info["cols"])
        assert height == pytest.approx(width / info["aspect_ratio"])

    def test_figure_size_scales_with_the_base_size(self):
        small, _, _ = cltplot.create_proportional_subplots(6, figsize_base=2, **FULL_HD)
        large, _, _ = cltplot.create_proportional_subplots(6, figsize_base=4, **FULL_HD)
        assert large.get_size_inches() == pytest.approx(2 * small.get_size_inches())

    def test_target_aspect_ratio(self):
        fig, _, info = cltplot.create_proportional_subplots(6, target_aspect_ratio=2.0)
        assert info["aspect_ratio"] == 2.0
        width, height = fig.get_size_inches()
        assert width / height == pytest.approx(2.0)

    def test_screen_is_detected_when_it_is_not_given(self, monkeypatch):
        spy_on_detection(monkeypatch, size=(1080, 1920))
        _, _, info = cltplot.create_proportional_subplots(5)
        assert info["aspect_ratio"] == pytest.approx(1080 / 1920)

    def test_without_a_display_a_full_hd_screen_is_assumed(self, no_display):
        with pytest.warns(UserWarning, match="1920x1080"):
            _, _, info = cltplot.create_proportional_subplots(5)
        assert info["aspect_ratio"] == pytest.approx(1920 / 1080)

    def test_axes_can_be_drawn_on(self):
        fig, axes, _ = cltplot.create_proportional_subplots(3, **FULL_HD)
        for i, ax in enumerate(axes[:3]):
            ax.plot([0, 1], [0, i])
        assert [len(ax.lines) for ax in axes[:3]] == [1, 1, 1]

    @pytest.mark.parametrize("n_plots", [0, -2])
    def test_number_of_plots_must_be_positive(self, n_plots):
        with pytest.raises(ValueError, match="positive"):
            cltplot.create_proportional_subplots(n_plots, **FULL_HD)

    def test_number_of_plots_must_be_an_integer(self):
        with pytest.raises(TypeError, match="integer"):
            cltplot.create_proportional_subplots(2.5, **FULL_HD)

    @pytest.mark.parametrize("base", [0, -4, float("nan")])
    def test_base_size_must_be_positive(self, base):
        with pytest.raises(ValueError, match="figsize_base"):
            cltplot.create_proportional_subplots(4, figsize_base=base, **FULL_HD)

    def test_invalid_layout_options_are_rejected(self):
        with pytest.raises(TypeError):
            cltplot.create_proportional_subplots(4, unknown_option=1)

    def test_no_figure_is_left_open_after_a_failure(self):
        before = len(plt.get_fignums())
        with pytest.raises(ValueError):
            cltplot.create_proportional_subplots(0, **FULL_HD)
        assert len(plt.get_fignums()) == before


####################################################################################################
def base_sizes(**kwargs):
    return cltplot.calculate_font_sizes(6, 4, **{**FULL_HD, **kwargs})


class TestCalculateFontSizes:
    def test_full_hd_reference_plot(self):
        fonts = base_sizes()
        scale = 130 / 96  # estimated DPI over the 96 DPI baseline
        assert fonts["title"] == pytest.approx(14 * scale, abs=0.06)
        assert fonts["axis_labels"] == pytest.approx(12 * scale, abs=0.06)
        assert fonts["tick_labels"] == pytest.approx(10 * scale, abs=0.06)
        # 0.3 inch colorbar over the 0.5 inch reference
        assert fonts["colorbar_title"] == pytest.approx(11 * 0.6 * scale, abs=0.06)
        assert fonts["colorbar_ticks"] == pytest.approx(9 * 0.6 * scale, abs=0.06)

    def test_docstring_examples(self):
        fonts = base_sizes()
        assert fonts["title"] == 19.0
        assert fonts["colorbar_title"] == 8.9
        quhd = cltplot.calculate_font_sizes(6, 4, screen_width=2560, screen_height=1440)
        assert quhd["_monitor_info"]["estimated_dpi"] == 109
        small = cltplot.calculate_font_sizes(
            3,
            2,
            screen_width=3840,
            screen_height=2160,
            colorbar_orientation="horizontal",
        )
        assert small["colorbar_title"] == 8.9

    def test_values_have_one_decimal(self):
        fonts = base_sizes()
        for key, value in fonts.items():
            if not key.startswith("_"):
                assert value == round(value, 1), key

    def test_all_sizes_are_floats(self):
        # Sizes that reach a bound used to be integers
        huge = cltplot.calculate_font_sizes(40, 30, **FULL_HD)
        tiny = cltplot.calculate_font_sizes(0.1, 0.1, **FULL_HD)
        for fonts in (huge, tiny, base_sizes()):
            for key, value in fonts.items():
                if not key.startswith("_"):
                    assert type(value) is float, (key, value)

    def test_keys(self):
        assert set(base_sizes()) == {
            "title",
            "axis_labels",
            "tick_labels",
            "colorbar_title",
            "colorbar_ticks",
            "_monitor_info",
        }

    def test_monitor_info(self):
        info = base_sizes()["_monitor_info"]
        assert info["screen_width"] == 1920
        assert info["screen_height"] == 1080
        assert info["estimated_dpi"] == 130
        assert info["dpi_scale_factor"] == pytest.approx(130 / 96, abs=0.01)
        assert info["area_scale_factor"] == 1.0
        assert info["plot_scale"] == pytest.approx(130 / 96, abs=0.01)
        assert info["colorbar_scale"] == pytest.approx(0.6 * 130 / 96, abs=0.01)

    def test_plot_area_scaling(self):
        # Four times the area doubles the plot scale: a 4 x 3 and an 8 x 6 inch plot
        screen = {"screen_width": 2560, "screen_height": 1440}
        small = cltplot.calculate_font_sizes(4, 3, **screen)
        double = cltplot.calculate_font_sizes(8, 6, **screen)
        assert small["_monitor_info"]["area_scale_factor"] == pytest.approx(
            0.71, abs=0.01
        )
        assert double["_monitor_info"]["area_scale_factor"] == pytest.approx(
            1.41, abs=0.01
        )
        assert double["tick_labels"] == pytest.approx(
            2 * small["tick_labels"], rel=0.02
        )

    def test_larger_plots_never_have_smaller_fonts(self):
        previous = None
        for width in (1, 2, 4, 6, 8, 12, 20):
            fonts = cltplot.calculate_font_sizes(width, width * 2 / 3, **FULL_HD)
            current = [fonts[k] for k in ("title", "axis_labels", "tick_labels")]
            if previous is not None:
                assert all(c >= p for c, p in zip(current, previous, strict=True))
            previous = current

    def test_denser_screens_never_have_smaller_fonts(self):
        low = cltplot.calculate_font_sizes(6, 4, screen_width=1920, screen_height=1080)
        high = cltplot.calculate_font_sizes(6, 4, screen_width=2880, screen_height=1800)
        assert high["title"] > low["title"]
        assert high["tick_labels"] > low["tick_labels"]

    @pytest.mark.parametrize(
        "width, height, expected",
        [
            (60, 40, {"title": 28.0, "axis_labels": 20.0, "tick_labels": 20.0}),
            (0.1, 0.1, {"title": 8.0, "axis_labels": 6.0, "tick_labels": 6.0}),
        ],
    )
    def test_plot_font_bounds(self, width, height, expected):
        fonts = cltplot.calculate_font_sizes(width, height, **FULL_HD)
        for key, value in expected.items():
            assert fonts[key] == value, key

    @pytest.mark.parametrize(
        "orientation, big, small",
        [
            ("vertical", (14.0, 10.0), (6.0, 5.0)),
            ("horizontal", (16.0, 12.0), (7.0, 6.0)),
        ],
    )
    def test_colorbar_font_bounds(self, orientation, big, small):
        size = "colorbar_width" if orientation == "vertical" else "colorbar_height"
        large = base_sizes(colorbar_orientation=orientation, **{size: 100.0})
        tiny = base_sizes(colorbar_orientation=orientation, **{size: 0.001})
        assert (large["colorbar_title"], large["colorbar_ticks"]) == big
        assert (tiny["colorbar_title"], tiny["colorbar_ticks"]) == small

    @pytest.mark.parametrize(
        "screen, factor",
        [((2880, 1800), 2.0), ((640, 480), 0.8)],
    )
    def test_dpi_scaling_is_bounded(self, screen, factor):
        fonts = cltplot.calculate_font_sizes(
            6, 4, screen_width=screen[0], screen_height=screen[1]
        )
        assert fonts["_monitor_info"]["dpi_scale_factor"] == factor

    def test_orientation_only_changes_the_colorbar_fonts(self):
        vertical = base_sizes(colorbar_orientation="vertical")
        horizontal = base_sizes(colorbar_orientation="horizontal")
        for key in ("title", "axis_labels", "tick_labels"):
            assert vertical[key] == horizontal[key]
        assert vertical["colorbar_title"] != horizontal["colorbar_title"]

    def test_vertical_colorbar_scales_with_its_width(self):
        narrow = base_sizes(colorbar_width=0.3)
        wide = base_sizes(colorbar_width=0.5)
        assert wide["colorbar_title"] > narrow["colorbar_title"]
        assert wide["colorbar_ticks"] > narrow["colorbar_ticks"]

    def test_horizontal_colorbar_scales_with_its_height(self):
        thin = base_sizes(colorbar_orientation="horizontal", colorbar_height=0.3)
        thick = base_sizes(colorbar_orientation="horizontal", colorbar_height=0.5)
        assert thick["colorbar_title"] > thin["colorbar_title"]
        assert thick["colorbar_ticks"] > thin["colorbar_ticks"]

    def test_default_colorbar_size(self):
        # 5% of a 10 inch wide plot is 0.5 inches
        default = cltplot.calculate_font_sizes(10, 4, **FULL_HD)
        explicit = cltplot.calculate_font_sizes(10, 4, colorbar_width=0.5, **FULL_HD)
        assert default == explicit
        # 5% of a 10 inch high plot is 0.5 inches
        default = cltplot.calculate_font_sizes(
            6, 10, colorbar_orientation="horizontal", **FULL_HD
        )
        explicit = cltplot.calculate_font_sizes(
            6, 10, colorbar_orientation="horizontal", colorbar_height=0.5, **FULL_HD
        )
        assert default == explicit

    def test_default_colorbar_has_a_minimum_size(self):
        default = cltplot.calculate_font_sizes(2, 2, **FULL_HD)
        explicit = cltplot.calculate_font_sizes(2, 2, colorbar_width=0.3, **FULL_HD)
        assert default == explicit

    def test_integer_and_float_sizes_agree(self):
        assert cltplot.calculate_font_sizes(6, 4, **FULL_HD) == (
            cltplot.calculate_font_sizes(6.0, 4.0, **FULL_HD)
        )

    @pytest.mark.parametrize("dtype", [np.int64, np.int32, np.float32, np.float64])
    def test_numpy_numbers_are_accepted(self, dtype):
        fonts = cltplot.calculate_font_sizes(dtype(6), dtype(4), **FULL_HD)
        assert fonts["title"] == 19.0

    def test_numpy_screen_sizes_are_accepted(self):
        fonts = cltplot.calculate_font_sizes(
            6, 4, screen_width=np.int64(1920), screen_height=np.int64(1080)
        )
        assert fonts["title"] == 19.0

    def test_screen_is_detected_when_it_is_not_given(self, monkeypatch):
        calls = spy_on_detection(monkeypatch, size=(2560, 1440))
        fonts = cltplot.calculate_font_sizes(6, 4)
        assert len(calls) == 1
        assert fonts["_monitor_info"]["screen_width"] == 2560
        assert fonts["_monitor_info"]["estimated_dpi"] == 109

    def test_screen_is_not_detected_when_it_is_given(self, monkeypatch):
        calls = spy_on_detection(monkeypatch)
        cltplot.calculate_font_sizes(6, 4, **FULL_HD)
        assert calls == []

    def test_given_screen_width_is_kept_when_the_height_is_detected(self, monkeypatch):
        spy_on_detection(monkeypatch, size=(3840, 2160))
        info = cltplot.calculate_font_sizes(6, 4, screen_width=2560)["_monitor_info"]
        assert (info["screen_width"], info["screen_height"]) == (2560, 2160)

    def test_detection_can_be_disabled(self, monkeypatch):
        calls = spy_on_detection(monkeypatch)
        info = cltplot.calculate_font_sizes(6, 4, auto_detect_monitor=False)[
            "_monitor_info"
        ]
        assert calls == []
        assert (info["screen_width"], info["screen_height"]) == (1920, 1080)

    def test_disabled_detection_keeps_the_given_dimension(self):
        info = cltplot.calculate_font_sizes(
            6, 4, screen_width=2560, auto_detect_monitor=False
        )["_monitor_info"]
        assert (info["screen_width"], info["screen_height"]) == (2560, 1080)
        info = cltplot.calculate_font_sizes(
            6, 4, screen_height=1440, auto_detect_monitor=False
        )["_monitor_info"]
        assert (info["screen_width"], info["screen_height"]) == (1920, 1440)

    def test_missing_detection_packages_are_reported(self, monkeypatch):
        def missing():
            raise ImportError("No module named 'screeninfo'")

        monkeypatch.setattr(cltplot, "get_current_monitor_size", missing)
        with pytest.raises(ImportError, match="auto_detect_monitor=False"):
            cltplot.calculate_font_sizes(6, 4)

    def test_without_a_display_a_full_hd_screen_is_assumed(self, no_display):
        with pytest.warns(UserWarning, match="1920x1080"):
            fonts = cltplot.calculate_font_sizes(6, 4)
        assert fonts == base_sizes()

    @pytest.mark.parametrize("bad", [0, -1, -0.5, float("nan"), float("inf")])
    def test_plot_size_must_be_positive_and_finite(self, bad):
        with pytest.raises(ValueError, match="plot_width"):
            cltplot.calculate_font_sizes(bad, 4, **FULL_HD)
        with pytest.raises(ValueError, match="plot_height"):
            cltplot.calculate_font_sizes(6, bad, **FULL_HD)

    @pytest.mark.parametrize("bad", ["6", None, [6], True])
    def test_plot_size_must_be_a_number(self, bad):
        with pytest.raises(TypeError, match="plot_width"):
            cltplot.calculate_font_sizes(bad, 4, **FULL_HD)
        with pytest.raises(TypeError, match="plot_height"):
            cltplot.calculate_font_sizes(6, bad, **FULL_HD)

    def test_colorbar_orientation_must_be_known(self):
        with pytest.raises(ValueError, match="colorbar_orientation"):
            cltplot.calculate_font_sizes(
                6, 4, colorbar_orientation="diagonal", **FULL_HD
            )

    @pytest.mark.parametrize("bad", [0, -0.3, float("nan")])
    def test_colorbar_size_must_be_positive(self, bad):
        with pytest.raises(ValueError, match="colorbar_width"):
            cltplot.calculate_font_sizes(6, 4, colorbar_width=bad, **FULL_HD)
        with pytest.raises(ValueError, match="colorbar_height"):
            cltplot.calculate_font_sizes(6, 4, colorbar_height=bad, **FULL_HD)

    @pytest.mark.parametrize("bad", [0, -1920])
    def test_screen_size_must_be_positive(self, bad):
        with pytest.raises(ValueError, match="positive"):
            cltplot.calculate_font_sizes(6, 4, screen_width=bad, screen_height=1080)


####################################################################################################
class TestGetPyvistaFonts:
    def test_font_sizes_are_integers(self):
        fonts = cltplot.get_pyvista_fonts(8, 6, **FULL_HD)
        assert set(fonts) == {
            "title",
            "axis_labels",
            "tick_labels",
            "colorbar_title",
            "colorbar_ticks",
        }
        assert all(type(v) is int for v in fonts.values())

    def test_debug_information_is_excluded(self):
        assert "_monitor_info" not in cltplot.get_pyvista_fonts(8, 6, **FULL_HD)

    def test_sizes_are_the_rounded_float_sizes(self):
        floats = cltplot.calculate_font_sizes(8, 6, **FULL_HD)
        fonts = cltplot.get_pyvista_fonts(8, 6, **FULL_HD)
        for key, value in fonts.items():
            assert value == int(round(floats[key])), key

    def test_options_are_passed_on(self):
        vertical = cltplot.get_pyvista_fonts(8, 6, **FULL_HD)
        horizontal = cltplot.get_pyvista_fonts(
            8, 6, colorbar_orientation="horizontal", **FULL_HD
        )
        assert vertical["colorbar_title"] != horizontal["colorbar_title"]
        assert vertical["title"] == horizontal["title"]

    def test_screen_dimensions_are_passed_on(self):
        full_hd = cltplot.get_pyvista_fonts(8, 6, **FULL_HD)
        retina = cltplot.get_pyvista_fonts(8, 6, screen_width=2880, screen_height=1800)
        assert retina["title"] > full_hd["title"]

    def test_invalid_sizes_are_rejected(self):
        with pytest.raises(ValueError):
            cltplot.get_pyvista_fonts(0, 6, **FULL_HD)

    def test_without_a_display_a_full_hd_screen_is_assumed(self, no_display):
        with pytest.warns(UserWarning, match="1920x1080"):
            fonts = cltplot.get_pyvista_fonts(8, 6)
        assert fonts == cltplot.get_pyvista_fonts(8, 6, **FULL_HD)


####################################################################################################
class TestGenerateSphericalCoords:
    def test_shape_and_type(self):
        coords = cltplot.generate_spherical_coords(25, np.random.default_rng(0))
        assert coords.shape == (25, 3)
        assert coords.dtype == np.float64

    def test_points_are_inside_the_sphere(self):
        coords = cltplot.generate_spherical_coords(
            5000, np.random.default_rng(1), radius=30.0
        )
        assert np.linalg.norm(coords, axis=1).max() <= 30.0

    def test_default_radius(self):
        coords = cltplot.generate_spherical_coords(5000, np.random.default_rng(1))
        norms = np.linalg.norm(coords, axis=1)
        assert norms.max() <= 50.0
        assert norms.max() > 49.0

    def test_docstring_example(self):
        coords = cltplot.generate_spherical_coords(3, np.random.default_rng(42), 50.0)
        np.testing.assert_allclose(
            np.round(coords, 2),
            [
                [8.86, -30.24, 21.82],
                [13.37, -27.73, -18.51],
                [18.25, -45.14, -2.4],
            ],
        )

    def test_same_seed_gives_the_same_points(self):
        first = cltplot.generate_spherical_coords(10, np.random.default_rng(7))
        second = cltplot.generate_spherical_coords(10, np.random.default_rng(7))
        np.testing.assert_array_equal(first, second)

    def test_different_seeds_give_different_points(self):
        first = cltplot.generate_spherical_coords(10, np.random.default_rng(7))
        second = cltplot.generate_spherical_coords(10, np.random.default_rng(8))
        assert not np.allclose(first, second)

    def test_successive_calls_continue_the_random_stream(self):
        rng = np.random.default_rng(3)
        first = cltplot.generate_spherical_coords(5, rng)
        second = cltplot.generate_spherical_coords(5, rng)
        assert not np.allclose(first, second)

    def test_radius_scales_the_points(self):
        small = cltplot.generate_spherical_coords(20, np.random.default_rng(5), 10.0)
        large = cltplot.generate_spherical_coords(20, np.random.default_rng(5), 40.0)
        np.testing.assert_allclose(large, 4 * small)

    def test_points_fill_the_volume_uniformly(self):
        radius = 20.0
        coords = cltplot.generate_spherical_coords(
            60000, np.random.default_rng(11), radius
        )
        norms = np.linalg.norm(coords, axis=1)
        # The share of the points within a fraction f of the radius is f cubed
        for fraction in (0.25, 0.5, 0.75, 0.9):
            share = np.mean(norms <= fraction * radius)
            assert share == pytest.approx(fraction**3, abs=0.01), fraction

    def test_directions_are_isotropic(self):
        radius = 20.0
        coords = cltplot.generate_spherical_coords(
            60000, np.random.default_rng(12), radius
        )
        np.testing.assert_allclose(coords.mean(axis=0), 0.0, atol=0.2)
        # Variance of each coordinate in a uniform ball: radius^2 / 5
        np.testing.assert_allclose(coords.var(axis=0), radius**2 / 5, rtol=0.03)
        correlation = np.corrcoef(coords.T)
        np.testing.assert_allclose(correlation - np.eye(3), 0.0, atol=0.02)

    def test_no_points(self):
        coords = cltplot.generate_spherical_coords(0, np.random.default_rng(0))
        assert coords.shape == (0, 3)

    def test_single_point(self):
        coords = cltplot.generate_spherical_coords(1, np.random.default_rng(0), 5.0)
        assert coords.shape == (1, 3)
        assert np.linalg.norm(coords) <= 5.0

    def test_zero_radius_puts_every_point_at_the_origin(self):
        coords = cltplot.generate_spherical_coords(4, np.random.default_rng(0), 0.0)
        np.testing.assert_array_equal(coords, np.zeros((4, 3)))

    def test_zero_length_direction_does_not_give_nan(self):
        class ZeroRng:
            def normal(self, size):
                return np.zeros(size)

            def uniform(self, low, high, size):
                return np.full(size, 0.5)

        coords = cltplot.generate_spherical_coords(3, ZeroRng(), 10.0)
        assert np.isfinite(coords).all()
        np.testing.assert_array_equal(coords, np.zeros((3, 3)))

    def test_radius_must_not_be_negative(self):
        with pytest.raises(ValueError, match="radius"):
            cltplot.generate_spherical_coords(5, np.random.default_rng(0), -1.0)

    @pytest.mark.parametrize("radius", [float("nan"), float("inf")])
    def test_radius_must_be_finite(self, radius):
        with pytest.raises(ValueError, match="radius"):
            cltplot.generate_spherical_coords(5, np.random.default_rng(0), radius)

    def test_number_of_points_must_not_be_negative(self):
        with pytest.raises(ValueError):
            cltplot.generate_spherical_coords(-1, np.random.default_rng(0))
