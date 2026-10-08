"""Tests for clabtoolkit.colorstools."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import clabtoolkit.colorstools as cltcol  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def ctab_dict():
    return {
        "index": [1, 2, 3],
        "name": ["ctx-lh-a", "ctx-lh-b", "ctx-rh-a"],
        "color": ["#ff0000", "#00ff00", "#0000ff"],
        "opacity": [1.0, 0.5, 1.0],
    }


@pytest.fixture
def lut_file(tmp_path):
    """FreeSurfer-style LUT with header lines and alpha values 0, 128 and 255."""
    path = tmp_path / "atlas.lut"
    path.write_text(
        "# Atlas test LUT\n"
        "# Created for the tests\n"
        "#No. Label Name:    R   G   B   A\n"
        "\n"
        "0   Unknown          0   0   0   0\n"
        "1   Left-Hippocampus 220 216 20  0\n"
        "2   Right-Hippocampus 220 216 20 128\n"
        "3   Left-Amygdala    103 255 255 255\n"
    )
    return path


@pytest.fixture
def tsv_file(tmp_path):
    path = tmp_path / "atlas.tsv"
    path.write_text(
        "index\tname\tcolor\topacity\n"
        "1\tctx-lh-a\t#ff0000\t1.0\n"
        "2\tctx-lh-b\t#00ff00\t0.5\n"
        "3\tctx-rh-a\t#0000ff\t1.0\n"
    )
    return path


####################################################################################################
# Section 1: color validation and conversion
####################################################################################################
class TestValidation:
    @pytest.mark.parametrize(
        "color",
        [
            "#FF5733",
            "#FF5733FF",
            "red",
            np.array([255, 87, 51]),
            np.array([255, 87, 51, 255]),
            np.array([1.0, 0.34, 0.2]),
            np.array([70.0, 130.0, 180.0]),
            [255, 87, 51],
            (255, 128, 0, 128),
            [1.0, 0.34, 0.5],
            [255.0, 128.0, 0.0],
        ],
    )
    def test_is_color_like_valid(self, color):
        assert cltcol.is_color_like(color) is True

    @pytest.mark.parametrize(
        "color",
        [
            "invalid_color",
            [256, 0, 0],
            [128.5, 0, 0],
            [1, 2],
            ["a", "b", "c"],
            np.array([70.5, 130.0, 180.0]),
            np.array([[1, 2, 3]]),
            np.array([-1, 0, 0]),
        ],
    )
    def test_is_color_like_invalid(self, color):
        assert cltcol.is_color_like(color) is False

    @pytest.mark.parametrize(
        "rgb, expected",
        [
            ([255, 128, 0], "0-255"),
            ([1.0, 0.5, 0.0], "0-1"),
            ([0, 1, 0], "0-1"),
            ([255, 0.5, 128], "invalid"),
            ([300, 200, 100], "invalid"),
            ((255, 128, 0, 128), "0-255"),
            (np.array([255, 87, 51]), "0-255"),
            (np.array([1.0, 0.34, 0.2]), "0-1"),
            (np.array([70.0, 130.0, 180.0]), "0-255"),
            (np.array([70.5, 130.0, 180.0]), "invalid"),
            ("not_a_list", "invalid"),
            ([255, 128], "invalid"),
        ],
    )
    def test_detect_rgb_range(self, rgb, expected):
        assert cltcol.detect_rgb_range(rgb) == expected

    @pytest.mark.parametrize(
        "rgb, expected",
        [
            ([255, 128, 0], True),
            ([128.0, 200.0, 50.0], True),
            ((255, 128, 0, 128), True),
            (np.array([255, 128, 0, 255]), True),
            (np.array([70.0, 130.0, 180.0]), True),
            ([np.int64(255), 128, 0], True),
            ([0.5, 0.3, 0.8], False),
            ([300, 200, 100], False),
            ([-1, 128, 0], False),
            ([255, 128], False),
            ("#ffffff", False),
        ],
    )
    def test_is_valid_rgb_255(self, rgb, expected):
        assert cltcol.is_valid_rgb_255(rgb) is expected  # A Python bool

    @pytest.mark.parametrize(
        "rgb, expected",
        [
            ([1.0, 0.5, 0.0], True),
            ([0, 1, 0], True),
            ((1.0, 0.5, 0.0, 1.0), True),
            (np.array([0, 1, 0]), True),
            (np.array([1.0, 0.5, 0.0, 0.5]), True),
            ([255, 128, 0], False),
            ([1.5, 0.5, 0.2], False),
            ([-0.1, 0.5, 0.2], False),
            (np.array([0, 2, 0]), False),
            ([0, 0], False),
        ],
    )
    def test_is_valid_rgb_01(self, rgb, expected):
        assert cltcol.is_valid_rgb_01(rgb) is expected  # A Python bool

    @pytest.mark.parametrize(
        "hex_color, expected",
        [
            ("#FF0000", True),
            ("#ffffff", True),
            ("#ABC123", True),
            ("#FFF", False),
            ("FF0000", False),
            ("#GG0000", False),
            ("#FF0000FF", False),
            ("", False),
            (None, False),
            (123, False),
        ],
    )
    def test_is_valid_hex_color(self, hex_color, expected):
        assert cltcol.is_valid_hex_color(hex_color) is expected

    def test_normalize_rgb(self):
        assert cltcol.normalize_rgb([255, 0, 51]) == [1.0, 0.0, 0.2]
        assert cltcol.normalize_rgb([1, 0.5, 0]) == [1.0, 0.5, 0.0]
        assert cltcol.normalize_rgb("x") is None


class TestConversion:
    @pytest.mark.parametrize(
        "rgb, expected",
        [
            ((255, 0, 0), "#ff0000"),
            ((1.0, 0.0, 0.0), "#ff0000"),
            ((0.5, 0.0, 1.0), "#8000ff"),
            ((0, 128, 255), "#0080ff"),
            ((np.int64(1), np.int64(2), np.int64(3)), "#010203"),
        ],
    )
    def test_rgb2hex(self, rgb, expected):
        assert cltcol.rgb2hex(*rgb) == expected

    def test_rgb2hex_mixed_python_and_numpy_integers(self):
        # Components taken from different sources must not be rejected
        assert cltcol.rgb2hex(np.int64(255), 0, 0) == "#ff0000"

    def test_rgb2hex_errors(self):
        with pytest.raises(TypeError):
            cltcol.rgb2hex(255, 0.0, 0)
        with pytest.raises(ValueError):
            cltcol.rgb2hex(256, 0, 0)
        with pytest.raises(ValueError):
            cltcol.rgb2hex(1.5, 0.0, 0.0)

    def test_hex2rgb(self):
        assert cltcol.hex2rgb("#FF5733") == (255, 87, 51)
        assert cltcol.hex2rgb("00ff00") == (0, 255, 0)

    def test_multi_conversions(self):
        rgb = cltcol.multi_hex2rgb(["#FF5733", "#33FF57"])
        assert rgb.tolist() == [[255, 87, 51], [51, 255, 87]]
        assert cltcol.multi_hex2rgb("#000000").tolist() == [[0, 0, 0]]
        assert cltcol.multi_rgb2hex([[255, 0, 0], [0, 255, 0], [0, 0, 255]]) == [
            "#ff0000",
            "#00ff00",
            "#0000ff",
        ]
        assert cltcol.multi_rgb2hex(["#FF0000", [0, 0, 255]]) == ["#ff0000", "#0000ff"]

    def test_invert_colors(self):
        out = cltcol.invert_colors([np.array([0.0, 0.0, 1.0]), np.array([0, 255, 243])])
        np.testing.assert_allclose(out[0], [1.0, 1.0, 0.0])
        assert out[1].tolist() == [255, 0, 12]
        assert cltcol.invert_colors(["#ff0000", [0, 0, 255]]) == [
            "#00ffff",
            [255, 255, 0],
        ]
        assert cltcol.invert_colors([[1.0, 0.0, 0.0]]) == [[0.0, 1.0, 1.0]]
        assert isinstance(cltcol.invert_colors(np.array([[0, 0, 255]])), np.ndarray)
        with pytest.raises(TypeError):
            cltcol.invert_colors("#ff0000")


class TestHarmonizeColors:
    def test_hex_output(self):
        colors = ["#FF5733", [255, 87, 51], (51, 87, 255), np.array([1.0, 0.0, 0.0])]
        assert cltcol.harmonize_colors(colors) == [
            "#ff5733",
            "#ff5733",
            "#3357ff",
            "#ff0000",
        ]
        assert cltcol.harmonize_colors((255, 87, 51)) == ["#ff5733"]
        assert cltcol.harmonize_colors("#FF573380") == ["#ff5733"]  # Alpha dropped
        assert cltcol.harmonize_colors(["red"]) == ["#ff0000"]

    def test_rgb_and_rgbnorm_output(self):
        colors = [(255, 87, 51, 255), "#3357ff"]
        rgb = cltcol.harmonize_colors(colors, output_format="rgb")
        assert rgb.dtype == np.uint8 and rgb.tolist() == [[255, 87, 51], [51, 87, 255]]
        norm = cltcol.harmonize_colors(colors, output_format="RGBNORM")
        np.testing.assert_allclose(norm, rgb / 255.0)
        two_d = cltcol.harmonize_colors(np.array([[0, 0, 0], [255, 255, 255]]), "rgb")
        assert two_d.shape == (2, 3)

    def test_whole_number_float_colors(self):
        # Colors accepted by is_color_like as 0-255 values must be converted, not
        # rejected or left unnormalized
        assert cltcol.harmonize_colors(np.array([70.0, 130.0, 180.0])) == ["#4682b4"]
        np.testing.assert_allclose(
            cltcol.harmonize_colors([[255.0, 128.0, 0.0]], output_format="rgbnorm"),
            [[1.0, 128 / 255, 0.0]],
        )

    def test_errors(self):
        with pytest.raises(ValueError):
            cltcol.harmonize_colors(["#ff0000"], output_format="cmyk")
        with pytest.raises(ValueError):
            cltcol.harmonize_colors([[300, 0, 0]])
        with pytest.raises(ValueError):
            cltcol.harmonize_colors(np.zeros((2, 2, 3)))
        with pytest.raises(TypeError):
            cltcol.harmonize_colors(5)

    def test_readjust_colors(self):
        colors = ["#FF5733", [255, 87, 51], np.array([51, 87, 255])]
        assert cltcol.readjust_colors(colors, "hex") == [
            "#ff5733",
            "#ff5733",
            "#3357ff",
        ]
        assert cltcol.readjust_colors(colors).tolist() == [
            [255, 87, 51],
            [255, 87, 51],
            [51, 87, 255],
        ]
        with pytest.raises(ValueError):
            cltcol.readjust_colors(colors, "lab")


####################################################################################################
# Color generation
####################################################################################################
class TestColorGeneration:
    @pytest.mark.parametrize("fmt", ["rgb", "rgbnorm", "hex"])
    def test_create_random_colors(self, fmt):
        colors = cltcol.create_random_colors(5, output_format=fmt, random_seed=1)
        assert len(colors) == 5
        again = cltcol.create_random_colors(5, output_format=fmt, random_seed=1)
        assert np.array_equal(np.asarray(colors), np.asarray(again))

    def test_create_random_colors_from_cmap(self):
        assert cltcol.create_random_colors(3, "hex", cmap="gray") == [
            "#000000",
            "#808080",
            "#ffffff",
        ]
        assert cltcol.create_random_colors(1, "rgb", cmap="gray").shape == (1, 3)

    def test_create_random_colors_errors(self):
        with pytest.raises(TypeError):
            cltcol.create_random_colors(2.0)
        with pytest.raises(ValueError):
            cltcol.create_random_colors(0)
        with pytest.raises(ValueError):
            cltcol.create_random_colors(2, output_format="cmyk")
        with pytest.raises(ValueError):
            cltcol.create_random_colors(2, cmap="not_a_colormap")

    def test_create_distinguishable_colors(self):
        rgb = cltcol.create_distinguishable_colors(8, random_seed=0)
        assert rgb.shape == (8, 3) and rgb.min() >= 0 and rgb.max() <= 255
        assert len({tuple(c) for c in rgb}) == 8
        hexes = cltcol.create_distinguishable_colors(4, output_format="hex")
        assert all(cltcol.is_valid_hex_color(h) for h in hexes)
        norm = cltcol.create_distinguishable_colors(3, output_format="rgbnorm")
        assert norm.max() <= 1.0
        assert np.array_equal(
            cltcol.create_distinguishable_colors(5, random_seed=3),
            cltcol.create_distinguishable_colors(5, random_seed=3),
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n": 0},
            {"n": 3, "output_format": "cmyk"},
            {"n": 3, "lightness_range": (0.9, 0.1)},
            {"n": 3, "saturation_range": (0, 1.5)},
        ],
    )
    def test_create_distinguishable_colors_errors(self, kwargs):
        with pytest.raises(ValueError):
            cltcol.create_distinguishable_colors(**kwargs)

    def test_get_predefined_distinguishable_colors(self):
        assert cltcol.get_predefined_distinguishable_colors(2, "hex") == [
            "#F3C300",
            "#875692",
        ]
        assert cltcol.get_predefined_distinguishable_colors(1).tolist() == [
            [243, 195, 0]
        ]
        assert cltcol.get_predefined_distinguishable_colors(3, "rgbnorm").max() <= 1
        with pytest.raises(ValueError):
            cltcol.get_predefined_distinguishable_colors(21)

    def test_get_colormaps_names(self):
        assert cltcol.get_colormaps_names(3) == ["viridis", "jet", "copper"]
        assert cltcol.get_colormaps_names(3, cmap_type="diverging") == [
            "PiYG",
            "PRGn",
            "BrBG",
        ]
        many = cltcol.get_colormaps_names(40, cmap_type="diverging")
        assert len(many) == 40 and many[12] == "PiYG"  # The list repeats
        with pytest.raises(ValueError):
            cltcol.get_colormaps_names(2, cmap_type="qualitative")

    def test_create_lut_dictionary(self):
        lut = cltcol.create_lut_dictionary([0, 3, 7])
        assert lut["index"] == [3, 7]
        assert lut["name"] == ["Region_3", "Region_7"]
        assert len(lut["color"]) == 2

    def test_generate_colortable(self):
        ctab = cltcol.generate_colortable(3)
        assert ctab["index"] == [1, 2, 3]
        assert ctab["name"][0] == "auto-roi-000001"
        assert ctab["opacity"] == [1.0] * 3 and ctab["headerlines"] == []
        with pytest.raises(ValueError):
            cltcol.generate_colortable(0)


####################################################################################################
# Values, color tables and plots
####################################################################################################
class TestValuesToColors:
    def test_hex_and_rgb(self):
        assert cltcol.values2colors([0, 10], cmap="gray") == ["#000000", "#ffffff"]
        rgb = cltcol.values2colors([0, 5, 10], cmap="gray", output_format="rgb")
        assert rgb[0].tolist() == [0, 0, 0] and rgb[2].tolist() == [255, 255, 255]
        norm = cltcol.values2colors(
            np.array([0, 10]), cmap="gray", output_format="rgbnorm"
        )
        np.testing.assert_allclose(norm[1], [1.0, 1.0, 1.0])

    def test_single_value_and_inversions(self):
        assert len(cltcol.values2colors([5.0])) == 1
        assert cltcol.values2colors([0, 10], cmap="gray", invert_clmap=True) == [
            "#ffffff",
            "#000000",
        ]
        assert cltcol.values2colors([0, 10], cmap="gray", invert_cl=True) == [
            "#ffffff",
            "#000000",
        ]

    def test_vmin_vmax_and_range(self):
        assert cltcol.values2colors([5], cmap="gray", vmin=0, vmax=10) == ["#808080"]
        rgb = cltcol.values2colors(
            [0, 5, 10], cmap="gray", output_format="rgb", range_min=1, range_max=9
        )
        assert rgb[0].tolist() == [200, 200, 200] and rgb[2].tolist() == [200, 200, 200]
        out = cltcol.values2colors([0, 10], range_min=50, range_color="#000000")
        assert out == ["#000000", "#000000"]

    def test_errors(self):
        with pytest.raises(TypeError):
            cltcol.values2colors("abc")
        with pytest.raises(ValueError):
            cltcol.values2colors([])
        with pytest.raises(ValueError):
            cltcol.values2colors([1, 2], output_format="cmyk")
        with pytest.raises(ValueError):
            cltcol.values2colors([1, 2], cmap="not_a_colormap")


class TestColorTables:
    def test_colors_to_table(self):
        table = cltcol.colors_to_table(["#ff0000", [0, 255, 0]])
        assert table.shape == (2, 5)
        assert table[:, 3].tolist() == [0, 0]  # Default alpha
        assert table[:, 4].tolist() == [255, 255 * 256]  # Packed RGB values
        table = cltcol.colors_to_table(
            ["#ff0000", "#00ff00"], alpha_values=255, values=[1, 2]
        )
        assert table[:, 3].tolist() == [255, 255] and table[:, 4].tolist() == [1, 2]

    def test_colors_to_table_errors(self):
        with pytest.raises(ValueError):
            cltcol.colors_to_table("#ff0000")
        with pytest.raises(ValueError):
            cltcol.colors_to_table(["#ff0000"], values=[1, 2])
        with pytest.raises(ValueError):
            cltcol.colors_to_table(["#ff0000", "#00ff00"], alpha_values=[1, 2, 3])

    def test_get_colors_from_colortable(self):
        ctab = cltcol.colors_to_table(
            ["#ff0000", "#00ff00"], alpha_values=255, values=[1, 2]
        )
        colors = cltcol.get_colors_from_colortable(np.array([1, 2, 9]), ctab)
        assert colors.tolist() == [
            [255, 0, 0, 255],
            [0, 255, 0, 255],
            [240, 240, 240, 0],  # Unlabeled vertices are gray
        ]
        with pytest.raises(ValueError):
            cltcol.get_colors_from_colortable(np.array([1]), ctab[:, :4])

    def test_colortable_visualization(self, tmp_path, ctab_dict, lut_file):
        fig = cltcol.colortable_visualization(
            [[255, 0, 0], [0, 255, 0]], ["Region 1", "Region 2"], columns=1
        )
        assert isinstance(fig, plt.Figure)
        out = tmp_path / "figs" / "ctab.png"
        cltcol.colortable_visualization(
            ctab_dict, export_path=str(out), alternating_bg=True
        )
        assert out.is_file()
        assert isinstance(cltcol.colortable_visualization(lut_file), plt.Figure)
        loader = cltcol.ColorTableLoader(ctab_dict)
        assert isinstance(cltcol.colortable_visualization(loader), plt.Figure)

    def test_colortable_visualization_errors(self, tmp_path):
        with pytest.raises(ValueError):
            cltcol.colortable_visualization([[255, 0, 0]])  # Names are required
        with pytest.raises(ValueError):
            cltcol.colortable_visualization([[255, 0, 0]], ["a", "b"])
        with pytest.raises(TypeError):
            cltcol.colortable_visualization([[255, 0, 0]], [1])
        with pytest.raises(FileNotFoundError):
            cltcol.colortable_visualization(tmp_path / "missing.lut")

    def test_visualize_colors(self):
        cltcol.visualize_colors(["#FF5733", [51, 255, 87]], label_position="above")
        assert plt.get_fignums()
        cltcol.visualize_colors([])  # Nothing to draw


####################################################################################################
# ColorTableLoader
####################################################################################################
class TestColorTableLoaderReading:
    def test_read_luttable(self, lut_file):
        lut = cltcol.ColorTableLoader.read_luttable(str(lut_file))
        assert lut["index"] == [0, 1, 2, 3]
        assert lut["name"][1] == "Left-Hippocampus"
        assert lut["color"][1] == "#dcd814"
        assert lut["opacity"] == [0, 0, 128, 255]  # Raw values from the file
        assert lut["headerlines"] == ["Atlas test LUT", "Created for the tests"]
        hippo = cltcol.ColorTableLoader.read_luttable(str(lut_file), "hippocampus")
        assert hippo["index"] == [1, 2]

    def test_read_tsvtable(self, tsv_file, tmp_path):
        tsv = cltcol.ColorTableLoader.read_tsvtable(str(tsv_file), filter_by_name="lh")
        assert tsv["index"] == [1, 2] and tsv["opacity"] == [1.0, 0.5]
        bad = tmp_path / "bad.tsv"
        bad.write_text("code\tlabel\n1\ta\n")
        with pytest.raises(ValueError, match="missing required columns"):
            cltcol.ColorTableLoader.read_tsvtable(str(bad))
        with pytest.raises(FileNotFoundError):
            cltcol.ColorTableLoader.read_tsvtable(str(tmp_path / "missing.tsv"))

    def test_load_colortable_lut_opacity(self, lut_file):
        # LUT alpha values are 0-255; 0 means fully opaque in most neuroimaging tools
        lut = cltcol.ColorTableLoader.load_colortable(lut_file)
        np.testing.assert_allclose(lut["opacity"], [1.0, 1.0, 128 / 255, 1.0])

    def test_load_colortable_formats(self, tmp_path, tsv_file):
        assert cltcol.ColorTableLoader.load_colortable(tsv_file)["opacity"] == [
            1.0,
            0.5,
            1.0,
        ]
        tsv_txt = tmp_path / "table.txt"  # Tab-separated with header, .txt extension
        tsv_txt.write_text("index\tname\tcolor\n1\ta\t#ff0000\n")
        assert cltcol.ColorTableLoader.load_colortable(tsv_txt)["name"] == ["a"]
        no_alpha = tmp_path / "no_alpha"  # Extensionless LUT without the alpha column
        no_alpha.write_text("1 a 1 2 3\n")
        lut = cltcol.ColorTableLoader.load_colortable(no_alpha)
        assert lut["color"] == ["#010203"] and lut["opacity"] == [1]

    def test_load_colortable_errors(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            cltcol.ColorTableLoader.load_colortable(tmp_path / "missing.lut")
        csv = tmp_path / "table.csv"
        csv.write_text("index,name\n")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            cltcol.ColorTableLoader.load_colortable(csv)
        empty = tmp_path / "empty.lut"
        empty.write_text("\n")
        with pytest.raises(ValueError, match="empty"):
            cltcol.ColorTableLoader.load_colortable(empty)
        comments = tmp_path / "comments.lut"
        comments.write_text("# only comments\n")
        with pytest.raises(ValueError, match="only comments"):
            cltcol.ColorTableLoader.load_colortable(comments)

    def test_init_from_files_and_dicts(self, lut_file, tsv_file, ctab_dict):
        assert cltcol.ColorTableLoader(lut_file).index == [0, 1, 2, 3]
        assert cltcol.ColorTableLoader(str(tsv_file)).name[0] == "ctx-lh-a"
        loader = cltcol.ColorTableLoader(ctab_dict)
        assert loader.opacity == [1.0, 0.5, 1.0] and loader.headerlines == []
        minimal = cltcol.ColorTableLoader({"index": [4, 5]})
        assert minimal.name == ["Region_4", "Region_5"]
        assert len(minimal.color) == 2 and minimal.opacity == [1.0, 1.0]

    def test_init_errors(self, ctab_dict):
        with pytest.raises(ValueError, match="index"):
            cltcol.ColorTableLoader({"name": ["a"]})
        with pytest.raises(ValueError, match="Length"):
            cltcol.ColorTableLoader({**ctab_dict, "name": ["a"]})
        with pytest.raises(ValueError):
            cltcol.ColorTableLoader(5)


class TestColorTableLoaderWriting:
    def test_lut_round_trip(self, tmp_path, ctab_dict):
        loader = cltcol.ColorTableLoader(ctab_dict)
        out = tmp_path / "out.lut"
        assert loader.export(out, out_format="lut") is None
        reloaded = cltcol.ColorTableLoader(out)
        assert reloaded.index == ctab_dict["index"]
        assert reloaded.name == ctab_dict["name"]
        assert reloaded.color == ctab_dict["color"]
        np.testing.assert_allclose(reloaded.opacity, ctab_dict["opacity"], atol=1 / 255)

    def test_export_to_lutctab(self, tmp_path, ctab_dict):
        loader = cltcol.ColorTableLoader(ctab_dict)
        lines = loader.export_to_lutctab(headerlines="# my header")
        assert lines[0] == "# my header" and lines[2].startswith("#No.")
        assert lines[4].split() == ["1", "ctx-lh-a", "255", "0", "0", "255"]
        assert lines[5].split()[-1] == "127"  # Opacity 0.5 written as 0-255
        out = tmp_path / "out.lut"
        assert loader.export_to_lutctab(out) == str(out)
        with pytest.raises(FileExistsError):
            loader.export_to_lutctab(out)
        loader.export_to_lutctab(out, append=True, headerlines=["# appended"])
        text = out.read_text()
        assert text.count("ctx-lh-a") == 2 and "# appended" in text
        with pytest.raises(FileNotFoundError):
            loader.export_to_lutctab(tmp_path / "missing" / "out.lut")
        with pytest.raises(ValueError):
            loader.export_to_lutctab(headerlines=5)

    def test_export_tsv_and_nilearn(self, tmp_path, ctab_dict):
        loader = cltcol.ColorTableLoader(ctab_dict)
        df = loader.export_to_tsvctab()
        assert isinstance(df, pd.DataFrame)
        assert df.columns.tolist() == ["index", "name", "color", "opacity"]
        out = tmp_path / "out.tsv"
        loader.export(out, out_format="tsv")
        assert cltcol.ColorTableLoader(out).opacity == [1.0, 0.5, 1.0]
        with pytest.raises(FileExistsError):
            loader.export_to_tsvctab(out)
        nil = tmp_path / "nilearn.tsv"
        loader.export(nil, out_format="nilearn", overwrite=False)
        assert pd.read_csv(nil, sep="\t").columns.tolist() == ["index", "name", "color"]

    def test_export_fsl(self, tmp_path, ctab_dict):
        loader = cltcol.ColorTableLoader(ctab_dict)
        out = tmp_path / "fsl.txt"
        loader.export(out, out_format="fsl", overwrite=False)  # New file
        first = out.read_text().splitlines()[0].split()
        assert first == ["1", "1.00000", "0.00000", "0.00000", "ctx-lh-a"]
        with pytest.raises(FileExistsError):
            loader.export_to_fslctab(out, overwrite=False)
        with pytest.raises(FileNotFoundError):
            loader.export_to_fslctab(tmp_path / "missing" / "fsl.txt")

    def test_export_unknown_format(self, tmp_path, ctab_dict):
        with pytest.raises(ValueError, match="Unsupported output format"):
            cltcol.ColorTableLoader(ctab_dict).export(tmp_path / "x", out_format="csv")

    def test_write_luttable(self, tmp_path, ctab_dict):
        lines = cltcol.ColorTableLoader.write_luttable(ctab_dict)
        assert any(line.split()[:2] == ["2", "ctx-lh-b"] for line in lines)
        out = tmp_path / "sub" / "legacy.lut"
        cltcol.ColorTableLoader.write_luttable(ctab_dict, str(out))
        assert cltcol.ColorTableLoader(out).name == ctab_dict["name"]

    def test_write_tsvtable(self, tmp_path, ctab_dict):
        out = tmp_path / "regions.tsv"
        data = {k: ctab_dict[k] for k in ("index", "name", "color")}
        assert cltcol.ColorTableLoader.write_tsvtable(data, str(out)) == str(out)
        rgb = {"index": [9], "name": ["x"], "color": [[0, 0, 255]]}
        cltcol.ColorTableLoader.write_tsvtable(rgb, str(tmp_path / "rgb.tsv"))
        assert pd.read_csv(tmp_path / "rgb.tsv", sep="\t")["color"].tolist() == [
            "#0000ff"
        ]
        with pytest.raises(ValueError, match="index"):
            cltcol.ColorTableLoader.write_tsvtable({"name": ["a"]}, str(out))
        with pytest.raises(ValueError, match="hexadecimal"):
            cltcol.ColorTableLoader.write_tsvtable(
                {"index": [1], "name": ["a"], "color": ["red"]}, str(tmp_path / "c.tsv")
            )

    def test_write_tsvtable_append(self, tmp_path):
        out = tmp_path / "regions.tsv"
        cltcol.ColorTableLoader.write_tsvtable(
            {"index": [1], "name": ["a"], "color": ["#ff0000"]}, str(out)
        )
        cltcol.ColorTableLoader.write_tsvtable(
            {"index": [2], "name": ["b"], "color": ["#00ff00"]},
            str(out),
            boolappend=True,
        )
        assert cltcol.ColorTableLoader.read_tsvtable(str(out))["name"] == ["a", "b"]
        with pytest.raises(ValueError, match="Cannot append"):
            cltcol.ColorTableLoader.write_tsvtable(
                {"index": [1], "name": ["a"]},
                str(tmp_path / "new.tsv"),
                boolappend=True,
            )

    def test_write_tsvtable_existing_file(self, tmp_path):
        out = tmp_path / "regions.tsv"
        cltcol.ColorTableLoader.write_tsvtable({"index": [1], "name": ["a"]}, str(out))
        # Without overwrite an existing file is kept
        cltcol.ColorTableLoader.write_tsvtable({"index": [2], "name": ["b"]}, str(out))
        assert cltcol.ColorTableLoader.read_tsvtable(str(out))["name"] == ["a"]
        cltcol.ColorTableLoader.write_tsvtable(
            {"index": [2], "name": ["b"]}, str(out), overwrite=True
        )
        assert cltcol.ColorTableLoader.read_tsvtable(str(out))["name"] == ["b"]
