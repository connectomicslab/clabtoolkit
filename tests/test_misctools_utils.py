"""Tests for clabtoolkit.misctools_utils (ExplorerDict, exist_file and exist_dir).

The notebook (HTML) output is checked through the IPython display calls, which are
replaced by a recorder: no notebook is needed.
"""

import re

import numpy as np
import pandas as pd
import pytest

import clabtoolkit.misctools_utils as mu
from clabtoolkit.misctools_utils import ExplorerDict

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text)


def sample(**kwargs) -> ExplorerDict:
    return ExplorerDict(
        {
            "subject": "sub-001",
            "age": 25,
            "scores": [1, 2, 3],
            "image": np.zeros((2, 3)),
            "small": np.array([1, 2]),
            "empty": np.array([]),
            "nested": {
                "Sessions": ["ses-01", "ses-02"],
                "pair": (1, {"deep": "x" * 80}),
                "none": None,
            },
        },
        **kwargs,
    )


@pytest.fixture(params=[True, False], ids=["with_colors", "plain"])
def colors(request, monkeypatch):
    monkeypatch.setattr(mu, "HAS_CLTCOLORS", request.param)
    return request.param


@pytest.fixture
def shown(monkeypatch):
    """Record what the notebook mode sends to IPython.display."""
    html = []
    monkeypatch.setattr(mu, "HAS_IPYTHON", True)
    monkeypatch.setattr(mu, "display", lambda obj: html.append(obj.data))
    return html


######################################################################################################
class TestColors:
    def test_the_color_module_is_found(self):
        assert mu.HAS_CLTCOLORS and mu.cltcolors.bcolors.ENDC

    def test_output_is_colored_by_default(self, capsys):
        sample(overwrite_mode="terminal").tree()
        assert "\x1b[" in capsys.readouterr().out
        sample(overwrite_mode="terminal").search("age")
        assert "\x1b[" in capsys.readouterr().out


######################################################################################################
class TestInit:
    def test_is_a_dict_with_a_name(self):
        data = ExplorerDict({"a": 1}, name="mine", overwrite_mode="terminal")
        assert data == {"a": 1} and data._name == "mine"
        assert not data._notebook_mode

    def test_notebook_mode_can_be_forced(self, shown):
        assert ExplorerDict(overwrite_mode="notebook")._notebook_mode

    def test_notebook_mode_needs_ipython(self, monkeypatch):
        monkeypatch.setattr(mu, "HAS_IPYTHON", False)
        assert not ExplorerDict(overwrite_mode="notebook")._notebook_mode

    def test_auto_detection_outside_a_notebook(self, monkeypatch):
        assert not ExplorerDict()._notebook_mode
        monkeypatch.setattr(mu, "HAS_IPYTHON", False)
        assert not ExplorerDict._is_notebook()


######################################################################################################
class TestToDataframe:
    def test_whole_dictionary(self):
        df = ExplorerDict({"a": [1, 2], "b": [3, 4]}).to_dataframe()
        assert isinstance(df, pd.DataFrame) and list(df.columns) == ["a", "b"]

    def test_exclude_and_include_keys(self):
        data = ExplorerDict({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        assert list(data.to_dataframe(exclude_keys=["b"]).columns) == ["a", "c"]
        assert list(data.to_dataframe(include_keys=["b"]).columns) == ["b"]


######################################################################################################
class TestTree:
    def test_terminal_tree(self, colors, capsys):
        sample(overwrite_mode="terminal").tree()
        out = strip_ansi(capsys.readouterr().out)
        assert "root" in out and "subject: sub-001" in out
        assert "shape=(2, 3)" in out and "[0]: 1" in out
        assert "..." in out  # the long string is truncated
        assert "empty array" in out

    def test_limits(self, colors, capsys):
        data = ExplorerDict(
            {f"k{i}": list(range(20)) for i in range(15)}, overwrite_mode="terminal"
        )
        data.tree(max_depth=1, max_items=3)
        out = strip_ansi(capsys.readouterr().out)
        assert "<max depth>" in out and "more items" in out

    def test_list_items_are_truncated(self, colors, capsys):
        ExplorerDict({"a": list(range(30))}, overwrite_mode="terminal").tree(
            max_items=4
        )
        assert "more items" in strip_ansi(capsys.readouterr().out)

    def test_without_type_information(self, colors, capsys):
        sample(overwrite_mode="terminal").tree(show_types=False, show_shapes=False)
        assert "dtype" not in capsys.readouterr().out

    def test_notebook_tree(self, shown):
        sample(overwrite_mode="notebook").tree(max_depth=3, max_items=2)
        assert shown and "<div" in shown[0] and "root" in shown[0]

    def test_notebook_tree_without_ipython_falls_back_to_the_terminal(
        self, monkeypatch, capsys
    ):
        data = sample(overwrite_mode="terminal")
        monkeypatch.setattr(mu, "HAS_IPYTHON", False)
        data._tree_html(None, 10, 50, True, True)
        assert "root" in capsys.readouterr().out


######################################################################################################
class TestSummary:
    def test_terminal_summary(self, colors, capsys):
        sample(overwrite_mode="terminal").summary()
        out = strip_ansi(capsys.readouterr().out)
        assert "Dictionary Summary: root" in out

    def test_verbose_terminal_summary(self, colors, capsys):
        sample(overwrite_mode="terminal").summary(verbose=True)
        assert strip_ansi(capsys.readouterr().out)

    def test_notebook_summary(self, shown):
        data = sample(overwrite_mode="notebook")
        data.summary()
        data.summary(verbose=True)
        assert len(shown) >= 2 and "<div" in shown[0]

    def test_statistics(self):
        stats = sample(overwrite_mode="terminal")._collect_stats(sample())
        assert stats["n_dicts"] == 3 and stats["n_lists"] == 3
        assert stats["max_depth"] == 4
        assert {"shape": (2, 3), "dtype": np.dtype(float), "size": 6} in stats["arrays"]
        assert stats["memory"] > 0

    @pytest.mark.parametrize(
        "size,text",
        [(512, "512.00 B"), (2048, "2.00 KB"), (3 * 1024**2, "3.00 MB")],
    )
    def test_format_bytes(self, size, text):
        assert ExplorerDict._format_bytes(size) == text

    def test_format_bytes_terabytes(self):
        assert ExplorerDict._format_bytes(5 * 1024**4) == "5.00 TB"


######################################################################################################
class TestStructure:
    def test_structure(self):
        structure = sample().structure(max_depth=None)
        assert structure["scores"] == ["<int>", "<int>", "<int>"]
        assert structure["image"] == "<ndarray: shape=(2, 3), dtype=float64>"
        assert structure["nested"]["pair"][1]["deep"] == "<str>"

    def test_max_depth(self):
        assert sample().structure(max_depth=1)["nested"] == "<dict>"

    def test_long_and_empty_lists(self):
        data = ExplorerDict({"long": list(range(10)), "none": [], "tup": ()})
        structure = data.structure()
        assert structure["long"] == ["<int>", "<... 8 more items ...>", "<int>"]
        assert structure["none"] == "<empty list>"
        assert structure["tup"] == "<empty tuple>"


######################################################################################################
class TestSearch:
    def test_finds_nested_keys_case_insensitively(self, colors, capsys):
        data = ExplorerDict({"a": {"Sessions": [{"session_id": 1}]}})
        assert data.search("session") == ["a.Sessions", "a.Sessions[0].session_id"]
        assert "Found 2 matches" in strip_ansi(capsys.readouterr().out)

    def test_case_sensitive(self, colors, capsys):
        data = ExplorerDict({"Session": 1, "session": 2})
        assert data.search("Session", case_sensitive=True) == ["Session"]

    def test_no_match(self, colors, capsys):
        assert ExplorerDict({"a": 1}).search("zzz") == []
        assert "No matches found" in strip_ansi(capsys.readouterr().out)

    def test_notebook_output(self, shown):
        data = ExplorerDict({"a": 1, "b": 2}, overwrite_mode="notebook")
        assert data.search("a") == ["a"]
        assert data.search("zzz") == []
        assert "Found 1 matches" in shown[0] and "No matches found" in shown[1]


######################################################################################################
class TestGetPath:
    def test_values_by_path(self):
        data = ExplorerDict({"a": {"b": [10, {"c": 3}]}})
        assert data.get_path("a.b[1].c") == 3
        assert data.get_path("a.b.0") == 10
        assert data.get_path("a") == {"b": [10, {"c": 3}]}

    def test_missing_path_is_reported(self, colors, capsys):
        assert ExplorerDict({"a": 1}).get_path("a.b.c") is None
        assert "Error accessing" in strip_ansi(capsys.readouterr().out)

    def test_missing_path_in_the_notebook(self, shown):
        data = ExplorerDict({"a": [1]}, overwrite_mode="notebook")
        assert data.get_path("a[5]") is None
        assert "Error accessing" in shown[0]


######################################################################################################
class TestExist:
    def test_exist_file(self, tmp_path):
        file = tmp_path / "f.txt"
        file.write_text("x")
        assert mu.exist_file(file) == str(file)
        assert mu.exist_file(tmp_path) == ""
        assert mu.exist_file(tmp_path / "missing.txt") == ""

    def test_exist_dir(self, tmp_path):
        assert mu.exist_dir(tmp_path) == str(tmp_path)
        assert mu.exist_dir(str(tmp_path / "missing")) == ""
        (tmp_path / "f.txt").write_text("x")
        assert mu.exist_dir(tmp_path / "f.txt") == ""
