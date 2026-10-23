"""Notebook (HTML) and terminal output of the inspection helpers of misctools.

The IPython display call is replaced by a recorder, so no notebook is needed.
"""

import importlib
import re
import types

import h5py
import numpy as np
import pytest

import clabtoolkit.misctools as cltmisc

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text)


class Base:
    """A base class."""

    def inherited_method(self):
        """Inherited from the base."""


class Example(Base):
    """An example class.

    With a longer description.
    """

    class_attribute = "x" * 80
    number = 3

    def __init__(self):
        self.value = [1, 2, 3]
        self._secret = 1

    @property
    def size(self):
        """The size."""
        return 3

    def load(self, path: str, retries: int = 2, *args, **kwargs) -> None:
        """Load something."""

    def _helper(self):
        """Private."""

    @staticmethod
    def no_doc():
        pass


def example_function(name: str, age=25):
    """Do something with a name."""


def make_module():
    module = types.ModuleType("fake_display_module", "A fake module.")
    module.Example = Example
    module.example_function = example_function
    module._private_function = lambda: None
    Example.__module__ = example_function.__module__ = "fake_display_module"
    return module


@pytest.fixture
def notebook(monkeypatch):
    """Notebook mode, recording the HTML sent to IPython.display."""
    html = []
    monkeypatch.setattr(cltmisc, "is_notebook", lambda: True)
    monkeypatch.setattr(cltmisc, "display", lambda obj: html.append(obj.data))
    yield html
    Example.__module__ = example_function.__module__ = __name__


@pytest.fixture(autouse=True)
def restore_modules():
    yield
    Example.__module__ = example_function.__module__ = __name__


######################################################################################################
class TestShowModuleContents:
    def test_terminal_options(self, capsys):
        module = make_module()
        cltmisc.show_module_contents(module, show_private=True, show_inherited=False)
        out = strip_ansi(capsys.readouterr().out)
        assert "Example" in out and "example_function" in out
        assert "inherited_method" not in out

    def test_terminal_with_inherited_methods(self, capsys):
        cltmisc.show_module_contents(make_module(), show_inherited=True)
        assert "inherited_method" in strip_ansi(capsys.readouterr().out)

    def test_notebook(self, notebook):
        cltmisc.show_module_contents(make_module())
        assert notebook and "Example" in notebook[0] and "<" in notebook[0]

    @pytest.mark.parametrize("private", [True, False])
    @pytest.mark.parametrize("inherited", [True, False])
    def test_notebook_options(self, notebook, private, inherited):
        cltmisc.show_module_contents(
            make_module(), show_private=private, show_inherited=inherited
        )
        html = "".join(notebook)
        assert ("inherited_method" in html) == inherited

    def test_notebook_of_a_real_module_by_name(self, notebook):
        cltmisc.show_module_contents("clabtoolkit.colorstools")
        assert "values2colors" in "".join(notebook)

    @pytest.mark.parametrize("in_notebook", [True, False])
    def test_errors(self, request, monkeypatch, capsys, in_notebook):
        html = []
        if in_notebook:
            monkeypatch.setattr(cltmisc, "is_notebook", lambda: True)
            monkeypatch.setattr(cltmisc, "display", lambda obj: html.append(obj.data))

        def message():
            return strip_ansi(capsys.readouterr().out) + "".join(html)

        cltmisc.show_module_contents("no_such_module_at_all")
        assert "could not be imported" in message()
        cltmisc.show_module_contents(5)
        assert "Invalid input" in message()

        def explode(name):
            raise RuntimeError("broken import")

        monkeypatch.setattr(importlib, "import_module", explode)
        cltmisc.show_module_contents("another_missing_module")
        assert "Error importing module" in message()


class TestShowObjectContent:
    def test_instance_with_everything(self, capsys):
        cltmisc.show_object_content(Example(), show_private=True, show_dunder=True)
        out = strip_ansi(capsys.readouterr().out)
        assert (
            "load" in out and "size" in out and "_secret" in out and "__init__" in out
        )

    def test_a_class_shows_its_mro(self, capsys):
        cltmisc.show_object_content(Example)
        out = strip_ansi(capsys.readouterr().out)
        assert "Base" in out and "class_attribute" in out

    @pytest.mark.parametrize(
        "flags",
        [
            {"show_methods": False},
            {"show_properties": False},
            {"show_attributes": False},
            {"show_methods": False, "show_properties": False},
        ],
    )
    def test_terminal_flags(self, capsys, flags):
        cltmisc.show_object_content(Example(), **flags)
        assert capsys.readouterr().out

    def test_function_and_module(self, capsys):
        cltmisc.show_object_content(example_function)
        cltmisc.show_object_content(np)
        assert capsys.readouterr().out

    def test_notebook(self, notebook):
        cltmisc.show_object_content(Example(), show_private=True, show_dunder=True)
        html = "".join(notebook)
        assert "load" in html and "size" in html and "<" in html

    @pytest.mark.parametrize(
        "flags",
        [
            {"show_methods": False},
            {"show_properties": False},
            {"show_attributes": False},
        ],
    )
    def test_notebook_flags_and_class(self, notebook, flags):
        cltmisc.show_object_content(Example, **flags)
        cltmisc.show_object_content(Example(), **flags)
        assert notebook


class TestH5Explorer:
    @pytest.fixture
    def path(self, tmp_path):
        path = tmp_path / "data.h5"
        with h5py.File(path, "w") as f:
            f.attrs.update({f"attr{i}": i for i in range(8)})
            f.attrs["text"] = "hello"
            f.attrs["array"] = np.arange(4)
            for i in range(6):
                f.create_dataset(f"d{i}", data=np.arange(i + 3))
            f.create_dataset("scalar", data=7)
            f.create_dataset("strings", data=np.array([b"a", b"b"]))
            f.create_dataset("big", data=np.zeros((50, 50)), compression="gzip")
            group = f.create_group("outer")
            group.create_group("inner").create_dataset("x", data=np.ones(3))
            group.attrs["units"] = "mm"
        return str(path)

    def test_limits_and_hidden_values(self, path, capsys):
        stats = cltmisc.h5explorer(
            path, max_datasets_per_group=3, max_attrs=2, show_values=False
        )
        out = strip_ansi(capsys.readouterr().out)
        assert stats["groups"] == 2 and stats["datasets"] >= 9
        assert "more" in out

    def test_shows_values_and_all_attributes(self, path, capsys):
        cltmisc.h5explorer(path, max_datasets_per_group=100, max_attrs=50)
        out = strip_ansi(capsys.readouterr().out)
        assert "units" in out and "scalar" in out and "inner" in out

    def test_simple_version_with_a_limit(self, path, capsys):
        cltmisc.h5explorer_simple(path, max_datasets_per_group=2)
        assert capsys.readouterr().out
