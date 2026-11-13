"""Tests for clabtoolkit.morphometrytools_dashboard (the Streamlit dashboard)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

import clabtoolkit.morphometrytools_dashboard as dash  # noqa: E402

SCRIPT = str(Path(dash.__file__))


####################################################################################################
# Helpers and fixtures
####################################################################################################
@pytest.fixture
def table(tmp_path):
    """Two models with different metrics, 4 participants, 2 hemispheres and 40 regions."""
    metrics = {"DTI": ["fa", "md"], "noddi": ["ndi", "odi"]}
    rng = np.random.default_rng(0)
    rows = [
        {
            "Participant": participant,
            "Hemisphere": hemi,
            "Model": model,
            "Metric": metric,
            "Supraregion": "ctx",
            "Region": f"ctx-{hemi}-region{region:02d}",
            "Value": rng.random(),
            "Median": rng.random(),
            "Empty": np.nan,
        }
        for participant in ("P01", "P02", "P03", "P04")
        for hemi in ("lh", "rh")
        for model, model_metrics in metrics.items()
        for metric in model_metrics
        for region in range(40)
    ]
    path = tmp_path / "table.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def open_dashboard(table):
    app = AppTest.from_file(SCRIPT, default_timeout=120).run()
    app.text_input[0].set_value(table).run()
    assert not app.exception
    return app


def add_filter(app, column):
    app.selectbox(key="var_to_add").set_value(column).run()
    next(b for b in app.button if "Add filter" in b.label).click().run()
    assert not app.exception
    return app.multiselect(key=f"flt_{column}")


def selector(app, label):
    return next(s for s in app.selectbox if s.label == label)


####################################################################################################
# Loading of the table
####################################################################################################
class TestTableHelpers:
    def test_drop_empty_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [np.nan, np.nan], "c": ["", " "]})
        assert list(dash.drop_empty_columns(df).columns) == ["a"]

    def test_shorten_region_names(self):
        df = pd.DataFrame(
            {
                "Supraregion": ["ctx", "ctx", "brain", "thal", "ctx", None],
                "Hemisphere": ["lh", "rh", "brain", "lh", "lh", "lh"],
                "Region": [
                    "ctx-lh-insula",
                    "ctx-rh-insula",
                    "brain-brain-wholebrain",
                    "ctx-lh-pulvinar",  # prefix of another supra-region
                    "ctx-lh-",  # nothing left without the prefix
                    "ctx-lh-cuneus",  # supra-region unknown
                ],
            }
        )
        out = dash.shorten_region_names(df)
        assert out["Region"].tolist() == [
            "insula",
            "insula",
            "wholebrain",
            "ctx-lh-pulvinar",
            "ctx-lh-",
            "ctx-lh-cuneus",
        ]
        # The input table is not modified
        assert df["Region"].iloc[0] == "ctx-lh-insula"

    def test_shorten_region_names_needs_the_three_columns(self):
        df = pd.DataFrame({"Hemisphere": ["lh"], "Region": ["ctx-lh-insula"]})
        assert dash.shorten_region_names(df) is df

    def test_with_alpha(self):
        assert dash.with_alpha("#FF7F0E", 0.5) == "rgba(255,127,14,0.5)"
        assert dash.with_alpha("rgb(48, 18, 59)", 0.25) == "rgba(48,18,59,0.25)"


####################################################################################################
# Dashboard
####################################################################################################
class TestDashboard:
    def test_needs_a_table(self):
        app = AppTest.from_file(SCRIPT, default_timeout=120).run()
        assert not app.exception
        assert "Upload a table" in app.info[0].value

    def test_overview(self, table):
        app = open_dashboard(table)
        counts = {m.label: m.value for m in app.metric}
        # 40 regions: the supra-region and the hemisphere are removed from their names
        assert counts["Regions"] == "40"
        assert counts["Rows"] == "1,280"
        # The empty column is discarded
        assert "Empty" not in app.selectbox(key="var_to_add").options

    def test_filters_offer_the_values_left_by_the_other_filters(self, table):
        app = open_dashboard(table)
        model = add_filter(app, "Model")
        assert model.options == ["DTI", "noddi"]
        model.set_value(["DTI"]).run()

        metric = add_filter(app, "Metric")
        assert len(metric.options) == 2  # only the metrics of DTI

        rows = next(m for m in app.metric if m.label == "Rows")
        assert rows.delta == "640 after filters"

        # ...and the other way around
        metric.set_value([metric.options[0]]).run()
        assert app.multiselect(key="flt_Model").options == ["DTI"]

    def test_filter_without_values_keeps_all_the_rows(self, table):
        app = open_dashboard(table)
        add_filter(app, "Model")
        rows = next(m for m in app.metric if m.label == "Rows")
        assert rows.delta is None or "after filters" not in rows.delta

    def test_selection_is_pruned_when_it_is_no_longer_available(self, table):
        app = open_dashboard(table)
        add_filter(app, "Model").set_value(["DTI"]).run()
        add_filter(app, "Metric").set_value(["fa"]).run()

        # noddi has no "fa": Model cannot offer it while Metric = fa
        app.session_state["flt_Model"] = ["noddi"]
        app.run()
        assert not app.exception
        assert app.multiselect(key="flt_Model").value == []

    def test_region_can_group_and_separate_the_plots(self, table):
        app = open_dashboard(table)
        assert "Region" in selector(app, "Group by (color)").options
        assert "Region" in selector(app, "Separate plots by").options

    @pytest.mark.parametrize("kind", dash.PLOT_TYPES)
    def test_plots_separated_in_panels(self, table, kind):
        app = open_dashboard(table)
        app.selectbox(key="plot_kind").set_value(kind).run()
        assert not app.exception
        split = selector(app, "Separate plots by")
        # Region is the variable correlated in the correlation plot
        split.set_value("Region" if "Region" in split.options else "Hemisphere").run()
        assert not app.exception
        assert len(app.get("plotly_chart")) == 1

    @pytest.mark.parametrize("kind", ["Box plot", "Violin plot"])
    def test_boxes_have_the_color_of_their_points(self, table, kind):
        import json

        app = open_dashboard(table)
        app.selectbox(key="plot_kind").set_value(kind).run()
        selector(app, "Group by (color)").set_value("Participant").run()
        assert not app.exception

        traces = json.loads(app.get("plotly_chart")[0].proto.spec)["data"]
        shapes = [
            t
            for t in traces
            if t["type"] == kind.split()[0].lower() and not t.get("jitter")
        ]
        fills = [t["fillcolor"] for t in shapes]
        assert len(shapes) == 4 and len(set(fills)) == 4
        for trace in shapes:
            assert trace["fillcolor"] == dash.with_alpha(trace["marker"]["color"], 0.55)
