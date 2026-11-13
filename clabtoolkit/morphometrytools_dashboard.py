"""
Streamlit dashboard to explore the tables produced by ``morphometrytools``.

Run it from the terminal with::

    clabtoolkit-morpho-dashboard /path/to/table.csv

from Python with::

    import clabtoolkit.morphometrytools as cltmorph
    cltmorph.launch_dashboard("/path/to/table.csv")

or directly with streamlit::

    streamlit run clabtoolkit/morphometrytools_dashboard.py -- --table /path/to/table.csv

It needs streamlit and plotly (``pip install "clabtoolkit[dashboard]"``).

Workflow
--------
1. Load a table (path or upload). It is read with ``misctools.smart_read_table``
   and every column that is entirely empty / NaN is discarded. The prefix
   ``<Supraregion>-<Hemisphere>-`` is removed from the names of the regions
   (``ctx-lh-superiorfrontal`` becomes ``superiorfrontal``).
2. In the left panel pick a variable from the drop-down and press
   **Add filter**. A categorical variable becomes a multi-select, a continuous
   one becomes a range slider. Filters can be removed individually. A
   multi-select without any value selected keeps all the values, and it only
   offers the values that exist in the rows kept by the other filters (e.g.
   after selecting a model, only the metrics of that model are offered).
3. The main panel shows the overview, the filtered table, summary statistics
   and an interactive plot.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import clabtoolkit.misctools as cltmisc
import clabtoolkit.morphometrytools as cltmorph

# A numeric column with at most this many distinct values is treated as categorical
MAX_NUMERIC_LEVELS = 12
MAX_POINTS = 6000  # jittered points drawn over box / violin / bar plots
# A variable with more distinct values than this cannot be used for the X axis,
# to group by color or to separate the plots
MAX_PLOT_LEVELS = 100
# Above this number of colors or panels the plot is drawn, but with a warning
CROWDED_LEVELS = 30
PLOT_TYPES = ["Box plot", "Violin plot", "Histogram", "Density", "Bar (mean ± std)", "Heatmap", "Correlation", "Scatter"]
CORR_METHODS = ["pearson", "spearman", "kendall"]
HEATMAP_AGGS = ["mean", "median", "std", "min", "max", "count"]
STAT_COLUMNS = ["Value", "Median", "Std", "Min", "Max"]
ALL = "All sessions"


# --------------------------------------------------------------------------- #
#                                Data helpers                                  #
# --------------------------------------------------------------------------- #
def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns with at least one value that is not NaN or an empty string."""
    as_text = df.astype("string").apply(lambda s: s.str.strip())
    keep = [c for c in df.columns if as_text[c].replace("", pd.NA).notna().any()]
    return df[keep]


def shorten_region_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the ``<Supraregion>-<Hemisphere>-`` prefix from the names of the regions
    (``ctx-lh-superiorfrontal`` -> ``superiorfrontal``). The supra-region and the
    hemisphere have their own columns, so the same region of both hemispheres gets
    the same name. The names without that prefix are left as they are.
    """
    if not {"Region", "Supraregion", "Hemisphere"} <= set(df.columns):
        return df

    def shorten(region, supra, hemi):
        if not isinstance(region, str) or pd.isna(supra) or pd.isna(hemi):
            return region
        prefix = f"{supra}-{hemi}-"
        return region[len(prefix):] if region.startswith(prefix) and len(region) > len(prefix) else region

    regions = [shorten(*row) for row in zip(df["Region"], df["Supraregion"], df["Hemisphere"])]
    return df.assign(Region=regions)


@st.cache_data(show_spinner="Reading table...")
def load_table(path: str) -> pd.DataFrame:
    """Read a table with ``smart_read_table``, drop its empty columns and shorten the region names."""
    return shorten_region_names(drop_empty_columns(cltmisc.smart_read_table(path)))


def metric_fullnames(df: pd.DataFrame) -> dict[str, str]:
    """``{metric: full name}`` for the metrics of the table (from config.json)."""
    if "Metric" not in df.columns:
        return {}
    names = df["Metric"].dropna().unique().tolist()
    return dict(zip(names, cltmorph.get_fullnames(names)))


def is_continuous(s: pd.Series) -> bool:
    """True for numeric columns with many distinct values."""
    return pd.api.types.is_numeric_dtype(s) and s.nunique() > MAX_NUMERIC_LEVELS


def filter_mask(df: pd.DataFrame, col: str) -> pd.Series | None:
    """Rows kept by the filter of a column, or None if the filter keeps all of them."""
    sel = st.session_state.get(f"flt_{col}")
    if is_continuous(df[col]):
        # The whole range is not a filter (so the rows without value are kept)
        if sel is None or (sel[0] <= df[col].min() and sel[1] >= df[col].max()):
            return None
        return df[col].between(*sel)
    # No value selected means all the values
    return df[col].isin(sel) if sel else None


def apply_filters(df: pd.DataFrame, filters: list[str], skip: str | None = None) -> pd.DataFrame:
    """Apply the widgets state of every active filter (except ``skip``) to the dataframe."""
    mask = pd.Series(True, index=df.index)
    for col in filters:
        if col == skip:
            continue
        col_mask = filter_mask(df, col)
        if col_mask is not None:
            mask &= col_mask
    return df[mask]


def sync_filters(df: pd.DataFrame) -> dict[str, list]:
    """
    Values that each categorical filter can offer: the ones present in the rows
    kept by all the other filters. The selected values that are no longer
    available are removed from the selection.
    """
    filters = st.session_state.filters
    categorical = [c for c in filters if not is_continuous(df[c])]
    options: dict[str, list] = {}
    # Removing a value from a filter can change the options of the other ones
    for _ in range(len(categorical) + 1):
        changed = False
        for col in categorical:
            kept_rows = apply_filters(df, filters, skip=col)
            options[col] = sorted(kept_rows[col].dropna().unique().tolist())
            sel = st.session_state.get(f"flt_{col}") or []
            valid = [v for v in sel if v in options[col]]
            if len(valid) != len(sel):
                st.session_state[f"flt_{col}"] = valid
                changed = True
        if not changed:
            break
    return options


# --------------------------------------------------------------------------- #
#                                  Sidebar                                     #
# --------------------------------------------------------------------------- #
def _remove_filter(col: str) -> None:
    st.session_state.filters.remove(col)
    st.session_state.pop(f"flt_{col}", None)


def _add_filter() -> None:
    col = st.session_state.get("var_to_add")
    if col and col not in st.session_state.filters:
        st.session_state.filters.append(col)


def sidebar_loader(default_path: str) -> str | None:
    """Sidebar block to select the input table. Returns the path to read."""
    st.header("Data")
    upload = st.file_uploader("Upload a table", type=["csv", "tsv", "txt"])
    path = st.text_input("...or path to a table", value=default_path)

    if upload is not None:
        # smart_read_table works on paths, so persist the upload once
        tmp = Path(tempfile.gettempdir()) / f"clab_dash_{upload.name}"
        tmp.write_bytes(upload.getvalue())
        return str(tmp)
    return path if path and Path(path).is_file() else None


def sidebar_filters(df: pd.DataFrame) -> None:
    """Sidebar block to add / edit / remove filters."""
    st.header("Filters")
    st.session_state.setdefault("filters", [])

    available = [c for c in df.columns if c not in st.session_state.filters]
    st.selectbox("Variable", available, key="var_to_add", index=0 if available else None)
    st.button("➕ Add filter", on_click=_add_filter, disabled=not available, width="stretch")

    options = sync_filters(df)
    names = metric_fullnames(df)
    for col in st.session_state.filters:
        s = df[col].dropna()
        with st.container(border=True):
            head, btn = st.columns([5, 1], vertical_alignment="center")
            head.markdown(f"**{col}**")
            btn.button("✖", key=f"rm_{col}", on_click=_remove_filter, args=(col,))
            if is_continuous(df[col]):
                lo, hi = float(s.min()), float(s.max())
                st.slider("range", lo, hi, (lo, hi), key=f"flt_{col}",
                          label_visibility="collapsed", format="%g")
            else:
                levels, total = options[col], s.nunique()
                col_names = names if col == "Metric" else {}
                st.multiselect("values", levels, key=f"flt_{col}",
                               label_visibility="collapsed",
                               placeholder=f"All ({len(levels):,})",
                               format_func=lambda v, n=col_names: n.get(v, v))
                if len(levels) < total:
                    st.caption(f"{len(levels):,} of {total:,} values exist with the other filters")


# --------------------------------------------------------------------------- #
#                                  Main panel                                  #
# --------------------------------------------------------------------------- #
def overview(df: pd.DataFrame, dff: pd.DataFrame) -> None:
    st.header("Overview")
    st.caption("Global counts for the whole table. The small note under each value "
               "shows how many remain after applying the filters.")

    # (label, column) - only the ones present in the table are shown
    counted = [("Participants", "Participant"), ("Sessions", "Session"),
               ("Metrics", "Metric"), ("Regions", "Region"),
               ("Hemispheres", "Hemisphere")]
    counted = [(lab, col) for lab, col in counted if col in df.columns]

    cols = st.columns(len(counted))
    for box, (label, col) in zip(cols, counted):
        total, kept = df[col].nunique(), dff[col].nunique()
        box.metric(label, f"{total:,}",
                   delta=f"{kept:,} after filters" if kept != total else None,
                   delta_color="off", delta_arrow="off")

    c1, c2, _ = st.columns([1, 1, 3])
    c1.metric("Rows", f"{len(df):,}",
              delta=f"{len(dff):,} after filters" if len(dff) != len(df) else None,
              delta_color="off", delta_arrow="off")
    c2.metric("Columns with data", len(df.columns))


def _categorical_options(d: pd.DataFrame, max_levels: int) -> list[str]:
    """Columns usable to group / split: not continuous, 2..max_levels distinct values."""
    return [c for c in d.columns
            if not is_continuous(d[c]) and 2 <= d[c].nunique() <= max_levels]


def _count_note(d: pd.DataFrame, col: str | None) -> str:
    """Small text with the number of distinct categories of a column."""
    if not col or col not in d.columns:
        return ""
    return f"{d[col].nunique():,} categories in {col}"


PALETTE = px.colors.qualitative.D3


def distinct_palette(n: int) -> list[str]:
    """Saturated, clearly different colors for ``n`` categories (10 -> D3, 24 -> Dark24, more -> Turbo)."""
    if n <= len(px.colors.qualitative.D3):
        return list(px.colors.qualitative.D3)
    if n <= len(px.colors.qualitative.Dark24):
        return list(px.colors.qualitative.Dark24)
    return px.colors.sample_colorscale("Turbo", [i / max(n - 1, 1) for i in range(n)])


def with_alpha(color: str, alpha: float) -> str:
    """The same color (``#rrggbb`` or ``rgb(r, g, b)``) with the given opacity."""
    r, g, b = px.colors.hex_to_rgb(color) if color.startswith("#") else px.colors.unlabel_rgb(color)
    return f"rgba({int(r)},{int(g)},{int(b)},{alpha})"


def theme_colors() -> dict:
    """Colors that follow the Streamlit light/dark theme."""
    try:
        dark = st.context.theme.type == "dark"
    except Exception:
        dark = False
    return {"dark": dark, "fg": "#f2f2f2" if dark else "#262730",
            "grid": "rgba(255,255,255,0.12)" if dark else "rgba(0,0,0,0.10)"}


def style_figure(fig: go.Figure, tc: dict, height: int, legend_title: str | None = None,
                 facets: bool = False) -> go.Figure:
    """Common look: transparent background, theme-aware text/lines, light grid."""
    fig.update_layout(
        template="plotly_dark" if tc["dark"] else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": tc["fg"]}, height=height, margin={"l": 10, "r": 10, "t": 40, "b": 10},
        legend_title_text=legend_title, hovermode="closest")
    fig.update_xaxes(linecolor=tc["fg"], gridcolor=tc["grid"], zeroline=False, automargin=True)
    fig.update_yaxes(linecolor=tc["fg"], gridcolor=tc["grid"], zeroline=False, automargin=True)
    if facets:  # "Session=01" -> "01"
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig


def kde_frame(data: pd.DataFrame, y: str, group: str | None, split: str | None,
              n_grid: int = 200, max_pts: int = 5000) -> pd.DataFrame:
    """Gaussian KDE of ``y`` for every (split, group) cell (no extrapolation beyond min/max)."""
    from scipy.stats import gaussian_kde

    keys = [k for k in (split, group) if k]
    rng = np.random.default_rng(0)
    rows = []
    for key, sub in (data.groupby(keys) if keys else [((), data)]):
        key = key if isinstance(key, tuple) else (key,)
        v = sub[y].to_numpy(dtype=float)
        if len(v) < 3 or np.ptp(v) == 0:
            continue
        if len(v) > max_pts:
            v = rng.choice(v, max_pts, replace=False)
        grid = np.linspace(v.min(), v.max(), n_grid)
        fr = pd.DataFrame({y: grid, "density": gaussian_kde(v)(grid)})
        for k, val in zip(keys, key):
            fr[k] = val
        rows.append(fr)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[y, "density", *keys])


def draw_matrices(panels: dict, xlabel: str, ylabel: str, colorscale: str, vmin: float, vmax: float,
                  cbar_label: str, ncols: int, fmt: str, tc: dict, lower_only: bool = False) -> go.Figure:
    """One interactive heatmap per panel on a shared color scale."""
    n = len(panels)
    ncols = max(1, min(ncols, n))
    nrows = -(-n // ncols)
    nr = max(m.shape[0] for m in panels.values())
    nc = max(m.shape[1] for m in panels.values())
    annot = nr * nc <= 150
    fig = make_subplots(rows=nrows, cols=ncols, horizontal_spacing=0.06,
                        vertical_spacing=min(0.12, 0.24 / nrows) if nrows > 1 else 0.0,
                        subplot_titles=[str(k) if k is not None else "" for k in panels])
    for i, (name, m) in enumerate(panels.items()):
        z = m.to_numpy(dtype=float)
        if lower_only:
            z = np.where(np.triu(np.ones(z.shape, dtype=bool), k=1), np.nan, z)
        trace = go.Heatmap(
            z=z, x=[str(c) for c in m.columns], y=[str(r) for r in m.index], coloraxis="coloraxis",
            xgap=1 if annot else 0, ygap=1 if annot else 0,
            hovertemplate=f"{xlabel}: %{{x}}<br>{ylabel}: %{{y}}<br>{cbar_label}: %{{z:.4g}}<extra></extra>")
        if annot:
            trace.update(texttemplate="%{z:" + fmt + "}", textfont={"size": 10})
        fig.add_trace(trace, row=i // ncols + 1, col=i % ncols + 1)
    fig.update_layout(coloraxis={"colorscale": colorscale, "cmin": vmin, "cmax": vmax,
                                 "colorbar": {"title": {"text": cbar_label}, "outlinewidth": 0, "thickness": 14}})
    fig.update_xaxes(type="category", tickmode="linear", dtick=1, tickangle=-90, title_text=xlabel,
                     tickfont={"size": 10 if nc <= 30 else 7}, showgrid=False)
    fig.update_yaxes(type="category", tickmode="linear", dtick=1, autorange="reversed", title_text=ylabel,
                     tickfont={"size": 10 if nr <= 30 else 7}, showgrid=False)
    height = nrows * int(np.clip(14 * nr + 170, 340, 1100))
    return style_figure(fig, tc, height)


def plot_section(dff: pd.DataFrame) -> None:
    st.header("Plot")
    if dff.empty:
        st.warning("No rows left after filtering.")
        return

    # ---- 1) what to plot: session, metric and statistic ------------------- #
    stats = [c for c in STAT_COLUMNS if c in dff.columns]
    if not stats:
        st.info("The table has none of the columns " + ", ".join(STAT_COLUMNS))
        return

    c1, c2, c3 = st.columns(3)
    d = dff
    if "Session" in d.columns:
        sessions = sorted(d["Session"].dropna().unique().tolist())
        ses = c1.selectbox("Session", [ALL] + sessions)
        c1.caption(f"{len(sessions):,} categories in Session")
        if ses != ALL:
            d = d[d["Session"] == ses]
    names = metric_fullnames(d)
    if "Metric" in d.columns:
        metrics = sorted(d["Metric"].dropna().unique().tolist(), key=lambda m: names[m].lower())
        show = lambda m: names.get(m, m)  # noqa: E731
        if st.session_state.get("plot_kind") in ("Heatmap", "Correlation"):
            # the heatmap can compare several metrics (e.g. Metric x Region)
            mets = c2.multiselect("Metrics", metrics, default=metrics[:1], format_func=show)
            c2.caption(f"{len(mets)} selected of {len(metrics):,} categories in Metric")
            if not mets:
                st.info("Select at least one metric.")
                return
        else:
            mets = [c2.selectbox("Metric", metrics, format_func=show)]
            c2.caption(f"{len(metrics):,} categories in Metric")
        d = d[d["Metric"].isin(mets)]
        if "Units" in d.columns and d["Units"].nunique() == 1:
            c2.caption(f"Units: {d['Units'].iloc[0]}")
    y = c3.selectbox("Variable to show", stats)
    c3.caption(f"{len(stats)} numeric variables available")

    # ---- 2) how to plot it ------------------------------------------------ #
    d = d.dropna(subset=[y])
    if d.empty:
        st.warning("No data for this selection.")
        return

    c1, c2, c3, c4 = st.columns(4)
    kind = c1.selectbox("Plot type", PLOT_TYPES, key="plot_kind")
    c1.caption(f"{len(PLOT_TYPES)} plot types")
    x = None
    ycat, agg_fn, obs, method = None, None, [], None
    if kind in ("Box plot", "Violin plot", "Bar (mean ± std)", "Heatmap", "Correlation"):
        opts = _categorical_options(d, MAX_PLOT_LEVELS)
        pref = {"Heatmap": "Metric" if "Metric" in opts else "Participant",
                "Correlation": "Metric" if "Metric" in opts else "Region"}.get(kind, "Hemisphere")
        xlabel = "Correlate levels of" if kind == "Correlation" else "X axis"
        x = c2.selectbox(xlabel, opts, index=opts.index(pref) if pref in opts else 0) if opts else None
        c2.caption(_count_note(d, x))
    elif kind == "Scatter":
        others = [c for c in stats if c != y]
        x = c2.selectbox("X axis (numeric)", others) if others else None
        c2.caption("numeric variable")
        if x is None:
            st.info("Scatter needs a second numeric variable.")
            return
    else:
        c2.selectbox("X axis", [y], disabled=True)

    # Variables to group by color or to separate the plots. The variables with a
    # single value in the selected rows cannot split them, so they are left out
    few = _categorical_options(d, MAX_PLOT_LEVELS)
    group = None
    if kind == "Correlation":
        defaults = [c for c in ("Participant", "Region") if c in d.columns and c != x]
        obs_opts = [c for c in _categorical_options(d, 10_000) + [c for c in d.columns if d[c].nunique() > 1 and c not in stats]
                    if c != x]
        obs_opts = list(dict.fromkeys(obs_opts))
        obs = c3.multiselect("Observations (rows)", obs_opts, default=[c for c in defaults if c in obs_opts],
                             help="Each unique combination of these variables is one observation; "
                                  "the levels of the variable on the left are correlated across them.")
        c3.caption(" · ".join(f"{o}: {d[o].nunique():,}" for o in obs) or "no variable selected")
        method = st.selectbox("Correlation method", CORR_METHODS)
        if not obs:
            st.info("Select at least one variable that defines the observations.")
            return
    elif kind == "Heatmap":
        yopts = [c for c in _categorical_options(d, MAX_PLOT_LEVELS) if c != x]
        if not yopts:
            st.info("A heatmap needs two categorical variables.")
            return
        ycat = c3.selectbox("Y axis", yopts, index=yopts.index("Region") if "Region" in yopts else 0)
        c3.caption(_count_note(d, ycat))
        agg_fn = st.selectbox("Aggregation of each cell", HEATMAP_AGGS,
                              help="Cells are colored by this summary of the chosen variable.")
    else:
        group = c3.selectbox("Group by (color)", [None] + [c for c in few if c != x],
                             help="Draws the groups with different colors inside each plot.")
        c3.caption(_count_note(d, group) or "no grouping")
    split = c4.selectbox("Separate plots by", [None] + [c for c in few if c not in (x, group, ycat, *obs)],
                         help="Creates one panel per level of this variable.")
    c4.caption(_count_note(d, split) or "single panel")
    ncols = 3
    if split:
        ncols = st.slider("Panels per row", 1, 6, 3)
    crowded = [f"{d[c].nunique()} {kind_} ({c})" for c, kind_ in ((group, "colors"), (split, "panels"))
               if c and d[c].nunique() > CROWDED_LEVELS]
    if crowded:
        st.caption("⚠️ " + " and ".join(crowded) + ": the plot can be slow and hard to read.")
    if kind in ("Box plot", "Violin plot", "Bar (mean ± std)", "Heatmap", "Correlation") and x is None:
        st.info("No categorical variable available for the X axis.")
        return

    st.caption(f"{len(d):,} rows plotted")

    # ---- 3) build the chart ---------------------------------------------- #
    cols = [c for c in dict.fromkeys([x, ycat, *obs, y, group, split]) if c]
    zscore = False
    if kind == "Heatmap" and "Metric" in d.columns and d["Metric"].nunique() > 1:
        zscore = st.checkbox("Standardize each metric (z-score)", value=True,
                             help="Metrics have different units; z-scoring each one makes their colors comparable.")
        if zscore:
            d = d.assign(**{y: d.groupby("Metric")[y].transform(lambda v: (v - v.mean()) / (v.std(ddof=0) or 1))})
    lower_only = False
    if kind == "Correlation":
        lower_only = st.checkbox("Show only the lower triangle", value=False)

    show_points = False
    if kind in ("Box plot", "Violin plot", "Bar (mean ± std)"):
        show_points = st.checkbox("Show individual points (jitter)", value=True,
                                  help=f"Up to {MAX_POINTS:,} points are drawn, spread evenly across the boxes. "
                                       "Statistics are always computed from all rows.")

    data = d[cols].copy()
    # categorical variables are always treated as labels (never as continuous hues)
    for c in (x, ycat, group, split, *obs):
        if c and c in data.columns and not (kind == "Scatter" and c == x):
            data[c] = data[c].map(names).fillna(data[c]) if c == "Metric" else data[c].astype(str)

    units = d["Units"].iloc[0] if "Units" in d.columns and d["Units"].nunique() == 1 else None
    ylabel = f"{y} ({units})" if units and not zscore else y
    if "Metric" in d.columns and len(mets) == 1:
        ylabel = f"{names.get(mets[0], mets[0])} – {ylabel}"

    tc = theme_colors()
    fg = tc["fg"]
    order = {}
    if x and kind != "Scatter":
        order[x] = sorted(data[x].unique())
    if group:
        order[group] = sorted(data[group].unique())
    if split:
        order[split] = sorted(data[split].unique())
    n_x = len(order.get(x, [1]))
    n_rows = -(-data[split].nunique() // ncols) if split else 1
    # The space between the rows of panels is a fraction of the figure, so it is
    # reduced with the number of rows (plotly fails if the rows do not fit)
    facet = {"facet_col": split, "facet_col_wrap": ncols,
             "facet_row_spacing": min(0.1, 0.2 / n_rows)} if split else {}
    height = 330 * n_rows if split else 520
    # box / violin / bar: without a group every X category gets its own color
    by_x = kind in ("Box plot", "Violin plot", "Bar (mean ± std)") and not group
    color_col = group or (x if by_x else None)
    common = dict(color=color_col, category_orders=order,
                  color_discrete_sequence=distinct_palette(data[color_col].nunique()) if color_col else [PALETTE[0]],
                  labels={y: ylabel}, **facet)

    if kind == "Box plot":
        fig = px.box(data, x=x, y=y, points=False, **common)
        # The outline has the color of the text, so the fill has to be set from the
        # color of each trace (plotly takes it from the outline otherwise)
        fig.for_each_trace(lambda t: t.update(fillcolor=with_alpha(t.marker.color, 0.55),
                                              line_color=fg, line_width=1.3))
    elif kind == "Violin plot":
        fig = px.violin(data, x=x, y=y, box=True, points=False, **common)
        fig.for_each_trace(lambda t: t.update(fillcolor=with_alpha(t.marker.color, 0.55)))
        fig.update_traces(line_color=fg, line_width=1, spanmode="hard", meanline_visible=False,
                          box_line_color=fg, box_fillcolor="rgba(0,0,0,0.25)")
    elif kind == "Bar (mean ± std)":
        keys = [k for k in dict.fromkeys([x, group, split]) if k]
        agg = data.groupby(keys, observed=True)[y].agg(["mean", "std"]).reset_index()
        fig = px.bar(agg, x=x, y="mean", error_y="std", barmode="group" if group else "overlay", **{**common, "labels": {"mean": f"mean {ylabel}"}})
        fig.update_traces(error_y={"color": fg, "thickness": 1.5, "width": 4})
        fig.update_yaxes(title_text=f"mean ± std of {ylabel}")
    elif kind == "Histogram":
        fig = px.histogram(data, x=y, nbins=40, barmode="overlay", opacity=0.65, **common)
        fig.update_traces(marker_line_color=fg, marker_line_width=0.6)
        fig.update_xaxes(title_text=ylabel)
        fig.update_yaxes(title_text="count")
    elif kind == "Density":
        kd = kde_frame(data, y, group, split)
        if kd.empty:
            st.warning("Not enough distinct values to estimate a density.")
            return
        fig = px.line(kd, x=y, y="density", **common)
        fig.update_traces(fill="tozeroy", line_width=1.6, opacity=0.8)
        fig.update_xaxes(title_text=ylabel)
    elif kind == "Scatter":
        sample = data.sample(min(len(data), 5000), random_state=0)
        fig = px.scatter(sample, x=x, y=y, opacity=0.55, **common)
        fig.update_traces(marker={"size": 6, "line": {"width": 0}})
    elif kind == "Heatmap":
        keys = [k for k in dict.fromkeys([ycat, x, split]) if k]
        agg = data.groupby(keys, observed=True)[y].agg(agg_fn).reset_index(name="cell")
        panels = {lvl: sub.pivot(index=ycat, columns=x, values="cell")
                  for lvl, sub in (agg.groupby(split) if split else [(None, agg)])}
        label = f"{agg_fn} of {y}" + (" (z)" if zscore and agg_fn != "count" else "")
        diverging = zscore and agg_fn != "count"
        lim = float(np.nanmax(np.abs(agg["cell"]))) if diverging else None
        fig = draw_matrices(panels, x, ycat, "RdBu_r" if diverging else "Viridis",
                            -lim if diverging else float(agg["cell"].min()),
                            lim if diverging else float(agg["cell"].max()), label, ncols, ".3g", tc)
    else:  # Correlation
        panels = {}
        for lvl, sub in (data.groupby(split) if split else [(None, data)]):
            wide = sub.pivot_table(index=obs, columns=x, values=y, aggfunc="mean", observed=True)
            wide = wide.loc[:, wide.notna().sum() >= 3]  # need a few observations per level
            if wide.shape[1] >= 2:
                panels[lvl] = wide.corr(method=method)
        if not panels:
            st.warning("Not enough overlapping observations to correlate. Try other observation variables.")
            return
        fig = draw_matrices(panels, x, x, "RdBu_r", -1.0, 1.0, f"{method} r", ncols, ".2f", tc,
                            lower_only=lower_only)

    if kind not in ("Heatmap", "Correlation"):
        if kind in ("Box plot", "Violin plot", "Bar (mean ± std)"):
            fig.update_xaxes(type="category", tickangle=-90 if n_x > 12 else 0,
                             tickfont={"size": 11 if n_x <= 30 else 8})
        if show_points:
            # stratified sample: at most `cap` points per (x, group, split) cell
            cell = [k for k in dict.fromkeys([x, group, split]) if k]
            cap = max(20, MAX_POINTS // max(data.groupby(cell).ngroups, 1))
            rnd = pd.Series(np.random.default_rng(0).random(len(data)), index=data.index)
            samp = data.loc[rnd.sort_values().index].groupby(cell, sort=False).head(cap)
            strip = px.strip(samp, x=x, y=y, **common)
            strip.update_traces(jitter=0.6, pointpos=0, showlegend=False,
                                marker={"size": 6, "opacity": 0.75,
                                        "line": {"width": 0.5, "color": with_alpha("#000000", 0.45)}})
            fig.add_traces(strip.data)
        style_figure(fig, tc, height, legend_title=group, facets=bool(split))
        # side by side only when groups share an X slot; otherwise one box per slot
        mode = "group" if group else "overlay"
        fig.update_layout(boxmode=mode, violinmode=mode)
        if by_x:
            fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch", theme=None,
                    config={"displaylogo": False, "scrollZoom": True,
                            "toImageButtonOptions": {"format": "png", "scale": 2}})


def main(default_path: str = "") -> None:
    st.set_page_config(page_title="Morphometry explorer", page_icon=":bar_chart:",
                       layout="wide", initial_sidebar_state="expanded")
    st.title("Morphometry explorer")

    with st.sidebar:
        path = sidebar_loader(default_path)
    if path is None:
        st.info("Upload a table or enter a valid path in the left panel.", icon="ℹ️")
        st.stop()

    df = load_table(path)
    with st.sidebar:
        sidebar_filters(df)

    dff = apply_filters(df, st.session_state.filters)

    overview(df, dff)
    tab_table, tab_stats, tab_plot = st.tabs(["Table", "Summary", "Plot"])
    with tab_table:
        st.dataframe(dff, width="stretch")
        st.download_button("Download filtered table", dff.to_csv(index=False).encode(),
                           file_name="filtered_table.csv", mime="text/csv")
    with tab_stats:
        st.dataframe(dff.describe(include="all").T, width="stretch")
    with tab_plot:
        plot_section(dff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default="", help="Table to open at start-up")
    args, _ = parser.parse_known_args(sys.argv[1:])
    main(args.table)
