"""Helpers of the freesurfertools module.

Functions used by ``FreeSurferSubject.generate_qc_webpage`` to select and draw the
slices of the volumes, cut the surfaces with them, render the maps over the surfaces,
read the global measures and build the web page of the quality control report.
"""

import contextlib
import io
import json
import os
import re
import warnings
from datetime import datetime
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def qc_parse_n_slices(n_slices) -> tuple:
    """Number of slices along X, Y and Z from an integer or a sequence of three."""
    if isinstance(n_slices, (int, np.integer)) and not isinstance(n_slices, bool):
        values = (int(n_slices),) * 3
    elif isinstance(n_slices, (list, tuple, np.ndarray)) and len(n_slices) == 3:
        values = tuple(n_slices)
    else:
        raise ValueError(
            "n_slices must be an integer or a sequence of three integers (X, Y, Z)"
        )

    if any(
        isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0
        for v in values
    ):
        raise ValueError("The numbers of slices must be non-negative integers")

    if sum(values) == 0:
        raise ValueError("At least one orientation must have slices")

    return tuple(int(v) for v in values)


def qc_find_lut(lut_file: str | Path = None) -> str:
    """Color table to use: the given one or the FreeSurferColorLUT.txt of FreeSurfer."""
    if lut_file is None:
        fs_home = os.environ.get("FREESURFER_HOME")
        if not fs_home:
            raise ValueError(
                "The color table must be passed as lut_file or FREESURFER_HOME must "
                "be set to use its FreeSurferColorLUT.txt"
            )
        lut_file = os.path.join(fs_home, "FreeSurferColorLUT.txt")

    if not os.path.isfile(lut_file):
        raise FileNotFoundError(f"The color table does not exist: {lut_file}")

    return str(lut_file)


def qc_read_text(path: str, last_line: bool = False) -> str | None:
    """Content of a small text file (or its last non-empty line), None if it is missing."""
    if not os.path.isfile(path):
        return None

    with open(path, "rb") as f:
        if last_line:
            f.seek(0, os.SEEK_END)
            f.seek(max(f.tell() - 4096, 0))
        lines = f.read().decode("utf-8", errors="replace").strip().splitlines()

    if not lines:
        return None

    return lines[-1].strip() if last_line else " ".join(line.strip() for line in lines)


def qc_read_stats_measures(stats_file: str) -> dict:
    """
    Global measures (``# Measure`` lines) of a FreeSurfer stats file.

    Returns a dictionary with the value of each measure by its short name
    (``eTIV``, ``CortexVol``, ``WhiteSurfArea``...), empty if the file is missing.
    """
    measures = {}
    if not stats_file or not os.path.isfile(stats_file):
        return measures

    with open(stats_file, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("# Measure"):
                continue
            fields = [field.strip() for field in line[len("# Measure") :].split(",")]
            if len(fields) < 5:
                continue
            try:
                measures[fields[1]] = float(fields[-2])
            except ValueError:
                continue

    return measures


def qc_read_topology(log_file: str, surf_dir: str = None) -> dict:
    """
    Topology of the surfaces of each hemisphere before the topology correction.

    Reads from ``recon-all.log`` the Euler number and the number of defects found
    by ``mris_fix_topology`` (the last ones if the subject was processed several
    times). Without them in the log, the Euler number is computed from the
    ``orig.nofix`` surface of ``surf_dir``. The number of holes is
    ``1 - euler / 2``. The values that are not available are None.
    """
    topology = {
        hemi: {"euler": None, "holes": None, "defects": None} for hemi in ("lh", "rh")
    }

    if log_file and os.path.isfile(log_file):
        hemi = None
        pending = {"euler": [], "defects": []}  # values found outside a known section
        with open(log_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                section = re.match(r"#@# Fix Topology\s+(lh|rh)\b", line)
                if section:
                    hemi = section.group(1)
                    continue
                if line.startswith("#@#"):
                    hemi = None
                    continue
                for key, pattern in (
                    ("euler", r"before topology correction, eno=(-?\d+)"),
                    ("defects", r"^\s*(\d+) defects found"),
                ):
                    found = re.search(pattern, line)
                    if found and hemi:
                        topology[hemi][key] = int(found.group(1))
                    elif found:
                        pending[key].append(int(found.group(1)))

        # Logs without the section titles: the left hemisphere is processed first
        for key, values in pending.items():
            if len(values) >= 2 and all(topology[h][key] is None for h in ("lh", "rh")):
                topology["lh"][key], topology["rh"][key] = values[-2], values[-1]

    for hemi in ("lh", "rh"):
        nofix = os.path.join(surf_dir, f"{hemi}.orig.nofix") if surf_dir else ""
        if topology[hemi]["euler"] is None and os.path.isfile(nofix):
            vertices, faces = nib.freesurfer.read_geometry(nofix)
            edges = np.sort(
                np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
                axis=1,
            )
            n_edges = np.unique(edges, axis=0).shape[0]
            topology[hemi]["euler"] = int(vertices.shape[0] - n_edges + faces.shape[0])
        if topology[hemi]["euler"] is not None:
            topology[hemi]["holes"] = (2 - topology[hemi]["euler"]) // 2

    return topology


def qc_metric_label(metric: str) -> tuple:
    """
    Full name and units of a metric, as defined in ``config/config.json``.

    For example, ("Cortical thickness", "mm") for "thickness". See
    :func:`clabtoolkit.morphometrytools.get_fullnames` and
    :func:`clabtoolkit.morphometrytools.get_units`.
    """
    # Imported here because morphometrytools imports freesurfertools
    from . import morphometrytools as morpho

    return morpho.get_fullnames(metric)[0], morpho.get_units(metric)[0]


def qc_mean_thickness(thickness_file: str, cortex_label: str = None) -> float | None:
    """
    Mean cortical thickness of a hemisphere, from its vertex-wise thickness map.

    The mean is computed over the vertices of the cortex label
    (``label/?h.cortex.label``) when it exists, as FreeSurfer does in its
    stats files, and otherwise over the vertices with a thickness greater
    than zero, which leaves out the medial wall. Returns None if the map is
    missing, cannot be read or has no cortical vertices.
    """
    if not thickness_file or not os.path.isfile(thickness_file):
        return None

    try:
        values = nib.freesurfer.read_morph_data(thickness_file)
    except (OSError, ValueError):
        return None
    cortex = None
    if cortex_label and os.path.isfile(cortex_label):
        try:
            cortex = nib.freesurfer.read_label(cortex_label)
        except (OSError, ValueError, IndexError):
            cortex = None
    if cortex is not None and cortex.size and cortex.max() < values.size:
        values = values[cortex]
    else:
        values = values[values > 0]

    return float(values.mean()) if values.size else None


def qc_subject_mean_thickness(subject) -> dict:
    """Mean cortical thickness of both hemispheres of a FreeSurferSubject."""
    label_dir = os.path.join(subject.subjs_dir, subject.subj_id, "label")
    return {
        hemi: qc_mean_thickness(
            subject.fs_files["surf"][hemi]["map"]["thickness"],
            os.path.join(label_dir, f"{hemi}.cortex.label"),
        )
        for hemi in ("lh", "rh")
    }


def qc_stats_table(subject, table_file: str) -> pd.DataFrame:
    """
    Morphometry table of a FreeSurferSubject, saved as ``table_file``.

    The table is the one of ``freesurfertools.create_subject_stats_table``. An
    existing file is read instead of computing the table again, unless the
    ``aseg.stats`` of the subject is newer than it (the subject was processed
    again). The columns Atlas and Hemisphere never have missing values.
    """
    # Imported here because freesurfertools imports this module
    from . import freesurfertools as cltfree

    aseg = subject.fs_files["stats"]["global"]["aseg"]
    reuse = os.path.isfile(table_file) and (
        not aseg or os.path.getmtime(table_file) >= os.path.getmtime(aseg)
    )
    # The morphometry functions print their progress and warn about missing measures
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = cltfree.create_subject_stats_table(
            subject.subj_id,
            subject.subjs_dir,
            out_tab_file=table_file,
            overwrite=not reuse,
        )

    return qc_clean_stats_table(table)


def qc_clean_stats_table(table: pd.DataFrame) -> pd.DataFrame:
    """Stats table with empty strings instead of missing atlas and hemisphere names."""
    table = table.copy()
    for column in ("Atlas", "Hemisphere", "Source", "Metric", "Units", "Region"):
        if column in table.columns:
            table[column] = table[column].fillna("").astype(str)
    return table


def qc_table_value(
    table: pd.DataFrame, source: str, metric: str, atlas: str, region: str
) -> float | None:
    """Value of a region in a stats table, None if it is not there."""
    rows = table[
        (table["Source"] == source)
        & (table["Metric"] == metric)
        & (table["Atlas"] == atlas)
        & (table["Region"] == region)
    ]
    if rows.empty or pd.isna(rows["Value"].iloc[0]):
        return None
    return float(rows["Value"].iloc[0])


def qc_measures_from_table(table: pd.DataFrame) -> tuple:
    """
    Global measures that a stats table has, in the layout of ``qc_global_measures``.

    Returns the measures of the aseg (by their names in ``aseg.stats``), the
    measures of each hemisphere and the mean thickness of each hemisphere.
    Only the measures found in the table are returned.
    """
    aseg, aparc, thickness = {}, {"lh": {}, "rh": {}}, {}

    for key, region in (
        ("eTIV", "brain-brain-intracraneal"),
        ("BrainSegVol", "brain-brain-wholebrain"),
        ("TotalGrayVol", "gm-brain-graymatter"),
        ("CerebralWhiteMatterVol", "wm-brain-whitematter"),
    ):
        value = qc_table_value(table, "statsfile", "volume", "", region)
        if value is not None:
            aseg[key] = value

    for hemi in ("lh", "rh"):
        region = f"ctx-{hemi}-hemisphere"
        for key, source in (("WhiteSurfArea", "white"), ("PialSurfArea", "pial")):
            value = qc_table_value(table, source, "area", "lobes_desikan", region)
            if value is not None:
                aparc[hemi][key] = value
        value = qc_table_value(table, "vertices", "thickness", "lobes_desikan", region)
        if value is not None:
            thickness[hemi] = value

    return aseg, aparc, thickness


def qc_lobes_table(table: pd.DataFrame) -> list:
    """
    Thickness and surface areas of the hemispheres and their lobes, from a stats table.

    Each row is a dictionary with the name of the region and, for each
    hemisphere, the mean thickness (mm) and the white and pial surface areas
    (cm2). Empty if the table has no lobar measures.
    """
    lobes = table[table["Atlas"] == "lobes_desikan"]
    names = []
    for region in lobes["Region"]:
        name = region.split("-", 2)[-1]
        if name not in names:
            names.append(name)

    rows = []
    for name in names:
        row = {"region": name.capitalize()}
        for hemi in ("lh", "rh"):
            region = f"ctx-{hemi}-{name}"
            for key, source, metric, factor in (
                ("thickness", "vertices", "thickness", 1),
                ("white_area", "white", "area", 100),
                ("pial_area", "pial", "area", 100),
            ):
                value = qc_table_value(lobes, source, metric, "lobes_desikan", region)
                row[f"{key}_{hemi}"] = (
                    None if value is None else round(value / factor, 2)
                )
        rows.append(row)

    return rows


def qc_regions_data(table: pd.DataFrame) -> list:
    """
    Stats table arranged for the table of regions of the web page.

    Returns one group per source and metric (with its full name and units),
    each with the rows [region, hemisphere, value, median, std, min, max] of
    every atlas. The numbers keep five significant digits.
    """

    def number(value):
        return None if pd.isna(value) else float(f"{float(value):.5g}")

    stats = [c for c in ("Value", "Median", "Std", "Min", "Max") if c in table.columns]
    groups = []
    for (source, metric), group in table.groupby(["Source", "Metric"], sort=False):
        atlases = {}
        for atlas, rows in group.groupby("Atlas", sort=False):
            atlases[atlas or "global"] = [
                [row.Region, row.Hemisphere] + [number(getattr(row, c)) for c in stats]
                for row in rows.itertuples()
            ]
        groups.append(
            {
                "label": f"{qc_metric_label(metric)[0]} ({source})",
                "units": group["Units"].iloc[0],
                "atlases": atlases,
            }
        )

    return groups


def qc_cohort_outliers(
    tables: dict, threshold: float = 4.0, min_subjects: int = 20
) -> pd.DataFrame | None:
    """
    Regional measures of each subject that are far from the ones of the cohort.

    ``tables`` has the stats table of each subject. For each measure, the
    distance of a subject to the median of the cohort is expressed in robust
    standard deviations (1.4826 times the median absolute deviation); the
    values farther than ``threshold`` are returned, the farthest first. The
    measures compared are the mean thickness and the white surface area of
    the lobes and of the regions of the Desikan-Killiany atlas, and the
    volumes of the aseg.

    Returns None when fewer than ``min_subjects`` subjects have a table,
    because the statistics of the cohort would not be reliable.
    """
    if len(tables) < max(int(min_subjects), 3):
        return None

    selected = []
    for subj_id, table in tables.items():
        keep = (
            (table["Source"] == "vertices")
            & (table["Metric"] == "thickness")
            & table["Atlas"].isin(["lobes_desikan", "desikan"])
        )
        keep |= (
            (table["Source"] == "white")
            & (table["Metric"] == "area")
            & table["Atlas"].isin(["lobes_desikan", "desikan"])
        )
        keep |= (
            (table["Source"] == "statsfile")
            & (table["Metric"] == "volume")
            & (table["Atlas"] == "")
        )
        rows = table.loc[
            keep, ["Source", "Metric", "Units", "Atlas", "Region", "Value"]
        ]
        selected.append(rows.assign(subject_id=subj_id))

    values = pd.concat(selected, ignore_index=True).dropna(subset=["Value"])
    keys = ["Source", "Metric", "Atlas", "Region"]
    grouped = values.groupby(keys)["Value"]
    values["cohort_median"] = grouped.transform("median")
    values["robust_sd"] = 1.4826 * (
        values["Value"] - values["cohort_median"]
    ).abs().groupby([values[k] for k in keys]).transform("median")
    values["n_subjects"] = grouped.transform("count")
    values = values[(values["robust_sd"] > 0) & (values["n_subjects"] >= min_subjects)]
    values = values.assign(
        z=(values["Value"] - values["cohort_median"]) / values["robust_sd"]
    )

    outliers = values[values["z"].abs() > threshold]
    outliers = outliers.reindex(outliers["z"].abs().sort_values(ascending=False).index)
    outliers.columns = [c.lower() for c in outliers.columns]
    return outliers[
        ["subject_id", "source", "metric", "units", "atlas", "region", "value"]
        + ["cohort_median", "robust_sd", "z", "n_subjects"]
    ].reset_index(drop=True)


def qc_global_measures(
    aseg: dict, aparc: dict, topology: dict, mean_thickness: dict = None
) -> list:
    """
    Rows of the table of global measures of the quality control report.

    Each row is a dictionary with the name of the measure, its units and its
    values for the left hemisphere, the right hemisphere and the whole brain
    (None when it does not apply or is not available). The measures without
    any value are left out. ``mean_thickness`` gives the mean thickness of
    each hemisphere, computed from the thickness maps (see
    :func:`qc_mean_thickness`). The stats files of old FreeSurfer versions
    name some measures differently (``ICV``, ``CorticalWhiteMatterVol``,
    ``SurfArea``): those names are used when the current ones are missing.
    """

    def scaled(value, factor):
        return None if value is None else round(value / factor, 2)

    def total(lh, rh):
        return None if lh is None or rh is None else round(lh + rh, 2)

    def first(measures, keys):
        return next((measures[k] for k in keys if k in measures), None)

    rows = []

    def add(name, units, lh=None, rh=None, both=None):
        if any(v is not None for v in (lh, rh, both)):
            rows.append(
                {"name": name, "units": units, "lh": lh, "rh": rh, "total": both}
            )

    for name, keys in (
        ("Estimated total intracranial volume", ("eTIV", "ICV")),
        ("Brain segmentation volume", ("BrainSegVol",)),
        ("Cortical gray matter volume", ("CortexVol",)),
        (
            "Cerebral white matter volume",
            ("CerebralWhiteMatterVol", "CorticalWhiteMatterVol"),
        ),
        ("Subcortical gray matter volume", ("SubCortGrayVol",)),
        ("Total gray matter volume", ("TotalGrayVol",)),
    ):
        add(
            name,
            "cm³",
            scaled(first(aseg, ["lh" + k for k in keys]), 1000),
            scaled(first(aseg, ["rh" + k for k in keys]), 1000),
            scaled(first(aseg, keys), 1000),
        )

    area = {
        h: scaled(first(aparc.get(h, {}), ("WhiteSurfArea", "SurfArea")), 100)
        for h in ("lh", "rh")
    }
    add(
        "White surface area",
        "cm²",
        area["lh"],
        area["rh"],
        total(area["lh"], area["rh"]),
    )
    pial = {h: scaled(aparc.get(h, {}).get("PialSurfArea"), 100) for h in ("lh", "rh")}
    add(
        "Pial surface area",
        "cm²",
        pial["lh"],
        pial["rh"],
        total(pial["lh"], pial["rh"]),
    )

    thickness = mean_thickness or {}
    thickness_name, thickness_units = qc_metric_label("thickness")
    add(
        f"Mean {thickness_name[0].lower()}{thickness_name[1:]}",
        thickness_units,
        scaled(thickness.get("lh"), 1),
        scaled(thickness.get("rh"), 1),
    )

    lh, rh = topology.get("lh", {}), topology.get("rh", {})
    for name, key in (
        ("Surface holes before topology correction", "holes"),
        ("Surface defects before topology correction", "defects"),
    ):
        add(name, "", lh.get(key), rh.get(key), total(lh.get(key), rh.get(key)))

    return rows


def qc_read_rating(subj_dir: str) -> dict | None:
    """
    Rating of a subject, saved by the rating widget of its report.

    Reads the last ``*qc_rating*.json`` file of the ``qcqa/metadata`` folder of
    the subject. Returns None if there is none or it cannot be read.
    """
    rating_files = sorted(
        glob(os.path.join(subj_dir, "qcqa", "metadata", "*qc_rating*.json"))
    )
    if not rating_files:
        return None
    try:
        with open(rating_files[-1], encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return None
    return saved if isinstance(saved, dict) else None


def qc_rating_rows(subj_id: str, rating: dict | None) -> list:
    """
    Rows of the rating of a subject for the table of ratings of a cohort.

    The first row is the rating of the subject (item "subject") and the others
    the figures that were rated or commented.
    """
    if not rating:
        return []

    common = {
        "subject_id": subj_id,
        "rater": rating.get("rater") or "",
        "date": rating.get("date") or "",
    }
    rows = [
        {
            **common,
            "item": "subject",
            "rating": rating.get("rating") or "",
            "comment": rating.get("comment") or "",
        }
    ]
    reportlets = rating.get("reportlets")
    for item, value in (reportlets if isinstance(reportlets, dict) else {}).items():
        if isinstance(value, dict) and (value.get("rating") or value.get("comment")):
            rows.append(
                {
                    **common,
                    "item": item,
                    "rating": value.get("rating") or "",
                    "comment": value.get("comment") or "",
                }
            )
    return rows


def qc_cohort_other_row(name: str, note: str) -> dict:
    """
    Row of the index of a cohort for a folder that is not a FreeSurfer output folder.

    It has the same fields as the row of a subject (see ``qc_cohort_row``),
    with the status "not-freesurfer" and no measures, rating or report.
    """
    return {
        "subject_id": name,
        "status": "not-freesurfer",
        "holes": None,
        "defects": None,
        "outliers": None,
        "etiv_cm3": None,
        "thickness_lh_mm": None,
        "thickness_rh_mm": None,
        "rating": None,
        "rater": None,
        "report": None,
        "report_date": None,
        "note": note,
    }


def qc_cohort_row(subject, report: str = None) -> dict:
    """
    Row of a subject in the index of the quality control reports of a cohort.

    ``subject`` is a FreeSurferSubject whose processing status was checked and
    ``report`` the path to its individual report, if it exists. The measures
    are read from the recon-all.log, the aseg.stats and the thickness maps of
    the subject, and the rating from the file saved by the rating widget of the report in the
    ``qcqa/metadata`` folder of the subject.
    """
    subj_dir = os.path.join(subject.subjs_dir, subject.subj_id)
    stats = subject.fs_files["stats"]
    topology = qc_read_topology(
        os.path.join(subj_dir, "scripts", "recon-all.log"),
        os.path.join(subj_dir, "surf"),
    )
    aseg = qc_read_stats_measures(stats["global"]["aseg"])
    thickness = qc_subject_mean_thickness(subject)

    def both(key):
        values = [topology[hemi][key] for hemi in ("lh", "rh")]
        return None if None in values else sum(values)

    def rounded(value, factor=1.0, digits=2):
        return None if value is None else round(value / factor, digits)

    saved = qc_read_rating(subj_dir) or {}
    rating, rater = saved.get("rating") or None, saved.get("rater") or None

    report_date = None
    if report:
        report_date = datetime.fromtimestamp(os.path.getmtime(report)).strftime(
            "%Y-%m-%d %H:%M"
        )

    return {
        "subject_id": subject.subj_id,
        "status": subject.pstatus,
        "holes": both("holes"),
        "defects": both("defects"),
        "outliers": None,
        "etiv_cm3": rounded(aseg.get("eTIV", aseg.get("ICV")), 1000, 1),
        "thickness_lh_mm": rounded(thickness["lh"]),
        "thickness_rh_mm": rounded(thickness["rh"]),
        "rating": rating,
        "rater": rater,
        "report": report,
        "report_date": report_date,
    }


def qc_to_ras(data: np.ndarray, affine: np.ndarray) -> tuple:
    """Reorient a volume to the closest RAS orientation. Returns the data and its affine."""
    ornt = nib.orientations.io_orientation(affine)
    ras_affine = affine @ nib.orientations.inv_ornt_aff(ornt, data.shape[:3])
    return nib.orientations.apply_orientation(data, ornt), ras_affine


def qc_bbox(mask: np.ndarray, pad: int = 0) -> list:
    """Bounding box (first and last index, both included) of a mask along each axis."""
    box = []
    for axis in range(3):
        others = tuple(a for a in range(3) if a != axis)
        index = np.where(mask.any(axis=others))[0]
        lo, hi = (index[0], index[-1]) if index.size else (0, mask.shape[axis] - 1)
        box.append((max(int(lo) - pad, 0), min(int(hi) + pad, mask.shape[axis] - 1)))
    return box


def qc_get_slice(vol: np.ndarray, axis: int, index: int, crop: list) -> np.ndarray:
    """Cropped slice of a RAS volume, ready to display (superior or anterior on top)."""
    (x0, x1), (y0, y1), (z0, z1) = crop
    if axis == 0:
        plane = vol[index, y0 : y1 + 1, z0 : z1 + 1]
    elif axis == 1:
        plane = vol[x0 : x1 + 1, index, z0 : z1 + 1]
    else:
        plane = vol[x0 : x1 + 1, y0 : y1 + 1, index]
    return np.rot90(plane)


def qc_blend(
    gray: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    rgb_lut: np.ndarray = None,
    color: tuple = None,
) -> np.ndarray:
    """
    Overlay a slice of labels on a gray slice.

    With ``rgb_lut`` each label gets its color with opacity ``alpha``. Without
    it the labels are a mask, drawn in ``color`` with half the opacity and a
    solid outline.
    """
    rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    fg = labels > 0

    if rgb_lut is not None:
        rgb[fg] = (1 - alpha) * rgb[fg] + alpha * rgb_lut[labels[fg]]
    else:
        col = np.asarray(color, dtype=np.float32)
        rgb[fg] = (1 - alpha / 2) * rgb[fg] + alpha / 2 * col

        # Voxels of the mask with a neighbor outside it
        inner = fg.copy()
        inner[1:, :] &= fg[:-1, :]
        inner[:-1, :] &= fg[1:, :]
        inner[:, 1:] &= fg[:, :-1]
        inner[:, :-1] &= fg[:, 1:]
        rgb[fg & ~inner] = col

    return rgb.astype(np.uint8)


def qc_slice_mesh(
    vertices: np.ndarray, faces: np.ndarray, axis: int, index: int, crop: list
) -> np.ndarray:
    """
    Intersection of a mesh with a slice of the RAS volume.

    The vertices are in voxels of the RAS volume. Returns the segments, an
    array of shape (n, 2, 2), in pixels of the slice given by ``qc_get_slice``.
    """
    # Distance of the vertices of each triangle to the plane (never exactly zero)
    dist = vertices[:, axis] - index
    dist = np.where(dist == 0, 1e-6, dist)[faces]
    cut = (dist.min(axis=1) < 0) & (dist.max(axis=1) > 0)
    if not cut.any():
        return np.zeros((0, 2, 2))
    faces, d_start = faces[cut], dist[cut]

    # Point where each edge of the triangles crosses the plane (two edges per triangle)
    start, end = vertices[faces], vertices[np.roll(faces, -1, axis=1)]
    d_end = np.roll(d_start, -1, axis=1)
    crossing = d_start * d_end < 0
    t = d_start / np.where(crossing, d_start - d_end, 1.0)
    points = start + t[..., None] * (end - start)
    order = np.argsort(~crossing, axis=1, kind="stable")[:, :2]
    segments = points[np.arange(faces.shape[0])[:, None], order]

    # Coordinates in the cropped and rotated slice
    horizontal, vertical = [a for a in range(3) if a != axis]
    return np.stack(
        [
            segments[..., horizontal] - crop[horizontal][0],
            crop[vertical][1] - segments[..., vertical],
        ],
        axis=-1,
    )


def qc_plot_surface_map(layer: dict, out_file: str, views, window_size: tuple):
    """Figure of a vertex-wise map or an annotation over the surfaces of both hemispheres."""
    # Imported here because these modules import this one
    from . import surfacetools as cltsurf
    from . import visualizationtools as cltvis

    surfaces = {}
    for hemi in ("lh", "rh"):
        surf = cltsurf.Surface(layer["files"][f"{hemi}.{layer['surface']}"], hemi=hemi)
        map_file = layer["files"][layer["map_file"].format(hemi=hemi)]
        if layer["kind"] == "surface_annot":
            surf.load_annotation(map_file, layer["map"])
        else:
            surf.load_scalar_maps(map_file, maps_names=layer["map"])
        surfaces[hemi] = surf

    # The plotter prints the changes of the theme and the path of the figure
    with contextlib.redirect_stdout(io.StringIO()):
        plotter = cltvis.BrainPlotter()
        plotter.apply_theme("dark")
        # Room for the full name of the map above the colorbar
        plotter.figure_conf["colorbar_size"] = 0.35
        # Texts sized for the width of the figure (the defaults are for a screen)
        scale = window_size[0] / 1000
        for key, factor in (
            ("colorbar_font_size", 1.0),
            ("colorbar_title_font_size", 0.6),
        ):
            plotter.figure_conf[key] = int(
                round(plotter.figure_conf[key] * scale * factor)
            )
        plotter.plot_hemispheres(
            surfaces["rh"],
            surfaces["lh"],
            map_name=layer["map"],
            views=views,
            save_path=out_file,
            window_size=window_size,
            **layer["plot"],
        )
    if not os.path.isfile(out_file):
        raise RuntimeError("plot_hemispheres did not create the figure")


def qc_save_mosaic(
    panels: list,
    zooms: np.ndarray,
    out_file: str,
    dpi: int = 150,
    fig_width: float = 14.0,
    max_cols: int = 8,
):
    """
    Save a mosaic with one block of slices per orientation on a black background.

    ``panels`` has three lists (X, Y and Z) of (RGB image, label, lines) tuples,
    where lines is a list of (segments, color) in pixels of the image. An
    orientation with more than ``max_cols`` slices is split in several rows.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure

    markers = [("P", "A"), ("L", "R"), ("L", "R")]
    sizes = [(zooms[1], zooms[2]), (zooms[0], zooms[2]), (zooms[0], zooms[1])]
    gap = 0.06  # inches between rows

    # Position (in inches, from the top) of each panel
    boxes = []
    top = 0.0
    for axis, row in enumerate(panels):
        if not row:
            continue
        n_rows = int(np.ceil(len(row) / max_cols))
        n_cols = int(np.ceil(len(row) / n_rows))
        height, width = row[0][0].shape[:2]
        panel_w = fig_width / n_cols
        panel_h = panel_w * (height * sizes[axis][1]) / (width * sizes[axis][0])
        for i, (rgb, label, lines) in enumerate(row):
            r, c = divmod(i, n_cols)
            boxes.append(
                (
                    c * panel_w,
                    top + r * (panel_h + gap),
                    panel_w,
                    panel_h,
                    rgb,
                    label,
                    lines,
                    axis,
                )
            )
        top += n_rows * (panel_h + gap)

    fig_height = top - gap
    fig = Figure(figsize=(fig_width, fig_height), facecolor="black")
    for left, y, w, h, rgb, label, lines, axis in boxes:
        ax = fig.add_axes(
            [left / fig_width, 1 - (y + h) / fig_height, w / fig_width, h / fig_height]
        )
        ax.imshow(rgb, aspect="auto")
        for segments, line_color in lines:
            ax.add_collection(
                LineCollection(segments, colors=line_color, linewidths=0.7)
            )
        ax.set_xlim(-0.5, rgb.shape[1] - 0.5)
        ax.set_ylim(rgb.shape[0] - 0.5, -0.5)
        ax.set_axis_off()
        style = {"color": "white", "fontsize": 8, "transform": ax.transAxes}
        ax.text(0.03, 0.97, label, va="top", ha="left", **style)
        ax.text(0.03, 0.03, markers[axis][0], va="bottom", ha="left", **style)
        ax.text(0.97, 0.03, markers[axis][1], va="bottom", ha="right", **style)

    fig.savefig(out_file, dpi=dpi, facecolor="black")


QC_CSS = """
:root { --ink: #212529; --muted: #6c757d; --line: #dee2e6; --soft: #f8f9fa; --link: #0d6efd; }
* { box-sizing: border-box; }
html { scroll-padding-top: 72px; scroll-behavior: smooth; }
body { margin: 0; padding-top: 56px; color: var(--ink); background: #fff; line-height: 1.5;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }
nav { position: fixed; top: 0; left: 0; right: 0; z-index: 10; min-height: 56px; background: #212529;
  display: flex; align-items: center; gap: 4px 20px; flex-wrap: wrap; padding: 8px 24px; }
nav .brand { color: #fff; font-size: 1.25rem; margin-right: 12px; white-space: nowrap; }
nav a { color: rgba(255, 255, 255, .65); font-size: .95rem; white-space: nowrap; }
nav a:hover { color: #fff; text-decoration: none; }
nav .nav-right { margin-left: auto; display: flex; align-items: center; gap: 4px 18px; flex-wrap: wrap; }
nav button { font: inherit; font-size: .875rem; padding: 4px 12px; cursor: pointer; color: #fff;
  background: transparent; border: 1px solid rgba(255, 255, 255, .45); border-radius: 4px; }
nav button:hover { background: rgba(255, 255, 255, .12); }
[hidden] { display: none !important; }
kbd { font-family: SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size: .8em;
  padding: 1px 6px; border: 1px solid #ced4da; border-bottom-width: 2px; border-radius: 4px; background: var(--soft); }
.figure img { cursor: zoom-in; }
@media print {
  @page { margin: 12mm; }
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  html { scroll-padding-top: 0; }
  body { padding-top: 0; font-size: 10pt; }
  nav, .toolbar, .actions, .shortcuts, .reportlet .head button, details, .explorer { display: none !important; }
  main { max-width: none; padding: 0; }
  h1 { margin-top: 0; }
  h2 { break-after: avoid; margin-top: 24px; }
  .reportlet { break-inside: avoid; margin-top: 18px; }
  table, .summary li { break-inside: avoid; }
  .rating button:not([aria-pressed="true"]) { display: none; }
  .rating:not(:has(button[aria-pressed="true"])) .label::after { content: " not rated"; }
  .rating input, .overall textarea { border: 0; padding-left: 0; }
  .rating input:placeholder-shown, .overall textarea:placeholder-shown { display: none; }
  a { color: inherit; }
}
main { max-width: 1320px; margin: 0 auto; padding: 8px 24px 48px; }
h1 { font-size: 2rem; font-weight: 500; margin: 28px 0 4px; }
h2 { font-size: 1.6rem; font-weight: 500; margin: 44px 0 12px; padding-bottom: 8px;
  border-bottom: 1px solid var(--line); }
h3 { font-size: 1.2rem; font-weight: 500; margin: 0 0 4px; }
.lead { color: var(--muted); margin: 0 0 8px; }
.summary { list-style: none; margin: 0; padding: 0; }
.summary li { padding: 6px 0; border-bottom: 1px solid var(--soft); display: flex; gap: 12px; flex-wrap: wrap; }
.summary .key { flex: 0 0 210px; color: var(--muted); }
.summary .value { flex: 1 1 300px; overflow-wrap: anywhere; }
code, .mono { font-family: SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; font-size: .875em; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 4px; color: #fff; font-size: .85rem;
  font-weight: 600; letter-spacing: .02em; }
.badge.processed, .badge.ok { background: #198754; }
.badge.autorecon2, .badge.autorecon1 { background: #b97a00; }
.badge.unprocessed, .badge.missing, .badge.error { background: #dc3545; }
.badge.not-freesurfer { background: #6c757d; }
.reportlet { margin: 32px 0 0; }
.reportlet .head { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
.reportlet p { margin: 0 0 10px; color: #495057; max-width: 900px; }
.reportlet button { font: inherit; font-size: .875rem; padding: 5px 12px; margin-bottom: 10px; cursor: pointer;
  color: var(--ink); background: #fff; border: 1px solid #adb5bd; border-radius: 4px; }
.reportlet button:hover { background: var(--soft); }
.reportlet button[aria-pressed="true"] { background: var(--ink); color: #fff; border-color: var(--ink); }
.figure { background: #000; border-radius: 4px; overflow: hidden; }
.figure img { display: block; width: 100%; height: auto; }
.caption { font-size: .85rem; color: var(--muted); margin-top: 6px; overflow-wrap: anywhere; }
.alert { padding: 12px 16px; border-radius: 4px; border: 1px solid #f1aeb5; background: #f8d7da; color: #58151c; }
details { margin-top: 10px; border: 1px solid var(--line); border-radius: 4px; }
summary { padding: 8px 14px; cursor: pointer; background: var(--soft); font-size: .95rem; }
.legend { list-style: none; margin: 0; padding: 12px 14px; display: grid; gap: 4px 20px;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); font-size: .82rem; }
.legend li { display: flex; align-items: center; gap: 8px; min-width: 0; }
.legend .swatch { flex: 0 0 14px; height: 14px; border-radius: 2px; border: 1px solid rgba(0, 0, 0, .25); }
.legend .name { flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.legend .vol { color: var(--muted); font-variant-numeric: tabular-nums; white-space: nowrap; }
table.measures { border-collapse: collapse; width: 100%; max-width: 900px; font-size: .95rem; }
table.measures th, table.measures td { padding: 6px 12px; border-bottom: 1px solid var(--line); text-align: right;
  font-variant-numeric: tabular-nums; }
table.measures thead th { color: var(--muted); font-weight: 500; border-bottom-width: 2px; }
table.measures th:first-child { text-align: left; font-weight: 400; }
table.measures td.empty, table.measures td.units { color: var(--muted); }
table.measures td.units { text-align: left; }
.rating { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.rating .label { color: var(--muted); font-size: .875rem; min-width: 52px; }
.rating button, .actions button { font: inherit; font-size: .875rem; padding: 5px 14px; margin: 0; cursor: pointer;
  color: var(--ink); background: #fff; border: 1px solid #adb5bd; border-radius: 4px; }
.rating button:hover, .actions button:hover { background: var(--soft); }
.rating button[aria-pressed="true"] { color: #fff; }
.rating button[aria-pressed="true"][data-value="pass"] { background: #198754; border-color: #198754; }
.rating button[aria-pressed="true"][data-value="doubtful"] { background: #b97a00; border-color: #b97a00; }
.rating button[aria-pressed="true"][data-value="fail"] { background: #dc3545; border-color: #dc3545; }
.rating input, .overall textarea { font: inherit; font-size: .9rem; padding: 5px 10px; color: var(--ink);
  border: 1px solid #ced4da; border-radius: 4px; }
.rating input { flex: 1 1 240px; max-width: 520px; }
.overall { max-width: 900px; }
.overall textarea { display: block; width: 100%; margin: 12px 0; resize: vertical; }
.actions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.actions .caption { margin: 0 0 0 8px; }
nav a[data-rating]::after { content: ""; display: inline-block; width: 8px; height: 8px; margin-left: 5px;
  border-radius: 50%; }
nav a[data-rating="pass"]::after { background: #2fbf71; }
nav a[data-rating="doubtful"]::after { background: #f0ad00; }
nav a[data-rating="fail"]::after { background: #ff5c6c; }
footer { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--line); color: var(--muted); font-size: .85rem; }
"""

QC_JS = """
document.querySelectorAll('button[data-target]').forEach(function (button) {
  var img = document.getElementById(button.dataset.target);
  button.addEventListener('click', function () {
    var hidden = button.getAttribute('aria-pressed') !== 'true';
    button.setAttribute('aria-pressed', hidden);
    button.textContent = hidden ? 'Show overlay' : 'Hide overlay';
    img.src = hidden ? img.dataset.base : img.dataset.overlay;
  });
});
"""


QC_RATING_JS = """
(function () {
  var key = 'clabtoolkit-qc:' + location.pathname;
  var empty = function () { return {rater: '', overall: '', comment: '', reportlets: {}}; };
  var state = empty();
  try { state = Object.assign(state, JSON.parse(localStorage.getItem(key) || '{}')); } catch (e) {}

  function item(id) {
    if (!state.reportlets[id]) { state.reportlets[id] = {rating: '', comment: ''}; }
    return state.reportlets[id];
  }
  function store() {
    try { localStorage.setItem(key, JSON.stringify(state)); } catch (e) {}
  }
  var boxes = document.querySelectorAll('.rating');
  var fields = document.querySelectorAll('[data-field]');

  function rate(id, value) {
    if (id) {
      item(id).rating = item(id).rating === value ? '' : value;
    } else {
      state.overall = state.overall === value ? '' : value;
    }
    store();
    refresh();
  }

  // Keyboard: j / k move between the figures, p / d / f rate the figure in view
  // and, with Shift, the subject
  var figures = Array.prototype.filter.call(
    document.querySelectorAll('section.reportlet'),
    function (section) { return section.querySelector('.rating'); }
  );
  function inView() {
    var line = window.innerHeight * 0.4, found = -1;
    figures.forEach(function (section, i) {
      var box = section.getBoundingClientRect();
      if (box.top <= line && box.bottom > line) { found = i; }
    });
    return found;
  }
  function move(step) {
    if (!figures.length) { return; }
    var i = inView(), target;
    if (i >= 0) {
      target = Math.max(0, Math.min(figures.length - 1, i + step));
    } else {
      target = figures[0].getBoundingClientRect().top > 0 ? 0 : figures.length - 1;
    }
    figures[target].scrollIntoView();
  }
  document.addEventListener('keydown', function (event) {
    var tag = event.target.tagName;
    if (event.ctrlKey || event.metaKey || event.altKey ||
        tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') { return; }
    var key = event.key.toLowerCase();
    var value = {p: 'pass', d: 'doubtful', f: 'fail'}[key];
    if (key === 'j') { move(1); }
    else if (key === 'k') { move(-1); }
    else if (value && event.shiftKey) { rate('', value); }
    else if (value && inView() >= 0) {
      rate(figures[inView()].querySelector('.rating').dataset.id, value);
    }
  });

  function owner(field) {
    var box = field.closest('.rating');
    return box && box.dataset.id ? item(box.dataset.id) : state;
  }
  function refresh() {
    var rated = 0, total = 0;
    boxes.forEach(function (box) {
      var id = box.dataset.id;
      var value = id ? item(id).rating : state.overall;
      box.querySelectorAll('button[data-value]').forEach(function (button) {
        button.setAttribute('aria-pressed', button.dataset.value === value);
      });
      if (id) {
        total += 1;
        rated += value ? 1 : 0;
        document.querySelectorAll('nav a[data-rated]').forEach(function (link) {
          if (link.dataset.rated !== id) { return; }
          if (value) { link.dataset.rating = value; } else { delete link.dataset.rating; }
        });
      }
    });
    fields.forEach(function (field) {
      var value = owner(field)[field.dataset.field] || '';
      if (field.value !== value) { field.value = value; }
    });
    document.getElementById('rating-count').textContent = rated + ' of ' + total + ' figures rated';
  }

  boxes.forEach(function (box) {
    var id = box.dataset.id;
    box.querySelectorAll('button[data-value]').forEach(function (button) {
      button.addEventListener('click', function () { rate(id, button.dataset.value); });
    });
  });
  fields.forEach(function (field) {
    field.addEventListener('input', function () {
      owner(field)[field.dataset.field] = field.value;
      store();
    });
  });

  document.getElementById('rating-download').addEventListener('click', function () {
    var body = document.body;
    var reportlets = {};
    boxes.forEach(function (box) {
      if (box.dataset.id) { reportlets[box.dataset.id] = item(box.dataset.id); }
    });
    var rating = {
      subject_id: body.dataset.subject,
      subjects_dir: body.dataset.subjectsDir,
      rater: state.rater,
      date: new Date().toISOString(),
      rating: state.overall,
      comment: state.comment,
      reportlets: reportlets
    };
    var link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([JSON.stringify(rating, null, 2)], {type: 'application/json'}));
    link.download = body.dataset.subject + '_qc_rating.json';
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(link.href);
  });
  document.getElementById('rating-clear').addEventListener('click', function () {
    if (!window.confirm('Clear the rating of this subject?')) { return; }
    state = empty();
    store();
    refresh();
  });

  refresh();
})();
"""


QC_STATS_JS = """
(function () {
  var node = document.getElementById('stats-data');
  if (!node) { return; }
  var groups = JSON.parse(node.textContent);
  var group = document.getElementById('stats-group');
  var atlas = document.getElementById('stats-atlas');
  var search = document.getElementById('stats-search');
  var shown = document.getElementById('stats-shown');
  var table = document.getElementById('stats-table');
  var body = table.tBodies[0];
  var sort = {column: -1, up: true};

  groups.forEach(function (g, i) { group.add(new Option(g.label, i)); });

  function fillAtlases() {
    atlas.innerHTML = '';
    Object.keys(groups[group.value].atlases).forEach(function (name) {
      atlas.add(new Option(name, name));
    });
  }
  function format(value) {
    if (value === null || value === undefined) { return '–'; }
    return Math.abs(value) >= 1000 ? value.toLocaleString('en-US', {maximumFractionDigits: 1}) : String(value);
  }
  function render() {
    var g = groups[group.value];
    var rows = (g.atlases[atlas.value] || []).slice();
    var text = search.value.trim().toLowerCase();
    if (text) {
      rows = rows.filter(function (row) { return row[0].toLowerCase().indexOf(text) !== -1; });
    }
    if (sort.column >= 0) {
      rows.sort(function (a, b) {
        var x = a[sort.column], y = b[sort.column];
        if (x === null || y === null) { return (x === null) - (y === null); }
        var order = typeof x === 'number' ? x - y : String(x).localeCompare(String(y), undefined, {numeric: true});
        return sort.up ? order : -order;
      });
    }
    body.innerHTML = '';
    rows.forEach(function (row) {
      var line = body.insertRow();
      for (var i = 0; i < 7; i++) {
        var cell = line.insertCell();
        if (i < 2) {
          cell.className = 'text';
          cell.textContent = row[i];
        } else {
          cell.textContent = format(row[i]);
          if (row[i] === null || row[i] === undefined) { cell.className = 'empty'; }
        }
      }
    });
    shown.textContent = rows.length + ' regions · units: ' + g.units;
  }

  Array.prototype.forEach.call(table.tHead.rows[0].cells, function (head, column) {
    head.addEventListener('click', function () {
      sort.up = sort.column === column ? !sort.up : true;
      sort.column = column;
      Array.prototype.forEach.call(table.tHead.rows[0].cells, function (other) {
        other.removeAttribute('aria-sort');
      });
      head.setAttribute('aria-sort', sort.up ? 'ascending' : 'descending');
      render();
    });
  });
  group.addEventListener('change', function () { fillAtlases(); render(); });
  atlas.addEventListener('change', render);
  search.addEventListener('input', render);

  fillAtlases();
  render();
})();
"""


QC_PAGE_JS = """
(function () {
  document.getElementById('print-button').addEventListener('click', function () { window.print(); });

  // Full-size image in a new tab
  document.querySelectorAll('.figure img').forEach(function (img) {
    img.title = 'Open the full-size image in a new tab';
    img.addEventListener('click', function () { window.open(img.currentSrc || img.src, '_blank'); });
  });

  // Previous and next subject, from the list written by the index of the cohort
  var cohort = window.QC_COHORT;
  if (!cohort || !cohort.subjects) { return; }
  var base = '../../qcqa/';
  var at = cohort.subjects.map(function (s) { return s.id; }).indexOf(document.body.dataset.subject);
  function link(id, target) {
    var node = document.getElementById(id);
    node.href = encodeURI(base + target);
    node.hidden = false;
  }
  link('nav-cohort', 'index.html');
  if (at > 0) { link('nav-prev', cohort.subjects[at - 1].report); }
  if (at >= 0 && at < cohort.subjects.length - 1) { link('nav-next', cohort.subjects[at + 1].report); }
  document.addEventListener('keydown', function (event) {
    var tag = event.target.tagName;
    if (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey ||
        tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') { return; }
    var node = {ArrowLeft: 'nav-prev', ArrowRight: 'nav-next'}[event.key];
    if (node && !document.getElementById(node).hidden) { document.getElementById(node).click(); }
  });
})();
"""


def qc_build_html(metadata: dict, legends: dict, regions: list = None) -> str:
    """
    Web page of the quality control report, from its metadata and region tables.

    ``regions`` is the morphometry table of the subject arranged by
    ``qc_regions_data``, for the table of regional measures.
    """
    from html import escape
    from urllib.parse import quote

    subj_id = escape(metadata["subject_id"])
    status = escape(metadata["processing_status"])
    reportlets = metadata["reportlets"]
    base_image = quote(reportlets[0]["image"])
    t1 = metadata["t1"]

    def row(key, value):
        return (
            f'<li><span class="key">{key}</span><span class="value">{value}</span></li>'
        )

    missing = [r["id"] for r in reportlets if r["status"] == "missing"]
    n_slices = metadata["n_slices"]
    summary = [
        row("Subject ID", f"<strong>{subj_id}</strong>"),
        row("Processing status", f'<span class="badge {status}">{status}</span>'),
        row("Subjects directory", f"<code>{escape(metadata['subjects_dir'])}</code>"),
        row("FreeSurfer version", escape(metadata["freesurfer_version"] or "unknown")),
        row(
            "Last line of recon-all.log",
            escape(metadata["recon_all_last_log_line"] or "not available"),
        ),
        row(
            "T1 image",
            " &times; ".join(str(d) for d in t1["dimensions"])
            + " voxels of "
            + " &times; ".join(f"{z:g}" for z in t1["voxel_size_mm"])
            + " mm",
        ),
    ]
    for report in reportlets:
        if "volume_cm3" in report:
            summary.append(
                row(
                    f"{escape(report['title'])} volume",
                    f"{report['volume_cm3']:.1f} cm&sup3;",
                )
            )
    summary += [
        row(
            "Files found",
            ", ".join(f"{escape(k)}: {v}" for k, v in metadata["files_found"].items()),
        ),
        row(
            "Missing reports",
            ", ".join(f"<code>{escape(m)}</code>" for m in missing)
            if missing
            else "none",
        ),
        row(
            "Slices",
            f"{n_slices['x']} sagittal (X), {n_slices['y']} coronal (Y), "
            f"{n_slices['z']} axial (Z)",
        ),
        row("Report generated", escape(metadata["generated"])),
    ]

    def cell(value):
        if value is None:
            return '<td class="empty">&ndash;</td>'
        return (
            f"<td>{value:,}</td>"
            if isinstance(value, int)
            else f"<td>{value:,.2f}</td>"
        )

    measures = metadata.get("measures") or []
    if measures:
        measures_table = (
            '<table class="measures"><thead><tr><th>Measure</th><th>Left</th>'
            "<th>Right</th><th>Total</th><th>Units</th></tr></thead><tbody>"
            + "".join(
                f"<tr><th>{escape(m['name'])}</th>{cell(m['lh'])}{cell(m['rh'])}"
                f'{cell(m["total"])}<td class="units">{escape(m["units"])}</td></tr>'
                for m in measures
            )
            + "</tbody></table>"
        )
    else:
        measures_table = (
            '<p class="lead">The stats files and the recon-all.log were not found.</p>'
        )

    # Morphometry table of the subject: lobes and all the regions
    stats_info = metadata.get("stats_table") or {"status": "disabled"}
    morphometry, morphometry_link = "", ""
    if stats_info["status"] != "disabled":
        morphometry_link = '<a href="#morphometry">Morphometry</a>'
        morphometry = '<h2 id="morphometry">Morphometry</h2>'
    if stats_info["status"] == "error":
        morphometry += (
            '<div class="alert"><span class="badge error">error</span> '
            f"{escape(stats_info['note'])}</div>"
        )
    elif stats_info["status"] == "ok":
        thick_name, thick_units = (escape(t) for t in qc_metric_label("thickness"))
        lobes = metadata.get("lobes") or []
        if lobes:
            keys = [
                f"{key}_{hemi}"
                for key in ("thickness", "white_area", "pial_area")
                for hemi in ("lh", "rh")
            ]
            morphometry += (
                '<p class="lead">Hemispheres and lobes of the Desikan-Killiany atlas.</p>'
                '<table class="measures lobes"><thead><tr><th rowspan="2">Region</th>'
                f'<th colspan="2">{thick_name} ({thick_units})</th>'
                '<th colspan="2">White surface area (cm&sup2;)</th>'
                '<th colspan="2">Pial surface area (cm&sup2;)</th></tr>'
                "<tr><th>Left</th><th>Right</th><th>Left</th><th>Right</th>"
                "<th>Left</th><th>Right</th></tr></thead><tbody>"
                + "".join(
                    f"<tr><th>{escape(lobe['region'])}</th>"
                    + "".join(cell(lobe[k]) for k in keys)
                    + "</tr>"
                    for lobe in lobes
                )
                + "</tbody></table>"
            )
        table_link = quote(stats_info["file"])
        morphometry += (
            f'<div class="caption">Get table file: <a href="{table_link}" '
            f'target="_blank">{escape(stats_info["file"])}</a> '
            f"({stats_info['n_rows']:,} rows)</div>"
        )
        if regions:
            data = json.dumps(regions, separators=(",", ":")).replace("</", "<\\/")
            morphometry += (
                "<details><summary>All regional measures</summary>"
                '<div class="toolbar explorer">'
                '<select id="stats-group" aria-label="Measure"></select>'
                '<select id="stats-atlas" aria-label="Atlas"></select>'
                '<input type="search" id="stats-search" placeholder="Filter by region" '
                'aria-label="Filter by region">'
                '<span id="stats-shown" class="caption"></span></div>'
                '<div class="table-wrap explorer"><table class="cohort" id="stats-table">'
                '<thead><tr><th class="text">Region</th><th class="text">Hemisphere</th>'
                "<th>Value</th><th>Median</th><th>Std</th><th>Min</th><th>Max</th>"
                "</tr></thead><tbody></tbody></table></div></details>"
                f'<script type="application/json" id="stats-data">{data}</script>'
            )

    def rating(rid, label):
        return (
            f'<div class="rating" data-id="{escape(rid)}"><span class="label">Rating</span>'
            '<button type="button" data-value="pass">Pass</button>'
            '<button type="button" data-value="doubtful">Doubtful</button>'
            '<button type="button" data-value="fail">Fail</button>'
            f'<input type="text" data-field="comment" placeholder="Comment" '
            f'aria-label="Comment on {escape(label)}"></div>'
        )

    sections = []
    for report in reportlets:
        rid = "reportlet-" + re.sub(r"[^A-Za-z0-9]+", "-", report["id"])
        head = f"<h3>{escape(report['title'])}</h3>"
        body = f"<p>{escape(report['description'])}</p>"

        if report["status"] != "ok":
            body += (
                f'<div class="alert"><span class="badge {report["status"]}">'
                f"{report['status']}</span> {escape(report['note'])}</div>"
            )
        else:
            image = quote(report["image"])
            if report["kind"] in ("anatomical", "surface_map", "surface_annot"):
                figure = f'<img src="{image}" alt="{escape(report["title"])}">'
            else:
                figure = (
                    f'<img id="{rid}-img" src="{image}" data-overlay="{image}" '
                    f'data-base="{base_image}" alt="{escape(report["title"])} over the T1 image">'
                )
                head += (
                    f'<button type="button" data-target="{rid}-img" '
                    'aria-pressed="false">Hide overlay</button>'
                )
            body += f'<div class="figure">{figure}</div>'
            body += (
                f'<div class="caption">Get figure file: <a href="{image}" '
                f'target="_blank">{escape(report["image"])}</a></div>'
            )

            unlisted = report.get("labels_not_in_color_table")
            if unlisted:
                body += (
                    '<div class="caption">Labels that are not in the color table '
                    f"(drawn in white): {', '.join(str(u) for u in unlisted)}</div>"
                )

            if report["id"] in legends:
                items = "".join(
                    f'<li><span class="swatch" style="background:{escape(r.color)}"></span>'
                    f'<span class="name" title="{escape(r.name)}">{escape(r.name)}</span>'
                    f'<span class="vol">{r.index} &middot; {r.volume_mm3 / 1000:.2f} cm&sup3;</span></li>'
                    for r in legends[report["id"]].itertuples()
                )
                table = quote(report["regions_table"])
                body += (
                    f"<details><summary>{report['n_regions']} regions "
                    f'(<a href="{table}" target="_blank">table</a>)</summary>'
                    f'<ul class="legend">{items}</ul></details>'
                )

        if report["status"] == "ok":
            body += rating(report["id"], report["title"])

        sections.append(
            f'<section class="reportlet" id="{rid}"><div class="head">{head}</div>{body}</section>'
        )

    nav_links = "".join(
        f'<a href="#reportlet-{re.sub(r"[^A-Za-z0-9]+", "-", r["id"])}" '
        f'data-rated="{escape(r["id"])}">{escape(r["id"])}</a>'
        for r in reportlets
    )
    color_table = metadata["color_table"]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{subj_id} - FreeSurfer visual report</title>
<style>{QC_CSS}{QC_COHORT_CSS}</style>
</head>
<body data-subject="{subj_id}" data-subjects-dir="{escape(metadata["subjects_dir"])}">
<nav>
  <span class="brand">{subj_id}</span>
  <a href="#summary">Summary</a>
  <a href="#measures">Measures</a>
  {morphometry_link}
  {nav_links}
  <a href="#rating">Rating</a>
  <a href="#about">About</a>
  <span class="nav-right">
    <a id="nav-prev" hidden title="Previous subject (left arrow)">&lsaquo; Previous</a>
    <a id="nav-cohort" hidden>Cohort</a>
    <a id="nav-next" hidden title="Next subject (right arrow)">Next &rsaquo;</a>
    <button type="button" id="print-button" title="Print the report or save it as PDF">Print / PDF</button>
  </span>
</nav>
<main>
  <h1>FreeSurfer: individual visual report</h1>
  <p class="lead">Subject <strong>{subj_id}</strong></p>

  <h2 id="summary">Summary</h2>
  <ul class="summary">
    {"".join(summary)}
  </ul>

  <h2 id="measures">Global measures</h2>
  {measures_table}

  {morphometry}

  <h2 id="visual-reports">Visual reports</h2>
  {"".join(sections)}

  <h2 id="rating">Rating</h2>
  <p class="lead">Rate each figure above and the subject as a whole. The rating is kept in
  this browser; download it to keep it with the subject (for example in the
  <code>qcqa/metadata</code> folder).</p>
  <div class="overall">
    <div class="rating" data-id=""><span class="label">Subject</span>
      <button type="button" data-value="pass">Pass</button>
      <button type="button" data-value="doubtful">Doubtful</button>
      <button type="button" data-value="fail">Fail</button>
      <input type="text" data-field="rater" placeholder="Rater" aria-label="Name of the rater">
    </div>
    <textarea data-field="comment" rows="3" placeholder="Comments about the subject"
      aria-label="Comments about the subject"></textarea>
    <div class="actions">
      <button type="button" id="rating-download">Download rating (JSON)</button>
      <button type="button" id="rating-clear">Clear rating</button>
      <span id="rating-count" class="caption"></span>
    </div>
    <div class="caption shortcuts">Keyboard: <kbd>j</kbd> / <kbd>k</kbd> next / previous figure &middot;
      <kbd>p</kbd> <kbd>d</kbd> <kbd>f</kbd> pass / doubtful / fail for the figure in view &middot;
      <kbd>Shift</kbd> + <kbd>p</kbd> <kbd>d</kbd> <kbd>f</kbd> for the subject &middot;
      <kbd>&larr;</kbd> / <kbd>&rarr;</kbd> previous / next subject
    </div>
  </div>

  <h2 id="about">About</h2>
  <ul class="summary">
    {row("Color table", f"<code>{escape(color_table)}</code>" if color_table else "not used")}
    {row("Overlay opacity", f"{metadata['overlay_alpha']:g}")}
    {row("Display convention", "Neurological: the left hemisphere is on the left of the image. The coordinates are scanner millimeters.")}
    {row("Metadata", '<a href="metadata/qc_metadata.json" target="_blank">metadata/qc_metadata.json</a>')}
  </ul>
  <footer>Generated with clabtoolkit on {escape(metadata["generated"])}.</footer>
</main>
<script src="../../qcqa/subjects.js"></script>
<script>{QC_JS}{QC_RATING_JS}{QC_STATS_JS}{QC_PAGE_JS}</script>
</body>
</html>
"""


QC_COHORT_CSS = """
.toolbar { display: flex; align-items: center; gap: 10px 16px; flex-wrap: wrap; margin: 16px 0; }
.toolbar button { font: inherit; font-size: .9rem; padding: 6px 12px; cursor: pointer; color: var(--ink);
  background: #fff; border: 1px solid #adb5bd; border-radius: 4px; }
.toolbar button:hover { background: var(--soft); }
.toolbar input, .toolbar select { font: inherit; font-size: .9rem; padding: 6px 10px; color: var(--ink);
  background: #fff; border: 1px solid #ced4da; border-radius: 4px; }
.toolbar input { flex: 0 1 280px; }
.toolbar .caption { margin: 0; }
.toolbar.filters { gap: 10px 32px; margin-top: 0; }
.chips { display: flex; align-items: center; gap: 6px 10px; flex-wrap: wrap; }
.chips .label { color: var(--muted); font-size: .875rem; }
.chip { display: inline-flex; align-items: center; gap: 5px; cursor: pointer; font-size: .875rem; white-space: nowrap; }
.chip input { margin: 0; cursor: pointer; }
.chip .n { color: var(--muted); font-variant-numeric: tabular-nums; }
.chip input:not(:checked) + .badge { background: #adb5bd; }
.chips button { font-size: .8rem; padding: 2px 8px; border-color: #ced4da; color: var(--muted); }
.badge.none { background: #6c757d; }
.hint { margin: 0 0 12px; }
table.cohort .pick-cell { width: 1%; padding-right: 0; text-align: center; cursor: default; }
table.cohort .pick-cell input { margin: 0; cursor: pointer; }
@media print { .hint, table.cohort .pick-cell { display: none; } }
.counts { display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 4px; }
.table-wrap { overflow-x: auto; }
table.cohort { border-collapse: collapse; width: 100%; font-size: .92rem; }
table.cohort th, table.cohort td { padding: 7px 12px; border-bottom: 1px solid var(--line); text-align: right;
  white-space: nowrap; font-variant-numeric: tabular-nums; }
table.cohort thead th { color: var(--muted); font-weight: 500; border-bottom-width: 2px; cursor: pointer;
  user-select: none; }
table.cohort thead th:hover { color: var(--ink); }
table.cohort thead th[aria-sort="ascending"]::after { content: " \\25B2"; font-size: .7em; }
table.cohort thead th[aria-sort="descending"]::after { content: " \\25BC"; font-size: .7em; }
table.cohort th.text, table.cohort td.text { text-align: left; }
table.cohort tbody tr:hover { background: var(--soft); }
table.cohort td.empty { color: var(--muted); }
table.cohort td.note { color: var(--muted); white-space: nowrap; text-align: left; font-size: .85rem; }
table.lobes { margin-bottom: 8px; }
table.lobes thead tr:first-child th { text-align: center; border-bottom-width: 1px; }
table.lobes thead tr:first-child th:first-child { text-align: left; vertical-align: bottom; }
table.lobes thead tr + tr th:first-child { text-align: right; }
.toolbar.explorer { margin: 12px 14px; }
.table-wrap.explorer { max-height: 520px; overflow: auto; margin: 0 14px 14px; }
.badge.pass { background: #198754; }
.badge.doubtful { background: #b97a00; }
.badge.fail { background: #dc3545; }
"""

QC_COHORT_JS = """
(function () {
  var table = document.getElementById('cohort-table');
  var body = table.tBodies[0];
  var rows = Array.prototype.slice.call(body.rows);
  var search = document.getElementById('cohort-search');
  var shown = document.getElementById('cohort-shown');
  var pickAll = document.getElementById('pick-all');
  var groups = {
    status: document.querySelectorAll('#cohort-status input'),
    rating: document.querySelectorAll('#cohort-rating input')
  };

  function checked(boxes) {
    return Array.prototype.filter.call(boxes, function (box) { return box.checked; })
                .map(function (box) { return box.value; });
  }
  function picked(row) { return row.querySelector('input.pick').checked; }
  function visible() { return rows.filter(function (row) { return !row.hidden; }); }

  // Rows of the checked statuses and ratings whose name has the text
  function filter() {
    var text = search.value.trim().toLowerCase();
    var statuses = checked(groups.status), ratings = checked(groups.rating);
    rows.forEach(function (row) {
      row.hidden = !((!text || row.dataset.subject.toLowerCase().indexOf(text) !== -1) &&
                     statuses.indexOf(row.dataset.status) !== -1 &&
                     ratings.indexOf(row.dataset.rating) !== -1);
    });
    count();
  }
  function count() {
    var rowsShown = visible(), selected = rowsShown.filter(picked).length;
    shown.textContent = selected + ' selected · ' + rowsShown.length + ' of ' + rows.length + ' rows shown';
    pickAll.checked = rowsShown.length > 0 && selected === rowsShown.length;
    pickAll.indeterminate = selected > 0 && selected < rowsShown.length;
  }

  search.addEventListener('input', filter);
  Object.keys(groups).forEach(function (name) {
    Array.prototype.forEach.call(groups[name], function (box) { box.addEventListener('change', filter); });
  });
  document.querySelectorAll('.chips button[data-set]').forEach(function (button) {
    button.addEventListener('click', function () {
      Array.prototype.forEach.call(groups[button.dataset.group], function (box) {
        box.checked = button.dataset.set === 'all';
      });
      filter();
    });
  });

  // Rows ticked by hand: the box of the header ticks or unticks the rows shown
  body.addEventListener('change', function (event) {
    if (event.target.classList.contains('pick')) { count(); }
  });
  pickAll.addEventListener('change', function () {
    visible().forEach(function (row) { row.querySelector('input.pick').checked = pickAll.checked; });
    count();
  });

  // Identifiers of the rows shown and ticked, in the order of the table
  function part(boxes) {
    var values = checked(boxes);
    if (values.length === boxes.length) { return ''; }
    return values.map(function (v) { return v === 'none' ? 'not-rated' : v; }).join('+') || 'none';
  }
  document.getElementById('cohort-download').addEventListener('click', function () {
    var rowsShown = visible();
    var ids = rowsShown.filter(picked).map(function (row) { return row.dataset.subject; });
    var name = ['fs_ids', part(groups.status) || 'all'];
    if (part(groups.rating)) { name.push(part(groups.rating)); }
    if (ids.length !== rowsShown.length || search.value.trim()) { name.push('selection'); }
    var link = document.createElement('a');
    link.href = URL.createObjectURL(new Blob([ids.join('\\n') + (ids.length ? '\\n' : '')], {type: 'text/plain'}));
    link.download = name.join('_') + '.txt';
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(link.href);
  });

  Array.prototype.forEach.call(table.tHead.rows[0].cells, function (head, column) {
    if (head.classList.contains('pick-cell')) { return; }
    head.addEventListener('click', function () {
      var up = head.getAttribute('aria-sort') !== 'ascending';
      Array.prototype.forEach.call(table.tHead.rows[0].cells, function (other) {
        other.removeAttribute('aria-sort');
      });
      head.setAttribute('aria-sort', up ? 'ascending' : 'descending');
      var numeric = !head.classList.contains('text');
      rows.sort(function (a, b) {
        var x = a.cells[column].dataset.value, y = b.cells[column].dataset.value;
        // The rows without a value always go last
        if (x === '' || y === '') { return (x === '') - (y === ''); }
        var order = numeric ? parseFloat(x) - parseFloat(y) : x.localeCompare(y, undefined, {numeric: true});
        return up ? order : -order;
      });
      rows.forEach(function (row) { body.appendChild(row); });
    });
  });

  filter();
})();
"""


def qc_build_cohort_html(
    rows: list,
    subjs_dir: str,
    generated: str,
    outliers: pd.DataFrame = None,
    outlier_info: dict = None,
    has_ratings: bool = False,
) -> str:
    """
    Index web page of the quality control reports of a cohort, from its rows.

    ``outliers`` is the table of ``qc_cohort_outliers`` (None if they were not
    looked for) and ``outlier_info`` the threshold, the minimum number of
    subjects and the number of morphometry tables that were found.
    ``has_ratings`` tells that the table of ratings of the cohort was saved.
    The rows with the status "not-freesurfer" are folders that are not
    FreeSurfer output folders: they are not counted as subjects.
    """
    from html import escape
    from urllib.parse import quote

    def cell(value, digits=None, css=""):
        if value is None:
            return f'<td class="empty {css}" data-value="">&ndash;</td>'
        text = f"{value:.{digits}f}" if digits is not None else str(value)
        return (
            f'<td class="{css}" data-value="{escape(str(value))}">{escape(text)}</td>'
        )

    def badge(value):
        if not value:
            return '<td class="empty text" data-value="">&ndash;</td>'
        value = escape(str(value))
        return (
            f'<td class="text" data-value="{value}">'
            f'<span class="badge {value}">{value}</span></td>'
        )

    lines = []
    for row in rows:
        subj_id = escape(row["subject_id"])
        if row["report"]:
            subject = f'<a href="{quote(row["report"])}">{subj_id}</a>'
            report = f'<a href="{quote(row["report"])}">open</a>'
        else:
            subject, report = subj_id, "&ndash;"
        rating = badge(row["rating"])
        if row["rating"] and row["rater"]:
            rating = rating.replace(
                "<td ", f'<td title="Rated by {escape(row["rater"])}" ', 1
            )
        note = (
            f'<td class="note">{escape(row["note"])}</td>'
            if row.get("note")
            else "<td></td>"
        )
        lines.append(
            f'<tr data-subject="{subj_id}" data-status="{escape(row["status"])}" '
            f'data-rating="{escape(row["rating"] or "none")}">'
            f'<td class="pick-cell"><input type="checkbox" class="pick" checked '
            f'aria-label="Select {subj_id}"></td>'
            f'<td class="text" data-value="{subj_id}">{subject}</td>'
            + badge(row["status"])
            + cell(row["holes"])
            + cell(row["defects"])
            + cell(row.get("outliers"))
            + cell(row["etiv_cm3"], 1)
            + cell(row["thickness_lh_mm"], 2)
            + cell(row["thickness_rh_mm"], 2)
            + rating
            + f'<td class="text{"" if row["report"] else " empty"}" '
            f'data-value="{escape(row["report_date"] or "")}">{report}'
            + (
                f' <span class="caption">{escape(row["report_date"])}</span>'
                if row["report_date"]
                else ""
            )
            + "</td>"
            + note
            + "</tr>"
        )

    thickness_name, thickness_units = (escape(t) for t in qc_metric_label("thickness"))

    # Regional measures far from the ones of the cohort
    info = outlier_info or {"threshold": 4.0, "min_subjects": 20, "n_tables": 0}
    reports = {row["subject_id"]: row["report"] for row in rows}
    if outliers is None:
        outlier_section = (
            f'<p class="lead">The outliers are looked for when at least '
            f"{info['min_subjects']} subjects have a morphometry table in their report "
            f"({info['n_tables']} found). Generate the individual reports with "
            "<code>stats_table=True</code> to compare the subjects.</p>"
        )
    elif outliers.empty:
        outlier_section = (
            f'<p class="lead">No regional measure of the {info["n_tables"]} subjects with a '
            f"morphometry table is farther than {info['threshold']:g} robust standard "
            "deviations from the median of the cohort.</p>"
        )
    else:
        max_rows = 300
        outlier_lines = []
        for item in outliers.head(max_rows).itertuples():
            subj_id = escape(item.subject_id)
            if reports.get(item.subject_id):
                subj_id = f'<a href="{quote(reports[item.subject_id])}">{subj_id}</a>'
            digits = 2 if abs(item.value) < 100 else 0
            outlier_lines.append(
                f'<tr><td class="text">{subj_id}</td>'
                f'<td class="text">{escape(qc_metric_label(item.metric)[0])} '
                f"({escape(item.units)})</td>"
                f'<td class="text">{escape(item.atlas or "aseg")}</td>'
                f'<td class="text">{escape(item.region)}</td>'
                f"<td>{item.value:,.{digits}f}</td><td>{item.cohort_median:,.{digits}f}</td>"
                f"<td>{item.z:+.1f}</td></tr>"
            )
        more = (
            f" The {max_rows} farthest ones are shown."
            if len(outliers) > max_rows
            else ""
        )
        outlier_section = (
            f'<p class="lead">{len(outliers)} regional measures of the {info["n_tables"]} '
            f"subjects with a morphometry table are farther than {info['threshold']:g} "
            "robust standard deviations from the median of the cohort."
            f'{more} Table: <a href="metadata/cohort_outliers.tsv" target="_blank">'
            "metadata/cohort_outliers.tsv</a>. Some flags are expected by chance, more "
            "in small cohorts, and volumes and areas depend on the size of the head.</p>"
            '<div class="table-wrap"><table class="cohort"><thead><tr>'
            '<th class="text">Subject</th><th class="text">Measure</th>'
            '<th class="text">Atlas</th><th class="text">Region</th><th>Value</th>'
            "<th>Cohort median</th><th>Distance (SD)</th></tr></thead><tbody>"
            + "".join(outlier_lines)
            + "</tbody></table></div>"
        )

    ratings_link = (
        '<a href="metadata/cohort_ratings.tsv" target="_blank">metadata/cohort_ratings.tsv</a>'
        if has_ratings
        else "no ratings found"
    )
    statuses = [
        "processed",
        "autorecon2",
        "autorecon1",
        "unprocessed",
        "not-freesurfer",
    ]
    counts = {s: sum(r["status"] == s for r in rows) for s in statuses}
    n_subjects = len(rows) - counts["not-freesurfer"]
    count_badges = "".join(
        f'<span class="badge {s}">{s}: {n}</span>' for s, n in counts.items() if n
    )
    n_reports = sum(bool(r["report"]) for r in rows)
    n_rated = sum(bool(r["rating"]) for r in rows)

    def chips(group, title, items):
        boxes = "".join(
            f'<label class="chip"><input type="checkbox" value="{value}" checked>'
            f'<span class="badge {value}">{label}</span>'
            f'<span class="n">{n}</span></label>'
            for value, label, n in items
        )
        return (
            f'<div class="chips" id="cohort-{group}" role="group" aria-label="{title}">'
            f'<span class="label">{title}</span>{boxes}'
            f'<button type="button" data-group="{group}" data-set="all">all</button>'
            f'<button type="button" data-group="{group}" data-set="none">none</button></div>'
        )

    status_chips = chips(
        "status", "Status", [(s, s, counts[s]) for s in statuses if counts[s]]
    )
    rating_chips = chips(
        "rating",
        "Rating",
        [
            (value, label, sum((r["rating"] or "none") == value for r in rows))
            for value, label in (
                ("pass", "pass"),
                ("doubtful", "doubtful"),
                ("fail", "fail"),
                ("none", "not rated"),
            )
        ],
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FreeSurfer cohort report</title>
<style>{QC_CSS}{QC_COHORT_CSS}@media print {{ @page {{ size: landscape; }} table.cohort {{ font-size: 8pt; }} }}</style>
</head>
<body>
<nav>
  <span class="brand">Cohort</span>
  <a href="#summary">Summary</a>
  <a href="#subjects">Subjects</a>
  <a href="#outliers">Outliers</a>
  <span class="nav-right">
    <button type="button" id="print-button" title="Print the report or save it as PDF">Print / PDF</button>
  </span>
</nav>
<main>
  <h1>FreeSurfer: cohort visual report</h1>
  <p class="lead"><code>{escape(subjs_dir)}</code></p>

  <h2 id="summary">Summary</h2>
  <ul class="summary">
    <li><span class="key">Subjects</span><span class="value">{n_subjects}</span></li>
    <li><span class="key">Other folders</span><span class="value">{counts["not-freesurfer"]} (not FreeSurfer output folders)</span></li>
    <li><span class="key">Processing status</span><span class="value counts">{count_badges or "&ndash;"}</span></li>
    <li><span class="key">Individual reports</span><span class="value">{n_reports} of {n_subjects}</span></li>
    <li><span class="key">Rated subjects</span><span class="value">{n_rated} of {n_subjects}</span></li>
    <li><span class="key">Table</span><span class="value"><a href="metadata/cohort_qc.tsv" target="_blank">metadata/cohort_qc.tsv</a></span></li>
    <li><span class="key">Ratings table</span><span class="value">{ratings_link}</span></li>
    <li><span class="key">Report generated</span><span class="value">{escape(generated)}</span></li>
  </ul>

  <h2 id="subjects">Subjects</h2>
  <div class="toolbar">
    <input type="search" id="cohort-search" placeholder="Filter by subject" aria-label="Filter by subject">
    <button type="button" id="cohort-download"
      title="Text file with the identifiers of the rows shown and ticked, one per line">Download IDs (.txt)</button>
    <span id="cohort-shown" class="caption"></span>
  </div>
  <div class="toolbar filters">
    {status_chips}
    {rating_chips}
  </div>
  <p class="caption hint">Untick a status or a rating to hide its rows, and untick the box of a row
    to leave it out. The download has the identifiers of the rows that are shown and ticked.</p>
  <div class="table-wrap">
  <table class="cohort" id="cohort-table">
    <thead><tr>
      <th class="pick-cell"><input type="checkbox" id="pick-all" checked
        aria-label="Select or unselect the rows shown" title="Select or unselect the rows shown"></th>
      <th class="text">Subject</th><th class="text">Status</th>
      <th title="Holes of both surfaces before the topology correction">Holes</th>
      <th title="Defects of both surfaces found by the topology correction">Defects</th>
      <th title="Regional measures far from the ones of the cohort">Outliers</th>
      <th title="Estimated total intracranial volume">eTIV (cm&sup3;)</th>
      <th title="Mean of the left hemisphere">{thickness_name} lh ({thickness_units})</th>
      <th title="Mean of the right hemisphere">{thickness_name} rh ({thickness_units})</th>
      <th class="text">Rating</th><th class="text">Report</th><th class="text">Notes</th>
    </tr></thead>
    <tbody>
      {"".join(lines)}
    </tbody>
  </table>
  </div>
  <h2 id="outliers">Outlier regions</h2>
  {outlier_section}

  <footer>Generated with clabtoolkit on {escape(generated)}. Click a column title of the table of subjects to sort it.</footer>
</main>
<script>{QC_COHORT_JS}
document.getElementById('print-button').addEventListener('click', function () {{ window.print(); }});
</script>
</body>
</html>
"""
