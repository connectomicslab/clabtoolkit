"""
Generate the clabtoolkit ecosystem figure.

A fan-shaped diagram (inspired by the scientific Python ecosystem figure) that
maps how clabtoolkit's modules build on top of Python and the wider scientific
stack. Concentric arcs, from the inside out:

    1. language     - Python
    2. core         - foundation modules + key external libraries
    3. I/O & 3D     - data ingestion, format handling, 3D structures
    4. analysis     - high-level neuroimaging analysis & workflow tools

Run from the repository root::

    python docs/source/_static/figures/make_ecosystem_figure.py

Outputs PNG (transparent + white) and SVG next to this script.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge


# --------------------------------------------------------------------------- #
# Layout configuration                                                        #
# --------------------------------------------------------------------------- #

# Each item: (label, color, weight, italic)
# `external` flag controls whether the label is drawn upright (italic external
# library names) or as a clabtoolkit module (bold italic, branded color).
LAYERS = [
    {
        "key": "language",
        "radius": (0.0, 1.05),
        "facecolor": "#FFF7E6",
        "edgecolor": "#E8C77B",
        "items": [
            ("Python", "#306998", "bold", False),
        ],
    },
    {
        "key": "core",
        "radius": (1.05, 2.20),
        "facecolor": "#EAF3FB",
        "edgecolor": "#9CC3E5",
        # split into two sub-rows: inner row = external libs, outer = clabtoolkit
        "items_inner": [
            ("NumPy",      "#4D77CF", "bold", False),
            ("SciPy",      "#0054A6", "bold", False),
            ("nibabel",    "#2C8A99", "bold", False),
            ("pandas",     "#150458", "bold", False),
            ("matplotlib", "#11557C", "bold", False),
        ],
        "items_outer": [
            ("misctools",   "#B5651D", "bold", True),
            ("bidstools",   "#B5651D", "bold", True),
            ("plottools",   "#B5651D", "bold", True),
            ("colorstools", "#B5651D", "bold", True),
        ],
    },
    {
        "key": "io_structures",
        "radius": (2.20, 3.20),
        "facecolor": "#F1ECF9",
        "edgecolor": "#B79CE0",
        "items_inner": [
            ("imagetools",      "#5B2C82", "bold", True),
            ("freesurfertools", "#5B2C82", "bold", True),
            ("dicomtools",      "#5B2C82", "bold", True),
            ("dwitools",        "#5B2C82", "bold", True),
        ],
        "items_outer": [
            ("surfacetools",      "#5B2C82", "bold", True),
            ("tracttools",        "#5B2C82", "bold", True),
            ("segmentationtools", "#5B2C82", "bold", True),
            ("pointstools",       "#5B2C82", "bold", True),
        ],
    },
    {
        "key": "analysis",
        "radius": (3.20, 4.30),
        "facecolor": "#FDECEC",
        "edgecolor": "#E5908F",
        "items_inner": [
            ("parcellationtools", "#A02C2C", "bold", True),
            ("morphometrytools",  "#A02C2C", "bold", True),
            ("connectivitytools", "#A02C2C", "bold", True),
            ("networktools",      "#A02C2C", "bold", True),
        ],
        "items_outer": [
            ("connectome",         "#A02C2C", "bold", True),
            ("visualizationtools", "#A02C2C", "bold", True),
            ("pipelinetools",      "#A02C2C", "bold", True),
            ("qcqatools",          "#A02C2C", "bold", True),
        ],
    },
]

# Fan geometry: wedges fill a full half-circle (0..180), but item placement is
# restricted to a narrower angular band so labels never crowd the bottom edges.
WEDGE_START = 0
WEDGE_END = 180
ITEM_START = 22
ITEM_END = 158


def _angles_for(n: int, offset_deg: float = 0.0) -> np.ndarray:
    """Evenly spaced angles within the item placement band.

    `offset_deg` shifts placements within the band (used to interleave rows).
    The result is clipped to stay within [ITEM_START, ITEM_END].
    """
    if n == 1:
        return np.array([(ITEM_START + ITEM_END) / 2.0])
    angles = np.linspace(ITEM_START, ITEM_END, n) + offset_deg
    return np.clip(angles, ITEM_START, ITEM_END)


def _place(items, radius: float, ax, base_fontsize: float, offset_deg: float = 0.0):
    angles = _angles_for(len(items), offset_deg)
    for angle_deg, (text, color, weight, italic) in zip(angles, items):
        theta = math.radians(angle_deg)
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        ax.text(
            x, y, text,
            ha="center", va="center",
            color=color, fontsize=base_fontsize, weight=weight,
            style="italic" if italic else "normal",
            zorder=5,
        )


def draw_ecosystem(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 9.2), dpi=200)
    ax.set_aspect("equal")
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-1.05, 5.15)
    ax.axis("off")

    # Draw rings outer-first so inner overpaints cleanly.
    for layer in reversed(LAYERS):
        r0, r1 = layer["radius"]
        ax.add_patch(Wedge(
            center=(0, 0),
            r=r1,
            theta1=WEDGE_START,
            theta2=WEDGE_END,
            width=r1 - r0,
            facecolor=layer["facecolor"],
            edgecolor=layer["edgecolor"],
            linewidth=1.4,
            zorder=2,
        ))

    # Place text on each ring. The innermost ring (Python) gets one row at the
    # ring center; outer rings get two staggered (interleaved) rows so labels
    # don't sit radially aligned.
    for layer in LAYERS:
        r0, r1 = layer["radius"]
        if "items" in layer:
            r_mid = (r0 + r1) / 2.0
            _place(layer["items"], r_mid, ax, base_fontsize=16)
        else:
            r_lo = r0 + (r1 - r0) * 0.30
            r_hi = r0 + (r1 - r0) * 0.74
            fs = 12.5 if r0 >= 2.5 else 12.0
            _place(layer["items_inner"], r_lo, ax, base_fontsize=fs)
            _place(layer["items_outer"], r_hi, ax, base_fontsize=fs)

    # Bottom-right captions: each label sits just below the +x axis, at the
    # outer radius of its ring, slightly tilted to evoke the curve. This is
    # the same convention as the reference scientific-Python figure.
    caption_y = -0.32
    captions = [
        (LAYERS[0]["radius"][1], "language",       4),
        (LAYERS[1]["radius"][1], "core",           7),
        (LAYERS[2]["radius"][1], "I/O & structures", 10),
        (LAYERS[3]["radius"][1], "analysis & viz", 13),
    ]
    for r, label, tilt in captions:
        ax.text(
            r, caption_y, label,
            ha="right", va="top",
            fontsize=13, color="#555555", style="italic",
            rotation=-tilt, rotation_mode="anchor",
        )

    # Title block
    ax.text(
        0, 4.92, "clabtoolkit ecosystem",
        ha="center", va="center",
        fontsize=27, weight="bold", color="#222222",
    )
    ax.text(
        0, 4.55,
        "a Python toolkit for connectomics, BIDS, and neuroimaging analysis",
        ha="center", va="center",
        fontsize=13, color="#666666", style="italic",
    )

    fig.tight_layout()

    png_path = output_dir / "clabtoolkit_ecosystem.png"
    png_white_path = output_dir / "clabtoolkit_ecosystem_white.png"
    svg_path = output_dir / "clabtoolkit_ecosystem.svg"

    fig.savefig(png_path, dpi=220, transparent=True, bbox_inches="tight")
    fig.savefig(png_white_path, dpi=220, facecolor="white", bbox_inches="tight")
    fig.savefig(svg_path, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {png_white_path}")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    draw_ecosystem(here)
