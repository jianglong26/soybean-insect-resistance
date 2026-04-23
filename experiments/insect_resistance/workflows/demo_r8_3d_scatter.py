"""Standalone method demo: median-plane 3D eight-region scatter with Z8 highlight.

No external input is required. The script builds a deterministic synthetic cloud,
splits the 3D space by three median planes into 8 regions, and only highlights Z8
using the same color as the project source code.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


R8_COLOR = "#17BECF"  # Keep the same Z8 color as source code.


def region_id(similarity: float, ndm: float, yield_value: float, sim_med: float, ndm_med: float, yld_med: float) -> int:
    """Return region id in [0..7] using the same bit-rule as the main pipeline.

    bit0: similarity >= median (high similarity)
    bit1: ndm <= median (early maturity)
    bit2: yield >= median (high yield)
    """
    return int(similarity >= sim_med) + 2 * int(ndm <= ndm_med) + 4 * int(yield_value >= yld_med)


def generate_demo_points() -> np.ndarray:
    """Create deterministic synthetic points spanning all 8 regions.

    Columns: [similarity, ndm, nocontrol_yield]
    """
    rng = np.random.default_rng(20260413)

    sim_med = 0.60
    ndm_med = 110.0
    yld_med = 1500.0

    points = []

    # 8 regions, 3 points each (24 total) as a clear demonstration set.
    for rid in range(8):
        high_sim = (rid & 1) > 0
        early = (rid & 2) > 0
        high_yield = (rid & 4) > 0

        sim_center = 0.72 if high_sim else 0.48
        ndm_center = 103.0 if early else 118.0
        yld_center = 1750.0 if high_yield else 1250.0

        sim = rng.normal(sim_center, 0.03, size=3)
        ndm = rng.normal(ndm_center, 2.0, size=3)
        yld = rng.normal(yld_center, 85.0, size=3)

        sim = np.clip(sim, 0.0, 1.0)
        for i in range(3):
            points.append([sim[i], ndm[i], yld[i]])

    return np.asarray(points, dtype=float)


def main() -> None:
    out_dir = Path("experiments/insect_resistance/outputs/method_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = generate_demo_points()
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # Use fixed split values for method illustration so the three split anchors
    # are visually aligned and stable across runs.
    sim_med = 0.65
    ndm_med = 110.0
    yld_med = 1600.0

    regions = np.array([region_id(xi, yi, zi, sim_med, ndm_med, yld_med) for xi, yi, zi in zip(x, y, z)])
    is_r8 = regions == 7

    fig = plt.figure(figsize=(10.8, 8.8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw all 8 octants as translucent volumes, with Z8 slightly emphasized.
    def _cuboid_faces(x0, x1, y0, y1, z0, z1):
        v000 = (x0, y0, z0)
        v001 = (x0, y0, z1)
        v010 = (x0, y1, z0)
        v011 = (x0, y1, z1)
        v100 = (x1, y0, z0)
        v101 = (x1, y0, z1)
        v110 = (x1, y1, z0)
        v111 = (x1, y1, z1)
        return [
            [v000, v001, v011, v010],
            [v100, v101, v111, v110],
            [v000, v001, v101, v100],
            [v010, v011, v111, v110],
            [v000, v010, v110, v100],
            [v001, v011, v111, v101],
        ]

    def _draw_median_planes(ax3d, x_med, y_med, z_med, xlim_v, ylim_v, zlim_v):
        yy, zz = np.meshgrid(np.linspace(ylim_v[0], ylim_v[1], 2), np.linspace(zlim_v[0], zlim_v[1], 2))
        xx = np.full_like(yy, x_med)
        ax3d.plot_surface(xx, yy, zz, color="#616161", alpha=0.15, linewidth=0.35, edgecolor="#424242", shade=False)

        xx2, zz2 = np.meshgrid(np.linspace(xlim_v[0], xlim_v[1], 2), np.linspace(zlim_v[0], zlim_v[1], 2))
        yy2 = np.full_like(xx2, y_med)
        ax3d.plot_surface(xx2, yy2, zz2, color="#64b5f6", alpha=0.15, linewidth=0.35, edgecolor="#1e88e5", shade=False)

        xx3, yy3 = np.meshgrid(np.linspace(xlim_v[0], xlim_v[1], 2), np.linspace(ylim_v[0], ylim_v[1], 2))
        zz3 = np.full_like(xx3, z_med)
        ax3d.plot_surface(xx3, yy3, zz3, color="#81c784", alpha=0.15, linewidth=0.35, edgecolor="#2e7d32", shade=False)

    # No scatter points: this figure is only for method illustration of Z8 location.

    # Fixed axis bounds (publication/demo style) to keep region boundaries clear.
    xlim = (0.35, 0.85)
    ylim = (98.0, 122.0)
    zlim = (1050.0, 1900.0)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    x_segments = [(xlim[0], sim_med), (sim_med, xlim[1])]
    y_segments = [(ndm_med, ylim[1]), (ylim[0], ndm_med)]
    z_segments = [(zlim[0], yld_med), (yld_med, zlim[1])]
    for xb in (0, 1):
        for yb in (0, 1):
            for zb in (0, 1):
                rid = xb + 2 * yb + 4 * zb
                if rid != 7:
                    continue
                x0, x1 = x_segments[xb]
                y0, y1 = y_segments[yb]
                z0, z1 = z_segments[zb]
                if (x1 - x0) <= 1e-12 or (y1 - y0) <= 1e-12 or (z1 - z0) <= 1e-12:
                    continue
                box = Poly3DCollection(
                    _cuboid_faces(x0, x1, y0, y1, z0, z1),
                    facecolors=R8_COLOR,
                    edgecolors="none",
                    linewidths=0.0,
                    alpha=0.14,
                )
                ax.add_collection3d(box)

    _draw_median_planes(ax, sim_med, ndm_med, yld_med, xlim, ylim, zlim)

    # Median plane split lines (same visual language as source code).
    ax.plot([xlim[0], xlim[1]], [ndm_med, ndm_med], [yld_med, yld_med], "--", color="#6E6E6E", linewidth=1.2)
    ax.plot([sim_med, sim_med], [ylim[0], ylim[1]], [yld_med, yld_med], "--", color="#6E6E6E", linewidth=1.2)
    ax.plot([sim_med, sim_med], [ndm_med, ndm_med], [zlim[0], zlim[1]], "--", color="#6E6E6E", linewidth=1.2)

    # Label the center of Z8 cuboid.
    r8_x = (sim_med + xlim[1]) * 0.5
    r8_y = (ylim[0] + ndm_med) * 0.5
    r8_z = (yld_med + zlim[1]) * 0.5
    ax.text(
        r8_x,
        r8_y,
        r8_z,
        "Z8 zone",
        fontsize=11,
        color=R8_COLOR,
        fontweight="bold",
    )

    ax.set_xlabel("Feature Similarity", fontsize=11, labelpad=8)
    ax.set_ylabel("NDM (days)", fontsize=11, labelpad=8)
    ax.set_zlabel("Performance Metric", fontsize=13, labelpad=7)
    ax.zaxis.label.set_clip_on(False)

    # Drop the two corner-most boundary ticks to avoid projection overlap like "0.85122".
    x_ticks = [0.35, 0.45, 0.55, 0.65, 0.75]
    y_ticks = [98, 104, 110, 116]
    z_ticks = [1200, 1400, 1600, 1800]

    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.zaxis.set_major_locator(FixedLocator(z_ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:.2f}" for t in x_ticks]))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{t:d}" for t in y_ticks]))
    ax.zaxis.set_major_formatter(FixedFormatter([f"{t:d}" for t in z_ticks]))

    # Disable any minor/offset labels so only explicit tick labels remain.
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.zaxis.set_minor_locator(NullLocator())
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)
    ax.zaxis.get_offset_text().set_visible(False)

    ax.view_init(elev=23, azim=-57)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.tick_params(axis="x", which="major", labelsize=12, pad=9)
    ax.tick_params(axis="y", which="major", labelsize=12, pad=3)
    ax.zaxis.set_tick_params(labelsize=12)

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=R8_COLOR, markeredgecolor="#2f2f2f", markersize=10, label="Z8 zone"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.94), frameon=True, fontsize=11)

    # Match spacing style of the average 3D scatter from source code.
    fig.subplots_adjust(left=0.02, right=0.86, top=0.6, bottom=0.1)

    png_path = out_dir / "r8_3d_scatter_method_demo.png"
    pdf_path = out_dir / "r8_3d_scatter_method_demo.pdf"
    fig.savefig(png_path, dpi=300, pad_inches=0.12, facecolor="white", bbox_inches="tight", bbox_extra_artists=[ax.zaxis.label])
    fig.savefig(pdf_path, dpi=300, pad_inches=0.12, facecolor="white", bbox_inches="tight", bbox_extra_artists=[ax.zaxis.label])
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()

