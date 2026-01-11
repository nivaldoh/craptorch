#!/usr/bin/env python3
import argparse
import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from craptorch.core.tensor import Tensor


def ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib import patches
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for visualization. Install it with:\n"
            "  pip install -r requirements-viz.txt"
        ) from exc
    return plt, patches


def parse_matrix(text):
    rows = []
    for raw_row in text.strip().split(";"):
        row = raw_row.replace(",", " ").strip()
        if not row:
            continue
        rows.append([float(val) for val in row.split()])
    if not rows:
        raise ValueError("Matrix text is empty.")
    row_lens = {len(row) for row in rows}
    if len(row_lens) != 1:
        raise ValueError("All rows must have the same number of columns.")
    return np.array(rows, dtype=np.float32)


def parse_shape(text):
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise ValueError("Shape must be like 2x3.")
    return int(parts[0]), int(parts[1])


def format_value(value, precision):
    val = float(value)
    rounded = round(val)
    if abs(val - rounded) < 1e-8:
        return str(int(rounded))
    return f"{val:.{precision}g}"


def split_output_path(output):
    base, ext = os.path.splitext(output)
    if not ext:
        return output, ".png"
    return base, ext


def annotate_matrix(ax, data, precision, show_numbers=True):
    n_rows, n_cols = data.shape
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(range(n_cols))
    ax.set_yticklabels(range(n_rows))
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    if not show_numbers:
        return
    for row in range(n_rows):
        for col in range(n_cols):
            ax.text(
                col,
                row,
                format_value(data[row, col], precision),
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )


def build_text_panel(a, b, i, j, precision):
    a_row = a[i, :]
    b_col = b[:, j]
    products = a_row * b_col
    terms = [f"{format_value(a_row[k], precision)}*{format_value(b_col[k], precision)}"
             for k in range(len(products))]
    sum_expr = " + ".join(terms) if terms else "0"
    sum_val = float(products.sum()) if products.size else 0.0
    lines = [
        f"C[{i},{j}] = sum_k A[{i},k] * B[k,{j}]",
        f"= {sum_expr}",
        f"= {format_value(sum_val, precision)}",
        "",
        f"A row {i}: [{', '.join(format_value(v, precision) for v in a_row)}]",
        f"B col {j}: [{', '.join(format_value(v, precision) for v in b_col)}]",
        "",
        f"Row combo: C[{i},:] = sum_k A[{i},k] * B[k,:]",
        f"Col combo: C[:,{j}] = sum_k A[:,k] * B[k,{j}]",
    ]
    return "\n".join(lines)


def plot_row_combo(ax, a, b, c, row_idx):
    cols = np.arange(b.shape[1])
    ax.axhline(0, color="gray", linewidth=0.6)
    for k in range(a.shape[1]):
        contrib = a[row_idx, k] * b[k, :]
        ax.plot(cols, contrib, marker="o", alpha=0.6, label=f"k={k}")
    ax.plot(cols, c[row_idx, :], marker="o", color="black", linewidth=2, label="sum")
    ax.set_title(f"C[{row_idx},:] from scaled B rows")
    ax.set_xlabel("col")
    ax.set_ylabel("value")
    if a.shape[1] <= 8:
        ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(True, alpha=0.3)


def plot_col_combo(ax, a, b, c, col_idx):
    rows = np.arange(a.shape[0])
    ax.axhline(0, color="gray", linewidth=0.6)
    for k in range(a.shape[1]):
        contrib = a[:, k] * b[k, col_idx]
        ax.plot(rows, contrib, marker="o", alpha=0.6, label=f"k={k}")
    ax.plot(rows, c[:, col_idx], marker="o", color="black", linewidth=2, label="sum")
    ax.set_title(f"C[:,{col_idx}] from scaled A cols")
    ax.set_xlabel("row")
    ax.set_ylabel("value")
    if a.shape[1] <= 8:
        ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(True, alpha=0.3)


def render_overview(a, b, c, cell, output, cmap, precision, show_numbers):
    plt, patches = ensure_matplotlib()
    i, j = cell

    fig, axes = plt.subplots(2, 3, figsize=(13, 9), constrained_layout=True)
    ax_a, ax_b, ax_c = axes[0]
    ax_row, ax_col, ax_text = axes[1]

    im_a = ax_a.imshow(a, cmap=cmap)
    ax_a.set_title(f"A {a.shape}")
    annotate_matrix(ax_a, a, precision, show_numbers)
    ax_a.add_patch(
        patches.Rectangle(
            (-0.5, i - 0.5),
            a.shape[1],
            1,
            fill=False,
            edgecolor="orange",
            linewidth=2,
        )
    )

    im_b = ax_b.imshow(b, cmap=cmap)
    ax_b.set_title(f"B {b.shape}")
    annotate_matrix(ax_b, b, precision, show_numbers)
    ax_b.add_patch(
        patches.Rectangle(
            (j - 0.5, -0.5),
            1,
            b.shape[0],
            fill=False,
            edgecolor="orange",
            linewidth=2,
        )
    )

    im_c = ax_c.imshow(c, cmap=cmap)
    ax_c.set_title(f"C = A @ B {c.shape}")
    annotate_matrix(ax_c, c, precision, show_numbers)
    ax_c.add_patch(
        patches.Rectangle(
            (j - 0.5, i - 0.5),
            1,
            1,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
    )

    plot_row_combo(ax_row, a, b, c, i)
    plot_col_combo(ax_col, a, b, c, j)

    ax_text.axis("off")
    text = build_text_panel(a, b, i, j, precision)
    ax_text.text(
        0.0,
        1.0,
        text,
        ha="left",
        va="top",
        transform=ax_text.transAxes,
        fontsize=10,
        family="monospace",
    )

    for im in (im_a, im_b, im_c):
        im.set_clim(
            min(a.min(), b.min(), c.min()),
            max(a.max(), b.max(), c.max()),
        )

    fig.savefig(output, dpi=150)
    return fig


def render_outer_products(a, b, c, output, cmap, precision, show_numbers, max_terms):
    plt, _ = ensure_matplotlib()
    shared = a.shape[1]
    if max_terms is not None and shared > max_terms:
        return None

    outer_mats = [np.outer(a[:, k], b[k, :]) for k in range(shared)]
    mats = outer_mats + [c]

    vmin = min(mat.min() for mat in mats)
    vmax = max(mat.max() for mat in mats)

    panels = len(mats)
    cols = min(3, panels)
    rows = int(math.ceil(panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, mat in enumerate(outer_mats):
        ax = axes[idx]
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"k={idx}: outer(A[:,{idx}], B[{idx},:])")
        annotate_matrix(ax, mat, precision, show_numbers)
        im.set_clim(vmin, vmax)

    sum_ax = axes[len(outer_mats)]
    im_sum = sum_ax.imshow(c, cmap=cmap, vmin=vmin, vmax=vmax)
    sum_ax.set_title("sum = C")
    annotate_matrix(sum_ax, c, precision, show_numbers)
    im_sum.set_clim(vmin, vmax)

    for ax in axes[len(mats):]:
        ax.axis("off")

    fig.savefig(output, dpi=150)
    return fig


def transform_points(points, matrix):
    return points @ matrix.T


def render_transform_view(a, b, c, output):
    plt, _ = ensure_matplotlib()
    grid_vals = np.linspace(-1, 1, 5)
    line_vals = np.linspace(-1, 1, 80)
    lines = []
    for x in grid_vals:
        lines.append(np.column_stack([np.full_like(line_vals, x), line_vals]))
    for y in grid_vals:
        lines.append(np.column_stack([line_vals, np.full_like(line_vals, y)]))

    mats = {
        "Input (I)": np.eye(2, dtype=np.float32),
        "After B": b,
        "After A": a,
        "After A @ B": c,
    }

    all_points = []
    for mat in mats.values():
        for line in lines:
            all_points.append(transform_points(line, mat))
        all_points.append(transform_points(np.array([[1, 0], [0, 1]], dtype=np.float32), mat))

    stacked = np.vstack(all_points)
    pad = 0.2
    x_min, x_max = stacked[:, 0].min() - pad, stacked[:, 0].max() + pad
    y_min, y_max = stacked[:, 1].min() - pad, stacked[:, 1].max() + pad

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    axes = axes.flatten()
    for ax, (title, mat) in zip(axes, mats.items()):
        for line in lines:
            transformed = transform_points(line, mat)
            ax.plot(transformed[:, 0], transformed[:, 1], color="lightgray", linewidth=0.8)
        e1 = mat @ np.array([1, 0], dtype=np.float32)
        e2 = mat @ np.array([0, 1], dtype=np.float32)
        ax.arrow(0, 0, e1[0], e1[1], color="tab:red", width=0.01, length_includes_head=True)
        ax.arrow(0, 0, e2[0], e2[1], color="tab:green", width=0.01, length_includes_head=True)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    fig.savefig(output, dpi=150)
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 2D matrix multiplication with a focused output cell."
    )
    parser.add_argument(
        "--a",
        help='Matrix A values, e.g. "1 2; 3 4".',
    )
    parser.add_argument(
        "--b",
        help='Matrix B values, e.g. "5 6; 7 8".',
    )
    parser.add_argument(
        "--a-shape",
        help="Random A shape like 2x3 (used if --a is not provided).",
    )
    parser.add_argument(
        "--b-shape",
        help="Random B shape like 3x2 (used if --b is not provided).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for generated matrices.",
    )
    parser.add_argument(
        "--min",
        dest="min_val",
        type=int,
        default=-2,
        help="Minimum random integer value (inclusive).",
    )
    parser.add_argument(
        "--max",
        dest="max_val",
        type=int,
        default=3,
        help="Maximum random integer value (inclusive).",
    )
    parser.add_argument(
        "--cell",
        default="0,0",
        help="Focused output cell as row,col (default: 0,0).",
    )
    parser.add_argument(
        "--views",
        default="overview,outer,transform",
        help="Comma-separated views: overview,outer,transform.",
    )
    parser.add_argument(
        "--max-outer-terms",
        type=int,
        default=6,
        help="Max shared-dim terms to show in the outer-product view.",
    )
    parser.add_argument(
        "--output",
        default="matmul.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving.",
    )
    parser.add_argument(
        "--cmap",
        default="Blues",
        help="Matplotlib colormap name.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Number formatting precision.",
    )
    parser.add_argument(
        "--no-annot",
        action="store_true",
        help="Disable numeric annotations in matrix cells.",
    )
    args = parser.parse_args()

    if args.min_val > args.max_val:
        raise SystemExit("--min must be <= --max.")

    if args.a:
        a = parse_matrix(args.a)
    elif args.a_shape:
        a_shape = parse_shape(args.a_shape)
        rng = np.random.default_rng(args.seed)
        a = rng.integers(args.min_val, args.max_val + 1, size=a_shape).astype(np.float32)
    else:
        a = np.array([[1, 2, -1], [0, 3, 4]], dtype=np.float32)

    if args.b:
        b = parse_matrix(args.b)
    elif args.b_shape:
        b_shape = parse_shape(args.b_shape)
        rng = np.random.default_rng(args.seed + 1)
        b = rng.integers(args.min_val, args.max_val + 1, size=b_shape).astype(np.float32)
    else:
        b = np.array([[2, 1], [0, -3], [1, 4]], dtype=np.float32)

    if a.ndim != 2 or b.ndim != 2:
        raise SystemExit("Only 2D matrices are supported.")

    if a.shape[1] != b.shape[0]:
        raise SystemExit(f"Incompatible shapes: {a.shape} @ {b.shape}")

    tensor_c = Tensor(a).matmul(Tensor(b))
    c = tensor_c.data

    cell_parts = args.cell.split(",")
    if len(cell_parts) != 2:
        raise SystemExit("--cell must be formatted as row,col.")
    cell = (int(cell_parts[0]), int(cell_parts[1]))
    if not (0 <= cell[0] < c.shape[0] and 0 <= cell[1] < c.shape[1]):
        raise SystemExit(f"Cell {cell} is out of bounds for output shape {c.shape}.")

    views = {view.strip() for view in args.views.split(",") if view.strip()}
    valid_views = {"overview", "outer", "transform"}
    unknown = views - valid_views
    if unknown:
        raise SystemExit(f"Unknown views: {', '.join(sorted(unknown))}")

    base, ext = split_output_path(args.output)
    outputs = []
    figures = []

    if "overview" in views:
        overview_path = f"{base}{ext}"
        fig = render_overview(
            a,
            b,
            c,
            cell,
            overview_path,
            args.cmap,
            args.precision,
            not args.no_annot,
        )
        outputs.append(overview_path)
        figures.append(fig)

    if "outer" in views:
        if args.max_outer_terms is not None and a.shape[1] > args.max_outer_terms:
            print(
                f"Skipped outer view: shared dim {a.shape[1]} "
                f"> max {args.max_outer_terms}."
            )
        else:
            outer_path = f"{base}_outer{ext}"
            fig = render_outer_products(
                a,
                b,
                c,
                outer_path,
                args.cmap,
                args.precision,
                not args.no_annot,
                args.max_outer_terms,
            )
            if fig is not None:
                outputs.append(outer_path)
                figures.append(fig)

    if "transform" in views:
        if a.shape == (2, 2) and b.shape == (2, 2):
            transform_path = f"{base}_transform{ext}"
            fig = render_transform_view(a, b, c, transform_path)
            outputs.append(transform_path)
            figures.append(fig)
        else:
            print("Skipped transform view: only available for 2x2 matrices.")

    if outputs:
        for path in outputs:
            print(f"Wrote {path}")

    if args.show and figures:
        plt, _ = ensure_matplotlib()
        plt.show()
    if figures:
        plt, _ = ensure_matplotlib()
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main()
