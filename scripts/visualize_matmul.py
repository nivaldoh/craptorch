#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from craptorch.core.tensor import Tensor


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
    ]
    return "\n".join(lines)


def visualize(a, b, cell, output, show, cmap, precision, show_numbers):
    try:
        import matplotlib.pyplot as plt
        from matplotlib import patches
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for visualization. Install it with:\n"
            "  pip install -r requirements-viz.txt"
        ) from exc

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Only 2D matrices are supported.")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible shapes: {a.shape} @ {b.shape}")

    tensor_c = Tensor(a).matmul(Tensor(b))
    c = tensor_c.data

    i, j = cell
    if not (0 <= i < c.shape[0] and 0 <= j < c.shape[1]):
        raise ValueError(f"Cell {cell} is out of bounds for output shape {c.shape}.")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_text = axes.flat

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
    if show:
        plt.show()
    plt.close(fig)


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

    cell_parts = args.cell.split(",")
    if len(cell_parts) != 2:
        raise SystemExit("--cell must be formatted as row,col.")
    cell = (int(cell_parts[0]), int(cell_parts[1]))

    visualize(
        a,
        b,
        cell,
        args.output,
        args.show,
        args.cmap,
        args.precision,
        not args.no_annot,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
