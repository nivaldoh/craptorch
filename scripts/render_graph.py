#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import craptorch.core.autograd  # enable autograd monkey-patching
from craptorch.core.graph import render_graph
from craptorch.core.tensor import Tensor


def example_basic():
    x = Tensor([2.0], requires_grad=True)
    y = x * 3
    z = y*2 + 1
    return z, {"x": x, "y": y, "z": z}


def example_branch():
    x = Tensor([2.0, -1.0, 0.5], requires_grad=True)
    y = x * 2
    z = x * x
    out = (y + z).sum()
    return out, {"x": x, "y": y, "z": z, "out": out}


def example_matmul():
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    w = Tensor(np.array([[0.1, 0.2], [0.3, 0.4]]), requires_grad=True)
    b = Tensor(np.array([0.5, -0.5]), requires_grad=True)
    out = (x.matmul(w) + b).sum()
    return out, {"x": x, "w": w, "b": b, "out": out}


def build_example(name):
    if name == "basic":
        return example_basic()
    if name == "branch":
        return example_branch()
    if name == "matmul":
        return example_matmul()
    raise ValueError(f"Unknown example: {name}")


def main():
    parser = argparse.ArgumentParser(description="Render a craptorch autograd graph.")
    parser.add_argument(
        "--example",
        choices=["basic", "branch", "matmul"],
        default="branch",
        help="Which example graph to render.",
    )
    parser.add_argument(
        "--output",
        default="graph",
        help="Base path (without extension) for the output files.",
    )
    parser.add_argument(
        "--format",
        default="png",
        help="Image format for Graphviz dot (e.g. png, svg).",
    )
    parser.add_argument(
        "--no-grad-stats",
        action="store_true",
        help="Hide gradient summary stats (min/max/norm).",
    )
    parser.add_argument(
        "--grad-values",
        action="store_true",
        help="Show full gradient values in labels.",
    )
    parser.add_argument(
        "--tensor-values",
        action="store_true",
        help="Show full tensor values in labels.",
    )
    parser.add_argument(
        "--rankdir",
        choices=["LR", "TB"],
        default="TB",
        help="Graph layout direction (LR or TB).",
    )
    args = parser.parse_args()

    output, nodes = build_example(args.example)

    trace = output.backward(trace=True)
    names = {tensor: name for name, tensor in nodes.items()}
    dot_path, img_path = render_graph(
        output,
        path=args.output,
        fmt=args.format,
        trace=trace,
        names=names,
        show_grad_stats=not args.no_grad_stats,
        show_grad_values=args.grad_values,
        show_tensor_values=args.tensor_values,
        rankdir=args.rankdir,
    )

    print(f"Wrote DOT: {dot_path}")
    if img_path is None:
        print("Graphviz 'dot' not found; rendered image not produced.")
    else:
        print(f"Wrote image: {img_path}")

    if args.example in {"branch", "matmul"}:
        print("Example grads:")
        for name, tensor in nodes.items():
            if tensor.grad is not None:
                print(f"  {name}.grad = {tensor.grad}")


if __name__ == "__main__":
    main()
