import subprocess
import shutil

import numpy as np

from craptorch.core.tensor import Tensor


def _escape_label(text):
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    return text.replace("\\\\n", "\\n")


def _format_array(arr, max_items=6):
    arr = np.array(arr)
    if arr.size <= max_items:
        return np.array2string(arr, separator=", ")
    return np.array2string(arr, separator=", ", threshold=4, edgeitems=2)


def _format_name(value):
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    return str(value)


def _format_array_full(arr):
    arr = np.array(arr)
    return np.array2string(arr, separator=", ", threshold=arr.size, max_line_width=1000000)


def _format_grad_summary(arr):
    arr = np.array(arr)
    if arr.size == 1:
        return f"{float(arr):.6g}"
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    norm = float(np.linalg.norm(arr))
    return f"shape={arr.shape}, min={min_val:.3g}, max={max_val:.3g}, norm={norm:.3g}"


def _format_const(value, max_len=60):
    text = repr(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def build_graph(output):
    tensor_nodes = {}
    op_nodes = {}
    const_nodes = {}
    edges = []
    visited = set()

    def add_tensor(tensor):
        key = id(tensor)
        if key not in tensor_nodes:
            tensor_nodes[key] = {"id": f"t{len(tensor_nodes)}", "tensor": tensor}
        return tensor_nodes[key]["id"]

    def add_op(fn):
        key = id(fn)
        if key not in op_nodes:
            op_nodes[key] = {"id": f"op{len(op_nodes)}", "fn": fn}
        return op_nodes[key]["id"]

    def add_const(value):
        key = id(value)
        if key not in const_nodes:
            const_nodes[key] = {"id": f"c{len(const_nodes)}", "value": value}
        return const_nodes[key]["id"]

    def visit(tensor):
        key = id(tensor)
        if key in visited:
            return
        visited.add(key)

        add_tensor(tensor)
        grad_fn = getattr(tensor, "_grad_fn", None)
        if grad_fn is None:
            return

        op_id = add_op(grad_fn)
        edges.append((op_id, tensor_nodes[key]["id"]))

        for item in getattr(grad_fn, "saved_tensors", []):
            if isinstance(item, Tensor):
                item_id = add_tensor(item)
                edges.append((item_id, op_id))
                visit(item)
            else:
                const_id = add_const(item)
                edges.append((const_id, op_id))

    visit(output)

    return {
        "tensors": tensor_nodes,
        "ops": op_nodes,
        "consts": const_nodes,
        "edges": edges,
    }


def _normalize_names(names):
    if not names:
        return {}
    normalized = {}
    for key, value in names.items():
        if isinstance(key, Tensor):
            normalized[id(key)] = value
        else:
            normalized[key] = value
    return normalized


def to_dot(
    output,
    trace=None,
    names=None,
    show_grad=True,
    show_grad_stats=True,
    show_grad_values=False,
    show_tensor_values=False,
    rankdir="LR",
):
    graph = build_graph(output)
    trace_steps = {}
    trace_grads = {}
    names = _normalize_names(names)

    if trace is not None:
        for idx, entry in enumerate(trace.get("order", [])):
            tensor = entry.get("tensor") if isinstance(entry, dict) else entry
            if isinstance(tensor, Tensor) and id(tensor) not in trace_steps:
                trace_steps[id(tensor)] = idx
        trace_grads = trace.get("grads", {})

    lines = [
        "digraph ComputationGraph {",
        f"  rankdir={rankdir};",
        '  node [fontname="Courier"];',
    ]

    for node in graph["tensors"].values():
        tensor = node["tensor"]
        label_parts = [f"shape={tensor.shape}"]
        name = names.get(id(tensor))
        if name:
            label_parts.insert(0, _format_name(name))
        if not tensor.requires_grad:
            label_parts.append("requires_grad=False")
        step = trace_steps.get(id(tensor))
        if step is not None:
            label_parts.append(f"step={step}")
        if show_tensor_values:
            label_parts.append(f"data={_format_array_full(tensor.data)}")
        if show_grad:
            grad_val = trace_grads.get(id(tensor))
            if grad_val is None and tensor.grad is not None:
                grad_val = tensor.grad
            if grad_val is not None:
                if show_grad_stats:
                    label_parts.append(f"grad={_format_grad_summary(grad_val)}")
                if show_grad_values:
                    label_parts.append(f"grad_values={_format_array_full(grad_val)}")
        label = _escape_label("\\n".join(label_parts))
        lines.append(f'  {node["id"]} [shape=ellipse,label="{label}"];')

    for node in graph["ops"].values():
        op_name = node["fn"].__class__.__name__
        label = _escape_label(op_name)
        lines.append(f'  {node["id"]} [shape=box,label="{label}"];')

    for node in graph["consts"].values():
        const_label = f"const\\n{_format_const(node['value'])}"
        label = _escape_label(const_label)
        lines.append(f'  {node["id"]} [shape=diamond,label="{label}"];')

    for src, dst in graph["edges"]:
        lines.append(f"  {src} -> {dst};")

    lines.append("}")
    return "\n".join(lines)


def render_graph(
    output,
    path="graph",
    fmt="png",
    trace=None,
    names=None,
    show_grad=True,
    show_grad_stats=True,
    show_grad_values=False,
    show_tensor_values=False,
    rankdir="LR",
):
    dot = to_dot(
        output,
        trace=trace,
        names=names,
        show_grad=show_grad,
        show_grad_stats=show_grad_stats,
        show_grad_values=show_grad_values,
        show_tensor_values=show_tensor_values,
        rankdir=rankdir,
    )
    dot_path = f"{path}.dot"
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write(dot)

    dot_bin = shutil.which("dot")
    if dot_bin is None:
        return dot_path, None

    out_path = f"{path}.{fmt}"
    subprocess.run([dot_bin, f"-T{fmt}", dot_path, "-o", out_path], check=False)
    return dot_path, out_path
