import argparse
import numpy as np
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from craptorch.core.tensor import Tensor
from craptorch.core.optimizers import SGD, Adam, AdamW

LOG_SCALE = True
PLOT_DPI = 140


def ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plots. Install it with:\n"
            "  pip install matplotlib"
        ) from exc
    return plt


def format_loss(value):
    if not np.isfinite(value):
        return "nan" if np.isnan(value) else "inf"
    abs_value = abs(value)
    if abs_value == 0:
        return "0"
    if abs_value < 1e-4 or abs_value >= 1e4:
        return f"{value:.3e}"
    return f"{value:.6f}"


def apply_log_scale(ax, values, use_log):
    if not use_log:
        return
    series = np.asarray(values, dtype=float)
    series = series[np.isfinite(series)]
    if series.size == 0:
        return
    if np.nanmin(series) > 0:
        ax.set_yscale("log")


def make_quadratic(weights):
    weights = np.asarray(weights, dtype=float)

    def loss(x):
        return 0.5 * np.sum(weights * (x ** 2))

    def grad(x):
        return weights * x

    return loss, grad


def rosenbrock_loss(x, a=1.0, b=100.0):
    x0, x1 = x[0], x[1]
    return (a - x0) ** 2 + b * (x1 - x0 ** 2) ** 2


def rosenbrock_grad(x, a=1.0, b=100.0):
    x0, x1 = x[0], x[1]
    grad0 = -2 * (a - x0) - 4 * b * x0 * (x1 - x0 ** 2)
    grad1 = 2 * b * (x1 - x0 ** 2)
    return np.array([grad0, grad1], dtype=float)


def build_optimizer_specs(base_lr):
    return [
        ("SGD", SGD, {"lr": base_lr}),
        ("SGD+Momentum", SGD, {"lr": base_lr, "momentum": 0.9}),
        ("Adam", Adam, {"lr": base_lr}),
        ("AdamW", AdamW, {"lr": base_lr, "weight_decay": 0.01}),
    ]


def run_optimizer(optimizer_class, kwargs, loss_fn, grad_fn, x_start, steps, target=None):
    param = Tensor(x_start.copy(), requires_grad=True)
    optimizer = optimizer_class([param], **kwargs)
    losses = []
    distances = []
    status = "ok"
    stop_step = steps

    for step in range(steps + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            loss = float(loss_fn(param.data))
        losses.append(loss)
        if target is not None:
            distances.append(float(np.linalg.norm(param.data - target)))
        else:
            distances.append(np.nan)
        if not np.isfinite(loss):
            status = "diverged"
            stop_step = step
            break

        if step < steps:
            with np.errstate(over="ignore", invalid="ignore"):
                grad = grad_fn(param.data)
            param.grad = Tensor(grad)
            optimizer.step()
            optimizer.zero_grad()

    if len(losses) < steps + 1:
        losses.extend([np.nan] * (steps + 1 - len(losses)))
    if len(distances) < steps + 1:
        distances.extend([np.nan] * (steps + 1 - len(distances)))

    return {
        "losses": losses,
        "distances": distances,
        "status": status,
        "stop_step": stop_step,
    }


def milestone_steps(total_steps):
    steps = [0, total_steps // 4, total_steps // 2, (3 * total_steps) // 4, total_steps]
    return sorted(set(step for step in steps if step >= 0))


def print_summary_table(results, steps):
    milestones = milestone_steps(steps)
    name_width = max(12, max(len(name) for name, _ in results) + 2)
    col_width = 12

    header = f"{'Optimizer':<{name_width}}"
    for step in milestones:
        header += f"{f'step{step}':<{col_width}}"
    print(header)
    print("-" * len(header))

    for name, data in results:
        losses = data["losses"]
        row = f"{name:<{name_width}}"
        for step in milestones:
            value = losses[step] if step < len(losses) else np.nan
            row += f"{format_loss(value):<{col_width}}"
        print(row)


def plot_scenarios(scenarios, scenario_results, output_path=None, show=True):
    if not show and not output_path:
        return

    plt = ensure_matplotlib()
    num_scenarios = len(scenarios)
    fig, axes = plt.subplots(
        2,
        num_scenarios,
        figsize=(5.2 * num_scenarios, 7.0),
        constrained_layout=True,
    )
    if num_scenarios == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for idx, scenario in enumerate(scenarios):
        results = scenario_results[idx]
        loss_series = []
        distance_series = []

        ax_loss = axes[0, idx]
        ax_dist = axes[1, idx]

        for name, data in results:
            steps = np.arange(len(data["losses"]))
            losses = np.asarray(data["losses"], dtype=float)
            distances = np.asarray(data["distances"], dtype=float)
            line = ax_loss.plot(steps, losses, label=name)[0]
            ax_dist.plot(steps, distances, color=line.get_color(), label=name)
            loss_series.append(losses)
            distance_series.append(distances)

        ax_loss.set_title(scenario["name"])
        ax_loss.set_xlabel("step")
        ax_loss.set_ylabel("loss")
        ax_loss.grid(True, alpha=0.3)

        distance_label = scenario.get("distance_label", "distance")
        ax_dist.set_xlabel("step")
        ax_dist.set_ylabel(distance_label)
        ax_dist.grid(True, alpha=0.3)

        if loss_series:
            apply_log_scale(ax_loss, np.concatenate(loss_series), LOG_SCALE)
        if distance_series:
            apply_log_scale(ax_dist, np.concatenate(distance_series), LOG_SCALE)

        ax_loss.legend(frameon=False, fontsize=8)

    if output_path:
        fig.savefig(output_path, dpi=PLOT_DPI)
    if show:
        plt.show()
    plt.close(fig)


def analyze_optimizer_convergence_behavior(show=True, output_path=None):
    """Analyze convergence behavior of different optimizers with richer diagnostics."""
    print("📊 Analyzing Optimizer Convergence Behavior...")

    quadratic_loss, quadratic_grad = make_quadratic([1.0, 1.0, 1.0])
    ill_loss, ill_grad = make_quadratic([1.0, 10.0, 100.0])

    scenarios = [
        {
            "name": "Quadratic (well-conditioned)",
            "description": "f(x) = 0.5 * sum(x^2)",
            "loss_fn": quadratic_loss,
            "grad_fn": quadratic_grad,
            "x_start": np.array([5.0, -3.0, 2.0], dtype=float),
            "target": np.zeros(3, dtype=float),
            "distance_label": "||x||",
            "steps": 80,
            "optimizers": build_optimizer_specs(base_lr=0.1),
        },
        {
            "name": "Quadratic (ill-conditioned)",
            "description": "f(x) = 0.5 * sum([1,10,100] * x^2)",
            "loss_fn": ill_loss,
            "grad_fn": ill_grad,
            "x_start": np.array([5.0, -3.0, 2.0], dtype=float),
            "target": np.zeros(3, dtype=float),
            "distance_label": "||x||",
            "steps": 120,
            "optimizers": build_optimizer_specs(base_lr=0.02),
        },
        {
            "name": "Rosenbrock (non-convex)",
            "description": "f(x,y) = (1-x)^2 + 100*(y-x^2)^2",
            "loss_fn": rosenbrock_loss,
            "grad_fn": rosenbrock_grad,
            "x_start": np.array([-1.2, 1.0], dtype=float),
            "target": np.array([1.0, 1.0], dtype=float),
            "distance_label": "||x - [1, 1]||",
            "steps": 200,
            "optimizers": build_optimizer_specs(base_lr=0.002),
        },
    ]

    scenario_results = []
    for scenario in scenarios:
        print("\n" + "=" * 70)
        print(f"Scenario: {scenario['name']}")
        print(f"{scenario['description']}")
        print(f"Start: {scenario['x_start']}  Steps: {scenario['steps']}")

        results = []
        for name, optimizer_class, kwargs in scenario["optimizers"]:
            data = run_optimizer(
                optimizer_class,
                kwargs,
                scenario["loss_fn"],
                scenario["grad_fn"],
                scenario["x_start"],
                scenario["steps"],
                target=scenario.get("target"),
            )
            results.append((name, data))
        scenario_results.append(results)

        print("\nLoss snapshots:")
        print_summary_table(results, scenario["steps"])

    print("\n💡 Key Insights:")
    print("- Well-conditioned quadratics highlight baseline convergence speed.")
    print("- Ill-conditioned quadratics show how optimizers handle scale differences.")
    print("- Rosenbrock exposes behavior on a curved, non-convex surface.")

    plot_scenarios(scenarios, scenario_results, output_path=output_path, show=show)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze optimizer convergence behavior with matplotlib plots."
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the plot (e.g. optimizer_analysis.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip opening an interactive window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analyze_optimizer_convergence_behavior(
        show=not args.no_show,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
