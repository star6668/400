import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from graph_lib import choose_p


def interpolate_value(group, p, column):
    group = group.sort_values("p")
    ps = group["p"].to_list()
    ys = group[column].to_list()

    if p <= ps[0]:
        return ys[0]
    if p >= ps[-1]:
        return ys[-1]

    for i in range(len(ps) - 1):
        p0, p1 = ps[i], ps[i + 1]
        if p0 <= p <= p1:
            ratio = (p - p0) / (p1 - p0)
            return ys[i] + ratio * (ys[i + 1] - ys[i])

    return ys[-1]


def choose_experimental_p(group, target, column):
    row_id = (group[column] - target).abs().argmin()
    return float(group.iloc[row_id]["p"])


def build_comparison(profiles, metric, targets_per_n):
    column = "w_mean" if metric == "clique" else "d_mean"
    rows = []

    for n, group in profiles.groupby("n"):
        group = group.sort_values("p")
        y_min = float(group[column].min())
        y_max = float(group[column].max())
        targets = [
            y_min + (y_max - y_min) * i / (targets_per_n + 1)
            for i in range(1, targets_per_n + 1)
        ]

        for target in targets:
            p_exp = choose_experimental_p(group, target, column)
            p_formula, _ = choose_p(target, metric, int(n))

            p_formula = min(max(p_formula, float(group["p"].min())), float(group["p"].max()))

            for method, p in [("experimentale", p_exp), ("formule", p_formula)]:
                obtained = interpolate_value(group, p, column)
                rows.append(
                    {
                        "metric": metric,
                        "n": int(n),
                        "target": target,
                        "method": method,
                        "p": p,
                        "obtained": obtained,
                        "abs_error": abs(obtained - target),
                        "rel_error_pct": 100 * abs(obtained - target) / target,
                    }
                )

    return pd.DataFrame(rows)


def print_summary(comparison):
    summary = (
        comparison.groupby(["metric", "method"])[["abs_error", "rel_error_pct"]]
        .mean()
        .round(3)
    )
    print(summary)


def plot_target_vs_obtained(comparison, metric, output_path):
    data = comparison[comparison["metric"] == metric]
    title = "Clique maximale" if metric == "clique" else "K-degenerescence"

    fig, ax = plt.subplots(figsize=(8, 6))

    for method, color in [("experimentale", "tab:blue"), ("formule", "tab:red")]:
        subset = data[data["method"] == method]
        ax.scatter(
            subset["target"],
            subset["obtained"],
            s=28,
            alpha=0.75,
            label=method,
            color=color,
        )

    low = min(data["target"].min(), data["obtained"].min())
    high = max(data["target"].max(), data["obtained"].max())
    ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.2, label="ideal")

    ax.set_xlabel("Valeur cible")
    ax.set_ylabel("Valeur obtenue")
    ax.set_title(f"{title} : cible vs valeur obtenue")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--summary",
    type=Path,
    default=Path(__file__).with_name("simple_p_profiles_summary.csv"),
)
parser.add_argument("--targets-per-n", type=int, default=12)
parser.add_argument("--output-dir", type=Path, default=Path("."))
args = parser.parse_args()

profiles = pd.read_csv(args.summary)
profiles = profiles[(profiles["p"] >= 0.01) & (profiles["p"] <= 0.9)].copy()

comparison = pd.concat(
    [
        build_comparison(profiles, "clique", args.targets_per_n),
        build_comparison(profiles, "degeneracy", args.targets_per_n),
    ],
    ignore_index=True,
)

print_summary(comparison)

args.output_dir.mkdir(parents=True, exist_ok=True)
clique_output = args.output_dir / "clique_form_exp.png"
degen_output = args.output_dir / "degen_form_exp.png"

plot_target_vs_obtained(comparison, "clique", clique_output)
plot_target_vs_obtained(comparison, "degeneracy", degen_output)

print(f"\nGraphiques sauvegardes :")
print(f"- {clique_output}")
print(f"- {degen_output}")


