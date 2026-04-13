#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch


def load_run_data(path: Path) -> dict:
    return torch.load(path, weights_only=False, map_location="cpu")


def mean_time_per_token(run_data: dict, method_key: str) -> float:
    return float(np.mean([r[method_key].time_per_output_token for r in run_data["responses"]]))


def mean_acceptance_length(run_data: dict, method_key: str) -> float:
    return float(np.mean([np.mean(r[method_key].acceptance_lengths) for r in run_data["responses"]]))


def flatten_acceptance_lengths(run_data: dict, method_key: str) -> list[int]:
    return [length for response in run_data["responses"] for length in response[method_key].acceptance_lengths]


def short_model_name(model_name: str) -> str:
    return model_name.rsplit("/", maxsplit=1)[-1]


def best_run_data(sdpa, flash, method_key):
    if mean_time_per_token(sdpa, method_key) <= mean_time_per_token(flash, method_key):
        return sdpa
    return flash


def pair_run_paths(runs_dir: Path):
    sdpa_paths = {}
    flash_paths = {}
    for path in sorted(runs_dir.glob("*.pt")):
        if path.name.endswith("__sdpa.pt"):
            sdpa_paths[path.name.removesuffix("__sdpa.pt")] = path
        elif path.name.endswith("__flash_attn.pt"):
            flash_paths[path.name.removesuffix("__flash_attn.pt")] = path
    pairs = []
    for key in sorted(set(sdpa_paths) & set(flash_paths)):
        pairs.append((key, sdpa_paths[key], flash_paths[key]))
    return pairs


def find_run_pair(runs_dir: Path, dataset: str, model: str, temperature: float) -> tuple[dict, dict]:
    for _, sdpa_path, flash_path in pair_run_paths(runs_dir):
        sdpa = load_run_data(sdpa_path)
        args = sdpa["args"]
        if (
            args["dataset"] == dataset
            and short_model_name(args["model_name_or_path"]) == model
            and np.isclose(args["temperature"], temperature)
        ):
            return sdpa, load_run_data(flash_path)
    raise ValueError(
        f"No paired run found for dataset={dataset!r}, model={model!r}, temperature={temperature!r}."
    )


def collect_plot_data(runs_dir: Path) -> list[dict]:
    results = []
    for _, sdpa_path, flash_path in pair_run_paths(runs_dir):
        sdpa = load_run_data(sdpa_path)
        flash = load_run_data(flash_path)

        best_baseline = best_run_data(sdpa, flash, "baseline")
        baseline_tpt = mean_time_per_token(best_baseline, "baseline")

        best_dflash = best_run_data(sdpa, flash, "dflash")
        dflash_speedup = baseline_tpt / mean_time_per_token(best_dflash, "dflash")
        dflash_acceptance = mean_acceptance_length(best_dflash, "dflash")

        # All DDTree tree budgets (use sdpa, consistent with make_latex_table.py)
        ddtree_keys = sorted(
            [k for k in sdpa["responses"][0] if k.startswith("ddtree_tb")],
            key=lambda k: int(k.removeprefix("ddtree_tb")),
        )
        ddtree_speedups = {}
        ddtree_acceptances = {}
        for k in ddtree_keys:
            tb = int(k.removeprefix("ddtree_tb"))
            ddtree_speedups[tb] = baseline_tpt / mean_time_per_token(sdpa, k)
            ddtree_acceptances[tb] = mean_acceptance_length(sdpa, k)

        args = sdpa["args"]
        results.append({
            "dataset": args["dataset"],
            "model": short_model_name(args["model_name_or_path"]),
            "temperature": args["temperature"],
            "dflash_speedup": dflash_speedup,
            "dflash_acceptance": dflash_acceptance,
            "ddtree_speedups": ddtree_speedups,
            "ddtree_acceptances": ddtree_acceptances,
        })
    return results


DATASET_DISPLAY_NAMES = {
    "aime24": "AIME 2024",
    "aime25": "AIME 2025",
    "alpaca": "Alpaca",
    "gsm8k": "GSM8K",
    "humaneval": "HumanEval",
    "livecodebench": "LiveCodeBench",
    "math500": "MATH-500",
    "mbpp": "MBPP",
    "mt-bench": "MT-Bench",
    "swe-bench": "SWE-bench Lite",
}

MODEL_DISPLAY_NAMES = {
    "Qwen3-4B": "Qwen3-4B",
    "Qwen3-8B": "Qwen3-8B",
    "Qwen3-Coder-30B-A3B-Instruct": "Qwen3-30B-MoE",
}

# Grayscale for DFlash and greens for DDTree, with distinct luminance per model
MODEL_COLORS_DFLASH = {
    "Qwen3-4B":                      "#D9D9D9",  # light gray
    "Qwen3-8B":                      "#969696",  # medium gray
    "Qwen3-Coder-30B-A3B-Instruct":  "#525252",  # dark gray
}
MODEL_COLORS_DDTREE = {
    "Qwen3-4B":                      "#A1D99B",  # light green
    "Qwen3-8B":                      "#66BD63",  # medium green
    "Qwen3-Coder-30B-A3B-Instruct":  "#238443",  # dark green
}


def _setup_latex_rcparams() -> None:
    """Configure matplotlib for LaTeX-quality text, with graceful fallback."""
    try:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{lmodern}",
        })
        # Smoke-test: render a tiny figure to verify LaTeX works
        fig_test = plt.figure(figsize=(0.1, 0.1))
        fig_test.text(0.5, 0.5, r"$x$")
        fig_test.savefig("/dev/null", format="png")
        plt.close(fig_test)
    except Exception:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "serif"],
            "mathtext.fontset": "cm",
        })


def _fmt_speedup(val: float, use_tex: bool) -> str:
    if use_tex:
        return rf"${val:.1f}\times$"
    return f"{val:.1f}\u00d7"


def _safe_stem_token(value: str) -> str:
    return (
        value.lower()
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace(" ", "_")
    )


def plot_case_study(
    results: list[dict],
    output: Path,
    dataset: str,
    model: str,
    temperature: float,
) -> None:
    """Single-panel case-study figure with dual y-axes."""
    filtered = [
        r
        for r in results
        if r["dataset"] == dataset
        and r["model"] == model
        and np.isclose(r["temperature"], temperature)
    ]
    if not filtered:
        print(
            f"No data found for dataset={dataset!r}, model={model!r}, temperature={temperature!r}.",
            file=sys.stderr,
        )
        return
    if len(filtered) > 1:
        raise ValueError(
            f"Expected one matching run for dataset={dataset!r}, model={model!r}, temperature={temperature!r}, "
            f"but found {len(filtered)}."
        )
    r = filtered[0]

    _setup_latex_rcparams()
    use_tex = plt.rcParams.get("text.usetex", False)
    plt.rcParams.update({
        "axes.labelsize": 19,
        "axes.titlesize": 19,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 18.5,
    })

    tree_budgets = sorted(r["ddtree_speedups"])
    speedups = [r["ddtree_speedups"][tb] for tb in tree_budgets]
    acceptances = [r["ddtree_acceptances"][tb] for tb in tree_budgets]

    speed_color = "#0072B2"
    accept_color = "#D55E00"
    baseline_color = "#6F6F6F"

    fig, ax_speed = plt.subplots(figsize=(8.2, 6.2))
    ax_accept = ax_speed.twinx()

    speed_curve = ax_speed.plot(
        tree_budgets,
        speedups,
        color=speed_color,
        marker="o",
        markersize=5.0,
        linewidth=3.2,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=4,
        label="DDTree Speedup",
    )[0]
    speed_base = ax_speed.axhline(
        r["dflash_speedup"],
        color=speed_color,
        linestyle=(0, (4, 2)),
        linewidth=2.4,
        alpha=0.95,
        zorder=2,
        label="DFlash Speedup",
    )

    accept_curve = ax_accept.plot(
        tree_budgets,
        acceptances,
        color=accept_color,
        marker="s",
        markersize=4.8,
        linewidth=3.2,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=5,
        label="DDTree Acc. Length",
    )[0]
    accept_base = ax_accept.axhline(
        r["dflash_acceptance"],
        color=accept_color,
        linestyle=(0, (4, 2)),
        linewidth=2.4,
        alpha=0.95,
        zorder=3,
        label="DFlash Acc. Length",
    )

    tau_label = r"Acceptance Length ($\tau$)" if use_tex else "Acceptance Length (\u03c4)"
    ax_speed.set_xscale("log", base=2)
    ax_speed.set_xticks(tree_budgets, [str(tb) for tb in tree_budgets])
    ax_speed.set_xlabel("Node Budget")
    ax_speed.set_ylabel("Speedup", color=speed_color)
    ax_accept.set_ylabel(tau_label, color=accept_color)

    ax_speed.grid(axis="y", color="#E6E6E6", linewidth=0.6, zorder=0)
    ax_speed.set_axisbelow(True)
    ax_speed.axhline(y=1.0, color=baseline_color, linestyle=":", linewidth=0.9, alpha=0.8, zorder=1)
    ax_speed.set_xlim(tree_budgets[0] * 0.88, tree_budgets[-1] * 1.14)
    speed_bottom = 4.0
    accept_bottom = 4.0
    speed_top = max(speedups) * 1.12
    accept_top = max(acceptances) * 1.08
    common_top = max(speed_top, accept_top)
    ax_speed.set_ylim(speed_bottom, common_top)
    ax_accept.set_ylim(accept_bottom, common_top)
    ax_speed.set_yticks(np.arange(speed_bottom, common_top + 1e-6, 1.0))
    ax_accept.set_yticks(np.arange(accept_bottom, common_top + 1e-6, 1.0))
    ax_speed.tick_params(axis="y", colors=speed_color)
    ax_accept.tick_params(axis="y", colors=accept_color)
    ax_speed.tick_params(axis="x", colors="#555555")

    ax_speed.spines["top"].set_visible(False)
    ax_accept.spines["top"].set_visible(False)
    ax_speed.spines["left"].set_color(speed_color)
    ax_accept.spines["right"].set_color(accept_color)
    ax_speed.spines["bottom"].set_color("#CCCCCC")

    legend_handles = [speed_curve, speed_base, accept_curve, accept_base]
    legend_labels = [handle.get_label() for handle in legend_handles]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        frameon=False,
        ncol=2,
        columnspacing=1.6,
        handlelength=2.8,
    )

    fig.subplots_adjust(left=0.14, right=0.86, bottom=0.18, top=0.78)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    print(f"Saved case-study plot to {output}")


def plot_acceptance_distribution(
    runs_dir: Path,
    output: Path,
    dataset: str,
    model: str,
    temperature: float,
) -> None:
    """Acceptance-length histogram for DFlash vs best-speedup DDTree budget."""
    sdpa, flash = find_run_pair(runs_dir, dataset, model, temperature)
    best_baseline = best_run_data(sdpa, flash, "baseline")
    baseline_tpt = mean_time_per_token(best_baseline, "baseline")
    best_dflash = best_run_data(sdpa, flash, "dflash")
    ddtree_keys = sorted(
        [k for k in sdpa["responses"][0] if k.startswith("ddtree_tb")],
        key=lambda k: int(k.removeprefix("ddtree_tb")),
    )
    best_ddtree_key = max(
        ddtree_keys,
        key=lambda key: baseline_tpt / mean_time_per_token(sdpa, key),
    )
    best_budget = int(best_ddtree_key.removeprefix("ddtree_tb"))

    _setup_latex_rcparams()
    plt.rcParams.update({
        "axes.labelsize": 19,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 18.5,
    })

    dflash_lengths = flatten_acceptance_lengths(best_dflash, "dflash")
    ddtree_lengths = flatten_acceptance_lengths(sdpa, best_ddtree_key)

    max_accept_len = max(max(dflash_lengths), max(ddtree_lengths))
    x_values = np.arange(1, max_accept_len + 1)
    dflash_counts = np.bincount(dflash_lengths, minlength=max_accept_len + 1)[1:max_accept_len + 1]
    ddtree_counts = np.bincount(ddtree_lengths, minlength=max_accept_len + 1)[1:max_accept_len + 1]
    dflash_dist = dflash_counts / dflash_counts.sum()
    ddtree_dist = ddtree_counts / ddtree_counts.sum()

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    width = 0.38
    ax.bar(
        x_values - width / 2,
        dflash_dist,
        width=width,
        color="#4E79A7",
        edgecolor="white",
        linewidth=0.7,
        label="DFlash",
        zorder=3,
    )
    ax.bar(
        x_values + width / 2,
        ddtree_dist,
        width=width,
        color="#59A14F",
        edgecolor="white",
        linewidth=0.7,
        label=f"DFlash+DDTree (B={best_budget})",
        zorder=3,
    )

    ax.set_xlabel("Acceptance Length")
    ax.set_ylabel("Fraction of Decoding Rounds")
    ax.set_xticks(x_values)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{100 * val:.0f}%"))
    ax.set_xlim(0.4, max_accept_len + 0.6)
    ax.set_ylim(0.0, max(dflash_dist.max(), ddtree_dist.max()) * 1.12)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6, zorder=0)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.98), frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#555555")

    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.18, top=0.78)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    print(f"Saved acceptance distribution plot to {output}")


def plot_bar_speeds(results: list[dict], output: Path) -> None:
    """Publication-quality vertical bar chart of T=0.0 speedups across all datasets and models."""
    if not results:
        print("No data to plot.", file=sys.stderr)
        return

    results = [r for r in results if np.isclose(r["temperature"], 0.0)]
    if not results:
        print("No T=0.0 data to plot.", file=sys.stderr)
        return

    _setup_latex_rcparams()
    use_tex = plt.rcParams.get("text.usetex", False)

    plt.rcParams.update({
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8.5,
    })

    # Aggregate: for each (dataset, model) at T=0.0, take the best DDTree budget.
    agg: dict[tuple[str, str], dict[str, float]] = {}
    for r in results:
        key = (r["dataset"], r["model"])
        if key not in agg:
            agg[key] = {"dflash": 0.0, "ddtree": 0.0}
        agg[key]["dflash"] = max(agg[key]["dflash"], r["dflash_speedup"])
        best_ddtree = max(r["ddtree_speedups"].values()) if r["ddtree_speedups"] else 0.0
        agg[key]["ddtree"] = max(agg[key]["ddtree"], best_ddtree)

    models = sorted(set(k[1] for k in agg))
    n_models = len(models)

    # Group datasets by domain, sort by speedup within each group
    DATASET_ORDER = {
        # Math/Reasoning
        "math500": 0, "gsm8k": 1, "aime24": 2, "aime25": 3,
        # Code
        "humaneval": 4, "mbpp": 5, "livecodebench": 6, "swe-bench": 7,
        # General
        "mt-bench": 8, "alpaca": 9,
    }
    all_datasets = set(k[0] for k in agg)
    datasets = sorted(all_datasets, key=lambda d: DATASET_ORDER.get(d, 100))
    n_datasets = len(datasets)

    # Bar geometry: all DFlash bars on the left, all DDTree bars on the right,
    # with a small gap between the two halves.
    # Layout per group: [DF_0, DF_1, DF_2, <gap>, DDT_0, DDT_1, DDT_2]
    gap_units = 0.5          # gap expressed as multiples of bar_width
    total_units = n_models * 2 + gap_units
    bar_width = 0.78 / total_units
    gap = gap_units * bar_width
    x = np.arange(n_datasets)

    # Pre-collect vals for value-label pass
    all_ddtree: dict[tuple[int, int], float] = {}  # (model_idx, dataset_idx) -> val

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Two separate passes so legend order is: all DFlash entries, then all DDTree entries
    for i, model in enumerate(models):
        dflash_vals = []
        for dataset in datasets:
            dflash_vals.append(agg.get((dataset, model), {"dflash": 0.0})["dflash"])
        color = MODEL_COLORS_DFLASH.get(model, "#AAAAAA")
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        # DFlash slot i → left half
        df_offset = (i - (n_models - 1) / 2 - (n_models + gap_units) / 2 + 0.5) * bar_width
        ax.bar(
            x + df_offset, dflash_vals, bar_width,
            label=f"{display_name} DFlash",
            color=color, edgecolor=color, linewidth=0.4, zorder=3,
        )

    for i, model in enumerate(models):
        ddtree_vals = []
        for j, dataset in enumerate(datasets):
            val = agg.get((dataset, model), {"ddtree": 0.0})["ddtree"]
            ddtree_vals.append(val)
            all_ddtree[(i, j)] = val
        color = MODEL_COLORS_DDTREE.get(model, "#555555")
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        # DDTree slot i → right half (shifted by n_models + gap)
        ddt_offset = (i - (n_models - 1) / 2 + (n_models + gap_units) / 2 + 0.5) * bar_width
        ax.bar(
            x + ddt_offset, ddtree_vals, bar_width,
            label=f"{display_name} DFlash+DDTree",
            color=color, edgecolor=color, linewidth=0.4, zorder=3,
        )

    # Handles from loop: [4B-DF, 8B-DF, 30B-DF, 4B-DDT, 8B-DDT, 30B-DDT]
    # ncol=3 fills column-first, so interleave to get Row1=all-DF, Row2=all-DDT:
    # → [4B-DF, 4B-DDT, 8B-DF, 8B-DDT, 30B-DF, 30B-DDT]
    handles, labels = ax.get_legend_handles_labels()
    interleaved_h, interleaved_l = [], []
    for i in range(n_models):
        interleaved_h += [handles[i], handles[n_models + i]]
        interleaved_l += [labels[i], labels[n_models + i]]
    ax.legend(
        interleaved_h, interleaved_l,
        fontsize=7.4, ncol=3,
        loc="lower right", bbox_to_anchor=(1.0, 1.0),
        frameon=True, framealpha=0.95, edgecolor="#DDDDDD",
        handletextpad=0.3, columnspacing=0.6, labelspacing=0.3,
    )

    # Value labels: only the best DFlash+DDTree per dataset
    for j, dataset in enumerate(datasets):
        best_val = 0.0
        best_model_idx = 0
        for i, model in enumerate(models):
            val = all_ddtree.get((i, j), 0.0)
            if val > best_val:
                best_val = val
                best_model_idx = i
        ddt_offset = (best_model_idx - (n_models - 1) / 2 + (n_models + gap_units) / 2 + 0.5) * bar_width
        bar_x = x[j] + ddt_offset
        ax.text(
            bar_x, best_val + 0.08,
            _fmt_speedup(best_val, use_tex),
            ha="center", va="bottom", fontsize=5.5, fontweight="bold",
            color="#444444",
        )

    # Baseline reference line
    ax.axhline(y=1.0, color="#999999", linestyle="--", linewidth=0.8, alpha=0.5, zorder=2)

    # Grid
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", color="#E8E8E8", linewidth=0.4, zorder=0)

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(
        [DATASET_DISPLAY_NAMES.get(d, d) for d in datasets],
        rotation=35, ha="right", fontsize=8,
    )

    # Y-axis
    ax.set_ylabel("Speedup Relative to Autoregressive Decoding", fontsize=10)
    ax.set_ylim(bottom=0)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#555555")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=300)
    print(f"Saved bar plot to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--case-study",
        action="store_true",
        help="Generate a two-panel figure of speedup and acceptance length vs node budget",
    )
    parser.add_argument(
        "--acceptance-distribution",
        action="store_true",
        help="Generate an acceptance-length histogram for DFlash vs the best DDTree budget",
    )
    parser.add_argument("--dataset", default="math500",
                        help="Dataset to plot when --case-study is set")
    parser.add_argument("--model", default="Qwen3-8B",
                        help="Model to plot when --case-study is set")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Temperature to plot when --case-study is set")
    parser.add_argument(
        "--bar",
        action="store_true",
        help="Generate a single grouped bar chart summarizing all datasets and models",
    )
    args = parser.parse_args()

    results = collect_plot_data(args.runs_dir)
    selected_modes = []
    if args.case_study:
        selected_modes.append("case_study")
    if args.acceptance_distribution:
        selected_modes.append("acceptance_distribution")
    if args.bar:
        selected_modes.append("bar")
    if not selected_modes:
        selected_modes = ["case_study", "acceptance_distribution", "bar"]

    if args.output is not None and len(selected_modes) != 1:
        parser.error("--output can only be used when exactly one plot mode is selected.")

    for mode in selected_modes:
        if mode == "case_study":
            output = args.output or Path(
                f"paper/plots/{_safe_stem_token(args.dataset)}_{_safe_stem_token(args.model)}_budget_tradeoff.pdf"
            )
            plot_case_study(
                results,
                output,
                dataset=args.dataset,
                model=args.model,
                temperature=args.temperature,
            )
        elif mode == "acceptance_distribution":
            output = args.output or Path(
                f"paper/plots/{_safe_stem_token(args.dataset)}_{_safe_stem_token(args.model)}_acceptance_histogram.pdf"
            )
            plot_acceptance_distribution(
                args.runs_dir,
                output,
                dataset=args.dataset,
                model=args.model,
                temperature=args.temperature,
            )
        elif mode == "bar":
            output = args.output or Path("paper/plots/speedup_bar.pdf")
            plot_bar_speeds(results, output)


if __name__ == "__main__":
    main()
