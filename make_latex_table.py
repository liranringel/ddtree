#!/usr/bin/env python3

import argparse
from itertools import groupby
from pathlib import Path
import sys

import numpy as np
import torch


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


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def load_run_data(path: Path) -> dict[str, object]:
    return torch.load(path, weights_only=False, map_location="cpu")


def mean_time_per_token(run_data: dict[str, object], method_key: str) -> float:
    return float(np.mean([response[method_key].time_per_output_token for response in run_data["responses"]]))


def mean_acceptance_length(run_data: dict[str, object], method_key: str) -> float:
    return float(np.mean([np.mean(response[method_key].acceptance_lengths) for response in run_data["responses"]]))


def best_run_data(sdpa_run_data: dict[str, object], flash_run_data: dict[str, object], method_key: str) -> dict[str, object]:
    if mean_time_per_token(sdpa_run_data, method_key) <= mean_time_per_token(flash_run_data, method_key):
        return sdpa_run_data
    return flash_run_data


def method_label(method_key: str) -> str:
    if method_key == "dflash":
        return "DFlash"
    if method_key.startswith("ddtree_tb"):
        return f"DFlash+DDTree ({method_key.removeprefix('ddtree_tb')})"
    return method_key


def short_model_name(model_name: str) -> str:
    return model_name.rsplit("/", maxsplit=1)[-1]


def display_dataset_name(dataset: str) -> str:
    return DATASET_DISPLAY_NAMES.get(dataset, dataset)


def pair_run_paths(runs_dir: Path) -> list[tuple[str, Path, Path]]:
    sdpa_paths = {}
    flash_paths = {}

    for path in sorted(runs_dir.glob("*.pt")):
        if path.name.endswith("__sdpa.pt"):
            sdpa_paths[path.name.removesuffix("__sdpa.pt")] = path
        elif path.name.endswith("__flash_attn.pt"):
            flash_paths[path.name.removesuffix("__flash_attn.pt")] = path

    pair_keys = sorted(set(sdpa_paths) | set(flash_paths))
    pairs = []
    for pair_key in pair_keys:
        sdpa_path = sdpa_paths.get(pair_key)
        flash_path = flash_paths.get(pair_key)
        if sdpa_path is None or flash_path is None:
            print(f"Skipping incomplete pair: {pair_key}", file=sys.stderr)
            continue
        pairs.append((pair_key, sdpa_path, flash_path))
    return pairs


def build_rows(runs_dir: Path) -> list[tuple[str, str, str, str, float, float]]:
    rows = []

    for _, sdpa_path, flash_path in pair_run_paths(runs_dir):
        sdpa_run_data = load_run_data(sdpa_path)
        flash_run_data = load_run_data(flash_path)

        if sdpa_run_data["target_attn_implementation"] != "sdpa":
            raise ValueError(f"{sdpa_path} does not look like an sdpa run")
        if flash_run_data["target_attn_implementation"] != "flash_attention_2":
            raise ValueError(f"{flash_path} does not look like a flash_attn run")

        best_baseline_run_data = best_run_data(sdpa_run_data, flash_run_data, "baseline")
        best_baseline_time_per_token = mean_time_per_token(best_baseline_run_data, "baseline")
        best_dflash_run_data = best_run_data(sdpa_run_data, flash_run_data, "dflash")

        args = sdpa_run_data["args"]
        dataset = args["dataset"]
        model_name = short_model_name(args["model_name_or_path"])
        temperature = args["temperature"]

        dflash_speedup = best_baseline_time_per_token / mean_time_per_token(best_dflash_run_data, "dflash")
        dflash_acceptance = mean_acceptance_length(best_dflash_run_data, "dflash")
        rows.append((dataset, model_name, str(temperature), "DFlash", dflash_speedup, dflash_acceptance))

        ddtree_method_keys = [method_key for method_key in sdpa_run_data["responses"][0] if method_key.startswith("ddtree_tb")]
        best_ddtree_method_key = max(
            ddtree_method_keys,
            key=lambda method_key: best_baseline_time_per_token / mean_time_per_token(sdpa_run_data, method_key),
        )
        best_ddtree_speedup = best_baseline_time_per_token / mean_time_per_token(sdpa_run_data, best_ddtree_method_key)
        best_ddtree_acceptance = mean_acceptance_length(sdpa_run_data, best_ddtree_method_key)
        rows.append(
            (
                dataset,
                model_name,
                str(temperature),
                "DFlash+DDTree",
                best_ddtree_speedup,
                best_ddtree_acceptance,
            )
        )

    return sorted(rows)


DATASET_ORDER = {
    "math500": 0, "gsm8k": 1, "aime24": 2, "aime25": 3,
    "humaneval": 4, "mbpp": 5, "livecodebench": 6, "swe-bench": 7,
    "mt-bench": 8, "alpaca": 9,
}


def make_latex_table(rows: list[tuple[str, str, str, str, float, float]]) -> str:
    datasets = sorted(set(r[0] for r in rows), key=lambda d: DATASET_ORDER.get(d, 100))
    models = sorted(set(r[1] for r in rows))
    temperatures = sorted(set(r[2] for r in rows))
    methods = sorted(set(r[3] for r in rows), key=lambda method: (method != "DFlash", method))

    # lookup: (temp, dataset, model, method) -> (speedup, acceptance)
    lookup: dict[tuple[str, str, str, str], tuple[float, float]] = {}
    for dataset, model, temp, method, speedup, acceptance in rows:
        lookup[(temp, dataset, model, method)] = (speedup, acceptance)

    total_cols = 1 + 2 * len(methods) * len(models)
    model_span = 2 * len(methods)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Speedup over autoregressive decoding and mean acceptance length ($\tau$).}",
        r"\label{tab:benchmark-results}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l " + " ".join(["".join(["rc"] * len(methods))] * len(models)) + "}",
        r"\toprule",
    ]

    model_headers = []
    model_cmidrules = []
    method_headers = [r"\textbf{Dataset}"]
    method_cmidrules = []
    metric_headers = [""]

    current_col = 2
    for model in models:
        model_headers.append(rf"\multicolumn{{{model_span}}}{{c}}{{\textbf{{{latex_escape(model)}}}}}")
        model_cmidrules.append(rf"\cmidrule(lr){{{current_col}-{current_col + model_span - 1}}}")
        for method in methods:
            method_headers.append(rf"\multicolumn{{2}}{{c}}{{{latex_escape(method)}}}")
            method_cmidrules.append(rf"\cmidrule(lr){{{current_col}-{current_col + 1}}}")
            metric_headers.extend([r"Speedup", r"$\tau$"])
            current_col += 2

    lines.append(r" & " + " & ".join(model_headers) + r" \\")
    lines.append(" ".join(model_cmidrules))
    lines.append(" & ".join(method_headers) + r" \\")
    lines.append(" ".join(method_cmidrules))
    lines.append(" & ".join(metric_headers) + r" \\")

    for temp in temperatures:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{total_cols}}}{{l}}{{\textit{{Temperature = {temp}}}}} \\")

        for dataset in datasets:
            cells = [latex_escape(display_dataset_name(dataset))]
            for model in models:
                available_values = [
                    lookup[(temp, dataset, model, method)]
                    for method in methods
                    if (temp, dataset, model, method) in lookup
                ]
                best_sp = max((sp for sp, _ in available_values), default=None)
                best_acc = max((acc for _, acc in available_values), default=None)
                highlight_best = len(available_values) > 1

                for method in methods:
                    value = lookup.get((temp, dataset, model, method))
                    if value is None:
                        cells.extend(["--", "--"])
                        continue

                    speedup, acceptance = value
                    speedup_str = f"{speedup:.2f}$\\times$"
                    acceptance_str = f"{acceptance:.2f}"
                    if highlight_best and speedup == best_sp:
                        speedup_str = r"\textbf{" + speedup_str + "}"
                    if highlight_best and acceptance == best_acc:
                        acceptance_str = r"\textbf{" + acceptance_str + "}"
                    cells.append(speedup_str)
                    cells.append(acceptance_str)

            lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",  # close \resizebox
        r"\end{table*}",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = build_rows(args.runs_dir)
    table = make_latex_table(rows)

    if args.output is None:
        print(table)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table + "\n")


if __name__ == "__main__":
    main()
