import argparse
import json
import os

from tabulate import tabulate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile results for a specific benchmark mode."
    )
    parser.add_argument(
        "benchmark_mode",
        type=str,
        help="Benchmark mode to filter results (e.g., lighting-ll).",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="work-dir",
        help="Directory where results are stored. Default: work_dirs",
    )
    parser.add_argument(
        "--metrics",
        type=str.upper,
        choices=["AP", "AR", "ALL"],
        default="AP",
        help="Metrics to include in the table (AP, AR, ALL). Default: AP",
    )
    return parser.parse_args()


def find_results_dirs(results_dir, model_name, benchmark_mode, model_resolution):
    all_results = [
        f
        for f in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, f))
    ]
    return [
        r
        for r in all_results
        if r.startswith(f"{model_name}_poseadapt-{benchmark_mode}_")
        and r.endswith(f"_{model_resolution}")
    ]


def load_latest_results(result_path):
    experiment_dirs = [
        d
        for d in sorted(os.listdir(result_path), reverse=True)
        if os.path.isdir(os.path.join(result_path, d)) and d != "experiences"
    ]
    if not experiment_dirs:
        return None

    latest_dir = os.path.join(result_path, experiment_dirs[0])
    results_file = os.path.join(latest_dir, f"{experiment_dirs[0]}.json")
    if not os.path.exists(results_file):
        return None

    with open(results_file, "r") as f:
        return json.load(f)


def filter_metrics(results, metric_type):
    # Keep only selected metric keys
    return {
        k: v
        for k, v in results.items()
        if (metric_type == "ALL" and (k.endswith("/AP", "/AR")))
        or (metric_type == "AP" and k.endswith("/AP"))
        or (metric_type == "AR" and k.endswith("/AR"))
    }


def main():
    args = parse_args()

    # Config
    model_name = "rtmpose-t_8xb32-10e"
    model_resolution = "256x192"
    results_dir = args.work_dir

    # Gather results
    results_list = find_results_dirs(
        results_dir, model_name, args.benchmark_mode, model_resolution
    )
    summary = {}

    for result in sorted(results_list):
        results = load_latest_results(os.path.join(results_dir, result))
        if not results:
            continue

        results = filter_metrics(results, args.metrics)

        # Strategy extraction
        cl_strategy = (
            result.split(f"{model_name}_poseadapt-{args.benchmark_mode}")[-1]
            .rsplit(f"_{model_resolution}", 1)[0]
            .lstrip("_")
        ) or "ft"

        # Avoid name collisions
        i = 0
        base_strategy = cl_strategy
        while cl_strategy in summary:
            i += 1
            cl_strategy = f"{base_strategy}_{i}"

        summary[cl_strategy] = results

    # Compute means
    for cl_strategy, results in summary.items():
        ap_vals = [v for k, v in results.items() if k.endswith("/AP")]
        ar_vals = [v for k, v in results.items() if k.endswith("/AR")]
        if ap_vals and args.metrics in ("AP", "ALL"):
            results["mean/AP"] = sum(ap_vals) / len(ap_vals)
        if ar_vals and args.metrics in ("AR", "ALL"):
            results["mean/AR"] = sum(ar_vals) / len(ar_vals)

    # Prepare rows
    rows = []
    all_metrics = set()
    for cl_strategy, results in summary.items():
        row = {"Strategy": cl_strategy.upper()}
        row.update(results)
        all_metrics.update(results.keys())
        rows.append(row)

    # Sort rows
    def sort_key(row):
        if row["Strategy"] == "PT":
            return (0, row["Strategy"])
        if row["Strategy"] == "FT":
            return (1, row["Strategy"])
        return (2, row["Strategy"])

    rows.sort(key=sort_key)

    # Sort metrics so means come last
    metrics = sorted(all_metrics, key=lambda x: (x.startswith("mean/"), x))
    headers = ["Strategy"] + metrics

    table_data = []
    for row in rows:
        table_data.append(
            [
                row["Strategy"],
                *[f"{row.get(m, float('nan')) * 100:0.2f}" for m in metrics],
            ]
        )

        # Insert a separator after FT row
        if row["Strategy"] == "FT":
            table_data.append(["-" * len(h) for h in headers])

    # Print results
    print(
        tabulate(
            table_data,
            headers=headers,
            tablefmt="latex_booktabs",
            stralign="center",
            numalign="center",
        )
    )
    print(f"\nBenchmark: {args.benchmark_mode} ({model_resolution})")


if __name__ == "__main__":
    main()
