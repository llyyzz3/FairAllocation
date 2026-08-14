import json
import os
import re
import shutil
import sys
import time

import numpy as np

from algorithms.ef2x_four_agents import EF2XFourAgents
from data.generator import (
    FixedGenerator,
    IdenticalGenerator,
    MallowsGenerator,
    NormalGenerator,
    OrderedGenerator,
    UniformGenerator,
)
from evaluation.metrics import (
    EF1Checker,
    EFXChecker,
    MaxAlphaEFkX,
)


RESULTS_DIR = "results"
CONSOLE_LOG_DIR = os.path.join(RESULTS_DIR, "console_logs")
RESULT_FILE_DIR = os.path.join(RESULTS_DIR, "result_files")


class TeeOutput:
    """Write console output to both terminal and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


class ConsoleLogCapture:
    def __init__(self, log_path):
        self.log_path = log_path
        self.log_file = None
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        sys.stdout = TeeOutput(self.original_stdout, self.log_file)
        sys.stderr = TeeOutput(self.original_stderr, self.log_file)
        return self.log_path

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.log_file.close()


class EF2XFourAgentsBenchmarkAdapter:
    """Small benchmark adapter around the project EF2X four-agent implementation."""

    def __init__(self):
        self.name = "EF2X Four Agents"
        self.last_result = None

    def run(self, instance):
        if instance.n != 4:
            raise ValueError("EF2X Four Agents only supports n=4.")

        algorithm = EF2XFourAgents(instance.valuations)
        self.last_result = algorithm.solve()
        return {
            agent: sorted(self.last_result.allocation[agent])
            for agent in range(4)
        }

    def get_metadata(self):
        return EF2XFourAgents([[0.0], [0.0], [0.0], [0.0]]).get_metadata()


def prepare_result_dirs():
    os.makedirs(CONSOLE_LOG_DIR, exist_ok=True)
    os.makedirs(RESULT_FILE_DIR, exist_ok=True)


def migrate_existing_result_files():
    if not os.path.isdir(RESULTS_DIR):
        return

    for filename in os.listdir(RESULTS_DIR):
        source_path = os.path.join(RESULTS_DIR, filename)
        if not os.path.isfile(source_path) or not filename.endswith(".json"):
            continue

        target_path = os.path.join(RESULT_FILE_DIR, filename)
        if os.path.exists(target_path):
            name, ext = os.path.splitext(filename)
            target_path = os.path.join(RESULT_FILE_DIR, f"{name}_{time.time_ns()}{ext}")

        shutil.move(source_path, target_path)


def make_console_log_path():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(CONSOLE_LOG_DIR, f"benchmark_4agents_console_{timestamp}.txt")


def named(generator, name):
    generator.name = name
    return generator


def safe_filename_part(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9_.=-]+", "_", value)
    return value.strip("_") or "unknown"


def to_jsonable(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def save_experiment_result(instance, allocation, metrics, algo_name, scenario):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = (
        f"{safe_filename_part(scenario['name'])}_"
        f"{safe_filename_part(instance.dist_type)}_"
        f"{safe_filename_part(algo_name)}_"
        f"n{instance.n}_m{instance.m}_{time.time_ns()}.json"
    )
    filepath = os.path.join(RESULT_FILE_DIR, filename)

    report = {
        "metadata": {
            "algo_name": algo_name,
            "dist_type": instance.dist_type,
            "n": instance.n,
            "m": instance.m,
            "timestamp": timestamp,
            "scenario": scenario,
        },
        "valuations": instance.valuations.tolist(),
        "allocation": to_jsonable(allocation),
        "metrics": to_jsonable(metrics),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    return filepath


def format_metric_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return str(value)


def evaluate_metrics(instance, allocation, algo, metrics_to_check, runtime):
    results_metrics = {"runtime": runtime}
    metric_parts = []

    for metric in metrics_to_check:
        value = metric.compute(instance, allocation)
        results_metrics[metric.name] = value
        metric_parts.append(f"{metric.name}: {format_metric_value(value):>7}")

    if algo.last_result is not None:
        results_metrics["internal_ef2x_all_goods"] = True
        results_metrics["internal_checker_definition"] = "all_goods"
        results_metrics["algorithm_metadata"] = algo.get_metadata()
        results_metrics["iterations"] = algo.last_result.iterations
        results_metrics["final_stage"] = algo.last_result.final_stage.value
        results_metrics["stage_sequence"] = [
            info.stage.value for info in algo.last_result.stage_trace
        ]
        metric_parts.extend(
            [
                "InternalEF2X(all-goods):    True",
                f"Iterations: {algo.last_result.iterations:>7}",
            ]
        )

    return results_metrics, " | ".join(metric_parts)


def run_case(n, m, env, algo, metrics_to_check, scenario):
    print("=" * 90)
    print(f"{scenario['name']} | {n} agents, {m} items")
    print("=" * 90)
    print(f"\n>>> environment: {env.name}")
    print(f"    [running algorithm]: {algo.name:34}", end="")

    instance = env.generate(n, m)
    start_time = time.time()

    try:
        allocation = algo.run(instance)
        runtime = time.time() - start_time
        results_metrics, metric_str = evaluate_metrics(
            instance=instance,
            allocation=allocation,
            algo=algo,
            metrics_to_check=metrics_to_check,
            runtime=runtime,
        )
        save_experiment_result(instance, allocation, results_metrics, algo.name, scenario)
        print(f" | {metric_str} | Time: {runtime:.4f}s")
    except Exception as exc:
        runtime = time.time() - start_time
        failure_metrics = {
            "runtime": runtime,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "algorithm_metadata": algo.get_metadata(),
        }
        save_experiment_result(instance, {}, failure_metrics, algo.name, scenario)
        print(f" | FAILED: {type(exc).__name__}: {str(exc)[:240]} | Time: {runtime:.4f}s")


def build_fixed_cases():
    return [
        (
            15,
            FixedGenerator(
                valuations=[
                    [6, 0, 20, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 18, 0, 20, 0, 0, 2, 9, 0, 7, 1, 0, 0, 19],
                    [0, 0, 0, 0, 0, 0, 14, 11, 0, 8, 12, 0, 9, 10, 0],
                    [0, 0, 0, 3, 20, 18, 0, 5, 0, 0, 14, 0, 0, 0, 6],
                ],
                name="Fixed-SparseZeroHeavyRegression-m15",
            ),
        ),
        (
            16,
            FixedGenerator(
                valuations=[
                    [10, 10, 10, 10, 5, 5, 5, 5, 1, 1, 1, 1, 0, 0, 0, 0],
                    [10, 10, 10, 10, 5, 5, 5, 5, 1, 1, 1, 1, 0, 0, 0, 0],
                    [10, 10, 10, 10, 5, 5, 5, 5, 1, 1, 1, 1, 0, 0, 0, 0],
                    [10, 10, 10, 10, 5, 5, 5, 5, 1, 1, 1, 1, 0, 0, 0, 0],
                ],
                name="Fixed-IdenticalIntegerTie-m16",
            ),
        ),
        (
            8,
            FixedGenerator(
                valuations=[
                    [20, 19, 18, 1, 1, 1, 1, 1],
                    [20, 19, 1, 18, 1, 1, 1, 1],
                    [20, 1, 19, 1, 18, 1, 1, 1],
                    [20, 1, 1, 19, 1, 18, 1, 1],
                ],
                name="Fixed-ConflictingTopGoods-m8",
            ),
        ),
        (
            50,
            FixedGenerator(
                valuations=[
                    [(17 * good + 3 * agent + 11) % 21 for good in range(50)]
                    for agent in range(4)
                ],
                name="Fixed-LargeDeterministic-m50",
            ),
        ),
    ]


def build_random_environments():
    return [
        named(UniformGenerator(v_min=0, v_max=2000), "Uniform(0,2000)-4Agents"),
        named(UniformGenerator(v_min=1, v_max=50), "Uniform(1,50)-4Agents"),
        named(NormalGenerator(mean=800, std=350), "Normal(mean=800,std=350)-4Agents"),
        named(NormalGenerator(mean=300, std=500), "Normal(mean=300,std=500)-4Agents"),
        MallowsGenerator(phi=0.0),
        MallowsGenerator(phi=0.25),
        MallowsGenerator(phi=0.75),
        MallowsGenerator(phi=1.0),
        named(IdenticalGenerator(v_min=0, v_max=1200), "Identical(0,1200)-4Agents"),
        named(OrderedGenerator(v_min=0, v_max=1200), "Ordered(0,1200)-4Agents"),
    ]


def build_metrics():
    return [
        EF1Checker(),
        EFXChecker(),
        MaxAlphaEFkX(k=1),
    ]


def run_4agents_benchmark(repeats=1, base_seed=20260714):
    algo = EF2XFourAgentsBenchmarkAdapter()
    metrics_to_check = build_metrics()
    fixed_cases = build_fixed_cases()
    random_envs = build_random_environments()
    random_ms = [8, 12, 16, 24, 32, 50]

    total_cases = len(fixed_cases) + repeats * len(random_ms) * len(random_envs)
    case_idx = 0

    print("EF2X Four Agents benchmark")
    print("All generated instances use n=4")
    print(f"Fixed cases: {len(fixed_cases)}")
    print(f"Random sizes: {random_ms}")
    print(f"Random environments: {len(random_envs)}")

    for m, env in fixed_cases:
        case_idx += 1
        seed = base_seed + case_idx
        np.random.seed(seed)
        scenario = {
            "name": f"fixed_{env.name}",
            "benchmark": "benchmark_4agents",
            "seed": seed,
            "repeat": 1,
        }
        print(f"\n[case {case_idx}/{total_cases}] fixed seed={seed}")
        run_case(4, m, env, algo, metrics_to_check, scenario)

    for repeat in range(repeats):
        for m in random_ms:
            for env_idx, env in enumerate(random_envs):
                case_idx += 1
                seed = base_seed + 100_000 * (repeat + 1) + 1_000 * m + env_idx
                np.random.seed(seed)
                scenario = {
                    "name": f"random_r{repeat + 1}_{env.name}_m{m}",
                    "benchmark": "benchmark_4agents",
                    "seed": seed,
                    "repeat": repeat + 1,
                }
                print(f"\n[case {case_idx}/{total_cases}] seed={seed}")
                run_case(4, m, env, algo, metrics_to_check, scenario)

    print("\n" + "=" * 90)
    print(f"All 4-agent experiments completed. JSON reports are in {RESULT_FILE_DIR}/.")


if __name__ == "__main__":
    prepare_result_dirs()
    migrate_existing_result_files()
    console_log_path = make_console_log_path()

    with ConsoleLogCapture(console_log_path):
        print(f"Console log will be saved to: {console_log_path}")
        run_4agents_benchmark(repeats=1, base_seed=20260714)
        print(f"Console log saved to: {console_log_path}")
