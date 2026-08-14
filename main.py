import json
import os
import re
import shutil
import sys
import time

import numpy as np

from algorithms.EFKX import ApproximateEFkX
from algorithms.baselines import EnvyCycleElimination, EFX_0618_Approx, RoundRobin
from algorithms.ef2x_four_agents import EF2XFourAgents
from algorithms.kgoods import KRoundRobinWithECE
from data.generator import (
    FixedGenerator,
    IdenticalGenerator,
    MallowsGenerator,
    NormalGenerator,
    OrderedGenerator,
    SharedTopGoodsGenerator,
    SparseZeroHeavyGenerator,
    SplidditSQLDataset,
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

# False: only isDemo=0 instances. True: both demo and non-demo instances.
SPLIDDIT_INCLUDE_DEMO = True


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
    """Benchmark adapter for the complete four-agent EF2X algorithm."""

    def __init__(self):
        self.name = "EF2X Four Agents"
        self.supported_num_agents = {4}
        self.last_result = None

    def run(self, instance):
        if instance.n != 4:
            raise ValueError("EF2X Four Agents 只适用于恰好 4 个 agents。")
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
            target_path = os.path.join(
                RESULT_FILE_DIR,
                f"{name}_{time.time_ns()}{ext}",
            )

        shutil.move(source_path, target_path)


def make_console_log_path():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(CONSOLE_LOG_DIR, f"benchmark_console_{timestamp}.txt")


def named(generator, name):
    """Give parameterized generators unique names in result files."""
    generator.name = name
    return generator


def safe_filename_part(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9_.=-]+", "_", value)
    return value.strip("_") or "unknown"


def to_jsonable(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def save_experiment_result(instance, allocation, metrics, algo_name, scenario):
    """Save one experiment result without overwriting other grid runs."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    unique_id = time.time_ns()
    dist_name = safe_filename_part(instance.dist_type)
    safe_algo_name = safe_filename_part(algo_name)
    scenario_name = safe_filename_part(scenario["name"])
    filename = (
        f"{scenario_name}_{dist_name}_{safe_algo_name}_"
        f"n{instance.n}_m{instance.m}_{unique_id}.json"
    )
    filepath = os.path.join(RESULT_FILE_DIR, filename)

    os.makedirs(RESULT_FILE_DIR, exist_ok=True)
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


def evaluate_all_metrics(instance, allocation, algo, metrics_to_check, runtime):
    results_metrics = {"runtime": runtime}
    metric_parts = []

    for metric in metrics_to_check:
        value = metric.compute(instance, allocation)
        results_metrics[metric.name] = value
        metric_parts.append(f"{metric.name}: {format_metric_value(value):>7}")

    alpha_efx_metric = MaxAlphaEFkX(k=1)
    alpha_efx_value = alpha_efx_metric.compute(instance, allocation)
    results_metrics[alpha_efx_metric.name] = alpha_efx_value
    metric_parts.append(f"{alpha_efx_metric.name}: {alpha_efx_value:>7.3f}")

    if isinstance(algo, (KRoundRobinWithECE, ApproximateEFkX)):
        eval_k = algo.eval_k
        alpha_efkx_metric = MaxAlphaEFkX(k=eval_k)
        alpha_efkx_value = alpha_efkx_metric.compute(instance, allocation)
        results_metrics[alpha_efkx_metric.name] = alpha_efkx_value

        if isinstance(algo, KRoundRobinWithECE):
            guaranteed_alpha = eval_k / (eval_k + 1)
        else:
            guaranteed_alpha = (eval_k + 1) / (eval_k + 2)

        guarantee_passed = alpha_efkx_value + 1e-9 >= guaranteed_alpha
        results_metrics["theoretical_guarantee"] = guaranteed_alpha
        results_metrics["guarantee_passed"] = guarantee_passed
        results_metrics[f"GuaranteeEF{eval_k}X"] = guarantee_passed

        status = "PASS" if guarantee_passed else "FAIL"
        metric_parts.extend(
            [
                f"{alpha_efkx_metric.name}: {alpha_efkx_value:>7.3f}",
                f"Guarantee>={guaranteed_alpha:.3f}: {status:>4}",
            ]
        )

    if isinstance(algo, EF2XFourAgentsBenchmarkAdapter) and algo.last_result is not None:
        results_metrics["internal_ef2x_all_goods"] = True
        results_metrics["metric_efx_definition"] = "evaluation.metrics.EFXChecker"
        results_metrics["metric_efkx_definition"] = "evaluation.metrics.MaxAlphaEFkX uses positive-goods deletion"
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


def build_algorithms(k_values):
    algorithms = [
        RoundRobin(),
        EnvyCycleElimination(),
        EFX_0618_Approx(),
        EF2XFourAgentsBenchmarkAdapter(),
    ]

    for k in k_values:
        algorithms.append(KRoundRobinWithECE(k=k))
        if k >= 2:
            algorithms.append(ApproximateEFkX(k=k))

    return algorithms


def run_experiment_suite(n, m, data_envs, algos_to_test, metrics_to_check, scenario=None):
    """Backward-compatible runner for a single (n, m) setting."""
    scenario = scenario or {"name": f"single_n{n}_m{m}", "seed": None, "repeat": 0}

    print("=" * 90)
    print(f"{scenario['name']} | {n} agents, {m} items")
    print("=" * 90)

    for env in data_envs:
        print(f"\n>>> environment: {env.name}")
        instance = env.generate(n, m)

        for algo in algos_to_test:
            supported = getattr(algo, "supported_num_agents", None)
            if supported is not None and n not in supported:
                print(f"    [skipping algorithm]: {algo.name:34} | unsupported n={n}")
                continue
            print(f"    [running algorithm]: {algo.name:34}", end="")

            start_time = time.time()
            try:
                allocation = algo.run(instance)
            except Exception as exc:
                runtime = time.time() - start_time
                failure_metrics = {
                    "runtime": runtime,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                if isinstance(algo, EF2XFourAgentsBenchmarkAdapter):
                    failure_metrics["internal_checker_definition"] = "all_goods"
                    failure_metrics["algorithm_metadata"] = algo.get_metadata()
                save_experiment_result(instance, {}, failure_metrics, algo.name, scenario)
                print(
                    f" | FAILED: {type(exc).__name__}: {str(exc)[:240]} "
                    f"| Time: {runtime:.4f}s"
                )
                continue

            runtime = time.time() - start_time

            results_metrics, metric_str = evaluate_all_metrics(
                instance=instance,
                allocation=allocation,
                algo=algo,
                metrics_to_check=metrics_to_check,
                runtime=runtime,
            )

            save_experiment_result(instance, allocation, results_metrics, algo.name, scenario)
            print(f" | {metric_str} | Time: {runtime:.4f}s")


def run_benchmark_grid(
    size_grid,
    data_envs,
    k_values,
    metrics_to_check,
    repeats=1,
    base_seed=42,
):
    total_cases = len(size_grid) * len(data_envs) * repeats
    algorithms = build_algorithms(k_values)
    case_idx = 0

    print(f"Benchmark grid: {total_cases} generated instances")
    print(f"Algorithms per instance: {len(algorithms)}")
    print(f"k values: {list(k_values)}")

    for repeat in range(repeats):
        for n, m in size_grid:
            for env_idx, env in enumerate(data_envs):
                case_idx += 1
                seed = base_seed + repeat * 100_000 + n * 1_000 + m * 10 + env_idx
                np.random.seed(seed)

                scenario = {
                    "name": f"grid_r{repeat + 1}_{env.name}_n{n}_m{m}",
                    "seed": seed,
                    "repeat": repeat + 1,
                    "k_values": list(k_values),
                }

                print(f"\n[case {case_idx}/{total_cases}] seed={seed}")
                run_experiment_suite(
                    n=n,
                    m=m,
                    data_envs=[env],
                    algos_to_test=algorithms,
                    metrics_to_check=metrics_to_check,
                    scenario=scenario,
                )

    print("\n" + "=" * 90)
    print(f"All experiments completed. JSON reports are in {RESULT_FILE_DIR}/.")


def run_real_dataset(instances, k_values, metrics_to_check, include_demo=False):
    """Run the unchanged algorithm/metric suite once on every loaded instance."""
    algorithms = build_algorithms(k_values)
    dataset_scope = "all instances" if include_demo else "isDemo=0 instances"
    print(f"Spliddit benchmark: {len(instances)} complete {dataset_scope}")
    print(f"Algorithms per instance: {len(algorithms)}")

    for index, instance in enumerate(instances, start=1):
        source_id = instance.source_instance_id
        is_demo = instance.source_is_demo
        scenario = {
            "name": f"spliddit_instance_{source_id}",
            "seed": None,
            "repeat": 1,
            "source": f"Spliddit SQL dump ({dataset_scope})",
            "source_instance_id": source_id,
            "source_is_demo": is_demo,
            "include_demo": include_demo,
            "k_values": list(k_values),
        }
        print(
            f"\n[Spliddit case {index}/{len(instances)}] "
            f"source_instance_id={source_id}, isDemo={int(is_demo)}"
        )
        run_experiment_suite(
            n=instance.n,
            m=instance.m,
            data_envs=[FixedGenerator(instance.valuations, name=instance.dist_type)],
            algos_to_test=algorithms,
            metrics_to_check=metrics_to_check,
            scenario=scenario,
        )

    print("\n" + "=" * 90)
    print(f"All real-world experiments completed. JSON reports are in {RESULT_FILE_DIR}/.")


if __name__ == "__main__":
    prepare_result_dirs()
    migrate_existing_result_files()
    console_log_path = make_console_log_path()

    with ConsoleLogCapture(console_log_path):
        print(f"Console log will be saved to: {console_log_path}")

        # Fixed case for debugging known edge behavior. It is only valid for n=4, m=13.
        fixed_valuations = np.array(
            [
                [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1, 10],
                [100, 95, 85, 65, 55, 45, 35, 25, 15, 10, 5, 1, 10],
                [90, 100, 85, 75, 65, 55, 45, 35, 25, 15, 10, 5, 10],
                [95, 90, 100, 80, 70, 60, 50, 40, 30, 20, 10, 5, 10],
            ],
            dtype=float,
        )

        data_environments = [
            named(UniformGenerator(v_min=0, v_max=1000), "Uniform(0,1000)"),
            named(UniformGenerator(v_min=100, v_max=1000), "Uniform(100,1000)"),
            named(NormalGenerator(mean=500, std=100), "Normal(mean=500,std=100)"),
            named(NormalGenerator(mean=300, std=500), "Normal(mean=300,std=500)"),
            MallowsGenerator(phi=0.0),
            MallowsGenerator(phi=0.1),
            MallowsGenerator(phi=0.5),
            MallowsGenerator(phi=0.9),
            MallowsGenerator(phi=1.0),
            named(IdenticalGenerator(v_min=0, v_max=1000), "Identical(0,1000)"),
            named(OrderedGenerator(v_min=0, v_max=1000), "Ordered(0,1000)"),
            SparseZeroHeavyGenerator(
                zero_probability=0.8,
                v_min=1,
                v_max=1000,
            ),
            SharedTopGoodsGenerator(
                shared_top_fraction=0.2,
                regular_min=0,
                regular_max=500,
                top_min=750,
                top_max=1000,
            ),
        ]

        # Keep m >= n because ApproximateEFkX creates a one-good seed per agent.
        # size_grid = [
        #     (3, 10),
        #     (4, 13),
        #     (5, 20),
        #     (6, 25),
        #     (8, 40),
        # ]
        size_grid = [
            (3, 12),
            (4, 16),
            (5, 20),
            (6, 24),
            (7, 28),
            (8, 32)
        ]

        k_values = [1, 2, 3, 4, 5]

        my_metrics = [
            EF1Checker(),
            EFXChecker(),
        ]

        # Original synthetic benchmark (temporarily disabled while testing the
        # newly added real-world dataset).
        run_benchmark_grid(
            size_grid=size_grid,
            data_envs=data_environments,
            k_values=k_values,
            metrics_to_check=my_metrics,
            repeats=5,
            base_seed=42,
        )

        # spliddit_dataset = SplidditSQLDataset(
        #     r"D:\Courses\Semester1\TTDS\cw3\pythonProject\data"
        #     r"\spliddit-2026-07-18-goods.sql"
        # )
        # real_instances = spliddit_dataset.load_instances(
        #     require_m_at_least_n=False,
        #     require_strictly_more_items=True,
        #     require_positive_valuations=True,
        #     include_demo=SPLIDDIT_INCLUDE_DEMO,
        # )
        # run_real_dataset(
        #     instances=real_instances,
        #     k_values=k_values,
        #     metrics_to_check=my_metrics,
        #     include_demo=SPLIDDIT_INCLUDE_DEMO,
        # )

        print(f"Console log saved to: {console_log_path}")

        # Optional deterministic manual case. Uncomment when you want a small
        # hand-crafted EF2X-style sanity test.
        # run_experiment_suite(
        #     n=4,
        #     m=13,
        #     data_envs=[FixedGenerator(valuations=fixed_valuations, name="Manual-EF2X-Test")],
        #     algos_to_test=build_algorithms(k_values),
        #     metrics_to_check=my_metrics,
        #     scenario={"name": "manual_fixed_case", "seed": None, "repeat": 1},
        # )

