import argparse
import importlib
import itertools
import json
import os
import time

from config.config import prepare_run_config, load_run_config
from engine.simulation.interface import run_simulation


def generate_full_variable_space(bounds, step_sizes):
    """
    Generate grid combinations for the variable space.
    bounds: dict with variable: [min, max]
    step_sizes: dict with variable: step
    """
    ranges = {
        var: [round(bounds[var][0] + i * step_sizes[var], 5)
              for i in range(int((bounds[var][1] - bounds[var][0]) / step_sizes[var]) + 1)]
        for var in bounds
    }
    grid = list(itertools.product(*ranges.values()))
    keys = list(ranges.keys())
    return [dict(zip(keys, vals)) for vals in grid]


def run_pareto_exploration(test_mode=False):
    # Load run config
    config = load_run_config()
    process_list = config.get("process_names", ["htc", "direct_combustion"])  # Add your process names here
    bounds = config.get("decision_variable_bounds", {})
    step_sizes = config.get("grid_step_size", {k: 0.1 for k in bounds})  # Default 0.1 step

    feed_data = config.get("feed_composition", {})
    model_config = config.get("model_config", {})

    print("📊 Generating variable combinations...")
    variable_combinations = generate_full_variable_space(bounds, step_sizes)
    print(f"🔍 Total combinations to run: {len(variable_combinations)}")

    results = []

    start_time = time.time()
    for i, x_input in enumerate(variable_combinations):
        for process_name in process_list:
            try:
                output = run_simulation(process_name, model_config, x_input, feed_data, test_mode=test_mode)
                result_entry = {
                    "process": process_name,
                    "x_input": x_input,
                    "outputs": output
                }
                results.append(result_entry)
                print(f"[{i+1}/{len(variable_combinations)}] ✅ {process_name} | x = {x_input} → {output}")
            except Exception as e:
                print(f"[{i+1}/{len(variable_combinations)}] ❌ Failed {process_name} | x = {x_input}: {e}")

    elapsed = time.time() - start_time
    print(f"\n🎉 Pareto sweep completed in {elapsed:.2f} seconds.")
    print(f"📁 Total successful runs: {len(results)}")

    # Optional: Save results to file
    with open("results/pareto_results.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="MIRA Pareto Runner")
    parser.add_argument("--test", action="store_true", help="Run with mock simulation mode")
    args = parser.parse_args()

    print("🚀 Running Pareto Exploration...")
    run_pareto_exploration(test_mode=args.test)


if __name__ == "__main__":
    main()
