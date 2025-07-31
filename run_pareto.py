import argparse
import time
import itertools
import json
import os

from config.config import prepare_run_config, load_run_config
from engine.simulation.interface import run_simulation


def generate_variable_grid(bounds, step=0.1):
    """Generate full variable grid from bounds."""
    grid_axes = []
    for lb, ub in bounds:
        axis = [round(lb + i * step, 5) for i in range(int((ub - lb) / step) + 1)]
        grid_axes.append(axis)
    return list(itertools.product(*grid_axes))


def run_pareto(test_mode=False):
    # Prompt user if needed
    prepare_run_config()
    config = load_run_config()

    process_list = config["process_system"]
    var_bounds = config["var_bounds"]
    x_names = config["system_manipulated_var_details"]
    feed = config["feed"]
    feed_comp = config["feed_comp"]
    model_config = config["model_config"]

    step = 0.1  # Set your preferred step size here

    print("\n📊 Generating grid...")
    grid = generate_variable_grid(var_bounds, step=step)
    print(f"🔍 Total combinations: {len(grid)}")

    results = []
    start_time = time.time()

    for i, x_vals in enumerate(grid):
        x_input = {var: val for proc in process_list for var, val in zip(x_names[proc], x_vals)}
        for proc in process_list:
            try:
                output = run_simulation(proc, model_config, x_input, feed_comp, test_mode)
                result = {
                    "process": proc,
                    "x_input": x_input,
                    "outputs": output
                }
                results.append(result)
                print(f"[{i+1}/{len(grid)}] ✅ {proc} | x = {x_input} → {output}")
            except Exception as e:
                print(f"[{i+1}/{len(grid)}] ❌ {proc} failed | x = {x_input} | {e}")

    duration = time.time() - start_time
    print(f"\n⏱️ Completed in {duration:.2f} seconds. Successful runs: {len(results)}")

    os.makedirs("results", exist_ok=True)
    with open("results/pareto_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("📁 Results saved to results/pareto_results.json")


def main():
    parser = argparse.ArgumentParser(description="MIRA Pareto Runner")
    parser.add_argument("--test", action="store_true", help="Run in test mode with mock simulation")
    args = parser.parse_args()

    print("🚀 Running Pareto Simulation Sweep...")
    run_pareto(test_mode=args.test)


if __name__ == "__main__":
    main()
