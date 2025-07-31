import os
import time
import json
import itertools
import pandas as pd
import argparse

from config.config import load_yaml_config
from process_models.htc.htc_process import run_htc_model


def generate_grid(bounds, steps):
    """Generate grid from bounds and step sizes."""
    axes = []
    for key in bounds:
        lb, ub = bounds[key]["bounds"]
        step = bounds[key]["step"]
        axis = [round(lb + i * step, 5) for i in range(int((ub - lb) / step) + 1)]
        axes.append(axis)
    return list(itertools.product(*axes))


def choose_feed(feed_csv):
    df = pd.read_csv(feed_csv)
    df.columns = df.columns.str.strip()
    feed_names = df["Feed"].dropna().unique().tolist()

    print("🍃 Available feedstocks:")
    for i, name in enumerate(feed_names, 1):
        print(f"{i}. {name}")

    try:
        choice = int(input("🔢 Enter the number of your selected feedstock: "))
        if 1 <= choice <= len(feed_names):
            feed_name = feed_names[choice - 1]
        else:
            raise ValueError
    except Exception:
        print("⚠️ Invalid input. Defaulting to first feed.")
        feed_name = feed_names[0]

    row = df[df["Feed"] == feed_name].iloc[0].to_dict()
    row.pop("Feed", None)
    values = list(df[df["Feed"] == feed_name].iloc[0].values)
    print(f"\n✅ You selected: {feed_name}")
    print("🔬 Feed composition:")
    for k, v in feed_dict.items():
        print(f"   {k:<20} = {v}")

    input("\n👉 Press Enter to confirm and start the simulation...")

    return feed_name, row, values


def run_htc_pareto_selected_feed(test_mode=False):
    config = load_yaml_config()
    model_config = config["MODEL_CONFIG"]["htc"]
    manipulated_vars = model_config["MANIPULATED_VARIABLES"]
    var_keys = list(manipulated_vars.keys())

    # Select feed
    feed_csv = os.path.join(os.path.dirname(__file__), "..", "data", "datasets", "feed_data.csv")
    feed_name, feed_dict, feed_array = choose_feed(feed_csv)

    # Generate grid
    grid = generate_grid(manipulated_vars, manipulated_vars)

    print("\n🧮 Variable ranges:")
    for var in var_keys:
        bounds = manipulated_vars[var]["bounds"]
        step = manipulated_vars[var]["step"]
        count = int((bounds[1] - bounds[0]) / step) + 1
        print(f"   {var:<15}: {bounds[0]} → {bounds[1]} (step {step}) → {count} points")

        print(f"\n📊 Running HTC for feed '{feed_name}' with total combinations: {len(grid)}")

    # Preview first 2 runs
    preview_n = 2
    print(f"\n🔍 Running preview for the first {preview_n} cases...")
    for i in range(preview_n):
        x_vals = grid[i]
        x_input = {var: val for var, val in zip(var_keys, x_vals)}
        particle_position = [None, x_input.get("temp"), x_input.get("char_routing")]
        try:
            output = run_htc_model(model_config, particle_position, feed_array)
            print(f"\n🔹 Preview {i+1}: x = {x_input}")
            print("   ↪ Products :", output.get("products", {}))
            print("   ↪ Emissions:", output.get("emissions", {}))
        except Exception as e:
            print(f"\n❌ Preview {i+1} failed: {x_input} → {e}")

    input(f"\n✅ Preview complete. Press Enter to run remaining {len(grid) - preview_n} combinations, or Ctrl+C to abort...")


    results = []
    start_time = time.time()

    for i, values in enumerate(grid):
        x_input = {var: val for var, val in zip(var_keys, values)}
        particle_position = [None, x_input.get("temp"), x_input.get("char_routing")]

        try:
            output = run_htc_model(model_config, particle_position, feed_array)
            results.append({
                "feed": feed_name,
                "x_input": x_input,
                "outputs": output
            })
            print(f"[{i+1}/{len(grid)}] ✅ {x_input} → {output['products']}")
        except Exception as e:
            print(f"[{i+1}/{len(grid)}] ❌ Failed {x_input} → {e}")

    elapsed = time.time() - start_time
    print(f"\n⏱️ HTC Pareto run completed in {elapsed:.2f} seconds. Successful runs: {len(results)}")

    os.makedirs("logs", exist_ok=True)
    out_file = f"logs/pareto/pareto_htc_{feed_name}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"📁 Results saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="HTC Pareto Runner (Single Feed)")
    parser.add_argument("--test", action="store_true", help="Run in test mode with mock simulation")
    args = parser.parse_args()

    print("🚀 Starting HTC Pareto run for selected feed...")
    run_htc_pareto_selected_feed(test_mode=args.test)


if __name__ == "__main__":
    main()
