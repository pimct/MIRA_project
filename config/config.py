import os
import yaml
import pandas as pd
import json

# === Path Setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # → .../MIRA/config
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))  # → .../MIRA

YAML_PATH = os.path.join(BASE_DIR, "config.yaml")
FEED_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "datasets", "feed_data.csv")
RUN_CONFIG_PATH = os.path.join(BASE_DIR, "run_config.json")


# === Load YAML Config ===
def load_yaml_config(yaml_path=YAML_PATH):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


# === Feed Selection ===
def load_feed_data(config, csv_path=FEED_CSV_PATH):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if "Feed" not in df.columns:
        raise ValueError("❌ Column 'Feed' not found in the CSV.")

    feed_names = df["Feed"].dropna().unique().tolist()
    default_feed = config.get("default_feedstock", feed_names[0])

    print("🍃 Available feedstocks:")
    for i, name in enumerate(feed_names, 1):
        print(f"{i}. {name}")
    print(f"(Default = {default_feed})")

    try:
        choice = input("Enter the number of your selected feedstock (press Enter for default): ").strip()
        if choice == "":
            feed_name = default_feed
        else:
            index = int(choice)
            if 1 <= index <= len(feed_names):
                feed_name = feed_names[index - 1]
            else:
                raise ValueError
        row = df[df["Feed"] == feed_name].iloc[0].to_dict()
        row.pop("Feed", None)
        print(f"\n✅ You selected: {feed_name}")
        return feed_name, row
    except Exception:
        print(f"\n⚠️ Invalid input. Defaulting to: {default_feed}")
        row = df[df["Feed"] == default_feed].iloc[0].to_dict()
        row.pop("Feed", None)
        return default_feed, row


# === Process System Selection ===
def select_process_system(config):
    options = config.get("process_system", {})
    keys = list(options.keys())
    default_key = config.get("default_system", keys[0])

    print("🔧 Please select a process system pathway:")
    for i, key in enumerate(keys, 1):
        print(f"{i}. {key} → {options[key]}")
    print(f"(Default = {default_key})")

    try:
        choice = input("Enter the number of your choice (press Enter for default): ").strip()
        if choice == "":
            selected_key = default_key
        else:
            index = int(choice)
            if 1 <= index <= len(keys):
                selected_key = keys[index - 1]
            else:
                raise ValueError
        selected_process = options[selected_key]
        print(f"\n✅ You selected: {selected_key} → {selected_process}")
        return selected_process
    except Exception:
        selected_process = options[default_key]
        print(f"\n⚠️ Invalid input. Defaulting to: {default_key} → {selected_process}")
        return selected_process



# === Scenario Selection ===
def select_scenario(config):
    scenarios = config.get("scenarios", {})
    keys = list(scenarios.keys())
    default_scenario = config.get("default_scenario", keys[0])  # fallback to first if no default key

    print("\n🌍 Please select an optimization scenario:")
    for i, key in enumerate(keys, 1):
        print(f"{i}. {key} → {scenarios[key]['objective_weights']}")

    try:
        choice = int(input("Enter the number of your choice (1–3): "))
        if 1 <= choice <= len(keys):
            selected = keys[choice - 1]
            print(f"\n✅ You selected scenario: {selected}")
            return selected, scenarios[selected]["objective_weights"]
        else:
            raise ValueError  # fall through to default
    except ValueError:
        print(f"\n⚠️ Invalid input. Defaulting to: {default_scenario}")
        return default_scenario, scenarios[default_scenario]["objective_weights"]


def prepare_run_config():
    config = load_yaml_config()
    selected_process = select_process_system(config)
    scenario_name, objective_weights = select_scenario(config)
    feed_name, feed_comp = load_feed_data(config, FEED_CSV_PATH)

    run_config = {}

    # === Collect manipulated variables and their bounds ===
    system_manipulated_var = []
    system_manipulated_var_details = {}
    var_bounds = []

    selected_model_config = {}
    model_config_all = config.get("MODEL_CONFIG", {})

    for proc in selected_process:
        if proc in model_config_all:
            selected_model_config[proc] = model_config_all[proc]

    for proc in selected_process:
        manipulated_vars = selected_model_config.get(proc, {}).get("MANIPULATED_VARIABLES", {})
        var_names = list(manipulated_vars.keys())
        system_manipulated_var_details[proc] = var_names

        for var_name in var_names:
            bounds = manipulated_vars[var_name]["bounds"]
            var_bounds.append(bounds)
            system_manipulated_var.append(bounds[0])  # Lower bound as initial value

    # === Get product prices from YAML ===
    product_prices = config.get("product_prices", {})
    if not product_prices:
        print("⚠️ Warning: No product_prices found in config.yaml")

    # === Construct run_config ===
    run_config["system_manipulated_var"] = system_manipulated_var
    run_config["system_manipulated_var_details"] = system_manipulated_var_details
    run_config["var_bounds"] = var_bounds
    run_config["feed"] = feed_name
    run_config["feed_comp"] = feed_comp
    run_config["process_system"] = selected_process
    run_config["scenario"] = scenario_name
    run_config["objective_weights"] = objective_weights
    run_config["optimizer_config"] = config.get("OPTIMIZER_CONFIG", {})
    run_config["product_prices"] = product_prices
    run_config["model_config"] = selected_model_config

    save_run_config(run_config)



# === Save run_config to JSON ===
def save_run_config(run_config, output_path=RUN_CONFIG_PATH):
    with open(output_path, "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"✅ Saved run_config to {output_path}")

# === Load config ===
def load_run_config(path=RUN_CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# # === Main Entry ===
# if __name__ == "__main__":
#     run_config = prepare_run_config()
#     save_run_config(run_config)


