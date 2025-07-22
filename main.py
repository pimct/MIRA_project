#from optimization.pso import run_pso
from config import PSO_CONFIG, FEEDS, DATA_PATHS, MODEL_PATHS, ANN_COLUMN_INDICES
from model_training.train_htc_ann import train_and_save_ann_model
import json
import os
import joblib

def select_system():
    print("🌐 Initialize MIRA Framework")
    print("Available systems:")
    print("1. Waste-to-Energy (HTC vs Direct Incineration)")
    selection = input("Select system to analyze (1): ")
    if selection.strip() != "1":
        print("❌ Invalid or unsupported system. Only option 1 is available now.")
        exit()
    print("✅ Waste-to-Energy system selected.\n")

def select_feed():
    print("📦 Available feedstocks:")
    feed_keys = list(FEEDS.keys())
    for idx, feed in enumerate(feed_keys):
        print(f"{idx + 1}. {feed}")

    try:
        selection = int(input("Select feedstock by number: ")) - 1
        if selection < 0 or selection >= len(feed_keys):
            raise ValueError
    except ValueError:
        print("❌ Invalid selection. Exiting.")
        exit()

    selected_feed_name = feed_keys[selection]
    selected_feed_data = FEEDS[selected_feed_name]
    print(f"✅ Selected feedstock: {selected_feed_name}\n")
    return selected_feed_name, selected_feed_data

def main():
    select_system()

    # === Ask if user wants to train ANN model or use existing one ===
    if os.path.exists(MODEL_PATHS["ann_model"]):
        choice = input("Existing ANN model found. Do you want to re-train it? (y/n): ").lower()
    else:
        choice = "y"

    if choice == 'y':
        print("🔧 Training ANN model for HTC pathway...")
        train_and_save_ann_model(DATA_PATHS["htc_dataset"], MODEL_PATHS, ANN_COLUMN_INDICES)
        print("✅ ANN model training complete. Model saved to 'models/ann/'.\n")
    else:
        print("✅ Using existing ANN model at 'models/ann/'.\n")

    # === Ask for optimization ===
    choice = input("Do you want to start PSO optimization now? (y/n): ").lower()
    if choice != 'y':
        print("🛑 Optimization skipped. Exiting MIRA.")
        return

    # === Feed selection before optimization ===
    selected_feed_name, selected_feed_data = select_feed()

    print("🚀 Starting PSO Optimization...")
    best_particle, best_metrics, history = run_pso(PSO_CONFIG, {selected_feed_name: selected_feed_data})

    print("\n✅ Optimization Complete")
    print("Best Particle:", best_particle)
    print("Best Metrics:", best_metrics)

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/best_solution.json", "w") as f:
        json.dump({
            "particle": best_particle,
            "metrics": best_metrics
        }, f, indent=2)

    with open("results/iteration_logs.csv", "w") as f:
        f.write("iteration,revenue,co2\n")
        for i, (rev, co2) in enumerate(history):
            f.write(f"{i},{rev},{co2}\n")

    print("\n📁 Results saved to 'results/' folder")

if __name__ == "__main__":
    main()
