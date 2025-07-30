# run.py

import argparse
import time
from engine.optimizer.pso.pso_runner import run_pso
from config.config import prepare_run_config

def main():
    parser = argparse.ArgumentParser(description="MIRA Optimization Runner")
    parser.add_argument("--mode", type=str, default="pso", choices=["pso", "prepare"], help="Mode to run")
    parser.add_argument("--test", action="store_true", help="Run in test mode with mock simulation")

    args = parser.parse_args()

    if args.mode == "prepare":
        print("🛠️  Preparing run configuration...")
        prepare_run_config()
    elif args.mode == "pso":
        print("🚀 Running PSO Optimization...")
        start_time = time.time()  # Start timer
        run_pso()
        end_time = time.time()    # End timer
        elapsed = end_time - start_time
        print(f"⏱️  PSO Optimization completed in {elapsed:.2f} seconds.")
    else:
        print("❌ Unknown mode:", args.mode)


if __name__ == "__main__":
    main()
