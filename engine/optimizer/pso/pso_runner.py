# engine/optimizer/pso/pso_runner.py

import importlib
import os
import json
import pandas as pd
from config.config import load_run_config, prepare_run_config
from engine.optimizer.pso.particle import (
    initialize_swarm,
    get_var_ranges_by_process,
    quantize
)
from engine.optimizer.pso.velocity_update import update_velocity
from engine.optimizer.pso.fitness import evaluate_fitness
from engine.optimizer.pso.logger import log_particle_state
from engine.simulation.interface import run_simulation  # ✅ unified interface

def get_effective_vars(particle, num_vars):
    return particle["position"][1:1 + num_vars]

def run_pso():
    prepare_run_config()
    config = load_run_config()
    test_mode = config.get("test_mode", False)
    track_minmax = config.get("track_minmax", False)

    process_list = config.get("process_system", [])
    feed_comp = list(config.get("feed_comp", {}).values())
    model_config_map = config.get("model_config", {})

    pso_config = config.get("optimizer_config", {}).get("pso", {})
    population_size = pso_config.get("num_particles", 10)
    max_iter = pso_config.get("max_iter", 30)

    swarm = initialize_swarm(population_size, process_list, config)
    log = []
    convergence_log = []

    gbest = None
    best_score = float("inf")
    best_iteration = -1

    minmax_tracker = {
        "revenue": [float("inf"), float("-inf")],
        "co2": [float("inf"), float("-inf")]
    }

    print("\n🐝 Initial Swarm Preview:")
    for p in swarm:
        print(f"Particle {p['id']:2d} | Process: {process_list[p['process_index']]:<10} | Position: {p['position']}")

    for t in range(max_iter):
        for p in swarm:
            process_index = int(p["position"][0])
            process_name = process_list[process_index]
            model_config = model_config_map[process_name]

            mv_dict = model_config.get("MANIPULATED_VARIABLES", {})
            var_keys = list(mv_dict.keys())
            var_ranges = [mv_dict[k]["bounds"] for k in var_keys]
            var_steps = [mv_dict[k].get("step", None) for k in var_keys]
            num_vars = len(var_ranges)

            x_vars = get_effective_vars(p, num_vars)
            input_vector = [process_index] + x_vars
            results = run_simulation(process_name, model_config, input_vector, feed_comp, test_mode)

            revenue = results.get("revenue", 0.0)
            co2 = results.get("co2_emission", 0.0)

            if track_minmax:
                minmax_tracker["revenue"][0] = min(minmax_tracker["revenue"][0], revenue)
                minmax_tracker["revenue"][1] = max(minmax_tracker["revenue"][1], revenue)
                minmax_tracker["co2"][0] = min(minmax_tracker["co2"][0], co2)
                minmax_tracker["co2"][1] = max(minmax_tracker["co2"][1], co2)

            revenue, co2, score = evaluate_fitness(results, config, minmax_tracker if track_minmax else None)

            p["revenue"] = revenue
            p["co2_emission"] = co2
            p["score"] = score

            if score < p["pbest_score"]:
                p["pbest_score"] = score
                p["pbest_position"] = p["position"].copy()
                p["pbest_revenue"] = revenue
                p["pbest_co2"] = co2

            log_particle_state(log, t, p)

            if score < best_score:
                gbest = p
                best_score = score
                best_iteration = t

        gbest_process_index = int(gbest["pbest_position"][0])
        gbest_process_name = process_list[gbest_process_index]

        print(f"📈 Iteration {t + 1}: Best Score = {best_score:.4f}")
        print(f"   ↪ Process: {gbest_process_name}")
        print(f"   ↪ Revenue = {gbest['pbest_revenue']:.2f}, CO₂ = {gbest['pbest_co2']:.2f}")

        convergence_log.append({
            "iteration": t + 1,
            "score": best_score,
            "revenue": gbest["pbest_revenue"],
            "co2": gbest["pbest_co2"],
            "process": gbest_process_name
        })

        for p in swarm:
            process_index = int(p["position"][0])
            process_name = process_list[process_index]
            model_config = model_config_map[process_name]
            mv_dict = model_config.get("MANIPULATED_VARIABLES", {})
            var_keys = list(mv_dict.keys())
            var_ranges = [mv_dict[k]["bounds"] for k in var_keys]
            var_steps = [mv_dict[k].get("step", None) for k in var_keys]
            num_vars = len(var_ranges)

            sliced_velocity = p["velocity"][:num_vars + 1]
            sliced_position = p["position"][:num_vars + 1]
            sliced_pbest = p["pbest_position"][:num_vars + 1]
            sliced_gbest = gbest["pbest_position"][:num_vars + 1]

            new_v = update_velocity(sliced_velocity, sliced_position, sliced_pbest, sliced_gbest)
            p["velocity"] = new_v + [0.0] * (len(p["position"]) - len(new_v))

            for j in range(1, num_vars + 1):
                p["position"][j] += p["velocity"][j]
                step = var_steps[j - 1]
                bounds = var_ranges[j - 1]
                p["position"][j] = quantize(p["position"][j], step, *bounds)

    os.makedirs("logs", exist_ok=True)
    pd.DataFrame(log).to_csv("logs/pso_log.csv", index=False)
    pd.DataFrame(convergence_log).to_csv("logs/convergence.csv", index=False)

    with open("logs/best_result.json", "w") as f:
        json.dump({
            "iteration": best_iteration + 1,
            "score": gbest["pbest_score"],
            "position": gbest["pbest_position"],
            "revenue": gbest["pbest_revenue"],
            "co2_emission": gbest["pbest_co2"]
        }, f, indent=2)

    print(f"\n🏁 Optimization completed — best solution found at iteration {best_iteration + 1}.")
    print("🏆 Best Score       :", gbest['pbest_score'])
    print("🔧 Best Parameters  :", gbest['pbest_position'])
    print("💰 Revenue          :", gbest['pbest_revenue'])
    print("🌫️  CO2 Emission     :", gbest['pbest_co2'])

if __name__ == "__main__":
    run_pso()
