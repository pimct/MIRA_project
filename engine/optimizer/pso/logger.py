# engine/optimizer/pso/logger.py
#
# def log_particle_state(log, iteration, particle):
#     log.append({
#         "iteration": iteration,
#         "particle_id": particle["id"],
#         "process_index": particle["process_index"],
#         "parameters": particle["position"],
#         "revenue": particle["revenue"],
#         "co2_emission": particle["co2_emission"],
#         "score": particle["score"],
#         "pbest_position": particle["pbest_position"],
#         "pbest_score": particle["pbest_score"]
#     })


import json
import os

def log_particle_state(log, iteration, particle, output_dir="logs"):
    # Append the particle state to the log
    particle_state = {
        "iteration": iteration,
        "particle_id": particle["id"],
        "process_index": particle["process_index"],
        "parameters": particle["position"],
        "revenue": particle["revenue"],
        "co2_emission": particle["co2_emission"],
        "score": particle["score"],
        "pbest_position": particle["pbest_position"],
        "pbest_score": particle["pbest_score"]
    }
    log.append(particle_state)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write the particle state to a JSON file
    json_file_path = os.path.join(output_dir, f"particle_{particle['id']}_iteration_{iteration}.json")
    with open(json_file_path, "w") as json_file:
        json.dump(particle_state, json_file, indent=2)