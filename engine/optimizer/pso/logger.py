# engine/optimizer/pso/logger.py

def log_particle_state(log, iteration, particle):
    log.append({
        "iteration": iteration,
        "particle_id": particle["id"],
        "process_index": particle["process_index"],
        "parameters": particle["position"],
        "revenue": particle["revenue"],
        "co2_emission": particle["co2_emission"],
        "score": particle["score"],
        "pbest_position": particle["pbest_position"],
        "pbest_score": particle["pbest_score"]
    })
