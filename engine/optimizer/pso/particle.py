# engine/optimizer/pso/particle.py

import random


def get_var_ranges_by_process(config, process_name):
    """
    Extract manipulated variable bounds for a specific process from config.
    Returns: list of [min, max] for each manipulated variable.
    """
    model_config = config["model_config"].get(process_name, {})
    mv_dict = model_config.get("MANIPULATED_VARIABLES", {})
    return [mv_dict[key]["bounds"] for key in mv_dict]


def get_var_steps_by_process(config, process_name):
    """
    Extract step size for each manipulated variable.
    Returns: list of step values (or None if unspecified).
    """
    model_config = config["model_config"].get(process_name, {})
    mv_dict = model_config.get("MANIPULATED_VARIABLES", {})
    return [mv_dict[key].get("step", None) for key in mv_dict]


def quantize(value, step, min_val, max_val):
    """
    Round a value to the nearest step within given bounds.
    """
    if step is None:
        return value
    rounded = round((value - min_val) / step) * step + min_val
    return max(min(rounded, max_val), min_val)


def initialize_swarm(n_particles, process_list, config):
    """
    Initialize a swarm of particles. If both 'htc' and 'combustion' are in process_list,
    assign exactly one particle to combustion, the rest to htc.
    """
    swarm = []

    include_both = set(process_list) >= {"htc", "combustion"}


    for i in range(n_particles):
        # === Assign process ===
        if include_both:
            if i == 0:
                process_name = "combustion"
            else:
                process_name = "htc"
        else:
            process_name = random.choice(process_list)

        process_index = process_list.index(process_name)

        # === Get variable space ===
        var_ranges = get_var_ranges_by_process(config, process_name)
        var_steps = get_var_steps_by_process(config, process_name)

        # === Quantized initialization ===
        x_vars = []
        for (low, high), step in zip(var_ranges, var_steps):
            value = random.uniform(low, high)
            value = quantize(value, step, low, high)
            x_vars.append(value)

        position = [process_index] + x_vars
        velocity = [0.0] * len(position)

        particle = {
            "id": i,
            "process_index": process_index,
            "position": position,
            "velocity": velocity,
            "revenue": None,
            "co2_emission": None,
            "score": None,
            "pbest_position": position.copy(),
            "pbest_score": float("inf"),
            "pbest_revenue": None,
            "pbest_co2": None
        }
        swarm.append(particle)

    return swarm
