# optimization/pso.py

import numpy as np
import random

def run_pso(config, feeds, feed_name, fitness_function):
    bounds = config["x_bounds"]
    n_particles = config["num_particles"]
    max_iter = config["max_iter"]
    dim = len(bounds)

    # Initialize particles and velocities
    particles = np.array([
        [random.uniform(*bounds[f"x{i+1}"]) for i in range(dim)]
        for _ in range(n_particles)
    ])
    velocities = np.zeros_like(particles)

    # Evaluate initial fitness
    pbest = particles.copy()
    pbest_scores = [fitness_function(p, feed_name, feeds, config) for p in pbest]

    # Define normalization function
    def normalize_all(scores):
        scores_array = np.array(scores)
        revenue_all = scores_array[:, 0]
        co2_all = scores_array[:, 1]
        rev_min, rev_max = revenue_all.min(), revenue_all.max()
        co2_min, co2_max = co2_all.min(), co2_all.max()

        def normalize(score):
            rev, co2 = score
            rev_n = (rev - rev_min) / (rev_max - rev_min + 1e-8)
            co2_n = (co2 - co2_min) / (co2_max - co2_min + 1e-8)
            return (
                    config['objective_weights']['revenue'] * rev_n +
                    config['objective_weights']['co2'] * co2_n
            )

        return normalize

    normalize = normalize_all(pbest_scores)
    gbest_idx = np.argmin([normalize(s) for s in pbest_scores])
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_scores[gbest_idx]

    # PSO parameters
    w = 0.5
    c1 = 1.5
    c2 = 1.5
    history = []

    for iter in range(max_iter):
        for i in range(n_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = (
                    w * velocities[i]
                    + c1 * r1 * (pbest[i] - particles[i])
                    + c2 * r2 * (gbest - particles[i])
            )
            particles[i] += velocities[i]

            # Clamp particles to bounds
            for d in range(dim):
                low, high = bounds[f"x{d+1}"]
                particles[i][d] = np.clip(particles[i][d], low, high)

            score = fitness_function(particles[i], feed_name, feeds, config)
            if normalize(score) < normalize(pbest_scores[i]):
                pbest[i] = particles[i].copy()
                pbest_scores[i] = score

        # Update global best
        normalize = normalize_all(pbest_scores)
        gbest_idx = np.argmin([normalize(s) for s in pbest_scores])
        gbest = pbest[gbest_idx].copy()
        gbest_score = pbest_scores[gbest_idx]

        history.append(gbest_score)
        print(f"Iteration {iter + 1}/{max_iter}: Revenue = {gbest_score[0]:.4f}, CO2 = {gbest_score[1]:.4f}")

    return gbest.tolist(), {
        "revenue": gbest_score[0],
        "co2": gbest_score[1]
    }, history
