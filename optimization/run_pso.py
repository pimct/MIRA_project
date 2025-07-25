# run_pso.py (specific to HTC system)
from engine.optimizer.pso.pso import run_pso as general_run_pso
from optimization.fitness_function import evaluate_fitness


def run_pso(config, feeds, feed_name):
    return general_run_pso(
        config=config,
        feeds=feeds,
        feed_name=feed_name,
        fitness_function=evaluate_fitness
    )
