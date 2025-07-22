import numpy as np
from models.ann.ann_predictor import run_ann_models
from simulation.aspen_runner import run_aspen_simulation

def evaluate_fitness(particle, feed_name, feeds, config):
    x1, x2, x3, _ = particle
    x1 = int(round(x1))  # Ensure binary
    feed_features = feeds[feed_name]
    moisture = feed_features[1] / 100.0  # e.g., 9.56% → 0.0956
    dry_mass = 1.0 - moisture  # fraction of dry matter in 1 kg-wet

    prices = config["PRODUCT_PRICES"]
    model_dir = config["ann_model_dir"]
    ann_models = run_ann_models(model_dir)

    if x1 == 0:
        # Direct combustion pathway
        sim_outputs = run_aspen_simulation("direct.apw", pathway="direct")
        # Assume outputs already per kg wet-basis
        y_co2 = sim_outputs["co2"]
        y_elect = sim_outputs["electricity"]  # kWh/kg-wet
        y_char = 0  # no char output in this case

    else:
        # HTC + Valorization pathway
        ann_input = np.array([x2, x3] + feed_features[1:])  # skip dummy zero
        ann_outputs = {name: model.predict(ann_input.reshape(1, -1))[0] for name, model in ann_models.items()}

        char_yield = ann_outputs["charyield"]  # dry basis fraction
        char_HHV = ann_outputs["charHHV"]      # MJ/kg-dry-char
        co2 = ann_outputs["co2"]               # kg/kg-dry-feed
        elect = ann_outputs["electricity"]     # kWh/kg-dry-feed (estimated)

        # Convert to per kg-wet-feed
        y_co2 = co2 * dry_mass
        y_char = char_yield * dry_mass * char_HHV  # MJ/kg-wet
        y_elect = elect * dry_mass                # kWh/kg-wet

    revenue = (y_char * prices["char"]) + (y_elect * prices["electricity"])

    return revenue, y_co2  # For multi-objective optimization
