# engine/optimizer/pso/fitness.py

def normalize(value, min_val, max_val):
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def evaluate_fitness(results, config, minmax_tracker=None):
    # === Revenue Calculation ===
    product_prices = config.get("product_prices", {})
    revenue = 0.0
    for product, value in results.items():
        if product == "co2_emission":
            continue
        price = product_prices.get(product, 0.0)
        revenue += value * price

    # === CO2 Emission ===
    co2 = results.get("co2_emission", 0.0)

    # === Weights and Normalization Settings ===
    weights = config.get("objective_weights", {})
    bounds = config.get("normalization_bounds", {})
    track = config.get("track_minmax", False)

    if track and minmax_tracker:
        rev_min, rev_max = minmax_tracker["revenue"]
        co2_min, co2_max = minmax_tracker["co2"]
    else:
        rev_min = bounds.get("revenue", {}).get("min", 0)
        rev_max = bounds.get("revenue", {}).get("max", 1)
        co2_min = bounds.get("co2", {}).get("min", 0)
        co2_max = bounds.get("co2", {}).get("max", 1)

    # === Normalize and Compute Score ===
    revenue_norm = normalize(revenue, rev_min, rev_max)
    co2_norm = normalize(co2, co2_min, co2_max)

    score = (
            weights.get("revenue", 0.0) * revenue_norm +
            weights.get("co2", 0.0) * co2_norm
    )

    return revenue, co2, score
