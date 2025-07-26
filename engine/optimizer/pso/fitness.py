def normalize(value, min_val, max_val):
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def evaluate_fitness(results, config, minmax_tracker=None, verbose=False):
    """
    Evaluate fitness for PSO using revenue and CO2 emission.

    Args:
        results (dict): Aspen simulation output {product: value}
        config (dict): Config dictionary with pricing and weights
        minmax_tracker (dict, optional): Tracker for normalization bounds
        verbose (bool): If True, print debug logs

    Returns:
        tuple: (revenue, co2_emission, fitness_score)
    """
    # === Load config ===
    product_prices = config.get("product_prices", {})
    weights = config.get("objective_weights", {})
    bounds = config.get("normalization_bounds", {})
    track = config.get("track_minmax", False)

    # === Revenue Calculation ===
    revenue = 0.0
    for product, price in product_prices.items():
        value = results.get(product)
        if value is None:
            if verbose:
                print(f"⚠️ Missing value for product '{product}', assuming 0.0")
            value = 0.0
        revenue += value * price
        if verbose:
            print(f"💰 Revenue contribution: {product} = {value:.3f} × {price} → {value * price:.3f}")

    # === CO2 Emission ===
    co2 = results.get("co2_emission", 0.0)
    if co2 is None:
        if verbose:
            print("⚠️ CO2 emission not found, assuming 0.0")
        co2 = 0.0

    if minmax_tracker:
        minmax_tracker["revenue"][0] = min(minmax_tracker["revenue"][0], revenue)
        minmax_tracker["revenue"][1] = max(minmax_tracker["revenue"][1], revenue)
        minmax_tracker["co2"][0] = min(minmax_tracker["co2"][0], co2)
        minmax_tracker["co2"][1] = max(minmax_tracker["co2"][1], co2)

    # === Min-max Normalization ===
    if track and minmax_tracker:
        rev_min, rev_max = minmax_tracker["revenue"]
        co2_min, co2_max = minmax_tracker["co2"]
    else:
        rev_min = bounds.get("revenue", {}).get("min", 0.0)
        rev_max = bounds.get("revenue", {}).get("max", 1.0)
        co2_min = bounds.get("co2", {}).get("min", 0.0)
        co2_max = bounds.get("co2", {}).get("max", 1.0)

    revenue_norm = normalize(revenue, rev_min, rev_max)
    co2_norm = normalize(co2, co2_min, co2_max)

    # === Weighted fitness score ===
    score = (
            weights.get("revenue", 0.0) * revenue_norm +
            weights.get("co2", 0.0) * co2_norm
    )

    if verbose:
        print(f"📊 Normalized Revenue: {revenue_norm:.4f} (raw: {revenue})")
        print(f"📉 Normalized CO2: {co2_norm:.4f} (raw: {co2})")
        print(f"🎯 Fitness Score: {score:.4f}")

    return revenue, co2, score
