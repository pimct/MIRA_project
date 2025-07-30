def normalize(value, min_val, max_val):
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def evaluate_fitness(results, config, minmax_tracker=None, verbose=False):
    """
    Evaluate fitness using revenue (from products) and CO2 emissions (per wet feed basis).

    Args:
        results (dict): Structured result with 'products' and 'emissions' keys
        config (dict): Contains product_prices, weights, bounds
        minmax_tracker (dict, optional): Updates dynamic normalization bounds
        verbose (bool): Print debug info

    Returns:
        tuple: (revenue, co2_emission, fitness_score)
    """
    products = results.get("products", {})
    emissions = results.get("emissions", {})

    product_prices = config.get("product_prices", {})
    weights = config.get("objective_weights", {})
    bounds = config.get("normalization_bounds", {})
    track = config.get("track_minmax", False)

    # === Revenue Calculation ===
    revenue = 0.0
    for product, price in product_prices.items():
        value = products.get(product, 0.0)
        product_revenue = value * price
        revenue += product_revenue
        if verbose:
            print(f"💰 Revenue: {product} = {value:.4f} × {price} → {product_revenue:.4f}")

    # === CO2 Emission (kg per kg wet feed)
    co2 = emissions.get("CO2_emission", 0.0)
    if verbose:
        print(f"🌫️ CO2 Emission (kg/kg wet feed): {co2:.4f}")

    # === Track bounds
    if minmax_tracker:
        minmax_tracker["revenue"][0] = min(minmax_tracker["revenue"][0], revenue)
        minmax_tracker["revenue"][1] = max(minmax_tracker["revenue"][1], revenue)
        minmax_tracker["co2"][0] = min(minmax_tracker["co2"][0], co2)
        minmax_tracker["co2"][1] = max(minmax_tracker["co2"][1], co2)

    # === Apply min-max normalization
    if minmax_tracker:
        rev_min, rev_max = minmax_tracker["revenue"]
        co2_min, co2_max = minmax_tracker["co2"]
    else:
        rev_min = bounds.get("revenue", {}).get("min", 0.0)
        rev_max = bounds.get("revenue", {}).get("max", 1.0)
        co2_min = bounds.get("co2", {}).get("min", 0.0)
        co2_max = bounds.get("co2", {}).get("max", 1.0)

    revenue_norm = normalize(revenue, rev_min, rev_max)
    co2_norm = normalize(co2, co2_min, co2_max)

    score = (
            weights.get("revenue", 0.0) * revenue_norm +
            weights.get("co2", 0.0) * co2_norm
    )

    if verbose:
        print(f"📊 Normalized Revenue: {revenue_norm:.4f} (raw: {revenue:.4f})")
        print(f"📉 Normalized CO2: {co2_norm:.4f} (raw: {co2:.4f})")
        print(f"🎯 Fitness Score: {score:.4f}")

    return revenue, co2, score
