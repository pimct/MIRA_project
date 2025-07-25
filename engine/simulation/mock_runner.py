#
def run_simulation(model_config, input_vector, feed_comp):
    process_index = input_vector[0]
    temp = input_vector[1] if len(input_vector) > 1 else 300
    x_char = input_vector[2] if len(input_vector) > 2 else 0.3

    # Simple mock behavior
    char = 0.01 * temp + 100 * x_char
    electricity = - 0.2 * temp - 10 * x_char
    co2 = 1000 - 0.5 * temp + 20 * x_char

    return {
        "char": char,
        "electricity": electricity,
        "co2_emission": co2
    }
