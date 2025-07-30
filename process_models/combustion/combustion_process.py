import numpy as np
from engine.simulation.hybrid_runner import run_simulation

def prepare_x_input(feed_comp):
    """
    Prepare input vector for HTC process.

    Args:
        feed_comp (list): Proximate/ultimate analysis values
        temp (float): Operating temperature
        x_char_split (float): Char yield fraction
        tr (float): Residence time in hours
        solid_loading (float): Solid loading in wt%

    Returns:
        np.array: Input vector for HTC process
    """
    # Ensure feed_comp is a numpy array for consistency
    # feed_comp order:  # [Moisture, C, H, N, S, O, VM, FC, Ash]

    feed_comp = np.array(feed_comp)    # Validate input dimensions

    # Combustion input vector
    x_input = np.array([
            #  Proximate analysis (Moisture, FC, VM, Ash)
            feed_comp[0],
            feed_comp[7],
            feed_comp[6],
            feed_comp[8],
            #  Ultimate analysis (Ash, C, H, N, Cl, S, O)
            feed_comp[8],
            feed_comp[1],   # C_in (wt%)
            feed_comp[2],    # H_in
            feed_comp[3],    # N_in
            0,
            feed_comp[4],    # S_in
            feed_comp[5],   # O_in
            feed_comp[4],    # S_organic
        ])





    return x_input


def run_combustion_model(model_config, particle_position,feed_comp):
    """
    Run HTC model with Aspen + ANN hybrid setup.

    Args:
        model_config (dict): Configuration for the HTC model
        particle_position (list): Particle position values, including manipulated variables

    Returns:
        dict: {"y1": CO2_emission, "y2": revenue}
    """



    x_input = prepare_x_input(feed_comp)
    process_name = "combustion"


    # # Run Aspen model
    # results = run_simulation(process_name,x_input)
    #
    # return results

    # Run Aspen simulation
    raw_results = run_simulation(process_name, x_input)

    # Assume combustion feed basis is 100 kg/hr wet feed
    wet_feed_basis = 100.0

    electricity_kwh_hr = raw_results.get("electricity", 0.0)
    co2_kg_hr = raw_results.get("CO2_emission", 0.0)

    electricity_per_wet = round(electricity_kwh_hr / wet_feed_basis, 4)
    co2_per_wet = round(co2_kg_hr / wet_feed_basis, 4)

    return {
        "raw": {
            "electricity": round(electricity_kwh_hr, 4),
            "CO2_emission": round(co2_kg_hr, 4)
        },
        "products": {
            "electricity": electricity_per_wet # kWh per wet feed
        },
        "emissions": {
            "CO2_emission": co2_per_wet # kg CO2 per wet feed
        }
    }