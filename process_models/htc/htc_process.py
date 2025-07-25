import numpy as np
from engine.simulation.hybrid_runner import run_simulation



def prepare_x_input(feed_comp,temp, x_char_split, time, solid_loading):
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
    feed_comp = np.array(feed_comp[1:])    # Validate input dimensions
    x_input = np.array([
        *feed_comp,
        temp,                # HTC temperature (°C)
        time,                 # Residence time (hr) - placeholder, adjust as needed
        solid_loading,                # Solid loading (wt%) - placeholder, adjust as needed
        x_char_split         # Char split fraction
    ])
    return x_input



def run_htc_model(model_config, particle_position,feed_comp):
    """
    Run HTC model with Aspen + ANN hybrid setup.

    Args:
        model_config (dict): Configuration for the HTC model
        particle_position (list): Particle position values, including manipulated variables

    Returns:
        dict: {"y1": CO2_emission, "y2": revenue}
    """

    _, temp, x_char_split = particle_position
    print(temp)
    print(x_char_split)
    # Display warning if x_char_split is not valid
    if x_char_split > 1.0:
        print(f"⚠️ Warning: x_char_split = {x_char_split:.3f} is > 1. This may be invalid.")


    time = model_config["CONSTANTS"]["time_constant"]
    solid_loading = model_config["CONSTANTS"]["solid_loading"]

    x_input = prepare_x_input(feed_comp, temp, x_char_split, time, solid_loading)
    process_name = "htc"


    # Run Aspen model
    results = run_simulation(process_name,x_input)

    return results

