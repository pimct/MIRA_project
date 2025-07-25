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

    feed_comp = np.array(feed_comp)    # Validate input dimensions
    feed_flow = 100 - feed_comp[0]


    if len(feed_comp) != 9:
        raise ValueError("feed_comp must have exactly 10 elements")
    # Sample HTC input vector
    #     x_input = np.array([
    #         100 - Moisture,   # flow waste
    #         55,     # MOISTURE (wt%)
    #         53.4,   # C_in (wt%)
    #         6.2,    # H_in
    #         3.0,    # N_in
    #         0.3,    # S_in
    #         37.1,   # O_in
    #         54.6,   # VM_in (%)
    #         9.6,    # FC_in
    #         35.8,   # Ash_in (%)
    #     ])
    #
    #     result = prepare_aspen_inputs("htc", x_input)
    x_input = np.array([
        feed_flow,
        *feed_comp,
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


    # Run Aspen model
    results = run_simulation(process_name,x_input)

    return results
