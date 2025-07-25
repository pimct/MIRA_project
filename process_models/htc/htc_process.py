#import os
#import json
import numpy as np

#from engine.simulation.prepare_paths import prepare_aspen_inputs, prepare_aspen_outputs
from engine.simulation.hybrid_runner import run_simulation
#from engine.simulation.hybrid_runner import AspenRunnerManager


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
    # Ensure feed_comp is a numpy array for consistency

    feed_comp = np.array(feed_comp[1:])    # Validate input dimensions

    if len(feed_comp) != 8:
        raise ValueError("feed_comp must have exactly 10 elements")
# Sample HTC input vector
#     x_input = np.array([
#         53.4,   # C_in (wt%)
#         6.2,    # H_in
#         3.0,    # N_in
#         0.3,    # S_in
#         37.1,   # O_in
#         54.6,   # VM_in (%)
#         9.6,    # FC_in
#         35.8,   # Ash_in (%)
#         260.0,  # HTC temperature (°C)
#         2.0,    # Residence time (hr)
#         30.0,   # Solid loading (wt%)
#         0.5     # Char split fraction
#     ])
#
#     result = prepare_aspen_inputs("htc", x_input)
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

    #runner = AspenRunnerManager(process_names=["htc", "combustion"])

    # Debug one simulation visually
    # results = runner.run_simulation(process_name, x_input=None, visible=False)
    # print(results)
    # print("\n📊 Simulation Results:")
    # for key, value in results.items():
    #     print(f"🔹 {key}: {value}")

    #runner.close_all()


# Extract results (y1 = CO2, y2 = revenue)
    return results

