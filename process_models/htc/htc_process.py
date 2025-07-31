import numpy as np
from engine.simulation.hybrid_runner import run_simulation
from ann_models.htc.ann_postprocess import PostProcessor


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
    Results normalized to 1 kg wet feed.

    Returns:
        dict: {
            "raw": {...},
            "products": {...},
            "emissions": {...}
        }
    """

    _, temp, x_char_split = particle_position
    # Display warning if x_char_split is not valid
    if x_char_split > 1.0:
        print(f"⚠️ Warning: x_char_split = {x_char_split:.3f} is > 1. This may be invalid.")


    time = model_config["CONSTANTS"]["time_constant"]
    solid_loading = model_config["CONSTANTS"]["solid_loading"]
    moisture = feed_comp[0]  # wt% moisture

    x_input = prepare_x_input(feed_comp, temp, x_char_split, time, solid_loading)
    process_name = "htc"


    # # Run Aspen model
    # results = run_simulation(process_name,x_input)
    #
    #
    # return results
    #

    # === Run Aspen simulation
    raw_results = run_simulation(process_name, x_input)

    # === Post-process: char HHV
    post = PostProcessor(process_name)
    post_result = post.predict_and_postprocess(x_input)
    char_hhv = post_result["postprocessed"]["char_HHV"]
    print(f"Char HHV: {char_hhv:.3f} MJ/kg")

    # === Normalize basis: per 1 kg wet feed
    wet_feed_flowrate = solid_loading / (1 - moisture / 100)

    char_kg_hr = raw_results.get("char", 0.0)
    elec_kwh_hr = raw_results.get("electricity", 0.0)
    co2_kg_hr = raw_results.get("CO2_emission") or 1000.0 # Penalize missing CO2 emission data (run crash)

    char_kg_per_kg_wet = char_kg_hr / wet_feed_flowrate
    char_heat_per_kg_wet = round(char_kg_per_kg_wet * char_hhv, 3)
    elec_kwh_per_kg_wet = round(elec_kwh_hr / wet_feed_flowrate, 4)
    co2_per_kg_wet = round(co2_kg_hr / wet_feed_flowrate, 4)

    return {
        "raw": {
            "char": round(char_kg_hr, 4),
            "electricity": round(elec_kwh_hr, 4),
            "CO2_emission": round(co2_kg_hr, 4)
        },
        "products": {
            "char": char_heat_per_kg_wet,      # MJ per kg wet feed
            "electricity": elec_kwh_per_kg_wet # kWh per kg wet feed
        },
        "emissions": {
            "CO2_emission": co2_per_kg_wet # kg CO2 per kg wet feed
        }
    }