import time
import numpy as np
import win32com.client
from config.config import FEEDS, MODEL_PATHS, TIME_CONST, SOLID_CONST
from optimization.aspen_paths import ASPEN_PATHS
from models.ann.prepare_direct_combustion_inputs import prepare_direct_combustion_inputs
from models.ann.prepare_htc_aspen_inputs import prepare_htc_aspen_inputs

def run_simulation(select_pathway, feed_name, T_HTC=None, char_split=None, verbose=False):
    assert select_pathway in ["htc", "direct"], "select_pathway must be 'htc' or 'direct'"
    feed_data = FEEDS.get(feed_name)
    if not feed_data:
        raise ValueError(f"Feedstock '{feed_name}' not found in config")

    feed_comp = feed_data["composition"]

    # === Prepare input data
    if select_pathway == "direct":
        input_dict = prepare_direct_combustion_inputs(feed_comp)
        apw_path = MODEL_PATHS["aspen_direct"]
    else:
        if T_HTC is None or char_split is None:
            raise ValueError("T_HTC and char_split must be specified for HTC pathway")
        htc_input = np.array([
            *feed_comp, T_HTC, TIME_CONST, SOLID_CONST, char_split
        ])
        input_dict = prepare_htc_aspen_inputs(htc_input)
        apw_path = MODEL_PATHS["aspen_htc"]

    # === Launch Aspen
    try:
        aspen = win32com.client.Dispatch("Apwn.Document")
        aspen.InitFromArchive2(str(apw_path))
    except Exception as e:
        raise RuntimeError(f"❌ Failed to launch/load Aspen: {e}")

    # === Apply input values
    for path, value in zip(input_dict["path"], input_dict["value"]):
        try:
            aspen.Tree.FindNode(path).Value = value
        except Exception as e:
            if verbose:
                print(f"⚠️ Could not set {path}: {e}")

    # === Run simulation
    try:
        aspen.Reinit()
        aspen.Engine.Run2(1)
        while aspen.Engine.IsRunning:
            time.sleep(0.2)
    except Exception as e:
        aspen.Close(False)
        raise RuntimeError(f"❌ Simulation failed: {e}")

    # === Extract outputs
    power = None
    char_flow = None
    try:
        power_path = ASPEN_PATHS[select_pathway]["outputs"]["power"]
        power = aspen.Tree.FindNode(power_path).Value

        if select_pathway == "htc":
            char_path = ASPEN_PATHS["htc"]["outputs"]["char"]
            char_flow = aspen.Tree.FindNode(char_path).Value

        if verbose:
            print(f"✅ Power: {power} kW")
            if char_flow is not None:
                print(f"✅ Char: {char_flow} kg/h")

    except Exception as e:
        if verbose:
            print(f"⚠️ Failed to extract results: {e}")

    aspen.Close(False)

    return {
        "power_kW": -1*power,
        "char_kgph": char_flow
    }
