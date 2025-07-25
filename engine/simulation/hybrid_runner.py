import time
import os
import win32com.client
from engine.simulation.prepare_paths import prepare_aspen_inputs, prepare_aspen_outputs


def run_simulation(process_name, x_input, verbose=True, visible=False):
    """
    Run Aspen simulation for a given process using Aspen Plus .apw file.

    Args:
        process_name (str): Process name (e.g., "htc")
        x_input (np.ndarray): Input data (from ANN or direct)
        verbose (bool): Whether to print logs

    Returns:
        dict: Simulation output results {name: value}
    """
    # === Path to .apw file ===
    apw_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "aspen_models", process_name, f"{process_name}.apw"))
    if verbose:
        print(f"📦 Aspen file path: {apw_path}")

    # === Prepare input values ===
    input_dict = prepare_aspen_inputs(process_name, x_input)

    # === Launch Aspen ===
    try:
        aspen = win32com.client.Dispatch("Apwn.Document")
        aspen.InitFromArchive2(str(apw_path))
    except Exception as e:
        raise RuntimeError(f"❌ Failed to launch Aspen: {e}")

    # === Apply inputs ===
    for path, value in zip(input_dict["path"], input_dict["value"]):
        try:
            aspen.Tree.FindNode(path).Value = value
        except Exception as e:
            if verbose:
                print(f"⚠️ Could not set {path} = {value}: {e}")

    # === Run simulation ===
    try:
        aspen.Reinit()
        aspen.Engine.Run2(1)
        while aspen.Engine.IsRunning:
            time.sleep(0.2)
    except Exception as e:
        aspen.Close(False)
        raise RuntimeError(f"❌ Simulation failed: {e}")

    # === Extract outputs ===
    output_dict = prepare_aspen_outputs(process_name)
    result = {}
    for name, path in zip(output_dict["name"], output_dict["path"]):
        try:
            value = aspen.Tree.FindNode(path).Value
            result[name] = value
            if verbose:
                print(f"✅ {name}: {value}")
        except Exception as e:
            result[name] = None
            if verbose:
                print(f"⚠️ Failed to read {name} from {path}: {e}")

    aspen.Close(False)
    return result


