import time
import os
import win32com.client
from engine.simulation.prepare_paths import prepare_aspen_inputs, prepare_aspen_outputs

def safe_set(aspen, path, value, verbose=False, index=None):
    node = aspen.Tree.FindNode(path)
    if node is None:
        if verbose:
            prefix = f"[{index:02d}]" if index is not None else ""
            print(f"❌ {prefix} Node not found: {path}")
        return False
    try:
        node.Value = value
        return True
    except Exception as e:
        if verbose:
            prefix = f"[{index:02d}]" if index is not None else ""
            print(f"⚠️ {prefix} Failed to set {path} = {value}: {e}")
        return False

def run_simulation(process_name, x_input, verbose=True, visible=False):
    """
    Run Aspen simulation for a given process using Aspen Plus .apw file.

    Args:
        process_name (str): Process name (e.g., "htc")
        x_input (np.ndarray): Input data (from ANN or direct)
        verbose (bool): Whether to print logs
        visible (bool): Whether to show Aspen GUI

    Returns:
        dict: Simulation output results {name: value}
    """
    # === Path to .apw file ===
    apw_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "aspen_models", process_name, f"{process_name}.apw"))
    # if verbose:
    #     print(f"📦 Aspen file path: {apw_path}")

    # === Prepare input values ===
    input_dict = prepare_aspen_inputs(process_name, x_input)
    paths = input_dict["path"]
    values = input_dict["value"]

    if len(paths) != len(values):
        raise ValueError("❌ Input path/value length mismatch.")

    # Log non-numeric values
    for i, (p, v) in enumerate(zip(paths, values)):
        if not isinstance(v, (int, float)):
            print(f"⚠️ Non-numeric value at index {i}: {p} = {v}")

    # === Launch Aspen ===
    try:
        aspen = win32com.client.Dispatch("Apwn.Document")
        if visible:
            aspen.Visible = True
        aspen.InitFromArchive2(str(apw_path))
        time.sleep(3)  # Allow Aspen to initialize
    except Exception as e:
        raise RuntimeError(f"❌ Failed to launch Aspen: {e}")

    # === Apply inputs with safety check ===
    for i, (path, value) in enumerate(zip(paths, values)):
        safe_set(aspen, path, value, verbose=verbose, index=i)

    # === Run simulation with robust wait logic ===
    try:
        aspen.Reinit()
        aspen.Engine.Run2(1)

        # Wait for Aspen to start running (in case it's too fast)
        timeout_start = time.time()
        while not aspen.Engine.IsRunning and time.time() - timeout_start < 3:
            time.sleep(0.1)

        # Wait for Aspen to finish
        timeout_start = time.time()
        while aspen.Engine.IsRunning and time.time() - timeout_start < 60:
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
            # if verbose:
            #     print(f"✅ {name}: {value}")
        except Exception as e:
            result[name] = None
            if verbose:
                print(f"⚠️ Failed to read {name} from {path}: {e}")

    aspen.Save()
    aspen.Close(False)
    return result
