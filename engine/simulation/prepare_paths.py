import os
import yaml
import numpy as np
from ann_models.htc.htc_model import HTCPostProcessor  # example HTC handler

# Optional: map processes to postprocessors (HTC-specific logic)
POSTPROCESSORS = {
    "htc": HTCPostProcessor,  # must implement `.predict_and_postprocess(x_input)`
    # Add other process classes in the future
}


def load_path_mapping(process_name):
    """Load Aspen path mappings from YAML."""
    path_file = os.path.join("aspen_models", process_name, f"{process_name}_paths.yaml")
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"❌ Aspen path mapping not found for process '{process_name}'")
    with open(path_file, "r") as f:
        return yaml.safe_load(f)


def prepare_aspen_inputs(process_name, x_input):
    """
    Generic Aspen input preparation for any supported process.
    If the process uses an ANN model, it will apply post-processing.

    Args:
        process_name (str): e.g., "htc", "combustion"
        x_input (np.ndarray): model input or direct Aspen input

    Returns:
        dict: {"path": [...], "value": [...]}
    """
    path_map = load_path_mapping(process_name)
    path_dict = path_map.get("input_paths")
    if path_dict is None:
        raise KeyError(f"No 'input_paths' found in {process_name}_paths.yaml")

    input_paths = path_map.get("input_paths")
    if input_paths is None:
        raise KeyError(f"No 'input_paths' found in {process_name}_paths.yaml")
    # === Use ANN if postprocessor exists ===
    if process_name in POSTPROCESSORS:
        processor = POSTPROCESSORS[process_name]()
        result = processor.predict_and_postprocess(x_input)
        values = result["values"]
    else:
        values = x_input.tolist()

    # Final format
    return {
        "path": input_paths,
        "value": values
    }

def prepare_aspen_outputs(process_name):
    """
    Load and return the Aspen output paths for a given process.

    Returns:
        dict: {"path": [...], "name": [...]}
    """
    path_map = load_path_mapping(process_name)
    output_paths = path_map.get("output_paths")
    if output_paths is None:
        raise KeyError(f"No 'output_paths' found in {process_name}_paths.yaml")

    flat_paths = []
    names = []
    for key, value in output_paths.items():
        if isinstance(value, list):
            flat_paths.extend(value)
            names.extend([f"{key}_{i}" for i in range(len(value))])
        else:
            flat_paths.append(value)
            names.append(key)

    return {
        "path": flat_paths,
        "name": names
    }