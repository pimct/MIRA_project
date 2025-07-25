import importlib
import os
import yaml
import numpy as np

# === Auto-detect and load postprocessor if model exists ===
def detect_postprocessor(process_name):
    """
    Auto-detect and load a postprocessor class for a given process.

    Returns:
        postprocessor class or None
    """
    # Update model_dir to reflect the correct path in the main project directory
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ann_models", process_name))
    postprocess_file = os.path.join(model_dir, "ann_postprocess.py")

    # Check if the postprocessor file exists
    if not os.path.exists(postprocess_file):
        print(f"⚠️ Postprocessor file not found for '{process_name}': {postprocess_file}")
        return None

    try:
        module_path = f"ann_models.{process_name}.ann_postprocess"
        class_name = f"PostProcessor"
        module = importlib.import_module(module_path)
        post_class = getattr(module, class_name)
        return post_class
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"⚠️ ANN postprocessor not found for '{process_name}': {e}")
        return None


def load_path_mapping(process_name):
    """Load Aspen path mappings from YAML."""
    path_file = os.path.join("aspen_models", process_name, f"{process_name}_paths.yaml")
    if not os.path.exists(path_file):
        raise FileNotFoundError(f"❌ Aspen path mapping not found for process '{process_name}'")
    with open(path_file, "r") as f:
        return yaml.safe_load(f)


# === Prepare inputs for Aspen (use ANN if available) ===
def prepare_aspen_inputs(process_name, x_input):
    """
    Generic Aspen input preparation. Uses ANN postprocessor if detected.

    Args:
        process_name (str): e.g., "htc", "combustion"
        x_input (np.ndarray): input vector

    Returns:
        dict: {"path": [...], "value": [...]}
    """
    path_map = load_path_mapping(process_name)
    input_paths = path_map.get("input_paths")
    if input_paths is None:
        raise KeyError(f"No 'input_paths' found in {process_name}_paths.yaml")

    # Try to use ANN postprocessor
    post_class = detect_postprocessor(process_name)
    if post_class:
        print(f"🤖 Using ANN postprocessor for '{process_name}'")
        processor = post_class()
        result = processor.predict_and_postprocess(x_input)
        values = result["values"]
    else:
        print(f"🔁 No ANN postprocessor found for '{process_name}', using raw input.")



        values = x_input.tolist()

    return {
        "path": input_paths,
        "value": values
    }

# === Prepare output paths for Aspen ===
def prepare_aspen_outputs(process_name):
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

