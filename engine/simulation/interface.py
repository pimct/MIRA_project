import importlib
def run_simulation(process_name, model_config, x_input, feed_comp, test_mode=False):
    if test_mode:
        from engine.simulation.mock_runner import run_simulation
    else:
        module_path = f"process_models.{process_name}.{process_name}_process"
        process_module = importlib.import_module(module_path)
        run_simulation = getattr(process_module, f"run_{process_name}_model")
    return run_simulation(model_config, x_input, feed_comp)
