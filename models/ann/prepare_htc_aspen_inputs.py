from models.ann.ann_predictor import ANNModelSklearn

def generate_htc_parameter_paths():
    paths = []
    for i in range(4):
        paths.append(f"\\Data\\Streams\\FEED\\Input\\ELEM\\NCPSD\\WASTES\\PROXANAL\\#{i}")
    for i in range(7):
        paths.append(f"\\Data\\Streams\\FEED\\Input\\ELEM\\NCPSD\\WASTES\\ULTANAL\\#{i}")
    for i in range(3):
        paths.append(f"\\Data\\Streams\\FEED\\Input\\ELEM\\NCPSD\\WASTES\\SULFANAL\\#{i}")
    paths.append("\\Data\\Streams\\FEED\\Input\\FLOW\\NCPSD\\WASTES")
    paths.append("\\Data\\Blocks\\HTC\\Input\\TEMP")
    paths.append("\\Data\\Blocks\\STMHX1\\Input\\VALUE")
    paths.append("\\Data\\Streams\\DUMMY1\\Input\\FLOW\\MIXED\\C")
    paths.append("\\Data\\Streams\\DUMMY1\\Input\\FLOW\\MIXED\\H2")
    paths.append("\\Data\\Streams\\DUMMY1\\Input\\FLOW\\MIXED\\N2")
    for i in range(4):
        paths.append(f"\\Data\\Blocks\\HTC\\Input\\ELEM\\NCPSD\\CHAR\\PROXANAL\\#{i}")
    for i in range(7):
        paths.append(f"\\Data\\Blocks\\HTC\\Input\\ELEM\\NCPSD\\CHAR\\ULTANAL\\#{i}")
    for i in range(4):
        paths.append(f"\\Data\\Blocks\\HTC\\Input\\ELEM\\NCPSD\\ORGANICS\\PROXANAL\\#{i}")
    for i in range(7):
        paths.append(f"\\Data\\Blocks\\HTC\\Input\\ELEM\\NCPSD\\ORGANICS\\ULTANAL\\#{i}")
    paths.append("\\Data\\Blocks\\HTC\\Input\\ELEM\\NCPSD\\CHAR\\SULFANAL\\#2")
    paths.append("\\Data\\Blocks\\HTC\\Input\\ELEM\\NCPSD\\ORGANICS\\SULFANAL\\#2")
    paths.append("\\Data\\Blocks\\X\\Input\\FRAC\\CHARPROD")
    return paths

def prepare_htc_aspen_inputs(x_input, ann_models, solid_feed_rate, temp, x_char_split):
    """
    Combines HTC prediction, post-processing, and Aspen parameter packaging.

    Args:
        x_input (list or np.array): Input vector (11 features)
        ann_models (dict): Dictionary of trained ANN models
        solid_feed_rate (float): Feed flowrate (kg/hr)
        temp (float): HTC temperature (°C)
        x_char_split (float): Fraction of char routed to energy recovery

    Returns:
        dict: {"path": [...], "value": [...]} for Aspen INP injection
    """
    # Step 1: Predict ANN outputs
    ann_output = ANNModelSklearn(x_input, ann_models)

    # Step 2: Post-process to mass-consistent HTC result
    htc_result = htc_ann_postprocess(x_input, ann_output)

    # Step 3: Prepare full Aspen input vector
    feed_prop = [
        0,
        x_input[6],
        x_input[5],
        x_input[7],
        x_input[7],
        x_input[0],
        x_input[1],
        x_input[2],
        0,
        x_input[3],
        x_input[4],
        0.2,
        0,
        0
    ]

    htc_sim = [
        solid_feed_rate,
        temp,
        temp,
        htc_result['char_yield'],
        htc_result['org_yield'],
        htc_result['water_yield'],
        *htc_result['char_prox'],
        *htc_result['char_ult'],
        *htc_result['org_prox'],
        *htc_result['org_ult'],
        htc_result['char_ult'][5],   # S in char
        htc_result['org_ult'][5]     # S in organics
    ]

    values = feed_prop + htc_sim + [x_char_split]

    return {
        "path": generate_htc_parameter_paths(),
        "value": values
    }

def htc_ann_postprocess(x_input, y_ann_output):
    """
    Post-process ANN outputs for HTC product stream calculation.
    Ensures mass and elemental balances are closed for Aspen simulation.

    Parameters:
    - x_input: ndarray of shape (11,) → input vector used for ANN (including HTC temp, feed comp, etc.)
    - y_ann_output: ndarray of shape (10,) → ANN outputs: [CharYield, C, H, O, N, S, Volatile, FixedC, Ash, HHV]

    Returns:
    - yChar: [Yield, C, H, O, S, Ash, Vol, FC]
    - yOrg:  [Yield, C, H, O, S, Ash, Vol, FC]
    - char_ult_final: [C, H, O, N, Cl, S, Ash]
    - char_prox_final: [Moisture, Volatile, FixedC, Ash]
    - org_ult_final: same structure
    - org_prox_final: same structure
    - mass_yields: dict with 'char', 'org', and 'water' (kg/kg-wet)
    """

    x = np.array(x_input).flatten()
    y = np.array(y_ann_output).flatten()

    # === Extract ANN outputs ===
    charyield, C, H, O, N, S, volatile, fixedC, ash, HHV = y
    y_htc = np.array([charyield, C, H, O, N, S, volatile, fixedC, ash])
    y_htc[y_htc < 0] = 0
    if y_htc[0] > 100:
        y_htc[0] = 99

    # === Dry feed basis
    mass_dryfeed = x[10] / 100
    moisture_char = 0
    charyield = y_htc[0] * mass_dryfeed / (100 - moisture_char)
    mass_drychar = charyield * (100 - moisture_char) / 100

    # === Ultimate analysis
    feed_cl = 100 - sum([x[7], x[0], x[1], x[2], 0, x[3], x[4]])
    feed_ult = np.array([x[7], x[0], x[1], x[2], feed_cl, x[3], x[4]])
    char_ult = np.array([y_htc[8], y_htc[1], y_htc[2], y_htc[3], 0, y_htc[4], y_htc[5]])

    if np.sum(char_ult) > 100:
        char_ult[1:] *= (100 - char_ult[0]) / np.sum(char_ult[1:])

    feed_element = mass_dryfeed * feed_ult / 100
    char_element = mass_drychar * char_ult / 100
    diff = feed_element - char_element

    if np.any(diff < 0):
        char_element[diff < 0] = feed_element[diff < 0]
        char_ult = char_element * 100 / np.sum(char_element)
        mass_drychar = np.sum(char_element)

    prod_charyield = mass_drychar * 100 / (100 - moisture_char)
    char_prox = [moisture_char,
                 y_htc[6] * (100 - char_ult[0]) / np.sum(y_htc[6:8]),
                 y_htc[7] * (100 - char_ult[0]) / np.sum(y_htc[6:8]),
                 char_ult[0]]

    org_element = feed_element - mass_drychar * char_ult / 100
    org_ult = np.clip(org_element * 100 / np.sum(org_element), 0, None)
    org_prox = [0, 0, 100 - org_ult[0], org_ult[0]]

    yChar = [prod_charyield * 100 * 100 / x[10]] + char_ult[[1, 2, 3, 5, 6]].tolist() + char_prox[1:]
    yOrg = [(x[10] - prod_charyield * 100) * 100 / x[10]] + org_ult[[1, 2, 3, 5, 6]].tolist() + org_prox[1:]

    # Final rearranged output
    char_prox_final = [0, yChar[7], yChar[6], yChar[8]]
    char_ult_final = [yChar[8], yChar[1], yChar[2], yChar[3], 0, yChar[4], yChar[5]]
    org_prox_final = [0, yOrg[7], yOrg[6], yOrg[8]]
    org_ult_final = [yOrg[8], yOrg[1], yOrg[2], yOrg[3], 0, yOrg[4], yOrg[5]]

    # Yields (kg/kg-wet-feed)
    feed_massflow = x[10]
    char_yield = yChar[0] * feed_massflow / 100 / 100
    org_yield = feed_massflow / 100 - char_yield
    water_yield = 1 - feed_massflow / 100

    mass_yields = {
        "char": round(char_yield, 6),
        "organics": round(org_yield, 6),
        "water": round(water_yield, 6)
    }

    return {
        "yChar": yChar,
        "yOrg": yOrg,
        "char_ult": char_ult_final,
        "char_prox": char_prox_final,
        "org_ult": org_ult_final,
        "org_prox": org_prox_final,
        "mass_yields": mass_yields,
        "HHV": HHV  # Optional: may be used downstream
    }