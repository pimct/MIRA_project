# prepare parameters for HTC simulation in Aspen Plus.

import numpy as np
from engine.simulation.ann_predictor import ANNModelSklearn
from optimization.aspen_paths import ASPEN_PATHS

def prepare_htc_aspen_inputs(input):
    """
    # === Index mapping for input and processed feature vectors ===

    # input (model input):
    # [ 0] C_in         (wt%)
    # [ 1] H_in         (wt%)
    # [ 2] N_in         (wt%)
    # [ 3] S_in         (wt%)
    # [ 4] O_in         (wt%)
    # [ 5] VM_in        (%)
    # [ 6] FC_in        (%)
    # [ 7] Ash_in       (%)
    # [ 8] T_HTC        (°C)
    # [ 9] t_res        (h)
    # [10] Solid_load   (%)
    # [11] Char_split  (fraction of char routed to energy recovery, e.g. 0.5)

    # feed_prop (processed composition):
    # [ 0] Unused
    # [ 1] FC_in        (%)
    # [ 2] VM_in        (%)
    # [ 3] Ash_in       (%)
    # [ 4] Ash_in       (%)
    # [ 5] C_in         (wt%)
    # [ 6] H_in         (wt%)
    # [ 7] N_in         (wt%)
    # [ 8] Cl_in        (wt%) = 100 - (C + H + N + S + O)
    # [ 9] S_in         (wt%)
    # [10] O_in         (wt%)
    # [11] PYRITIC Sulfur (assumed 0)
    # [12] SULFATE Sulfur (assumed 0)
    # [13] ORGANIC Sulfur (assumed = S_in)

    Combines HTC prediction, post-processing, and Aspen parameter packaging.

    Returns:
        dict: {"path": [...], "value": [...]} for Aspen INP injection
    """
    # Create paths for Aspen input
    paths_config = ASPEN_PATHS["htc"]["inputs"]

    # Combine all path entries from categorized input sections
    paths = (
            paths_config["prox"]
            + paths_config["ult"]
            + paths_config["sulfur"]
            + [
                paths_config["flow"],
                paths_config["temp"],
                paths_config["solid_rate"],
                paths_config["gas_C"],
                paths_config["gas_H2"],
                paths_config["gas_N2"],
            ]
            + paths_config["char_prox"]
            + paths_config["char_ult"]
            + paths_config["org_prox"]
            + paths_config["org_ult"]
            + [
                paths_config["char_sulfur"],
                paths_config["org_sulfur"],
                paths_config["char_split"],
            ]
    )


    # Ensure input is a numpy array
    x_input = input[:-1]
    ann_input = x_input
    x_char_split = input[-1]
    # Step 0: load ANN model
    ann_model = ANNModelSklearn()
    # Step 1: Predict ANN outputs
    ann_output = ann_model.predict(ann_input)

    # Step 2: Post-process to mass-consistent HTC result


    # Step 3: Prepare full Aspen input vector
    # Extract ash and calculate scaling factor
    ash = x_input[7]
    scale = (100 - ash) / 100
    x_input[0:5] = x_input[0:5] * scale
    htc_result = htc_ann_postprocess(x_input, ann_output)

    feed_prop = [
        0,                     # Unused
        x_input[6],            # FC
        x_input[5],            # VM
        x_input[7],            # Ash (proximate
        x_input[7],            # Ash (ultimate)
        x_input[0],            # C
        x_input[1],            # H
        x_input[2],            # N
        100 - sum(x_input[i] for i in [0, 1, 2, 3, 4]),  # Cl = 100 - (C + H + N + S + O)
        x_input[3],            # S
        x_input[4],            # O
        0,                     # Unused
        0,                     # Unused
        x_input[3]             # S again
    ]

    htc_sim = [
        x_input[10],                          # Solid load (%)
        x_input[8],                           # HTC temp (°C)
        x_input[8],                           # HTC temp (°C) duplicated
        htc_result["mass_yields"]["char"],   # Char yield (kg/kg-wet)
        htc_result["mass_yields"]["organics"],  # Organics yield (kg/kg-wet)
        htc_result["mass_yields"]["water"],     # Water yield (kg/kg-wet)

        *htc_result["char_prox"],            # Moisture, Volatile, FixedC, Ash (char)
        *htc_result["char_ult"],             # C, H, O, N, Cl, S, Ash (char)
        *htc_result["org_prox"],             # Moisture, Volatile, FixedC, Ash (organics)
        *htc_result["org_ult"],              # C, H, O, N, Cl, S, Ash (organics)

        htc_result["char_ult"][5],           # S in char
        htc_result["org_ult"][5]             # S in organics
    ]


    values = feed_prop + htc_sim + [x_char_split]

    return {
        "path": paths,
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
    charyield, C, H, O, N, S, volatile, fixedC, ash = y
    HHV = np.dot(np.array([C, H,N,S,O,ash]),np.array([0.3491, 1.1783,0.1005,-0.1034,-0.0151,-0.0211])) # Ref: Channiwala SA, 2002
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