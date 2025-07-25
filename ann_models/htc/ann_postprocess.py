import numpy as np
from engine.simulation.ann_predictor import ANNModelSklearn

# def atom_bal(feed_ultimate: dict, char_predicted: dict, char_yield: float, solid_loading: float):
#     """
#     Atom balance: char + organic = solid_loading (dry mass basis), excludes water.
#
#     Args:
#         feed_ultimate (dict): Elemental wt% of feed (C, H, O, etc.).
#         char_predicted (dict): Elemental wt% of ANN-predicted char.
#         char_yield (float): Char mass yield (% of dry feed).
#         solid_loading (float): Total dry solids input (e.g. 20 means 20 wt% solids, 80 wt% water).
#
#     Returns:
#         tuple: (char_balanced_wt%, organic_balanced_wt%)
#     """
#     atomic_weights = {
#         'CARBON': 12.01,
#         'HYDROGEN': 1.008,
#         'OXYGEN': 16.00,
#         'NITROGEN': 14.01,
#         'SULFUR': 32.06,
#         'CHLORINE': 35.45
#     }
#
#     element_order = ['ASH', 'CARBON', 'HYDROGEN', 'NITROGEN', 'CHLORINE', 'SULFUR', 'OXYGEN']
#     atom_elements = [el for el in element_order if el != 'ASH']
#
#     def wt_to_mol(wt_dict, total_mass):
#         return {el: (wt_dict[el] / 100) * total_mass / atomic_weights[el] for el in atom_elements}
#
#     def mol_to_wt(mol_dict):
#         return {el: mol_dict[el] * atomic_weights[el] for el in atom_elements}
#
#     # --- Define total dry input mass as solid_loading (e.g., 20 means 20 g solid + 80 g water) ---
#     feed_mass = 100
#     char_mass = char_yield*feed_mass/100 # char_yield value is 100% basis, so we need to divide by 100
#
#     # --- Convert wt% to moles ---
#     feed_mol = wt_to_mol(feed_ultimate, feed_mass)
#     char_mol = wt_to_mol(char_predicted, char_mass)
#
#     # --- Limit char to atom availability ---
#     ratios = {el: feed_mol[el] / char_mol[el] if char_mol[el] > 0 else 1.0 for el in atom_elements}
#     min_ratio = min(1.0, min(ratios.values()))
#
#     # --- Scale char atoms down if needed ---
#     char_mol_corrected = {el: char_mol[el] * min_ratio for el in atom_elements}
#     char_wt_corrected = mol_to_wt(char_mol_corrected)
#
#     # --- Organic = remaining atoms ---
#     organic_mol = {el: feed_mol[el] - char_mol_corrected[el] for el in atom_elements}
#     organic_wt = mol_to_wt(organic_mol)
#
#     # --- Normalize to wt% (of each product mass) ---
#     char_total = sum(char_wt_corrected.values())
#     org_total = sum(organic_wt.values())
#
#     char_normalized = {el: char_wt_corrected.get(el, 0.0) / char_total * 100 for el in element_order}
#     organic_normalized = {el: organic_wt.get(el, 0.0) / org_total * 100 for el in element_order}
#
#     return char_normalized, organic_normalized

def atom_bal(feed_ultimate: dict, char_predicted: dict, char_yield: float, solid_loading: float):
    """
    Atom balance using mass-based logic inspired by simulate_htc.
    Returns char and organic elemental compositions (wt%).

    Args:
        feed_ultimate (dict): Elemental wt% of feed (C, H, O, etc.).
        char_predicted (dict): Elemental wt% of predicted char.
        char_yield (float): Char yield (% of dry feed).
        solid_loading (float): Feed wt% dry basis (e.g., 20 means 20 g solid per 100 g total feed).

    Returns:
        tuple: (char_balanced_wt%, organic_balanced_wt%)
    """
    # === Atomic elements order and feed/char arrays ===
    element_order = ['ASH', 'CARBON', 'HYDROGEN', 'NITROGEN', 'CHLORINE', 'SULFUR', 'OXYGEN']

    # Convert feed to array
    feed_vec = np.array([feed_ultimate[el] for el in element_order])
    char_vec = np.array([char_predicted[el] for el in element_order])

    # === Mass basis: dry feed mass fraction ===
    mass_dry_feed = solid_loading / 100.0  # e.g., 20/100 = 0.2
    mass_dry_char = char_yield * mass_dry_feed / 100.0  # e.g., 40% of dry = 0.08

    # === Atom balance logic from simulate_htc ===
    feed_element = mass_dry_feed * feed_vec / 100.0
    char_element = mass_dry_char * char_vec / 100.0
    diff = feed_element - char_element

    # If any negative, truncate char_element to match feed supply
    if np.any(diff < 0):
        char_element[diff < 0] = feed_element[diff < 0]
        # Normalize char composition again
        total_char_mass = np.sum(char_element)
        char_vec = char_element * 100 / total_char_mass
        mass_dry_char = total_char_mass

    # Recalculate organics by difference
    org_element = feed_element - char_element
    total_org_mass = np.sum(org_element)
    org_vec = org_element * 100 / total_org_mass if total_org_mass > 0 else np.zeros_like(org_element)

    # Repack into dicts
    char_bal = {el: char_vec[i] for i, el in enumerate(element_order)}
    org_bal = {el: org_vec[i] for i, el in enumerate(element_order)}

    return char_bal, org_bal


class PostProcessor:
    def __init__(self, process_name="htc"):
        self.model = ANNModelSklearn(process_name)
        self.process_name = process_name

    def predict_and_postprocess(self, x_input):
        # === ANN Prediction ===
        ann_output = self.model.predict(x_input[:-1])

        # === Parse ANN output ===
        char_yield = ann_output[0]
        char_prox = [0, ann_output[7], ann_output[6], ann_output[8]]  # Moisture, FC, VM, Ash
        char_ult = {
            "CARBON": ann_output[1],
            "HYDROGEN": ann_output[2],
            "NITROGEN": ann_output[3],
            "SULFUR": ann_output[4],
            "OXYGEN": ann_output[5],
            "CHLORINE": 0.0,
            "ASH": ann_output[8]
        }

        # === Input Features ===
        solid_loading = x_input[10]  # feed wt% on dry basis
        x_char_split = x_input[-1]   # char split (decision var)

        feed_ultimate = {
            "CARBON": x_input[0],
            "HYDROGEN": x_input[1],
            "NITROGEN": x_input[2],
            "SULFUR": x_input[3],
            "OXYGEN": x_input[4],
            "ASH": x_input[7],
            "CHLORINE": 0.0
        }

        # === Atom balance ===
        char_bal, org_bal = atom_bal(feed_ultimate, char_ult, char_yield, solid_loading)
        # === Mass yields based on corrected atom_bal2 ===
        # Total feed mass basis = 100 g
        mass_dry_feed = solid_loading                          # g (dry solid part)
        mass_dry_char = char_yield * mass_dry_feed / 100       # g
        mass_dry_organics = mass_dry_feed - mass_dry_char      # g
        mass_water = 100 - solid_loading                       # g

        char_yield = solid_loading* char_yield / 100.0         # Convert to mass basis (e.g., 20% feed means 20 g total, so char yield is 20 g * char_yield/100)
        organic_yield = mass_dry_organics                      # g
        water_yield = mass_water                               # g

        # === Values in order matching htc_path.yaml ===
        values = []

        # --- Feed PROX (Moisture, FC, VM, Ash) ---
        values += [0, x_input[6], x_input[5], x_input[7]]

        # --- Feed ULT (ASH first) ---
        values += [x_input[7], x_input[0], x_input[1], x_input[2], 0.0, x_input[3], x_input[4]]

        # --- Sulfur Analysis (organic sulfur) ---
        values += [x_input[3]]

        # --- Feed Flowrate and HTC Temp and  HX Temp Input  ---
        values += [x_input[10], x_input[8], x_input[8]]

        # --- Dummy Gas Stream Yields ---
        values += [char_yield, organic_yield, water_yield]

        # --- Char PROX ---
        values += char_prox

        # --- Char ULT (ASH first) ---
        values += [
            char_bal["ASH"],
            char_bal["CARBON"],
            char_bal["HYDROGEN"],
            char_bal["NITROGEN"],
            char_bal["CHLORINE"],
            char_bal["SULFUR"],
            char_bal["OXYGEN"],
        ]
        # --- Char sulfur (organic)
        values += [char_bal["SULFUR"]]
        # --- Organic PROX (placeholders if not modeled) ---
        values +=  [0.0, 0.0, 100, 0.0] # Moisture, FC, VM, Ash


        # --- Organic ULT (ASH first as 0.0) ---
        values += [
            org_bal["ASH"],
            org_bal["CARBON"],
            org_bal["HYDROGEN"],
            org_bal["NITROGEN"],
            org_bal["CHLORINE"],
            org_bal["SULFUR"],
            org_bal["OXYGEN"],
        ]

        # --- Organic sulfur (organic)
        values += [org_bal["SULFUR"]]

        # --- Char Split (x_char_split) ---
        values += [x_char_split]

        return {
            "values": values,
            "postprocessed": {
                "mass_yields": {
                    "char": char_yield,
                    "organics": organic_yield,
                    "water": water_yield
                },
                "char_bal": char_bal,
                "org_bal": org_bal
            }
        }
