import numpy as np
from engine.simulation.ann_predictor import ANNModelSklearn



def atom_bal(feed_ultimate: dict, char_predicted: dict, char_yield: float):
    """
    Perform atom balance correction on ANN-predicted char composition (with char yield),
    and compute remaining atoms as an 'organic' product.

    Args:
        feed_ultimate (dict): Feed ultimate analysis (wt%) including ASH (assumed 1g basis).
        char_predicted (dict): ANN-predicted char composition (wt%) including ASH (per g char).
        char_yield (float): Fraction of char mass per 1g of feed (e.g., 0.35 for 35%).

    Returns:
        tuple: (char_composition, organic_composition), both as ordered dicts (wt%)
    """

    # Atomic weights (g/mol)
    atomic_weights = {
        'CARBON': 12.01,
        'HYDROGEN': 1.008,
        'OXYGEN': 16.00,
        'NITROGEN': 14.01,
        'SULFUR': 32.06,
        'CHLORINE': 35.45
    }

    # Aspen-compatible order
    element_order = ['ASH', 'CARBON', 'HYDROGEN', 'NITROGEN', 'CHLORINE', 'SULFUR', 'OXYGEN']
    atom_elements = [el for el in element_order if el != 'ASH']

    # Helpers
    def wt_to_mol(wt_dict, total_mass):
        return {el: (wt_dict[el] / 100) * total_mass / atomic_weights[el] for el in atom_elements}

    def mol_to_wt(mol_dict, total_mass=None):
        wt_dict = {el: mol_dict[el] * atomic_weights[el] for el in atom_elements}
        if total_mass:
            # Normalize to 100%
            wt_dict = {el: (wt / total_mass * 100) for el, wt in wt_dict.items()}
        return wt_dict

    # Step 1: Convert feed and char to moles
    feed_mass = 1.0  # assume 1g feed
    char_mass = char_yield
    organic_mass = 1.0 - char_yield

    feed_mol = wt_to_mol(feed_ultimate, feed_mass)
    char_mol = wt_to_mol(char_predicted, char_mass)

    # Step 2: Calculate limiting ratio to scale char moles
    ratios = {el: feed_mol[el] / char_mol[el] if char_mol[el] > 0 else 1.0 for el in atom_elements}
    min_ratio = min(1.0, min(ratios.values()))

    # Step 3: Scale char moles
    char_mol_corrected = {el: char_mol[el] * min_ratio for el in atom_elements}
    char_wt_corrected = mol_to_wt(char_mol_corrected)
    char_wt_corrected['ASH'] = (char_predicted.get('ASH', 0.0) / 100) * char_mass  # real ash mass

    # Step 4: Organic = remaining atoms
    organic_mol = {el: feed_mol[el] - char_mol_corrected[el] for el in atom_elements}
    organic_wt = mol_to_wt(organic_mol)
    organic_wt['ASH'] = 0.0

    # Step 5: Normalize to wt% of each phase
    char_total_mass = sum(char_wt_corrected.values())
    organic_total_mass = sum(organic_wt.values())

    char_normalized = {el: char_wt_corrected.get(el, 0.0) / char_total_mass * 100 for el in element_order}
    organic_normalized = {el: organic_wt.get(el, 0.0) / organic_total_mass * 100 for el in element_order}

    return char_normalized, organic_normalized


class PostProcessor:
    def __init__(self, process_name="htc"):
        self.model = ANNModelSklearn(process_name)
        self.process_name = process_name

    def predict_and_postprocess(self, x_input, input_paths=None):
        ann_output = self.model.predict(x_input[:-1])
        x_char_split = x_input[-1]

        # Feed ultimate analysis
        feed_ultimate = {
            "CARBON": x_input[0],
            "HYDROGEN": x_input[1],
            "NITROGEN": x_input[2],
            "SULFUR": x_input[3],
            "OXYGEN": x_input[4],
            "ASH": x_input[7],
            "CHLORINE": 0.0
        }

        # Atom balance correction
        char_pred = ann_output["char_ultimate"]
        char_pred["ASH"] = ann_output.get("char_ash", x_input[7])
        char_bal, org_bal = atom_bal(feed_ultimate, char_pred, x_char_split)

        # Prepare values
        feed_prop = [
            0,
            x_input[6],  # FC
            x_input[5],  # VM
            x_input[7],  # Ash
            x_input[7],  # Ash again for ULT
            x_input[0],  # C
            x_input[1],  # H
            x_input[2],  # N
            100 - sum(x_input[i] for i in [0,1,2,3,4]),  # O
            x_input[3],  # S
            x_input[4],  # O
            0,
            0,
            x_input[3],  # S again
        ]

        htc_sim = [
            x_input[10],  # Solid loading
            x_input[8],   # HTC temp
            x_input[8],   # HX temp
            ann_output["mass_yields"]["char"],
            ann_output["mass_yields"]["organics"],
            ann_output["mass_yields"]["water"],
            *ann_output["char_prox"],
            *[char_bal[k] for k in ["CARBON", "HYDROGEN", "NITROGEN", "CHLORINE", "SULFUR", "OXYGEN"]],
            *ann_output["org_prox"],
            *[org_bal[k] for k in ["CARBON", "HYDROGEN", "NITROGEN", "CHLORINE", "SULFUR", "OXYGEN"]],
            char_bal["SULFUR"],
            org_bal["SULFUR"]
        ]

        values = feed_prop + htc_sim + [x_char_split]

        # 🔍 Optional: Validate length
        if input_paths is not None and len(values) != len(input_paths):
            raise ValueError(f"Mismatch: {len(values)} values vs {len(input_paths)} paths")

        return {
            "values": values,
            "postprocessed": {
                "mass_yields": ann_output["mass_yields"],
                "char_bal": char_bal,
                "org_bal": org_bal
            }
        }
