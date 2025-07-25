from engine.simulation.ann_predictor import ANNModelSklearn


def atom_bal(feed_ultimate: dict, char_predicted: dict, char_yield: float, solid_loading: float):
    """
    Atom balance considering char + organic from feed, excluding inert water.
    """
    atomic_weights = {
        'CARBON': 12.01,
        'HYDROGEN': 1.008,
        'OXYGEN': 16.00,
        'NITROGEN': 14.01,
        'SULFUR': 32.06,
        'CHLORINE': 35.45
    }

    element_order = ['ASH', 'CARBON', 'HYDROGEN', 'NITROGEN', 'CHLORINE', 'SULFUR', 'OXYGEN']
    atom_elements = [el for el in element_order if el != 'ASH']

    def wt_to_mol(wt_dict, total_mass):
        return {el: (wt_dict[el] / 100) * total_mass / atomic_weights[el] for el in atom_elements}

    def mol_to_wt(mol_dict, total_mass=None):
        wt_dict = {el: mol_dict[el] * atomic_weights[el] for el in atom_elements}
        if total_mass:
            wt_dict = {el: (wt / total_mass * 100) for el, wt in wt_dict.items()}
        return wt_dict

    feed_mass = 1.0  # always 1g feed
    char_mass = char_yield
    organic_mass = solid_loading - char_yield  # correct for input solid only

    feed_mol = wt_to_mol(feed_ultimate, feed_mass)
    char_mol = wt_to_mol(char_predicted, char_mass)

    ratios = {el: feed_mol[el] / char_mol[el] if char_mol[el] > 0 else 1.0 for el in atom_elements}
    min_ratio = min(1.0, min(ratios.values()))

    char_mol_corrected = {el: char_mol[el] * min_ratio for el in atom_elements}
    char_wt_corrected = mol_to_wt(char_mol_corrected)
    char_wt_corrected['ASH'] = (char_predicted.get('ASH', 0.0) / 100) * char_mass

    organic_mol = {el: feed_mol[el] - char_mol_corrected[el] for el in atom_elements}
    organic_wt = mol_to_wt(organic_mol)
    organic_wt['ASH'] = 0.0

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
        solid_loading = x_input[10]  # wt% of feed (e.g., 20 means 20 feed, 80 water)

        char_yield = ann_output["mass_yields"]  # wt% of feed (e.g., 0.35)
        organic_yield = solid_loading - char_yield  # balance from feed
        water_yield = 100.0 - solid_loading  # water is inert input/output

        # Feed ultimate analysis (1g basis)
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

        char_bal, org_bal = atom_bal(feed_ultimate, char_pred, char_yield, solid_loading)

        values = []

        # === Feed PROX (4): FC, VM, Ash, Ash again ===
        values += [x_input[6], x_input[5], x_input[7], x_input[7]]

        # === Feed ULT (7): C, H, N, Cl (0), S, O, (0 filler) ===
        values += [x_input[0], x_input[1], x_input[2], 0.0, x_input[3], x_input[4], 0.0]

        # === SULFANAL (3): dummy, dummy, S ===
        values += [0.0, 0.0, x_input[3]]

        # === Feed flow & HTC temp (2) ===
        values += [x_input[10], x_input[8]]

        # === HX temp (1) ===
        values += [x_input[8]]

        # === Dummy streams = yields (3): char, organic, water ===
        values += [char_yield, organic_yield, water_yield]

        # === Char PROX (4) ===
        values += ann_output["char_prox"]

        # === Char ULT (7): ASH first, then elements ===
        values += [
            char_pred["ASH"],
            char_bal["CARBON"],
            char_bal["HYDROGEN"],
            char_bal["NITROGEN"],
            char_bal["CHLORINE"],
            char_bal["SULFUR"],
            char_bal["OXYGEN"],
        ]

        # === Org PROX (4) ===
        values += ann_output["org_prox"]

        # === Org ULT (7): ASH first (0), then elements ===
        values += [
            0.0,
            org_bal["CARBON"],
            org_bal["HYDROGEN"],
            org_bal["NITROGEN"],
            org_bal["CHLORINE"],
            org_bal["SULFUR"],
            org_bal["OXYGEN"],
        ]

        # === SULFUR paths (2): char S, org S ===
        values += [char_bal["SULFUR"], org_bal["SULFUR"]]

        # === Char split (1) ===
        values += [x_char_split]

        # --- Validation ---
        if input_paths is not None and len(values) != len(input_paths):
            raise ValueError(f"Mismatch: {len(values)} values vs {len(input_paths)} paths")

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
