import numpy as np
from engine.simulation.ann_predictor import ANNModelSklearn
from .htc_postprocess import atom_bal

class PostProcessor:
    def __init__(self, process_name="htc"):
        self.model = ANNModelSklearn(process_name)
        self.process_name = process_name

    def predict_and_postprocess(self, x_input):
        # === ANN Prediction ===
        ann_output = self.model.predict(x_input[:-1])

        # === Parse ANN output ===
        char_yield = ann_output[0]
        char_prox = [ann_output[7], ann_output[6], ann_output[8], ann_output[8]]  # FC, VM, Ash, Ash
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
        organic_yield = solid_loading - char_yield
        water_yield = 100.0 - solid_loading

        # === Values in order matching htc_path.yaml ===
        values = []

        # --- Feed PROX (FC, VM, Ash, Ash) ---
        values += [x_input[6], x_input[5], x_input[7], x_input[7]]

        # --- Feed ULT (ASH first) ---
        values += [x_input[7], x_input[0], x_input[1], x_input[2], 0.0, x_input[3], x_input[4]]

        # --- Sulfur Analysis (dummy, dummy, S) ---
        values += [0.0, 0.0, x_input[3]]

        # --- Feed Flowrate and HTC Temp ---
        values += [x_input[10], x_input[8]]

        # --- HX Temp Input ---
        values += [x_input[8]]

        # --- Dummy Gas Stream Yields ---
        values += [char_yield, organic_yield, water_yield]

        # --- Char PROX ---
        values += char_prox

        # --- Char ULT (ASH first) ---
        values += [
            char_ult["ASH"],
            char_bal["CARBON"],
            char_bal["HYDROGEN"],
            char_bal["NITROGEN"],
            char_bal["CHLORINE"],
            char_bal["SULFUR"],
            char_bal["OXYGEN"],
        ]

        # --- Organic PROX (placeholders if not modeled) ---
        org_prox = [0.0, 0.0, 0.0, 0.0]
        values += org_prox

        # --- Organic ULT (ASH first as 0.0) ---
        values += [
            0.0,
            org_bal["CARBON"],
            org_bal["HYDROGEN"],
            org_bal["NITROGEN"],
            org_bal["CHLORINE"],
            org_bal["SULFUR"],
            org_bal["OXYGEN"],
        ]

        # --- Sulfur Distribution (char S, organic S) ---
        values += [char_bal["SULFUR"], org_bal["SULFUR"]]

        # --- Char Split (x_char_split) ---
        values += [x_char_split]

        return values
