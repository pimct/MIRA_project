import numpy as np
from engine.simulation.ann_predictor import ANNModelSklearn


class PostProcessor:
    def __init__(self, process_name="htc"):
        self.model = ANNModelSklearn(process_name)

    def predict_and_postprocess(self, x_input):
        # Sample HTC input vector
        #     x_input = np.array([
        #         53.4,   # C_in (wt%)
        #         6.2,    # H_in
        #         3.0,    # N_in
        #         0.3,    # S_in
        #         37.1,   # O_in
        #         54.6,   # VM_in (%)
        #         9.6,    # FC_in
        #         35.8,   # Ash_in (%)
        #         260.0,  # HTC temperature (°C)
        #         2.0,    # Residence time (hr)
        #         30.0,   # Solid loading (wt%)
        #         0.5     # Char split fraction
        #     ])

        ann_output = self.model.predict(x_input[:-1])
        x_char_split = x_input[-1]
        htc_result = self._htc_ann_postprocess(x_input[:-1], ann_output)
        print(htc_result) ####### recheck


        # Build Aspen input values
        feed_prop = [
            0,
            x_input[6],
            x_input[5],
            x_input[7],
            x_input[7],
            x_input[0],
            x_input[1],
            x_input[2],
            100 - sum(x_input[i] for i in [0, 1, 2, 3, 4]),
            x_input[3],
            x_input[4],
            0,
            0,
            x_input[3]
        ]

        htc_sim = [
            x_input[10],
            x_input[8],
            x_input[8],
            htc_result["mass_yields"]["char"],
            htc_result["mass_yields"]["organics"],
            htc_result["mass_yields"]["water"],
            *htc_result["char_prox"],
            *htc_result["char_ult"],
            *htc_result["org_prox"],
            *htc_result["org_ult"],
            htc_result["char_ult"][5],
            htc_result["org_ult"][5]
        ]

        values = feed_prop + htc_sim + [x_char_split]

        return {
            "values": values,
            "postprocessed": htc_result
        }

    def _htc_ann_postprocess(self, x_input, y_ann_output):
        x = np.array(x_input).flatten()
        y = np.array(y_ann_output).flatten()

        charyield, C, H, O, N, S, volatile, fixedC, ash = y
        HHV = np.dot(np.array([C, H, N, S, O, ash]), np.array([0.3491, 1.1783, 0.1005, -0.1034, -0.0151, -0.0211]))

        y_htc = np.array([charyield, C, H, O, N, S, volatile, fixedC, ash])
        y_htc[y_htc < 0] = 0
        if y_htc[0] > 100:
            y_htc[0] = 99

        mass_dryfeed = x[10] / 100
        moisture_char = 0
        charyield = y_htc[0] * mass_dryfeed / (100 - moisture_char)
        mass_drychar = charyield * (100 - moisture_char) / 100

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
        char_prox = [
            moisture_char,
            y_htc[6] * (100 - char_ult[0]) / np.sum(y_htc[6:8]),
            y_htc[7] * (100 - char_ult[0]) / np.sum(y_htc[6:8]),
            char_ult[0]
        ]

        org_element = feed_element - mass_drychar * char_ult / 100
        org_ult = np.clip(org_element * 100 / np.sum(org_element), 0, None)
        org_prox = [0, 0, 100 - org_ult[0], org_ult[0]]

        yChar = [prod_charyield * 100 * 100 / x[10]] + char_ult[[1, 2, 3, 5, 6]].tolist() + char_prox[1:]
        yOrg = [(x[10] - prod_charyield * 100) * 100 / x[10]] + org_ult[[1, 2, 3, 5, 6]].tolist() + org_prox[1:]

        char_prox_final = [0, yChar[7], yChar[6], yChar[8]]
        char_ult_final = [yChar[8], yChar[1], yChar[2], yChar[3], 0, yChar[4], yChar[5]]
        org_prox_final = [0, yOrg[7], yOrg[6], yOrg[8]]
        org_ult_final = [yOrg[8], yOrg[1], yOrg[2], yOrg[3], 0, yOrg[4], yOrg[5]]

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
            "HHV": HHV
        }
