# Central data store for adsorption constants, experiments, and stream info

ADS = {
    # ---------------------------
    # Stream conditions (feed)
    # ---------------------------
    "stream": {
        "volumetric_flow_rate": {"value": 12414.45481, "unit": "L/min"},
        "pressure":             {"value": 1114.575,    "unit": "mbar"},  # stored as mbar
        "temperature":          {"value": 317.2885084, "unit": "K"},

        # Inlet flow rate (total and per component)
        "inlet_flow_rate": {
            "total": {"value": 14425461.4, "unit": "mol/h"},
            "components": {
                "N2":  {"value": 10514182.87,  "unit": "mol/h"},
                "O2":  {"value": 1976008.913,  "unit": "mol/h"},
                "CO2": {"value": 1154351.393,   "unit": "mol/h"},
                "H2O": {"value": 780918.2245,   "unit": "mol/h"},
            },
        },

        # Inlet mole fraction (dimensionless; should sum ~1)
        "inlet_mole_fraction": {
            "N2":  0.728862847,
            "O2":  0.136980638,
            "CO2": 0.0800218,
            "H2O": 0.054134714,
        },

        # Chosen basis for pressure conversion
        "pressure_basis": "bar",   # options: "mbar", "bar", "Pa"
    },

    # ---------------------------
    # Solid adsorption parameters
    # ---------------------------
    "retention_time": {"value": 234.0, "unit": "s"},  # retention time in seconds
    "sorbent": {
        "mass": 50,              # ton
    },

    # ---------------------------
    # Isotherm constants (CO2+N2, Toth)
    # ---------------------------
    "isotherm_CO2": {
        "q": 21.6816,                 # mol/kg
        "delta": 0.004896,            # -
        "alpha0": 3.89808e-3,            # (consistent with pressure basis)
        "delta_H_ads_kJ_per_mol": -21.0,
        "t_t": 0.8234,
    },
    "isotherm_N2": {
        "q": 85.8139,               # mol/kg
        "delta": 0.0114,            # -
        "alpha0": 0.007685847,      # (consistent with pressure basis)
        "delta_H_ads_kJ_per_mol": -6.9,
        "t_t": 0.9493,
    },

    # ---------------------------
    # Modified Avrami datasets
    # ---------------------------
    "modified_avrami": {
        "CO2": {
            "n_MA": 0.74,
            "data": {   # T [K] : k_MA [1/s]
                296: 2715,
                308: 3443,
                318: 4486,
            },
        },
        "N2": {
            "n_MA": 0.71,
            "data": {
                296: 2890,
                308: 2524,
                318: 5395,
            },
        },
    },

    # ---------------------------
    # Metadata
    # ---------------------------
    "meta": {
        "units": {
            "k_MA": "1/s",
            "T": "K",
            "q": "mol/kg",
            "pressure": ["mbar", "bar", "Pa"],
            "volumetric_flow": "L/min",
            "molar_flow": "mol/h",
        },
        "notes": (
            "Arrhenius fit uses ln(k/1000) vs 1/T. "
            "Total P can be converted based on ADS['stream']['pressure_basis'] "
            "and P_CO2 = y_CO2 * P_total."
        ),
    },
}


# ---------- Helper for pressure conversion ----------
def get_pressure(unit: str = None) -> float:
    """
    Get total pressure in desired unit.
    unit options: 'mbar', 'bar', 'Pa'.
    If None, uses ADS['stream']['pressure_basis'].
    """
    P_mbar = ADS["stream"]["pressure"]["value"]  # always stored in mbar
    if unit is None:
        unit = ADS["stream"]["pressure_basis"]

    if unit == "mbar":
        return P_mbar
    elif unit == "bar":
        return P_mbar / 1000.0
    elif unit == "Pa":
        return P_mbar * 100.0
    else:
        raise ValueError(f"Unknown unit {unit}")
