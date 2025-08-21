import numpy as np
from adsorption_data import ADS, get_pressure
from core import fit_arrhenius, simulate_binary_co2_n2

def make_constant_stream_from_ADS():
    P_total_bar = float(get_pressure("bar"))
    y = ADS["stream"]["inlet_mole_fraction"]
    T = float(ADS["stream"]["temperature"]["value"])
    return {"P_total": P_total_bar, "y": {"CO2": y["CO2"], "N2": y["N2"]}, "T": T}

# Arrhenius params
av_CO2 = ADS["modified_avrami"]["CO2"]; av_N2 = ADS["modified_avrami"]["N2"]
Ea_CO2, a_CO2 = fit_arrhenius(av_CO2["data"]); Ea_N2, a_N2 = fit_arrhenius(av_N2["data"])
n_CO2, n_N2 = av_CO2["n_MA"], av_N2["n_MA"]

# Convert feed flows to mol/s from ADS (mol/h → mol/s)
Fin_CO2 = ADS["stream"]["inlet_flow_rate"]["components"]["CO2"]["value"] / 3600.0
Fin_N2  = ADS["stream"]["inlet_flow_rate"]["components"]["N2"]["value"]  / 3600.0

# Use retention time from times array (e.g., 0→234 s)
times = np.linspace(0.0, ADS["retention_time"]["value"], 100)

res = simulate_binary_co2_n2(
    times=times,
    stream_prof=make_constant_stream_from_ADS(),
    n_ma_co2=n_CO2, Ea_co2=Ea_CO2, a_co2=a_CO2,
    n_ma_n2=n_N2,  Ea_n2=Ea_N2,  a_n2=a_N2,
    co2_iso=ADS["isotherm_CO2"], n2_iso=ADS["isotherm_N2"],
    q0_co2=0.0, q0_n2=0.0,
    sorbent_mass_ton=ADS["sorbent"]["mass"],
    inlet_co2_molar_flow_mol_s=Fin_CO2,
    inlet_n2_molar_flow_mol_s=Fin_N2,
)

print(f"Solid circulation rate: {res['solid_circulation_rate_kg_s']:.3f} kg/s")
print(f"CO2 capture rate: {res['capture_rate_co2_mol_s']:.3f} mol/s")
print(f"N2  capture rate: {res['capture_rate_n2_mol_s']:.3f} mol/s")
print(f"CO2 inlet flow: {Fin_CO2:.3f} mol/s")
print(f"CO2 % captured vs feed: {res.get('pct_feed_captured_co2', float('nan')):.3f}%")
print(f"N2  % captured vs feed: {res.get('pct_feed_captured_n2',  float('nan')):.3f}%")


#########  recheck the adsorption results- we neede to reccalculate the CO2 outlet flow
# co2 accumulated in the solid (0-retention time) = co2 inlet(0-retention time) - co2 adsorbed (0-retention time)