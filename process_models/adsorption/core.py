# core.py
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Iterable, Optional

R = 8.314  # J/mol/K

# =========================
# Validation & Fitting
# =========================

def _check_avrami_map(avrami_map: Dict[int, float]) -> None:
    """Ensure Avrami dataset is valid for Arrhenius fitting."""
    if not avrami_map or len(avrami_map) < 2:
        raise ValueError("Need at least two temperature points for Arrhenius fit.")
    for T, k in avrami_map.items():
        if T is None or T <= 0:
            raise ValueError(f"Invalid temperature {T}. Must be > 0 K.")
        if k is None or k <= 0:
            raise ValueError(f"Invalid k_MA at T={T}: {k}. Must be > 0.")

def fit_arrhenius(avrami_map: Dict[int, float]) -> Tuple[float, float]:
    """
    Fit y = m*(1/T) + b with:
      x = 1/T (K^-1), y = ln(k/1000).
    Returns:
      Ea [J/mol] = -R * m
      a  [1/s / 1000] = exp(b)
    """
    _check_avrami_map(avrami_map)
    T = np.array(sorted(avrami_map), dtype=float)
    k = np.array([avrami_map[t] for t in T], dtype=float)
    x = 1.0 / T
    y = np.log(k / 1000.0)
    m, b = np.polyfit(x, y, 1)
    Ea = -R * m                # J/mol
    a  = float(np.exp(b))      # pre-exponential in (1/s)/1000
    return Ea, a

def k_ma(T: float, Ea: float, a: float) -> float:
    """k_MA(T) = a * exp(-Ea/(R*T)) * 1000  [1/s]."""
    if T <= 0:
        raise ValueError("T must be > 0 K")
    return float(a * np.exp(-Ea / (R * T)))


# =========================
# Stream helpers
# =========================

def _stream_temperature(stream: Dict) -> float:
    """
    Extract temperature [K] from a stream record.
    Accepts: {"T": 308}, {"temperature": 308}, {"temperature": {"value": 308, "unit": "K"}}
    """
    if "T" in stream and stream["T"] is not None:
        return float(stream["T"])
    t = stream.get("temperature")
    if t is None:
        raise KeyError("Stream must include temperature as 'T' or 'temperature'.")
    if isinstance(t, dict) and "value" in t:
        return float(t["value"])
    return float(t)

def _partial_pressure(stream: Dict, species: str) -> float:
    """
    Returns partial pressure of 'species' in same unit as P_total.
    Accepts:
      {"P_total": ..., "y": {"CO2": ..., "N2": ...}}
      {"P_total": ..., "y_CO2": ...} / {"y_N2": ...}
      {"P_CO2": ...} / {"P_N2": ...}  (direct partial pressure)
    """
    key_direct = f"P_{species}"
    if key_direct in stream and stream[key_direct] is not None:
        return float(stream[key_direct])

    if "P_total" in stream and stream["P_total"] is not None:
        Ptot = float(stream["P_total"])
        y_key = f"y_{species}"
        if y_key in stream and stream[y_key] is not None:
            return Ptot * float(stream[y_key])
        ymap = stream.get("y") or {}
        if isinstance(ymap, dict) and (species in ymap):
            return Ptot * float(ymap[species])

    raise KeyError(f"Stream must provide partial pressure or mole fraction for {species}.")


# =========================
# Isotherms (Competitive Tóth)
# =========================

def _b_of_T(alpha0: float, dH_kJmol: float, T: float, use_negative_dH: bool = True) -> float:
    """
    b(T) = alpha0 * exp(sign * ΔH / (R T)).
    If ΔH is stored as a POSITIVE magnitude (exothermic), set use_negative_dH=True (default),
    so affinity decreases with T.
    """
    if T <= 0:
        raise ValueError("T must be > 0 K")
    sign = -1.0 if use_negative_dH else 1.0
    return float(alpha0 * np.exp(sign * (dH_kJmol * 1000.0) / (R * T)))

def _qeq_toth_component_from_P(
        P_i: float, P_j: float, params_i: Dict, params_j: Dict, T: float,
        use_negative_dH: bool = True
) -> float:
    """
    Binary competitive Tóth for component i in presence of j:
      q_i^eq = q_i * (b_i P_i) / [ 1 + (b_i P_i)^{t_i} + (b_j P_j)^{t_j} ]^{1/t_i}
    """
    b_i = _b_of_T(params_i["alpha0"], params_i["delta_H_ads_kJ_per_mol"], T, use_negative_dH)
    b_j = _b_of_T(params_j["alpha0"], params_j["delta_H_ads_kJ_per_mol"], T, use_negative_dH)

    t_i = float(params_i["t_t"])
    t_j = float(params_j["t_t"])
    q_i_T = float(params_i["q"]) * np.exp(-params_i["delta"] * T)

    x_i = (b_i * P_i) ** t_i
    x_j = (b_j * P_j) ** t_j
    denom = 1.0 + x_i + x_j
    return float(q_i_T * (b_i * P_i) / denom)

def qeq_toth_CO2N2_from_stream(
        stream: Dict,
        T: float,
        co2_params: Dict,
        n2_params: Dict,
        use_negative_dH: bool = True
) -> tuple[float, float]:
    """Return (q_eq_CO2, q_eq_N2) [mol/kg] from a binary CO2/N2 stream using competitive Tóth."""
    P_CO2 = _partial_pressure(stream, "CO2")
    try:
        P_N2 = _partial_pressure(stream, "N2")
    except KeyError:
        P_N2 = 0.0
    qeq_co2 = _qeq_toth_component_from_P(P_CO2, P_N2, co2_params, n2_params, T, use_negative_dH)
    qeq_n2  = _qeq_toth_component_from_P(P_N2, P_CO2, n2_params, co2_params, T, use_negative_dH)

    return qeq_co2, qeq_n2


# =========================
# Avrami kinetics
# =========================

def avrami_step(q: float, qeq: float, k: float, n: float, dt: float) -> float:
    """Incremental Modified Avrami update: q_{t+dt} = q + (qeq - q) * (1 - exp(-(k*dt)^n))."""
    if dt <= 0:
        return float(q)
    phi = 1.0 - np.exp(- (max(k, 0.0) * dt/1000)**max(n, 1e-12))
    return float(q + qeq * phi)

def avrami_closed_form(t: np.ndarray, q0: float, qeq: float, k: float, n: float) -> np.ndarray:
    """
    Closed-form Modified Avrami for constant qeq and k over time:
      q(t) = qeq - (qeq - q0) * exp(-(k t)^n)
    """
    t = np.asarray(t, dtype=float)
    a = k * t / 1000.0  # convert k from 1/s to 1/ms
    b = a**n
    c = np.exp(-b)
    phi = 1.0 - np.exp(- ((max(k, 0.0) * t/1000)**max(n, 1e-12)))
    return (qeq * (phi))


# =========================
# Binary simulator (CO2 + N2)
# =========================

def simulate_binary_co2_n2(
        times: Iterable[float],
        *,
        # Preferred: constant or time-varying stream profile
        stream_prof: Optional[Iterable[Dict]] = None,  # single dict (constant) or list of dicts (time-varying)
        # Kinetics (CO2)
        n_ma_co2: float,
        Ea_co2: float,
        a_co2: float,
        # Kinetics (N2)
        n_ma_n2: float,
        Ea_n2: float,
        a_n2: float,
        # Isotherm params
        co2_iso: Dict,
        n2_iso: Dict,
        # Initial loadings [mol/kg]
        q0_co2: float = 0.0,
        q0_n2: float = 0.0,
        # --- NEW: for per-time capture calculation ---
        sorbent_mass_ton: Optional[float] = None,            # ADS["sorbent"]["mass"] (ton)
        inlet_co2_molar_flow_mol_s: Optional[float] = None,  # ADS["stream"]["inlet_flow_rate"]["components"]["CO2"] in mol/s
        inlet_n2_molar_flow_mol_s: Optional[float] = None,   # ADS["stream"]["inlet_flow_rate"]["components"]["N2"] in mol/s
) -> Dict[str, np.ndarray]:
    """
    Binary CO2/N2 adsorption vs time with competitive Tóth + Modified Avrami.

    Returns dict with:
      - t
      - q_co2, q_n2                       [mol/kg]
      - qeq_co2, qeq_n2                   [mol/kg]
      - k_MA_co2, k_MA_n2                 [1/s]
      - q_co2_mmol_g, q_n2_mmol_g         [mmol/g] (numerically = mol/kg)
      - NEW (scalars at final time):
          solid_circulation_rate_kg_s
          capture_rate_co2_mol_s, capture_rate_n2_mol_s, capture_rate_total_mol_s
          pct_feed_captured_co2, pct_feed_captured_n2      [%]
    """
    times = np.asarray(list(times), dtype=float)
    nT = len(times)
    if nT < 2:
        raise ValueError("`times` must have at least two points.")
    tau = float(times[-1] - times[0])
    if tau <= 0:
        raise ValueError("Retention time must be positive (times[-1] > times[0]).")

    q_co2   = np.zeros(nT, dtype=float)
    q_n2    = np.zeros(nT, dtype=float)
    qeq_co2 = np.zeros(nT, dtype=float)
    qeq_n2  = np.zeros(nT, dtype=float)
    k_co2   = np.zeros(nT, dtype=float)
    k_n2    = np.zeros(nT, dtype=float)

    q_co2[0] = q0_co2
    q_n2[0]  = q0_n2

    def _finalize_and_return():
        out = {
            "t": times,
            "q_co2": q_co2, "q_n2": q_n2,
            "qeq_co2": qeq_co2, "qeq_n2": qeq_n2,
            "k_MA_co2": k_co2, "k_MA_n2": k_n2,
            "q_co2_mmol_g": q_co2.copy(), "q_n2_mmol_g": q_n2.copy(),
        }
        # Per-time capture using final qt and retention-time-based solid circulation
        if sorbent_mass_ton is not None:
            m_dot_kg_s = float(sorbent_mass_ton) * 1000.0 / tau
            out["solid_circulation_rate_kg_s"] = m_dot_kg_s
            cap_co2 = q_co2[-1] * m_dot_kg_s
            cap_n2  = q_n2[-1]  * m_dot_kg_s
            out["capture_rate_co2_mol_s"] = cap_co2
            out["capture_rate_n2_mol_s"]  = cap_n2
            out["capture_rate_total_mol_s"] = cap_co2 + cap_n2

            # % captured vs inlet molar flow (if provided), safe divide
            if inlet_co2_molar_flow_mol_s is not None and inlet_co2_molar_flow_mol_s > 0:
                out["pct_feed_captured_co2"] = 100.0 * cap_co2 / inlet_co2_molar_flow_mol_s
            if inlet_n2_molar_flow_mol_s is not None and inlet_n2_molar_flow_mol_s > 0:
                out["pct_feed_captured_n2"]  = 100.0 * cap_n2  / inlet_n2_molar_flow_mol_s
        return out

    # ---- Constant stream → closed-form
    if stream_prof is not None and (isinstance(stream_prof, dict) or (hasattr(stream_prof, "__len__") and len(stream_prof) == 1)):
        stream0 = stream_prof if isinstance(stream_prof, dict) else list(stream_prof)[0]
        T = _stream_temperature(stream0)

        k_const_co2 = k_ma(T, Ea_co2, a_co2)
        k_const_n2  = k_ma(T, Ea_n2,  a_n2)
        qeq_const_co2, qeq_const_n2 = qeq_toth_CO2N2_from_stream(stream0, T, co2_iso, n2_iso, use_negative_dH=True)

        k_co2[:] = k_const_co2
        k_n2[:]  = k_const_n2
        qeq_co2[:] = qeq_const_co2
        qeq_n2[:]  = qeq_const_n2

        q_co2[:] = avrami_closed_form(times, q0_co2, qeq_const_co2, k_const_co2, n_ma_co2)
        q_n2[:]  = avrami_closed_form(times, q0_n2,  qeq_const_n2,  k_const_n2,  n_ma_n2)

        return _finalize_and_return()

    # ---- Time-varying stream → incremental loop
    if stream_prof is not None:
        stream_list = list(stream_prof)
        if len(stream_list) != nT:
            raise ValueError("`stream_prof` length must match `times` length.")

        T0 = _stream_temperature(stream_list[0])
        k_co2[0] = k_ma(T0, Ea_co2, a_co2)
        k_n2[0]  = k_ma(T0, Ea_n2,  a_n2)
        qeq_co2[0], qeq_n2[0] = qeq_toth_CO2N2_from_stream(stream_list[0], T0, co2_iso, n2_iso, use_negative_dH=True)

        for i in range(1, nT):
            dt = times[i] - times[i-1]
            if dt < 0:
                raise ValueError("`times` must be non-decreasing.")

            Ti = _stream_temperature(stream_list[i])
            k_co2[i] = k_ma(Ti, Ea_co2, a_co2)
            k_n2[i]  = k_ma(Ti, Ea_n2,  a_n2)
            qeq_co2[i], qeq_n2[i] = qeq_toth_CO2N2_from_stream(stream_list[i], Ti, co2_iso, n2_iso, use_negative_dH=True)

            q_co2[i] = avrami_step(q_co2[i-1], qeq_co2[i], k_co2[i], n_ma_co2, dt)
            q_n2[i]  = avrami_step(q_n2[i-1],  qeq_n2[i],  k_n2[i],  n_ma_n2,  dt)

        return _finalize_and_return()

    raise ValueError("Provide `stream_prof` (single dict or list) for binary simulation.")


# =========================
# Visualization (model vs experiment)
# =========================

import matplotlib.pyplot as plt

def _pressure_to_bar(value: float, unit: str) -> float:
    unit = (unit or "").lower()
    if unit == "bar":  return float(value)
    if unit == "mbar": return float(value) / 1000.0
    if unit == "pa":   return float(value) / 1e5
    return float(value)

def visualize_validation(
        ADS: Dict,
        *,
        get_pressure_fn=None,                 # e.g., data.get_pressure
        exp_t_s: Optional[np.ndarray] = None, # experimental time [s]
        exp_q_over_qe: Optional[np.ndarray] = None,
        csv_path: Optional[str] = None,       # CSV with columns: t_seconds,q_over_qe
        t_min_max: tuple[float, float] = (1e1, 1e4),
        n_points: int = 200,
        ax=None,
):
    """
    Build model curve q/qe vs time and overlay experimental points.
    Uses the constant-stream fast path if ADS['stream'] is constant.
    """
    # Arrhenius for CO2 & N2
    av_co2 = ADS["modified_avrami"]["CO2"]
    av_n2  = ADS["modified_avrami"]["N2"]
    Ea_CO2, a_CO2 = fit_arrhenius(av_co2["data"])
    Ea_N2,  a_N2  = fit_arrhenius(av_n2["data"])

    # Time grid
    tmin, tmax = t_min_max
    t = np.logspace(np.log10(tmin), np.log10(tmax), n_points)

    # Constant stream from ADS
    if get_pressure_fn is not None:
        P_total_bar = float(get_pressure_fn("bar"))
    else:
        P_meta = ADS["stream"]["pressure"]
        P_total_bar = _pressure_to_bar(P_meta["value"], P_meta.get("unit", "bar"))

    ymap = ADS["stream"]["inlet_mole_fraction"]
    stream_const = {
        "P_total": P_total_bar,
        "y": {"CO2": float(ymap.get("CO2", 0.0)), "N2": float(ymap.get("N2", 0.0))},
        "T": float(ADS["stream"]["temperature"]["value"]),
    }

    # Isotherms
    CO2_ISO = ADS["isotherm_CO2"]
    N2_ISO  = ADS["isotherm_N2"]

    # Simulate (binary), then form CO2 q/qe
    res = simulate_binary_co2_n2(
        times=t,
        stream_prof=stream_const,
        n_ma_co2=av_co2["n_MA"], Ea_co2=Ea_CO2, a_co2=a_CO2,
        n_ma_n2=av_n2["n_MA"],  Ea_n2=Ea_N2,  a_n2=a_N2,
        co2_iso=CO2_ISO, n2_iso=N2_ISO,
        q0_co2=0.0, q0_n2=0.0,
    )
    q_sim = res["q_co2"]; qeq_sim = np.clip(res["qeq_co2"], 1e-12, None)
    qqe_sim = np.clip(q_sim / qeq_sim, 0.0, 1.0)

    # Experimental data (optional)
    if (exp_t_s is None or exp_q_over_qe is None) and csv_path is not None:
        try:
            data = np.genfromtxt(csv_path, delimiter=",", names=True)
            if data.dtype.names is None:
                data = np.genfromtxt(csv_path, delimiter=",")
                exp_t_s = data[:, 0].astype(float); exp_q_over_qe = data[:, 1].astype(float)
            else:
                cols = [c.lower() for c in data.dtype.names]
                t_col = data.dtype.names[cols.index("t_seconds")] if "t_seconds" in cols else data.dtype.names[0]
                q_col = data.dtype.names[cols.index("q_over_qe")] if "q_over_qe" in cols else data.dtype.names[1]
                exp_t_s = np.asarray(data[t_col], dtype=float)
                exp_q_over_qe = np.asarray(data[q_col], dtype=float)
        except Exception:
            raw = np.loadtxt(csv_path, delimiter=",")
            exp_t_s = raw[:, 0].astype(float); exp_q_over_qe = raw[:, 1].astype(float)

    # Plot
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4)); created_fig = True
    else:
        fig = ax.figure

    ax.plot(t, qqe_sim, label="Model (q/qe)")
    if exp_t_s is not None and exp_q_over_qe is not None and len(exp_t_s):
        ax.scatter(exp_t_s, exp_q_over_qe, marker="x", label="Experiment", zorder=3)

    ax.set_xscale("log")
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("log(t [s])")
    ax.set_ylabel("q_t / q_e")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.tight_layout()

    if created_fig:
        plt.show()

    return {"res": res, "qqe_sim": qqe_sim, "fig": fig, "ax": ax}
