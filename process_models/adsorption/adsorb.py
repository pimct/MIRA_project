from typing import NamedTuple
import math

class Outputs(NamedTuple):
    B8_N2FRACS: float   # corresponds to Excel cell B8
    B10_CO2FRACS: float # corresponds to Excel cell B10

def compute_outputs() -> Outputs:
    """
    Fully reproduces the Excel logic for B8 and B10 using
    constants and input values defined inside this function.
    """

    # ===== Aspen inputs (from your Excel B1:B3, B14:B17, B22:B25) =====
    V       = 12414.454809
    P       = 1114.575
    T       = 317.28850841
    N2FLOW  = 10514182.872
    O2FLOW  = 1976008.9131
    CO2FLOW = 1154351.3928
    H2OFLOW = 780918.22451
    YN      = 0.72886284734
    YO      = 0.13698063847
    YCO2    = 0.080021800383
    YH      = 0.054134713805

    # ===== Constants (cached values from your sheet) =====
    L26  = 50000000.0
    L29  = 234.13988903598405  # I141
    L38  = 4.344997486997632
    N38  = 0.74
    S34  = 2.182602187999699
    Y38  = 2.524
    AA38 = 0.71
    S35  = 0.16934233142866437
    H141 = 100.0

    # ===== Helper functions =====
    def I_k(k: int) -> float:
        # Linear ramp from I41=0 to I141=L29 in 100 steps (k=41..141)
        dI = L29 / H141
        return L29 - (141 - k) * dI

    def J_k(k: int) -> float:
        return I_k(k) / 60.0

    # Constant across rows:
    O_const  = CO2FLOW * 1000.0 / 60.0
    AB_const = N2FLOW  * 1000.0 / 60.0

    # Initialize at row 41
    J_prev = J_k(41)
    S_prev = 0.0
    K_prev = (1.0 - math.exp(-((L38 * I_k(41) / 1000.0) ** N38))) * S34
    L_prev = K_prev * L26
    Y_prev = (1.0 - math.exp(-((Y38 * I_k(41) / 1000.0) ** AA38))) * S35 * L26

    # Sweep rows 42..141
    for k in range(42, 142):
        Jk = J_k(k)
        dJ = abs(Jk - J_prev)

        # CO2 branch
        Kk = (1.0 - math.exp(-((L38 * I_k(k) / 1000.0) ** N38))) * S34
        Lk = Kk * L26
        Mk = Lk - L_prev
        Nk = Mk / dJ if dJ > 0 else 0.0
        Pk = max(O_const - Nk, 0.0)
        S_k = S_prev + (O_const - Pk) * dJ

        # N2 branch
        Xk = (1.0 - math.exp(-((Y38 * I_k(k) / 1000.0) ** AA38))) * S35
        Yk = Xk * L26

        # Roll
        S_prev, L_prev, J_prev, Y_prev = S_k, Lk, Jk, Yk

    # Final row values
    S141 = S_prev
    J141 = J_prev
    Y141 = Y_prev
    J140 = J_k(140)
    Y140 = (1.0 - math.exp(-((Y38 * I_k(140) / 1000.0) ** AA38))) * S35 * L26

    # ===== Compute T141 and AG141 =====
    denom = abs(J141 - J_k(41))  # J41 = 0
    T141 = S141 / denom if denom > 0 else 0.0

    Z141  = Y141 - Y140
    dJ140 = abs(J141 - J140)
    AA141 = Z141 / dJ140 if dJ140 > 0 else 0.0
    AC141 = max(AB_const - AA141, 0.0)
    AF141 = (AB_const - AC141) * (J141 - J140)
    AG141 = AF141 / denom if denom > 0 else 0.0

    # ===== Final Excel outputs =====
    factor = 60.0 / 1000.0
    B8  = factor * AG141 / N2FLOW  if N2FLOW  else 0.0
    B10 = factor * T141  / CO2FLOW if CO2FLOW else 0.0

    return Outputs(B8, B10)


# Example run:
if __name__ == "__main__":
    result = compute_outputs()
    print("B8 =", result.B8_N2FRACS)
    print("B10 =", result.B10_CO2FRACS)
