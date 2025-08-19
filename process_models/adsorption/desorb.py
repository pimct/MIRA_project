from typing import NamedTuple

class Outputs(NamedTuple):
    B10_CO2FRACS: float  # corresponds to Excel cell B10

def compute_outputs() -> Outputs:
    """
    Reproduces the Excel logic for B10 using constants and inputs
    defined inside this function.

    Excel relation:
      B10 = Y42 * 60 / 1000 / CO2FLOW

    Note: In your desorb.xlsm, Y42 is stored as a numeric literal
    (no formula). If you want B10 to update from inputs, please
    share the formula/rule for Y42 and I’ll wire it in.
    """

    # ===== Inputs from your Excel (B1:B3, B14:B17, B22:B25, B47:B48) =====
    V       = 753.86334954     # B1
    P       = 1013.0           # B2
    T       = 317.81389782     # B3
    N2FLOW  = 320.90718577     # B14
    O2FLOW  = 0.0              # B15
    CO2FLOW = 844220.89574     # B16
    H2OFLOW = 0.0              # B17
    YN      = 0.0003799778586  # B22
    YO      = 0.0              # B23
    YCO2    = 0.99962002214    # B24
    YH      = 0.0              # B25
    TSOLID  = 317.81389782     # B47
    SOLIDF  = 11621036.517     # B48

    # ===== Constant from your sheet row 42 (stored as value, no formula) =====
    Y42 = 4_289_694.478

    # ===== Excel-equivalent output =====
    factor = 60.0 / 100.0 / 10.0  # == 60/1000
    B10 = (Y42 * factor) / CO2FLOW if CO2FLOW else 0.0

    return Outputs(B10_CO2FRACS=B10)

# Example run
if __name__ == "__main__":
    out = compute_outputs()
    print("B10 =", out.B10_CO2FRACS)  # ~0.30487479044734217 with the values above
