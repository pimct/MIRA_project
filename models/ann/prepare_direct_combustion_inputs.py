# prepare parameters for direct combustion simulation in Aspen Plus.

# def generate_directcomb_parameter_paths():
#     paths = []
#     # Proximate analysis: FC, VM, Ash (Prox)
#     for i in range(4):
#         paths.append(f"\\Data\\Streams\\FEED\\Input\\ELEM\\NCPSD\\WASTES\\PROXANAL\\#{i}")
#     # Ultimate analysis: Ash (Ult), C, H, N, Cl, S, O
#     for i in range(7):
#         paths.append(f"\\Data\\Streams\\FEED\\Input\\ELEM\\NCPSD\\WASTES\\ULTANAL\\#{i}")
#     # Sulfur analysis: 3 entries
#     for i in range(3):
#         paths.append(f"\\Data\\Streams\\FEED\\Input\\ELEM\\NCPSD\\WASTES\\SULFANAL\\#{i}")
#     return paths

from optimization.aspen_paths import ASPEN_PATHS

def prepare_direct_combustion_inputs(x_input):
    """
    Prepare input values for direct combustion Aspen simulation.

    Input vector x_input:
    [ 0] C_in         (wt%)
    [ 1] H_in         (wt%)
    [ 2] N_in         (wt%)
    [ 3] S_in         (wt%)
    [ 4] O_in         (wt%)
    [ 5] VM_in        (%)
    [ 6] FC_in        (%)
    [ 7] Ash_in       (%)
    """
    # Use predefined paths from config
    paths = (
            ASPEN_PATHS["direct"]["inputs"]["prox"]
            + ASPEN_PATHS["direct"]["inputs"]["ult"]
            + ASPEN_PATHS["direct"]["inputs"]["sulfur"]
    )

    # Rearranged input values for Aspen stream
    values = [
        0,                         # PROXANAL #0: Unused
        x_input[6],                # PROXANAL #1: FC
        x_input[5],                # PROXANAL #2: VM
        x_input[7],                # PROXANAL #3: Ash (Proximate)

        x_input[7],                # ULTANAL #0: Ash (Ultimate)
        x_input[0],                # ULTANAL #1: C
        x_input[1],                # ULTANAL #2: H
        x_input[2],                # ULTANAL #3: N
        100 - sum(x_input[i] for i in [0, 1, 2, 3, 4]),  # ULTANAL #4: Cl
        x_input[3],                # ULTANAL #5: S
        x_input[4],                # ULTANAL #6: O

        0,                         # SULFANAL #0: Unused
        0,                         # SULFANAL #1: Unused
        x_input[3]                 # SULFANAL #2: S again
    ]

    return {
        #"path": generate_directcomb_parameter_paths(),
        "path": paths,
        "value": values
    }
