ASPEN_PATHS = {
    "direct": {
        "inputs": {
            "prox": [f"\\Data\\Streams\\FEED\\Input\\ELEM\\NCPSD\\WASTES\\PROXANAL\\#{i}" for i in range(4)],
            "ult":  [f"\\Data\\Streams\\FEED\\Input\\ELEM\\NCPSD\\WASTES\\ULTANAL\\#{i}" for i in range(7)],
            "sulfur": [f"\\Data\\Streams\\FEED\\Input\\ELEM\\NCPSD\\WASTES\\SULFANAL\\#{i}" for i in range(3)],
        },
        "outputs": {
            "power": "\\Data\\Streams\\WNET\\Output\\POWER_OUT"
        }
    },

    "htc": {
        "inputs": {
            "prox": [f"\\Data\\Streams\\FEED\\Input\\ELEM\\NCPSD\\WASTES\\PROXANAL\\#{i}" for i in range(4)],
            "ult":  [f"\\Data\\Streams\\FEED\\Input\\ELEM\\NCPSD\\WASTES\\ULTANAL\\#{i}" for i in range(7)],
            "sulfur": [f"\\Data\\Streams\\FEED\\Input\\ELEM\\NCPSD\\WASTES\\SULFANAL\\#{i}" for i in range(3)],
            "flow": "\\Data\\Streams\\FEED\\Input\\FLOW\\NCPSD\\WASTES",
            "temp": "\\Data\\Blocks\\HTC\\Input\\TEMP",
            "solid_rate": "\\Data\\Blocks\\STMHX1\\Input\\VALUE",
            "char_split": "\\Data\\Blocks\\X\\Input\\FRAC\\CHARPROD",
            "gas_C": "\\Data\\Streams\\DUMMY1\\Input\\FLOW\\MIXED\\C",
            "gas_H2": "\\Data\\Streams\\DUMMY1\\Input\\FLOW\\MIXED\\H2",
            "gas_N2": "\\Data\\Streams\\DUMMY1\\Input\\FLOW\\MIXED\\N2",
            "char_prox": [f"\\Data\\Blocks\\HTC\\Input\\ELEM\\NCPSD\\CHAR\\PROXANAL\\#{i}" for i in range(4)],
            "char_ult": [f"\\Data\\Blocks\\HTC\\Input\\ELEM\\NCPSD\\CHAR\\ULTANAL\\#{i}" for i in range(7)],
            "org_prox": [f"\\Data\\Blocks\\HTC\\Input\\ELEM\\NCPSD\\ORGANICS\\PROXANAL\\#{i}" for i in range(4)],
            "org_ult": [f"\\Data\\Blocks\\HTC\\Input\\ELEM\\NCPSD\\ORGANICS\\ULTANAL\\#{i}" for i in range(7)],
            "char_sulfur": "\\Data\\Blocks\\HTC\\Input\\ELEM\\NCPSD\\CHAR\\SULFANAL\\#2",
            "org_sulfur": "\\Data\\Blocks\\HTC\\Input\\ELEM\\NCPSD\\ORGANICS\\SULFANAL\\#2"
        },
        "outputs": {
            "power": "\\Data\\Streams\\WNET\\Output\\POWER_OUT",
            "char": "\\Data\\Streams\\CHARPROD\\Output\\MASSFLOW\\NCPSD\\CHAR"
        }
    }
}
