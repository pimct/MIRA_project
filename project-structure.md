MIRA_project/
├── LICENSE                  # Project license information
├── README.md                # Project overview and usage instructions
├── requirements.txt         # List of Python dependencies
├── project-structure.md     # Documentation of the project structure
├── run.py                   # Main entry point for running optimization or preparation
├── run_pareto.py            # Script for running Pareto optimization
├── test.py                  # Test script for the project
├── __pycache__/             # Python bytecode cache (auto-generated)
├── ann_models/              # Artificial Neural Network model files
│   ├── readme.md            # Documentation for ANN models
│   └── htc/                 # HTC-specific ANN models
├── aspen_models/            # Aspen simulation models
│   ├── combustion/          # Combustion process models
│   ├── digestion/           # Digestion process models
│   ├── htc/                 # HTC process models
│   ├── ptx/                 # PTX process models
│   └── pyrolysis/           # Pyrolysis process models
├── config/                  # Configuration files and scripts
│   ├── config.py            # Python configuration module
│   ├── config.yaml          # YAML configuration file
│   ├── run_config.json      # JSON configuration for runs
│   └── __pycache__/         # Bytecode cache for config modules
├── data/                    # Data storage
│   ├── datasets/            # Input datasets
│   └── figures/             # Generated figures and plots
├── docs/                    # Project documentation
├── engine/                  # Core engine modules
│   ├── data_io/             # Data input/output utilities
│   ├── evaluation/          # Model evaluation code
│   ├── model_training/      # Model training scripts
│   ├── optimizer/           # Optimization algorithms (e.g., PSO)
│   └── simulation/          # Simulation modules
├── logs/                    # Logs and results from runs
│   ├── best_result.json     # Best optimization result
│   ├── convergence.csv      # Convergence data for optimization
│   └── particle_*_iteration_*.json # Per-particle, per-iteration logs
├── optimization/            # Optimization-related scripts and modules
├── process_models/          # Process model definitions
└── visualization/           # Visualization scripts and tools
