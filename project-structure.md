📦 MIRA_project/
├── .idea/                       # IDE configuration files (e.g., IntelliJ/PyCharm)
├── ann_models/                  # Trained Artificial Neural Network models
│   └── htc/                     # ANN models specific to HTC process
│       └── readme.md
├── aspen_models/               # Aspen Plus simulation files
│   ├── combustion/
│   ├── digestion/
│   ├── htc/
│   ├── ptx/
│   └── pyrolysis/
├── config/                     # Configuration files
│   ├── config.py               # Python utilities for configuration loading
│   ├── config.yaml             # Main YAML config file
│   └── run_config.json         # Scenario-specific runtime configuration
├── data/                       # Input and output data files
│   ├── datasets/               # Input feedstock or experimental datasets
│   └── figures/                # Saved plots and visualizations
├── engine/                     # Core engine logic
│   ├── model_training/         # ANN training pipelines (if implemented)
│   ├── optimizer/              # Optimization engine
│   │   └── pso/                # Particle Swarm Optimization implementation
│   │       ├── fitness.py
│   │       ├── logger.py
│   │       ├── particle.py
│   │       ├── pso.py
│   │       ├── pso_runner.py
│   │       └── velocity_update.py
│   └── simulation/             # Aspen & ANN interface modules
│       ├── ann_predictor.py    # ANN inference engine
│       ├── hybrid_runner.py    # Hybrid ANN–Aspen simulation
│       ├── interface.py        # Interface functions for simulation
│       ├── mock_runner.py      # Mock simulation (for testing)
│       └── prepare_paths.py    # Aspen input/output path preparation
├── logs/                       # Run logs (e.g., optimization logs, results)
├── process_models/             # Unit operation model folders
│   ├── combustion/
│   ├── digestion/
│   ├── htc/
│   ├── ptx/
│   └── pyrolysis/
├── visualization/              # Plotting and result visualization scripts
├── LICENSE
├── project-structure.md        # 📄 You are here
├── README.md                   # Project overview and instructions
├── requirements.txt            # Python dependencies
├── run.py                      # Main entrypoint for running PSO optimization
└── run_pareto.py               # Script for running Pareto front optimization
