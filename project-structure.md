MIRA/
├── config/                             # Global configs and experiment templates
│   ├── config.yaml                     # Global constants, file paths
│   ├── run_config.json                 # Example full-run input (feed, scenario, system)
│   ├── feed_data.csv                   # Feedstock compositions and availability
│   └── objective_weights.json          # Weights for CO₂, revenue, etc. per scenario

├── engine/                              # Optimization, simulation, evaluation core
│   ├── optimizer/
│   │   ├── pso.py                      # PSO algorithm
│   │   ├── decision_space.py           # Decision variables per process
│   │   └── scenario_manager.py         # CO₂-, revenue-, or balance-focused control
│   ├── simulation/
│   │   ├── prepare_paths.py            # Prepare setup and extract results from Aspen streams
│   │   ├── hybrid_runner.py            # Main engine for combining ANN + Aspen
│   │   ├── surrogate_runner.py         # Run-only ANN for speed
│   │   └── process_interface.py        # Base class template for all process types
│   ├── evaluation/
│   │   └── objective_functions.py      # CO₂, revenue, energy, emergy, etc.
│   ├── data_io/
│   │    ├── data_loader.py               # Load configs, feed info
│   │    └── result_writer.py            # Save and checkpoint results
│   └── model_training/                           ◀️ NEW: Central training module
│       ├── train_htc_ann.py                      # HTC-specific ANN training script
│       ├── train_pyrolysis_ann.py                # 🔄 Future process
│       ├── architecture_optimizer.py             # Shared logic for tuning nodes/layers
│       ├── ann_utils.py                          # Shared normalization, plotting, scoring
│       └── README.md                             # How to use training scripts

├── process_models/                      # All valorization processes (modular)
│   ├── htc/
│   │   ├── htc_process.py              # Implements ProcessInterface
│   │   ├── prepare_inputs.py
│   │   ├── run_aspen.py
│   │   └── ann_predictor.py
│   ├── combustion/
│   │   ├── combustion_process.py
│   │   └── run_aspen.py
│   ├── pyrolysis/                      # 🔄 Future
│   │   ├── pyrolysis_process.py
│   │   └── run_aspen.py
│   ├── digestion/                      # 🔄 Future
│   └── ptx/                            # 🔄 Future

├── aspen_models/                        # Process-specific Aspen Plus files & paths
│   ├── htc/
│   │   ├── HTC_250.apw
│   │   ├── HTC_template.inp
│   │   └── htc_paths.yaml              # Aspen variable paths (input/output)
│   ├── combustion/
│   │   ├── Direct_combustion.apw
│   │   └── combustion_paths.yaml
│   ├── pyrolysis/
│   │   ├── Pyrolysis_model.apw
│   │   └── pyrolysis_paths.yaml
│   └── README.md                       # Model version history and usage notes

├── ann_models/                           # Trained surrogate models by process
│
│   ├── htc/                              # HTC-specific ANN and scalers
│   │   ├── htc_model.pkl
│   │   ├── htc_scaler_x.pkl
│   │   ├── htc_scaler_y.pkl
│   │   └── metadata.json                # Optional: model config, date, metrics
│   ├── pyrolysis/                        # Pyrolysis model files (future)
│   │   ├── pyrolysis_model.pkl
│   │   ├── pyrolysis_scaler_x.pkl
│   │   ├── pyrolysis_scaler_y.pkl
│   │   └── metadata.json
│   ├── combustion/                       # Combustion model files (if any)
│   │   └── ...
│
│   └── README.md                         # Documentation of model versioning

├── interfaces/                          # CLI, dashboard, and exhibit control
│   ├── cli.py                          # Command-line interface
│   ├── dashboard_plot.py               # Plots: 2D/3D surfaces, Pareto, radar
│   └── exhibit_board/                 # Optional LED control for public showcase
│       ├── led_controller.py
│       └── layout_map.json

├── data/                                # Output and logs
│   ├── datasets/                         # ⬅️ Raw datasets for model training
│   │   ├── htc_dataset.csv
│   │   ├── pyrolysis_dataset.csv         # 🔄 Future
│   │   └── README.md                     # Describes columns, units, origin
│   ├── results/                          # ⬅️ Outputs from PSO/Hybrid simulations
│   │   ├── CO2-focused_htc_combustion.csv
│   │   ├── Revenue-focused_htc.csv
│   │   ├── Balanced_pyrolysis_combustion.csv
│   │   ├── checkpoints/                 # ⬅️ Optional: autosaved during long PSO
│   │   │   ├── run_001.csv
│   │   │   └── run_002.csv
│   │   └── README.md                    # Format of result files
│   ├── logs/                             # ⬅️ Run-time logs (e.g., simulation, error trace)
│   │   ├── mira_run_2025_07_23.log
│   │   └── debug_simulation_temp.log
│   └── figures/                          # ⬅️ Plots exported during training or analysis
│

├── docs/                                # Documentation and guides
│   ├── setup_guide.md
│   ├── simulation_structure.md         # How to connect Aspen <-> model
│   ├── process_extension_guide.md      # How to add new process modules
│   └── scenarios.md

├── utils/                               # Utility tools
│   ├── logging_config.py
│   ├── unit_conversion.py
│   └── interpolate_surfaces.py         # RBF interpolation for plotting

├── main.py                              # Entry point for running simulations
└── README.md
