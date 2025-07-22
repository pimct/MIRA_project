MIRA_PSO_Optimization/
├── main.py                      # Entry point: run PSO
├── config.py                    # Central config: bounds, feeds, prices
│
├── data/
│   └── htc_dataset.csv          # Raw ANN training data
│
├── model_training/
│   └── train_htc_ann.py         # Trains + selects ANN architecture
│   └── hidden_nodes_mse_oceanblue.png
│
├── models/
│   ├── ann/                     # Saved ANN model + scalers
│   │   ├── htc_ann_model.pkl
│   │   ├── htc_input_scaler.pkl
│   │   └── htc_output_scaler.pkl
│   │
│   └── aspen/                   # Aspen simulation files
│       ├── direct.apw
│       └── htc.apw
│
├── simulation/
│   ├── ann_predictor.py         # Load ANN model and predict
│   ├── aspen_runner.py          # Interface to run Aspen simulation
│   └── result_parser.py         # Extract revenue, CO₂, etc. from Aspen
│
├── optimization/
│   ├── pso.py                   # PSO algorithm
│   ├── fitness_function.py      # Cost function using ANN + Aspen
│   └── history_logger.py        # Log particles, best results
│
├── results/
│   ├── best_solution.json       # Best result from optimization
│   └── iteration_logs.csv       # PSO iteration history
│
└── README.md                    # Project overview and instructions
