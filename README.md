# MIRA: Multi-objective Integrated Resource Allocation for Waste Valorization

This repository contains the implementation of **MIRA (Multi-objective Integrated Resource Allocation)**, a hybrid decision-support framework developed for the intelligent optimization of hydrochar production and energy recovery from various waste streams. This tool was developed as part of the research article:

> **"An Intelligent Plant-Wide Decision-Support Framework for Waste Valorization: Optimizing Hydrochar Production and Energy Recovery"**  
> *Prathana Nimmanterdwong et al.*

## 🔍 Overview

MIRA integrates **artificial neural networks (ANNs)**, **thermodynamic process modeling**, and **particle swarm optimization (PSO)** to optimize process configurations for waste-to-energy systems. It enables dynamic, feedstock-specific optimization across multiple valorization routes, ensuring both **thermodynamic consistency** and **multi-objective performance**.

### **Key Features**

- **Hybrid modeling**: Combines data-driven ANN models with Aspen Plus simulations to maintain mass and energy conservation.
- **Multi-objective optimization**: Simultaneously balances environmental and economic outcomes using PSO.
- **Scenario-based analysis**: Supports comparative optimization under CO₂-focused, revenue-focused, and balanced objectives.
- **Feedstock flexibility**: Validated on three distinct waste types:
  - Organic Household Waste Digestate (OHWD)
  - Municipal Solid Waste (MSW)
  - Agricultural Residue (AGR)
- **Thermochemical pathways**:
  - Direct combustion with energy recovery (as implemented in Thailand’s Phuket WtE plant)
  - Hydrothermal Carbonization (HTC) followed by power generation

### 📊 Optimization Outputs

- Electricity generation 
- Economic returns
- Optimal HTC temperature and char routing fraction (`x_char`, ranging from **0.1 to 0.5**)
- Trade-offs visualized across environmental and economic performance metrics

### 🧠 Method Used

- Python for ANN training and PSO algorithm implementation
- Aspen Plus for thermodynamic modeling and validation
- NumPy, Pandas, Matplotlib for data processing and visualization
- Interfacing via COM automation for coupling Python with Aspen Plus

### 🧪 Reproducibility

This code base is modular and designed for extension to other waste streams and thermochemical processes. Future versions aim to incorporate:
_Future versions aim to incorporate:_
- Logistics and supply chain models  
- Additional conversion technologies (e.g., pyrolysis, gasification)  
- Grid/network integration for energy systems
_Core improvements:_
- Run-continue feature for recovering from simulation crashes


## 📄 Citation

If you use this framework in your research or project, please cite the following paper:

Nimmanterdwong, Prathana and Srifa, Atthapon and Prechthai, Tawach and Tuntiwiwattanapun, Nattapong and Piemjaiswang, Ratchanon and Yu, Bor-Yih and Pornaroontham, Phuwadej and Sema, Teerawat and Chalermsinsuwan, Benjapon and Piumsomboon, Pornpote, An Intelligent Plant-Wide Decision-Support Framework for Waste Valorization: Optimizing Hydrochar Production and Energy Recovery. Available at SSRN: https://ssrn.com/abstract=5335234 or http://dx.doi.org/10.2139/ssrn.5335234


