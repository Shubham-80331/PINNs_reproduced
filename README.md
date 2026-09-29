# Physics-Informed Neural Networks (PINNs) Reproduction

This repository is a modern PyTorch reproduction of the paper:
**Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations** by *M. Raissi, P. Perdikaris, and G.E. Karniadakis*.

## Structure

- `data/`: Contains the original `.mat` datasets from the paper.
- `src/`: Core neural network framework and helper utilities.
- `scripts/`: Scripts demonstrating the various equations.
- `results/`: Output folder for plots and figures.

## Usage

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Setup**
   The datasets are retrieved from the original author's repository. Ensure `setup_data.py` has been run or that the `data/` directory contains the `.mat` files.

3. **Running the Models**
   Navigate to the `scripts/` directory and run any of the Python files:
   ```bash
   python continuous_burgers.py
   python continuous_schrodinger.py
   python continuous_navier_stokes.py
   python discrete_allen_cahn.py
   python discrete_kdv.py
   ```

## Reproduced Examples

### Continuous Time Models
- **Burgers' Equation**: Data-driven solution of the 1D Burgers' equation showing shock formation.
- **Schrodinger Equation**: Data-driven solution with periodic boundary conditions and complex variables.
- **Navier-Stokes Equation**: Data-driven discovery of parameters and unknown pressure field from scattered velocity data.

### Discrete Time Models
- **Allen-Cahn Equation**: Data-driven solution using high-order implicit Runge-Kutta time stepping for massive time steps.
- **Korteweg-de Vries (KdV) Equation**: Data-driven discovery from two distinct temporal snapshots.
