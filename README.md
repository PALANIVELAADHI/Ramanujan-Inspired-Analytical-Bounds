# Ramanujan-Bounded Logical Error Rates in SiC Quantum Architectures

This repository provides an **analytical–numerical framework** for bounding logical error rates in silicon carbide (SiC) defect-based quantum architectures at room temperature.  
The approach combines **Hardy–Ramanujan partition asymptotics**, **mock theta function corrections**, and **high-performance Monte Carlo simulations** to model thermally correlated defect ensembles and their impact on logical error probabilities.

The framework is designed to bridge **combinatorial number theory** and **scalable quantum hardware modeling**, with particular relevance to wafer-scale SiC platforms.

---

## Key Features

- **Ramanujan-inspired analytical bounds** on logical error probabilities
- **Mock theta function corrections** capturing irregular thermal and combinatorial effects
- **High-performance Monte Carlo core (C++)** for large SiC lattices
- **Python driver layer** for analytical corrections, visualization, and parameter sweeps
- Designed for **room-temperature operation** and fault-tolerant threshold analysis

---

## Repository Structure

```text
/src
 ├── mc_core.cpp        # High-performance Monte Carlo engine (C++)
 ├── mc_core.pyx        # Cython interface to C++ core
 ├── setup.py           # Build script for C++/Python integration
 └── run_simulation.py  # Python driver with mock theta corrections
