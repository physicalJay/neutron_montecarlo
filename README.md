# Neutron Monte Carlo — Reactor Transport (Python)

This project implements a Monte Carlo simulation for neutron transport in fissile materials
(e.g., U-235, U-238, natural uranium), estimating parameters like k-effective and mean free path.

## Features
- Multi-generation neutron tracking
- Energy-dependent cross-section sampling (ENDF/B-based)
- Comparison of fissile materials and moderator performance
- Visualization: flux spectrum, neutron energy histogram, k_eff per generation

## Getting Started
```bash
git clone https://github.com/physicalJay/neutron_montecarlo.git
cd neutron_montecarlo
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python src/nuclear_mc_main.py
```

## Repository Structure
```
neutron_montecarlo/
├── README.md
├── requirements.txt
├── src/
│   ├── nuclear_mc_main.py
│   ├── homogeneous_reactor.py
│   ├── endf_parser.py
│   └── experimental/        # Scratch/test scripts (optional)
├── data/
│   └── ENDF_samples/        # Sample processed cross-section files
├── results/                 # Output figures, data tables
└── docs/
    ├── theory_notes.md
    └── summary.pdf (optional)
```

## License
MIT
