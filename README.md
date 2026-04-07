# RNA Riboswitch Inverse Design

From-scratch implementation of RNA riboswitch inverse design via bicompatible
sequence sampling, based on the theoretical framework of Huang, Barrett & Reidys
(2021) — "The energy-spectrum of bicompatible sequences."

## Overview

Given two RNA secondary structures in dot-bracket notation (ON-state, OFF-state),
this system outputs ranked RNA sequences predicted to function as riboswitches.

**Core components (all implemented from scratch):**
- Full Turner 2004 nearest-neighbor energy model
- Zuker MFE folding algorithm
- Bicompatible Boltzmann sequence sampler (Huang & Reidys DP)
- Multi-stage sequence optimizer

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Usage

```bash
ribo-design "(((...)))" "((....))" --samples 1000 --top-k 20
```

## References

- Huang, F.W., Barrett, C.L. & Reidys, C.M. (2021). The energy-spectrum of
  bicompatible sequences. *Algorithms for Molecular Biology*, 16, 4.
- Mathews, D.H. et al. (2004). Incorporating chemical modification constraints
  into a dynamic programming algorithm for prediction of RNA secondary structure.
  *PNAS*, 101(20), 7287-7292.
