# Code for "Repeated Viewing of a Narrative Movie Changes Event Timescales in The Brain"

The code in this repository can be used to reproduce the results of Al-Zahli, Aly, and Baldassano, "Repeated Viewing of a Narrative Movie Changes Event Timescales in The Brain." Preprint. August, 2025.

Data from "Learning Naturalistic Temporal Structure in the Posterior Medial Network" was preprocessed using FSL.
All results reported in the manuscript can be reproduced by running main.py.

> ⚠️ Note: Running all permutations, viewings, and event counts across subjects and clips will be computationally intensive (potentially multiple days depending on configuration). We recommend modifying the permutation loops to take advantage of parallelization if available.

## Environment & Dependencies

This code was originally run with:

Python version: 3.7.12;
brainiak version: 0.11

