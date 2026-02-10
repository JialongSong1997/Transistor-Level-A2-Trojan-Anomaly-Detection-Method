# Transistor-Level A2 Trojan Detection via Isolation Forest and VF2

This repository provides a two-stage framework for **transistor-level A2 hardware Trojan detection** based on **graph analysis** and **unsupervised anomaly screening**.  
The proposed pipeline combines **Isolation Forest–based pre-screening** with **VF2 subgraph isomorphism matching** to efficiently and accurately identify A2 Trojan structures from CDL netlists.

---

## Overview

The detection framework consists of two main modules:

### 1. IF_A2.py — Isolation Forest–Based Pre-screening

`IF_A2.py` is responsible for **preprocessing CDL netlists** and performing a coarse-grained screening of suspicious transistors (or nets).

- Parses the input `.cdl` netlist and constructs a graph representation.
- Extracts structural features of transistors/nets.
- Applies the **Isolation Forest** algorithm to identify anomalous candidates.
- Outputs a list of suspicious transistors/nets, which are used as expansion centers in the next stage.

This step significantly reduces the search space for subsequent graph matching.

---

### 2. VF2_A2.py — Subgraph Expansion and VF2 Matching

`VF2_A2.py` takes the suspicious candidates detected by `IF_A2.py` as input and performs fine-grained structural verification.

- Expands a local subgraph around each suspicious transistor/net.
- Uses the **VF2 subgraph isomorphism algorithm** to match the expanded subgraph against a predefined **A2 Trojan template**.
- Outputs the final detection results, indicating whether the A2 Trojan structure is present.

This stage focuses on **structural consistency** and provides accurate Trojan identification.

---

## Workflow

```text
CDL Netlist
     │
     ▼
IF_A2.py (Isolation Forest Screening)
     │
     ▼
Suspicious Transistor / Net List
     │
     ▼
VF2_A2.py (Subgraph Expansion + VF2 Matching)
     │
     ▼
Final A2 Trojan Detection Result

---

## Citation
If you find this code useful in your research, please consider citing the following paper:
J. Song, J. Zhang, X. Hu, Y. Zhang, J. He and J. Tan, ”Transistor-Level A2 Trojan
Anomaly Detection Method” IEEE International Conference onElectronic Engineering and Information Systems

---
