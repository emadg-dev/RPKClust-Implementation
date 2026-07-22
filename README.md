# RPKClust: Robust Protocol Keyword Clustering

## Project Purpose
This project provides a complete, scratch-built Python implementation of the **RPKClust** algorithm, designed to cluster unknown binary protocol messages without relying on computationally expensive Multiple Sequence Alignment (MSA). This implementation is built for university-level research presentations and benchmark comparisons.

## Algorithm Overview
RPKClust exploits the structural properties of network protocols, which typically consist of a Fixed-Offset Region (FOR) followed by a Non-Fixed-Offset Region (NFOR). 
1. **Boundary Identification:** Uses 6 heuristic semantic rules (Constant, Sequence, Timestamp, Sparse, Address, Checksum) to identify the boundary $B$ between the FOR and NFOR.
2. **Candidate Generation:** Slides a window over the FOR to extract candidates, and uses TLV (Type-Length-Value) pattern matching to extract variable-length candidates from the NFOR.
3. **Bayesian Inference:** Evaluates candidates using a Two-Stage Bayesian model. It computes a Bit-usage Likelihood ($p_{bit}$), a Position Likelihood ($p_{offset}$), and a prior probability ($p_f$) to rank candidates and select the true protocol keyword.
4. **Clustering:** Groups messages based on the values of the highest-ranked keyword.

## Installation
Ensure you have Python 3.8+ installed.
```bash
pip install -r requirements.txt