"""
Final error and minimum gap vs circuit length (hs).

For each circuit length L (the GHZ circuit H(0), CNOT(0 -> 1), padded with
identities up to length L), runs the evolution of
history_state.HistoryStateEvolution with T = L^4 and plots the error
1 - |<history state|psi(T)>|^2, the minimum gap of H(s) along the path, and
the value of s where it occurs.

Output:
    images/history_state/overlap_scaling/hs_{overlap,min_gap,min_gap_positions}_vs_L.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from eigensolvers import find_blocks, lowest_eigenpairs
from history_state import (
    HistoryState,
    HistoryStateEvolution,
)

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
path = f"{repo}/images/history_state/overlap_scaling/"
os.makedirs(path, exist_ok=True)

nqubits = 2
step = 1

nL = 8
gateslist = [("H", 0), ("CNOT", 1, 0)]
overlaps = []
min_gap = []
min_gap_positions = []
for i in range(nL):
    print(f"Running for L={i+2}")
    gateslisti = gateslist + [("I", 0)] * i
    evo = HistoryStateEvolution(
        gateslisti,
        n_qubits=nqubits,
        T=(len(gateslisti)) ** 4,
        step=step,
    )
    evo.evolve_state()
    hist_state = HistoryState(gateslisti, n_qubits=nqubits)
    overlaps.append(
        np.abs(np.vdot(hist_state.history_state(), evo.final_state)) ** 2
    )
    H = evo.evolve_H()
    # Two lowest levels at every step, block by block (the blocks of the
    # union of all the patterns are valid for every H in the list)
    blocks = find_blocks(*H)
    gap_list = []
    for i in range(len(H)):
        eigenvalues = lowest_eigenpairs(H[i], k=2, blocks=blocks)[0]
        gap_list.append(eigenvalues[1] - eigenvalues[0])
    min_gap_positions.append(np.argmin(gap_list) * step / (len(gateslisti)) ** 4)
    min_gap.append(np.min(gap_list))
errors = [1 - overlap for overlap in overlaps]


plt.figure(figsize=(7, 5))
plt.plot(range(2, nL + 2, 1), errors, ".-", linewidth=1, markersize=4.5)
plt.title("Error in Overlap of Final State with History State vs L")
plt.yscale("log")
plt.xlabel(r"$L$")
plt.ylabel(r"$1-|\langle \psi_{evo}(T)|\eta\rangle|^2$")
plt.grid()
plt.savefig(f"{path}hs_overlap_vs_L.png")
plt.show()


plt.figure(figsize=(7, 5))
plt.plot(range(2, nL + 2, 1), min_gap, ".-", linewidth=1, markersize=4.5)
plt.title("Minimum Gap in Energies vs L")
# plt.yscale("log")
plt.xlabel(r"$L$")
plt.ylabel(r"$\Delta_{min}$")
plt.grid()
plt.savefig(f"{path}hs_min_gap_vs_L.png")
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(range(2, nL + 2, 1), min_gap_positions, ".-", linewidth=1, markersize=4.5)
plt.title("Minimum Gap Positions vs L")
# plt.yscale("log")
plt.xlabel(r"$L$")
plt.ylabel(r"$s_{min \Delta}$")
plt.grid()
plt.savefig(f"{path}hs_min_gap_positions_vs_L.png")
plt.show()
