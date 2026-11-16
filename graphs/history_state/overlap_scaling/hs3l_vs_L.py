"""
Final error and minimum gap vs circuit length (hs3l).

For each circuit length L (the GHZ circuit H(0), CNOT(0 -> 1), padded with
identities up to length L), runs the evolution of
history_state.HistoryStateEvolution3Local with T = L^4 and plots the error
1 - |<history state|psi(T)>|^2 and the minimum gap of H(s) along the path.

Output:
    images/history_state/overlap_scaling/hs3l_{overlap,min_gap}_vs_L.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from eigensolvers import find_blocks, lowest_eigenpairs
from history_state import (
    HistoryState,
    HistoryStateEvolution3Local,
)

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
path = f"{repo}/images/history_state/overlap_scaling/"
os.makedirs(path, exist_ok=True)

nqubits = 2
step = 1

nL = 9
gateslist = [("H", 0), ("CNOT", 1, 0)]
overlaps = []
min_gap = []
for i in range(nL):
    gateslisti = gateslist + [("I", 0)] * i
    evo = HistoryStateEvolution3Local(
        gateslisti,
        n_qubits=nqubits,
        T=(len(gateslisti)) ** 4,
        step=step,
        J=(len(gateslisti)) ** 6,
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
    min_gap.append(np.min(gap_list))
errors = [1 - overlap for overlap in overlaps]


plt.figure(figsize=(7, 5))
plt.plot(range(2, nL + 2, 1), errors, ".-", linewidth=1, markersize=4.5)
plt.title("Error in Overlap of Final State with History State vs L")
plt.yscale("log")
plt.xlabel(r"$L$")
plt.ylabel(r"$1-|\langle \psi_{evo}(T)|\eta\rangle|^2$")
plt.grid()
plt.savefig(f"{path}hs3l_overlap_vs_L.png")
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(range(2, nL + 2, 1), min_gap, ".-", linewidth=1, markersize=4.5)
plt.title("Minimum Gap in Energies vs L")
# plt.yscale("log")
plt.xlabel(r"$L$")
plt.ylabel(r"$\Delta_{min}$")
plt.grid()
plt.savefig(f"{path}hs3l_min_gap_vs_L.png")
plt.show()
