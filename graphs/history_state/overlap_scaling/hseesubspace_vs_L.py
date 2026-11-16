"""
Final error and minimum gap vs circuit length (hseesubspace).

For each circuit length L (the GHZ circuit H(0), CNOT(0 -> 1), padded with
identities up to length L), runs the moving-well evolution in the history
subspace (history_state.HistoryStateEvolutionExactEquivSubspace) with
T = tau L and amplitude ratio eta, and plots the error
1 - |<history state|psi(T)>|^2 and the minimum gap along the path.

Output:
    images/history_state/overlap_scaling/hseesubspace_{overlap,min_gap}_vs_L_eta=<eta>.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from eigensolvers import find_blocks, lowest_eigenpairs
from history_state import (
    HistoryState,
    HistoryStateEvolutionExactEquivSubspace,
    vector_basis_change,
)

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
path = f"{repo}/images/history_state/overlap_scaling/"
os.makedirs(path, exist_ok=True)

nqubits = 2
T1 = 50
T2 = 50
step = 1
tau = 200
eta = 50.0

nL = 20
gateslist = [("H", 0), ("CNOT", 1, 0)]
overlaps = []
min_gap = []

for i in range(nL):
    print(f"Running for L={len(gateslist) + i}")
    gateslisti = gateslist + [("I", 0)] * i
    evo = HistoryStateEvolutionExactEquivSubspace(
        gateslisti,
        n_qubits=nqubits,
        T1=T1,
        T2=T2,
        T=tau * (len(gateslisti)),
        step=step,
        eta=eta,
        tau=tau,
    )
    states, times, H = evo.evolution()
    hist_state = HistoryState(gateslisti, n_qubits=nqubits, eta=eta)
    overlaps.append(
        np.abs(
            np.vdot(
                hist_state.history_state(),
                vector_basis_change(states[-1], gateslisti, nqubits),
            )
        )
        ** 2
    )
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
plt.savefig(f"{path}hseesubspace_overlap_vs_L_eta={eta}.png")
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(range(2, nL + 2, 1), min_gap, ".-", linewidth=1, markersize=4.5)
plt.title("Minimum Gap in Energies vs L")
# plt.yscale("log")
plt.xlabel(r"$L$")
plt.ylabel(r"$\Delta_{min}$")
plt.grid()
plt.savefig(f"{path}hseesubspace_min_gap_vs_L_eta={eta}.png")
plt.show()
