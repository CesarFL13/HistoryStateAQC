"""
Final error vs total time (hs3l).

Runs the evolution of history_state.HistoryStateEvolution3Local (J = L^4)
for the 4-qubit GHZ circuit H(0), CNOT(0 -> 1), CNOT(1 -> 2), CNOT(2 -> 3)
on a grid of total times T and plots the error
1 - |<history state|psi(T)>|^2, with the reference T = L^4.

Output:
    images/history_state/overlap_scaling/hs3l_overlap_vs_T.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from history_state import HistoryState, HistoryStateEvolution3Local

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
path = f"{repo}/images/history_state/overlap_scaling/"
os.makedirs(path, exist_ok=True)

Tlist = np.linspace(0.1, 4**5, 100)
overlaps = []
gateslist = [("H", 0), ("CNOT", 1, 0), ("CNOT", 2, 1), ("CNOT", 3, 2)]
n_qubits = 4
hist_state = HistoryState(gateslist, n_qubits=n_qubits)

for T in Tlist:
    evo = HistoryStateEvolution3Local(
        gateslist, n_qubits=n_qubits, T=T, step=0.25, J=(len(gateslist)) ** 4
    )
    evo.evolve_state()
    overlaps.append(
        np.abs(np.vdot(hist_state.history_state(), evo.final_state)) ** 2
    )

errors = [1 - overlap for overlap in overlaps]


plt.figure(figsize=(7, 5))
plt.plot(Tlist, errors, ".-", linewidth=1, markersize=4.5)
plt.title("Error in Overlap of Final State with History State vs Total Time")
plt.yscale("log")
plt.vlines(
    (len(gateslist)) ** 4,
    color="red",
    linestyle="--",
    label=r"$T = L^4$",
    ymin=0,
    ymax=1,
)
plt.xlabel(r"$T$")
plt.ylabel(r"$1-|\langle \psi_{evo}(T)|\eta\rangle|^2$")
plt.legend()
plt.grid()
plt.savefig(f"{path}hs3l_overlap_vs_T.png")
plt.show()
