"""
Final error vs the time per clock step tau (hsee).

Runs the three-stage moving-well evolution of
history_state.HistoryStateEvolutionExactEquiv for the 3-qubit GHZ circuit
H(0), CNOT(0 -> 1), CNOT(1 -> 2) on a grid of tau, with T = tau L, and plots
the error 1 - |<history state|psi(T)>|^2.

Output:
    images/history_state/overlap_scaling/hsee_overlap_vs_tau.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from history_state import HistoryState, HistoryStateEvolutionExactEquiv

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
path = f"{repo}/images/history_state/overlap_scaling/"
os.makedirs(path, exist_ok=True)

taulist = np.linspace(0.1, 3**5, 100)
nqubits = 3
T1 = 40
T2 = 40
step = 0.5

overlaps = []
gateslist = [("H", 0), ("CNOT", 1, 0), ("CNOT", 2, 1)]
hist_state = HistoryState(gateslist, n_qubits=nqubits)

for t in taulist:
    T = t * len(gateslist)
    evo = HistoryStateEvolutionExactEquiv(
        gateslist, n_qubits=nqubits, T1=T1, T2=T2, T=T, step=step, tau=t
    )
    states = evo.evolution()[0]
    overlaps.append(
        np.abs(np.vdot(hist_state.history_state(), states[-1])) ** 2
    )

errors = [1 - overlap for overlap in overlaps]


plt.figure(figsize=(7, 5))
plt.plot(taulist, errors, ".-", linewidth=1, markersize=4.5)
plt.title("Error in Overlap of Final State with History State vs Tau")
plt.yscale("log")
plt.xlabel(r"$\tau$")
plt.ylabel(r"$1-|\langle \psi_{evo}(T)|\eta\rangle|^2$")
plt.grid()
plt.savefig(f"{path}hsee_overlap_vs_tau.png")
plt.show()
