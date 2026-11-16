"""
Error vs total time for several time steps (hs, random circuit).

Takes the index k of the circuit length L = Llist[k] as its only argument
(python hs_randqc.py k). For each time step in steplist, computes the error
1 - |<history state|psi(T)>|^2 of the adiabatic evolution on a grid of total
times T (Tlist).

Model: history_state.HistoryStateEvolution, the history-state Hamiltonian of
Aharonov et al. with a unary clock.

Circuit: X(0), H(1), CNOT(0 -> 1), Y(1), padded with identities I(0) up to
length L.

Saves Tlist, steplist and errorslist, plotted by
graphs/history_state/step_search/step_search.py. Parameters at the top of
the script.

Output:
    generated/history_state/step_search/hs_randqc_N={n_qubits}_L={L}.npz
"""

import os
import sys
import time

import numpy as np

from history_state import HistoryState, HistoryStateEvolution

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

start = time.perf_counter()

steplist = [0.1, 0.25, 0.5, 1, 2]
Llist = [3, 4, 5, 6]
k = int(sys.argv[1])
L = Llist[k]
gateslist = [("X", 0), ("H", 1), ("CNOT", 1, 0), ("Y", 1)] + [("I", 0)] * (L - 2)
n_qubits = 2
hist_state = HistoryState(gateslist, n_qubits=n_qubits)
errorslist = []
Tlist = np.linspace(0.1, L**4, 100)

for step in steplist:
    overlaps = []
    for T in Tlist:
        evo = HistoryStateEvolution(gateslist, n_qubits=n_qubits, T=T, step=step)
        evo.evolve_state()
        overlaps.append(
            np.abs(np.vdot(hist_state.history_state(), evo.final_state))
            ** 2
        )

    errors = [1 - overlap for overlap in overlaps]
    errorslist.append(errors)

results_dir = f"{repo}/generated/history_state/step_search/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

np.savez(
    f"{results_dir}hs_randqc_N={n_qubits}_L={L}.npz",
    Tlist=Tlist,
    steplist=np.array(steplist),
    errorslist=np.array(errorslist, dtype=object),
)
end = time.perf_counter()

elapsed = end - start
print("Completed!! ")
print(f"Results saved in {results_dir}hs_randqc_N={n_qubits}_L={L}.npz")
print(f"Elapsed time: {elapsed:.2f} seconds")
