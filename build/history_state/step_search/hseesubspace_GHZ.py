"""
Error vs total time for several time steps (hseesubspace, GHZ circuit).

Takes the index k of the circuit length L = Llist[k] as its only argument
(python hseesubspace_GHZ.py k). For each time step in steplist, computes the
error 1 - |<history state|psi(T)>|^2 of the adiabatic evolution on a grid of
total times T (Tlist).

Model: history_state.HistoryStateEvolutionExactEquivSubspace, the
three-stage moving-well evolution restricted to the (L + 1)-dimensional
history subspace, tau = T / L.

Circuit: H(0), CNOT(0 -> 1), padded with identities I(0) up to length L.

Saves Tlist, steplist and errorslist, plotted by
graphs/history_state/step_search/step_search.py. Parameters at the top of
the script.

Output:
    generated/history_state/step_search/hseesubspace_GHZ_N={n_qubits}_T1={T1}_T2={T2}_eta={eta}_L={L}.npz
"""

import os
import sys
import time

import numpy as np

from history_state import (
    HistoryState,
    HistoryStateEvolutionExactEquivSubspace,
    vector_basis_change,
)

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

start = time.perf_counter()

steplist = [0.1, 0.25, 0.5, 1, 2]
Llist = [3, 4, 5, 6, 7]
k = int(sys.argv[1])
L = Llist[k]
gateslist = [("H", 0), ("CNOT", 1, 0)] + [("I", 0)] * (L - 2)
n_qubits = 2
errorslist = []
Tlist = np.linspace(0.1, L**4, 100)
T1 = 40
T2 = 40
eta = 1.0

hist_state = HistoryState(gateslist, n_qubits=n_qubits, eta=eta)

for step in steplist:
    overlaps = []
    for T in Tlist:
        evo = HistoryStateEvolutionExactEquivSubspace(
            gateslist,
            n_qubits=n_qubits,
            T1=T1,
            T2=T2,
            T=T,
            step=step,
            tau=T / L,
            eta=eta,
        )
        states = evo.evolution()[0]
        overlaps.append(
            np.abs(
                np.vdot(
                    hist_state.history_state(),
                    vector_basis_change(states[-1], gateslist, n_qubits),
                )
            )
            ** 2
        )

    errors = [1 - overlap for overlap in overlaps]
    errorslist.append(errors)

results_dir = f"{repo}/generated/history_state/step_search/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

np.savez(
    f"{results_dir}hseesubspace_GHZ_N={n_qubits}_T1={T1}_T2={T2}_eta={eta}_L={L}.npz",
    Tlist=Tlist,
    steplist=np.array(steplist),
    errorslist=np.array(errorslist, dtype=object),
)
end = time.perf_counter()

elapsed = end - start
print("Completed!! ")
print(
    f"Results saved in {results_dir}hseesubspace_GHZ_N={n_qubits}_T1={T1}_T2={T2}_eta={eta}_L={L}.npz"
)
print(f"Elapsed time: {elapsed:.2f} seconds")
