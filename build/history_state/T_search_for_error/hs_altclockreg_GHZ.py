"""
Total time needed for a target error vs circuit length (hs_altclockreg, GHZ
circuit).

For each circuit length L, the total time T of the adiabatic evolution is
increased in steps of 10 until the error 1 - |<history state|psi(T)>|^2
falls below eps (giving up above T = 1e6).

Model: history_state_altclockreg.HistoryStateEvolution, the history-state
Hamiltonian with a binary clock register.

Circuit: H(0), CNOT(0 -> 1), padded with identities I(0) up to length L.

Saves Llist and Tresults, plotted by
graphs/history_state/T_search_for_error/T_search_for_error.py. Parameters at
the top of the script.

Output:
    generated/history_state/T_search_for_error/hs_altclockreg_GHZ_N={nqubits}_step={step}_error={eps}.npz
"""

import os
import time

import numpy as np

from history_state_altclockreg import (
    HistoryState,
    HistoryStateEvolution,
)

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

start = time.perf_counter()

nqubits = 2
step = 1
eps = 1e-3

nL = 15
gateslist = [("H", 0), ("CNOT", 1, 0)]

Tresults = []
Llist = []
overlaps = []
for i in range(nL):
    gateslisti = gateslist + [("I", 0)] * i
    L = len(gateslisti)
    T = 0
    error = eps + 1

    while error > eps:
        T += 10
        if T > 1e6:
            print("Error did not go below eps for L=", i + 2)
            break
        evo = HistoryStateEvolution(
            gateslisti,
            n_qubits=nqubits,
            T=T,
            step=step,
        )
        evo.evolve_state()
        hist_state = HistoryState(gateslisti, n_qubits=nqubits)
        error = 1 - (
            np.abs(np.vdot(hist_state.history_state(), evo.final_state))
            ** 2
        )
    Llist.append(L)
    Tresults.append(T)
    print(f"L={L}, T={T}, error={error}")

results_dir = f"{repo}/generated/history_state/T_search_for_error/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

np.savez(
    f"{results_dir}hs_altclockreg_GHZ_N={nqubits}_step={step}_error={eps}.npz",
    Llist=Llist,
    Tresults=Tresults,
)
end = time.perf_counter()

elapsed = end - start
print("Completed!! ")
print(
    f"Results saved in {results_dir}hs_altclockreg_GHZ_N={nqubits}_step={step}_error={eps}.npz"
)
print(f"Elapsed time: {elapsed:.2f} seconds")
