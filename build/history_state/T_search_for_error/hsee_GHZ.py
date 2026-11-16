"""
Total time needed for a target error vs circuit length (hsee, GHZ circuit).

For each circuit length L, the total time T of the adiabatic evolution is
increased in steps of 10 until the error 1 - |<history state|psi(T)>|^2
falls below eps (giving up above T = 1e6).

Model: history_state.HistoryStateEvolutionExactEquiv, the three-stage
evolution with a moving potential well (H0 -> HI(t) -> H1), tau = T / L.

Circuit: H(0), CNOT(0 -> 1), padded with identities I(0) up to length L.

Saves Llist and Tresults, plotted by
graphs/history_state/T_search_for_error/T_search_for_error.py. Parameters at
the top of the script.

Output:
    generated/history_state/T_search_for_error/hsee_GHZ_N={nqubits}_T1={T1}_T2={T2}_eta={eta}_step={step}_error={eps}.npz
"""

import os
import time

import numpy as np

from history_state import (
    HistoryState,
    HistoryStateEvolutionExactEquiv,
)

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

start = time.perf_counter()

nqubits = 2
step = 1
eps = 1e-3
T1 = 40
T2 = 40
eta = 5.0
nL = 6
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
        tau = T / L
        if T > 1e6:
            print("Error did not go below eps for L=", i + 2)
            break
        evo = HistoryStateEvolutionExactEquiv(
            gateslisti,
            n_qubits=nqubits,
            T1=T1,
            T2=T2,
            T=T,
            step=step,
            tau=tau,
            eta=eta,
        )
        states = evo.evolution()[0]
        hist_state = HistoryState(gateslisti, n_qubits=nqubits, eta=eta)
        error = 1 - (
            np.abs(np.vdot(hist_state.history_state(), states[-1]))
            ** 2
        )
    Llist.append(L)
    Tresults.append(T)
    print(f"L={L}, T={T}, error={error}")

results_dir = f"{repo}/generated/history_state/T_search_for_error/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

np.savez(
    f"{results_dir}hsee_GHZ_N={nqubits}_T1={T1}_T2={T2}_eta={eta}_step={step}_error={eps}.npz",
    Llist=Llist,
    Tresults=Tresults,
)
end = time.perf_counter()

elapsed = end - start
print("Completed!! ")
print(
    f"Results saved in {results_dir}hsee_GHZ_N={nqubits}_T1={T1}_T2={T2}_eta={eta}_step={step}_error={eps}.npz"
)
print(f"Elapsed time: {elapsed:.2f} seconds")
