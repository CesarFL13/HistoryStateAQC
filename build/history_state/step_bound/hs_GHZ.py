"""
Time step from the commutator bound vs circuit length (hs, GHZ circuit).

For each circuit length L, the time step sqrt(12 eps / ||[H0, H1]||) given
by the bound on the error of the discretized evolution in terms of the
commutator norm (quantum.commutator_norm_hermitian). No evolution is run.

Model: history_state.HistoryStateHamiltonian, the history-state Hamiltonian
of Aharonov et al. with a unary clock.

Circuit: H(0), CNOT(0 -> 1), padded with identities I(0) up to length L.

Saves Llist and steplist, plotted by
graphs/history_state/step_bound/step_bound.py. Parameters at the top of the
script.

Output:
    generated/history_state/step_bound/hs_GHZ_N={nqubits}_error={eps}.npz
"""

import os
import time

import numpy as np

from history_state import (
    HistoryStateHamiltonian,
)
from quantum import commutator_norm_hermitian

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

start = time.perf_counter()

nqubits = 2

nL = 13
gateslist = [("H", 0), ("CNOT", 1, 0)]

eps = 1e-4
Llist = range(2, nL + 2, 1)
steplist = []

for L in Llist:
    gateslisti = gateslist + [("I", 0)] * (L - 2)
    H = HistoryStateHamiltonian(gateslisti, n_qubits=nqubits, eta=1.0)
    H0 = H.H0()
    H1 = H.H1()
    comm_norm = commutator_norm_hermitian(H0, H1)
    steplist.append(np.sqrt(12 * eps / comm_norm))

results_dir = f"{repo}/generated/history_state/step_bound/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

np.savez(
    f"{results_dir}hs_GHZ_N={nqubits}_error={eps}.npz",
    Llist=Llist,
    steplist=steplist,
)
end = time.perf_counter()

elapsed = end - start
print("Completed!! ")
print(f"Results saved in {results_dir}hs_GHZ_N={nqubits}_error={eps}.npz")
print(f"Elapsed time: {elapsed:.2f} seconds")
