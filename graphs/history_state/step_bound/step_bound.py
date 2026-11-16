"""
Time step from the commutator bound vs circuit length.

Reads
generated/history_state/step_bound/<variant>_<circuit>_N=<n>_error=<eps>.npz
(from the scripts of build/history_state/step_bound/) and plots the time
step vs L. The variant, circuit and parameters are chosen at the top of the
script.

Output:
    images/history_state/step_bound/<variant>_<circuit>_N=<n>_error=<eps>.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

nqubits = 2
eps = 1e-4
type = "hs"
circuit = "GHZ"  # "GHZ" o "randqc"
altclockreg = True  # True o False

if altclockreg:
    type += "_altclockreg"
results = np.load(
    f"{repo}/generated/history_state/step_bound/{type}_{circuit}_N={nqubits}_error={eps}.npz"
)
Llist = results["Llist"]
steplist = results["steplist"]


plt.figure(figsize=(7, 5))
plt.plot(Llist, steplist, ".-", linewidth=1, markersize=4.5)
plt.title(r"Step needed for $\varepsilon=$" + str(eps) + r" vs L")
# plt.yscale("log")
plt.xlabel(r"$L$")
plt.ylabel(r"$\Delta t$")
plt.grid()
path = f"{repo}/images/history_state/step_bound/"
if not os.path.exists(path):
    os.makedirs(path)
plt.savefig(f"{path}{type}_{circuit}_N={nqubits}_error={eps}.png")
plt.show()
