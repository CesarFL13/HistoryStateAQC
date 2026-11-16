"""
Error vs total time for several time steps, as an animation over L.

Reads
generated/history_state/step_search/<variant>_<circuit>_N=<n>_..._L=<L>.npz
for each L in L_values (from the scripts of
build/history_state/step_search/) and makes a gif with one frame per L,
showing the error vs T for each time step. The variant, circuit and
parameters are chosen at the top of the script.

Output:
    images/history_state/step_search/<variant>_<circuit>_N=<n>[...].gif
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ============================================
# LISTA DE VALORES DE L
# ============================================

L_values = [3, 4, 5, 6, 7, 8, 9, 10, 11,12]
nqubits = 2
typehs = "hs"  # "hs", "hs3l", "hsee" o "hseesubspace"
circuit = "GHZ"  # "GHZ" o "randqc"
altclockreg = False  # True o False
if altclockreg:
    type = typehs + "_altclockreg"
else:
    type = typehs
T1 = 40
T2 = 40
eta = 5.0
# ============================================
# FUNCIÓN QUE LEE LOS DATOS
# ============================================


def compute_data(L):
    if typehs == "hs" or typehs == "hs3l":
        results = np.load(
            f"{repo}/generated/history_state/step_search/{type}_{circuit}_N={nqubits}_L={L}.npz",
            allow_pickle=True,
        )
    elif typehs == "hsee" or typehs == "hseesubspace":
        results = np.load(
            f"{repo}/generated/history_state/step_search/{type}_{circuit}_N={nqubits}_T1={T1}_T2={T2}_eta={eta}_L={L}.npz",
            allow_pickle=True,
        )
    steplist = results["steplist"]
    errorslist = results["errorslist"]
    Tlist = results["Tlist"]

    return steplist, errorslist, Tlist


# ============================================
# GIF CON BUCLE IDA/VUELTA
# ============================================

sequence = list(range(len(L_values)))


fig_gif, ax_gif = plt.subplots(figsize=(7, 5))


def animate(i):
    ax_gif.clear()

    idx = sequence[i]
    L = L_values[idx]

    steplist, errorslist, Tlist = compute_data(L)

    for step, errors in zip(steplist, errorslist):
        ax_gif.plot(
            Tlist, errors, ".-", linewidth=1, markersize=4.5, label=f"step = {step}"
        )

    ax_gif.set_title(f"Error in Overlap (L = {L})")
    ax_gif.set_yscale("log")
    ax_gif.set_xlabel(r"$T$")
    ax_gif.set_ylabel(r"$1-|\langle \psi_{evo}(T)|\eta\rangle|^2$")
    ax_gif.legend()
    ax_gif.grid()


ani = animation.FuncAnimation(fig_gif, animate, frames=len(sequence), interval=1200)
path = f"{repo}/images/history_state/step_search/"
if not os.path.exists(path):
    os.makedirs(path)

if typehs == "hs" or typehs == "hs3l":
    ani.save(
        f"{path}{type}_{circuit}_N={nqubits}.gif",
        writer="pillow",
        fps=0.3,
    )
    print(
        f"GIF guardado como {type}_{circuit}_N={nqubits}.gif en la carpeta images/history_state/step_search/"
    )
elif typehs == "hsee" or typehs == "hseesubspace":
    ani.save(
        f"{path}{type}_{circuit}_N={nqubits}_T1={T1}_T2={T2}_eta={eta}.gif",
        writer="pillow",
        fps=0.3,
    )
    print(
        f"GIF guardado como {type}_{circuit}_N={nqubits}_T1={T1}_T2={T2}_eta={eta}.gif en la carpeta images/history_state/step_search/"
    )
