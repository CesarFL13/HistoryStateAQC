"""
Total time needed for a target error vs circuit length.

Reads generated/history_state/T_search_for_error/<variant>_<circuit>_...npz
(from the scripts of build/history_state/T_search_for_error/) and plots T vs
L with a fit: a power law T ~ L^m in log-log scale (graph = "log"), or a
degree-4 polynomial or a L^4 (mode) otherwise. The variant, circuit and
parameters are chosen at the top of the script and must match an existing
file.

Output:
    images/history_state/T_search_for_error/<name of the data file>[_log].png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

nqubits = 2
step = 1
eps = 1e-3
typehs = "hs"  # "hs", "hs3l", "hsee" o "hseesubspace"
circuit = "GHZ"  # "GHZ" o "randqc"
graph = "log"


altclockreg = True  # True o False
if altclockreg:
    type = typehs + "_altclockreg"
else:
    type = typehs


T1 = 40
T2 = 40
eta = 5.0


v = "_v2"  # "",  "_v2" o "_v3"


if typehs == "hs" or typehs == "hs3l":
    results = np.load(
        f"{repo}/generated/history_state/T_search_for_error/{type}_{circuit}_N={nqubits}_step={step}_error={eps}{v}.npz",
        allow_pickle=True,
    )
elif typehs == "hsee" or typehs == "hseesubspace":
    results = np.load(
        f"{repo}/generated/history_state/T_search_for_error/{type}_{circuit}_N={nqubits}_T1={T1}_T2={T2}_eta={eta}_step={step}_error={eps}{v}.npz",
        allow_pickle=True,
    )

Llist = results["Llist"]
Tresults = results["Tresults"]


# --- MODELOS ---
def model_x4(x, a):
    return a * x**4


def model_poly4(x, a, b, c, d, e):
    return a * x**4 + b * x**3 + c * x**2 + d * x + e


# --- ELEGIR MODELO ---
mode = "poly"  # "x4" o "poly"

if mode == "x4":
    model = model_x4
elif mode == "poly":
    model = model_poly4
else:
    raise ValueError("Modo no válido")

# --- CURVA SUAVE ---
L_fit = np.linspace(min(Llist) * 0.9, max(Llist) * 1.05, 200)

# --- AJUSTE SEGÚN TIPO DE GRÁFICA ---
if graph == "log":
    # Ajuste lineal en log-log → T ~ L^m
    logL = np.log(Llist)
    logT = np.log(Tresults)

    m, b = np.polyfit(logL, logT, 1)

    T_fit = np.exp(b) * L_fit**m
    label_fit = rf"$\sim L^{{{m:.2f}}}$"

else:
    # Ajuste normal (polinómico o x^4)
    params, _ = curve_fit(model, Llist, Tresults)
    T_fit = model(L_fit, *params)

    if mode == "x4":
        a = params[0]
        label_fit = rf"${a:.2e} \, L^4$"

    elif mode == "poly":
        a, b, c, d, e = params
        label_fit = rf"${a:.2e}L^4 + {b:.2e}L^3 + {c:.2e}L^2 + {d:.2e}L + {e:.2e}$"


# --- PLOT ---
plt.figure(figsize=(7, 5))
plt.plot(Llist, Tresults, ".-", linewidth=1, markersize=4.5, label="Simulation points")
plt.plot(L_fit, T_fit, "--", label=label_fit)

plt.title(r"T needed for $\varepsilon=$" + str(eps) + r" vs L")
if graph == "log":
    plt.xscale("log")
    plt.yscale("log")

plt.xlabel(r"$L$")
plt.ylabel(r"$T$")

plt.grid()
plt.legend()

path = f"{repo}/images/history_state/T_search_for_error/"
if not os.path.exists(path):
    os.makedirs(path)
if graph == "log":
    if typehs == "hs" or typehs == "hs3l":
        plt.savefig(
            f"{path}{type}_{circuit}_N={nqubits}_step={step}_error={eps}{v}_log.png"
        )
    elif typehs == "hsee" or typehs == "hseesubspace":
        plt.savefig(
            f"{path}{type}_{circuit}_N={nqubits}_T1={T1}_T2={T2}_eta={eta}_step={step}_error={eps}{v}_log.png"
        )
else:
    if typehs == "hs" or typehs == "hs3l":
        plt.savefig(
            f"{path}{type}_{circuit}_N={nqubits}_step={step}_error={eps}{v}.png"
        )
    elif typehs == "hsee" or typehs == "hseesubspace":
        plt.savefig(
            f"{path}{type}_{circuit}_N={nqubits}_T1={T1}_T2={T2}_eta={eta}_step={step}_error={eps}{v}.png"
        )
plt.show()
