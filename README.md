# HistoryStateAQC

Numerical study of the equivalence between the quantum circuit model and
adiabatic quantum computation (Aharonov et al.; Wu et al.). A circuit is
encoded in a history state, the ground state of a final Hamiltonian H1, which
is reached by an adiabatic evolution from an easy Hamiltonian H0. The scripts
measure how well the evolution reaches the history state (the error
1 − |⟨history state|ψ(T)⟩|²) and how the total time T, the time step and the
spectral gap scale with the circuit length L.

## Setup

Python 3 (tested with 3.13) with numpy, scipy and matplotlib. From the root
of the repo:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                # installs src/ as the package historystateaqc
```

After the editable install, the modules of `src/` are imported by name from
anywhere (`from history_state import HistoryStateEvolution`), and changes in
`src/` take effect without reinstalling.

## Layout

| Folder | Contents |
|---|---|
| [`src/`](src/README.md) | The library: history-state Hamiltonians, adiabatic evolution, eigensolvers, plotting. |
| [`build/`](build/README.md) | Scripts that compute results and save them to `generated/`. |
| [`graphs/`](graphs/README.md) | Scripts that make the figures (from `generated/` or computing directly) and save them to `images/`. |
| `generated/` | Numerical results (`.npz`) written by `build/`. Created when a script runs; not in git. |
| `images/` | Figures written by `graphs/`. Created when a script runs; not in git. |

## Organization

`build/`, `graphs/`, `generated/` and `images/` share the same tree:

```
<folder>/history_state/<study>/<file>
```

- **study**: what is being measured (see the table below). A study keeps the
  same path in every folder, e.g. `build/history_state/step_search/hs_GHZ.py`
  writes `generated/history_state/step_search/hs_GHZ_N=2_L=4.npz`, which
  `graphs/history_state/step_search/step_search.py` plots into
  `images/history_state/step_search/`.
- **variant**: the model variant goes first in the file name (`hs3l_vs_L.py`,
  `hs3l_GHZ_N=2_L=4.npz`), and parts already given by the folders are dropped.
- Scripts build every path from the repo root
  (`repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))`),
  so they can be run from any working directory.

### Studies

| Study | What it measures | build | graphs |
|---|---|---|---|
| `overlap_scaling` | Final error (and minimum gap of H(s)) vs L, T, J or tau | – | ✓ |
| `T_search_for_error` | Smallest total time T that reaches a target error, vs L | ✓ | ✓ |
| `step_search` | Error vs T for several time steps of the evolution | ✓ | ✓ |
| `step_bound` | Time step from the bound with the commutator norm ‖[H0, H1]‖, vs L | ✓ | ✓ |

### Variants

| Prefix | Model |
|---|---|
| `hs` | Aharonov et al.: unary clock of L qubits, H0 = H_input + H_clockinit + H_clock, H1 = H_input + H_clock + H_gates (`history_state.py`) |
| `hs3l` | 3-local version, with the clock terms weighted by J |
| `hsee` | "Exact equivalence": three-stage evolution H0 → H_I(t) → H1, where H_I has a Gaussian potential well that moves along the clock with speed 1/tau (times T1, T2) |
| `hseesubspace` | `hsee` restricted to the (L + 1)-dimensional history subspace |
| `hs_altclockreg`, `hsee_altclockreg` | `hs` and `hsee` with a binary clock register of ⌊log2 L⌋ + 1 qubits (`history_state_altclockreg.py`) |

Circuits of the `build/` scripts: `GHZ` (H on qubit 0 and a CNOT) or `randqc`
(a fixed 4-gate circuit), padded with identities up to length L. `_v2` and
`_v3` are other paddings of the GHZ circuit (identities in the middle, X X
pairs).

## Running

```bash
python build/history_state/T_search_for_error/hs_GHZ.py
python build/history_state/step_search/hs_GHZ.py 2      # index of L in Llist
python graphs/history_state/T_search_for_error/T_search_for_error.py
python graphs/history_state/overlap_scaling/hs_vs_L.py
```

Parameters (number of qubits, circuit lengths, time step, target error, ...)
are set in variables at the top of each script; every script has a docstring
that says what it computes, which model it uses, and which files it reads and
writes. The scripts of `graphs/` that read `generated/` need the matching
results of `build/` (same variant, circuit and parameters).

## Conventions

- New scripts go in `<folder>/history_state/<study>/`, named with the variant
  first; new results and figures go to the same study in `generated/` and
  `images/`.
- Paths are always built from `repo`, never hardcoded.
- Each script starts with a docstring like the existing ones.
