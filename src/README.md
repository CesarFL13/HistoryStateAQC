# src/

The library, installed with `pip install -e .` (package `historystateaqc`).
Modules are imported by name: `from history_state import HistoryStateEvolution`.

Each model has a `...Hamiltonian` class (H0, H1, ...) and an `...Evolution`
class that runs the adiabatic evolution from H0 to H1 (through
`AdiabaticEvolution`). The variant prefixes used in `build/` and `graphs/` are
listed in the [root README](../README.md#variants). Every module starts with a
docstring that describes it.

## Common tools

| Module | Contents |
|---|---|
| `quantum.py` | Basis states, projectors, gates on n qubits (`build_gate`), commutator norm. |
| `adiabatic_evolution.py` | `AdiabaticEvolution`: discretized evolution under H(s) = (1 − s) H0 + s H1, spectrum along the path, plots. |
| `eigensolvers.py` | Lowest eigenpairs of sparse Hermitian matrices, diagonalizing block by block (the Hamiltonians split into many invariant blocks with degenerate levels, where plain Krylov methods fail). |
| `evolution_recorder.py` | Sparse record of the probabilities during an evolution, to avoid storing every state vector. |
| `plotting_utils.py` | Animation and heatmap of the basis-state probabilities. |

## Models

| Module | Variants |
|---|---|
| `history_state.py` | `hs` (`HistoryStateHamiltonian`), `hs3l` (`…3Local`), `hsee` (`…ExactEquiv`), `hseesubspace` (`…ExactEquivSubspace`), and `HistoryState` (the target history state, with amplitudes eta^t if eta ≠ 1). |
| `history_state_altclockreg.py` | `hs_altclockreg`, `hsee_altclockreg`: the same models with a binary clock register. |
