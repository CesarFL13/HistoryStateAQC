# graphs/

Scripts that make the figures, saved to `images/history_state/<study>/`
(same path as the script). Some read the results of `build/` from
`generated/history_state/<study>/`; the `overlap_scaling` ones run the
evolutions themselves. Parameters are set at the top of each script, and its
docstring describes what it plots and its input and output files.

## history_state/overlap_scaling/

Run the adiabatic evolution and plot the error 1 − |⟨history state|ψ(T)⟩|²
(and, in the `_vs_L` scripts, the minimum gap of H(s)):

| Script | Varies | Fixed |
|---|---|---|
| `hs_vs_L.py`, `hs3l_vs_L.py`, `hsee_vs_L.py`, `hseesubspace_vs_L.py` | circuit length L (GHZ padded with identities) | T = L⁴ (or tau L for `hsee*`) |
| `hs_vs_T.py`, `hs3l_vs_T.py` | total time T | circuit |
| `hs3l_vs_J.py` | clock weight J of the 3-local model | T = L⁴ |
| `hsee_vs_tau.py` | time per clock step tau of the moving well | T1, T2 |

Output: `images/history_state/overlap_scaling/<variant>_{overlap,min_gap,min_gap_positions}_vs_<x>[_eta=…].png`.

## history_state/T_search_for_error/, step_search/, step_bound/

Plot the results of the scripts of the same name in `build/history_state/`:
T needed vs L with a fit, error vs T for each time step (a gif with one frame
per L), and the time step from the bound vs L. The variant, circuit and
parameters to plot are chosen at the top of the script and must match an
existing file in `generated/history_state/<study>/`.
