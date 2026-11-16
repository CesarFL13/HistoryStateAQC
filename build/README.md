# build/

Scripts that compute results and save them to
`generated/history_state/<study>/` (same path as the script). The script name
is the prefix of the files it writes (`hs3l_GHZ.py` → `hs3l_GHZ_N=2_L=4.npz`).
Parameters are set at the top of each script, and its docstring describes the
model, the circuit and the output file.

The circuit (`GHZ` or `randqc`) is padded with identities up to length L
(`_v2`: identities in the middle; `_v3`: padded with X X pairs).

## history_state/T_search_for_error/

For each L, increases the total time T until the error
1 − |⟨history state|ψ(T)⟩|² falls below `eps`. Saves `Llist`, `Tresults`.

Output: `{variant}_{circuit}_N={n}_step={step}_error={eps}[...].npz`, with
`_T1=…_T2=…_eta=…` for `hsee*` and the `_v2` / `_v3` suffix of the GHZ
paddings.

## history_state/step_search/

For one L (its index in `Llist`, given as the first command-line argument:
`python hs_GHZ.py 2`), computes the error vs T for several time steps of the
evolution. Saves `Tlist`, `steplist`, `errorslist`. Output:
`{variant}_{circuit}_N={n}_L={L}[...].npz`.

## history_state/step_bound/

For each L, the time step `sqrt(12 eps / ‖[H0, H1]‖)` given by the bound on
the error of the discretized evolution in terms of the commutator norm. No
evolution is run. Saves `Llist`, `steplist`. Output:
`{variant}_{circuit}_N={n}_error={eps}.npz`.
