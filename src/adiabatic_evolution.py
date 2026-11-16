"""
Adiabatic evolution under H(s) = (1 - s) H0 + s H1.

AdiabaticEvolution evolves an initial state on a time grid of step
`step` up to the total time T (s = t / T), applying exp(-i H dt) with H
at the midpoint of each step (scipy expm_multiply), and caches what its
plotting methods need: final state, energies, overlap with the ground
state of H1, a sparse record of the probabilities (evolution_recorder)
and the lowest levels of H(s) (eigensolvers). EvolutionProgress prints
the progress messages of the long loops.

The model classes (history_state, history_state_altclockreg) inherit
from it and provide H0 and H1; subclasses can also override H(s) and the
time grid (self.t, self.s).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import expm_multiply

from eigensolvers import (
    LU_BUDGET_MB,
    find_blocks,
    lowest_eigenpairs,
    n_blocks,
    select_eig_method,
)
from evolution_recorder import EvolutionRecorder
from plotting_utils import plot_state_evolution, plot_state_heatmap


class EvolutionProgress:
    """
    Progress messages of an evolution: one when it is created (the start)
    and one every `every` time steps with the current t, plus the last step.

    Used by the evolutions made of several consecutive stages (the
    ExactEquiv classes), which report a single global time and step count:
    call it as progress(step, t) with the global step index and time.
    """

    def __init__(
        self, verbose, every, total_time, total_steps, dim=None, label="evolution"
    ):
        self.verbose = verbose
        self.every = every
        self.total_time = total_time
        self.total_steps = total_steps
        self.label = label
        self._last = 0
        if verbose:
            size = "" if dim is None else f", dim = {dim}"
            print(
                f"[{label}] start: T = {total_time:g}, {total_steps} steps{size}",
                flush=True,
            )

    def __call__(self, step, t):
        # step 0 is the start; a step shared by two stages is reported once
        if not self.verbose or step <= self._last:
            return
        if step % self.every == 0 or step == self.total_steps:
            self._last = step
            print(
                f"[{self.label}] t = {t:g} / {self.total_time:g}  "
                f"(step {step}/{self.total_steps})",
                flush=True,
            )


class AdiabaticEvolution:

    def __init__(
        self,
        H0,
        H1,
        initial_state,
        T,
        step,
        eig_method="auto",
        lu_budget_mb=LU_BUDGET_MB,
        verbose=False,
        verbose_every=5,
    ):
        self.T = T
        self.step = step

        # Progress messages of evolve_state: one when it starts and one
        # every `verbose_every` time steps with the current t. Both can be
        # changed at any time (evo.verbose = True).
        self.verbose = verbose
        self.verbose_every = verbose_every

        # If set (a function of the step index), the progress of this
        # evolution is reported to it instead of being printed: used when it
        # is one stage of a longer evolution (see EvolutionProgress).
        self._progress_hook = None

        # Eigensolver for the ground state of H1 and the spectrum of H(s):
        # "auto", "dense", "shift-invert" or "lobpcg" (see eigensolvers.py).
        # lu_budget_mb: max memory of the shift-invert LU before "auto"
        # switches to lobpcg.
        self.eig_method = eig_method
        self.lu_budget_mb = lu_budget_mb

        self.t = step * np.arange(round(T / step) + 1)
        self.s = self.t / T

        # Hamiltonians: keep them sparse
        self.H0 = H0 if issparse(H0) else csr_matrix(H0)
        self.H1 = H1 if issparse(H1) else csr_matrix(H1)

        # State vectors are dense 1D NumPy arrays
        self.initial_state = np.asarray(initial_state, dtype=complex).reshape(-1)

        # Normalize initial state
        initial_norm = np.linalg.norm(self.initial_state)

        if initial_norm == 0:
            raise ValueError("initial_state cannot be the zero vector.")

        self.initial_state /= initial_norm

        # ---------------------------------------------------------
        # Caches
        # ---------------------------------------------------------

        # Full list of state vectors. Only filled if the evolution is run
        # with store_states=True (it needs n_steps * dim complex numbers).
        self.states = None

        # State at the end of the evolution (s = 1). Always kept, even when
        # store_states=False.
        self.final_state = None

        # Energy of evolved state at every s
        self.energies = None

        # |<E0^{H1}|psi(s)>|^2 at every s (computed during the evolution)
        self.overlaps = None

        # Sparse record of the probabilities |psi_i|^2 at every step
        # (EvolutionRecorder), used by plot_probabilities.
        self.recorder = None

        # Ground state of H1 (computed once, on demand)
        self._h1_ground_state_cache = None

        # Lowest eigenvalues of H(s)
        #
        # Shape:
        #     (n_eigenvalues, len(s))
        #
        self.eigenvalues = None

        # Number of eigenvalues currently stored
        self.n_eigenvalues = 0

    def H(self, s):
        """Hamiltonian at interpolation parameter s."""
        return (1 - s) * self.H0 + s * self.H1

    def evolve_H(self):
        """
        Return the Hamiltonian at every value of s.

        Warning:
            This stores all Hamiltonians in memory.
            For large systems, prefer self.H(s) on demand.
        """
        return [self.H(si) for si in self.s]

    def _h1_ground_state(self):
        """Ground state of H1 (cached)."""
        if self._h1_ground_state_cache is None:
            _, eigenvectors = lowest_eigenpairs(
                self.H1, k=1, method=self.eig_method, lu_budget_mb=self.lu_budget_mb
            )
            self._h1_ground_state_cache = eigenvectors[:, 0]
        return self._h1_ground_state_cache

    def _progress(self, verbose, label, i):
        """Progress message of step i of a loop over the time grid: one at
        the start and one every self.verbose_every steps (and at the end)."""
        if not verbose:
            return
        if self._progress_hook is not None:
            self._progress_hook(i)
            return
        last = len(self.t) - 1
        if i == 0:
            print(
                f"[{label}] start: T = {self.t[-1]:g}, {last} steps, "
                f"dim = {self.H0.shape[0]}",
                flush=True,
            )
        elif i % self.verbose_every == 0 or i == last:
            print(
                f"[{label}] t = {self.t[i]:g} / {self.t[-1]:g}  (step {i}/{last})",
                flush=True,
            )

    def clear_evolution_cache(self):
        """Forget states, energies, overlaps and the probability record, so
        the next evolve_state call runs the evolution again (e.g. with other
        tol_store / top_k)."""
        self.states = None
        self.final_state = None
        self.energies = None
        self.overlaps = None
        self.recorder = None

    def evolve_state(
        self,
        n_eigenvalues=None,
        store_states=False,
        tol_store=1e-8,
        top_k=None,
        verbose=None,
    ):
        """
        Evolve the initial state and optionally calculate the lowest
        n_eigenvalues of H(s) at every evolution step.

        The eigenstate corresponding to the physical state being
        followed is tracked using its overlap with the state from
        the previous step.

        Parameters
        ----------
        n_eigenvalues : int or None
            Number of lowest eigenvalues of H(s) to compute and cache.
        store_states : bool
            If True, keep every state vector in self.states (memory:
            n_steps * dim * 16 bytes). If False (default), only the current
            state is kept in memory during the evolution and self.states
            stays None. Calling evolve_state(store_states=True) after a
            lean run repeats the evolution (it is deterministic) to fill it.
        tol_store : float
            Probabilities <= tol_store are not recorded in self.recorder.
            Choose it well below the tolerance you will plot with.
        top_k : int or None
            Hard cap on the number of states recorded per step.
        verbose : bool or None
            Print a message when the evolution starts and the current t
            every self.verbose_every time steps (same for the spectrum
            loop). None (default) uses self.verbose. Nothing is printed for
            the parts that are already cached.

        tol_store and top_k only take effect when the evolution actually
        runs; use clear_evolution_cache() to repeat it with other values.

        Cached quantities:

            self.states        (only if store_states=True)
            self.final_state   (always: the state at s = 1)
            self.energies
            self.overlaps
            self.recorder
            self.eigenvalues

        Returns self.states (None if the states were not stored).
        """

        # =========================================================
        # 1. EVOLUTION
        # =========================================================

        if verbose is None:
            verbose = self.verbose

        rerun_for_states = store_states and self.states is None

        if self.energies is None or rerun_for_states:

            # Keep the recorder settings if we are only re-running to
            # store the states.
            if rerun_for_states and self.recorder is not None:
                tol_store = self.recorder.tol_store
                top_k = self.recorder.top_k

            target = self._h1_ground_state()

            recorder = EvolutionRecorder(tol_store=tol_store, top_k=top_k)

            state = self.initial_state.copy()
            states = [state.copy()] if store_states else None
            energies = []
            overlaps = []

            for i, si in enumerate(self.s):

                self._progress(verbose, "evolve_state", i)

                H = self.H(si)

                # -------------------------------------------------
                # Energy of evolved state
                # -------------------------------------------------

                Hpsi = H @ state

                energy = np.vdot(state, Hpsi).real

                energies.append(energy)

                # Overlap with the ground state of H1
                overlaps.append(np.abs(np.vdot(target, state)) ** 2)

                # Sparse record of |psi_i|^2
                recorder.add(state)

                # No evolution after final point
                if i == len(self.s) - 1:
                    break

                # Midpoint Hamiltonian for Trotter step. Step size and
                # midpoint are read from the grid, so subclasses can use a
                # different grid (e.g. a rescaled total time) by setting
                # self.t and self.s. For the uniform grid this is
                # dt = self.step and s_mid = si + self.step / (2 * self.T).
                dt = self.t[i + 1] - self.t[i]
                s_mid = (si + self.s[i + 1]) / 2

                H_mid = self.H(s_mid)

                new_state = expm_multiply(-1j * H_mid * dt, state)

                # -------------------------------------------------
                # Optional normalization
                #
                # expm_multiply should preserve the norm for a
                # Hermitian Hamiltonian, but this removes small
                # accumulated numerical errors.
                # -------------------------------------------------

                new_state /= np.linalg.norm(new_state)

                state = new_state

                if store_states:
                    states.append(state)

            self.states = states
            self.final_state = state
            self.energies = np.asarray(energies)
            self.overlaps = np.asarray(overlaps)
            self.recorder = recorder

        # =========================================================
        # 2. SPECTRUM
        # =========================================================

        if n_eigenvalues is None:
            return self.states

        matrix_dimension = self.H0.shape[0]

        # (eigensolvers.lowest_eigenpairs switches to dense diagonalization
        # when n_eigenvalues is close to the dimension, so any value up to it
        # is valid, as it was with the original dense eigvalsh)
        if n_eigenvalues > matrix_dimension:
            raise ValueError(
                "n_eigenvalues cannot be larger than the matrix dimension."
            )

        # =========================================================
        # CHANGE 1
        #
        # Use a larger Krylov subspace than the minimum required.
        #
        # This is particularly useful when several eigenvalues are
        # degenerate or very close to each other.
        # =========================================================

        ncv = min(matrix_dimension, max(4 * n_eigenvalues, 20))

        # =========================================================
        # If enough eigenvalues are already cached, do nothing.
        #
        # Example:
        #
        #   first call:  n=5
        #   second call: n=2
        #
        # =========================================================

        if self.eigenvalues is not None and self.n_eigenvalues >= n_eigenvalues:
            return self.states

        # =========================================================
        # CHANGE 2
        #
        # Number of eigenvalues we actually calculate.
        #
        # We calculate exactly the requested number. The larger
        # ncv gives ARPACK more room to resolve degeneracies.
        # =========================================================

        k = n_eigenvalues

        eigenvalues_list = [[] for _ in range(n_eigenvalues)]

        # =========================================================
        # CHANGE 3
        #
        # This is the state whose physical branch we want to track.
        #
        # At s=0 this is explicitly the user's initial_state.
        # This is important in the case of a degenerate ground state.
        # =========================================================

        target_state = self.initial_state.copy()

        # Make absolutely sure it is normalized
        target_state /= np.linalg.norm(target_state)

        # =========================================================
        # If we already have some eigenvalues cached, preserve them.
        #
        # Example:
        #
        #   first call: n=2
        #   second call: n=5
        #
        # The first two eigenvalues are kept.
        # =========================================================

        old_n = self.n_eigenvalues

        # Eigensolver, chosen once for the whole evolution: the pattern of
        # every H(s) lies inside that of |H0| + |H1|, so its block structure
        # (and, for a single block, a generic mid-evolution H(s)) represents
        # all of them.
        method, blocks = self.eig_method, None
        if method in ("auto", "blocks"):
            blocks = find_blocks(self.H0, self.H1)
            if method == "auto" and n_blocks(blocks) == 1:
                method, blocks = select_eig_method(
                    self.H(self.s[len(self.s) // 2]), lu_budget_mb=self.lu_budget_mb
                ), None
            else:
                method = "blocks"

        for i, si in enumerate(self.s):

            self._progress(verbose, "evolve_state, spectrum", i)

            H = self.H(si)

            # -----------------------------------------------------
            # We DO NOT use target_state as v0 at s=0.
            #
            # Instead, at every subsequent step we use the state
            # selected at the previous step. The larger ncv and the
            # tight tolerance help resolve close/degenerate
            # eigenvalues (shift-invert mode).
            # -----------------------------------------------------

            eigenvalues, eigenvectors = lowest_eigenpairs(
                H,
                k=k,
                method=method,
                v0=target_state if i > 0 else None,
                tol=1e-10,
                ncv=ncv,
                lu_budget_mb=self.lu_budget_mb,
                blocks=blocks,
            )

            # -----------------------------------------------------
            # Sort eigenvalues and eigenvectors
            # -----------------------------------------------------

            order = np.argsort(eigenvalues)

            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            # =====================================================
            # CHANGE 7
            #
            # Track the physical state by overlap rather than
            # assuming eigenvectors[:, 0] is the state we want.
            #
            # This is the key change for degenerate ground states.
            # =====================================================

            overlaps = np.abs(eigenvectors.conj().T @ target_state) ** 2

            best_index = np.argmax(overlaps)

            target_state = eigenvectors[:, best_index].copy()

            # Normalize
            target_state /= np.linalg.norm(target_state)

            # -----------------------------------------------------
            # Store eigenvalues
            # -----------------------------------------------------

            if old_n == 0:

                # No previous spectrum
                for j in range(n_eigenvalues):
                    eigenvalues_list[j].append(eigenvalues[j])

            else:

                # Keep cached eigenvalues where available
                for j in range(old_n):
                    eigenvalues_list[j].append(self.eigenvalues[j, i])

                # Add newly requested eigenvalues
                for j in range(old_n, n_eigenvalues):
                    eigenvalues_list[j].append(eigenvalues[j])

        self.eigenvalues = np.asarray(eigenvalues_list)

        self.n_eigenvalues = n_eigenvalues

        return self.states

    def plot_overlap(self, verbose=None):
        """Plot overlap with the ground state of H1.

        verbose : bool or None, passed to evolve_state (None uses self.verbose).
        """

        # Overlaps are computed during the evolution (no need for the states)
        self.evolve_state(verbose=verbose)

        plt.figure(figsize=(7, 5))

        plt.plot(self.s, self.overlaps, ".-", linewidth=1, markersize=4.5)

        plt.title(r"Overlap with Ground State of $H_1$")

        plt.xlabel(r"$s$")

        plt.ylabel(r"$|\langle \psi_{\mathrm{evo}}(s)" r"|E_0^{H_1}\rangle|^2$")

        plt.ylim(-0.05, 1.05)
        plt.grid()
        plt.show()

    def plot_probabilities(
        self,
        N,
        L,
        kind="animation",
        tol=1e-3,
        max_states=None,
        digit_base=None,
        width=None,
        tags=None,
        times=None,
        tol_store=1e-8,
        top_k=None,
        verbose=None,
        **plot_kwargs,
    ):
        """
        Plot the evolution of the basis-state probabilities, grouping
        physical and virtual states (see plotting_utils.py).

        Parameters
        ----------
        N, L : int
            N bits + N digits in [0, L] per bitstring (see plotting_utils).
        kind : "animation" or "heatmap"
        tol : float
            Only states with probability > tol at some time are shown.
        max_states : int, tuple or None
            Cap on the number of states shown (see plotting_utils).
        digit_base : int or None
            Number of values of each digit of the second part
            (default L + 1).
        times : array or None
            Values shown as time (default self.t).
        tol_store, top_k :
            Recording parameters, only used if the evolution has not been
            run yet (see evolve_state).
        verbose : bool or None
            Passed to evolve_state (None uses self.verbose).
        **plot_kwargs :
            Passed to plot_state_evolution / plot_state_heatmap
            (title, save_path, duration, figsize, ...).

        Returns whatever the plotting function returns
        ((ani, fig) for the animation, (fig, ax) for the heatmap).
        """
        self.evolve_state(tol_store=tol_store, top_k=top_k, verbose=verbose)

        prob_data = self.recorder.to_prob_data(
            N, L, digit_base=digit_base, width=width, tol=tol
        )

        if times is None:
            times = self.t

        common = dict(
            N=N,
            L=L,
            width=width,
            tags=tags,
            tol=tol,
            max_states=max_states,
            times=times,
        )

        if kind == "animation":
            return plot_state_evolution(prob_data, **common, **plot_kwargs)
        if kind == "heatmap":
            return plot_state_heatmap(prob_data, **common, **plot_kwargs)
        raise ValueError("kind must be 'animation' or 'heatmap'.")

    def plot_spectral_values(self, n=2, legend=True, min_gap=False, verbose=None):
        """
        Plot the lowest n eigenvalues and the energy of the
        evolved state.

        verbose : bool or None, passed to evolve_state (None uses self.verbose).
        """

        # Calculates only what is not already cached
        self.evolve_state(n_eigenvalues=n, verbose=verbose)

        eigenvalues_list = self.eigenvalues[:n]

        if min_gap:

            gaps = eigenvalues_list[1] - eigenvalues_list[0]

            min_gap_index = np.argmin(gaps)

            min_gap_value = gaps[min_gap_index]
            min_gap_s = self.s[min_gap_index]

            print(f"Minimum energy gap: {min_gap_value}")

            print(f"Location of minimum energy gap: " f"{min_gap_s}")

        plt.figure(figsize=(7, 5))

        for i in range(n):

            plt.plot(self.s, eigenvalues_list[i], label=rf"$E_{{{i}}}$", linewidth=1.5)

        # Energy of evolved state is already cached
        plt.plot(
            self.s, self.energies, ".", markersize=5, label="Energy of Evolved State"
        )

        plt.title("Spectral Values During Adiabatic Evolution")

        plt.xlabel(r"$s$")
        plt.ylabel(r"$E$")

        if legend:
            plt.legend()

        plt.grid()
        plt.show()

    def plot_energy_variance(self, legend=True, verbose=None):
        """Plot energy variance during the evolution.

        Needs the full state vectors, so it runs the evolution with
        store_states=True if they are not stored yet (memory heavy for
        large systems).

        verbose : bool or None, passed to evolve_state (None uses self.verbose).
        """

        states = self.evolve_state(store_states=True, verbose=verbose)

        energy_variance_list = []

        for si, state in zip(self.s, states):

            H = self.H(si)

            Hpsi = H @ state

            # <H>
            mean_energy = np.vdot(state, Hpsi).real

            # <H^2>
            #
            # <psi|H^2|psi> = ||H|psi>||^2
            #
            mean_energy_squared = np.vdot(Hpsi, Hpsi).real

            variance = mean_energy_squared - mean_energy**2

            # Protect against tiny negative floating-point
            # errors.
            energy_variance_list.append(max(variance, 0.0))

        plt.figure(figsize=(7, 5))

        plt.plot(
            self.s,
            energy_variance_list,
            ".-",
            linewidth=1.5,
            markersize=4.5,
            label="Energy Variance",
        )

        plt.title("Energy Variance During Adiabatic Evolution")

        plt.xlabel(r"$s$")
        plt.ylabel(r"$\Delta E^2$")

        if legend:
            plt.legend()

        plt.grid()
        plt.show()
