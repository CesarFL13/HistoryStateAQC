"""
History-state Hamiltonians with a unary clock (Aharonov et al.) and
variants.

A circuit is a list of gates (name, target[, control[, theta]]) on
n_qubits; the clock has L = len(gateslist) qubits, clock step t being
|1^t 0^(L-t)>. HistoryState builds the target history state
sum_t |alpha_t>|t> / sqrt(L + 1), |alpha_t> the state after t gates
(amplitudes proportional to eta^t if eta != 1).

Hamiltonians and their adiabatic evolutions:

hs: HistoryStateHamiltonian, HistoryStateEvolution
    H0 = H_input + H_clockinit + H_clock,
    H1 = H_input + H_clock + H_gates.
hs3l: HistoryStateHamiltonian3Local, HistoryStateEvolution3Local
    3-local gate terms, clock terms weighted by J.
hsee: HistoryStateHamiltonianExactEquiv, HistoryStateEvolutionExactEquiv
    Three stages: H0 -> HI(0) during T1, HI(t) during T, HI(T) -> H1
    during T2. HI(t) has a Gaussian potential well centred at clock
    step t / tau, which drags the state along the clock; it is
    written in the basis of the states |alpha_t>|t>.
hseesubspace: HistoryStateHamiltonianExactEquivSubspace,
              HistoryStateEvolutionExactEquivSubspace
    The same in the (L + 1)-dimensional history subspace;
    vector_basis_change maps its states back to the full space.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, issparse, kron
from scipy.sparse.linalg import expm_multiply

import quantum as q
from adiabatic_evolution import AdiabaticEvolution, EvolutionProgress
from eigensolvers import find_blocks, lowest_eigenpairs


def vector_basis_change(state, gateslist, n_qubits):
    """Performs the change of basis from the gamma basis to the computational basis."""
    if issparse(state):
        state = state.toarray()
    state = np.asarray(state).reshape(-1)
    hist_state = HistoryState(gateslist, n_qubits)
    new_state = np.zeros(hist_state.dim, dtype=complex)
    alpha_list = hist_state.alpha_list()
    for t in range(hist_state.L + 1):
        new_state += state[t] * np.kron(
            alpha_list[t],
            q.basis_state("1" * t + "0" * (hist_state.L - t), hist_state.L),
        )
    return new_state


class HistoryState:
    def __init__(self, gateslist, n_qubits, eta=1.0):
        self.gateslist = gateslist
        self.n_qubits = n_qubits
        self.L = len(gateslist)
        self.qubitsdim = 2**n_qubits  # Dimension of the qubits Hilbert space
        self.clockdim = 2**self.L  # Dimension of the clock Hilbert space
        self.dim = 2 ** (n_qubits + self.L)  # Total dimension
        self.eta = eta

    def alpha_list(self):
        alpha_0 = q.basis_state(
            "0" * self.n_qubits, self.n_qubits
        )  # Initial state is |00...0⟩
        alpha_list = [alpha_0]
        for t in range(self.L):
            name, target = self.gateslist[t][0], self.gateslist[t][1]
            if len(self.gateslist[t]) >= 3:
                control = self.gateslist[t][2]
            else:
                control = None

            if len(self.gateslist[t]) >= 4:
                theta = self.gateslist[t][3]
            else:
                theta = None
            gate_matrix = q.build_gate(
                name, self.n_qubits, target, control=control, theta=theta
            )
            alpha_l = gate_matrix @ alpha_list[-1]
            alpha_list.append(alpha_l)
        return alpha_list

    def history_state(self):
        """Constructs the history state corresponding to the given gateslist."""
        state = np.zeros(self.dim, dtype=complex)
        alpha_list = self.alpha_list()
        if self.eta == 1.0:
            for t in range(self.L + 1):
                state += np.kron(
                    alpha_list[t], q.basis_state("1" * t + "0" * (self.L - t), self.L)
                )
            state = (1 / np.sqrt(self.L + 1)) * state
        else:
            for t in range(self.L + 1):
                state += (self.eta**t) * np.kron(
                    alpha_list[t], q.basis_state("1" * t + "0" * (self.L - t), self.L)
                )
            state = (
                np.sqrt((self.eta**2 - 1) / (self.eta ** (2 * self.L + 2) - 1))
            ) * state
        return state


class HistoryStateHamiltonian:
    def __init__(self, gateslist, n_qubits, eta=1.0):
        self.gateslist = gateslist
        self.n_qubits = n_qubits
        self.L = len(gateslist)
        self.qubitsdim = 2**n_qubits  # Dimension of the qubits Hilbert space
        self.clockdim = 2**self.L  # Dimension of the clock Hilbert space
        self.dim = 2 ** (n_qubits + self.L)  # Total dimension
        self.eta = eta

    def H_clock(self):
        """Constructs the clock Hamiltonian."""
        H_clock = csr_matrix((self.clockdim, self.clockdim), dtype=complex)
        for t in range(self.L - 1):
            projector = q.basis_projector("01", "01", 2)
            H_clock += kron(
                q.identity(2**t),
                kron(projector, q.identity(2 ** (self.L - t - 2))),
            )
        H_clock = kron(
            q.identity(self.qubitsdim), H_clock
        )  # Extend to the full Hilbert space
        return H_clock

    def H_input(self):
        """Constructs the input Hamiltonian."""
        H_input = csr_matrix((self.dim, self.dim), dtype=complex)
        for i in range(self.n_qubits):
            projector = kron(
                kron(q.identity(2**i), q.basis_projector("1", "1", 1)),
                q.identity(2 ** (self.n_qubits - i - 1)),
            )
            H_input += kron(
                kron(projector, q.basis_projector("0", "0", 1)),
                q.identity(2 ** (self.L - 1)),
            )  # Extend to the full Hilbert space
        return H_input

    def H_clockinit(self):
        """Constructs the clock initialization Hamiltonian."""
        projector = q.basis_projector("1", "1", 1)
        H_clockinit = kron(
            kron(q.identity(self.qubitsdim), projector),
            q.identity(2 ** (self.L - 1)),
        )  # Extend to the full Hilbert space
        return H_clockinit

    def H_gates(self):
        """Constructs the gate Hamiltonian."""
        H_gates = csr_matrix((self.dim, self.dim), dtype=complex)

        for i in range(self.L):
            name, target = self.gateslist[i][0], self.gateslist[i][1]
            if len(self.gateslist[i]) >= 3:
                control = self.gateslist[i][2]
            else:
                control = None

            if len(self.gateslist[i]) >= 4:
                theta = self.gateslist[i][3]
            else:
                theta = None
            gate_matrix = q.build_gate(
                name, self.n_qubits, target, control=control, theta=theta
            )
            if i != 0 and i != self.L - 1:
                H_gates += (
                    self.eta
                    * kron(
                        q.identity(self.qubitsdim),
                        kron(
                            q.identity(2 ** (i - 1)),
                            kron(
                                q.basis_projector("100", "100", 3),
                                q.identity(2 ** (self.L - 2 - i)),
                            ),
                        ),
                    )
                    - kron(
                        gate_matrix,
                        kron(
                            q.identity(2 ** (i - 1)),
                            kron(
                                q.basis_projector("100", "110", 3),
                                q.identity(2 ** (self.L - 2 - i)),
                            ),
                        ),
                    )
                    - kron(
                        gate_matrix.transpose().conj(),
                        kron(
                            q.identity(2 ** (i - 1)),
                            kron(
                                q.basis_projector("110", "100", 3),
                                q.identity(2 ** (self.L - 2 - i)),
                            ),
                        ),
                    )
                    + (1 / self.eta)
                    * kron(
                        q.identity(self.qubitsdim),
                        kron(
                            q.identity(2 ** (i - 1)),
                            kron(
                                q.basis_projector("110", "110", 3),
                                q.identity(2 ** (self.L - 2 - i)),
                            ),
                        ),
                    )
                )
            elif i == 0:
                H_gates += (
                    self.eta
                    * kron(
                        q.identity(self.qubitsdim),
                        kron(
                            q.basis_projector("00", "00", 2),
                            q.identity(2 ** (self.L - 2)),
                        ),
                    )
                    - kron(
                        gate_matrix,
                        kron(
                            q.basis_projector("00", "10", 2),
                            q.identity(2 ** (self.L - 2)),
                        ),
                    )
                    - kron(
                        gate_matrix.transpose().conj(),
                        kron(
                            q.basis_projector("10", "00", 2),
                            q.identity(2 ** (self.L - 2)),
                        ),
                    )
                    + (1 / self.eta)
                    * kron(
                        q.identity(self.qubitsdim),
                        kron(
                            q.basis_projector("10", "10", 2),
                            q.identity(2 ** (self.L - 2)),
                        ),
                    )
                )

            else:
                H_gates += (
                    self.eta
                    * kron(
                        q.identity(2 ** (self.n_qubits + self.L - 2)),
                        q.basis_projector("10", "10", 2),
                    )
                    - kron(
                        gate_matrix,
                        kron(
                            q.identity(2 ** (self.L - 2)),
                            q.basis_projector("10", "11", 2),
                        ),
                    )
                    - kron(
                        gate_matrix.transpose().conj(),
                        kron(
                            q.identity(2 ** (self.L - 2)),
                            q.basis_projector("11", "10", 2),
                        ),
                    )
                    + (1 / self.eta)
                    * kron(
                        q.identity(2 ** (self.n_qubits + self.L - 2)),
                        q.basis_projector("11", "11", 2),
                    )
                )

        return (1 / 2) * H_gates

    def H0(self):
        """Constructs the initial Hamiltonian H0."""
        return self.H_input() + self.H_clockinit() + self.H_clock()

    def H1(self):
        """Constructs the final Hamiltonian H1."""
        return self.H_input() + self.H_clock() + self.H_gates()


class HistoryStateHamiltonian3Local(HistoryStateHamiltonian):
    def __init__(self, gateslist, n_qubits, J=1.0):
        HistoryStateHamiltonian.__init__(self, gateslist, n_qubits)
        self.J = J

    def H_gates(self):
        """Constructs the gate Hamiltonian."""
        H_gates = csr_matrix((self.dim, self.dim), dtype=complex)

        for i in range(self.L):
            name, target = self.gateslist[i][0], self.gateslist[i][1]
            if len(self.gateslist[i]) >= 3:
                control = self.gateslist[i][2]
            else:
                control = None

            if len(self.gateslist[i]) >= 4:
                theta = self.gateslist[i][3]
            else:
                theta = None
            gate_matrix = q.build_gate(
                name, self.n_qubits, target, control=control, theta=theta
            )

            if i != 0 and i != self.L - 1:
                H_gates += (
                    kron(
                        q.identity(self.qubitsdim),
                        kron(
                            q.identity(2 ** (i - 1)),
                            kron(
                                q.basis_projector("100", "100", 3),
                                q.identity(2 ** (self.L - 2 - i)),
                            ),
                        ),
                    )
                    - kron(
                        gate_matrix,
                        kron(
                            q.identity(2 ** (i)),
                            kron(
                                q.basis_projector("0", "1", 1),
                                q.identity(2 ** (self.L - 1 - i)),
                            ),
                        ),
                    )
                    - kron(
                        gate_matrix.transpose().conj(),
                        kron(
                            q.identity(2 ** (i)),
                            kron(
                                q.basis_projector("1", "0", 1),
                                q.identity(2 ** (self.L - 1 - i)),
                            ),
                        ),
                    )
                    + kron(
                        q.identity(self.qubitsdim),
                        kron(
                            q.identity(2 ** (i - 1)),
                            kron(
                                q.basis_projector("110", "110", 3),
                                q.identity(2 ** (self.L - 2 - i)),
                            ),
                        ),
                    )
                )
            elif i == 0:
                H_gates += (
                    kron(
                        q.identity(self.qubitsdim),
                        kron(
                            q.basis_projector("00", "00", 2),
                            q.identity(2 ** (self.L - 2)),
                        ),
                    )
                    - kron(
                        gate_matrix,
                        kron(
                            q.basis_projector("0", "1", 1),
                            q.identity(2 ** (self.L - 1)),
                        ),
                    )
                    - kron(
                        gate_matrix.transpose().conj(),
                        kron(
                            q.basis_projector("1", "0", 1),
                            q.identity(2 ** (self.L - 1)),
                        ),
                    )
                    + kron(
                        q.identity(self.qubitsdim),
                        kron(
                            q.basis_projector("10", "10", 2),
                            q.identity(2 ** (self.L - 2)),
                        ),
                    )
                )
            else:
                H_gates += (
                    kron(
                        q.identity(2 ** (self.n_qubits + self.L - 2)),
                        q.basis_projector("10", "10", 2),
                    )
                    - kron(
                        gate_matrix,
                        kron(
                            q.identity(2 ** (self.L - 1)),
                            q.basis_projector("0", "1", 1),
                        ),
                    )
                    - kron(
                        gate_matrix.transpose().conj(),
                        kron(
                            q.identity(2 ** (self.L - 1)),
                            q.basis_projector("1", "0", 1),
                        ),
                    )
                    + kron(
                        q.identity(2 ** (self.n_qubits + self.L - 2)),
                        q.basis_projector("11", "11", 2),
                    )
                )
        return (1 / 2) * H_gates

    def H0(self):
        """Constructs the initial Hamiltonian H0."""
        return self.H_input() + self.H_clockinit() + self.J * self.H_clock()

    def H1(self):
        """Constructs the final Hamiltonian H1."""
        return self.H_input() + self.J * self.H_clock() + self.H_gates()


class HistoryStateHamiltonianExactEquiv(HistoryStateHamiltonian):
    def __init__(self, gateslist, n_qubits, eta=1.0, tau=1.0):
        HistoryStateHamiltonian.__init__(self, gateslist, n_qubits, eta=eta)
        self.tau = tau

    def M(self):
        """Constructs the change of basis matrix M from the gamma basis to the computational basis."""
        M = np.zeros((self.dim, self.L + 1), dtype=complex)
        alpha_list = HistoryState(
            self.gateslist, self.n_qubits, eta=self.eta
        ).alpha_list()
        for t in range(self.L + 1):
            M[:, t] = np.kron(alpha_list[t], q.basis_state("1" * t + "0" * (self.L - t), self.L))
        return csr_matrix(M)

    def HI(self, t):
        """Constructs the intermediate Hamiltonian HI."""
        HI = np.zeros((self.L + 1, self.L + 1), dtype=complex)
        for m in range(self.L + 1):
            HI[m, m] = (1 / (2 * self.eta)) + (self.eta / 2) * (
                1 - np.exp(-((t / self.tau - m) ** 2))
            )
        for m in range(self.L):
            HI[m, m + 1] = -0.5
            HI[m + 1, m] = -0.5

        M = self.M()
        return M @ csr_matrix(HI) @ M.transpose().conj()


class HistoryStateHamiltonianExactEquivSubspace:
    def __init__(self, gateslist, n_qubits, eta=1.0):
        self.gateslist = gateslist
        self.n_qubits = n_qubits
        self.L = len(gateslist)
        self.qubitsdim = 2**n_qubits  # Dimension of the qubits Hilbert space
        self.clockdim = 2**self.L  # Dimension of the clock Hilbert space
        self.dim = 2 ** (n_qubits + self.L)  # Total dimension
        self.eta = eta

    def H0(self):
        """Constructs the initial Hamiltonian H0."""
        H0 = np.zeros((self.L + 1, self.L + 1), dtype=complex)
        for t in range(1, self.L + 1):
            H0[t, t] = 1.0
        return csr_matrix(H0)

    def H1(self):
        """Constructs the final Hamiltonian H1."""
        H1 = np.zeros((self.L + 1, self.L + 1), dtype=complex)
        for t in range(self.L + 1):
            if t != 0 and t != self.L:
                H1[t, t] = (self.eta + (1 / self.eta)) / 2
            elif t == 0:
                H1[t, t] = self.eta / 2
            else:
                H1[t, t] = 1 / (2 * self.eta)
        for t in range(self.L):
            H1[t, t + 1] = -0.5
            H1[t + 1, t] = -0.5
        return csr_matrix(H1)

    def HI(self, t):
        """Constructs the intermediate Hamiltonian HI."""
        HI = np.zeros((self.L + 1, self.L + 1), dtype=complex)
        for m in range(self.L + 1):
            HI[m, m] = (1 / (2 * self.eta)) + (self.eta / 2) * (
                1 - np.exp(-((t / self.tau - m) ** 2))
            )
        for m in range(self.L):
            HI[m, m + 1] = -0.5
            HI[m + 1, m] = -0.5
        return csr_matrix(HI)


class HistoryStateEvolution(AdiabaticEvolution, HistoryStateHamiltonian):
    def __init__(self, gateslist, n_qubits, T, step, verbose=False, verbose_every=5):
        HistoryStateHamiltonian.__init__(self, gateslist, n_qubits)
        initial_state = np.zeros(self.dim, dtype=complex)
        initial_state[0] = 1.0  # Initial state is |00...0⟩|0⟩
        AdiabaticEvolution.__init__(
            self,
            self.H0(),
            self.H1(),
            initial_state,
            T,
            step,
            verbose=verbose,
            verbose_every=verbose_every,
        )

    def output_state(self):
        """Returns the obtained output state of the circuit after the adiabatic evolution."""
        self.evolve_state()
        # Project onto the last clock state (clock = least significant part)
        output = self.final_state.reshape(self.qubitsdim, self.clockdim)[:, -1]
        output = (1 / np.linalg.norm(output)) * output
        return output


class HistoryStateEvolution3Local(AdiabaticEvolution, HistoryStateHamiltonian3Local):
    def __init__(
        self, gateslist, n_qubits, T, step, J=1.0, verbose=False, verbose_every=5
    ):
        self.J = J
        HistoryStateHamiltonian3Local.__init__(self, gateslist, n_qubits, J=self.J)
        initial_state = np.zeros(self.dim, dtype=complex)
        initial_state[0] = 1.0  # Initial state is |00...0⟩|0⟩
        AdiabaticEvolution.__init__(
            self,
            self.H0(),
            self.H1(),
            initial_state,
            T,
            step,
            verbose=verbose,
            verbose_every=verbose_every,
        )

    def output_state(self):
        """Returns the obtained output state of the circuit after the adiabatic evolution."""
        self.evolve_state()
        # Project onto the last clock state (clock = least significant part)
        output = self.final_state.reshape(self.qubitsdim, self.clockdim)[:, -1]
        output = (1 / np.linalg.norm(output)) * output
        return output


class HistoryStateEvolutionExactEquiv(HistoryStateHamiltonianExactEquiv):
    def __init__(
        self,
        gateslist,
        n_qubits,
        T1,
        T2,
        T,
        step,
        eta=1.0,
        tau=1.0,
        verbose=False,
        verbose_every=5,
    ):
        self.T1 = T1
        self.T2 = T2
        self.T = T
        self.step = step
        # Progress messages of evolution() (see AdiabaticEvolution)
        self.verbose = verbose
        self.verbose_every = verbose_every
        HistoryStateHamiltonianExactEquiv.__init__(
            self, gateslist, n_qubits, eta=eta, tau=tau
        )

    def evolution(self, verbose=None):
        """Performs the adiabatic evolution from H0 to H1, using HI as intermediate Hamiltonian."""
        initial_state = np.zeros(self.dim, dtype=complex)
        initial_state[0] = 1.0  # Initial state is |00...0⟩|0⟩
        if verbose is None:
            verbose = self.verbose
        n1, n2 = round(self.T1 / self.step), round(self.T / self.step)
        progress = EvolutionProgress(
            verbose,
            self.verbose_every,
            self.T1 + self.T + self.T2,
            n1 + n2 + round(self.T2 / self.step),
            dim=initial_state.size,
        )
        evo1 = AdiabaticEvolution(
            self.H0(), self.HI(0), initial_state, self.T1, self.step
        )
        evo1._progress_hook = lambda i: progress(i, evo1.t[i])
        states = list(evo1.evolve_state(store_states=True, verbose=verbose))
        H = evo1.evolve_H()
        times = list(self.step * np.arange(round(self.T1 / self.step) + 1))
        for t in list(self.step * np.arange(round(self.T / self.step) + 1) + self.T1)[
            1:
        ]:
            # * Temporal evolution is done using Trotterization.
            H.append(self.HI(t - self.T1))
            new_state = expm_multiply(
                -1j * self.HI(t - self.step / 2 - self.T1) * self.step, states[-1]
            )
            new_state /= np.linalg.norm(new_state)

            states.append(new_state)
            times.append(t)
            progress(len(times) - 1, t)

        evo2 = AdiabaticEvolution(
            self.HI(self.T), self.H1(), states[-1], self.T2, self.step
        )
        evo2._progress_hook = lambda i: progress(
            n1 + n2 + i, self.T1 + self.T + evo2.t[i]
        )
        states += evo2.evolve_state(store_states=True, verbose=verbose)[1:]
        H += evo2.evolve_H()[1:]
        times += list(
            self.step * np.arange(round(self.T2 / self.step) + 1) + self.T1 + self.T
        )[1:]
        return states, times, H

    def plot_overlap(self, verbose=None):
        states, t = self.evolution(verbose)[:2]
        s = [(1 / (self.T1 + self.T + self.T2)) * time for time in t]
        objective_state = lowest_eigenpairs(self.H1(), k=1)[1][:, 0]
        plt.figure(figsize=(7, 5))
        overlaps = []
        for i in range(len(states)):
            overlaps.append(np.abs(np.vdot(objective_state, states[i])) ** 2)
        plt.plot(s, overlaps, ".-", linewidth=1, markersize=4.5)
        plt.title(r"Overlap with Ground State of $H_1$ during Evolution")
        plt.xlabel(r"$s$")
        plt.ylabel(r"$|\langle \psi_{evo}(s)|E_0^{H_1}\rangle|^2$")
        plt.ylim(-0.05, 1.05)
        plt.vlines(
            self.T1 / (self.T1 + self.T + self.T2),
            color="black",
            linestyle="--",
            ymin=-0.05,
            ymax=1.05,
            linewidth=1,
        )
        plt.vlines(
            (self.T1 + self.T) / (self.T1 + self.T + self.T2),
            color="black",
            linestyle="--",
            ymin=-0.05,
            ymax=1.05,
            linewidth=1,
        )
        plt.grid()
        plt.show()

    def plot_spectral_values(self, n=2, legend=True, verbose=None):
        states, t, H = self.evolution(verbose)
        s = [(1 / (self.T1 + self.T + self.T2)) * time for time in t]
        # Lowest n levels of every H, block by block: the blocks of the union
        # of all their sparsity patterns are valid for each of them
        blocks = find_blocks(*H)
        eigenvalues_list = [[] for _ in range(n)]
        for i in range(len(H)):
            eigenvalues = lowest_eigenpairs(H[i], k=n, blocks=blocks)[0]
            for j in range(n):
                eigenvalues_list[j].append(eigenvalues[j])
        evo_energy = []
        for i in range(len(H)):
            evo_energy.append(
                np.vdot(states[i], H[i] @ states[i]).real
            )
        plt.figure(figsize=(7, 5))
        for i in range(n):
            plt.plot(s, eigenvalues_list[i], label=r"$E_{%d}$" % i, linewidth=1.5)
        plt.plot(
            s,
            evo_energy,
            ".",
            markersize=5,
            label="Energy of Evolved State",
        )
        plt.title("Spectral Values During Adiabatic Evolution")
        plt.xlabel(r"$s$")
        plt.ylabel(r"$E$")
        plt.vlines(
            self.T1 / (self.T1 + self.T + self.T2),
            color="black",
            linestyle="--",
            ymin=int(min(eigenvalues_list[:][0])) - 0.5,
            ymax=int(max(eigenvalues_list[:][n - 1])) + 0.5,
            linewidth=1,
        )
        plt.vlines(
            (self.T1 + self.T) / (self.T1 + self.T + self.T2),
            color="black",
            linestyle="--",
            ymin=int(min(eigenvalues_list[:][0])) - 0.5,
            ymax=int(max(eigenvalues_list[:][n - 1])) + 0.5,
            linewidth=1,
        )
        if legend:
            plt.legend()
        plt.grid()
        plt.show()

    def output_state(self, verbose=None):
        final_state = self.evolution(verbose)[0][-1]
        # Project onto the last clock state (clock = least significant part)
        output = final_state.reshape(self.qubitsdim, self.clockdim)[:, -1]
        output = (1 / np.linalg.norm(output)) * output
        return output


class HistoryStateEvolutionExactEquivSubspace(
    HistoryStateHamiltonianExactEquivSubspace
):
    def __init__(
        self,
        gateslist,
        n_qubits,
        T1,
        T2,
        T,
        step,
        eta=1.0,
        tau=1.0,
        verbose=False,
        verbose_every=5,
    ):
        self.tau = tau
        self.T1 = T1
        self.T2 = T2
        self.T = T
        self.step = step
        # Progress messages of evolution() (see AdiabaticEvolution)
        self.verbose = verbose
        self.verbose_every = verbose_every
        HistoryStateHamiltonianExactEquivSubspace.__init__(
            self, gateslist, n_qubits, eta=eta
        )

    def evolution(self, verbose=None):
        """Performs the adiabatic evolution from H0 to H1, using HI as intermediate Hamiltonian."""
        initial_state = np.zeros(self.L + 1, dtype=complex)
        initial_state[0] = 1.0  # Initial state is |00...0⟩|0⟩
        if verbose is None:
            verbose = self.verbose
        n1, n2 = round(self.T1 / self.step), round(self.T / self.step)
        progress = EvolutionProgress(
            verbose,
            self.verbose_every,
            self.T1 + self.T + self.T2,
            n1 + n2 + round(self.T2 / self.step),
            dim=initial_state.size,
        )
        evo1 = AdiabaticEvolution(
            self.H0(), self.HI(0), initial_state, self.T1, self.step
        )
        evo1._progress_hook = lambda i: progress(i, evo1.t[i])
        states = list(evo1.evolve_state(store_states=True, verbose=verbose))
        H = evo1.evolve_H()
        times = list(self.step * np.arange(round(self.T1 / self.step) + 1))
        for t in list(self.step * np.arange(round(self.T / self.step) + 1) + self.T1)[
            1:
        ]:
            # * Temporal evolution is done using Trotterization.
            H.append(self.HI(t - self.T1))
            new_state = expm_multiply(
                -1j * self.HI(t - self.step / 2 - self.T1) * self.step, states[-1]
            )
            new_state /= np.linalg.norm(new_state)
            states.append(new_state)
            times.append(t)
            progress(len(times) - 1, t)

        evo2 = AdiabaticEvolution(
            self.HI(self.T), self.H1(), states[-1], self.T2, self.step
        )
        evo2._progress_hook = lambda i: progress(
            n1 + n2 + i, self.T1 + self.T + evo2.t[i]
        )
        states += evo2.evolve_state(store_states=True, verbose=verbose)[1:]
        H += evo2.evolve_H()[1:]
        times += list(
            self.step * np.arange(round(self.T2 / self.step) + 1) + self.T1 + self.T
        )[1:]
        return states, times, H

    def plot_overlap(self, verbose=None):
        states, t = self.evolution(verbose)[:2]
        s = [(1 / (self.T1 + self.T + self.T2)) * time for time in t]
        objective_state = lowest_eigenpairs(self.H1(), k=1)[1][:, 0]
        plt.figure(figsize=(7, 5))
        overlaps = []
        for i in range(len(states)):
            overlaps.append(np.abs(np.vdot(objective_state, states[i])) ** 2)
        plt.plot(s, overlaps, ".-", linewidth=1, markersize=4.5)
        plt.title(r"Overlap with Ground State of $H_1$ during Evolution")
        plt.xlabel(r"$s$")
        plt.ylabel(r"$|\langle \psi_{evo}(s)|E_0^{H_1}\rangle|^2$")
        plt.ylim(-0.05, 1.05)
        plt.vlines(
            self.T1 / (self.T1 + self.T + self.T2),
            color="black",
            linestyle="--",
            ymin=-0.05,
            ymax=1.05,
            linewidth=1,
        )
        plt.vlines(
            (self.T1 + self.T) / (self.T1 + self.T + self.T2),
            color="black",
            linestyle="--",
            ymin=-0.05,
            ymax=1.05,
            linewidth=1,
        )
        plt.grid()
        plt.show()

    def plot_spectral_values(self, n=2, legend=True, verbose=None):
        states, t, H = self.evolution(verbose)
        s = [(1 / (self.T1 + self.T + self.T2)) * time for time in t]
        # Lowest n levels of every H, block by block: the blocks of the union
        # of all their sparsity patterns are valid for each of them
        blocks = find_blocks(*H)
        eigenvalues_list = [[] for _ in range(n)]
        for i in range(len(H)):
            eigenvalues = lowest_eigenpairs(H[i], k=n, blocks=blocks)[0]
            for j in range(n):
                eigenvalues_list[j].append(eigenvalues[j])
        evo_energy = []
        for i in range(len(H)):
            evo_energy.append(
                np.vdot(states[i], H[i] @ states[i]).real
            )
        plt.figure(figsize=(7, 5))
        for i in range(n):
            plt.plot(s, eigenvalues_list[i], label=r"$E_{%d}$" % i, linewidth=1.5)
        plt.plot(
            s,
            evo_energy,
            ".",
            markersize=5,
            label="Energy of Evolved State",
        )
        plt.title("Spectral Values During Adiabatic Evolution")
        plt.xlabel(r"$s$")
        plt.ylabel(r"$E$")
        plt.vlines(
            self.T1 / (self.T1 + self.T + self.T2),
            color="black",
            linestyle="--",
            ymin=int(min(eigenvalues_list[:][0])) - 0.5,
            ymax=int(max(eigenvalues_list[:][n - 1])) + 0.5,
            linewidth=1,
        )
        plt.vlines(
            (self.T1 + self.T) / (self.T1 + self.T + self.T2),
            color="black",
            linestyle="--",
            ymin=int(min(eigenvalues_list[:][0])) - 0.5,
            ymax=int(max(eigenvalues_list[:][n - 1])) + 0.5,
            linewidth=1,
        )
        if legend:
            plt.legend()
        plt.grid()
        plt.show()

    def output_state(self, verbose=None):
        final_state = self.evolution(verbose)[0][-1]
        final_state_comp_basis = vector_basis_change(
            final_state, self.gateslist, self.n_qubits
        )
        # Project onto the last clock state (clock = least significant part)
        output = final_state_comp_basis.reshape(self.qubitsdim, self.clockdim)[:, -1]
        output = (1 / np.linalg.norm(output)) * output
        return output
