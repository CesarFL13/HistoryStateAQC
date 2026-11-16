"""
Basic tools for n-qubit registers.

Conversion between binary strings and integers, computational basis
states and projectors (basis_state, basis_projector), identity, gates
on n qubits as sparse matrices (build_gate) and the norm of the
commutator of two Hermitian matrices (commutator_norm_hermitian).
"""

import numpy as np

from scipy.sparse import csr_matrix, kron
from scipy.sparse import identity as sparse_identity
from scipy.sparse.linalg import LinearOperator, eigsh


def binary_to_decimal(binary_string):
    """Convert a binary string to its decimal representation."""
    return int(binary_string, 2)


def decimal_to_binary(decimal_number, nqubits):
    """Convert a decimal number to a binary string with leading zeros."""
    return format(decimal_number, f"0{nqubits}b")


def identity(n):
    """Return an n x n sparse identity matrix (built sparse, never dense)."""
    return sparse_identity(n, dtype=complex, format="csr")


def basis_state(state, nqubits, dtype="bin"):
    """
    Return a computational basis state as a dense 1D NumPy array.

    The returned vector has shape (2**nqubits,), not (2**nqubits, 1).
    """
    dim = 2**nqubits

    if dtype == "bin":
        state_index = binary_to_decimal(state)
    elif dtype == "dec":
        state_index = state
    else:
        raise ValueError("dtype must be 'bin' or 'dec'")

    if not 0 <= state_index < dim:
        raise ValueError(f"State index {state_index} outside range [0, {dim - 1}]")

    state_vector = np.zeros(dim, dtype=complex)
    state_vector[state_index] = 1.0

    return state_vector


def basis_projector(state_origin, state_end, nqubits, dtype="bin"):
    """
    Return the sparse operator |state_end><state_origin|.
    """
    dim = 2**nqubits

    if dtype == "bin":
        state_index = binary_to_decimal(state_origin)
        state_index2 = binary_to_decimal(state_end)
    elif dtype == "dec":
        state_index = state_origin
        state_index2 = state_end
    else:
        raise ValueError("dtype must be 'bin' or 'dec'")

    if not 0 <= state_index < dim:
        raise ValueError(
            f"Origin state index {state_index} outside range [0, {dim - 1}]"
        )

    if not 0 <= state_index2 < dim:
        raise ValueError(f"End state index {state_index2} outside range [0, {dim - 1}]")

    return csr_matrix(
        ([1.0 + 0j], ([state_index2], [state_index])),
        shape=(dim, dim),
    )


def _single_qubit_gate(gate_matrix, n_qubits, target):
    """
    Construct a full sparse single-qubit gate.
    """
    if not 0 <= target < n_qubits:
        raise ValueError(f"Target qubit {target} outside range [0, {n_qubits - 1}]")

    return kron(
        kron(
            identity(2**target),
            csr_matrix(gate_matrix),
            format="csr",
        ),
        identity(2 ** (n_qubits - target - 1)),
        format="csr",
    )


def build_gate(name, n_qubits, target, control=None, theta=None):
    """
    Build a sparse matrix representation of a quantum gate.
    """
    if not 0 <= target < n_qubits:
        raise ValueError(f"Target qubit {target} outside range [0, {n_qubits - 1}]")

    if name == "I":
        return identity(2**n_qubits)

    elif name == "X":
        gate_matrix = np.array(
            [[0, 1], [1, 0]],
            dtype=complex,
        )

        return _single_qubit_gate(
            gate_matrix,
            n_qubits,
            target,
        )

    elif name == "Y":
        gate_matrix = np.array(
            [[0, -1j], [1j, 0]],
            dtype=complex,
        )

        return _single_qubit_gate(
            gate_matrix,
            n_qubits,
            target,
        )

    elif name == "Z":
        gate_matrix = np.array(
            [[1, 0], [0, -1]],
            dtype=complex,
        )

        return _single_qubit_gate(
            gate_matrix,
            n_qubits,
            target,
        )

    elif name == "H":
        gate_matrix = (1 / np.sqrt(2)) * np.array(
            [[1, 1], [1, -1]],
            dtype=complex,
        )

        return _single_qubit_gate(
            gate_matrix,
            n_qubits,
            target,
        )

    elif name == "CNOT":
        if control is None:
            raise ValueError("Control qubit must be specified for CNOT gate.")

        if control == target:
            raise ValueError("Control and target qubits must be different.")

        X = csr_matrix(
            np.array(
                [[0, 1], [1, 0]],
                dtype=complex,
            )
        )

        P0 = basis_projector("0", "0", 1)
        P1 = basis_projector("1", "1", 1)

        if target < control:

            # |0><0| acts on the control qubit (not on the target)
            part_0 = kron(
                kron(
                    identity(2**control),
                    P0,
                    format="csr",
                ),
                identity(2 ** (n_qubits - control - 1)),
                format="csr",
            )

            part_1 = kron(
                kron(
                    kron(
                        kron(
                            identity(2**target),
                            X,
                            format="csr",
                        ),
                        identity(2 ** (control - target - 1)),
                        format="csr",
                    ),
                    P1,
                    format="csr",
                ),
                identity(2 ** (n_qubits - control - 1)),
                format="csr",
            )

            return part_0 + part_1

        else:  # control < target

            part_0 = kron(
                kron(
                    identity(2**control),
                    P0,
                    format="csr",
                ),
                identity(2 ** (n_qubits - control - 1)),
                format="csr",
            )

            part_1 = kron(
                kron(
                    kron(
                        kron(
                            identity(2**control),
                            P1,
                            format="csr",
                        ),
                        identity(2 ** (target - control - 1)),
                        format="csr",
                    ),
                    X,
                    format="csr",
                ),
                identity(2 ** (n_qubits - target - 1)),
                format="csr",
            )

            return part_0 + part_1

    elif name == "RX":
        if theta is None:
            raise ValueError("theta must be specified for RX gate.")

        # RX(theta) = exp(-i theta X / 2)
        gate_matrix = np.array(
            [
                [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                [-1j * np.sin(theta / 2), np.cos(theta / 2)],
            ],
            dtype=complex,
        )

        return _single_qubit_gate(
            gate_matrix,
            n_qubits,
            target,
        )

    elif name == "RZ":
        if theta is None:
            raise ValueError("theta must be specified for RZ gate.")

        gate_matrix = np.array(
            [
                [np.exp(-1j * theta / 2), 0],
                [0, np.exp(1j * theta / 2)],
            ],
            dtype=complex,
        )

        return _single_qubit_gate(
            gate_matrix,
            n_qubits,
            target,
        )

    elif name == "RZZ":
        if theta is None:
            raise ValueError("theta must be specified for RZZ gate.")

        if control is None:
            raise ValueError("Control qubit must be specified for RZZ gate.")

        if control == target:
            raise ValueError("Control and target qubits must be different.")

        gate_matrix = csr_matrix(
            np.array(
                [
                    [np.exp(-1j * theta / 2), 0],
                    [0, np.exp(1j * theta / 2)],
                ],
                dtype=complex,
            )
        )

        gate_matrix2 = csr_matrix(
            np.array(
                [
                    [np.exp(1j * theta / 2), 0],
                    [0, np.exp(-1j * theta / 2)],
                ],
                dtype=complex,
            )
        )

        P0 = basis_projector("0", "0", 1)
        P1 = basis_projector("1", "1", 1)

        if target < control:

            part_0 = kron(
                kron(
                    kron(
                        kron(
                            identity(2**target),
                            P0,
                            format="csr",
                        ),
                        identity(2 ** (control - target - 1)),
                        format="csr",
                    ),
                    gate_matrix,
                    format="csr",
                ),
                identity(2 ** (n_qubits - control - 1)),
                format="csr",
            )

            part_1 = kron(
                kron(
                    kron(
                        kron(
                            identity(2**target),
                            P1,
                            format="csr",
                        ),
                        identity(2 ** (control - target - 1)),
                        format="csr",
                    ),
                    gate_matrix2,
                    format="csr",
                ),
                identity(2 ** (n_qubits - control - 1)),
                format="csr",
            )

        else:  # control < target

            part_0 = kron(
                kron(
                    kron(
                        kron(
                            identity(2**control),
                            P0,
                            format="csr",
                        ),
                        identity(2 ** (target - control - 1)),
                        format="csr",
                    ),
                    gate_matrix,
                    format="csr",
                ),
                identity(2 ** (n_qubits - target - 1)),
                format="csr",
            )

            part_1 = kron(
                kron(
                    kron(
                        kron(
                            identity(2**control),
                            P1,
                            format="csr",
                        ),
                        identity(2 ** (target - control - 1)),
                        format="csr",
                    ),
                    gate_matrix2,
                    format="csr",
                ),
                identity(2 ** (n_qubits - target - 1)),
                format="csr",
            )

        return part_0 + part_1

    else:
        raise ValueError(f"Unsupported gate: {name}")


def commutator_norm_hermitian(H0, H1):
    """
    Return the largest absolute eigenvalue of i[H0, H1].

    For Hermitian H0 and H1, i[H0, H1] is Hermitian.
    The result is therefore the spectral norm of the commutator.
    """
    n = H0.shape[0]

    # Random dense initial vector
    v0 = np.random.randn(n) + 1j * np.random.randn(n)

    v0 /= np.linalg.norm(v0)

    def matvec(x):
        return 1j * (H0 @ (H1 @ x) - H1 @ (H0 @ x))

    A_op = LinearOperator(
        shape=(n, n),
        matvec=matvec,
        dtype=np.complex128,
    )

    eigvals = eigsh(
        A_op,
        k=1,
        which="LM",
        maxiter=200000,
        v0=v0,
        return_eigenvectors=False,
    )

    return np.abs(eigvals[0])
