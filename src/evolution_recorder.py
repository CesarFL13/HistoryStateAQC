"""
Sparse record of an evolution, so that not every state vector
(n_steps x D complex numbers) has to be kept in memory.

Idea: at each step, the probabilities |psi_i|^2 are computed from the
vector psi and ONLY the relevant ones are kept, as triples
(step, state_index, probability). They are then converted to the list
of dictionaries {bitstring: prob} expected by plot_state_evolution and
plot_state_heatmap, converting to a bitstring only the indices that
are going to be drawn.

Assumption about the order of the indices (adjust it if it is
different):
  index = number in mixed radix with digits (b_1..b_N, d_1..d_N), where
  b_k are binary, d_k go from 0 to digit_base-1, and the first digit is
  the most significant (as np.unravel_index with shape (2,)*N +
  (digit_base,)*N).
"""

import numpy as np


def index_to_bitstring(i, dims, width, N):
    """Índice entero -> bitstring 'N bits binarios + N dígitos (width chars)'."""
    coords = np.unravel_index(int(i), dims)
    first = "".join(str(int(c)) for c in coords[:N])
    second = "".join(f"{int(c):0{width}d}" for c in coords[N:])
    return first + second


class EvolutionRecorder:
    """
    Parameters
    ----------
    tol_store : float
        Solo se guardan los estados con probabilidad > tol_store en ese paso.
        Elígelo bastante por debajo de la `tol` con la que vayas a dibujar
        (p.ej. 1e-8 si dibujas con 1e-3).
    top_k : int or None
        Cota dura de memoria: como mucho `top_k` estados por paso (los de
        mayor probabilidad). Útil si en algún tramo la evolución se
        "esparce" por muchísimos estados (cerca del mínimo del gap).
    """

    def __init__(self, tol_store=1e-8, top_k=None):
        self.tol_store = tol_store
        self.top_k = top_k
        self.dim = None
        self.n_steps = 0
        self.lost = []  # probabilidad descartada en cada paso (control)
        self._steps, self._idx, self._p = [], [], []

    # -- llamada dentro del bucle de evolución ------------------------------
    def add(self, psi):
        p = psi.real ** 2 + psi.imag ** 2
        if self.dim is None:
            self.dim = p.size
        idx = np.flatnonzero(p > self.tol_store)
        if self.top_k is not None and idx.size > self.top_k:
            keep = np.argpartition(p[idx], -self.top_k)[-self.top_k:]
            idx = idx[keep]
        pk = p[idx]
        self.lost.append(max(1.0 - float(pk.sum()), 0.0))
        self._steps.append(np.full(idx.size, self.n_steps, dtype=np.int32))
        self._idx.append(idx.astype(np.int64))
        self._p.append(pk.astype(np.float32))
        self.n_steps += 1

    # -- conversión al formato de las funciones de dibujo -------------------
    def to_prob_data(self, N, L, digit_base=None, width=None, tol=None):
        """
        Devuelve list[dict[str, float]] (un dict por paso).
        `tol`: si se indica, solo se convierten los estados cuya probabilidad
        máxima en el tiempo supera `tol` (recomendable = la tol del gráfico).
        """
        if digit_base is None:
            digit_base = L + 1
        if width is None:
            width = len(str(L))
        dims = (2,) * N + (digit_base,) * N
        if int(np.prod(dims)) != self.dim:
            raise ValueError(
                f"La dimensión del espacio ({self.dim}) no coincide con "
                f"2^N * digit_base^N = {int(np.prod(dims))}. Revisa N, L o "
                f"digit_base."
            )

        t = np.concatenate(self._steps)
        idx = np.concatenate(self._idx)
        p = np.concatenate(self._p)

        uniq, inv = np.unique(idx, return_inverse=True)
        maxp = np.zeros(uniq.size, dtype=np.float32)
        np.maximum.at(maxp, inv, p)
        selected = maxp > (tol if tol is not None else -1.0)

        names = {
            int(k): index_to_bitstring(uniq[k], dims, width, N)
            for k in np.flatnonzero(selected)
        }
        prob_data = [dict() for _ in range(self.n_steps)]
        keep = selected[inv]
        for ti, k, pi in zip(t[keep].tolist(), inv[keep].tolist(), p[keep].tolist()):
            prob_data[ti][names[k]] = pi
        return prob_data

    # -- guardado / carga ---------------------------------------------------
    def save(self, path, **extra):
        """Guarda las tripletas comprimidas. `extra`: arrays adicionales
        (p.ej. times=..., energies=...)."""
        np.savez_compressed(
            path,
            steps=np.concatenate(self._steps), idx=np.concatenate(self._idx),
            p=np.concatenate(self._p), lost=np.asarray(self.lost),
            dim=self.dim, n_steps=self.n_steps, **extra,
        )

    @classmethod
    def load(cls, path):
        z = np.load(path)
        rec = cls()
        rec.dim, rec.n_steps = int(z["dim"]), int(z["n_steps"])
        rec.lost = z["lost"].tolist()
        rec._steps, rec._idx, rec._p = [z["steps"]], [z["idx"]], [z["p"]]
        return rec
