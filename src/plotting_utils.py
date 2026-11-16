"""
Animation (and static heatmap) of the time evolution of the
probabilities of the computational basis states, distinguishing
between "physical" and "virtual" states.

Assumed format of the bitstrings:
  - The first N characters are binary (0/1).
  - The next N "digits" can take integer values between 0 and L
    (each one takes `width` characters; by default width = len(str(L))).
  - A state is PHYSICAL if those N digits are all equal to each other
    (i.e. the second part is 'i'*N for some i in [0, L]).
    Otherwise it is VIRTUAL.

If your encoding is different (e.g. a separator between digits, or
another criterion), it is enough to change `parse_state` /
`classify_state`.

Requires: matplotlib, numpy
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Símbolo de "l" en las etiquetas de los estados físicos. Por defecto la
# ℓ cursiva de mathtext, que no se confunde con un 1 (a diferencia de una
# "l" normal en la fuente por defecto). Cámbialo con el argumento
# `l_symbol` de las funciones de dibujo: p.ej. "l" (sin distinguir),
# r"$l$" (l cursiva) o r"$\mathrm{\ell}$".
L_SYMBOL = r"$\ell$"


# ---------------------------------------------------------------------------
# Utilidades de exportación
# ---------------------------------------------------------------------------


def _ensure_ffmpeg():
    """True si matplotlib puede usar ffmpeg. Si no está en el PATH, intenta
    usar el binario que trae el paquete `imageio-ffmpeg` (pip install
    imageio-ffmpeg), que no requiere tocar el PATH."""
    if animation.writers.is_available("ffmpeg"):
        return True
    try:
        import imageio_ffmpeg

        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return False
    return animation.writers.is_available("ffmpeg")


# ---------------------------------------------------------------------------
# Parsing / clasificación de estados
# ---------------------------------------------------------------------------


def parse_state(bitstring, N, width):
    """Separa un bitstring en (primera_parte:str, segunda_parte:list[int])."""
    first = bitstring[:N]
    second_str = bitstring[N:]
    second_vals = [int(second_str[i * width : (i + 1) * width]) for i in range(N)]
    return first, second_vals


def classify_state(bitstring, N, L, width=None):
    """Devuelve 'physical' o 'virtual' según la regla de la segunda parte."""
    if width is None:
        width = len(str(L))
    _, second_vals = parse_state(bitstring, N, width)
    return "physical" if len(set(second_vals)) == 1 else "virtual"


def default_tag(bitstring, N, L, width=None, l_symbol=L_SYMBOL):
    """Etiqueta legible por defecto, en notación de ket: parte binaria +
    índice l (si es físico) o los valores de la segunda parte (si es
    virtual), entre '|' y '⟩'."""
    if width is None:
        width = len(str(L))
    first, second_vals = parse_state(bitstring, N, width)
    if len(set(second_vals)) == 1:
        inner = f"{first};{l_symbol}={second_vals[0]}"
    else:
        inner = f"{first};{','.join(map(str, second_vals))}"
    return f"|{inner}\u27e9"


def _split_max_states(max_states):
    """Normaliza `max_states` a un par (max_físicos, max_virtuales).
    Acepta None (sin límite), un entero (se aplica a CADA grupo por
    separado) o una tupla/lista (max_físicos, max_virtuales), donde cada
    elemento puede ser None."""
    if max_states is None:
        return None, None
    if isinstance(max_states, (tuple, list)):
        if len(max_states) != 2:
            raise ValueError(
                "max_states debe ser None, un entero o (max_fis, max_virt)."
            )
        return max_states[0], max_states[1]
    return int(max_states), int(max_states)


def _group_label(name, shown, total):
    """Etiqueta de grupo; indica cuántos se muestran si se ha truncado."""
    if shown < total:
        return f"{name} (top {shown} de {total})"
    return name


def _prepare_groups(
    prob_data, N, L, width, tags, tol, max_states=None, l_symbol=L_SYMBOL
):
    """Lógica común a la animación y al heatmap: qué estados mostrar, en
    qué orden y con qué etiqueta, separados en físicos/virtuales.

    Devuelve (ordered_states, labels, n_phys, n_virt, total_phys, total_virt),
    donde total_* es el nº de estados que superaban `tol` antes de aplicar
    `max_states`."""
    if width is None:
        width = len(str(L))
    max_phys, max_virt = _split_max_states(max_states)

    all_states = set()
    for d in prob_data:
        all_states.update(d.keys())

    max_prob = {s: max(d.get(s, 0.0) for d in prob_data) for s in all_states}
    relevant = {s for s, p in max_prob.items() if p > tol}

    group = {s: classify_state(s, N, L, width) for s in relevant}
    phys_relevant = sorted(
        (s for s in relevant if group[s] == "physical"), key=lambda s: -max_prob[s]
    )
    virt_relevant = sorted(
        (s for s in relevant if group[s] == "virtual"), key=lambda s: -max_prob[s]
    )

    if not phys_relevant and not virt_relevant:
        raise ValueError(
            "Ningún estado supera la tolerancia especificada en ningún instante. "
            "Prueba a bajar `tol`."
        )

    # Límite de estados a mostrar: nos quedamos con los de mayor probabilidad
    # máxima (las listas ya están ordenadas de mayor a menor).
    total_phys, total_virt = len(phys_relevant), len(virt_relevant)
    if max_phys is not None:
        phys_relevant = phys_relevant[:max_phys]
    if max_virt is not None:
        virt_relevant = virt_relevant[:max_virt]

    ordered_states = phys_relevant + virt_relevant
    labels = [
        (tags.get(s) if tags and s in tags else default_tag(s, N, L, width, l_symbol))
        for s in ordered_states
    ]

    return (
        ordered_states,
        labels,
        len(phys_relevant),
        len(virt_relevant),
        total_phys,
        total_virt,
    )


# ---------------------------------------------------------------------------
# Animación (barras que evolucionan en el tiempo)
# ---------------------------------------------------------------------------


def plot_state_evolution(
    prob_data,
    N,
    L,
    width=None,
    tags=None,
    tol=1e-3,
    max_states=None,
    l_symbol=L_SYMBOL,
    times=None,
    duration=30.0,
    max_fps=30,
    interval=None,
    save_path=None,
    title="Evolución de probabilidades",
    figsize=None,
    dpi=110,
    phys_color="tab:blue",
    virt_color="tab:orange",
    bar_width_per_state=0.45,
    min_fig_width=9,
    max_label_fontsize=9,
    min_label_fontsize=5,
):
    """
    Anima la evolución temporal de las probabilidades de los estados de la
    base computacional, agrupando los estados "físicos" y "virtuales", y
    mostrando sólo aquellos cuya probabilidad supera `tol` en algún instante.

    Parameters
    ----------
    prob_data : list[dict[str, float]]
        Lista con un diccionario por paso temporal: {bitstring: probabilidad}.
    N : int
        Número de qubits/valores en cada una de las dos partes del bitstring.
    L : int
        Valor máximo que puede tomar cada entero de la segunda parte.
    width : int or None
        Nº de caracteres que ocupa cada entero de la segunda parte. Por
        defecto len(str(L)).
    tags : dict[str, str] or None
        Etiqueta legible para bitstrings concretos (sobreescribe la etiqueta
        automática, que ya viene en notación de ket: |bits,l=k⟩ para
        físicos, |bits,v1,v2,...⟩ para virtuales). No hace falta que cubra
        todos los estados.
    tol : float
        Probabilidad mínima para que un estado se muestre (para decidir si
        entra en el eje x, y también frame a frame).
    max_states : int, tuple[int, int] or None
        Máximo de estados a mostrar, quedándose con los de mayor probabilidad
        máxima a lo largo del tiempo. Un entero se aplica a cada grupo por
        separado (así ambos grupos siguen representados); una tupla
        (max_físicos, max_virtuales) permite límites distintos (usa None
        en una posición para no limitar ese grupo). Si se trunca, la etiqueta
        del grupo indica "top K de M". Se aplica DESPUÉS de filtrar por `tol`.
    l_symbol : str
        Símbolo usado para "l" en las etiquetas automáticas de los estados
        físicos (|bits;l=k⟩). Por defecto r"$\\ell$" (ℓ cursiva), que no se
        confunde con un 1. Usa "l" para la letra normal. No afecta a las
        etiquetas que pases en `tags`.
    times : list[float] or None
        Valores de tiempo para cada paso. Si None, se usan índices 0..N-1.
    duration : float
        Duración (segundos) de la animación, sea cual sea el nº de pasos
        temporales. Si hay más pasos de los que caben a `max_fps`, se
        muestra un subconjunto equiespaciado (siempre incluye el primero y
        el último); si hay menos, se muestran todos, más despacio.
    max_fps : float
        Máximo de frames por segundo (y por tanto de frames dibujados:
        duration * max_fps).
    interval : float or None
        Milisegundos entre frames. Si se pasa explícitamente, se muestran
        todos los pasos con ese intervalo y se ignoran `duration` y
        `max_fps` (la duración escala entonces con el nº de pasos).
    save_path : str or None
        Si se indica, guarda la animación ahí (.gif vía Pillow, sin
        necesidad de ffmpeg; .mp4 si tienes ffmpeg instalado).
    title : str
        Título del gráfico.
    figsize : tuple or None
        Si None (por defecto), el ancho se calcula automáticamente en
        función del nº de estados a mostrar (ver `bar_width_per_state` y
        `min_fig_width`), para que las barras no se aprieten cuando hay
        muchos estados relevantes. Pásalo explícitamente para fijarlo tú.
    dpi : int
        Resolución de la figura.
    phys_color, virt_color :
        Color único (nombre o hex de matplotlib) para todas las barras
        físicas y todas las virtuales, respectivamente.
    bar_width_per_state : float
        Pulgadas de ancho de figura reservadas por cada estado mostrado,
        usado para el auto-dimensionado cuando `figsize` es None.
    min_fig_width : float
        Ancho mínimo de la figura (pulgadas), aunque haya pocos estados.
    max_label_fontsize, min_label_fontsize : float
        El tamaño de letra de las etiquetas del eje x se reduce
        automáticamente cuantos más estados haya, entre estos dos límites.

    Returns
    -------
    ani, fig
    """
    n_frames = len(prob_data)
    if n_frames == 0:
        raise ValueError("prob_data está vacío.")
    if times is None:
        times = list(range(n_frames))
    # Duración fija: si hay más pasos de los que caben en `duration` segundos
    # a `max_fps`, se submuestrean (equiespaciados, con el primero y el
    # último); si hay menos, se muestran todos bajando los fps.
    if interval is None:
        if duration <= 0 or max_fps <= 0:
            raise ValueError("duration y max_fps deben ser positivos.")
        n_shown = min(n_frames, max(1, round(duration * max_fps)))
        frames = np.linspace(0, n_frames - 1, n_shown).round().astype(int)
        fps = n_shown / duration
        interval = 1000.0 / fps
    else:
        frames = np.arange(n_frames)
        fps = 1000.0 / interval
    if width is None:
        width = len(str(L))

    ordered_states, labels, n_phys, n_virt, total_phys, total_virt = _prepare_groups(
        prob_data, N, L, width, tags, tol, max_states, l_symbol
    )
    n_total = n_phys + n_virt

    gap = 1.0
    x_phys = np.arange(n_phys)
    x_virt = np.arange(n_virt) + n_phys + gap
    x_positions = np.concatenate([x_phys, x_virt]) if n_virt else x_phys

    bar_colors = [phys_color] * n_phys + [virt_color] * n_virt

    # Auto-dimensionado: cuantos más estados relevantes haya, más ancha se
    # hace la figura (en vez de que las barras se encojan en un ancho fijo).
    if figsize is None:
        fig_width = max(min_fig_width, n_total * bar_width_per_state)
        figsize = (fig_width, 6)
    label_fontsize = max(
        min_label_fontsize, min(max_label_fontsize, 300 / max(n_total, 1))
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    bars = ax.bar(
        x_positions,
        np.zeros(len(ordered_states)),
        color=bar_colors,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=label_fontsize)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Probabilidad")
    ax.axhline(tol, color="grey", linestyle=":", linewidth=1, label=f"tol = {tol:g}")

    if n_phys and n_virt:
        sep_x = (x_phys[-1] + x_virt[0]) / 2
        ax.axvline(sep_x, color="black", linestyle="--", linewidth=1)
    if n_phys:
        ax.text(
            x_phys.mean(),
            1.02,
            _group_label("Físicos", n_phys, total_phys),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="tab:blue",
            transform=ax.get_xaxis_transform(),
        )
    if n_virt:
        ax.text(
            x_virt.mean(),
            1.02,
            _group_label("Virtuales", n_virt, total_virt),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="tab:orange",
            transform=ax.get_xaxis_transform(),
        )

    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )
    # El título va en la figura (fig.suptitle), no en el eje (ax.set_title):
    # así queda por encima de las etiquetas "Físicos"/"Virtuales" en vez de
    # solaparse con ellas, sea cual sea el tamaño de la figura.
    fig.suptitle(title, fontsize=13, y=0.97)
    ax.legend(loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    def update(frame):
        d = prob_data[frame]
        for bar, s in zip(bars, ordered_states):
            p = d.get(s, 0.0)
            if p >= tol:
                bar.set_height(p)
                bar.set_alpha(1.0)
            else:
                bar.set_height(0.0)
                bar.set_alpha(0.25)
        time_text.set_text(f"t = {times[frame]:.3g}")
        return list(bars) + [time_text]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames.tolist(),
        interval=interval,
        blit=False,
    )

    if save_path:
        if save_path.endswith(".gif"):
            ani.save(save_path, writer=animation.PillowWriter(fps=fps))
        elif save_path.endswith(".mp4"):
            if not _ensure_ffmpeg():
                raise RuntimeError(
                    "No se encuentra ffmpeg, necesario para guardar .mp4. "
                    "Opciones: `pip install imageio-ffmpeg` (lo más simple; "
                    "se detecta automáticamente), `conda install -c "
                    "conda-forge ffmpeg`, o instalar ffmpeg y añadirlo al "
                    "PATH. Mientras tanto puedes guardar como .gif."
                )
            # h264 + yuv420p: el formato que PowerPoint reproduce de forma
            # fiable, con su propio reproductor (con barra de tiempo
            # arrastrable, play/pausa) al insertarlo como vídeo.
            ani.save(
                save_path,
                writer="ffmpeg",
                fps=fps,
                extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
            )
        else:
            ani.save(save_path)

    return ani, fig


# ---------------------------------------------------------------------------
# Heatmap estático (estados en Y, tiempo en X, color = probabilidad)
# ---------------------------------------------------------------------------


def plot_state_heatmap(
    prob_data,
    N,
    L,
    width=None,
    tags=None,
    tol=1e-3,
    max_states=None,
    l_symbol=L_SYMBOL,
    times=None,
    title="Evolución de probabilidades (heatmap)",
    figsize=None,
    dpi=110,
    cmap="viridis",
    save_path=None,
    fig_width=10,
    row_height_per_state=0.35,
    min_fig_height=5,
    max_label_fontsize=9,
    min_label_fontsize=5,
):
    """
    Versión estática: un heatmap con los estados relevantes en el eje Y
    (físicos arriba, virtuales abajo, separados por una línea) y el tiempo
    en el eje X. El color codifica la probabilidad en cada instante.

    Aquí lo que se satura con muchos estados es el eje Y (las etiquetas se
    amontonan), así que si `figsize` es None (por defecto) la altura de la
    figura se calcula automáticamente en función del nº de estados a
    mostrar (`row_height_per_state`, `min_fig_height`), y el tamaño de
    letra de las etiquetas se reduce entre `max_label_fontsize` y
    `min_label_fontsize`. Pasa `figsize` explícitamente para fijarlo tú.

    Resto de parámetros: igual que en `plot_state_evolution` (salvo
    `interval`, que no aplica aquí, y `cmap`, el colormap de matplotlib
    para las probabilidades).

    Returns
    -------
    fig, ax
    """
    n_frames = len(prob_data)
    if n_frames == 0:
        raise ValueError("prob_data está vacío.")
    if times is None:
        times = list(range(n_frames))
    if width is None:
        width = len(str(L))

    ordered_states, labels, n_phys, n_virt, total_phys, total_virt = _prepare_groups(
        prob_data, N, L, width, tags, tol, max_states, l_symbol
    )
    n_total = n_phys + n_virt

    # matriz (n_estados x n_frames)
    matrix = np.array([[d.get(s, 0.0) for d in prob_data] for s in ordered_states])

    if figsize is None:
        fig_height = max(min_fig_height, n_total * row_height_per_state)
        figsize = (fig_width, fig_height)
    label_fontsize = max(
        min_label_fontsize, min(max_label_fontsize, 300 / max(n_total, 1))
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    extent = [times[0], times[-1], len(ordered_states) - 0.5, -0.5]
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=0,
        vmax=matrix.max(),
        extent=extent,
        interpolation="nearest",
    )

    ax.set_yticks(range(len(ordered_states)))
    ax.set_yticklabels(labels, fontsize=label_fontsize)
    ax.set_xlabel("Tiempo")
    ax.set_title(title)

    # separador horizontal entre físicos y virtuales
    if n_phys and n_virt:
        ax.axhline(n_phys - 0.5, color="white", linestyle="--", linewidth=1.5)

    # Etiquetas de grupo a la derecha. Los físicos ocupan las primeras
    # `n_phys` filas (arriba, fila 0 = arriba en imshow) y los virtuales
    # las últimas `n_virt` (abajo). En coordenadas de ejes (transAxes),
    # y=1 es arriba y y=0 es abajo, así que convertimos el índice de fila
    # medio de cada grupo a esa escala invertida.
    n_total = max(len(ordered_states) - 1, 1)
    if n_phys:
        mean_row_phys = (n_phys - 1) / 2
        ax.text(
            1.01,
            1 - mean_row_phys / n_total,
            _group_label("Físicos", n_phys, total_phys),
            rotation=90,
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color="tab:blue",
            transform=ax.transAxes,
        )
    if n_virt:
        mean_row_virt = n_phys + (n_virt - 1) / 2
        ax.text(
            1.01,
            1 - mean_row_virt / n_total,
            _group_label("Virtuales", n_virt, total_virt),
            rotation=90,
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
            color="tab:orange",
            transform=ax.transAxes,
        )

    cbar = fig.colorbar(im, ax=ax, pad=0.12)
    cbar.set_label("Probabilidad")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax
