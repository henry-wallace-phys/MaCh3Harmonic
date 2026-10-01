import jax
import numpy as np
import re
from collections.abc import Sequence
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import harmonic as hm
from logging import getLogger

from mach3_harmonic.file_io.chain_reader import filter_keys
from mach3_harmonic.stats.evidence import batched_sample, PREDICT_BATCH

from getdist import MCSamples, plots

COLOUR_MACH3 = "#56B4E9"
COLOUR_HARMONIC = "#DB7138"

# Line styles for the flow's CI markers, cycled if there are more levels than styles
FLOW_CI_STYLES = ["--", ":", "-.", (0, (1, 3)), (0, (5, 1, 1, 1))]


def to_latex_label(name: str) -> str:
    """Turn a MaCh3 parameter name into something getdist can render.

    getdist wraps labels in $...$, so raw underscores/spaces would break LaTeX.
    Fancy names using ROOT TLatex syntax (#Delta, #theta...) are converted to LaTeX.
    """
    if "#" in name:
        return name.replace("#", "\\")
    return r"\mathrm{" + name.replace("_", r"\_").replace(" ", r"\ ") + "}"


def _format_pct(level: float) -> str:
    """0.68 -> '68%', 0.997 -> '99.7%'."""
    return f"{100 * level:.1f}".rstrip("0").rstrip(".") + "%"


def _interval(stats, name: str, level_index: int) -> tuple[float, float]:
    """(lower, upper) of the getdist marginalised limit at contours[level_index]."""
    lim = stats.parWithName(name).limits[level_index]
    return lim.lower, lim.upper


def _plot_1d(
    chain: MCSamples,
    flow: MCSamples,
    names: list[str],
    plot_name: str,
    title: str | None = None,
    log_floor: float = 1e-4,
):
    """One page per parameter, drawn with getdist's plot_1d.

    Top panel: linear density. Bottom panel: the same densities on a log
    y-axis, which exposes how well the flow reproduces the tails (these set the
    outer CIs). Every level in chain.contours is shown: chain CIs are shaded
    bands (lighter for wider intervals), flow CIs are getdist x-markers.
    """
    chain_stats = chain.getMargeStats()
    flow_stats = flow.getMargeStats()
    levels = list(chain.contours)
    n_levels = len(levels)
    pct = [_format_pct(c) for c in levels]
    band_alphas = np.linspace(0.30, 0.08, n_levels)
    flow_styles = [FLOW_CI_STYLES[i % len(FLOW_CI_STYLES)] for i in range(n_levels)]

    plot_kwargs = dict(
        colors=[COLOUR_MACH3, COLOUR_HARMONIC],
        ls=["-", "--"],
        lws=[2.0, 2.0],
        normalized=True,  # unit area, so the log panel compares real tail mass
    )

    # Legend is the same on every page, so build it once
    handles = [
        plt.Line2D([], [], color=COLOUR_MACH3, lw=2),
        plt.Line2D([], [], color=COLOUR_HARMONIC, lw=2, ls="--"),
    ]
    texts = [chain.label or "MCMC chain", flow.label or "Flow"]
    for p, alpha in zip(pct, band_alphas):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=COLOUR_MACH3, alpha=alpha))
        texts.append(f"Chain {p} CI")
    for p, ls in zip(pct, flow_styles):
        handles.append(plt.Line2D([], [], color=COLOUR_HARMONIC, ls=ls, lw=1.2))
        texts.append(f"Flow {p} CI")

    with PdfPages(plot_name) as pdf:
        for name in names:
            g = plots.get_subplot_plotter(width_inch=8)
            g.settings.axes_fontsize = 11
            g.settings.axes_labelsize = 14
            g.settings.legend_fontsize = 11
            g.settings.figure_legend_frame = False
            g.settings.prob_y_ticks = True  # we need readable ticks on the log axis
            g.settings.norm_prob_label = "Probability density"
            g.make_figure(nx=1, ny=2, sharex=True, ystretch=0.6)
            ax_lin, ax_log = g.get_axes((0, 0)), g.get_axes((1, 0))  # (row, col)

            ci_chain = [_interval(chain_stats, name, i) for i in range(n_levels)]
            ci_flow = [_interval(flow_stats, name, i) for i in range(n_levels)]

            for ax in (ax_lin, ax_log):
                g.plot_1d([chain, flow], name, ax=ax, **plot_kwargs)
                for (lo, hi), alpha in zip(ci_chain, band_alphas):
                    ax.axvspan(lo, hi, color=COLOUR_MACH3, alpha=alpha, lw=0, zorder=0)
                for (lo, hi), ls in zip(ci_flow, flow_styles):
                    g.add_x_marker([lo, hi], color=COLOUR_HARMONIC, ls=ls, lw=1.2, ax=ax)

            # Log panel: rescale y so the tails are visible
            ymax = ax_log.get_ylim()[1]
            ax_log.set_yscale("log")
            ax_log.set_ylim(ymax * log_floor, ymax * 2)
            ax_log.set_ylabel("Density (log)", fontsize=g.settings.axes_labelsize)
            ax_lin.set_xlabel("")
            ax_lin.tick_params(labelbottom=False)

            ax_lin.legend(handles, texts, loc="upper right", frameon=False,
                          fontsize=g.settings.legend_fontsize)

            # Numerical CI comparison
            rows = [f"{'CI':<7}{'chain':<22}flow"]
            for p, (cl, cu), (fl, fu) in zip(pct, ci_chain, ci_flow):
                rows.append(f"{p:<7}{f'[{cl:+.3g}, {cu:+.3g}]':<22}[{fl:+.3g}, {fu:+.3g}]")
            ax_lin.text(0.02, 0.97, "\n".join(rows), transform=ax_lin.transAxes,
                        va="top", ha="left", family="monospace", fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

            label = chain.paramNames.parWithName(name).label
            ax_lin.set_title(f"{title}: ${label}$" if title else f"${label}$",
                             fontsize=15, pad=10)
            g.fig.subplots_adjust(hspace=0.06)

            pdf.savefig(g.fig, bbox_inches="tight")
            plt.close(g.fig)


def _plot_triangle(
    chain: MCSamples,
    flow: MCSamples,
    names: list[str],
    plot_name: str,
    title: str | None = None,
):
    """Triangle plot of the MCMC chain (filled) against flow samples (lines)."""
    ndim = len(names)
    n_levels = len(chain.contours)

    g = plots.get_subplot_plotter(width_inch=max(8.0, 1.8 * ndim))
    g.settings.axes_fontsize = 11
    g.settings.axes_labelsize = 14
    g.settings.legend_fontsize = 16
    # Each extra level darkens the fill; ease off so 3+ bands stay distinguishable
    g.settings.alpha_filled_add = 0.7 if n_levels <= 2 else 0.5
    g.settings.figure_legend_frame = False
    g.settings.num_plot_contours = n_levels  # draw every level set on the samples

    g.triangle_plot(
        [chain, flow],
        params=names,
        filled=[True, False],
        contour_colors=[COLOUR_MACH3, COLOUR_HARMONIC],
        contour_lws=[1.0, 1.8],
        contour_ls=["-", "--"],
        legend_loc="upper right",
        title_limit=1,  # chain interval at the first contour level above each 1D panel
    )
    if title:
        g.fig.suptitle(title, fontsize=18, y=1.01)

    g.export(plot_name)
    plt.close(g.fig)


def plot_flow_vs_chain(
    chain_samples: np.ndarray,
    flow_samples: np.ndarray,
    param_names,
    param_labels,
    plot_name: str,
    title: str | None = None,
    flow_label: str = "Normalising flow",
    contours: Sequence[float] = (0.68, 0.95, 0.997),
):
    contours = sorted(float(c) for c in contours)
    if not contours:
        raise ValueError("Need at least one contour level")
    if any(c <= 0 or c >= 1 for c in contours):
        raise ValueError(f"Contour levels must lie strictly in (0, 1), got {contours}")

    names = [re.sub(r"\W", "_", str(n)) for n in param_names]
    labels = [to_latex_label(str(l)) for l in param_labels]

    settings = {
        "smooth_scale_1D": 0.3,
        "smooth_scale_2D": 0.3,
        "ignore_rows": 0,
        "contours": contours,
    }
    chain = MCSamples(samples=chain_samples, names=names, labels=labels,
                      label="MCMC chain", settings=settings)
    flow = MCSamples(samples=flow_samples, names=names, labels=labels,
                     label=flow_label, settings=settings)

    triangle_file = f"triangle_{plot_name}"
    getLogger().info(f"Making triangle plot : {triangle_file}")
    _plot_triangle(chain, flow, names, triangle_file, title)

    oned_file = f"1d_{plot_name}"
    getLogger().info(f"Making 1D plots      : {oned_file} ({len(names)} pages)")
    _plot_1d(chain, flow, names, oned_file, title)


def plot_flow(model, chains: hm.Chains | np.ndarray, param_names, param_labels,
              plot_name: str, title: str | None = None, n_max: int = 100_000,
              seed: int = 0, sample_batch_size: int = PREDICT_BATCH,
              pars_to_ignore: list[str] | None = None):
    """Triangle + 1D comparison of the flow against a chain (flat or hm.Chains).

    Parameters matching a wildcard in pars_to_ignore are left out of the plots
    only; the flow is still sampled in every dimension.
    """
    chain_flat = np.asarray(chains.samples if isinstance(chains, hm.Chains) else chains)
    chain_flat = chain_flat.reshape(-1, chain_flat.shape[-1])
    flow_samples = batched_sample(model, min(n_max, len(chain_flat)),
                                  jax.random.PRNGKey(seed), batch_size=sample_batch_size)

    param_names = list(param_names)
    kept = set(filter_keys(param_names, pars_to_ignore or []))
    keep = [i for i, n in enumerate(param_names) if n in kept]
    if not keep:
        raise ValueError(f"Plotting.pars_to_ignore {pars_to_ignore} removes every parameter")

    plot_flow_vs_chain(
        chain_flat[:, keep], flow_samples[:, keep],
        [param_names[i] for i in keep], [list(param_labels)[i] for i in keep], plot_name,
        title=title, flow_label=f"Normalising flow (T = {model.temperature})",
    )