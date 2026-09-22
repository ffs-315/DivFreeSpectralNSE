import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pathlib import Path


figures_dir = Path(
    "/home/jbq/shenfun_projects/Div_free_Spectral_Methods/Figures"
)
figures_dir.mkdir(parents=True, exist_ok=True)


# The timing arrays contain ten entries, so N=72 is included here.
N_values = np.array(
    [8, 16, 24, 32, 40, 48, 56, 64, 72, 80],
    dtype=int,
)

time_imex = np.array(
    [1.29, 1.39, 1.50, 1.93, 3.30,
     4.53, 8.85, 15.36, 23.04, 33.81],
    dtype=float,
)

time_pc = np.array(
    [1.42, 1.61, 1.74, 2.37, 3.62,
     5.52, 11.39, 18.04, 26.86, 38.82],
    dtype=float,
)


if not (
    N_values.size == time_imex.size == time_pc.size
):
    raise ValueError(
        "N_values, time_imex, and time_pc must have the same length."
    )

if np.any(time_imex <= 0.0) or np.any(time_pc <= 0.0):
    raise ValueError("All measured times must be positive.")


# The two timing curves begin to separate clearly at N=32.  A power law
# T(N) = C*N**p is fitted by least squares in logarithmic coordinates.
fit_min_N = 32
fit_mask = N_values >= fit_min_N

pcg_power, pcg_log_coefficient = np.polyfit(
    np.log(N_values[fit_mask]),
    np.log(time_imex[fit_mask]),
    1,
)
scipy_power, scipy_log_coefficient = np.polyfit(
    np.log(N_values[fit_mask]),
    np.log(time_pc[fit_mask]),
    1,
)

fit_N = np.geomspace(fit_min_N, N_values[-1], 200)
pcg_fit = np.exp(pcg_log_coefficient)*fit_N**pcg_power
scipy_fit = np.exp(scipy_log_coefficient)*fit_N**scipy_power


fig, ax = plt.subplots(figsize=(7, 5))

ax.loglog(
    N_values,
    time_imex,
    color="blue",
    linestyle=":",
    linewidth=3.5,
    marker="D",
    markersize=10,
    markerfacecolor="white",
    markeredgecolor="blue",
    markeredgewidth=2.5,
    label=rf"IMEX-CN scheme",
    zorder=2,
)

ax.loglog(
    N_values,
    time_pc,
    color="green",
    linestyle=":",
    linewidth=3.5,
    marker="o",
    markersize=10,
    markerfacecolor="white",
    markeredgecolor="green",
    markeredgewidth=2.5,
    label=rf"PC-CN scheme",
    zorder=2,
)

# Solid lines show the least-squares power-law fits over N >= fit_min_N.
ax.loglog(
    fit_N,
    pcg_fit,
    color="black",
    linestyle="-",
    linewidth= 3.5,
    label="_nolegend_",
    zorder=3,
    alpha=0.6,
)

ax.loglog(
    fit_N,
    scipy_fit,
    color="red",
    linestyle="-",
    linewidth= 3.5,
    label="_nolegend_",
    zorder=3,
    alpha=0.6,
)


# Vertical arrows identify the empirical powers on the fitted lines.
arrow_settings = [
    (
        44.0,
        np.exp(pcg_log_coefficient)*44.0**pcg_power,
        pcg_power,
        "black",
        10.0,
    ),
    (
        56.0,
        np.exp(scipy_log_coefficient)*56.0**scipy_power,
        scipy_power,
        "red",
        2.5,
    ),
]

for arrow_N, arrow_bottom, fitted_power, color, height_factor in (
    arrow_settings
):
    arrow_top = height_factor*arrow_bottom

    ax.annotate(
        "",
        xy=(arrow_N, arrow_top),
        xytext=(arrow_N, arrow_bottom),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linestyle": (0, (3, 3)),
            "linewidth": 2.5,
            "mutation_scale": 14,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=4,
    )

    ax.annotate(
        rf"$\mathrm{{N}}^{{{fitted_power:.2f}}}$",
        xy=(arrow_N, arrow_top),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=18,
        color=color,
        zorder=5,
    )


ax.set_xlim(7.5, 84.0)
ax.set_ylim(1.0, 160.0)

ax.set_xlabel(r"$\mathrm{Spatial\ polynomial\ degree\ N}$", fontsize=18)
shown_x_ticks = np.array([8, 16, 24, 32, 48, 64, 80])
ax.set_xticks(shown_x_ticks)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.tick_params(axis="both", which="major", labelsize=18)
ax.tick_params(axis="both", which="minor", labelsize=15)

legend = ax.legend(
    loc="upper left",
    fontsize=18,
    frameon=True,
    framealpha=0.3,
    title=rf"$\mathrm{{CPU \, \, time \,(s)}}$",
    title_fontsize=18,
)
legend.get_frame().set_edgecolor("black")

fig.subplots_adjust(
    left=0.16,
    right=0.98,
    bottom=0.16,
    top=0.98,
)

figure_file = figures_dir / "imex_pc_total_time_comparison.png"
fig.savefig(
    figure_file,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.02,
    pil_kwargs={"optimize": True, "compress_level": 9},
)

print(f"Figure saved to: {figure_file}")
print(
    f"PCG fit for N >= {fit_min_N}: "
    f"T = {np.exp(pcg_log_coefficient):.4e} N^{pcg_power:.4f}"
)
print(
    f"Sparse direct fit for N >= {fit_min_N}: "
    f"T = {np.exp(scipy_log_coefficient):.4e} N^{scipy_power:.4f}"
)

plt.show()
