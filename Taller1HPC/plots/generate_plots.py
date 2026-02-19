#!/usr/bin/env python3
"""
generate_plots.py — Performance analysis plots for 2D Poisson Jacobi solver
Universidad Distrital Francisco José de Caldas — Sistemas Distribuidos 2026-1
Hardware: Apple M3 (arm64), 128KB L1D, 16MB L2, 128-byte cache lines
"""

import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
DATA_O0 = BASE / "benchmark/results/results_all.csv"
DATA_O2 = BASE / "benchmark/results/results_O2.csv"
OUT     = BASE / "plots"
OUT.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  12,
    "legend.fontsize": 10,
    "figure.dpi":      150,
    "axes.grid":       True,
    "grid.alpha":      0.35,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Load data ─────────────────────────────────────────────────────────────────
def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["N"]            = int(r["N"])
            r["block"]        = int(r["block"])
            r["iters"]        = int(r["iters"])
            r["time_s"]       = float(r["time_s"].strip())
            r["ms_per_iter"]  = float(r["ms_per_iter"].strip())
            r["max_diff"]     = float(r["max_diff"].strip())
            rows.append(r)
    return rows

d0 = load_csv(DATA_O0)   # no optimization flags
d2 = load_csv(DATA_O2)   # -O2

def query(data, **kw):
    """Filter rows matching keyword conditions."""
    out = data
    for k, v in kw.items():
        if isinstance(v, list):
            out = [r for r in out if r[k] in v]
        else:
            out = [r for r in out if r[k] == v]
    return out

def get_ms(data, **kw):
    rows = query(data, **kw)
    return rows[0]["ms_per_iter"] if rows else None

SIZES  = [512, 1024, 2048, 4096]
COLORS = {"ptr":    "#2196F3",
          "flat":   "#4CAF50",
          "vector": "#FF9800",
          "f90":    "#9C27B0"}
HATCHES = {"ij": "", "ji": "///"}

# =============================================================================
# FIGURE 1 — Impacto del orden de bucles: ms/iter vs N (log-log)
# Para cada variante: línea sólida = orden bueno, punteada = orden malo
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle("Figura 1 — Impacto del Orden de Bucles en el Tiempo por Iteración\n"
             "(Apple M3 arm64, sin flags vs con -O2)", fontweight="bold")

for ax_idx, (data, label) in enumerate([(d0, "Sin flags (-O0)"), (d2, "Con -O2")]):
    ax = axes[ax_idx]

    # C++ flat
    ms_ij = [get_ms(data, N=n, storage="flat", order="ij", block=0) for n in SIZES]
    ms_ji = [get_ms(data, N=n, storage="flat", order="ji", block=0) for n in SIZES]
    ax.plot(SIZES, ms_ij, "o-",  color=COLORS["flat"],   lw=2, label="C++ flat (orden ij ✓)")
    ax.plot(SIZES, ms_ji, "o--", color=COLORS["flat"],   lw=2, label="C++ flat (orden ji ✗)", alpha=0.7)

    # C++ ptr
    ms_ij = [get_ms(data, N=n, storage="ptr", order="ij", block=0) for n in SIZES]
    ms_ji = [get_ms(data, N=n, storage="ptr", order="ji", block=0) for n in SIZES]
    ax.plot(SIZES, ms_ij, "s-",  color=COLORS["ptr"],    lw=2, label="C++ ptr  (orden ij ✓)")
    ax.plot(SIZES, ms_ji, "s--", color=COLORS["ptr"],    lw=2, label="C++ ptr  (orden ji ✗)", alpha=0.7)

    # Fortran
    ms_ji = [get_ms(data, N=n, lang="fortran", order="ji", block=0) for n in SIZES]
    ms_ij = [get_ms(data, N=n, lang="fortran", order="ij", block=0) for n in SIZES]
    ax.plot(SIZES, ms_ji, "^-",  color=COLORS["f90"],    lw=2, label="Fortran  (orden ji ✓)")
    ax.plot(SIZES, ms_ij, "^--", color=COLORS["f90"],    lw=2, label="Fortran  (orden ij ✗)", alpha=0.7)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(n) for n in SIZES])
    ax.set_xlabel("Tamaño de malla N")
    ax.set_ylabel("Tiempo por iteración (ms)")
    ax.set_title(label)
    ax.legend(loc="upper left", fontsize=9)

    # Add ratio annotations for N=4096
    for storage, color in [("flat", COLORS["flat"]), ("ptr", COLORS["ptr"])]:
        good = get_ms(data, N=4096, storage=storage, order="ij", block=0)
        bad  = get_ms(data, N=4096, storage=storage, order="ji", block=0)
        if good and bad:
            ratio = bad / good
            ax.annotate(f"×{ratio:.1f}", xy=(4096, bad), xytext=(-40, 5),
                        textcoords="offset points", color=color, fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT / "fig1_loop_order_impact.png", bbox_inches="tight")
plt.close()
print("Saved fig1_loop_order_impact.png")

# =============================================================================
# FIGURE 2 — Factor de penalización por orden incorrecto vs N
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_title("Figura 2 — Penalización por Orden Incorrecto de Bucles\n"
             "(ratio tiempo_malo / tiempo_bueno, -O2)", fontweight="bold")

variants = [
    ("C++ ptr",    d2, "ptr",    "ij", "ji",  COLORS["ptr"],    "o-"),
    ("C++ flat",   d2, "flat",   "ij", "ji",  COLORS["flat"],   "s-"),
    ("C++ vector", d2, "vector", "ij", "ji",  COLORS["vector"], "D-"),
    ("Fortran",    d2, "f90",    "ji", "ij",  COLORS["f90"],    "^-"),
]
for name, data, stor, good_ord, bad_ord, color, marker in variants:
    if stor == "f90":
        ratios = [get_ms(data, N=n, lang="fortran", order=bad_ord, block=0) /
                  get_ms(data, N=n, lang="fortran", order=good_ord, block=0)
                  for n in SIZES]
    else:
        ratios = [get_ms(data, N=n, storage=stor, order=bad_ord, block=0) /
                  get_ms(data, N=n, storage=stor, order=good_ord, block=0)
                  for n in SIZES]
    ax.plot(SIZES, ratios, marker, color=color, lw=2, markersize=8, label=name)
    for x, y in zip(SIZES, ratios):
        ax.annotate(f"{y:.1f}×", xy=(x, y), xytext=(4, 4),
                    textcoords="offset points", fontsize=9, color=color)

ax.axhline(1.0, color="gray", lw=1, linestyle=":", label="sin penalización")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(SIZES)
ax.set_xticklabels([str(n) for n in SIZES])
ax.set_xlabel("Tamaño de malla N")
ax.set_ylabel("Penalización (× veces más lento)")
ax.legend()
ax.set_ylim(0.8, None)
plt.tight_layout()
plt.savefig(OUT / "fig2_loop_penalty_vs_N.png", bbox_inches="tight")
plt.close()
print("Saved fig2_loop_penalty_vs_N.png")

# =============================================================================
# FIGURE 3 — Comparación de tipos de almacenamiento en C++
# ms/iter para N=512 y N=4096 (par de barras: sin flags y con -O2)
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle("Figura 3 — Comparación de Tipos de Almacenamiento en C++\n"
             "(orden ij — bueno para row-major)", fontweight="bold")

for ax, N in zip(axes, [512, 4096]):
    storages = ["ptr", "flat", "vector"]
    x        = np.arange(len(storages))
    w        = 0.35

    ms_o0 = [get_ms(d0, N=N, storage=s, order="ij", block=0) for s in storages]
    ms_o2 = [get_ms(d2, N=N, storage=s, order="ij", block=0) for s in storages]

    b0 = ax.bar(x - w/2, ms_o0, w, label="Sin flags (-O0)",
                color=[COLORS[s] for s in storages], alpha=0.55, edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + w/2, ms_o2, w, label="Con -O2",
                color=[COLORS[s] for s in storages], alpha=1.0, edgecolor="black", linewidth=0.8)

    for bar, val in zip(b0, ms_o0):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*max(ms_o0),
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, color="gray")
    for bar, val in zip(b2, ms_o2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*max(ms_o0),
                f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["double**\n(ptr)", "new double[]\n(flat)", "std::vector\n(vector)"])
    ax.set_ylabel("ms / iteración")
    ax.set_title(f"N = {N}")
    ax.legend(loc="upper right")

    # Speedup labels
    for i, (v0, v2) in enumerate(zip(ms_o0, ms_o2)):
        if v0 and v2:
            speedup = v0 / v2
            ax.text(i, max(ms_o0) * 1.08, f"↑{speedup:.1f}×",
                    ha="center", fontsize=9, color="red", fontweight="bold")

plt.tight_layout()
plt.savefig(OUT / "fig3_storage_comparison.png", bbox_inches="tight")
plt.close()
print("Saved fig3_storage_comparison.png")

# =============================================================================
# FIGURE 4 — Efecto del blocking/tiling: ms/iter vs tamaño de bloque B
# Para distintos N, ambos lenguajes, con -O2
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Figura 4 — Efecto del Cache Tiling (Blocking) vs Tamaño de Bloque B\n"
             "Con -O2 — Apple M3 (L1D=128KB, L2=16MB, línea caché=128B)",
             fontweight="bold")

block_sizes = [0, 8, 16, 32, 64]
block_labels = ["sin\nblocking", "B=8", "B=16", "B=32", "B=64"]

# Caché L1 capacity lines
L1_KB = 128
for ax, N in zip(axes.flat, SIZES):
    x = np.arange(len(block_sizes))

    # C++ flat ij
    ms_cpp = [get_ms(d2, N=N, storage="flat", order="ij", block=b) for b in block_sizes]
    # Fortran ji
    ms_f90 = [get_ms(d2, N=N, lang="fortran", order="ji", block=b) for b in block_sizes]

    ax.plot(x, ms_cpp, "o-", color=COLORS["flat"],   lw=2, markersize=8, label="C++ flat (ij)")
    ax.plot(x, ms_f90, "^-", color=COLORS["f90"],    lw=2, markersize=8, label="Fortran (ji)")

    # Annotate values
    for xi, y in zip(x, ms_cpp):
        if y: ax.annotate(f"{y:.2f}", (xi, y), xytext=(0, 6),
                          textcoords="offset points", ha="center", fontsize=8,
                          color=COLORS["flat"])
    for xi, y in zip(x, ms_f90):
        if y: ax.annotate(f"{y:.2f}", (xi, y), xytext=(0, -14),
                          textcoords="offset points", ha="center", fontsize=8,
                          color=COLORS["f90"])

    # Annotate optimal block size (M3 L1 analysis)
    # Working set per BxB block: (B+2)^2 * 3 arrays * 8 bytes
    for b in [8, 16, 32, 64]:
        ws_kb = (b+2)**2 * 3 * 8 / 1024
        if ws_kb <= L1_KB:
            xi = block_sizes.index(b)
            ax.axvline(xi, color="green", lw=0.8, alpha=0.4, linestyle=":")

    ax.set_xticks(x)
    ax.set_xticklabels(block_labels)
    ax.set_xlabel("Tamaño de bloque B")
    ax.set_ylabel("ms / iteración")
    ax.set_title(f"N = {N}  (malla {N}×{N})")
    ax.legend(loc="upper right")

    # Working set annotation
    row_kb = (N+2) * 8 / 1024
    ax.text(0.02, 0.97, f"1 fila = {row_kb:.1f} KB\nL1 = {L1_KB} KB",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig(OUT / "fig4_blocking_analysis.png", bbox_inches="tight")
plt.close()
print("Saved fig4_blocking_analysis.png")

# =============================================================================
# FIGURE 5 — C++ vs Fortran: todas las variantes, buen orden, con -O2
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Figura 5 — C++ vs Fortran (orden óptimo, sin blocking, -O2)\n"
             "Escalado con N", fontweight="bold")

configs = [
    ("C++ ptr   (ij)", [get_ms(d2, N=n, storage="ptr",    order="ij", block=0) for n in SIZES], COLORS["ptr"],    "o-"),
    ("C++ flat  (ij)", [get_ms(d2, N=n, storage="flat",   order="ij", block=0) for n in SIZES], COLORS["flat"],   "s-"),
    ("C++ vector(ij)", [get_ms(d2, N=n, storage="vector", order="ij", block=0) for n in SIZES], COLORS["vector"], "D-"),
    ("Fortran   (ji)", [get_ms(d2, N=n, lang="fortran",   order="ji", block=0) for n in SIZES], COLORS["f90"],    "^-"),
]

for name, ms_vals, color, marker in configs:
    ax.plot(SIZES, ms_vals, marker, color=color, lw=2, markersize=9, label=name)

# Add N^2 reference line
ref_N = np.array(SIZES, dtype=float)
ref_y = get_ms(d2, N=512, storage="flat", order="ij", block=0) * (ref_N / 512)**2
ax.plot(SIZES, ref_y, "k--", lw=1, alpha=0.5, label="O(N²) referencia")

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(SIZES)
ax.set_xticklabels([str(n) for n in SIZES])
ax.set_xlabel("Tamaño de malla N")
ax.set_ylabel("Tiempo por iteración (ms)")
ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(OUT / "fig5_cpp_vs_fortran.png", bbox_inches="tight")
plt.close()
print("Saved fig5_cpp_vs_fortran.png")

# =============================================================================
# FIGURE 6 — Speedup de -O2 vs sin flags (por variante y N)
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 5))
ax.set_title("Figura 6 — Aceleración por Optimización del Compilador (-O2 vs Sin Flags)\n"
             "Orden bueno para cada variante", fontweight="bold")

configs = [
    ("C++ ptr",    "ptr",    "cpp",    "ij"),
    ("C++ flat",   "flat",   "cpp",    "ij"),
    ("C++ vector", "vector", "cpp",    "ij"),
    ("Fortran",    "f90",    "fortran","ji"),
]

x = np.arange(len(SIZES))
w = 0.22
offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
colors_list = [COLORS["ptr"], COLORS["flat"], COLORS["vector"], COLORS["f90"]]

for i, ((name, stor, lang, order), offset, color) in enumerate(zip(configs, offsets, colors_list)):
    speedups = []
    for n in SIZES:
        if lang == "fortran":
            v0 = get_ms(d0, N=n, lang="fortran", order=order, block=0)
            v2 = get_ms(d2, N=n, lang="fortran", order=order, block=0)
        else:
            v0 = get_ms(d0, N=n, storage=stor, order=order, block=0)
            v2 = get_ms(d2, N=n, storage=stor, order=order, block=0)
        speedups.append(v0 / v2 if (v0 and v2) else 0)

    bars = ax.bar(x + offset, speedups, w, label=name, color=color,
                  edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f"{val:.1f}×", ha="center", va="bottom", fontsize=8,
                rotation=90, fontweight="bold")

ax.axhline(1.0, color="gray", lw=1.2, linestyle="--", label="sin aceleración")
ax.set_xticks(x)
ax.set_xticklabels([f"N={n}" for n in SIZES])
ax.set_ylabel("Factor de aceleración (-O2 / sin flags)")
ax.legend(loc="upper left")
ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
plt.tight_layout()
plt.savefig(OUT / "fig6_compiler_speedup.png", bbox_inches="tight")
plt.close()
print("Saved fig6_compiler_speedup.png")

# =============================================================================
# FIGURE 7 — Modelo Roofline (Apple M3)
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 6))
ax.set_title("Figura 7 — Modelo Roofline — Apple M3 arm64\n"
             "Jacobi 2D Poisson (varios N y configuraciones, -O2)",
             fontweight="bold")

# M3 specs
PEAK_GFLOPS = 4.05 * 4 * 2 * 2 * 2   # GHz × perf_cores × FMA_units × FMA_ops × NEON_doubles
# More conservatively: 4 perf cores × 4.05GHz × 2 NEON FMA/cycle × 2 doubles/FMA = 65 GFLOPS
PEAK_GFLOPS_CONSERVATIVE = 4 * 4.05 * 2 * 2
BW_GBS = 100.0           # ~100 GB/s unified memory bandwidth

# Roofline
ai_range = np.logspace(-2, 3, 1000)
perf_compute = np.minimum(ai_range * BW_GBS, PEAK_GFLOPS_CONSERVATIVE)

ax.loglog(ai_range, perf_compute, "k-", lw=2.5, label=f"Roofline (pico={PEAK_GFLOPS_CONSERVATIVE:.0f} GFLOPS, BW={BW_GBS:.0f} GB/s)")
ax.axvline(PEAK_GFLOPS_CONSERVATIVE / BW_GBS, color="gray", lw=1, linestyle=":",
           label=f"Ridge point = {PEAK_GFLOPS_CONSERVATIVE/BW_GBS:.2f} FLOP/byte")

# Arithmetic intensity of Jacobi stencil: ~8 FLOPs / (8 reads+writes × 8 bytes)
# Theoretical (perfect cache): 8 / (8×8) = 0.125 FLOP/byte
AI_THEORETICAL = 8.0 / (8 * 8)
ax.axvline(AI_THEORETICAL, color="red", lw=1.5, linestyle="--",
           label=f"AI teórico = {AI_THEORETICAL:.3f} FLOP/byte")

# Actual measured points
for n, color, marker in zip(SIZES, ["#2196F3","#4CAF50","#FF9800","#9C27B0"], ["o","s","D","^"]):
    ms = get_ms(d2, N=n, storage="flat", order="ij", block=0)
    if ms:
        t_s = ms / 1000.0
        flops = n * n * 8
        bytes_transferred = n * n * 8 * 8  # 8 doubles accessed per point
        gflops = flops / t_s / 1e9
        ai_actual = flops / bytes_transferred
        ax.plot(ai_actual, gflops, marker, color=color, markersize=12,
                label=f"C++ flat N={n}: {gflops:.2f} GFLOPS", zorder=5)

# Fortran points
for n, marker in zip([2048, 4096], ["v", "P"]):
    ms = get_ms(d2, N=n, lang="fortran", order="ji", block=0)
    if ms:
        t_s = ms / 1000.0
        flops = n * n * 8
        bytes_transferred = n * n * 8 * 8
        gflops = flops / t_s / 1e9
        ai_actual = flops / bytes_transferred
        ax.plot(ai_actual, gflops, marker, color=COLORS["f90"], markersize=12,
                label=f"Fortran N={n}: {gflops:.2f} GFLOPS", zorder=5)

ax.set_xlabel("Intensidad Aritmética (FLOP/byte)")
ax.set_ylabel("Rendimiento (GFLOPS)")
ax.set_xlim(0.01, 1000)
ax.set_ylim(0.1, 300)
ax.legend(fontsize=8, loc="upper left")
ax.text(0.012, 0.15, "Memory-Bound", color="blue", fontsize=10, style="italic")
ax.text(2.5, 0.15, "Compute-Bound", color="green", fontsize=10, style="italic")
plt.tight_layout()
plt.savefig(OUT / "fig7_roofline.png", bbox_inches="tight")
plt.close()
print("Saved fig7_roofline.png")

# =============================================================================
# FIGURE 8 — Tabla de rendimiento como heatmap (ms/iter, -O2)
# Filas = (lang, storage, order), Columnas = N
# =============================================================================
rows_hm = [
    ("C++ ptr   ij",    [get_ms(d2, N=n, storage="ptr",    order="ij", block=0) for n in SIZES]),
    ("C++ ptr   ji",    [get_ms(d2, N=n, storage="ptr",    order="ji", block=0) for n in SIZES]),
    ("C++ flat  ij",    [get_ms(d2, N=n, storage="flat",   order="ij", block=0) for n in SIZES]),
    ("C++ flat  ji",    [get_ms(d2, N=n, storage="flat",   order="ji", block=0) for n in SIZES]),
    ("C++ vector ij",   [get_ms(d2, N=n, storage="vector", order="ij", block=0) for n in SIZES]),
    ("C++ vector ji",   [get_ms(d2, N=n, storage="vector", order="ji", block=0) for n in SIZES]),
    ("Fortran   ji",    [get_ms(d2, N=n, lang="fortran",   order="ji", block=0) for n in SIZES]),
    ("Fortran   ij",    [get_ms(d2, N=n, lang="fortran",   order="ij", block=0) for n in SIZES]),
]

labels_hm = [r[0] for r in rows_hm]
data_hm   = np.array([r[1] for r in rows_hm], dtype=float)

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_title("Figura 8 — Tiempo por Iteración (ms) — Heatmap Comparativo (-O2)\n"
             "Verde=rápido, Rojo=lento", fontweight="bold")

im = ax.imshow(data_hm, aspect="auto", cmap="RdYlGn_r")
ax.set_xticks(range(len(SIZES)))
ax.set_xticklabels([f"N={n}" for n in SIZES])
ax.set_yticks(range(len(labels_hm)))
ax.set_yticklabels(labels_hm, fontsize=10)

# Annotate cells
for i in range(len(labels_hm)):
    for j in range(len(SIZES)):
        val = data_hm[i, j]
        color = "white" if val > np.percentile(data_hm, 75) else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=10, color=color, fontweight="bold")

plt.colorbar(im, ax=ax, label="ms / iteración")
plt.tight_layout()
plt.savefig(OUT / "fig8_heatmap.png", bbox_inches="tight")
plt.close()
print("Saved fig8_heatmap.png")

print("\n=== All figures generated ===")
print(f"Output directory: {OUT}")

# =============================================================================
# Print summary tables for LaTeX
# =============================================================================
print("\n" + "="*70)
print("TABLE 1 — ms/iter sin flags (-O0)")
print("="*70)
print(f"{'Variant':<20}", end="")
for n in SIZES: print(f" N={n:>5}", end="")
print()
for stor, order, lang in [("ptr","ij","cpp"),("ptr","ji","cpp"),
                           ("flat","ij","cpp"),("flat","ji","cpp"),
                           ("vector","ij","cpp"),("vector","ji","cpp"),
                           ("f90","ji","fortran"),("f90","ij","fortran")]:
    if lang == "fortran":
        name = f"Fortran ({order})"
        vals = [get_ms(d0, N=n, lang="fortran", order=order, block=0) for n in SIZES]
    else:
        name = f"C++ {stor} ({order})"
        vals = [get_ms(d0, N=n, storage=stor, order=order, block=0) for n in SIZES]
    print(f"{name:<20}", end="")
    for v in vals: print(f"  {v:>6.2f}", end="") if v else print(f"  {'?':>6}", end="")
    print()

print("\n" + "="*70)
print("TABLE 2 — ms/iter con -O2")
print("="*70)
print(f"{'Variant':<20}", end="")
for n in SIZES: print(f" N={n:>5}", end="")
print()
for stor, order, lang in [("ptr","ij","cpp"),("ptr","ji","cpp"),
                           ("flat","ij","cpp"),("flat","ji","cpp"),
                           ("vector","ij","cpp"),("vector","ji","cpp"),
                           ("f90","ji","fortran"),("f90","ij","fortran")]:
    if lang == "fortran":
        name = f"Fortran ({order})"
        vals = [get_ms(d2, N=n, lang="fortran", order=order, block=0) for n in SIZES]
    else:
        name = f"C++ {stor} ({order})"
        vals = [get_ms(d2, N=n, storage=stor, order=order, block=0) for n in SIZES]
    print(f"{name:<20}", end="")
    for v in vals: print(f"  {v:>6.3f}", end="") if v else print(f"  {'?':>6}", end="")
    print()

print("\n" + "="*70)
print("TABLE 3 — Blocking (C++ flat ij, -O2)")
print("="*70)
print(f"{'Block B':<10}", end="")
for n in SIZES: print(f" N={n:>5}", end="")
print()
for b in [0, 8, 16, 32, 64]:
    name = f"B={b}" if b > 0 else "No tiling"
    print(f"{name:<10}", end="")
    for n in SIZES:
        v = get_ms(d2, N=n, storage="flat", order="ij", block=b)
        print(f"  {v:>6.3f}", end="") if v else print(f"  {'?':>6}", end="")
    print()

print("\n" + "="*70)
print("TABLE 4 — Blocking (Fortran ji, -O2)")
print("="*70)
print(f"{'Block B':<10}", end="")
for n in SIZES: print(f" N={n:>5}", end="")
print()
for b in [0, 8, 16, 32, 64]:
    name = f"B={b}" if b > 0 else "No tiling"
    print(f"{name:<10}", end="")
    for n in SIZES:
        v = get_ms(d2, N=n, lang="fortran", order="ji", block=b)
        print(f"  {v:>6.3f}", end="") if v else print(f"  {'?':>6}", end="")
    print()
