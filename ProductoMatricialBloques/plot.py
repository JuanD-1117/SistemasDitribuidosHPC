#!/usr/bin/env python3
"""
plot.py — Gráficas comparativas: Naive vs Técnica de Bloques
Requiere: results.csv generado por ./benchmark

Instalar dependencias: pip install pandas matplotlib numpy
Ejecutar:              python3 plot.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ── Verificar datos ────────────────────────────────────────────
if not os.path.exists("results.csv"):
    print("ERROR: No se encontró results.csv")
    print("Ejecutar primero: ./benchmark")
    sys.exit(1)

df = pd.read_csv("results.csv")
print("Datos cargados:")
print(df.to_string())

# ── Constantes ────────────────────────────────────────────────
L1_KB       = 128
L1_BYTES    = L1_KB * 1024
s_theory    = np.sqrt(L1_BYTES / (3 * 4))          # 104.5
s_best_pow2 = 64                                    # mayor pot.2 <= s_theory

# ── Separar métodos ───────────────────────────────────────────
naive = df[df["method"] == "naive"].sort_values("N").reset_index(drop=True)
block = df[df["method"] == "block"].sort_values(["N", "S"]).reset_index(drop=True)

# Mejor bloque por N (mínimo tiempo)
best_idx   = block.groupby("N")["time_sec"].idxmin()
best_block = block.loc[best_idx].reset_index(drop=True)

# Tabla comparativa
comp = naive[["N", "time_sec", "gflops"]].copy()
comp.columns = ["N", "t_naive", "gf_naive"]
comp = comp.merge(
    best_block[["N", "S", "time_sec", "gflops"]],
    on="N"
).rename(columns={"time_sec": "t_block", "gflops": "gf_block", "S": "S_best"})
comp["speedup"] = comp["t_naive"] / comp["t_block"]

print("\nResumen:")
print(comp[["N", "t_naive", "t_block", "S_best", "speedup", "gf_naive", "gf_block"]].to_string(index=False))

sizes = sorted(df["N"].unique())

# ── Paleta ────────────────────────────────────────────────────
C_NAIVE  = "#E74C3C"
C_BLOCK  = "#27AE60"
C_THEORY = "#2980B9"
C_P2     = "#E67E22"

# ══════════════════════════════════════════════════════════════
# FIGURA PRINCIPAL  (2×2)
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 13))
fig.patch.set_facecolor("white")
fig.suptitle(
    "Multiplicación de Matrices: Naive vs Técnica de Bloques\n"
    f"Cache L1 = {L1_KB} KB  |  s teórico = {s_theory:.1f}  |  "
    f"Mejor pot.2 = S={s_best_pow2}  |  Mac (darwin)",
    fontsize=14, fontweight="bold", y=0.98,
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.38)

x     = np.arange(len(sizes))
width = 0.35

# ── Helper: etiquetas sobre barras ────────────────────────────
def add_labels(ax, bars, color, fmt=".3f", suffix="s"):
    for bar in bars:
        h = bar.get_height()
        if h == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.04,          # 4 % por encima (válido en escala log)
            f"{h:{fmt}}{suffix}",
            ha="center", va="bottom", fontsize=8,
            fontweight="bold", color=color,
        )

# ────────────────────────────────────────────────────────────
# GRÁFICA 1 — Tiempo vs N  (escala log)
# ────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])

b1 = ax1.bar(x - width/2, comp["t_naive"], width,
             label="Naive", color=C_NAIVE, alpha=0.87,
             edgecolor="black", linewidth=0.7, zorder=3)
b2 = ax1.bar(x + width/2, comp["t_block"], width,
             label="Bloques (S óptimo)", color=C_BLOCK, alpha=0.87,
             edgecolor="black", linewidth=0.7, zorder=3)

add_labels(ax1, b1, C_NAIVE)
add_labels(ax1, b2, "#1a6b3a")

ax1.set_yscale("log")
ax1.set_xlabel("Tamaño de Matriz N×N", fontsize=11)
ax1.set_ylabel("Tiempo (s) [escala log]", fontsize=10)
ax1.set_title("Tiempo de Ejecución", fontsize=13, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels([f"{n}×{n}" for n in sizes], fontsize=10)
ax1.legend(fontsize=10)
ax1.grid(axis="y", alpha=0.35, zorder=0)
ax1.set_axisbelow(True)

# ────────────────────────────────────────────────────────────
# GRÁFICA 2 — Speedup vs N
# ────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])

bars_sp = ax2.bar(x, comp["speedup"], color=C_BLOCK, alpha=0.87,
                  edgecolor="black", linewidth=0.7, zorder=3)

for bar, (_, row) in zip(bars_sp, comp.iterrows()):
    h = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        h + 0.04 * comp["speedup"].max(),
        f"{h:.2f}×\n(S={int(row['S_best'])})",
        ha="center", va="bottom", fontsize=9,
        fontweight="bold", color="#1a6b3a",
    )

ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5,
            label="Sin mejora (1×)", zorder=2)
ax2.set_xlabel("Tamaño de Matriz N×N", fontsize=11)
ax2.set_ylabel("Speedup = t_naive / t_block", fontsize=10)
ax2.set_title("Factor de Aceleración (Speedup)", fontsize=13, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels([f"{n}×{n}" for n in sizes], fontsize=10)
ax2.legend(fontsize=10)
ax2.grid(axis="y", alpha=0.35, zorder=0)
ax2.set_axisbelow(True)

# ────────────────────────────────────────────────────────────
# GRÁFICA 3 — Tiempo vs tamaño de bloque S
# ────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])

palette = ["#8E44AD", "#E67E22", "#16A085"]
markers = ["o", "s", "^"]

for idx, N in enumerate(sizes):
    data_N = block[block["N"] == N].sort_values("S")
    if not data_N.empty:
        ax3.plot(data_N["S"], data_N["time_sec"],
                 marker=markers[idx % 3], color=palette[idx % 3],
                 linewidth=2.5, markersize=9,
                 label=f"N={N}", zorder=3)
        # Marcar mínimo
        min_row = data_N.loc[data_N["time_sec"].idxmin()]
        ax3.scatter(min_row["S"], min_row["time_sec"],
                    s=120, color=palette[idx % 3], marker="*",
                    zorder=5)

# Líneas de referencia
ax3.axvline(x=s_theory, color=C_THEORY, linestyle="--", linewidth=2.0,
            label=f"s clásico = {s_theory:.0f}  (3s²·4≤L1)", zorder=4)
ax3.axvline(x=s_best_pow2, color=C_P2, linestyle=":", linewidth=2.0,
            label=f"S={s_best_pow2} (mejor pot.2 clásica)", zorder=4)

# Líneas corregidas para cada N
linestyles_corr = [(0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]
for idx, N_ref in enumerate(sizes):
    s_corr = L1_BYTES / (3 * N_ref * 4)
    ax3.axvline(x=s_corr, color=palette[idx % 3],
                linestyle=linestyles_corr[idx % 3], linewidth=1.6,
                label=f"s corr. N={N_ref} = {s_corr:.1f}", zorder=4)

# Región "fuera de L1 clásica"
ax3.axvspan(s_theory, block["S"].max() + 10, alpha=0.06,
            color="red", label="Excede L1 (fórmula clásica)", zorder=0)

ax3.set_xlabel("Tamaño de Bloque S", fontsize=11)
ax3.set_ylabel("Tiempo de Ejecución (s)", fontsize=10)
ax3.set_title("Efecto del Tamaño de Bloque", fontsize=13, fontweight="bold")
ax3.set_xticks(sorted(block["S"].unique()))
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3, zorder=0)
ax3.set_axisbelow(True)

# ────────────────────────────────────────────────────────────
# GRÁFICA 4 — GFLOPS + análisis teórico
# ────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])

b3 = ax4.bar(x - width/2, comp["gf_naive"], width,
             label="Naive", color=C_NAIVE, alpha=0.87,
             edgecolor="black", linewidth=0.7, zorder=3)
b4 = ax4.bar(x + width/2, comp["gf_block"], width,
             label="Bloques (S óptimo)", color=C_BLOCK, alpha=0.87,
             edgecolor="black", linewidth=0.7, zorder=3)

add_labels(ax4, b3, C_NAIVE,   fmt=".3f", suffix="")
add_labels(ax4, b4, "#1a6b3a", fmt=".3f", suffix="")

ax4.set_xlabel("Tamaño de Matriz N×N", fontsize=11)
ax4.set_ylabel("GFLOPS  (2·N³ / tiempo)", fontsize=10)
ax4.set_title("Rendimiento (GFLOPS)", fontsize=13, fontweight="bold")
ax4.set_xticks(x)
ax4.set_xticklabels([f"{n}×{n}" for n in sizes], fontsize=10)
ax4.legend(fontsize=10)
ax4.grid(axis="y", alpha=0.35, zorder=0)
ax4.set_axisbelow(True)

# Cuadro con análisis: fórmula clásica vs corregida
lines = [
    "─── Fórmula clásica (bloques contiguos) ───",
    f"  3·s²·4 ≤ L1  →  s ≤ √({L1_BYTES}/12)",
    f"  s_clásico = {s_theory:.1f}  (pot.2: S=64)",
    "",
    "─── Fórmula corregida (row-major) ────────",
    "  Huella real ≈ 3·s·N·4 bytes",
    "  (bloques NO contiguos; stride = N)",
    "  3·s·N·4 ≤ L1  →  s ≤ L1/(3·N·4)",
]
for N_ref in sorted(df["N"].unique()):
    s_corr = L1_BYTES / (3 * N_ref * 4)
    emp_best = comp.loc[comp["N"] == N_ref, "S_best"].values
    s_emp = int(emp_best[0]) if len(emp_best) else "?"
    lines.append(f"  N={N_ref:4d}: s≤{s_corr:5.1f}  empírico S={s_emp}")
lines += ["", "  → El S empírico coincide con la", "    fórmula CORREGIDA, no con la clásica."]

textstr = "\n".join(lines)
ax4.text(0.97, 0.97, textstr, transform=ax4.transAxes,
         fontsize=7.8, verticalalignment="top", horizontalalignment="right",
         fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFFDE7",
                   alpha=0.95, edgecolor="#BDBDBD", linewidth=1))

# ── Guardar ───────────────────────────────────────────────────
out_png = "benchmark_results.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nGráfica guardada: {out_png}")
plt.show()
