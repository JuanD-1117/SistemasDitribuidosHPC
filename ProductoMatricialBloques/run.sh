#!/bin/bash
# run.sh — Compila, ejecuta el benchmark y genera gráficas
# Uso: bash run.sh

set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "=================================================="
echo "  Benchmark: Multiplicacion de Matrices"
echo "  Cache L1 = 128 KB"
echo "=================================================="
echo ""

# ── Compilar ──────────────────────────────────────────────────
echo ">>> [1/4] Compilando con -O3..."
g++ -O3 -std=c++11 -o mult1     mult1.cpp     && echo "    mult1     OK"
g++ -O3 -std=c++11 -o mult3     mult3.cpp     && echo "    mult3     OK"
g++ -O3 -std=c++11 -o benchmark benchmark.cpp && echo "    benchmark OK"

# ── Standalone: mult1 ─────────────────────────────────────────
echo ""
echo ">>> [2/4] Ejecutando mult1 (Naive, N=1024)..."
echo "    -----------------------------------------"
./mult1

# ── Standalone: mult3 ─────────────────────────────────────────
echo ""
echo ">>> [3/4] Ejecutando mult3 (Bloques, N=1024, S=64)..."
echo "    -----------------------------------------"
./mult3

# ── Benchmark completo ────────────────────────────────────────
echo ""
echo ">>> [4/4] Ejecutando benchmark completo (N=256,512,1024)..."
echo "    (puede tardar ~1-3 min para N=1024 naive)"
echo "    -----------------------------------------"
./benchmark

# ── Gráficas ──────────────────────────────────────────────────
echo ""
echo ">>> Generando gráficas con Python..."
python3 plot.py

echo ""
echo "=================================================="
echo "  Completado!"
echo "  Gráfica: benchmark_results.png"
echo "  Datos:   results.csv"
echo "=================================================="
