# Producto Matricial Bajo Técnica de Bloques

**Sistemas Distribuidos 2026 — Universidad Distrital**

Comparativa de rendimiento entre multiplicación de matrices *naive* y con técnica de bloques (*cache blocking*), con énfasis en la optimización de la jerarquía de caché L1.

## Descripción

| Archivo | Descripción |
|---|---|
| `mult1.cpp` | Implementación naive (triple loop), N=1024 |
| `mult3.cpp` | Implementación por bloques S=64, N=1024 |
| `benchmark.cpp` | Benchmark completo: N={256,512,1024}, S={8,16,32,64} |
| `plot.py` | Gráficas comparativas con matplotlib |
| `run.sh` | Script de compilación y ejecución completa |
| `results.csv` | Datos medidos (generado por benchmark) |
| `benchmark_results.png` | Gráficas comparativas |
| `informe.tex` | Informe LaTeX completo (Overleaf) |

## Compilar y ejecutar

```bash
bash run.sh
```

O paso a paso:
```bash
g++ -O3 -std=c++11 -o mult1     mult1.cpp
g++ -O3 -std=c++11 -o mult3     mult3.cpp
g++ -O3 -std=c++11 -o benchmark benchmark.cpp

./mult1        # Naive  N=1024
./mult3        # Bloques N=1024, S=64
./benchmark    # Benchmark completo → results.csv
python3 plot.py
```

## Resultados (macOS, L1 = 128 KB)

| N | Naive | Bloques (S óptimo) | Speedup |
|---|---|---|---|
| 256×256   | 0.021 s | 0.0095 s (S=32) | **2.21×** |
| 512×512   | 0.196 s | 0.076 s  (S=32) | **2.57×** |
| 1024×1024 | 1.746 s | 0.595 s  (S=16) | **2.93×** |

## Hallazgo clave: fórmula corregida para el tamaño de bloque

La fórmula clásica `3·s²·4 ≤ L1` predice S=64 como óptimo, pero ignora que los bloques en formato *row-major* no son contiguos en memoria. La huella real es `3·s·N·4 bytes`, lo que da la condición corregida:

```
s_optimo ≤ L1 / (3 · N · sizeof(float))
```

| N    | s teórico clásico | s corregido | S empírico óptimo |
|------|---|---|---|
| 256  | 104.5 | 42.7 | **32** ✓ |
| 512  | 104.5 | 21.3 | **32** ✓ |
| 1024 | 104.5 | 10.7 | **16** ✓ |

## Dependencias

- C++11 (`g++` / `clang++`)
- Python 3 + `numpy`, `pandas`, `matplotlib`
