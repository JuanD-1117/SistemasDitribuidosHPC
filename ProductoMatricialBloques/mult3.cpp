// mult3.cpp — Multiplicación de matrices (técnica de bloques)
// Compilar: g++ -O3 -std=c++11 -o mult3 mult3.cpp
// Ejecutar: ./mult3

#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>

#define N   1024
#define S   64     // Tamaño de bloque

// ──────────────────────────────────────────────────────────────
// Análisis teórico del tamaño de bloque (S)
//
//   Para que los 3 sub-bloques A_ik, B_kj, C_ij quepan en L1:
//     3 * S^2 * sizeof(float) <= L1_cache
//     3 * S^2 * 4             <= 128 * 1024
//     S <= sqrt(131072 / 12) ≈ 104.5
//
//   Mejor potencia de 2 <= 104.5 → S = 64
//     Verificación S=64 : 3 * 64^2  * 4 =  49 152 bytes =  48 KB  ✓ (< 128 KB)
//     Verificación S=128: 3 * 128^2 * 4 = 196 608 bytes = 192 KB  ✗ (> 128 KB)
// ──────────────────────────────────────────────────────────────

float A[N * N], B[N * N], C[N * N];

int main() {
    const int L1_BYTES   = 128 * 1024;
    const int mem_bloque = 3 * S * S * (int)sizeof(float);

    std::cout << "==========================================" << std::endl;
    std::cout << "  Multiplicacion de Matrices (Bloques)" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "N = " << N << ",  S = " << S << std::endl;
    std::cout << "Cache L1 = " << L1_BYTES / 1024 << " KB" << std::endl;
    std::cout << "Memoria por bloque: 3 * " << S << "^2 * 4 = "
              << mem_bloque << " bytes = " << mem_bloque / 1024.0 << " KB" << std::endl;
    std::cout << "Cabe en L1: " << (mem_bloque <= L1_BYTES ? "SI ✓" : "NO ✗") << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    // Inicializar matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)(i % 100) / 100.0f;
        B[i] = (float)((i + 1) % 100) / 100.0f;
    }
    memset(C, 0, sizeof(C));

    auto t_inicio = std::chrono::high_resolution_clock::now();

    // Triple loop externo: recorre bloques (I, J, K)
    // Triple loop interno: producto dentro del bloque S×S
    // (igual que la imagen 3)
    for (int l = 0; l < N / S; ++l) {
        for (int J = 0; J < N / S; ++J) {
            for (int K = 0; K < N / S; ++K) {
                for (int i = 0; i < S; ++i) {
                    for (int j = 0; j < S; ++j) {
                        float sum = 0;
                        for (int k = 0; k < S; ++k) {
                            sum += A[(l * S + i) * N + (K * S + k)]
                                 * B[(K * S + k) * N + (J * S + j)];
                        }
                        C[(l * S + i) * N + (J * S + j)] += sum;
                    }
                }
            }
        }
    }

    auto t_fin = std::chrono::high_resolution_clock::now();
    double t      = std::chrono::duration<double>(t_fin - t_inicio).count();
    double gflops = 2.0 * N * N * N / t / 1e9;

    std::cout << "Tiempo:  " << t      << " s"      << std::endl;
    std::cout << "GFLOPS:  " << gflops << std::endl;
    std::cout << "C[0] = "   << C[0]   << " (verificacion)" << std::endl;

    return 0;
}
