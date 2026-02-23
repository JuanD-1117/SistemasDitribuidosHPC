// benchmark.cpp — Comparativa completa: Naive vs Bloques
// Prueba múltiples tamaños N y tamaños de bloque S
// Genera results.csv para graficar con plot.py
//
// Compilar: g++ -O3 -std=c++11 -o benchmark benchmark.cpp
// Ejecutar: ./benchmark

#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

// ============================================================
// MULTIPLICACION NAIVE
// C[i][j] = sum_k  A[i][k] * B[k][j]
// ============================================================
void matmul_naive(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

// ============================================================
// MULTIPLICACION POR BLOQUES
// Sub-matrices de tamaño S×S
// ============================================================
void matmul_block(const float* A, const float* B, float* C, int N, int S) {
    memset(C, 0, (size_t)N * N * sizeof(float));
    for (int l = 0; l < N / S; ++l)
        for (int J = 0; J < N / S; ++J)
            for (int K = 0; K < N / S; ++K)
                for (int i = 0; i < S; ++i)
                    for (int j = 0; j < S; ++j) {
                        float sum = 0.0f;
                        for (int k = 0; k < S; ++k)
                            sum += A[(l*S+i)*N + (K*S+k)]
                                 * B[(K*S+k)*N + (J*S+j)];
                        C[(l*S+i)*N + (J*S+j)] += sum;
                    }
}

int main() {
    const int L1_BYTES   = 128 * 1024;      // 128 KB cache L1
    const int FLOAT_BYTES = (int)sizeof(float);
    double s_theory = sqrt((double)L1_BYTES / (3.0 * FLOAT_BYTES));

    // ── Cabecera con análisis teórico ──────────────────────────
    printf("=====================================================\n");
    printf("   Benchmark: Multiplicacion de Matrices N×N\n");
    printf("=====================================================\n");
    printf("Cache L1 = %d KB = %d bytes\n", L1_BYTES / 1024, L1_BYTES);
    printf("\n[Analisis teorico del tamano optimo de bloque S]\n");
    printf("  Condicion: 3 * s^2 * sizeof(float) <= L1\n");
    printf("  => s <= sqrt(L1 / (3 * sizeof(float)))\n");
    printf("  => s <= sqrt(%d / %d)  = %.2f\n",
           L1_BYTES, 3 * FLOAT_BYTES, s_theory);
    printf("  => s_teorico = %d  (parte entera)\n\n", (int)s_theory);

    int test_s[] = {8, 16, 32, 64, 128};
    for (int s : test_s) {
        int mem = 3 * s * s * FLOAT_BYTES;
        printf("  S=%3d:  3*%3d^2*4 = %6d bytes = %5.1f KB  %s\n",
               s, s, mem, mem / 1024.0,
               mem <= L1_BYTES ? "[CABE en L1 ✓]" : "[NO CABE    ✗]");
    }
    printf("\n[ATENCION: formula corregida para almacenamiento row-major]\n");
    printf("  Los bloques NO son contiguos en memoria (stride = N floats).\n");
    printf("  La huella real en cache de 3 bloques S×S es ~3*S*N*4 bytes.\n");
    printf("  Condicion corregida: 3 * S * N * sizeof(float) <= L1\n");
    printf("  => S <= L1 / (3 * N * sizeof(float))\n");
    int sizes_preview[] = {256, 512, 1024};
    for (int n : sizes_preview) {
        double s_corr = (double)L1_BYTES / (3.0 * n * FLOAT_BYTES);
        printf("  N=%4d => S_optimo ~%.1f (pot.2: %d)\n",
               n, s_corr,
               (int)s_corr >= 32 ? 32 : (int)s_corr >= 16 ? 16 : 8);
    }
    printf("=====================================================\n\n");

    // ── Tamaños y bloques a probar ─────────────────────────────
    int sizes[]      = {256, 512, 1024};
    int block_sizes[] = {8, 16, 32, 64};

    ofstream csv("results.csv");
    csv << "N,method,S,time_sec,gflops\n";

    for (int N : sizes) {
        double mat_mb = (double)N * N * FLOAT_BYTES / (1024.0 * 1024.0);
        printf("--- N = %d  (cada matriz: %.2f MB) ---\n", N, mat_mb);

        // Alocación e inicialización
        vector<float> A(N * N), B(N * N), C(N * N);
        srand(42);
        for (int i = 0; i < N * N; i++) {
            A[i] = (float)rand() / RAND_MAX;
            B[i] = (float)rand() / RAND_MAX;
        }

        // Repeticiones: más para matrices pequeñas
        int reps   = (N <= 256) ? 5 : (N <= 512) ? 3 : 1;
        double flops = 2.0 * N * N * N;

        // ── NAIVE ─────────────────────────────────────────────
        matmul_naive(A.data(), B.data(), C.data(), N); // warm-up

        auto t0 = high_resolution_clock::now();
        for (int r = 0; r < reps; r++)
            matmul_naive(A.data(), B.data(), C.data(), N);
        double t_naive = duration<double>(high_resolution_clock::now() - t0).count() / reps;
        double gf_naive = flops / t_naive / 1e9;

        printf("  Naive:         %9.4f s  | %7.4f GFLOPS\n", t_naive, gf_naive);
        csv << N << ",naive,0," << fixed << setprecision(6)
            << t_naive << "," << gf_naive << "\n";

        // ── BLOQUES con distintos S ────────────────────────────
        for (int S : block_sizes) {
            if (N % S != 0) continue;

            int  mem_bytes = 3 * S * S * FLOAT_BYTES;
            bool fits      = mem_bytes <= L1_BYTES;

            matmul_block(A.data(), B.data(), C.data(), N, S); // warm-up

            auto tb0 = high_resolution_clock::now();
            for (int r = 0; r < reps; r++)
                matmul_block(A.data(), B.data(), C.data(), N, S);
            double t_block = duration<double>(high_resolution_clock::now() - tb0).count() / reps;
            double gf_block = flops / t_block / 1e9;
            double speedup  = t_naive / t_block;

            printf("  Block S=%3d:   %9.4f s  | %7.4f GFLOPS  | speedup=%5.2fx  [%s L1]\n",
                   S, t_block, gf_block, speedup,
                   fits ? "cabe en" : "excede ");

            csv << N << ",block," << S << ","
                << t_block << "," << gf_block << "\n";
        }
        printf("\n");
    }

    csv.close();
    printf("Resultados guardados en: results.csv\n");
    return 0;
}
