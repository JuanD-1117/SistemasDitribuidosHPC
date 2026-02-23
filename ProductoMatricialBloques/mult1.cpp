// mult1.cpp — Multiplicación de matrices (implementación naive / "ingenua")
// Compilar: g++ -O3 -std=c++11 -o mult1 mult1.cpp
// Ejecutar: ./mult1

#include <iostream>
#include <chrono>

#define N 1024

float A[N * N], B[N * N], C[N * N];

int main() {
    // Inicializar matrices con valores de prueba
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)(i % 100) / 100.0f;
        B[i] = (float)((i + 1) % 100) / 100.0f;
        C[i] = 0.0f;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "  Multiplicacion de Matrices (Naive)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "N = " << N << "  (matriz " << N << "x" << N << ")" << std::endl;

    auto t_inicio = std::chrono::high_resolution_clock::now();

    // Triple loop naive  (igual que la imagen 1)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    auto t_fin = std::chrono::high_resolution_clock::now();
    double t = std::chrono::duration<double>(t_fin - t_inicio).count();
    double gflops = 2.0 * N * N * N / t / 1e9;

    std::cout << "Tiempo:  " << t      << " s"       << std::endl;
    std::cout << "GFLOPS:  " << gflops << std::endl;
    std::cout << "C[0] = "   << C[0]   << " (verificacion)" << std::endl;

    return 0;
}
