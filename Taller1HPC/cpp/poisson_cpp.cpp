// =============================================================================
// poisson_cpp.cpp — Jacobi solver for 2D Poisson equation
// ∇²φ(x,y) = sin(πx)sin(πy),  φ=0 on ∂Ω,  Ω=[0,1]×[0,1]
//
// Universidad Distrital Francisco José de Caldas — Sistemas Distribuidos 2026-1
//
// Architecture: Apple M3 (arm64)
//   L1D: 128 KB (perf cores),  L2: 16 MB (perf cores)
//   Cache line: 128 bytes = 16 doubles
//   Storage: C++ row-major (contiguous rows)
//
// Usage:
//   ./poisson_cpp  N  STORAGE  ORDER  BLOCK  ITERS
//
//   N       : interior grid size (N×N points), e.g. 512
//   STORAGE : 0 = double** (pointer-to-pointer)
//             1 = new double[N*N] (flat contiguous)
//             2 = std::vector<double> (contiguous)
//   ORDER   : 0 = i-outer / j-inner  (row-major, GOOD for C++ row-major storage)
//             1 = j-outer / i-inner  (col-major, BAD  for C++ row-major storage)
//   BLOCK   : 0 = no tiling;  B>0 = B×B tile size
//   ITERS   : >0 = fixed iterations for benchmarking
//              0 = run until ||phi_new - phi_old||_inf < 1e-6
//
// Output (CSV row, no header):
//   N, storage_name, order_name, block, iterations, time_s, ms_per_iter, max_diff
// =============================================================================

#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
static const double PI    = M_PI;
static const double TOL   = 1.0e-6;

// ─────────────────────────────────────────────────────────────────────────────
// Convenience typedefs
using Clock = std::chrono::high_resolution_clock;
using Sec   = std::chrono::duration<double>;

// =============================================================================
// STORAGE VARIANT 0 — double** (pointer to pointer)
// Each row is a separately allocated array.
// Row i starts at phi[i] (possibly non-contiguous between rows).
// Indexing: phi[i][j]   (natural 2-D syntax)
// =============================================================================
static double run_ptr(int N, int order, int B, int max_iter,
                      int& out_iters, double& out_diff)
{
    const int NP = N + 2;          // total points per dimension (incl. boundary)
    const double h  = 1.0 / (N + 1);
    const double h2 = h * h;

    // Allocate three (NP × NP) arrays via pointer-to-pointer
    auto alloc2d = [&]() -> double** {
        double** a = new double*[NP];
        for (int i = 0; i < NP; ++i) a[i] = new double[NP]();
        return a;
    };
    auto free2d = [&](double** a) {
        for (int i = 0; i < NP; ++i) delete[] a[i];
        delete[] a;
    };

    double** phi     = alloc2d();
    double** phi_new = alloc2d();
    double** f       = alloc2d();

    // Initialise source term f[i][j] = sin(π·i·h)·sin(π·j·h)
    for (int i = 1; i <= N; ++i)
        for (int j = 1; j <= N; ++j)
            f[i][j] = std::sin(PI * i * h) * std::sin(PI * j * h);

    // ── Timed Jacobi iterations ──────────────────────────────────────────────
    auto t0 = Clock::now();

    int iter = 0;
    double md = 1.0;

    while (iter < max_iter) {
        md = 0.0;

        if (B == 0) {
            // ── No tiling ────────────────────────────────────────────────────
            if (order == 0) {
                // i-outer / j-inner (GOOD: sequential j access, row-major)
                for (int i = 1; i <= N; ++i)
                    for (int j = 1; j <= N; ++j) {
                        double v = 0.25 * (phi[i-1][j] + phi[i+1][j]
                                         + phi[i][j-1] + phi[i][j+1]
                                         - h2 * f[i][j]);
                        double d = std::fabs(v - phi[i][j]);
                        phi_new[i][j] = v;
                        if (d > md) md = d;
                    }
            } else {
                // j-outer / i-inner (BAD: stride-N access in i, row-major)
                for (int j = 1; j <= N; ++j)
                    for (int i = 1; i <= N; ++i) {
                        double v = 0.25 * (phi[i-1][j] + phi[i+1][j]
                                         + phi[i][j-1] + phi[i][j+1]
                                         - h2 * f[i][j]);
                        double d = std::fabs(v - phi[i][j]);
                        phi_new[i][j] = v;
                        if (d > md) md = d;
                    }
            }
        } else {
            // ── B×B tiling (always i-inner, j-inner within block) ────────────
            int nb = (N + B - 1) / B;
            for (int bi = 0; bi < nb; ++bi) {
                int i0 = bi * B + 1, i1 = std::min(N, (bi + 1) * B);
                for (int bj = 0; bj < nb; ++bj) {
                    int j0 = bj * B + 1, j1 = std::min(N, (bj + 1) * B);
                    for (int i = i0; i <= i1; ++i)
                        for (int j = j0; j <= j1; ++j) {
                            double v = 0.25 * (phi[i-1][j] + phi[i+1][j]
                                             + phi[i][j-1] + phi[i][j+1]
                                             - h2 * f[i][j]);
                            double d = std::fabs(v - phi[i][j]);
                            phi_new[i][j] = v;
                            if (d > md) md = d;
                        }
                }
            }
        }

        // O(1) pointer swap — no data copying
        std::swap(phi, phi_new);
        ++iter;
        if (max_iter == 0 && md < TOL) break;
    }

    double elapsed = Sec(Clock::now() - t0).count();
    out_iters = iter;
    out_diff  = md;

    free2d(phi); free2d(phi_new); free2d(f);
    return elapsed;
}

// =============================================================================
// STORAGE VARIANT 1 — flat new double[NP*NP]  (contiguous block)
// Row-major layout: phi[i*NP + j]
// Indexing macro: IDX(i,j) = i*NP + j
// =============================================================================
static double run_flat(int N, int order, int B, int max_iter,
                       int& out_iters, double& out_diff)
{
    const int NP = N + 2;
    const double h  = 1.0 / (N + 1);
    const double h2 = h * h;

    double* phi     = new double[NP * NP]();
    double* phi_new = new double[NP * NP]();
    double* f       = new double[NP * NP]();

    // Row-major index
    auto IDX = [NP](int i, int j) -> int { return i * NP + j; };

    for (int i = 1; i <= N; ++i)
        for (int j = 1; j <= N; ++j)
            f[IDX(i,j)] = std::sin(PI * i * h) * std::sin(PI * j * h);

    auto t0 = Clock::now();

    int iter = 0;
    double md = 1.0;

    while (iter < max_iter) {
        md = 0.0;

        if (B == 0) {
            if (order == 0) {
                // i-outer / j-inner (GOOD for row-major)
                for (int i = 1; i <= N; ++i)
                    for (int j = 1; j <= N; ++j) {
                        double v = 0.25 * (phi[IDX(i-1,j)] + phi[IDX(i+1,j)]
                                         + phi[IDX(i,j-1)] + phi[IDX(i,j+1)]
                                         - h2 * f[IDX(i,j)]);
                        double d = std::fabs(v - phi[IDX(i,j)]);
                        phi_new[IDX(i,j)] = v;
                        if (d > md) md = d;
                    }
            } else {
                // j-outer / i-inner (BAD for row-major: stride NP per step)
                for (int j = 1; j <= N; ++j)
                    for (int i = 1; i <= N; ++i) {
                        double v = 0.25 * (phi[IDX(i-1,j)] + phi[IDX(i+1,j)]
                                         + phi[IDX(i,j-1)] + phi[IDX(i,j+1)]
                                         - h2 * f[IDX(i,j)]);
                        double d = std::fabs(v - phi[IDX(i,j)]);
                        phi_new[IDX(i,j)] = v;
                        if (d > md) md = d;
                    }
            }
        } else {
            int nb = (N + B - 1) / B;
            for (int bi = 0; bi < nb; ++bi) {
                int i0 = bi * B + 1, i1 = std::min(N, (bi + 1) * B);
                for (int bj = 0; bj < nb; ++bj) {
                    int j0 = bj * B + 1, j1 = std::min(N, (bj + 1) * B);
                    for (int i = i0; i <= i1; ++i)
                        for (int j = j0; j <= j1; ++j) {
                            double v = 0.25 * (phi[IDX(i-1,j)] + phi[IDX(i+1,j)]
                                             + phi[IDX(i,j-1)] + phi[IDX(i,j+1)]
                                             - h2 * f[IDX(i,j)]);
                            double d = std::fabs(v - phi[IDX(i,j)]);
                            phi_new[IDX(i,j)] = v;
                            if (d > md) md = d;
                        }
                }
            }
        }

        std::swap(phi, phi_new);
        ++iter;
        if (max_iter == 0 && md < TOL) break;
    }

    double elapsed = Sec(Clock::now() - t0).count();
    out_iters = iter;
    out_diff  = md;

    delete[] phi; delete[] phi_new; delete[] f;
    return elapsed;
}

// =============================================================================
// STORAGE VARIANT 2 — std::vector<double>  (contiguous, bounds-checked in debug)
// Same row-major layout as variant 1: phi[i*NP + j]
// std::vector guarantees contiguous storage (C++11 §23.3.6.1)
// =============================================================================
static double run_vector(int N, int order, int B, int max_iter,
                         int& out_iters, double& out_diff)
{
    const int NP = N + 2;
    const double h  = 1.0 / (N + 1);
    const double h2 = h * h;

    std::vector<double> phi(NP * NP, 0.0);
    std::vector<double> phi_new(NP * NP, 0.0);
    std::vector<double> f(NP * NP, 0.0);

    auto IDX = [NP](int i, int j) -> int { return i * NP + j; };

    for (int i = 1; i <= N; ++i)
        for (int j = 1; j <= N; ++j)
            f[IDX(i,j)] = std::sin(PI * i * h) * std::sin(PI * j * h);

    auto t0 = Clock::now();

    int iter = 0;
    double md = 1.0;

    while (iter < max_iter) {
        md = 0.0;

        if (B == 0) {
            if (order == 0) {
                for (int i = 1; i <= N; ++i)
                    for (int j = 1; j <= N; ++j) {
                        double v = 0.25 * (phi[IDX(i-1,j)] + phi[IDX(i+1,j)]
                                         + phi[IDX(i,j-1)] + phi[IDX(i,j+1)]
                                         - h2 * f[IDX(i,j)]);
                        double d = std::fabs(v - phi[IDX(i,j)]);
                        phi_new[IDX(i,j)] = v;
                        if (d > md) md = d;
                    }
            } else {
                for (int j = 1; j <= N; ++j)
                    for (int i = 1; i <= N; ++i) {
                        double v = 0.25 * (phi[IDX(i-1,j)] + phi[IDX(i+1,j)]
                                         + phi[IDX(i,j-1)] + phi[IDX(i,j+1)]
                                         - h2 * f[IDX(i,j)]);
                        double d = std::fabs(v - phi[IDX(i,j)]);
                        phi_new[IDX(i,j)] = v;
                        if (d > md) md = d;
                    }
            }
        } else {
            int nb = (N + B - 1) / B;
            for (int bi = 0; bi < nb; ++bi) {
                int i0 = bi * B + 1, i1 = std::min(N, (bi + 1) * B);
                for (int bj = 0; bj < nb; ++bj) {
                    int j0 = bj * B + 1, j1 = std::min(N, (bj + 1) * B);
                    for (int i = i0; i <= i1; ++i)
                        for (int j = j0; j <= j1; ++j) {
                            double v = 0.25 * (phi[IDX(i-1,j)] + phi[IDX(i+1,j)]
                                             + phi[IDX(i,j-1)] + phi[IDX(i,j+1)]
                                             - h2 * f[IDX(i,j)]);
                            double d = std::fabs(v - phi[IDX(i,j)]);
                            phi_new[IDX(i,j)] = v;
                            if (d > md) md = d;
                        }
                }
            }
        }

        std::swap(phi, phi_new);   // O(1): swaps internal pointers
        ++iter;
        if (max_iter == 0 && md < TOL) break;
    }

    double elapsed = Sec(Clock::now() - t0).count();
    out_iters = iter;
    out_diff  = md;

    return elapsed;
}

// =============================================================================
// VALIDATION — compare numerical solution to analytical φ* = -sin(πx)sin(πy)/(2π²)
// =============================================================================
static void validate(int N)
{
    const int NP = N + 2;
    const double h  = 1.0 / (N + 1);
    const double h2 = h * h;

    std::vector<double> phi(NP * NP, 0.0);
    std::vector<double> phi_new(NP * NP, 0.0);
    std::vector<double> f(NP * NP, 0.0);

    auto IDX = [NP](int i, int j) -> int { return i * NP + j; };

    for (int i = 1; i <= N; ++i)
        for (int j = 1; j <= N; ++j)
            f[IDX(i,j)] = std::sin(PI * i * h) * std::sin(PI * j * h);

    double md = 1.0;
    int iter = 0;

    while (md >= TOL && iter < 2000000) {
        md = 0.0;
        for (int i = 1; i <= N; ++i)
            for (int j = 1; j <= N; ++j) {
                double v = 0.25 * (phi[IDX(i-1,j)] + phi[IDX(i+1,j)]
                                 + phi[IDX(i,j-1)] + phi[IDX(i,j+1)]
                                 - h2 * f[IDX(i,j)]);
                double d = std::fabs(v - phi[IDX(i,j)]);
                phi_new[IDX(i,j)] = v;
                if (d > md) md = d;
            }
        std::swap(phi, phi_new);
        ++iter;
    }

    // Compute error against analytical solution
    double err = 0.0;
    double exact_norm = 0.0;
    for (int i = 1; i <= N; ++i)
        for (int j = 1; j <= N; ++j) {
            double x = i * h, y = j * h;
            double exact = -std::sin(PI * x) * std::sin(PI * y) / (2.0 * PI * PI);
            double e = std::fabs(phi[IDX(i,j)] - exact);
            if (e > err) err = e;
            if (std::fabs(exact) > exact_norm) exact_norm = std::fabs(exact);
        }

    std::cerr << "[VALIDATION] N=" << N
              << "  iters=" << iter
              << "  ||phi_new-phi_old||_inf=" << md
              << "  ||err||_inf=" << err
              << "  rel_err=" << err / exact_norm
              << "  phi_max=" << phi[IDX(N/2+1, N/2+1)]
              << "  exact_max=" << -1.0/(2.0*PI*PI)
              << "\n";
}

// =============================================================================
// MAIN
// =============================================================================
int main(int argc, char* argv[])
{
    // Defaults
    int N        = 512;
    int storage  = 0;   // 0=ptr, 1=flat, 2=vector
    int order    = 0;   // 0=ij (good for C++), 1=ji (bad for C++)
    int B        = 0;   // 0=no tiling
    int iters    = 100; // fixed iteration count; 0=convergence mode

    if (argc > 1) N       = std::atoi(argv[1]);
    if (argc > 2) storage = std::atoi(argv[2]);
    if (argc > 3) order   = std::atoi(argv[3]);
    if (argc > 4) B       = std::atoi(argv[4]);
    if (argc > 5) iters   = std::atoi(argv[5]);

    // Special case: validation mode (N < 0)
    if (N < 0) {
        validate(-N);
        return 0;
    }

    // When iters==0 use convergence mode (very large limit)
    int max_iter = (iters > 0) ? iters : 10000000;

    int    out_iters;
    double out_diff;
    double elapsed;

    switch (storage) {
        case 0: elapsed = run_ptr   (N, order, B, max_iter, out_iters, out_diff); break;
        case 1: elapsed = run_flat  (N, order, B, max_iter, out_iters, out_diff); break;
        case 2: elapsed = run_vector(N, order, B, max_iter, out_iters, out_diff); break;
        default:
            std::cerr << "Invalid storage type " << storage << "\n";
            return 1;
    }

    const char* stor_names[] = {"ptr", "flat", "vector"};
    const char* ord_names[]  = {"ij",  "ji"};

    double ms_per_iter = (out_iters > 0) ? (elapsed / out_iters * 1000.0) : 0.0;

    // CSV output: N,lang,storage,order,block,iters,time_s,ms_per_iter,max_diff
    std::cout << N                    << ","
              << "cpp"                << ","
              << stor_names[storage]  << ","
              << ord_names[order]     << ","
              << B                   << ","
              << out_iters           << ","
              << elapsed             << ","
              << ms_per_iter         << ","
              << out_diff            << "\n";

    return 0;
}
