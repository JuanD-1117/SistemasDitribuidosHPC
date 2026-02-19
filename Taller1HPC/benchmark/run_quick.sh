#!/usr/bin/env bash
# =============================================================================
# run_quick.sh — Quick benchmark for testing (fewer sizes/blocks)
# Runs in ~2-5 minutes on Apple M3
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CPP="${ROOT_DIR}/cpp/poisson_cpp"
F90="${ROOT_DIR}/fortran/poisson_f90"
OUT="${SCRIPT_DIR}/results/results_all.csv"

if [[ ! -x "$CPP" ]] || [[ ! -x "$F90" ]]; then
    echo "ERROR: Run 'make all' first."
    exit 1
fi

mkdir -p "${SCRIPT_DIR}/results"
echo "N,lang,storage,order,block,iters,time_s,ms_per_iter,max_diff" > "$OUT"

# Quick iteration counts (more iters for stable timing)
declare -A ITERS=([512]=200 [1024]=80 [2048]=20 [4096]=5)

SIZES=(512 1024 2048 4096)
BLOCKS=(0 8 16 32 64)

echo "=== Quick Benchmark — Apple M3 arm64 ==="
echo ""

for N in "${SIZES[@]}"; do
    IT=${ITERS[$N]}
    echo "── N = $N (${IT} iterations) ──────────────────────────────"

    # C++ storage variants, both loop orders, no tiling
    for S in 0 1 2; do
        for O in 0 1; do
            printf "  C++ storage=%d order=%d block=0  ... " "$S" "$O"
            r=$("$CPP" "$N" "$S" "$O" 0 "$IT" 2>/dev/null)
            echo "$r" >> "$OUT"
            ms=$(echo "$r" | awk -F',' '{printf "%.4f", $8}')
            echo "${ms} ms/iter"
        done
    done

    # C++ flat (best C++ storage), all block sizes
    for B in "${BLOCKS[@]}"; do
        [[ "$B" == "0" ]] && continue   # already done above
        printf "  C++ flat   order=0 block=%-2d ... " "$B"
        r=$("$CPP" "$N" 1 0 "$B" "$IT" 2>/dev/null)
        echo "$r" >> "$OUT"
        ms=$(echo "$r" | awk -F',' '{printf "%.4f", $8}')
        echo "${ms} ms/iter"
    done

    # Fortran, both orders, no tiling + all blocks
    for O in 0 1; do
        printf "  F90        order=%d block=0  ... " "$O"
        r=$("$F90" "$N" "$O" 0 "$IT" 2>/dev/null)
        echo "$r" >> "$OUT"
        ms=$(echo "$r" | awk -F',' '{printf "%.4f", $8}')
        echo "${ms} ms/iter"
    done

    for B in "${BLOCKS[@]}"; do
        [[ "$B" == "0" ]] && continue
        printf "  F90        order=0 block=%-2d ... " "$B"
        r=$("$F90" "$N" 0 "$B" "$IT" 2>/dev/null)
        echo "$r" >> "$OUT"
        ms=$(echo "$r" | awk -F',' '{printf "%.4f", $8}')
        echo "${ms} ms/iter"
    done

    echo ""
done

echo "Results saved to: ${OUT}"
echo ""

# Print summary table
echo "=== Performance Summary ==="
echo "N    | lang    | storage | order | block | ms/iter"
echo "-----|---------|---------|-------|-------|--------"
awk -F',' 'NR>1 {
    printf "%-4s | %-7s | %-7s | %-5s | %-5s | %.4f\n",
           $1, $2, $3, $4, $5, $8
}' "$OUT" | sort -t'|' -k1 -n -k7 -n
