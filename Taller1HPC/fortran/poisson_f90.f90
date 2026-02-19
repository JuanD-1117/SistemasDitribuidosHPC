! =============================================================================
! poisson_f90.f90 — Jacobi solver for 2D Poisson equation
! ∇²φ(x,y) = sin(πx)sin(πy),  φ=0 on ∂Ω,  Ω=[0,1]×[0,1]
!
! Universidad Distrital Francisco José de Caldas — Sistemas Distribuidos 2026-1
!
! Architecture: Apple M3 (arm64)
!   L1D: 128 KB (perf cores),  L2: 16 MB (perf cores)
!   Cache line: 128 bytes = 16 doubles
!   Storage: Fortran COLUMN-MAJOR (contiguous columns)
!     phi(i,j) and phi(i+1,j) are adjacent in memory (same column)
!     phi(i,j) and phi(i,j+1) are N+2 elements apart (different column)
!
! Usage:
!   ./poisson_f90  N  ORDER  BLOCK  ITERS
!
!   N     : interior grid size (N×N interior points)
!   ORDER : 0 = j-outer / i-inner  (column-major, GOOD for Fortran)
!           1 = i-outer / j-inner  (row-major,    BAD  for Fortran)
!   BLOCK : 0 = no tiling;  B>0 = B×B tile size
!   ITERS : >0 = fixed iterations for benchmarking
!            0 = converge until ||phi_new - phi_old||_inf < 1e-6
!
! Output (CSV row, no header):
!   N, lang, storage, order, block, iterations, time_s, ms_per_iter, max_diff
! =============================================================================

program poisson_jacobi
    use, intrinsic :: iso_fortran_env, only: real64, int64
    implicit none

    ! ── Parameters ─────────────────────────────────────────────────────────────
    real(real64), parameter :: PI  = 4.0_real64 * atan(1.0_real64)
    real(real64), parameter :: TOL = 1.0e-6_real64

    ! ── Variables ──────────────────────────────────────────────────────────────
    integer :: N, loop_order, B, fixed_iters, max_iter
    integer :: i, j, bi, bj, i0, i1, j0, j1, nb, iter

    ! Pointer arrays for O(1) swap (no data copying)
    ! In Fortran POINTER arrays:  phi(i,j) stored column-major
    !   → phi(1,1), phi(2,1), phi(3,1), ... phi(N+2,1), phi(1,2), ...
    real(real64), pointer     :: phi(:,:), phi_new(:,:), f_arr(:,:), tmp_ptr(:,:)

    real(real64) :: h, h2, max_diff, d, val_new
    real(real64) :: elapsed, ms_per_iter

    integer(int64) :: t_start, t_end, t_rate
    character(len=64) :: arg

    ! ── Parse command-line arguments ────────────────────────────────────────────
    N           = 512
    loop_order  = 0
    B           = 0
    fixed_iters = 100

    if (command_argument_count() >= 1) then
        call get_command_argument(1, arg); read(arg,*) N
    end if
    if (command_argument_count() >= 2) then
        call get_command_argument(2, arg); read(arg,*) loop_order
    end if
    if (command_argument_count() >= 3) then
        call get_command_argument(3, arg); read(arg,*) B
    end if
    if (command_argument_count() >= 4) then
        call get_command_argument(4, arg); read(arg,*) fixed_iters
    end if

    ! ── Allocate arrays  (index 1..N+2, boundary at 1 and N+2) ─────────────────
    ! Interior points: i,j = 2..N+1  →  x_i = (i-1)*h,  y_j = (j-1)*h
    allocate(phi    (N+2, N+2))
    allocate(phi_new(N+2, N+2))
    allocate(f_arr  (N+2, N+2))

    phi     = 0.0_real64
    phi_new = 0.0_real64
    f_arr   = 0.0_real64

    h  = 1.0_real64 / (N + 1)
    h2 = h * h

    ! ── Initialise source term ──────────────────────────────────────────────────
    ! f(i,j) = sin(π*(i-1)*h) * sin(π*(j-1)*h)
    ! Interior: i,j = 2..N+1  →  (i-1)*h = h,2h,...,Nh  =  x_1,...,x_N
    !
    ! NOTE: In Fortran column-major, this double loop
    !   do j=2,N+1; do i=2,N+1; ... end do; end do
    ! is the EFFICIENT order (sequential memory access for inner i-loop)
    do j = 2, N+1
        do i = 2, N+1
            f_arr(i,j) = sin(PI * real(i-1, real64) * h) &
                       * sin(PI * real(j-1, real64) * h)
        end do
    end do

    max_iter = max(1, fixed_iters)
    if (fixed_iters == 0) max_iter = 10000000

    ! ── Timed Jacobi iterations ─────────────────────────────────────────────────
    call system_clock(t_start, t_rate)

    iter     = 0
    max_diff = 1.0_real64

    do while (iter < max_iter)
        max_diff = 0.0_real64

        if (B == 0) then
            ! ── No tiling ──────────────────────────────────────────────────────
            if (loop_order == 0) then
                ! j-outer / i-inner (GOOD for Fortran column-major)
                ! phi(i-1,j), phi(i,j), phi(i+1,j) are CONTIGUOUS in column
                do j = 2, N+1
                    do i = 2, N+1
                        val_new = 0.25_real64 * ( phi(i-1,j) + phi(i+1,j) &
                                                + phi(i,j-1) + phi(i,j+1) &
                                                - h2 * f_arr(i,j) )
                        d = abs(val_new - phi(i,j))
                        phi_new(i,j) = val_new
                        if (d > max_diff) max_diff = d
                    end do
                end do
            else
                ! i-outer / j-inner (BAD for Fortran column-major)
                ! phi(i,j-1), phi(i,j), phi(i,j+1): stride N+2 per step → cache misses
                do i = 2, N+1
                    do j = 2, N+1
                        val_new = 0.25_real64 * ( phi(i-1,j) + phi(i+1,j) &
                                                + phi(i,j-1) + phi(i,j+1) &
                                                - h2 * f_arr(i,j) )
                        d = abs(val_new - phi(i,j))
                        phi_new(i,j) = val_new
                        if (d > max_diff) max_diff = d
                    end do
                end do
            end if

        else
            ! ── B×B tiling ─────────────────────────────────────────────────────
            ! For Fortran column-major: tile with j-outer, i-inner (good order)
            ! Tile (bi,bj) processes phi(i0..i1, j0..j1)
            nb = (N + B - 1) / B
            do bj = 0, nb-1
                j0 = bj * B + 2;  j1 = min(N+1, (bj+1)*B + 1)
                do bi = 0, nb-1
                    i0 = bi * B + 2;  i1 = min(N+1, (bi+1)*B + 1)
                    do j = j0, j1
                        do i = i0, i1
                            val_new = 0.25_real64 * ( phi(i-1,j) + phi(i+1,j) &
                                                    + phi(i,j-1) + phi(i,j+1) &
                                                    - h2 * f_arr(i,j) )
                            d = abs(val_new - phi(i,j))
                            phi_new(i,j) = val_new
                            if (d > max_diff) max_diff = d
                        end do
                    end do
                end do
            end do
        end if

        ! O(1) pointer swap — no data copying (critical for performance parity with C++)
        tmp_ptr => phi
        phi     => phi_new
        phi_new => tmp_ptr

        iter = iter + 1
        if (fixed_iters == 0 .and. max_diff < TOL) exit

    end do

    call system_clock(t_end)
    elapsed = real(t_end - t_start, real64) / real(t_rate, real64)

    if (iter > 0) then
        ms_per_iter = elapsed / real(iter, real64) * 1000.0_real64
    else
        ms_per_iter = 0.0_real64
    end if

    ! ── CSV output ──────────────────────────────────────────────────────────────
    ! N, lang, storage, order, block, iterations, time_s, ms_per_iter, max_diff
    if (loop_order == 0) then
        write(*,'(I0,",fortran,f90,ji,",I0,",",I0,",",F14.8,",",F14.8,",",ES12.4)') &
            N, B, iter, elapsed, ms_per_iter, max_diff
    else
        write(*,'(I0,",fortran,f90,ij,",I0,",",I0,",",F14.8,",",F14.8,",",ES12.4)') &
            N, B, iter, elapsed, ms_per_iter, max_diff
    end if

    deallocate(phi, phi_new, f_arr)

end program poisson_jacobi
