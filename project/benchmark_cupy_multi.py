import config
from poisson_cupy_multi import solve_cupy_2gpu


def run_benchmark():
    config.BENCHMARK_MODE = False
    config.MAX_ITER = 200000

    BASE_TOL = 1e-7
    BASE_SIZE = 50
    sizes = [50, 100, 200, 400, 800, 1000, 2000]

    print("=======================================================================")
    print("  CuPy 2-GPU Jacobi Benchmark (y-split)")
    print("=======================================================================")
    print(f"{'Grid':<10} | {'Tol Used':<10} | {'Steps':<10} | {'Time (s)':<10}")
    print("-" * 60)

    for size in sizes:
        tol = BASE_TOL / ((size / BASE_SIZE) ** 2)

        iters, t = solve_cupy_2gpu(
            nx=size,
            ny=size,
            max_iter=config.MAX_ITER,
            tol=tol
        )

        print(f"{size}x{size:<5} | {tol:.1e} | {iters:<10} | {t:<10.4f}")

    print("-" * 60)


if __name__ == "__main__":
    run_benchmark()
