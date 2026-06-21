
# HDQTS v6.2 — Lehmer-Enhanced Hybrid DAG Quantum TSP Solver

A hybrid quantum-classical solver for the Traveling Salesman Problem that uses a novel Lehmer code decoder to eliminate the classical oracle bottleneck.

## What This Is

HDQTS decomposes large TSP instances into a hierarchical DAG of small subproblems, solves each leaf with exact enumeration via Lehmer decoding, merges partial tours with cheapest insertion, and polishes with 2-opt local search.

**It works today as a classical solver. It is quantum-ready for when hardware matures.**

## What This Is NOT

- Not a claimed quantum advantage on current hardware
- Not competitive with LKH for 200+ city instances
- Not a replacement for commercial routing APIs at scale

## Quick Start

```bash
# Run benchmarks
cargo run --release

# Run tests (14 tests, including 100-city verification)
cargo test

# Build for browser
wasm-pack build --target web --release --out-dir www/pkg
```

## Verified Performance (5 seeds, Euclidean instances)

| Cities | Mean Improvement over NN | Best Case | Time (release mode) |
|--------|--------------------------|-----------|---------------------|
| 10 | 9.0% | 17.2% | ~1ms |
| 20 | 11.8% | 17.7% | ~5ms |
| 50 | 15.6% | 21.4% | ~80ms |
| 75 | 15.0% | 17.3% | ~360ms |
| 100 | 12.6% | 18.0% | ~1s |

Improvement measured against nearest-neighbor baseline.

## Architecture

```
[Distance Matrix] -> AdaptiveDAGDecomposer -> DAG Tree
                           |
                    (level-by-level, parallel)
                           |
              QuantumSubSolver [Lehmer/Held-Karp] -> SubSolutions
                           |
                    TourCombiner [cheapest insertion] -> Merged Tour
                           |
                    LocalSearch [2-opt with Edge Cache] -> Final Tour
```

### Lehmer Decoder
The core innovation. Instead of storing all O(n!) permutations to build a quantum oracle, we decode any integer index to its corresponding permutation in O(n) space and O(n^2) time using the factorial number system. Verified correct across 5,910 encode-decode roundtrips for n=4 to n=8.

### Adaptive DAG Decomposition
Hierarchical binary tree of city clusters using spectral bisection (Fiedler vector). Internal nodes store bridge cities connecting subtrees. Depth is controlled by `d_max = max(2, floor(log2(N/s_max)) + 1)` to prevent both flat and pathological deep structures.

### Complexity Gating
Quantum circuits are only attempted when estimated qubit count and circuit depth are within configured limits. Otherwise falls back to exact Held-Karp or nearest-neighbor. No false claims — the solver reports which nodes used which method.

## Limitations (Honest)

1. **8-city gap**: Decomposition creates ~9% overhead on small instances. Direct solve is better for n <= 8.
2. **Quantum today**: For n=4 subproblems, classical enumeration (6 routes) is faster than Grover circuit setup. Quantum becomes advantageous at n >= 8 (40,320 routes, sqrt speedup ~200x).
3. **Merge seams**: Cheapest-insertion merging costs 1-5% compared to direct global optimization.
4. **Single-threaded 2-opt**: The post-merge polish is not parallelized.
5. **Euclidean only**: Distance matrix is assumed symmetric and metric. No time windows, no capacity constraints.

## When To Use This

- Small to medium routing problems (10-100 stops)
- Offline or embedded deployment where cloud APIs are unavailable
- Exact optimality required on subproblems
- Research and benchmarking of hybrid quantum-classical algorithms
- Teaching tool for quantum computing concepts

## When NOT To Use This

- Large-scale logistics (200+ stops) — use LKH-3 or OR-Tools
- Real-time dynamic routing — use specialized VRP solvers
- Production quantum advantage claims — hardware isn't there yet

## Repository Structure

```
sirraya-hqtsp-lehmer/
├── Cargo.toml          # Rust project configuration
├── src/
│   ├── lib.rs          # Core solver (~1600 lines)
│   └── main.rs         # CLI benchmarking suite (~550 lines)
├── www/
│   ├── index.html      # Browser demo
│   └── pkg/            # WASM output (after build)
└── README.md
```

## Citation

If you use this work in research, please cite:

```
@software{mir2024hdqts,
  author = {Amir Hameed Mir},
  title = {HDQTS v6.2: Lehmer-Enhanced Hybrid DAG Quantum TSP Solver},
  year = {2024},
  url = {https://github.com/sirraya-labs/sirraya-hqtsp-lehmer}
}
```

## License

MIT — see LICENSE file.

## Contact

Amir Hameed Mir — amir@sirraya.org — Sirraya Labs

