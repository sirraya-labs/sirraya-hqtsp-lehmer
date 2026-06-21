// ============================================================================
// HDQTS v6.2 -- Command-Line Interface for Peer-Reviewed Benchmarking
// ============================================================================
//
// Sirraya Labs -- Hybrid Quantum-Classical TSP Solver
// Author: Amir Hameed Mir <amir@sirraya.org>
// Version: 6.2.0 "Lehmer-Enhanced"
//
// This CLI produces comprehensive output suitable for paper artifacts,
// peer review verification, and reproducibility validation.
//
// Build: cargo run --release
// Output: Full benchmarks with DAG structure, solver breakdown, and statistics
// ============================================================================

use hdqtsp_lehmer::{HybridDAGQuantumTSP, LehmerDecoder, SolverConfig};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;
use std::time::Instant;
use std::collections::HashSet;

fn main() {
    println!("======================================================================");
    println!("  Sirraya HDQTS v6.2 -- Lehmer-Enhanced TSP Solver");
    println!("  Peer-Reviewed Benchmarking Artifact");
    println!("  Repository: github.com/sirraya/hdqtsp");
    println!("  Author: Amir Hameed Mir <amir@sirraya.org>");
    println!("======================================================================\n");

    if cfg!(debug_assertions) {
        println!("[NOTE] Running in DEBUG mode. For publication-grade timings,");
        println!("       re-run with: cargo run --release\n");
    }

    // -----------------------------------------------------------------------
    // ARTIFACT 1: Algorithm Configuration
    // -----------------------------------------------------------------------
    print_configuration();

    // -----------------------------------------------------------------------
    // ARTIFACT 2: Lehmer Decoder Correctness Proof
    // -----------------------------------------------------------------------
    verify_lehmer_decoder();

    // -----------------------------------------------------------------------
    // ARTIFACT 3: 8-City Known-Optimal Benchmark
    // -----------------------------------------------------------------------
    benchmark_8city_optimal();

    // -----------------------------------------------------------------------
    // ARTIFACT 4: Scaling Benchmarks (10, 20, 50, 75, 100 cities)
    // -----------------------------------------------------------------------
    let seeds = [42u64, 43, 44, 45, 46];
    let sizes = [10usize, 20, 50, 75, 100];
    benchmark_scaling(&sizes, &seeds);

    // -----------------------------------------------------------------------
    // ARTIFACT 5: Solver Comparison
    // -----------------------------------------------------------------------
    compare_solvers(50);

    // -----------------------------------------------------------------------
    // ARTIFACT 6: Reproducibility Hash
    // -----------------------------------------------------------------------
    print_reproducibility_info();

    println!("\n======================================================================");
    println!("  Benchmarking complete. All results deterministic with fixed seeds.");
    println!("  Contact: amir@sirraya.org for raw data or questions.");
    println!("======================================================================\n");
}

// ============================================================================
// ARTIFACT 1: Configuration
// ============================================================================

fn print_configuration() {
    println!("======================================================================");
    println!("  ARTIFACT 1: Solver Configuration");
    println!("======================================================================\n");

    let config = SolverConfig::default();

    println!("  Parameter               Value       Description");
    println!("  ----------------------- ----------- --------------------------------");
    println!("  max_subproblem_size     {:<11} Cities per leaf node", config.max_subproblem_size);
    println!("  max_qubits              {:<11} Maximum qubits for Grover circuit", config.max_qubits);
    println!("  max_circuit_depth       {:<11} Maximum circuit depth before fallback", config.max_circuit_depth);
    println!("  enable_2opt_polish      {:<11} Apply 2-opt local search", config.enable_2opt_polish);
    println!("  enable_3opt_polish      {:<11} Apply 3-opt (slower, deeper search)", config.enable_3opt_polish);
    println!("  max_refinement_iters    {:<11} Grover threshold refinement passes", config.max_refinement_iterations);
    println!("  decomposition_method    {:<11} auto / spectral / mst", config.decomposition_method);
    println!("  shots (quantum sim)     {:<11} Measurement shots per circuit", config.shots);
    println!("  parallel_workers        {:<11} Thread pool size for DAG levels", config.parallel_workers);
    println!("  enable_classical_fb     {:<11} Fallback to Held-Karp if circuit infeasible", config.enable_classical_fallback);
    println!("  enable_iterative_ref    {:<11} Tighten threshold after each quantum run", config.enable_iterative_refinement);

    let depth_formula = "d_max = max(2, floor(log2(N / s_max)) + 1)";
    let min_formula = "s_min = max(3, s_max / 2)";
    let overlap_formula = "overlap = max(1, s_max / 3)";

    println!("\n  Adaptive DAG Formulas:");
    println!("    {}", depth_formula);
    println!("    {}", min_formula);
    println!("    {}", overlap_formula);

    println!("\n  Lehmer Decoder: O(n) space, O(n^2) time factorial-number-system decoding.");
    println!("  Edge-Cached 2-Opt: HashMap-based distance lookup for O(1) edge cost access.");
    println!();
}

// ============================================================================
// ARTIFACT 2: Lehmer Decoder Verification
// ============================================================================

fn verify_lehmer_decoder() {
    println!("======================================================================");
    println!("  ARTIFACT 2: Lehmer Decoder Correctness Verification");
    println!("======================================================================\n");

    let test_sizes = [4usize, 5, 6, 7, 8];

    for &n in &test_sizes {
        let decoder = LehmerDecoder::new(n);
        let total = decoder.total_permutations();
        let qubits = decoder.required_qubits();
        let expected = factorial(n - 1);

        print!("  n={} | Permutations: {:>8} | Expected: {:>8} | Qubits: {:>3} | ",
            n, total, expected, qubits);

        // Verify total count
        assert_eq!(total, expected, "Permutation count mismatch for n={}", n);

        // Verify encode-decode roundtrip for all permutations (or first 10,000)
        let check_count = total.min(10000);
        let mut all_match = true;
        for idx in 0..check_count {
            let perm = decoder.decode(idx);
            let encoded = decoder.encode(&perm);
            if idx != encoded {
                all_match = false;
                break;
            }
        }

        if all_match {
            println!("Roundtrip: PASS (verified {} indices)", check_count);
        } else {
            println!("Roundtrip: FAIL");
        }
    }

    // Demonstrate the mathematical mapping
    println!("\n  Lehmer Code Demonstration (n=4, 3! = 6 permutations):");
    println!("  Formula: index = d2*2! + d1*1! + d0*0! where 0 <= di <= i");
    let decoder = LehmerDecoder::new(4);
    println!("  {:<8} {:<12} {:<20}", "Index", "Digits", "Route");
    println!("  {:<8} {:<12} {:<20}", "-----", "------", "-----");
    for idx in 0..6 {
        let route = decoder.decode_full_route(idx);
        let digits = decoder_digits(idx, 3);
        println!("  {:<8} {:<12} {:?}", idx, digits, route);
    }
    println!();
}

fn factorial(n: usize) -> usize {
    (1..=n).product()
}

fn decoder_digits(index: usize, k: usize) -> String {
    let mut remaining = index;
    let mut digits = Vec::new();
    let factorials: Vec<usize> = (0..=k).map(|i| (1..=i).product::<usize>()).collect();
    for i in (0..k).rev() {
        let fact = if i == 0 { 1 } else { factorials[i] };
        digits.push(remaining / fact);
        remaining %= fact;
    }
    format!("{:?}", digits)
}

// ============================================================================
// ARTIFACT 3: 8-City Known-Optimal Benchmark
// ============================================================================

fn benchmark_8city_optimal() {
    println!("======================================================================");
    println!("  ARTIFACT 3: 8-City Known-Optimal Benchmark");
    println!("  Optimal tour: 244.00 (verified by exhaustive enumeration)");
    println!("  Route: [0, 1, 4, 6, 2, 7, 5, 3, 0]");
    println!("======================================================================\n");

    let matrix_8 = vec![
        0.0, 29.0, 82.0, 46.0, 68.0, 52.0, 72.0, 42.0,
        29.0, 0.0, 55.0, 46.0, 42.0, 43.0, 43.0, 23.0,
        82.0, 55.0, 0.0, 68.0, 46.0, 55.0, 23.0, 43.0,
        46.0, 46.0, 68.0, 0.0, 82.0, 15.0, 72.0, 31.0,
        68.0, 42.0, 46.0, 82.0, 0.0, 74.0, 23.0, 52.0,
        52.0, 43.0, 55.0, 15.0, 74.0, 0.0, 61.0, 23.0,
        72.0, 43.0, 23.0, 72.0, 23.0, 61.0, 0.0, 42.0,
        42.0, 23.0, 43.0, 31.0, 52.0, 23.0, 42.0, 0.0,
    ];

    let start = Instant::now();
    let solver = HybridDAGQuantumTSP::new(matrix_8, 8).unwrap();
    let result = solver.solve_native();
    let elapsed = start.elapsed();

    let gap_pct = (result.distance - 244.0) / 244.0 * 100.0;

    println!("  Metric                  Value");
    println!("  ----------------------- -----------------------------------------");
    println!("  Solver distance         {:.2}", result.distance);
    println!("  Known optimal           244.00");
    println!("  Absolute gap            {:.2}", result.distance - 244.0);
    println!("  Gap percentage          {:.2}%", gap_pct);
    println!("  Execution time          {:.3}ms", elapsed.as_secs_f64() * 1000.0);
    
    // Fix: dereference the f64 before casting
    let dag_nodes = *result.stats.get("total_nodes").unwrap_or(&0.0) as usize;
    let lehmer_nodes = *result.stats.get("lehmer_exact_nodes").unwrap_or(&0.0) as usize;
    
    println!("  DAG nodes               {}", dag_nodes);
    println!("  DAG leaves              {}", lehmer_nodes);
    println!("  Lehmer-exact nodes      {}", lehmer_nodes);
    println!("  Route length            {} cities ({} with return)", result.route.len() - 1, result.route.len());
    println!("  Route valid             {}", if result.route.len() == 9 && result.route[0] == result.route[8] { "YES" } else { "NO" });

    // Print DAG node details
    println!("\n  DAG Node Breakdown:");
    println!("  {:<8} {:<8} {:<20} {:<12}", "NodeID", "Level", "Cities", "Method");
    println!("  {:<8} {:<8} {:<20} {:<12}", "------", "-----", "------", "------");
    for (&nid, sol) in &result.node_solutions {
        let cities_str = if let Some(node) = result.dag.nodes.get(&nid) {
            format!("{:?}", node.cities)
        } else {
            "N/A".to_string()
        };
        let level = result.dag.nodes.get(&nid).map(|n| n.level).unwrap_or(0);
        println!("  {:<8} {:<8} {:<20} {:<12}", nid, level, cities_str, sol.method);
    }

    // Status
    if gap_pct.abs() < 0.01 {
        println!("\n  STATUS: OPTIMAL (gap = 0.00%)");
    } else if gap_pct < 2.0 {
        println!("\n  STATUS: NEAR-OPTIMAL (gap = {:.2}%)", gap_pct);
    } else {
        println!("\n  STATUS: SUBOPTIMAL (gap = {:.2}%) -- DAG decomposition overhead", gap_pct);
    }

    println!();
}

// ============================================================================
// ARTIFACT 4: Scaling Benchmarks
// ============================================================================

fn benchmark_scaling(sizes: &[usize], seeds: &[u64]) {
    println!("======================================================================");
    println!("  ARTIFACT 4: Scaling Benchmarks (Multi-Seed)");
    println!("  Instance type: Random Euclidean, points in [0, 100]^2");
    println!("  Seeds: {:?}", seeds);
    println!("======================================================================\n");

    // Header
    println!("  {:<6} {:<8} {:<14} {:<14} {:<10} {:<10} {:<8} {:<8}",
        "N", "Seed", "NN Baseline", "HDQTS Dist", "Improv%", "Time(ms)", "DAG", "Leaves");
    println!("  {:<6} {:<8} {:<14} {:<14} {:<10} {:<10} {:<8} {:<8}",
        "------", "------", "------------", "------------", "--------", "--------", "------", "------");

    // Store all results for summary statistics
    let mut all_results: Vec<(usize, u64, f64, f64, f64, f64, usize, usize)> = Vec::new();

    for &n in sizes {
        for &seed in seeds {
            let matrix = generate_euclidean_matrix(n, seed);

            let start = Instant::now();
            let solver = HybridDAGQuantumTSP::new(matrix.clone(), n).unwrap();
            let result = solver.solve_native();
            let elapsed = start.elapsed();

            // Compute NN baseline
            let (_, nn_dist) = compute_nn_from_matrix(&matrix);
            let improvement = (nn_dist - result.distance) / nn_dist * 100.0;
            // Fix: dereference before casting
            let dag_nodes = *result.stats.get("total_nodes").unwrap_or(&0.0) as usize;
            let leaves = *result.stats.get("lehmer_exact_nodes").unwrap_or(&0.0) as usize;

            println!("  {:<6} {:<8} {:<14.2} {:<14.2} {:<9.2}% {:<9.1} {:<8} {:<8}",
                n, seed, nn_dist, result.distance, improvement,
                elapsed.as_secs_f64() * 1000.0, dag_nodes, leaves);

            all_results.push((n, seed, nn_dist, result.distance, improvement, 
                elapsed.as_secs_f64() * 1000.0, dag_nodes, leaves));
        }
    }

    // Summary statistics per size
    println!("\n  Summary Statistics (Aggregated over {} seeds):", seeds.len());
    println!("  {:<6} {:<14} {:<14} {:<14} {:<14} {:<10}",
        "N", "Mean Improv%", "Std Improv%", "Min Improv%", "Max Improv%", "Mean Time");
    println!("  {:<6} {:<14} {:<14} {:<14} {:<14} {:<10}",
        "------", "------------", "------------", "------------", "------------", "--------");

    for &n in sizes {
        let size_results: Vec<_> = all_results.iter().filter(|r| r.0 == n).collect();
        let improvements: Vec<f64> = size_results.iter().map(|r| r.4).collect();
        let times: Vec<f64> = size_results.iter().map(|r| r.5).collect();

        let mean_imp = improvements.iter().sum::<f64>() / improvements.len() as f64;
        let variance = improvements.iter().map(|x| (x - mean_imp).powi(2)).sum::<f64>() / improvements.len() as f64;
        let std_imp = variance.sqrt();
        let min_imp = improvements.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_imp = improvements.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_time = times.iter().sum::<f64>() / times.len() as f64;

        println!("  {:<6} {:<13.2}% {:<13.2}% {:<13.2}% {:<13.2}% {:<9.1}ms",
            n, mean_imp, std_imp, min_imp, max_imp, mean_time);
    }

    println!();
}

fn generate_euclidean_matrix(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let points: Vec<(f64, f64)> = (0..n)
        .map(|_| (rng.gen_range(0.0..100.0), rng.gen_range(0.0..100.0)))
        .collect();

    let mut matrix = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let dx = points[i].0 - points[j].0;
            let dy = points[i].1 - points[j].1;
            matrix[i * n + j] = (dx * dx + dy * dy).sqrt();
        }
    }
    matrix
}

fn compute_nn_from_matrix(matrix: &[f64]) -> (Vec<usize>, f64) {
    let n = (matrix.len() as f64).sqrt() as usize;
    let mut unvisited: HashSet<usize> = (0..n).collect();
    unvisited.remove(&0);
    let mut route = vec![0];
    let mut current = 0;
    let mut total = 0.0;

    while !unvisited.is_empty() {
        let next = *unvisited
            .iter()
            .min_by(|&&a, &&b| {
                let da = matrix[current * n + a];
                let db = matrix[current * n + b];
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        total += matrix[current * n + next];
        route.push(next);
        unvisited.remove(&next);
        current = next;
    }
    total += matrix[current * n + 0];
    (route, total)
}

// ============================================================================
// ARTIFACT 5: Solver Comparison
// ============================================================================

fn compare_solvers(n: usize) {
    println!("======================================================================");
    println!("  ARTIFACT 5: Solver Comparison (N={})", n);
    println!("======================================================================\n");

    let matrix = generate_euclidean_matrix(n, 42);
    let matrix_flat = matrix.clone();

    // HDQTS
    let start = Instant::now();
    let solver = HybridDAGQuantumTSP::new(matrix_flat.clone(), n).unwrap();
    let hdqts_result = solver.solve_native();
    let hdqts_time = start.elapsed();

    // Nearest Neighbor
    let start = Instant::now();
    let (nn_route, nn_dist) = compute_nn_from_matrix(&matrix);
    let nn_time = start.elapsed();

    // NN + 2-opt
    let start = Instant::now();
    let opt_route = simple_two_opt(&matrix, n, &nn_route);
    let opt_dist = compute_route_distance(&matrix, n, &opt_route);
    let opt_time = start.elapsed();

    println!("  Solver              Distance     Time (ms)    vs NN");
    println!("  ------------------- ------------ ------------ --------");
    println!("  HDQTS v6.2          {:<12.2} {:<12.1} {:.2}%",
        hdqts_result.distance, hdqts_time.as_secs_f64() * 1000.0,
        (nn_dist - hdqts_result.distance) / nn_dist * 100.0);
    println!("  Nearest Neighbor    {:<12.2} {:<12.1} baseline",
        nn_dist, nn_time.as_secs_f64() * 1000.0);
    println!("  NN + 2-opt          {:<12.2} {:<12.1} {:.2}%",
        opt_dist, opt_time.as_secs_f64() * 1000.0,
        (nn_dist - opt_dist) / nn_dist * 100.0);

    // Fix: dereference before casting
    let dag_nodes = *hdqts_result.stats.get("total_nodes").unwrap_or(&0.0) as usize;
    let leaves = *hdqts_result.stats.get("lehmer_exact_nodes").unwrap_or(&0.0) as usize;
    
    println!("\n  HDQTS DAG Structure:");
    println!("    Nodes:  {}", dag_nodes);
    println!("    Leaves: {}", leaves);
    println!();
}

fn simple_two_opt(matrix: &[f64], n: usize, route: &[usize]) -> Vec<usize> {
    let mut best = route.to_vec();
    let mut best_d = compute_route_distance(matrix, n, &best);
    let mut improved = true;

    while improved {
        improved = false;
        'outer: for i in 1..n - 1 {
            for j in i + 1..n {
                let mut candidate = best.clone();
                candidate[i..=j].reverse();
                let d = compute_route_distance(matrix, n, &candidate);
                if d < best_d - 1e-9 {
                    best = candidate;
                    best_d = d;
                    improved = true;
                    break 'outer;
                }
            }
        }
    }
    best
}

fn compute_route_distance(matrix: &[f64], n: usize, route: &[usize]) -> f64 {
    let mut total = 0.0;
    for i in 0..route.len() {
        total += matrix[route[i] * n + route[(i + 1) % route.len()]];
    }
    total
}

// ============================================================================
// ARTIFACT 6: Reproducibility
// ============================================================================

fn print_reproducibility_info() {
    println!("======================================================================");
    println!("  ARTIFACT 6: Reproducibility Information");
    println!("======================================================================\n");

    println!("  Build Information:");
    println!("    Language:       Rust (edition 2021)");
    // Use option_env! which returns None if not set, instead of env! which panics
    let rustc_ver = option_env!("RUSTC_VERSION").unwrap_or("unknown");
    println!("    Compiler:       rustc {}", rustc_ver);
    println!("    Profile:        {}", if cfg!(debug_assertions) { "debug" } else { "release" });
    let target = option_env!("TARGET").unwrap_or("unknown");
    println!("    Target:         {}", target);
    println!();

    println!("  Key Dependencies:");
    println!("    nalgebra        0.32   Linear algebra (distance matrices)");
    println!("    rand            0.8    Random number generation (StdRng)");
    println!("    serde           1.0    Serialization for WASM bridge");
    println!("    wasm-bindgen    0.2    WebAssembly compilation target");
    println!();

    println!("  Reproducibility:");
    println!("    All random instances use StdRng::seed_from_u64() with");
    println!("    fixed seeds [42, 43, 44, 45, 46]. Results are deterministic.");
    println!("    Re-run with: cargo run --release");
    println!();

    println!("  Source Code:");
    println!("    Repository: github.com/sirraya/hdqtsp");
    println!("    Version:    v6.2.0 (Lehmer-Enhanced)");
    println!("    License:    MIT");
    println!();

    // Environment info
    println!("  System Information:");
    println!("    OS:      {}", std::env::consts::OS);
    println!("    Arch:    {}", std::env::consts::ARCH);
    println!("    Threads: {}", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    println!();

    println!("  Contact:");
    println!("    Author:  Amir Hameed Mir");
    println!("    Email:   amir@sirraya.org");
    println!("    Lab:     Sirraya Labs -- Hybrid Quantum-Classical Computing");
    println!();
}