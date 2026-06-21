// ============================================================================
// HDQTS v6.2 — Complete Rust/WASM Implementation with Lehmer Decoder
// ============================================================================
//
// Sirraya Labs — Hybrid Quantum-Classical TSP Solver
// Author: Amir Hameed Mir <amir@sirraya.org>
// Version: 6.2.0 "Lehmer-Enhanced"
//
// Compile: wasm-pack build --target web --release --out-dir pkg
// Test: cargo test
// Run CLI: cargo run --release
//
// ============================================================================

use nalgebra::DMatrix;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::f64;
use wasm_bindgen::prelude::*;

// WASM allocator for smaller binary size
#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

// ============================================================================
// CROSS-PLATFORM TIMING (works in both WASM and native)
// ============================================================================

/// Get current time in milliseconds.
/// Uses js_sys in WASM, std::time in native.
#[cfg(target_arch = "wasm32")]
fn now_ms() -> f64 {
    js_sys::Date::now()
}

#[cfg(not(target_arch = "wasm32"))]
fn now_ms() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as f64
}

// ============================================================================
// LEHMER CODE DECODER
// ============================================================================

/// Lehmer Code Decoder for TSP route encoding.
///
/// Converts integer indices to unique permutations without storing all permutations.
/// This eliminates the O(n!) memory bottleneck.
///
/// HOW IT WORKS (Factorial Number System):
///   Any integer 0..(n-1)!-1 maps to a unique permutation of {1,...,n-1}
///   using the Lehmer code (also called "factoradic" representation).
///
/// Example for n=4, index=3:
///   Cities to permute: [1, 2, 3]
///   3 in factorial base: 3 = 1×2! + 1×1! + 0×0! → digits [1, 1, 0]
///   Available: [1, 2, 3]
///     Digit 1 → pick position 1 → 2, remaining [1, 3]
///     Digit 1 → pick position 1 → 3, remaining [1]
///     Digit 0 → pick position 0 → 1
///   Result: [2, 3, 1], Full route: [0, 2, 3, 1]
#[derive(Debug, Clone)]
pub struct LehmerDecoder {
    n: usize,
    k: usize,
    factorials: Vec<usize>,
    total_permutations: usize,
    required_qubits: usize,
}

impl LehmerDecoder {
    /// Create a new Lehmer decoder for n cities.
    pub fn new(n: usize) -> Self {
        let k = n.saturating_sub(1);
        
        let mut factorials = vec![1];
        for i in 1..=k {
            factorials.push(factorials.last().unwrap() * i);
        }
        
        let total_permutations = if k == 0 { 1 } else { factorials[k] };
        let required_qubits = if total_permutations <= 1 {
            1
        } else {
            (total_permutations as f64).log2().ceil() as usize
        };
        
        Self {
            n,
            k,
            factorials,
            total_permutations,
            required_qubits,
        }
    }
    
    /// Decode an integer index to a permutation of {1, ..., n-1}.
    pub fn decode(&self, index: usize) -> Vec<usize> {
        let index = if self.total_permutations > 0 {
            index % self.total_permutations
        } else {
            0
        };
        
        let mut available: Vec<usize> = (1..self.n).collect();
        let mut result = Vec::with_capacity(self.k);
        let mut remaining = index;
        
        for i in (0..self.k).rev() {
            let fact = self.factorials[i];
            let digit = remaining / fact;
            remaining %= fact;
            
            if digit < available.len() {
                result.push(available.remove(digit));
            } else {
                result.extend(available.drain(..));
                break;
            }
        }
        
        result
    }
    
    /// Encode a permutation back to its integer index.
    pub fn encode(&self, permutation: &[usize]) -> usize {
        let mut available: Vec<usize> = (1..self.n).collect();
        let mut index = 0;
        
        for (i, &city) in permutation.iter().enumerate() {
            if i >= self.k {
                break;
            }
            if let Some(pos) = available.iter().position(|&x| x == city) {
                index += pos * self.factorials[self.k - 1 - i];
                available.remove(pos);
            }
        }
        
        index
    }
    
    /// Decode to full TSP route starting at city 0.
    pub fn decode_full_route(&self, index: usize) -> Vec<usize> {
        let mut route = vec![0];
        route.extend(self.decode(index));
        route
    }
    
    /// Compute tour distance directly from index without storing route.
    pub fn compute_distance(&self, index: usize, matrix: &DMatrix<f64>) -> f64 {
        let route = self.decode_full_route(index);
        let mut total = 0.0;
        for i in 0..route.len() {
            let from = route[i];
            let to = route[(i + 1) % route.len()];
            total += matrix[(from, to)];
        }
        total
    }
    
    /// Find all route indices with distance ≤ threshold.
    pub fn find_good_indices(
        &self,
        matrix: &DMatrix<f64>,
        threshold: f64,
        max_to_check: Option<usize>,
    ) -> Vec<usize> {
        let max_check = max_to_check.unwrap_or(self.total_permutations.min(10000));
        let mut good_indices = Vec::new();
        
        if self.total_permutations <= max_check {
            for idx in 0..self.total_permutations {
                let dist = self.compute_distance(idx, matrix);
                if dist <= threshold {
                    good_indices.push(idx);
                }
            }
        } else {
            let mut rng = rand::thread_rng();
            let mut samples: HashSet<usize> = HashSet::new();
            while samples.len() < max_check {
                samples.insert(rng.gen_range(0..self.total_permutations));
            }
            for &idx in &samples {
                let dist = self.compute_distance(idx, matrix);
                if dist <= threshold {
                    good_indices.push(idx);
                }
            }
        }
        
        good_indices
    }
    
    /// Find the best (minimum distance) route by full enumeration.
    pub fn find_best_route(&self, matrix: &DMatrix<f64>) -> (usize, f64) {
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;
        
        for idx in 0..self.total_permutations {
            let dist = self.compute_distance(idx, matrix);
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }
        
        (best_idx, best_dist)
    }

    /// Get total number of permutations (n-1)!
    pub fn total_permutations(&self) -> usize {
        self.total_permutations
    }
    
    /// Get required qubits for encoding
    pub fn required_qubits(&self) -> usize {
        self.required_qubits
    }
}

// ============================================================================
// SHOT SCHEDULER — Adaptive quantum shot allocation
// ============================================================================

pub struct ShotScheduler {
    base_shots: usize,
}

impl ShotScheduler {
    pub fn new(base_shots: usize) -> Self {
        Self { base_shots }
    }
    
    pub fn shots_for(&self, decoder: &LehmerDecoder) -> usize {
        let total = decoder.total_permutations as f64;
        let grover_factor = (total.sqrt() / 4.0).min(4.0).max(1.0);
        (self.base_shots as f64 * grover_factor) as usize
    }
    
    pub fn quantum_advantage_viable(&self, decoder: &LehmerDecoder) -> bool {
        let total = decoder.total_permutations as f64;
        total.sqrt() > 10.0
    }
}

// ============================================================================
// INCREMENTAL DISTANCE CALCULATOR — Fast 2-opt delta computation
// ============================================================================

pub struct IncrementalDistance {
    matrix: DMatrix<f64>,
}

impl IncrementalDistance {
    pub fn new(matrix: DMatrix<f64>) -> Self {
        Self { matrix }
    }
    
    pub fn two_opt_delta(&self, route: &[usize], i: usize, j: usize) -> f64 {
        let n = route.len();
        if i == 0 || j >= n - 1 {
            return 0.0;
        }
        
        let a = route[i - 1];
        let b = route[i];
        let c = route[j];
        let d = route[(j + 1) % n];
        
        let removed = self.matrix[(a, b)] + self.matrix[(c, d)];
        let added = self.matrix[(a, c)] + self.matrix[(b, d)];
        
        added - removed
    }
    
    pub fn full_distance(&self, route: &[usize]) -> f64 {
        let mut total = 0.0;
        for i in 0..route.len() {
            total += self.matrix[(route[i], route[(i + 1) % route.len()])];
        }
        total
    }
}

// ============================================================================
// EDGE CACHE — Hash-based distance lookup for 2-opt
// ============================================================================

struct EdgeCache {
    cache: HashMap<(usize, usize), f64>,
}

impl EdgeCache {
    fn new() -> Self {
        Self { cache: HashMap::new() }
    }
    
    fn get(&mut self, matrix: &DMatrix<f64>, i: usize, j: usize) -> f64 {
        let key = if i < j { (i, j) } else { (j, i) };
        *self.cache.entry(key).or_insert_with(|| matrix[(i, j)])
    }
}

// ============================================================================
// CONFIGURATION
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    pub max_circuit_depth: usize,
    pub max_qubits: usize,
    pub enable_classical_fallback: bool,
    pub enable_iterative_refinement: bool,
    pub max_refinement_iterations: usize,
    pub enable_2opt_polish: bool,
    pub enable_3opt_polish: bool,
    pub decomposition_method: String,
    pub max_subproblem_size: usize,
    pub shots: usize,
    pub parallel_workers: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_circuit_depth: 200,
            max_qubits: 12,
            enable_classical_fallback: true,
            enable_iterative_refinement: true,
            max_refinement_iterations: 3,
            enable_2opt_polish: true,
            enable_3opt_polish: false,
            decomposition_method: "auto".to_string(),
            max_subproblem_size: 6,
            shots: 512,
            parallel_workers: 4,
        }
    }
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubSolution {
    pub success: bool,
    pub route: Vec<usize>,
    pub distance: f64,
    pub probability: f64,
    pub shots_used: usize,
    pub exec_time: f64,
    pub qubits_used: usize,
    pub node_id: usize,
    pub method: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DAGNode {
    pub node_id: usize,
    pub cities: Vec<usize>,
    pub level: usize,
    pub parent_ids: Vec<usize>,
    pub child_ids: Vec<usize>,
    pub connection_cities: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DAGGraph {
    pub nodes: HashMap<usize, DAGNode>,
    pub roots: Vec<usize>,
    pub leaves: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalResult {
    pub success: bool,
    pub route: Vec<usize>,
    pub distance: f64,
    pub exec_time: f64,
    pub node_solutions: HashMap<usize, SubSolution>,
    pub dag: DAGGraph,
    pub stats: HashMap<String, f64>,
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

fn route_distance(matrix: &DMatrix<f64>, route: &[usize], closed: bool) -> f64 {
    if route.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0;
    for i in 0..route.len() - 1 {
        total += matrix[(route[i], route[i + 1])];
    }
    if closed {
        total += matrix[(route[route.len() - 1], route[0])];
    }
    total
}

fn nearest_neighbor(matrix: &DMatrix<f64>, start: usize) -> (Vec<usize>, f64) {
    let n = matrix.nrows();
    let mut unvisited: HashSet<usize> = (0..n).collect();
    unvisited.remove(&start);
    let mut route = vec![start];
    let mut current = start;

    while !unvisited.is_empty() {
        let next = *unvisited
            .iter()
            .min_by(|&&a, &&b| {
                matrix[(current, a)]
                    .partial_cmp(&matrix[(current, b)])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        route.push(next);
        unvisited.remove(&next);
        current = next;
    }

    let dist = route_distance(matrix, &route, true);
    (route, dist)
}

fn combinations<T: Clone>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![vec![]];
    }
    if items.is_empty() {
        return vec![];
    }
    let mut result = Vec::new();
    let first = items[0].clone();
    for mut combo in combinations(&items[1..], k - 1) {
        combo.insert(0, first.clone());
        result.push(combo);
    }
    result.extend(combinations(&items[1..], k));
    result
}

fn held_karp(matrix: &DMatrix<f64>) -> (Vec<usize>, f64) {
    let n = matrix.nrows();
    
    if n == 1 {
        return (vec![0], 0.0);
    }
    if n == 2 {
        return (vec![0, 1], matrix[(0, 1)] + matrix[(1, 0)]);
    }

    let full_mask = (1 << n) - 1;
    let mut dp: HashMap<(usize, usize), f64> = HashMap::new();
    let mut parent: HashMap<(usize, usize), usize> = HashMap::new();

    for i in 1..n {
        dp.insert((1 << i, i), matrix[(0, i)]);
        parent.insert((1 << i, i), 0);
    }

    for size in 2..n {
        let subsets = combinations(&(1..n).collect::<Vec<_>>(), size);
        for subset in &subsets {
            let bits: usize = subset.iter().map(|&s| 1 << s).sum();
            for &last in subset {
                let prev_bits = bits ^ (1 << last);
                let mut best = f64::INFINITY;
                let mut best_prev = 0;
                for &prev in subset {
                    if prev == last {
                        continue;
                    }
                    if let Some(&cost) = dp.get(&(prev_bits, prev)) {
                        let total = cost + matrix[(prev, last)];
                        if total < best {
                            best = total;
                            best_prev = prev;
                        }
                    }
                }
                if best < f64::INFINITY {
                    dp.insert((bits, last), best);
                    parent.insert((bits, last), best_prev);
                }
            }
        }
    }

    let start_mask = full_mask ^ 1;
    let mut best_total = f64::INFINITY;
    let mut best_last = 0;
    for i in 1..n {
        if let Some(&cost) = dp.get(&(start_mask, i)) {
            let total = cost + matrix[(i, 0)];
            if total < best_total {
                best_total = total;
                best_last = i;
            }
        }
    }

    if best_last == 0 {
        return nearest_neighbor(matrix, 0);
    }

    let mut route = vec![];
    let mut bits = start_mask;
    let mut last = best_last;
    while last != 0 {
        route.push(last);
        let prev = parent[&(bits, last)];
        bits ^= 1 << last;
        last = prev;
    }
    route.push(0);
    route.reverse();
    
    (route, best_total)
}

// ============================================================================
// ENHANCED 2-OPT with Edge Cache
// ============================================================================

fn two_opt_cached(matrix: &DMatrix<f64>, route: &[usize]) -> Vec<usize> {
    let mut best = route.to_vec();
    let mut best_d = route_distance(matrix, &best, true);
    let mut cache = EdgeCache::new();
    let mut improved = true;
    let n = best.len();

    while improved {
        improved = false;
        'outer: for i in 1..n - 1 {
            for j in i + 1..n {
                if j == n - 1 && i == 0 {
                    continue;
                }
                
                let a = best[i - 1];
                let b = best[i];
                let c = best[j];
                let d = best[(j + 1) % n];
                
                let old_dist = cache.get(matrix, a, b) + cache.get(matrix, c, d);
                let new_dist = cache.get(matrix, a, c) + cache.get(matrix, b, d);
                
                if new_dist < old_dist - 1e-9 {
                    let mut candidate = best.clone();
                    candidate[i..=j].reverse();
                    let d_total = route_distance(matrix, &candidate, true);
                    if d_total < best_d - 1e-9 {
                        best = candidate;
                        best_d = d_total;
                        improved = true;
                        break 'outer;
                    }
                }
            }
        }
    }
    best
}

// ============================================================================
// ADAPTIVE DAG DECOMPOSER
// ============================================================================

pub struct AdaptiveDAGDecomposer {
    matrix: DMatrix<f64>,
    n: usize,
    max_sub: usize,
    max_depth: usize,
    min_cluster_size: usize,
    overlap_size: usize,
}

impl AdaptiveDAGDecomposer {
    pub fn new(matrix: DMatrix<f64>, max_sub: usize) -> Self {
        let n = matrix.nrows();
        let max_depth = (2usize).max(((n as f64 / max_sub as f64).log2() as usize) + 1);
        let min_cluster_size = (3usize).max(max_sub / 2);
        let overlap_size = (1usize).max(max_sub / 3);

        Self {
            matrix,
            n,
            max_sub,
            max_depth,
            min_cluster_size,
            overlap_size,
        }
    }

    pub fn decompose(&self) -> DAGGraph {
        if self.n <= self.max_sub {
            return self.single_node();
        }

        if let Some(mut dag) = self.balanced_spectral() {
            self.annotate_connections(&mut dag);
            return dag;
        }

        self.fallback_flat()
    }

    fn single_node(&self) -> DAGGraph {
        let cities: Vec<usize> = (0..self.n).collect();
        let node = DAGNode {
            node_id: 0,
            cities,
            level: 0,
            parent_ids: vec![],
            child_ids: vec![],
            connection_cities: vec![],
        };
        let mut nodes = HashMap::new();
        nodes.insert(0, node);
        DAGGraph {
            nodes,
            roots: vec![0],
            leaves: vec![0],
        }
    }

    fn balanced_spectral(&self) -> Option<DAGGraph> {
        let max_d = self.matrix.max();
        let mut sim = DMatrix::from_element(self.n, self.n, max_d) - &self.matrix;
        for i in 0..self.n {
            sim[(i, i)] = 0.0;
        }

        let deg = sim.column_sum();
        let lap = DMatrix::from_diagonal(&deg) - sim;

        let eigen = lap.symmetric_eigen();
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        let mut idx: Vec<usize> = (0..self.n).collect();
        idx.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

        if idx.len() < 3 {
            return None;
        }

        let coords: Vec<(f64, f64)> = (0..self.n)
            .map(|i| (eigenvectors[(i, idx[1])], eigenvectors[(i, idx[2])]))
            .collect();

        let mut dag = DAGGraph {
            nodes: HashMap::new(),
            roots: vec![0],
            leaves: vec![],
        };

        let root = DAGNode {
            node_id: 0,
            cities: vec![],
            level: 0,
            parent_ids: vec![],
            child_ids: vec![],
            connection_cities: vec![],
        };
        dag.nodes.insert(0, root);

        let mut node_id_counter = 1;
        let mut bridge_cities = vec![];
        let all_cities: Vec<usize> = (0..self.n).collect();

        self.build_dag_tree(
            &coords,
            &all_cities,
            0,
            &mut dag,
            0,
            &mut node_id_counter,
            &mut bridge_cities,
        );

        let root_cities: Vec<usize> = {
            let mut seen = HashSet::new();
            bridge_cities
                .into_iter()
                .filter(|c| seen.insert(*c))
                .take(self.max_sub)
                .collect()
        };

        if root_cities.is_empty() {
            dag.nodes.get_mut(&0).unwrap().cities = vec![0];
        } else {
            dag.nodes.get_mut(&0).unwrap().cities = root_cities;
        }

        dag.leaves = dag
            .nodes
            .iter()
            .filter(|(_, node)| node.child_ids.is_empty())
            .map(|(&id, _)| id)
            .collect();

        Some(dag)
    }

    fn build_dag_tree(
        &self,
        coords: &[(f64, f64)],
        cities: &[usize],
        depth: usize,
        dag: &mut DAGGraph,
        parent_id: usize,
        node_id_counter: &mut usize,
        bridge_cities: &mut Vec<usize>,
    ) {
        if depth >= self.max_depth
            || cities.len() <= self.max_sub
            || cities.len() <= self.min_cluster_size
        {
            let node = DAGNode {
                node_id: *node_id_counter,
                cities: cities.to_vec(),
                level: depth + 1,
                parent_ids: vec![parent_id],
                child_ids: vec![],
                connection_cities: vec![],
            };
            dag.nodes.get_mut(&parent_id).unwrap().child_ids.push(*node_id_counter);
            dag.nodes.insert(*node_id_counter, node);
            *node_id_counter += 1;
            return;
        }

        let mut sorted_cities: Vec<(usize, f64)> = cities
            .iter()
            .map(|&c| (c, coords[c].0))
            .collect();
        sorted_cities.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let median_idx = sorted_cities.len() / 2;
        let left: Vec<usize> = sorted_cities[..median_idx].iter().map(|(c, _)| *c).collect();
        let right: Vec<usize> = sorted_cities[median_idx..].iter().map(|(c, _)| *c).collect();

        if left.len() < self.min_cluster_size || right.len() < self.min_cluster_size {
            let node = DAGNode {
                node_id: *node_id_counter,
                cities: cities.to_vec(),
                level: depth + 1,
                parent_ids: vec![parent_id],
                child_ids: vec![],
                connection_cities: vec![],
            };
            dag.nodes.get_mut(&parent_id).unwrap().child_ids.push(*node_id_counter);
            dag.nodes.insert(*node_id_counter, node);
            *node_id_counter += 1;
            return;
        }

        let b1 = self.find_bridge_city(&left, &right);
        let b2 = self.find_bridge_city(&right, &left);
        bridge_cities.push(b1);
        bridge_cities.push(b2);

        let internal_id = *node_id_counter;
        let internal = DAGNode {
            node_id: internal_id,
            cities: vec![b1, b2],
            level: depth + 1,
            parent_ids: vec![parent_id],
            child_ids: vec![],
            connection_cities: vec![],
        };
        dag.nodes.get_mut(&parent_id).unwrap().child_ids.push(internal_id);
        dag.nodes.insert(internal_id, internal);
        *node_id_counter += 1;

        self.build_dag_tree(coords, &left, depth + 1, dag, internal_id, node_id_counter, bridge_cities);
        self.build_dag_tree(coords, &right, depth + 1, dag, internal_id, node_id_counter, bridge_cities);
    }

    fn find_bridge_city(&self, cluster_a: &[usize], cluster_b: &[usize]) -> usize {
        let mut best = f64::INFINITY;
        let mut best_city = cluster_a[0];
        for &c in cluster_a {
            let sum: f64 = cluster_b.iter().map(|&d| self.matrix[(c, d)]).sum();
            let avg_dist = sum / cluster_b.len() as f64;
            if avg_dist < best {
                best = avg_dist;
                best_city = c;
            }
        }
        best_city
    }

    fn fallback_flat(&self) -> DAGGraph {
        let cities: Vec<usize> = (0..self.n).collect();
        let mut nodes = HashMap::new();
        let mut node_id = 0;

        let root = DAGNode {
            node_id,
            cities: cities[..self.max_sub.min(self.n)].to_vec(),
            level: 0,
            parent_ids: vec![],
            child_ids: vec![],
            connection_cities: vec![],
        };
        nodes.insert(node_id, root);
        node_id += 1;

        let stride = self.max_sub - self.overlap_size;
        let mut i = self.max_sub - self.overlap_size;
        while i < self.n {
            let start = i.saturating_sub(self.overlap_size);
            let end = (i + self.max_sub - self.overlap_size).min(self.n);
            let chunk = cities[start..end].to_vec();
            if chunk.len() >= 2 {
                let node = DAGNode {
                    node_id,
                    cities: chunk,
                    level: 1,
                    parent_ids: vec![0],
                    child_ids: vec![],
                    connection_cities: vec![],
                };
                nodes.get_mut(&0).unwrap().child_ids.push(node_id);
                nodes.insert(node_id, node);
                node_id += 1;
            }
            i += stride;
        }

        let leaves: Vec<usize> = nodes
            .iter()
            .filter(|(_, node)| node.child_ids.is_empty())
            .map(|(&id, _)| id)
            .collect();

        DAGGraph {
            nodes,
            roots: vec![0],
            leaves,
        }
    }

    fn annotate_connections(&self, dag: &mut DAGGraph) {
        let mut updates = vec![];
        for (&nid, node) in &dag.nodes {
            for &pid in &node.parent_ids {
                if let Some(parent) = dag.nodes.get(&pid) {
                    let shared: Vec<usize> = node
                        .cities
                        .iter()
                        .filter(|c| parent.cities.contains(c))
                        .copied()
                        .collect();
                    if !shared.is_empty() {
                        updates.push((nid, shared));
                    }
                }
            }
        }
        for (nid, shared) in updates {
            if let Some(node) = dag.nodes.get_mut(&nid) {
                for c in shared {
                    if !node.connection_cities.contains(&c) {
                        node.connection_cities.push(c);
                    }
                }
            }
        }
    }
}

// ============================================================================
// TOUR COMBINER
// ============================================================================

pub struct TourCombiner {
    matrix: DMatrix<f64>,
    n: usize,
}

impl TourCombiner {
    pub fn new(matrix: DMatrix<f64>) -> Self {
        let n = matrix.nrows();
        Self { matrix, n }
    }

    pub fn combine(&self, leaf_routes: &[Vec<usize>]) -> Vec<usize> {
        if leaf_routes.is_empty() {
            return nearest_neighbor(&self.matrix, 0).0;
        }

        let mut ordered_cities = Vec::new();
        let mut all_cities: HashSet<usize> = HashSet::new();

        let mut sorted_routes = leaf_routes.to_vec();
        sorted_routes.sort_by_key(|r| -(r.len() as isize));

        for route in &sorted_routes {
            for &c in route {
                if all_cities.insert(c) {
                    ordered_cities.push(c);
                }
            }
        }

        if ordered_cities.len() == self.n {
            return ordered_cities;
        }

        let present: HashSet<usize> = ordered_cities.iter().copied().collect();
        let missing: Vec<usize> = (0..self.n).filter(|c| !present.contains(c)).collect();

        let mut tour = ordered_cities;
        for city in missing {
            let mut best_cost = f64::INFINITY;
            let mut best_pos = 1;
            for i in 0..tour.len() {
                let a = tour[i];
                let b = tour[(i + 1) % tour.len()];
                let cost = self.matrix[(a, city)] + self.matrix[(city, b)] - self.matrix[(a, b)];
                if cost < best_cost {
                    best_cost = cost;
                    best_pos = i + 1;
                }
            }
            tour.insert(best_pos, city);
        }

        tour
    }

    pub fn validate(&self, route: &[usize]) -> bool {
        let mut sorted = route.to_vec();
        sorted.sort();
        let expected: Vec<usize> = (0..self.n).collect();
        sorted == expected
    }
}

// ============================================================================
// MAIN SOLVER STRUCT
// ============================================================================

#[wasm_bindgen]
pub struct HybridDAGQuantumTSP {
    pub(crate) matrix: DMatrix<f64>,
    pub(crate) n: usize,
    pub(crate) config: SolverConfig,
    pub(crate) decomposer: AdaptiveDAGDecomposer,
    pub(crate) combiner: TourCombiner,
}

// ============================================================================
// NATIVE API — For CLI binaries and tests (not WASM-exposed)
// ============================================================================

impl HybridDAGQuantumTSP {
    /// Solve TSP in native mode and return FinalResult directly.
    /// Works in native binaries, CLI, and tests.
    pub fn solve_native(&self) -> FinalResult {
        let t0 = now_ms();

        if self.n <= self.config.max_subproblem_size {
            return self.solve_with_lehmer_native(t0);
        }

        let node_solutions = self.solve_dag();
        let dag = self.decomposer.decompose();
        
        let leaf_routes: Vec<Vec<usize>> = dag
            .leaves
            .iter()
            .filter_map(|&leaf_id| {
                node_solutions.get(&leaf_id).and_then(|sol| {
                    if sol.success && !sol.route.is_empty() {
                        Some(sol.route.clone())
                    } else {
                        None
                    }
                })
            })
            .collect();

        let mut merged = self.combiner.combine(&leaf_routes);
        if !self.combiner.validate(&merged) {
            merged = nearest_neighbor(&self.matrix, 0).0;
        }
        if self.config.enable_2opt_polish {
            merged = two_opt_cached(&self.matrix, &merged);
        }

        let closed = {
            let mut c = merged.clone();
            c.push(merged[0]);
            c
        };
        let dist = route_distance(&self.matrix, &merged, true);
        let elapsed = (now_ms() - t0) / 1000.0;
        let stats = self.build_stats(&node_solutions, elapsed);

        FinalResult {
            success: true,
            route: closed,
            distance: dist,
            exec_time: elapsed,
            node_solutions: node_solutions.clone(),
            dag,
            stats,
        }
    }

    fn solve_with_lehmer_native(&self, t0: f64) -> FinalResult {
        let decoder = LehmerDecoder::new(self.n);
        let (best_idx, best_dist) = decoder.find_best_route(&self.matrix);
        let route = decoder.decode_full_route(best_idx);

        let mut polished = route.clone();
        if self.config.enable_2opt_polish {
            polished = two_opt_cached(&self.matrix, &polished);
        }

        let closed = {
            let mut c = polished.clone();
            c.push(polished[0]);
            c
        };

        let elapsed = (now_ms() - t0) / 1000.0;

        let mut node_solutions = HashMap::new();
        node_solutions.insert(0, SubSolution {
            success: true,
            route: route.clone(),
            distance: best_dist,
            probability: 1.0,
            shots_used: 0,
            exec_time: elapsed,
            qubits_used: 0,
            node_id: 0,
            method: "lehmer_exact".to_string(),
            error: None,
        });

        let dag = self.decomposer.decompose();
        let stats = self.build_stats(&node_solutions, elapsed);

        FinalResult {
            success: true,
            route: closed,
            distance: best_dist,
            exec_time: elapsed,
            node_solutions: node_solutions.clone(),
            dag,
            stats,
        }
    }

    pub(crate) fn solve_dag(&self) -> HashMap<usize, SubSolution> {
        let dag = self.decomposer.decompose();
        let mut solutions = HashMap::new();

        let max_level = dag.nodes.values().map(|n| n.level).max().unwrap_or(0);

        for level in 0..=max_level {
            let level_nodes: Vec<usize> = dag
                .nodes
                .iter()
                .filter(|(_, node)| node.level == level)
                .map(|(&id, _)| id)
                .collect();

            let level_solutions = self.solve_group(&level_nodes, &solutions);
            solutions.extend(level_solutions);
        }

        solutions
    }

    fn solve_group(
        &self,
        node_ids: &[usize],
        parent_solutions: &HashMap<usize, SubSolution>,
    ) -> HashMap<usize, SubSolution> {
        node_ids
            .iter()
            .map(|&nid| {
                let sol = self.solve_node(nid, parent_solutions);
                (nid, sol)
            })
            .collect()
    }

    fn solve_node(
        &self,
        node_id: usize,
        _parent_solutions: &HashMap<usize, SubSolution>,
    ) -> SubSolution {
        let dag = self.decomposer.decompose();
        let node = match dag.nodes.get(&node_id) {
            Some(n) => n,
            None => {
                return SubSolution {
                    success: false,
                    route: vec![],
                    distance: f64::INFINITY,
                    probability: 0.0,
                    shots_used: 0,
                    exec_time: 0.0,
                    qubits_used: 0,
                    node_id,
                    method: "failed".to_string(),
                    error: Some("Node not found".to_string()),
                };
            }
        };

        if node.cities.len() < 2 {
            let dist = if node.cities.len() == 1 {
                0.0
            } else {
                route_distance(&self.matrix, &node.cities, true)
            };
            return SubSolution {
                success: true,
                route: node.cities.clone(),
                distance: dist,
                probability: 1.0,
                shots_used: 0,
                exec_time: 0.0,
                qubits_used: 0,
                node_id,
                method: "trivial".to_string(),
                error: None,
            };
        }

        let decoder = LehmerDecoder::new(node.cities.len());
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;

        for idx in 0..decoder.total_permutations {
            let local_route = decoder.decode_full_route(idx);
            let mut dist = 0.0;
            for i in 0..local_route.len() {
                let from = node.cities[local_route[i]];
                let to = node.cities[local_route[(i + 1) % local_route.len()]];
                dist += self.matrix[(from, to)];
            }
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }

        let local_route = decoder.decode_full_route(best_idx);
        let global_route: Vec<usize> = local_route.iter().map(|&i| node.cities[i]).collect();

        SubSolution {
            success: true,
            route: global_route,
            distance: best_dist,
            probability: 1.0,
            shots_used: 0,
            exec_time: 0.0,
            qubits_used: 0,
            node_id,
            method: "lehmer_exact".to_string(),
            error: None,
        }
    }

    fn build_stats(
        &self,
        node_solutions: &HashMap<usize, SubSolution>,
        elapsed: f64,
    ) -> HashMap<String, f64> {
        let total = node_solutions.len() as f64;
        let successful = node_solutions.values().filter(|s| s.success).count() as f64;
        let lehmer = node_solutions.values().filter(|s| s.method == "lehmer_exact").count() as f64;
        let trivial = node_solutions.values().filter(|s| s.method == "trivial").count() as f64;
        let failed = total - successful;

        let mut stats = HashMap::new();
        stats.insert("total_nodes".to_string(), total);
        stats.insert("successful_nodes".to_string(), successful);
        stats.insert("lehmer_exact_nodes".to_string(), lehmer);
        stats.insert("trivial_nodes".to_string(), trivial);
        stats.insert("failed_nodes".to_string(), failed);
        stats.insert("total_exec_time".to_string(), elapsed);
        stats
    }
}

// ============================================================================
// WASM BINDINGS — Exposed to JavaScript/browser
// ============================================================================

#[wasm_bindgen]
impl HybridDAGQuantumTSP {
    /// Create a new solver from a flat distance matrix (row-major order).
    #[wasm_bindgen(constructor)]
    pub fn new(distance_matrix: Vec<f64>, n: usize) -> Result<HybridDAGQuantumTSP, JsValue> {
        if distance_matrix.len() != n * n {
            return Err(JsValue::from_str(
                &format!(
                    "Distance matrix must be {} × {} ({} elements), got {} elements",
                    n, n, n * n, distance_matrix.len()
                )
            ));
        }

        let matrix = DMatrix::from_row_slice(n, n, &distance_matrix);
        let config = SolverConfig::default();
        let decomposer = AdaptiveDAGDecomposer::new(matrix.clone(), config.max_subproblem_size);
        let combiner = TourCombiner::new(matrix.clone());

        Ok(Self {
            matrix,
            n,
            config,
            decomposer,
            combiner,
        })
    }

    /// Create solver with custom configuration from JSON.
    pub fn with_config(
        distance_matrix: Vec<f64>,
        n: usize,
        config_js: JsValue,
    ) -> Result<HybridDAGQuantumTSP, JsValue> {
        let config: SolverConfig = serde_wasm_bindgen::from_value(config_js)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

        if distance_matrix.len() != n * n {
            return Err(JsValue::from_str("Distance matrix size mismatch"));
        }

        let matrix = DMatrix::from_row_slice(n, n, &distance_matrix);
        let decomposer = AdaptiveDAGDecomposer::new(matrix.clone(), config.max_subproblem_size);
        let combiner = TourCombiner::new(matrix.clone());

        Ok(Self {
            matrix,
            n,
            config,
            decomposer,
            combiner,
        })
    }

    /// Solve the TSP and return JSON-serializable result (for WASM/browser).
    pub fn solve(&self) -> JsValue {
        let t0 = now_ms();

        if self.n <= self.config.max_subproblem_size {
            return self.solve_with_lehmer_wasm(t0);
        }

        let node_solutions = self.solve_dag();
        let dag = self.decomposer.decompose();
        
        let leaf_routes: Vec<Vec<usize>> = dag
            .leaves
            .iter()
            .filter_map(|&leaf_id| {
                node_solutions.get(&leaf_id).and_then(|sol| {
                    if sol.success && !sol.route.is_empty() {
                        Some(sol.route.clone())
                    } else {
                        None
                    }
                })
            })
            .collect();

        let mut merged = self.combiner.combine(&leaf_routes);
        if !self.combiner.validate(&merged) {
            merged = nearest_neighbor(&self.matrix, 0).0;
        }
        if self.config.enable_2opt_polish {
            merged = two_opt_cached(&self.matrix, &merged);
        }

        let closed = {
            let mut c = merged.clone();
            c.push(merged[0]);
            c
        };
        let dist = route_distance(&self.matrix, &merged, true);
        let elapsed = (now_ms() - t0) / 1000.0;
        let stats = self.build_stats(&node_solutions, elapsed);

        serde_wasm_bindgen::to_value(&FinalResult {
            success: true,
            route: closed,
            distance: dist,
            exec_time: elapsed,
            node_solutions: node_solutions.clone(),
            dag,
            stats,
        })
        .unwrap_or(JsValue::NULL)
    }

    fn solve_with_lehmer_wasm(&self, t0: f64) -> JsValue {
        let decoder = LehmerDecoder::new(self.n);
        let (best_idx, best_dist) = decoder.find_best_route(&self.matrix);
        let route = decoder.decode_full_route(best_idx);

        let mut polished = route.clone();
        if self.config.enable_2opt_polish {
            polished = two_opt_cached(&self.matrix, &polished);
        }

        let closed = {
            let mut c = polished.clone();
            c.push(polished[0]);
            c
        };

        let elapsed = (now_ms() - t0) / 1000.0;

        let mut node_solutions = HashMap::new();
        node_solutions.insert(0, SubSolution {
            success: true,
            route: route.clone(),
            distance: best_dist,
            probability: 1.0,
            shots_used: 0,
            exec_time: elapsed,
            qubits_used: 0,
            node_id: 0,
            method: "lehmer_exact".to_string(),
            error: None,
        });

        let dag = self.decomposer.decompose();
        let stats = self.build_stats(&node_solutions, elapsed);

        serde_wasm_bindgen::to_value(&FinalResult {
            success: true,
            route: closed,
            distance: best_dist,
            exec_time: elapsed,
            node_solutions: node_solutions.clone(),
            dag,
            stats,
        })
        .unwrap_or(JsValue::NULL)
    }
}

// ============================================================================
// STANDALONE WASM FUNCTIONS
// ============================================================================

#[wasm_bindgen]
pub fn solve_tsp(distance_matrix: Vec<f64>, n: usize) -> JsValue {
    match HybridDAGQuantumTSP::new(distance_matrix, n) {
        Ok(solver) => solver.solve(),
        Err(e) => JsValue::from_str(&format!("Error: {:?}", e)),
    }
}

#[wasm_bindgen]
pub fn solve_tsp_with_config(distance_matrix: Vec<f64>, n: usize, config: JsValue) -> JsValue {
    match HybridDAGQuantumTSP::with_config(distance_matrix, n, config) {
        Ok(solver) => solver.solve(),
        Err(e) => JsValue::from_str(&format!("Error: {:?}", e)),
    }
}

#[wasm_bindgen]
pub fn version() -> String {
    "HDQTS v6.2 Rust/WASM — Lehmer-Enhanced Edition".to_string()
}

#[wasm_bindgen]
pub fn decode_route(n: usize, index: usize) -> Vec<usize> {
    let decoder = LehmerDecoder::new(n);
    decoder.decode_full_route(index)
}

#[wasm_bindgen]
pub fn total_routes(n: usize) -> usize {
    let decoder = LehmerDecoder::new(n);
    decoder.total_permutations
}

#[wasm_bindgen]
pub fn benchmark_all(distance_matrix: Vec<f64>, n: usize) -> JsValue {
    let matrix = DMatrix::from_row_slice(n, n, &distance_matrix);

    let t0 = now_ms();
    let solver = HybridDAGQuantumTSP::new(distance_matrix.clone(), n).unwrap();
    let hdqts_result = solver.solve();
    let hdqts_time = (now_ms() - t0) / 1000.0;

    let t0 = now_ms();
    let (nn_route, nn_dist) = nearest_neighbor(&matrix, 0);
    let nn_time = (now_ms() - t0) / 1000.0;

    let t0 = now_ms();
    let opt_route = two_opt_cached(&matrix, &nn_route);
    let opt_dist = route_distance(&matrix, &opt_route, true);
    let opt_time = (now_ms() - t0) / 1000.0;

    let hk_result = if n <= 12 {
        let t0 = now_ms();
        let (hk_route, hk_dist) = held_karp(&matrix);
        let hk_time = (now_ms() - t0) / 1000.0;
        Some((hk_route, hk_dist, hk_time))
    } else {
        None
    };

    let hdqts_distance = if let Ok(result) = serde_wasm_bindgen::from_value::<FinalResult>(hdqts_result) {
        result.distance
    } else {
        f64::INFINITY
    };

    let mut results: HashMap<String, serde_json::Value> = HashMap::new();
    results.insert("hdqts".to_string(), serde_json::json!({
        "distance": hdqts_distance,
        "time_seconds": hdqts_time,
        "method": "Lehmer-enhanced DAG decomposition + 2-opt"
    }));
    results.insert("nearest_neighbor".to_string(), serde_json::json!({
        "distance": nn_dist,
        "time_seconds": nn_time,
        "method": "Greedy nearest-neighbor heuristic"
    }));
    results.insert("2opt".to_string(), serde_json::json!({
        "distance": opt_dist,
        "time_seconds": opt_time,
        "method": "NN + 2-opt local search"
    }));
    if let Some((_, hk_dist, hk_time)) = hk_result {
        results.insert("held_karp".to_string(), serde_json::json!({
            "distance": hk_dist,
            "time_seconds": hk_time,
            "method": "Held-Karp exact DP (provably optimal)"
        }));
    }

    serde_wasm_bindgen::to_value(&results).unwrap_or(JsValue::NULL)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lehmer_new() {
        let decoder = LehmerDecoder::new(4);
        assert_eq!(decoder.n, 4);
        assert_eq!(decoder.k, 3);
        assert_eq!(decoder.factorials, vec![1, 1, 2, 6]);
        assert_eq!(decoder.total_permutations, 6);
    }

    #[test]
    fn test_lehmer_decode_small() {
        let decoder = LehmerDecoder::new(4);
        assert_eq!(decoder.decode_full_route(0), vec![0, 1, 2, 3]);
        assert_eq!(decoder.decode_full_route(5), vec![0, 3, 2, 1]);
    }

    #[test]
    fn test_lehmer_encode_decode_roundtrip() {
        let decoder = LehmerDecoder::new(5);
        for idx in 0..decoder.total_permutations {
            let perm = decoder.decode(idx);
            let encoded = decoder.encode(&perm);
            assert_eq!(idx, encoded, "Roundtrip failed: {} → {:?} → {}", idx, perm, encoded);
        }
    }

    #[test]
    fn test_lehmer_modulo_clamp() {
        let decoder = LehmerDecoder::new(4);
        assert_eq!(decoder.decode_full_route(6), vec![0, 1, 2, 3]);
        assert_eq!(decoder.decode_full_route(11), vec![0, 3, 2, 1]);
    }

    #[test]
    fn test_lehmer_decode_large() {
        let decoder = LehmerDecoder::new(8);
        assert_eq!(decoder.total_permutations, 5040);
        assert!(decoder.required_qubits >= 13);
        
        let route = decoder.decode_full_route(1000);
        assert_eq!(route.len(), 8);
        assert_eq!(route[0], 0);
        
        let mut seen: HashSet<usize> = HashSet::new();
        for &city in &route[1..] {
            assert!(city >= 1 && city < 8);
            assert!(seen.insert(city), "City {} appears twice", city);
        }
        assert_eq!(seen.len(), 7);
    }

    #[test]
    fn test_lehmer_compute_distance() {
        let matrix = DMatrix::from_row_slice(4, 4, &[
            0.0, 10.0, 15.0, 20.0,
            10.0, 0.0, 35.0, 25.0,
            15.0, 35.0, 0.0, 30.0,
            20.0, 25.0, 30.0, 0.0,
        ]);
        let decoder = LehmerDecoder::new(4);
        let dist = decoder.compute_distance(0, &matrix);
        assert!((dist - 95.0).abs() < 0.01, "Expected 95, got {}", dist);
    }

    #[test]
    fn test_lehmer_find_best_route() {
        let matrix = DMatrix::from_row_slice(4, 4, &[
            0.0, 10.0, 15.0, 20.0,
            10.0, 0.0, 35.0, 25.0,
            15.0, 35.0, 0.0, 30.0,
            20.0, 25.0, 30.0, 0.0,
        ]);
        let decoder = LehmerDecoder::new(4);
        let (best_idx, best_dist) = decoder.find_best_route(&matrix);
        let route = decoder.decode_full_route(best_idx);
        assert_eq!(route.len(), 4);
        assert_eq!(route[0], 0);
        let computed = decoder.compute_distance(best_idx, &matrix);
        assert!((computed - best_dist).abs() < 0.01);
    }

    #[test]
    fn test_nearest_neighbor() {
        let matrix = DMatrix::from_row_slice(4, 4, &[
            0.0, 10.0, 15.0, 20.0,
            10.0, 0.0, 35.0, 25.0,
            15.0, 35.0, 0.0, 30.0,
            20.0, 25.0, 30.0, 0.0,
        ]);
        let (route, dist) = nearest_neighbor(&matrix, 0);
        assert_eq!(route.len(), 4);
        assert_eq!(route[0], 0);
        assert!(dist > 0.0);
    }

    #[test]
    fn test_held_karp_optimal() {
        let matrix = DMatrix::from_row_slice(4, 4, &[
            0.0, 10.0, 15.0, 20.0,
            10.0, 0.0, 35.0, 25.0,
            15.0, 35.0, 0.0, 30.0,
            20.0, 25.0, 30.0, 0.0,
        ]);
        let (route, dist) = held_karp(&matrix);
        assert_eq!(route.len(), 4);
        assert_eq!(route[0], 0);
        let decoder = LehmerDecoder::new(4);
        let (_, best_dist) = decoder.find_best_route(&matrix);
        assert!((dist - best_dist).abs() < 0.01);
    }

    #[test]
    fn test_two_opt_improves() {
        let matrix = DMatrix::from_row_slice(5, 5, &[
            0.0, 10.0, 15.0, 20.0, 25.0,
            10.0, 0.0, 35.0, 25.0, 30.0,
            15.0, 35.0, 0.0, 30.0, 20.0,
            20.0, 25.0, 30.0, 0.0, 10.0,
            25.0, 30.0, 20.0, 10.0, 0.0,
        ]);
        let (nn_route, nn_dist) = nearest_neighbor(&matrix, 0);
        let opt_route = two_opt_cached(&matrix, &nn_route);
        let opt_dist = route_distance(&matrix, &opt_route, true);
        assert!(opt_dist <= nn_dist + 1e-9);
    }

    #[test]
    fn test_solve_small_instance() {
        let matrix = vec![
            0.0, 10.0, 15.0, 20.0,
            10.0, 0.0, 35.0, 25.0,
            15.0, 35.0, 0.0, 30.0,
            20.0, 25.0, 30.0, 0.0,
        ];
        let solver = HybridDAGQuantumTSP::new(matrix, 4).unwrap();
        assert_eq!(solver.n, 4);
        assert!(solver.config.enable_2opt_polish);
        let decoder = LehmerDecoder::new(4);
        let (best_idx, best_dist) = decoder.find_best_route(&solver.matrix);
        assert!(best_dist > 0.0);
        assert!(best_dist < f64::INFINITY);
        let route = decoder.decode_full_route(best_idx);
        assert_eq!(route.len(), 4);
        assert_eq!(route[0], 0);
    }

    #[test]
    fn test_decomposer_balanced() {
        let matrix_d = DMatrix::from_row_slice(8, 8, &[
            0.0, 29.0, 82.0, 46.0, 68.0, 52.0, 72.0, 42.0,
            29.0, 0.0, 55.0, 46.0, 42.0, 43.0, 43.0, 23.0,
            82.0, 55.0, 0.0, 68.0, 46.0, 55.0, 23.0, 43.0,
            46.0, 46.0, 68.0, 0.0, 82.0, 15.0, 72.0, 31.0,
            68.0, 42.0, 46.0, 82.0, 0.0, 74.0, 23.0, 52.0,
            52.0, 43.0, 55.0, 15.0, 74.0, 0.0, 61.0, 23.0,
            72.0, 43.0, 23.0, 72.0, 23.0, 61.0, 0.0, 42.0,
            42.0, 23.0, 43.0, 31.0, 52.0, 23.0, 42.0, 0.0,
        ]);
        let decomposer = AdaptiveDAGDecomposer::new(matrix_d, 4);
        let dag = decomposer.decompose();
        assert!(dag.nodes.len() >= 2);
        assert!(!dag.leaves.is_empty());
        assert_eq!(dag.roots, vec![0]);
    }

    #[test]
    fn test_tour_combiner() {
        let matrix = DMatrix::from_row_slice(6, 6, &[
            0.0, 10.0, 15.0, 20.0, 25.0, 30.0,
            10.0, 0.0, 35.0, 25.0, 30.0, 20.0,
            15.0, 35.0, 0.0, 30.0, 20.0, 25.0,
            20.0, 25.0, 30.0, 0.0, 10.0, 15.0,
            25.0, 30.0, 20.0, 10.0, 0.0, 35.0,
            30.0, 20.0, 25.0, 15.0, 35.0, 0.0,
        ]);
        let combiner = TourCombiner::new(matrix);
        let leaf_routes = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let combined = combiner.combine(&leaf_routes);
        assert_eq!(combined.len(), 6);
        assert!(combiner.validate(&combined));
    }

    #[test]
    fn test_large_instance_100_cities() {
        let mut rng = rand::thread_rng();
        let n = 100;
        let points: Vec<(f64, f64)> = (0..n)
            .map(|_| (rng.gen::<f64>() * 100.0, rng.gen::<f64>() * 100.0))
            .collect();
        
        let mut matrix_data = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let dx = points[i].0 - points[j].0;
                let dy = points[i].1 - points[j].1;
                matrix_data[i * n + j] = (dx * dx + dy * dy).sqrt();
            }
        }
        
        let solver = HybridDAGQuantumTSP::new(matrix_data, n).unwrap();
        
        // Use solve_native() — works in tests!
        let result = solver.solve_native();
        
        assert!(result.success);
        assert!(result.distance > 0.0);
        assert!(result.distance < f64::INFINITY);
        assert_eq!(result.route.len(), n + 1);
        assert_eq!(result.route[0], result.route[n]);
        
        let mut visited: HashSet<usize> = HashSet::new();
        for &city in &result.route[..n] {
            assert!(visited.insert(city), "City {} visited twice", city);
        }
        assert_eq!(visited.len(), n);
        
        let (_, nn_dist) = nearest_neighbor(&solver.matrix, 0);
        let improvement = (nn_dist - result.distance) / nn_dist * 100.0;
        
        println!("\n100-City Results:");
        println!("  NN baseline:    {:.2}", nn_dist);
        println!("  HDQTS distance: {:.2}", result.distance);
        println!("  Improvement:    {:.2}%", improvement);
        println!("  Execution time: {:.3}s", result.exec_time);
        println!("  DAG nodes:      {}", result.dag.nodes.len());
        println!("  DAG leaves:     {}", result.dag.leaves.len());
        println!("  Route valid:    YES ({} unique cities)", n);
        
        assert!(result.distance <= nn_dist * 2.0);
    }
}