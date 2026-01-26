use crate::util::{distance_type_from_str, normalize_vectors};
use cuvs::cluster::kmeans;
use cuvs::distance_type::DistanceType;
use cuvs::{ManagedTensor, Resources};
use ndarray::{Array1, Array2, ArrayBase, Ix1, OwnedRepr};
use pgrx::{debug1, info, warning};
use std::process::Command;
use std::time::Instant;

// ============================================================================================
// GPU MEMORY UTILITIES
// ============================================================================================

/// Query available GPU memory using nvidia-smi.
/// Returns (free_bytes, total_bytes) or None if query fails.
fn query_gpu_memory() -> Option<(usize, usize)> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().next()?;
    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

    if parts.len() >= 2 {
        let free_mb: usize = parts[0].parse().ok()?;
        let total_mb: usize = parts[1].parse().ok()?;
        Some((free_mb * 1024 * 1024, total_mb * 1024 * 1024))
    } else {
        None
    }
}

/// Get usable GPU memory for vector data.
/// Uses 80% of free memory to leave room for centroids, labels, and cuVS workspace.
fn get_gpu_memory_budget() -> usize {
    const DEFAULT_BUDGET: usize = 10_000_000_000; // 10GB fallback
    const MEMORY_USAGE_RATIO: f64 = 0.80; // Use 80% of free memory

    match query_gpu_memory() {
        Some((free_bytes, total_bytes)) => {
            let usable = (free_bytes as f64 * MEMORY_USAGE_RATIO) as usize;
            info!("   GPU memory: {:.1}GB free / {:.1}GB total -> using {:.1}GB for vectors",
                  free_bytes as f64 / 1e9,
                  total_bytes as f64 / 1e9,
                  usable as f64 / 1e9);
            usable
        }
        None => {
            warning!("   Could not query GPU memory (nvidia-smi failed), using {:.1}GB default",
                     DEFAULT_BUDGET as f64 / 1e9);
            DEFAULT_BUDGET
        }
    }
}

// ============================================================================================
// SECTION 1: EXISTING FLAT INDEXING LOGIC
// ============================================================================================

pub fn run_clustering_batch(
    vectors: Vec<f32>,
    vector_dims: u32,
    num_clusters: u32,
    kmeans_iterations: u32,
    kmeans_nredo: u32,
    distance_operator: &str,
    spherical_centroids: bool,
) -> (Vec<f32>, Vec<f32>) {
    info!("Clustering vectors on GPU");
    let start_time = Instant::now();
    let num_vectors = vectors.len() / vector_dims as usize;
    let res = Resources::new().expect("GPU Resource creation failed");

    // Shape is (rows, cols). Rows is determined by the length of the vector input divided by dimensions.
    let vectors_array  = Array2::from_shape_vec((num_vectors, vector_dims as usize),vectors).expect("shaping vectors failed");
    let dataset = ManagedTensor::from(&vectors_array)
        .to_device(&res)
        .expect("vectors->tensor transformation failed");
    debug1!("⏱️ copied vec to gpu at: {:.2?}", start_time.elapsed());

    let mut centroids_host = Array2::<f32>::zeros((num_clusters as usize, vector_dims as usize));
    let mut centroids_gpu = ManagedTensor::from(&centroids_host)
        .to_device(&res)
        .expect("centroids(empty)->GPU transfer failed");

    let mut labels_host = Array1::<i32>::zeros(num_vectors);
    let mut labels_gpu = ManagedTensor::from(&labels_host)
        .to_device(&res)
        .expect("labels(empty)->GPU transfer failed");

    let distance_operator_cuvs = distance_type_from_str(&distance_operator)
        .expect(format!("invalid distance operator: {distance_operator}").as_str());

    let kmeans_params = kmeans::Params::new()
        .expect("kmeans params create failed")
        .set_n_clusters(num_clusters as i32)
        .set_max_iter(kmeans_iterations as i32)
        .set_n_init(kmeans_nredo as i32)
        .set_metric(distance_operator_cuvs)
        .set_hierarchical(true)
        .set_hierarchical_n_iters(kmeans_iterations as i32);

    debug1!(
        "⏱️ preparing/transferring data done at: {:.2?}",
        start_time.elapsed()
    );

    debug1!("running kmeans");
    let (inertia, n_iter) = kmeans::fit(&res, &kmeans_params, &dataset, &None, &mut centroids_gpu)
        .expect("kmeans training failed");
    debug1!("kmeans done with inertia: {inertia}, n_iter: {n_iter}");
    debug1!(
        "⏱️ kmeans training data done at: {:.2?}",
        start_time.elapsed()
    );

    let _inertia_pred = kmeans::predict(
        &res,
        &kmeans_params,
        &dataset,
        &None,
        &centroids_gpu,
        &mut labels_gpu,
        false,
    )
    .expect("kmeans prediction failed");
    debug1!(
        "⏱️ kmeans predict data done at: {:.2?}",
        start_time.elapsed()
    );

    debug1!("retrieve results from GPU");
    labels_gpu
        .to_host(&res, &mut labels_host)
        .expect("labels->host transfer failed");

    centroids_gpu
        .to_host(&res, &mut centroids_host)
        .expect("centroids->host transfer failed");
    debug1!(
        "⏱️ retrieved data from GPU at: {:.2?}",
        start_time.elapsed()
    );

    let weights = labels_to_weights(num_clusters, &labels_host);

    if spherical_centroids {
        debug1!("normalizing centroids");
        normalize_vectors(&mut centroids_host);
        debug1!("⏱️ normalized centroids at: {:.2?}", start_time.elapsed());
    }

    let centroids_owned: Vec<f32> = centroids_host.into_raw_vec().into();

    debug1!(
        "\tClustering (k-means) done in: {:.2?}",
        start_time.elapsed()
    );
    (centroids_owned, weights)
}

fn labels_to_weights(num_clusters: u32, labels_host: &ArrayBase<OwnedRepr<i32>, Ix1>) -> Vec<f32> {
    // Calculate weights based on cluster assignment counts
    let mut counts = vec![0.0; num_clusters as usize];
    for &label in labels_host.iter() {
        counts[label as usize] += 1.0;
    }
    counts
}

pub fn run_clustering_consolidate(
    vectors: Vec<f32>,
    weights: Vec<f32>,
    vector_dims: u32,
    num_clusters: u32,
    kmeans_iterations: u32,
    kmeans_nredo: u32,
    spherical_centroids: bool,
) -> Vec<f32> {
    info!("Clustering intermediate centroids on GPU");
    let start_time = Instant::now();
    let num_vectors = vectors.len() / vector_dims as usize;

    // cuvs setup
    let res = Resources::new().expect("GPU Resource creation failed");

    let vectors_array =
        Array2::from_shape_vec((num_vectors, vector_dims as usize), vectors).expect("shaping vectors failed");

    let weights_array =
        Array1::from_shape_vec(num_vectors, weights).expect("shaping vectors failed");

    let weights = ManagedTensor::from(&weights_array)
        .to_device(&res)
        .expect("weights(host)->GPU transfer failed");

    debug1!("⏱️ preparing vectors done at: {:.2?}", start_time.elapsed());

    let dataset = ManagedTensor::from(&vectors_array)
        .to_device(&res)
        .expect("vectors->tensor transformation failed");
    debug1!("⏱️ copied vectors to gpu at: {:.2?}", start_time.elapsed());

    let mut centroids_host = Array2::<f32>::zeros((num_clusters as usize, vector_dims as usize));
    let mut centroids_gpu = ManagedTensor::from(&centroids_host)
        .to_device(&res)
        .expect("centroids(empty)->GPU transfer failed");

    // Note: We use non-hierarchical kmeans here because only that supports
    // passing in weights (critical for accuracy when consolidating batches),
    // and non-hierarchical only works with L2Expanded distance.
    let kmeans_params = kmeans::Params::new()
        .expect("kmeans params create failed")
        .set_n_clusters(num_clusters as i32)
        .set_max_iter(kmeans_iterations as i32)
        .set_n_init(kmeans_nredo as i32)
        .set_metric(DistanceType::L2Expanded)
        .set_hierarchical(false);

    debug1!(
        "⏱️ preparing/transferring data done at: {:.2?}",
        start_time.elapsed()
    );

    debug1!("running kmeans");
    let (inertia, n_iter) = kmeans::fit(
        &res,
        &kmeans_params,
        &dataset,
        &Some(weights),
        &mut centroids_gpu,
    )
    .expect("kmeans training failed");
    debug1!("kmeans done with inertia: {inertia}, n_iter: {n_iter}");
    debug1!(
        "⏱️ kmeans training data done at: {:.2?}",
        start_time.elapsed()
    );

    debug1!("retrieve results from GPU");

    centroids_gpu
        .to_host(&res, &mut centroids_host)
        .expect("centroids->host transfer failed");
    debug1!(
        "⏱️ retrieved data from GPU at: {:.2?}",
        start_time.elapsed()
    );

    if spherical_centroids {
        debug1!("normalizing centroids");
        normalize_vectors(&mut centroids_host);
        debug1!("⏱️ normalized centroids at: {:.2?}", start_time.elapsed());
    }

    let centroids_owned: Vec<f32> = centroids_host.into_raw_vec().into();

    debug1!(
        "\tClustering (k-means) done in: {:.2?}",
        start_time.elapsed()
    );
    centroids_owned
}

// ============================================================================================
// SECTION 2: HIERARCHICAL CLUSTERING APPROACHES
// ============================================================================================

// --------------------------------------------------------------------------------------------
// BOTTOM-UP CLUSTERING (Recommended - matches VectorChord's approach but on GPU)
// --------------------------------------------------------------------------------------------
// Strategy: Train ALL leaf centroids in ONE GPU call, then cluster leaves into roots.
// This is fundamentally different from top-down which makes N separate GPU calls.
//
// VectorChord CPU approach:
//   - lloyd_k_means on CPU with rayon parallelism
//   - ~160k clusters from ~6.4M samples
//
// Our GPU approach (should be faster):
//   - ONE cuVS hierarchical k-means call: samples → leaves
//   - ONE cuVS k-means call: leaves → roots
//   - Reuse GPU context across all operations
// --------------------------------------------------------------------------------------------

/// Bottom-up hierarchical clustering result
pub struct BottomUpResult {
    /// Root/parent centroids (num_roots x dims)
    pub root_centroids: Vec<f32>,
    /// Leaf centroids (num_leaves x dims)
    pub leaf_centroids: Vec<f32>,
    /// Parent assignment for each leaf (which root each leaf belongs to)
    pub leaf_to_root: Vec<i32>,
}

/// Performs bottom-up hierarchical clustering on GPU.
///
/// This matches VectorChord's clustering strategy but leverages GPU acceleration.
/// If the dataset exceeds GPU memory, it processes in batches internally.
///
/// Strategy:
/// 1. If data fits GPU: Train ALL leaf centroids in one GPU call
/// 2. If data exceeds GPU: Batch process → intermediate centroids → consolidate
/// 3. Train root centroids from leaf centroids
/// 4. Assign leaves to roots
///
/// # Arguments
/// * `vectors` - Flat vector of all training data (num_vectors * dims)
/// * `vector_dims` - Dimensionality of each vector
/// * `num_leaves` - Number of leaf centroids to create (e.g., 160,000)
/// * `num_roots` - Number of root/parent centroids (e.g., 400). Pass 0 for flat index.
/// * `kmeans_iterations` - Max iterations for k-means
/// * `spherical_centroids` - Whether to L2-normalize centroids (for cosine similarity)
pub fn bottom_up(
    vectors: Vec<f32>,
    vector_dims: u32,
    num_leaves: u32,
    num_roots: u32,
    kmeans_iterations: u32,
    spherical_centroids: bool,
) -> BottomUpResult {
    let total_vectors = vectors.len() / vector_dims as usize;
    let is_hierarchical = num_roots > 0;

    info!("🚀 [BOTTOM-UP] Starting GPU hierarchical clustering");
    info!("   Input vectors: {}, Dims: {}", total_vectors, vector_dims);
    info!("   Target: {} leaves, {} roots", num_leaves, if is_hierarchical { num_roots } else { 0 });

    // Query actual GPU memory and calculate capacity
    let gpu_memory_budget = get_gpu_memory_budget();
    let bytes_per_vector = vector_dims as usize * 4;
    let max_vectors_for_gpu = gpu_memory_budget / bytes_per_vector;
    info!("   Max vectors for GPU: {} ({:.1}GB budget)",
          max_vectors_for_gpu, gpu_memory_budget as f64 / 1e9);

    let overall_start = Instant::now();

    // =========================================================================
    // Decide strategy: direct fit vs batched consolidation
    // =========================================================================
    let leaf_centroids_flat: Vec<f32> = if total_vectors <= max_vectors_for_gpu {
        // Dataset fits in GPU memory - train directly
        info!("📍 [PHASE 1] Direct GPU training ({} vectors fit in memory)", total_vectors);
        train_leaves_direct(
            vectors,
            vector_dims,
            num_leaves,
            kmeans_iterations,
            spherical_centroids,
        )
    } else {
        // Dataset exceeds GPU memory - batch process with consolidation
        let num_batches = (total_vectors + max_vectors_for_gpu - 1) / max_vectors_for_gpu;
        info!("📍 [PHASE 1] Batched GPU training ({} vectors in {} batches)",
              total_vectors, num_batches);
        train_leaves_batched(
            vectors,
            vector_dims,
            num_leaves,
            max_vectors_for_gpu,
            kmeans_iterations,
            spherical_centroids,
        )
    };

    info!("✅ [PHASE 1] Leaf training complete in {:.2?}", overall_start.elapsed());

    // =========================================================================
    // PHASE 2: Train root centroids from leaves (if hierarchical)
    // =========================================================================
    if !is_hierarchical {
        info!("🎉 [BOTTOM-UP] Flat index complete in {:.2?}", overall_start.elapsed());
        return BottomUpResult {
            root_centroids: Vec::new(),
            leaf_centroids: leaf_centroids_flat,
            leaf_to_root: vec![-1; num_leaves as usize],
        };
    }

    info!("📍 [PHASE 2] Training {} roots from {} leaves...", num_roots, num_leaves);
    let phase2_start = Instant::now();

    let res = Resources::new().expect("GPU Resource creation failed");

    let leaf_array = Array2::from_shape_vec(
        (num_leaves as usize, vector_dims as usize),
        leaf_centroids_flat.clone(),
    ).expect("Failed to reshape leaf centroids");

    let leaf_dataset_gpu = ManagedTensor::from(&leaf_array)
        .to_device(&res)
        .expect("Failed to transfer leaf centroids to GPU");

    let mut root_centroids_host = Array2::<f32>::zeros((num_roots as usize, vector_dims as usize));
    let mut root_centroids_gpu = ManagedTensor::from(&root_centroids_host)
        .to_device(&res)
        .expect("Failed to allocate root centroids on GPU");

    let root_params = kmeans::Params::new()
        .expect("Failed to create k-means params")
        .set_n_clusters(num_roots as i32)
        .set_max_iter(kmeans_iterations as i32)
        .set_metric(DistanceType::L2Expanded)
        .set_hierarchical(false);

    let (root_inertia, root_n_iter) = kmeans::fit(
        &res,
        &root_params,
        &leaf_dataset_gpu,
        &None,
        &mut root_centroids_gpu,
    ).expect("Root k-means training failed");

    info!("   K-means converged: inertia={:.2e}, iters={}", root_inertia, root_n_iter);

    root_centroids_gpu
        .to_host(&res, &mut root_centroids_host)
        .expect("Root centroids transfer failed");

    if spherical_centroids {
        normalize_vectors(&mut root_centroids_host);
    }

    info!("✅ [PHASE 2] Root training complete in {:.2?}", phase2_start.elapsed());

    // =========================================================================
    // PHASE 3: Assign each leaf to nearest root
    // =========================================================================
    info!("📍 [PHASE 3] Assigning {} leaves to {} roots...", num_leaves, num_roots);
    let phase3_start = Instant::now();

    let mut leaf_to_root_host = Array1::<i32>::zeros(num_leaves as usize);
    let mut leaf_to_root_gpu = ManagedTensor::from(&leaf_to_root_host)
        .to_device(&res)
        .expect("Failed to allocate leaf-to-root labels on GPU");

    kmeans::predict(
        &res,
        &root_params,
        &leaf_dataset_gpu,
        &None,
        &root_centroids_gpu,
        &mut leaf_to_root_gpu,
        false,
    ).expect("Leaf-to-root assignment failed");

    leaf_to_root_gpu
        .to_host(&res, &mut leaf_to_root_host)
        .expect("Leaf-to-root transfer failed");

    info!("✅ [PHASE 3] Assignment complete in {:.2?}", phase3_start.elapsed());
    info!("🎉 [BOTTOM-UP] Total clustering time: {:.2?}", overall_start.elapsed());

    BottomUpResult {
        root_centroids: root_centroids_host.into_raw_vec(),
        leaf_centroids: leaf_centroids_flat,
        leaf_to_root: leaf_to_root_host.into_raw_vec(),
    }
}

/// Train leaf centroids directly when data fits in GPU memory
fn train_leaves_direct(
    vectors: Vec<f32>,
    vector_dims: u32,
    num_leaves: u32,
    kmeans_iterations: u32,
    spherical_centroids: bool,
) -> Vec<f32> {
    let num_vectors = vectors.len() / vector_dims as usize;
    let start = Instant::now();

    let res = Resources::new().expect("GPU Resource creation failed");

    let vectors_array = Array2::from_shape_vec(
        (num_vectors, vector_dims as usize),
        vectors,
    ).expect("Failed to reshape vectors");

    let dataset_gpu = ManagedTensor::from(&vectors_array)
        .to_device(&res)
        .expect("Failed to transfer vectors to GPU");

    debug1!("   GPU transfer complete: {:.2?}", start.elapsed());

    let mut leaf_centroids_host = Array2::<f32>::zeros((num_leaves as usize, vector_dims as usize));
    let mut leaf_centroids_gpu = ManagedTensor::from(&leaf_centroids_host)
        .to_device(&res)
        .expect("Failed to allocate leaf centroids on GPU");

    // Use hierarchical k-means for large cluster counts (more efficient)
    let use_hierarchical = num_leaves > 256;
    let leaf_params = kmeans::Params::new()
        .expect("Failed to create k-means params")
        .set_n_clusters(num_leaves as i32)
        .set_max_iter(kmeans_iterations as i32)
        .set_metric(DistanceType::L2Expanded)
        .set_hierarchical(use_hierarchical)
        .set_hierarchical_n_iters(kmeans_iterations as i32);

    let (inertia, n_iter) = kmeans::fit(
        &res,
        &leaf_params,
        &dataset_gpu,
        &None,
        &mut leaf_centroids_gpu,
    ).expect("Leaf k-means training failed");

    info!("   K-means converged: inertia={:.2e}, iters={}", inertia, n_iter);

    leaf_centroids_gpu
        .to_host(&res, &mut leaf_centroids_host)
        .expect("Leaf centroids transfer failed");

    if spherical_centroids {
        normalize_vectors(&mut leaf_centroids_host);
    }

    leaf_centroids_host.into_raw_vec()
}

/// Train leaf centroids via batched processing when data exceeds GPU memory
fn train_leaves_batched(
    vectors: Vec<f32>,
    vector_dims: u32,
    num_leaves: u32,
    batch_size_vectors: usize,
    kmeans_iterations: u32,
    spherical_centroids: bool,
) -> Vec<f32> {
    let total_vectors = vectors.len() / vector_dims as usize;
    let num_batches = (total_vectors + batch_size_vectors - 1) / batch_size_vectors;

    // Target: 4x leaves as intermediate centroids for quality, spread across batches
    let total_intermediates = (num_leaves as usize).saturating_mul(4);
    let intermediates_per_batch = std::cmp::max(total_intermediates / num_batches, 64);

    info!("   Batching: {} batches, {} intermediates/batch", num_batches, intermediates_per_batch);

    let mut all_intermediates: Vec<f32> = Vec::new();
    let mut all_weights: Vec<f32> = Vec::new();

    // Process each batch
    for batch_idx in 0..num_batches {
        let start_vec = batch_idx * batch_size_vectors;
        let end_vec = std::cmp::min(start_vec + batch_size_vectors, total_vectors);
        let batch_count = end_vec - start_vec;

        let start_idx = start_vec * vector_dims as usize;
        let end_idx = end_vec * vector_dims as usize;
        let batch_vectors: Vec<f32> = vectors[start_idx..end_idx].to_vec();

        info!("   Batch {}/{}: {} vectors -> {} intermediates",
              batch_idx + 1, num_batches, batch_count, intermediates_per_batch);

        let (centroids, weights) = cluster_batch_to_intermediates(
            batch_vectors,
            vector_dims,
            intermediates_per_batch as u32,
            kmeans_iterations,
            spherical_centroids,
        );

        all_intermediates.extend(centroids);
        all_weights.extend(weights);
    }

    // Consolidate all intermediate centroids into final leaves
    let total_intermediate_count = all_intermediates.len() / vector_dims as usize;
    info!("   Consolidating {} intermediates -> {} leaves", total_intermediate_count, num_leaves);

    consolidate_to_leaves(
        all_intermediates,
        all_weights,
        vector_dims,
        num_leaves,
        kmeans_iterations,
        spherical_centroids,
    )
}

/// Cluster a batch of vectors to intermediate centroids
fn cluster_batch_to_intermediates(
    vectors: Vec<f32>,
    vector_dims: u32,
    num_clusters: u32,
    kmeans_iterations: u32,
    spherical_centroids: bool,
) -> (Vec<f32>, Vec<f32>) {
    let num_vectors = vectors.len() / vector_dims as usize;

    let res = Resources::new().expect("GPU Resource creation failed");

    let vectors_array = Array2::from_shape_vec(
        (num_vectors, vector_dims as usize),
        vectors,
    ).expect("Failed to reshape vectors");

    let dataset_gpu = ManagedTensor::from(&vectors_array)
        .to_device(&res)
        .expect("Failed to transfer vectors to GPU");

    let mut centroids_host = Array2::<f32>::zeros((num_clusters as usize, vector_dims as usize));
    let mut centroids_gpu = ManagedTensor::from(&centroids_host)
        .to_device(&res)
        .expect("Failed to allocate centroids on GPU");

    let mut labels_host = Array1::<i32>::zeros(num_vectors);
    let mut labels_gpu = ManagedTensor::from(&labels_host)
        .to_device(&res)
        .expect("Failed to allocate labels on GPU");

    // Use hierarchical for efficiency with many clusters
    let use_hierarchical = num_clusters > 256;
    let params = kmeans::Params::new()
        .expect("Failed to create k-means params")
        .set_n_clusters(num_clusters as i32)
        .set_max_iter(kmeans_iterations as i32)
        .set_metric(DistanceType::L2Expanded)
        .set_hierarchical(use_hierarchical)
        .set_hierarchical_n_iters(kmeans_iterations as i32);

    let (_inertia, _n_iter) = kmeans::fit(
        &res,
        &params,
        &dataset_gpu,
        &None,
        &mut centroids_gpu,
    ).expect("Batch k-means training failed");

    // Get labels to compute weights
    kmeans::predict(
        &res,
        &params,
        &dataset_gpu,
        &None,
        &centroids_gpu,
        &mut labels_gpu,
        false,
    ).expect("Batch k-means predict failed");

    centroids_gpu
        .to_host(&res, &mut centroids_host)
        .expect("Centroids transfer failed");

    labels_gpu
        .to_host(&res, &mut labels_host)
        .expect("Labels transfer failed");

    if spherical_centroids {
        normalize_vectors(&mut centroids_host);
    }

    // Compute weights (cluster sizes)
    let mut weights = vec![0.0f32; num_clusters as usize];
    for &label in labels_host.iter() {
        if label >= 0 && (label as usize) < weights.len() {
            weights[label as usize] += 1.0;
        }
    }

    (centroids_host.into_raw_vec(), weights)
}

/// Consolidate intermediate centroids into final leaf centroids using weighted k-means
fn consolidate_to_leaves(
    intermediates: Vec<f32>,
    weights: Vec<f32>,
    vector_dims: u32,
    num_leaves: u32,
    kmeans_iterations: u32,
    spherical_centroids: bool,
) -> Vec<f32> {
    let num_intermediates = intermediates.len() / vector_dims as usize;
    let start = Instant::now();

    let res = Resources::new().expect("GPU Resource creation failed");

    let intermediates_array = Array2::from_shape_vec(
        (num_intermediates, vector_dims as usize),
        intermediates,
    ).expect("Failed to reshape intermediates");

    let weights_array = Array1::from_shape_vec(num_intermediates, weights)
        .expect("Failed to reshape weights");

    let intermediates_gpu = ManagedTensor::from(&intermediates_array)
        .to_device(&res)
        .expect("Failed to transfer intermediates to GPU");

    let weights_gpu = ManagedTensor::from(&weights_array)
        .to_device(&res)
        .expect("Failed to transfer weights to GPU");

    let mut leaf_centroids_host = Array2::<f32>::zeros((num_leaves as usize, vector_dims as usize));
    let mut leaf_centroids_gpu = ManagedTensor::from(&leaf_centroids_host)
        .to_device(&res)
        .expect("Failed to allocate leaf centroids on GPU");

    // Non-hierarchical to support weights
    let params = kmeans::Params::new()
        .expect("Failed to create k-means params")
        .set_n_clusters(num_leaves as i32)
        .set_max_iter(kmeans_iterations as i32)
        .set_metric(DistanceType::L2Expanded)
        .set_hierarchical(false);

    let (inertia, n_iter) = kmeans::fit(
        &res,
        &params,
        &intermediates_gpu,
        &Some(weights_gpu),
        &mut leaf_centroids_gpu,
    ).expect("Consolidation k-means failed");

    info!("   Consolidation converged: inertia={:.2e}, iters={}", inertia, n_iter);

    leaf_centroids_gpu
        .to_host(&res, &mut leaf_centroids_host)
        .expect("Leaf centroids transfer failed");

    if spherical_centroids {
        normalize_vectors(&mut leaf_centroids_host);
    }

    debug1!("   Consolidation complete: {:.2?}", start.elapsed());
    leaf_centroids_host.into_raw_vec()
}

