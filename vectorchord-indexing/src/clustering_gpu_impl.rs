use crate::util::{distance_type_from_str, normalize_vectors};
use cuvs::cluster::kmeans;
use cuvs::distance_type::DistanceType;
use cuvs::{ManagedTensor, Resources};
use ndarray::{Array1, Array2, ArrayBase, ArrayView2, Ix1, OwnedRepr};
use pgrx::{debug1, info};
use std::process::Command;
use std::time::Instant;

// ============================================================================================
// GPU MEMORY UTILITIES
// ============================================================================================

/// Query available GPU memory and info using nvidia-smi.
/// Returns (free_bytes, total_bytes, gpu_index, gpu_name)
fn query_gpu_info() -> Option<(usize, usize, String, String)> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=index,name,memory.free,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().next()?;
    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

    if parts.len() >= 4 {
        let gpu_index = parts[0].to_string();
        let gpu_name = parts[1].to_string();
        let free_mb: usize = parts[2].parse().ok()?;
        let total_mb: usize = parts[3].parse().ok()?;
        Some((free_mb * 1024 * 1024, total_mb * 1024 * 1024, gpu_index, gpu_name))
    } else {
        None
    }
}

/// Get GPU memory info for logging
fn log_gpu_memory() {
    if let Some((free, total, idx, name)) = query_gpu_info() {
        info!("   GPU[{}] {}: {:.1}GB free / {:.1}GB total",
              idx, name, free as f64 / 1e9, total as f64 / 1e9);
    }
}

/// Create a new GPU Resources handle for reuse across multiple operations.
pub fn create_gpu_resources() -> Resources {
    Resources::new().expect("GPU Resource creation failed")
}

// ============================================================================================
// SECTION 1: FLAT INDEXING LOGIC (for single-level indexes)
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

    let vectors_array = Array2::from_shape_vec((num_vectors, vector_dims as usize), vectors)
        .expect("shaping vectors failed");
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

    debug1!("⏱️ preparing/transferring data done at: {:.2?}", start_time.elapsed());

    debug1!("running kmeans");
    let (inertia, n_iter) = kmeans::fit(&res, &kmeans_params, &dataset, &None, &mut centroids_gpu)
        .expect("kmeans training failed");
    debug1!("kmeans done with inertia: {inertia}, n_iter: {n_iter}");
    debug1!("⏱️ kmeans training data done at: {:.2?}", start_time.elapsed());

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
    debug1!("⏱️ kmeans predict data done at: {:.2?}", start_time.elapsed());

    debug1!("retrieve results from GPU");
    labels_gpu
        .to_host(&res, &mut labels_host)
        .expect("labels->host transfer failed");

    centroids_gpu
        .to_host(&res, &mut centroids_host)
        .expect("centroids->host transfer failed");
    debug1!("⏱️ retrieved data from GPU at: {:.2?}", start_time.elapsed());

    let weights = labels_to_weights(num_clusters, &labels_host);

    if spherical_centroids {
        debug1!("normalizing centroids");
        normalize_vectors(&mut centroids_host);
        debug1!("⏱️ normalized centroids at: {:.2?}", start_time.elapsed());
    }

    let centroids_owned: Vec<f32> = centroids_host.into_raw_vec().into();

    debug1!("\tClustering (k-means) done in: {:.2?}", start_time.elapsed());
    (centroids_owned, weights)
}

fn labels_to_weights(num_clusters: u32, labels_host: &ArrayBase<OwnedRepr<i32>, Ix1>) -> Vec<f32> {
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

    let res = Resources::new().expect("GPU Resource creation failed");

    let vectors_array = Array2::from_shape_vec((num_vectors, vector_dims as usize), vectors)
        .expect("shaping vectors failed");

    let weights_array = Array1::from_shape_vec(num_vectors, weights)
        .expect("shaping vectors failed");

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

    let kmeans_params = kmeans::Params::new()
        .expect("kmeans params create failed")
        .set_n_clusters(num_clusters as i32)
        .set_max_iter(kmeans_iterations as i32)
        .set_n_init(kmeans_nredo as i32)
        .set_metric(DistanceType::L2Expanded)
        .set_hierarchical(false);

    debug1!("⏱️ preparing/transferring data done at: {:.2?}", start_time.elapsed());

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
    debug1!("⏱️ kmeans training data done at: {:.2?}", start_time.elapsed());

    debug1!("retrieve results from GPU");

    centroids_gpu
        .to_host(&res, &mut centroids_host)
        .expect("centroids->host transfer failed");
    debug1!("⏱️ retrieved data from GPU at: {:.2?}", start_time.elapsed());

    if spherical_centroids {
        debug1!("normalizing centroids");
        normalize_vectors(&mut centroids_host);
        debug1!("⏱️ normalized centroids at: {:.2?}", start_time.elapsed());
    }

    let centroids_owned: Vec<f32> = centroids_host.into_raw_vec().into();

    debug1!("\tClustering (k-means) done in: {:.2?}", start_time.elapsed());
    centroids_owned
}

// ============================================================================================
// SECTION 2: TOP-DOWN HIERARCHICAL CLUSTERING
// ============================================================================================
// Strategy:
// 1. Train root centroids from a sample of vectors
// 2. Assign ALL vectors to their nearest root
// 3. For each root bucket, train leaf centroids
//
// This approach makes many small k-means calls instead of one huge call,
// which is more memory efficient and often faster.
// ============================================================================================

/// Train root centroids and assign all vectors to them.
/// Returns (root_centroids, assignments, root_training_duration, assignment_duration).
///
/// This combined function keeps centroids on GPU between fit() and predict(),
/// which is required for cuVS kmeans::predict to work correctly.
pub fn train_roots_and_assign_gpu(
    full_vectors: &[f32],
    vector_dims: u32,
    num_roots: u32,
    iterations: u32,
    spherical_centroids: bool,
) -> (Vec<f32>, Vec<i32>, std::time::Duration, std::time::Duration) {
    let total_vectors = full_vectors.len() / vector_dims as usize;
    let train_limit = 2_000_000; // Sample up to 2M vectors for root training

    let num_train = std::cmp::min(total_vectors, train_limit);
    let stride = if total_vectors > train_limit {
        total_vectors / train_limit
    } else {
        1
    };

    info!("🚀 [PHASE 1] Training {} roots from {} vectors (stride={} )",
          num_roots, num_train, stride);
    log_gpu_memory();

    let start = Instant::now();
    let res = Resources::new().expect("GPU Resource failed");

    // Create training buffer with strided sampling
    let mut train_data: Vec<f32> = Vec::with_capacity(num_train * vector_dims as usize);
    for i in 0..num_train {
        let src_idx = (i * stride) * vector_dims as usize;
        if src_idx + vector_dims as usize > full_vectors.len() {
            break;
        }
        train_data.extend_from_slice(&full_vectors[src_idx..src_idx + vector_dims as usize]);
    }

    let actual_train_count = train_data.len() / vector_dims as usize;

    let train_array = Array2::from_shape_vec(
        (actual_train_count, vector_dims as usize),
        train_data
    ).expect("reshape failed");

    let dataset = ManagedTensor::from(&train_array)
        .to_device(&res)
        .expect("transfer failed");

    let mut centroids_host = Array2::<f32>::zeros((num_roots as usize, vector_dims as usize));
    let mut centroids_gpu = ManagedTensor::from(&centroids_host)
        .to_device(&res)
        .expect("alloc failed");

    let params = kmeans::Params::new()
        .expect("params failed")
        .set_n_clusters(num_roots as i32)
        .set_max_iter(iterations as i32)
        .set_metric(DistanceType::L2Expanded)
        .set_n_init(1)
        .set_batch_samples(0)
        .set_batch_centroids(0);

    let (inertia, n_iter) = kmeans::fit(&res, &params, &dataset, &None, &mut centroids_gpu)
        .expect("k-means fit failed");

    let d_roots = start.elapsed();
    info!("✅ [PHASE 1] Roots trained in {:.2?} (inertia={:.2e}, iters={})",
          d_roots, inertia, n_iter);

    // ========================================================================
    // Normalize centroids BEFORE assignment (if spherical)
    // This ensures assignment uses the same centroids that will be stored/queried
    // ========================================================================
    let centroids_for_predict = if spherical_centroids {
        // Transfer to host, normalize, create new tensor
        centroids_gpu
            .to_host(&res, &mut centroids_host)
            .expect("retrieval failed");
        normalize_vectors(&mut centroids_host);

        // Create new tensor with normalized centroids
        ManagedTensor::from(&centroids_host)
            .to_device(&res)
            .expect("normalized centroids transfer failed")
    } else {
        // Use original tensor as-is (move ownership)
        centroids_gpu
    };

    // ========================================================================
    // PHASE 2: Assign all vectors to normalized centroids
    // ========================================================================
    info!("🚀 [PHASE 2] Assigning {} vectors to {} roots...", total_vectors, num_roots);
    let assign_start = Instant::now();

    let mut final_labels = Vec::with_capacity(total_vectors);
    let batch_size = 2_000_000; // Limited by cuVS internal memory allocation
    let mut processed = 0;

    while processed < total_vectors {
        let end = std::cmp::min(processed + batch_size, total_vectors);
        let current_batch_len = end - processed;

        if processed > 0 && processed % 10_000_000 == 0 {
            info!("   ... assigned {}/{} ({:.1}%)",
                  processed, total_vectors, 100.0 * processed as f64 / total_vectors as f64);
        }

        let slice_start = processed * vector_dims as usize;
        let slice_end = end * vector_dims as usize;
        let batch_slice = &full_vectors[slice_start..slice_end];

        // Use ArrayView to avoid copying - just borrow the slice
        let batch_view = ArrayView2::from_shape(
            (current_batch_len, vector_dims as usize),
            batch_slice
        ).expect("reshape failed");

        let batch_gpu = ManagedTensor::from(&batch_view)
            .to_device(&res)
            .expect("transfer failed");

        let mut labels_host = Array1::<i32>::zeros(current_batch_len);
        let mut labels_gpu = ManagedTensor::from(&labels_host)
            .to_device(&res)
            .expect("alloc failed");

        kmeans::predict(&res, &params, &batch_gpu, &None, &centroids_for_predict, &mut labels_gpu, false)
            .expect("predict failed");

        labels_gpu
            .to_host(&res, &mut labels_host)
            .expect("retrieval failed");

        final_labels.extend(labels_host.into_iter());
        processed += current_batch_len;
    }

    let d_assign = assign_start.elapsed();
    info!("✅ [PHASE 2] Assignment complete in {:.2?}", d_assign);

    // If spherical_centroids, centroids_host is already normalized from before predict
    // If not, retrieve from GPU now
    if !spherical_centroids {
        centroids_for_predict
            .to_host(&res, &mut centroids_host)
            .expect("retrieval failed");
    }

    (centroids_host.into_raw_vec(), final_labels, d_roots, d_assign)
}

/// Train leaf centroids for a single bucket using pre-created Resources.
/// Takes vector indices and the full dataset to avoid copying vectors into buckets.
pub fn train_leaves_for_bucket_gpu(
    res: &Resources,
    all_vectors: &[f32],
    bucket_indices: &[usize],
    vector_dims: u32,
    num_leaves: u32,
    iterations: u32,
    spherical_centroids: bool,
) -> Vec<f32> {
    let num_vecs = bucket_indices.len();

    if num_vecs < num_leaves as usize {
        // Return the actual vectors for small buckets (no k-means needed)
        let mut result = Vec::with_capacity(num_vecs * vector_dims as usize);
        for &idx in bucket_indices {
            let start = idx * vector_dims as usize;
            let end = start + vector_dims as usize;
            result.extend_from_slice(&all_vectors[start..end]);
        }
        return result;
    }

    if num_vecs == 0 {
        return Vec::new();
    }

    // Gather vectors for this bucket (only copy what we need for this bucket)
    let mut bucket_data: Vec<f32> = Vec::with_capacity(num_vecs * vector_dims as usize);
    for &idx in bucket_indices {
        let start = idx * vector_dims as usize;
        let end = start + vector_dims as usize;
        bucket_data.extend_from_slice(&all_vectors[start..end]);
    }

    let dataset_array = Array2::from_shape_vec(
        (num_vecs, vector_dims as usize),
        bucket_data
    ).expect("reshape failed");

    let dataset = ManagedTensor::from(&dataset_array)
        .to_device(res)
        .expect("transfer failed");

    let mut centroids_host = Array2::<f32>::zeros((num_leaves as usize, vector_dims as usize));
    let mut centroids_gpu = ManagedTensor::from(&centroids_host)
        .to_device(res)
        .expect("alloc failed");

    let params = kmeans::Params::new()
        .expect("params failed")
        .set_n_clusters(num_leaves as i32)
        .set_max_iter(iterations as i32)
        .set_metric(DistanceType::L2Expanded)
        .set_batch_samples(0)
        .set_batch_centroids(0);

    kmeans::fit(res, &params, &dataset, &None, &mut centroids_gpu)
        .expect("k-means fit failed");

    centroids_gpu
        .to_host(res, &mut centroids_host)
        .expect("retrieval failed");

    if spherical_centroids {
        normalize_vectors(&mut centroids_host);
    }

    centroids_host.into_raw_vec()
}

