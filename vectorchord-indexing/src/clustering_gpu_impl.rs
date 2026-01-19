use crate::util::{distance_type_from_str, normalize_vectors};
use cuvs::cluster::kmeans;
use cuvs::distance_type::DistanceType;
use cuvs::{ManagedTensor, Resources};
use ndarray::{Array1, Array2, ArrayBase, Ix1, OwnedRepr};
use pgrx::{debug1, info, warning};
use std::time::Instant;

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
    // shape is (rows, cols). rows is determined by the length of the vector input; so we divide by dimensions to get that value
    let vectors_array =
        Array2::from_shape_vec((num_vectors, vector_dims as usize), vectors.to_vec())
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

    debug1!(
        "⏱️ preparing/transferring data done at: {:.2?}",
        start_time.elapsed()
    );

    debug1!("running kemans");
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
        debug1!("⏱️ normlaized centroids at: {:.2?}", start_time.elapsed());
    }

    let centroids_owned: Vec<f32> = centroids_host.into_raw_vec().into();

    debug1!(
        "\tClustering (k-means) done in: {:.2?}",
        start_time.elapsed()
    );
    (centroids_owned, weights)
}

fn labels_to_weights(num_clusters: u32, labels_host: &ArrayBase<OwnedRepr<i32>, Ix1>) -> Vec<f32> {
    // calculate weights
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
    // shape is (rows, cols). rows is determined by the length of the vector input; so we divide by dimensions to get that value
    let vectors_array =
        Array2::from_shape_vec((num_vectors, vector_dims as usize), vectors.to_vec())
            .expect("shaping vectors failed");

    let weights_array =
        Array1::from_shape_vec(num_vectors, weights.to_vec()).expect("shaping vectors failed");

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

    // Note: we need to use non-hierarchical kmeans here since only that supports
    // passing in weights; which are critical for accuracy
    // and non-hiearchical only works with L2Expanded distance
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

    debug1!("running kemans");
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
        debug1!("⏱️ normlaized centroids at: {:.2?}", start_time.elapsed());
    }

    let centroids_owned: Vec<f32> = centroids_host.into_raw_vec().into();

    debug1!(
        "\tClustering (k-means) done in: {:.2?}",
        start_time.elapsed()
    );
    centroids_owned
}

/// clusters a the leaf centroids; i.e. the centroids being trained on the vectors in the table, into a set of parent centroids
/// to be used as the "top / root" level of the voronoi tree
/// the labels being assigned during prediction for from [0..(num_clusters-1)] these will be the parent IDs
/// i.e. an input centroids being assigned the label "0" belongs to the first cluster in our output
pub fn run_clustering_multilevel(
    vectors: &Vec<f32>,
    vector_dims: u32,
    num_clusters: u32,
    kmeans_iterations: u32,
    kmeans_nredo: u32,
    spherical_centroids: bool,
) -> (Vec<f32>, Vec<i32>) {
    info!("Clustering multilevel / leaf centroids on GPU");
    let start_time = Instant::now();
    let num_vectors = vectors.len() / vector_dims as usize;
    // cuvs setup
    let res = Resources::new().expect("GPU Resource creation failed");
    // shape is (rows, cols). rows is determined by the length of the vector input; so we divide by dimensions to get that value
    let vectors_array =
        Array2::from_shape_vec((num_vectors, vector_dims as usize), vectors.to_vec())
            .expect("shaping vectors failed");

    debug1!("⏱️ preparing vectors done at: {:.2?}", start_time.elapsed());

    let dataset = ManagedTensor::from(&vectors_array)
        .to_device(&res)
        .expect("vectors->tensor transformation failed");
    debug1!("⏱️ copied vectors to gpu at: {:.2?}", start_time.elapsed());
    let mut centroids_host = Array2::<f32>::zeros((num_clusters as usize, vector_dims as usize));
    let mut centroids_gpu = ManagedTensor::from(&centroids_host)
        .to_device(&res)
        .expect("centroids(empty)->GPU transfer failed");

    let mut labels_host = Array1::<i32>::zeros(num_vectors);
    let mut labels_gpu = ManagedTensor::from(&labels_host)
        .to_device(&res)
        .expect("labels(empty)->GPU transfer failed");

    // Note: we need to use non-hierarchical kmeans here since only that supports
    // passing in weights; which are critical for accuracy
    // and non-hiearchical only works with L2Expanded distance
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

    debug1!("running kemans");
    let (inertia, n_iter) = kmeans::fit(
        &res,
        &kmeans_params,
        &dataset,
        &None, // Note: we don't supply weights here on purpose. Benchmarks have shown that index accuracy drops if we use weights for this "parent clustering"
        &mut centroids_gpu,
    )
    .expect("kmeans training failed");
    debug1!("kmeans done with inertia: {inertia}, n_iter: {n_iter}");
    debug1!(
        "⏱️ kmeans training data done at: {:.2?}",
        start_time.elapsed()
    );

    // now run prediction to see into which clusters the individual vectors belong
    // these "labels" will then be used as the parent IDs in the centroids table.
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
    let labels_vec = labels_host.into_raw_vec().into();

    debug1!("retrieve results from GPU");
    //warning!("labels {:#?}", labels_vec);

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
        debug1!("⏱️ normlaized centroids at: {:.2?}", start_time.elapsed());
    }

    let centroids_owned: Vec<f32> = centroids_host.into_raw_vec().into();

    debug1!(
        "\tClustering (k-means) done in: {:.2?}",
        start_time.elapsed()
    );
    (centroids_owned, labels_vec)
}

// alfer changes

/// Helper to monitor GPU memory usage for logging
fn get_gpu_memory_info() -> (usize, usize) {
    let mut free = 0;
    let mut total = 0;
    unsafe {
        // Using raw CUDA binding to get accurate VRAM stats
        cuvs::ffi::cudaMemGetInfo(&mut free, &mut total);
    }
    (free, total)
}

/// Helper to log stage transitions with memory context
fn log_stage_start(stage: &str, details: &str) -> Instant {
    let (free, total) = get_gpu_memory_info();
    let free_gb = free as f64 / 1024.0 / 1024.0 / 1024.0;
    info!(
        "🚀 [STAGE START] {}: {} | VRAM Free: {:.2} GB / {:.2} GB",
        stage,
        details,
        free_gb,
        total as f64 / 1024.0 / 1024.0 / 1024.0
    );
    Instant::now()
}

// ============================================================================================
// PHASE 1: TRAIN ROOTS (Coarse Quantizer)
// ============================================================================================

/// Trains the top-level "Root" centroids (e.g., 400) using a safe subset of the data.
///
/// Strategy:
/// - We cap the training data at 1M vectors. Training 400 clusters on 1M vectors
///   provides a 2500:1 ratio, which is statistically perfect.
/// - Using 100M vectors here is wasteful and causes GPU OOM/Integer overflows.
pub fn train_roots_gpu(
    full_vectors: &Vec<f32>,
    vector_dims: u32,
    num_roots: u32,
    iterations: u32,
    distance_mode: &str, // "ip" or "l2"
) -> Vec<f32> {
    let total_count = full_vectors.len() / vector_dims as usize;

    // SAFETY: Cap training at 1M vectors to ensure RAFT stability
    let train_limit = 1_000_000;
    let num_train = std::cmp::min(total_count, train_limit);

    let start = log_stage_start("PHASE 1 (Roots)", &format!("Training {} roots on {} sampled vectors", num_roots, num_train));

    let res = Resources::new().expect("Failed to acquire GPU resources");

    // 1. Prepare Subset
    // We slice the host vector directly.
    let train_slice = &full_vectors[..(num_train * vector_dims as usize)];
    let train_array = Array2::from_shape_vec(
        (num_train, vector_dims as usize),
        train_slice.to_vec()
    ).expect("Failed to reshape training sample");

    let dataset = ManagedTensor::from(&train_array).to_device(&res).expect("Failed to move training sample to GPU");

    let mut centroids_gpu = ManagedTensor::from(&Array2::<f32>::zeros((num_roots as usize, vector_dims as usize)))
        .to_device(&res).expect("Failed to allocate centroids buffer");

    // 2. Configure Parameters
    // Note: Even for "ip" (Inner Product), we use L2 K-Means on normalized vectors.
    // L2 distance on the unit sphere is mathematically equivalent to Cosine/IP for clustering.
    let metric = if distance_mode == "ip" || distance_mode == "cosine" {
        DistanceType::L2Expanded
    } else {
        DistanceType::L2Expanded
    };

    let params = kmeans::Params::new().expect("Params failed")
        .set_n_clusters(num_roots as i32)
        .set_max_iter(iterations as i32)
        .set_metric(metric)
        .set_n_init(1) // 1 redo is usually enough for roots
        // Force batching to 0 to let RAFT auto-tune for small N
        .set_batch_samples(0)
        .set_batch_centroids(0);

    // 3. Fit
    info!("⚙️ [GPU Kernel] Running KMeans::fit (k={})", num_roots);
    kmeans::fit(&res, &params, &dataset, &None, &mut centroids_gpu)
        .expect("Root training failed");

    // 4. Retrieve
    let mut centroids_host = Array2::<f32>::zeros((num_roots as usize, vector_dims as usize));
    centroids_gpu.to_host(&res, &mut centroids_host).expect("Failed to retrieve roots");

    info!("✅ [PHASE 1 COMPLETE] Time: {:.2?}", start.elapsed());
    centroids_host.into_raw_vec()
}

// ============================================================================================
// PHASE 2: PARTITION (Assign 100M vectors to Roots)
// ============================================================================================

/// Assigns ALL vectors to their nearest root centroid.
///
/// Strategy:
/// - We cannot load 100M vectors into GPU at once.
/// - We process in batches (e.g., 2M vectors at a time).
/// - Returns a parallel Vec<i32> of labels corresponding to the input vectors.
pub fn assign_to_roots_gpu(
    all_vectors: &Vec<f32>,
    root_centroids: &Vec<f32>,
    vector_dims: u32,
    num_roots: u32,
) -> Vec<i32> {
    let total_vectors = all_vectors.len() / vector_dims as usize;
    let start = log_stage_start("PHASE 2 (Partition)", &format!("Assigning {} vectors to {} roots", total_vectors, num_roots));

    let res = Resources::new().expect("GPU Resource failed");

    // Convert roots to GPU once
    let roots_array = Array2::from_shape_vec((num_roots as usize, vector_dims as usize), root_centroids.clone())
        .expect("Roots shape mismatch");
    let roots_gpu = ManagedTensor::from(&roots_array).to_device(&res).expect("Roots transfer failed");

    // Output buffer for labels
    let mut final_labels = Vec::with_capacity(total_vectors);

    // BATCHING CONFIGURATION
    // 2M vectors * 768 dims * 4 bytes ~= 6GB VRAM per batch. Safe for RTX 6000.
    let batch_size = 2_000_000;
    let mut processed = 0;

    let params = kmeans::Params::new().expect("Params failed")
        .set_n_clusters(num_roots as i32)
        .set_metric(DistanceType::L2Expanded);

    while processed < total_vectors {
        let end = std::cmp::min(processed + batch_size, total_vectors);
        let batch_len = end - processed;

        info!("🔮 [Batch Process] Processing vectors {} to {} ({:.1}%)", processed, end, (processed as f64 / total_vectors as f64) * 100.0);

        // Slice batch
        let slice_start = processed * vector_dims as usize;
        let slice_end = end * vector_dims as usize;
        let batch_slice = &all_vectors[slice_start..slice_end];

        let batch_array = Array2::from_shape_vec((batch_len, vector_dims as usize), batch_slice.to_vec())
            .expect("Batch shape mismatch");

        let batch_gpu = ManagedTensor::from(&batch_array).to_device(&res).expect("Batch transfer failed");
        let mut labels_host = Array1::<i32>::zeros(batch_len);
        let mut labels_gpu = ManagedTensor::from(&labels_host).to_device(&res).expect("Labels alloc failed");

        // Predict
        kmeans::predict(&res, &params, &batch_gpu, &None, &roots_gpu, &mut labels_gpu, false)
            .expect("Predict batch failed");

        // Copy back
        labels_gpu.to_host(&res, &mut labels_host).expect("Labels retrieval failed");
        final_labels.extend(labels_host.into_iter());

        processed += batch_len;
    }

    info!("✅ [PHASE 2 COMPLETE] Time: {:.2?}", start.elapsed());
    final_labels
}

// ============================================================================================
// PHASE 3: TRAIN LEAVES (Refine each partition)
// ============================================================================================

/// Trains leaf centroids for a single bucket (partition).
///
/// Strategy:
/// - This function is called repeatedly (once per root).
/// - The input `bucket_vectors` is usually small (~250k vectors for 100M/400).
/// - We run standard K-Means here to find the fine-grained clusters.
pub fn train_leaves_for_bucket_gpu(
    bucket_vectors: &Vec<f32>,
    vector_dims: u32,
    num_leaves_this_bucket: u32,
    iterations: u32,
) -> Vec<f32> {
    // Edge case: If bucket is too small, return vectors as centroids or handle gracefully
    let num_vecs = bucket_vectors.len() / vector_dims as usize;
    if num_vecs < num_leaves_this_bucket as usize {
        warning!("⚠️ Bucket has fewer vectors ({}) than requested clusters ({}). Using input vectors as centroids.", num_vecs, num_leaves_this_bucket);
        // Pad with zeros or return what we have. For now, return what we have (this is rare in 100M datasets).
        return bucket_vectors.clone();
    }

    let res = Resources::new().expect("GPU Resource failed");

    let dataset_array = Array2::from_shape_vec((num_vecs, vector_dims as usize), bucket_vectors.clone())
        .expect("Bucket shape failed");
    let dataset = ManagedTensor::from(&dataset_array).to_device(&res).expect("Bucket xfer failed");

    let mut centroids_gpu = ManagedTensor::from(&Array2::<f32>::zeros((num_leaves_this_bucket as usize, vector_dims as usize)))
        .to_device(&res).expect("Leaf alloc failed");

    let params = kmeans::Params::new().expect("Params failed")
        .set_n_clusters(num_leaves_this_bucket as i32)
        .set_max_iter(iterations as i32)
        .set_metric(DistanceType::L2Expanded);

    kmeans::fit(&res, &params, &dataset, &None, &mut centroids_gpu)
        .expect("Leaf fit failed");

    let mut centroids_host = Array2::<f32>::zeros((num_leaves_this_bucket as usize, vector_dims as usize));
    centroids_gpu.to_host(&res, &mut centroids_host).expect("Leaf retrieval failed");

    centroids_host.into_raw_vec()
}