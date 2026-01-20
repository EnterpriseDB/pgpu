use crate::util::{distance_type_from_str, normalize_vectors};
use cuvs::cluster::kmeans;
use cuvs::distance_type::DistanceType;
use cuvs::{ManagedTensor, Resources};
use ndarray::{Array1, Array2, ArrayBase, Ix1, OwnedRepr};
use pgrx::{debug1, info, warning};
use std::time::Instant;

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

/// Clusters the leaf centroids (i.e., the centroids trained on the vectors in the table)
/// into a set of parent centroids to be used as the "top / root" level of the Voronoi tree.
/// The labels assigned during prediction range from [0..(num_clusters-1)].
/// These will serve as the Parent IDs.
// Currently not used but kept for potential future use.
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

    // Note: We use non-hierarchical kmeans here because only that supports
    // passing in weights, and non-hierarchical only works with L2Expanded distance.
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
        &None, // Note: We don't supply weights here. Benchmarks show accuracy drops if we use weights for "parent clustering".
        &mut centroids_gpu,
    )
    .expect("kmeans training failed");
    debug1!("kmeans done with inertia: {inertia}, n_iter: {n_iter}");
    debug1!(
        "⏱️ kmeans training data done at: {:.2?}",
        start_time.elapsed()
    );

    // Run prediction to assign individual vectors to clusters.
    // These "labels" will be used as the Parent IDs in the centroids table.
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
    (centroids_owned, labels_vec)
}

// ============================================================================================
// SECTION 2: NEW HIERARCHICAL INDEXING LOGIC (Top-Down)
// ============================================================================================

pub fn train_roots_gpu(
    full_vectors: &Vec<f32>,
    vector_dims: u32,
    num_roots: u32,
    iterations: u32,
    n_redo: u32,
) -> Vec<f32> {
    // 1. Define Total Vectors
    let total_vectors = full_vectors.len() / vector_dims as usize;
    let train_limit = 1_000_000;

    // 2. Calculate Stride
    let num_train = std::cmp::min(total_vectors, train_limit);
    let stride = if total_vectors > train_limit {
        total_vectors / train_limit
    } else {
        1
    };

    info!("🚀 [PHASE 1 START] Training {} Roots on {} sampled vectors (Subsampled from {} with stride {} )",
          num_roots, num_train, total_vectors, stride);

    let start = Instant::now();
    let res = Resources::new().expect("GPU Resource failed");

    // 3. Create Training Buffer
    let mut train_data: Vec<f32> = Vec::with_capacity(num_train * vector_dims as usize);

    // 4. Strided Copy Loop
    for i in 0..num_train {
        let src_idx = (i * stride) * vector_dims as usize;

        // Safety check
        if src_idx + vector_dims as usize > full_vectors.len() {
            break;
        }

        let vector_slice = &full_vectors[src_idx..src_idx + vector_dims as usize];
        train_data.extend_from_slice(vector_slice);
    }

    let actual_train_count = train_data.len() / vector_dims as usize;

    // 5. Create Array
    let train_array = Array2::from_shape_vec(
        (actual_train_count, vector_dims as usize),
        train_data
    ).expect("reshape failed");

    let dataset = ManagedTensor::from(&train_array).to_device(&res).expect("xfer failed");
    let mut centroids_gpu = ManagedTensor::from(
        &Array2::<f32>::zeros((num_roots as usize, vector_dims as usize))
    ).to_device(&res).expect("alloc failed");

    let params = kmeans::Params::new().expect("params failed")
        .set_n_clusters(num_roots as i32)
        .set_max_iter(iterations as i32)
        .set_metric(DistanceType::L2Expanded)
        .set_n_init(n_redo as i32)
        .set_batch_samples(0)
        .set_batch_centroids(0);

    kmeans::fit(&res, &params, &dataset, &None, &mut centroids_gpu).expect("fit failed");

    let mut centroids_host = Array2::<f32>::zeros((num_roots as usize, vector_dims as usize));
    centroids_gpu.to_host(&res, &mut centroids_host).expect("retrieval failed");

    info!("✅ [PHASE 1 DONE] Roots trained in {:.2?}", start.elapsed());
    centroids_host.into_raw_vec()
}

pub fn assign_to_roots_gpu(
    all_vectors: &Vec<f32>,
    root_centroids: &Vec<f32>,
    vector_dims: u32,
    num_roots: u32,
) -> Vec<i32> {
    let total_vectors = all_vectors.len() / vector_dims as usize;
    info!("🚀 [PHASE 2 START] Partitioning {} vectors into {} buckets...", total_vectors, num_roots);
    let start = Instant::now();
    let res = Resources::new().expect("GPU Resource failed");

    let roots_array = Array2::from_shape_vec((num_roots as usize, vector_dims as usize), root_centroids.clone()).expect("shape failed");
    let roots_gpu = ManagedTensor::from(&roots_array).to_device(&res).expect("xfer failed");

    let mut final_labels = Vec::with_capacity(total_vectors);
    let batch_size = 2_000_000;
    let mut processed = 0;

    let params = kmeans::Params::new().expect("params failed")
        .set_n_clusters(num_roots as i32)
        .set_metric(DistanceType::L2Expanded);

    while processed < total_vectors {
        let end = std::cmp::min(processed + batch_size, total_vectors);
        let current_batch_len = end - processed;

        if processed > 0 && processed % 10_000_000 == 0 {
             info!("📊 [PHASE 2] Partitioned {}/{} vectors ({:.1}%) - Elapsed: {:.2?}  ",
                 processed, total_vectors, (processed as f64 / total_vectors as f64) * 100.0, start.elapsed());
        }

        let slice_start = processed * vector_dims as usize;
        let slice_end = end * vector_dims as usize;
        let batch_slice = &all_vectors[slice_start..slice_end];
        let batch_array = Array2::from_shape_vec((current_batch_len, vector_dims as usize), batch_slice.to_vec()).expect("reshape failed");

        let batch_gpu = ManagedTensor::from(&batch_array).to_device(&res).expect("xfer failed");
        let mut labels_host = Array1::<i32>::zeros(current_batch_len);
        let mut labels_gpu = ManagedTensor::from(&labels_host).to_device(&res).expect("alloc failed");

        kmeans::predict(&res, &params, &batch_gpu, &None, &roots_gpu, &mut labels_gpu, false).expect("predict failed");

        labels_gpu.to_host(&res, &mut labels_host).expect("retrieval failed");
        final_labels.extend(labels_host.into_iter());
        processed += current_batch_len;
    }

    info!("✅ [PHASE 2 DONE] Partitioning complete in {:.2?}", start.elapsed());
    final_labels
}

pub fn train_leaves_for_bucket_gpu(
    bucket_vectors: &Vec<f32>,
    vector_dims: u32,
    num_leaves_this_bucket: u32,
    iterations: u32,
) -> Vec<f32> {
    let num_vecs = bucket_vectors.len() / vector_dims as usize;
    if num_vecs < num_leaves_this_bucket as usize {
        warning!("⚠️ Bucket too small ({} vectors < {} leaves). Returning raw vectors.", num_vecs, num_leaves_this_bucket);
        return bucket_vectors.clone();
    }

    let res = Resources::new().expect("GPU Resource failed");
    let dataset_array = Array2::from_shape_vec((num_vecs, vector_dims as usize), bucket_vectors.clone()).expect("reshape failed");
    let dataset = ManagedTensor::from(&dataset_array).to_device(&res).expect("xfer failed");
    let mut centroids_gpu = ManagedTensor::from(&Array2::<f32>::zeros((num_leaves_this_bucket as usize, vector_dims as usize))).to_device(&res).expect("alloc failed");

    let params = kmeans::Params::new().expect("params failed")
        .set_n_clusters(num_leaves_this_bucket as i32)
        .set_max_iter(iterations as i32)
        .set_metric(DistanceType::L2Expanded)
        .set_batch_samples(0)
        .set_batch_centroids(0);

    kmeans::fit(&res, &params, &dataset, &None, &mut centroids_gpu).expect("fit failed");

    let mut centroids_host = Array2::<f32>::zeros((num_leaves_this_bucket as usize, vector_dims as usize));
    centroids_gpu.to_host(&res, &mut centroids_host).expect("retrieval failed");

    centroids_host.into_raw_vec()
}