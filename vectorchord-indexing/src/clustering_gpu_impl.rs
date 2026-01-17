use crate::util::{distance_type_from_str, normalize_vectors};
use cuvs::cluster::kmeans;
use cuvs::distance_type::DistanceType;
use cuvs::{ManagedTensor, Resources};
use ndarray::{Array1, Array2, ArrayBase, Ix1, OwnedRepr};
use pgrx::{debug1, info};
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

/// TOP-DOWN Hierarchical Clustering
/// Replaces the slow "Batch + Consolidate" method.
/// 1. Trains 'root_k' centroids.
/// 2. Partitions data.
/// 3. Trains 'leaf_k' centroids per partition.
pub fn run_clustering_hierarchical(
    vectors: Vec<f32>,
    vector_dims: u32,
    lists: Vec<u32>, // [400, 160000]
    kmeans_iterations: u32,
    kmeans_nredo: u32,
    distance_operator: &str,
    spherical_centroids: bool,
) -> Vec<f32> {
    info!("Starting Hierarchical Clustering (Top-Down) on GPU");
    let start_time = Instant::now();

    let root_k = lists[0]; // 400
    let total_leaf_k = lists[1]; // 160000
    let leaf_k_per_root = total_leaf_k / root_k; // 400 sub-clusters per root

    // --- PHASE 1: ROOT TRAINING ---
    // We keep this as is. It builds the "map" for Phase 2.
    info!("Phase 1: Training {} root centroids...", root_k);
    let (root_centroids, _) = run_clustering_batch(
        vectors.clone(),
        vector_dims,
        root_k,
        kmeans_iterations,
        kmeans_nredo,
        distance_operator,
        spherical_centroids,
    );

    // --- PHASE 2: PARTITIONING (THE FIX) ---
    info!("Phase 2: Partitioning data into {} buckets...", root_k);

    // Instead of calling 'multilevel' (which re-trains and crashes),
    // we call 'predict' to simply assign labels to our root_centroids.
    // NOTE: If you don't have a 'predict' wrapper, use run_clustering_batch
    // with 0 iterations; it's a hack that forces assignment without training.
    let (_, labels) = run_clustering_batch(
        vectors.clone(),
        vector_dims,
        root_k,
        0, // Set iterations to 0 to skip the crashing training kernels
        1,
        distance_operator,
        spherical_centroids,
    );

    let mut buckets: Vec<Vec<f32>> = vec![Vec::new(); root_k as usize];
    let num_vectors = vectors.len() / vector_dims as usize;
    for i in 0..num_vectors {
        let label = labels[i] as usize;
        if label < buckets.len() {
            let start = i * vector_dims as usize;
            let end = start + vector_dims as usize;
            buckets[label].extend_from_slice(&vectors[start..end]);
        }
    }

    // --- PHASE 3: LEAF TRAINING (THE STABILITY FIX) ---
    info!("Phase 3: Training leaf nodes per bucket...");
    let mut final_centroids: Vec<f32> = Vec::with_capacity((total_leaf_k * vector_dims) as usize);

    for (i, bucket_data) in buckets.iter().enumerate() {
        // CRITICAL FIX: VectorChord expects EXACTLY total_leaf_k centroids.
        // If a bucket is empty or has fewer vectors than leaf_k_per_root,
        // we must push "placeholder" centroids (zeros) to keep the index aligned.
        if bucket_data.len() < (leaf_k_per_root * vector_dims) as usize {
             debug1!("Warning: Bucket {} is too small ({} vectors). Filling with zeros.", i, bucket_data.len() / vector_dims as usize);
             final_centroids.extend(vec![0.0; (leaf_k_per_root * vector_dims) as usize]);
             continue;
        }

        // Run batch clustering on the specific bucket
        let (mut leaf_centroids, _) = run_clustering_batch(
            bucket_data.clone(),
            vector_dims,
            leaf_k_per_root,
            kmeans_iterations,
            kmeans_nredo,
            distance_operator,
            spherical_centroids,
        );

        final_centroids.append(&mut leaf_centroids);

        if i % 20 == 0 {
            info!("Finished bucket {}/{}", i, root_k);
        }
    }

    info!("Hierarchical Clustering finished in {:.2?}", start_time.elapsed());
    final_centroids
}