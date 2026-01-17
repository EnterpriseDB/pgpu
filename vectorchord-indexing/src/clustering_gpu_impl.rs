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
    use_internal_hierarchy: bool,
) -> (Vec<f32>, Vec<f32>) {
    info!("Clustering vectors on GPU");
    let start_time = Instant::now();
    let num_vectors = vectors.len() / vector_dims as usize;
    let res = Resources::new().expect("GPU Resource creation failed");
    // shape is (rows, cols). rows is determined by the length of the vector input; so we divide by dimensions to get that value

    // Stage 1: GPU Transfer
    let t_gpu_transfer_start = Instant::now();
    let vectors_array =
        Array2::from_shape_vec((num_vectors, vector_dims as usize), vectors.to_vec())
            .expect("shaping vectors failed");

    let dataset = ManagedTensor::from(&vectors_array)
        .to_device(&res)
        .expect("vectors->tensor transformation failed");
    let t_gpu_transfer_elapsed = t_gpu_transfer_start.elapsed();
    debug1!("⏱️ copied vec to gpu at: {:.2?}", t_gpu_transfer_elapsed);

    // Memory Allocation
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

    // --- KMEANS PARAMETERS ---
    let mut kmeans_params = kmeans::Params::new()
        .expect("kmeans params create failed")
        .set_n_clusters(num_clusters as i32)
        .set_max_iter(kmeans_iterations as i32)
        .set_n_init(kmeans_nredo as i32)
        .set_metric(distance_operator_cuvs)
        .set_hierarchical(use_internal_hierarchy);

    // Only set internal hierarchical iters if flag is true
    if use_internal_hierarchy {
        kmeans_params = kmeans_params.set_hierarchical_n_iters(kmeans_iterations as i32);
    }

    debug1!("⏱️ preparing/transferring data done at: {:.2?}", start_time.elapsed());


    // Stage 2: KMeans Fit
    debug1!("running kemans");
    let t_fit_start = Instant::now();
    let (inertia, n_iter) = kmeans::fit(&res, &kmeans_params, &dataset, &None, &mut centroids_gpu)
        .expect("kmeans training failed");
    let t_fit_elapsed = t_fit_start.elapsed();
    debug1!("kmeans done with inertia: {inertia}, n_iter: {n_iter}");
    debug1!("⏱️ kmeans training data done at: {:.2?}", t_fit_elapsed);

    // Stage 3: KMeans Predict
    let t_predict_start = Instant::now();
    let _inertia_pred = kmeans::predict(&res, &kmeans_params, &dataset, &None, &centroids_gpu, &mut labels_gpu, false)
        .expect("kmeans prediction failed");
    let t_predict_elapsed = t_predict_start.elapsed();
    debug1!("⏱️ kmeans predict data done at: {:.2?}", t_predict_elapsed);

    // Stage 4: Retrieve results
    debug1!("retrieve results from GPU");
    let t_retrieve_start = Instant::now();
    labels_gpu.to_host(&res, &mut labels_host).expect("labels->host transfer failed");
    centroids_gpu.to_host(&res, &mut centroids_host).expect("centroids->host transfer failed");
    let t_retrieve_elapsed = t_retrieve_start.elapsed();
    debug1!("⏱️ retrieved data from GPU at: {:.2?}", t_retrieve_elapsed);

    // Stage 5: Normalization
    let t_norm_start = Instant::now();
    if spherical_centroids {
        debug1!("normalizing centroids");
        normalize_vectors(&mut centroids_host);
        debug1!("⏱️ normalized centroids at: {:.2?}", start_time.elapsed());
    }
    let t_norm_elapsed = t_norm_start.elapsed();
    let weights = labels_to_weights(num_clusters, &labels_host);
    let centroids_owned: Vec<f32> = centroids_host.into_raw_vec().into();
    debug1!("\tClustering (k-means) done in: {:.2?}", start_time.elapsed());

    // Stage 6: Summary Block
    if num_clusters > 100 {
        debug1!("==========================================================");
        debug1!("📉 FLAT BATCH CLUSTERING SUMMARY ");
        debug1!("----------------------------------------------------------");
        debug1!("GPU Data Transfer:          {:>12.2?}", t_gpu_transfer_elapsed);
        debug1!("K-Means GPU Fit:            {:>12.2?}", t_fit_elapsed);
        debug1!("K-Means GPU Predict:        {:>12.2?}", t_predict_elapsed);
        debug1!("Result Retrieval:           {:>12.2?}", t_retrieve_elapsed);
        debug1!("Normalization/Weights:      {:>12.2?}", t_norm_elapsed);
        debug1!("----------------------------------------------------------");
        debug1!("TOTAL BATCH TIME:           {:>12.2?}", start_time.elapsed());
        debug1!("==========================================================");
    }
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
    distance_operator: &str,
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
    let distance_operator_cuvs = crate::util::distance_type_from_str(&distance_operator)
        .unwrap_or(DistanceType::L2Expanded); // default to L2Expanded if unknow

    let kmeans_params = kmeans::Params::new()
        .expect("kmeans params create failed")
        .set_n_clusters(num_clusters as i32)
        .set_max_iter(kmeans_iterations as i32)
        .set_n_init(kmeans_nredo as i32)
        .set_metric(distance_operator_cuvs) // Use the dynamic metric
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
    distance_operator: &str,
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
    let distance_operator_cuvs = crate::util::distance_type_from_str(&distance_operator)
        .unwrap_or(DistanceType::L2Expanded);

    let kmeans_params = kmeans::Params::new()
        .expect("kmeans params create failed")
        .set_n_clusters(num_clusters as i32)
        .set_max_iter(kmeans_iterations as i32)
        .set_n_init(kmeans_nredo as i32)
        .set_metric(distance_operator_cuvs) // Use the dynamic metric
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
    lists: Vec<u32>,
    kmeans_iterations: u32,
    kmeans_nredo: u32,
    distance_operator: &str,
    spherical_centroids: bool,
    residual_quantization: bool,
) -> Vec<(Vec<f32>, i32)> {
    let global_start = std::time::Instant::now();
    debug1!("🚀 Starting Hierarchical Clustering (Top-Down) | RQ Mode: {} .", residual_quantization);

    let root_k = lists[0];
    let total_leaf_k = lists[1];
    let leaf_k_per_root = total_leaf_k / root_k;

    // Phase 1: Roots (Always training on absolute values)
    let p1_start = Instant::now();
    debug1!("[Phase 1] Training {} root centroids...", root_k);
    let (root_centroids_raw, _) = run_clustering_batch(vectors.clone(), vector_dims, root_k, kmeans_iterations, kmeans_nredo, distance_operator, spherical_centroids, false);
    let p1_elapsed = p1_start.elapsed();
    debug1!("✅ [Phase 1] Completed in {:.2?}", p1_elapsed);

    // Phase 2: Partitioning
    let p2_start = Instant::now();
    debug1!("[Phase 2] Partitioning data into {} buckets...", root_k);
    // Use batch logic with 0 iterations for a clean predict/partition pass
    let (_, labels) = run_clustering_batch(vectors.clone(), vector_dims, root_k, 0, 1, distance_operator, spherical_centroids, false);

    let mut buckets: Vec<Vec<f32>> = vec![Vec::new(); root_k as usize];
    let num_vectors = vectors.len() / vector_dims as usize;


    for i in 0..std::cmp::min(num_vectors, labels.len()) {
        let label = labels[i] as usize;
        if label < root_k as usize {
            let start = i * vector_dims as usize;

            if residual_quantization {
                // TRUE RQ: Subtract root before training to optimize the error
                let root_start = label * vector_dims as usize;
                let mut vec_res = vectors[start..start + vector_dims as usize].to_vec();
                let root_vec = &root_centroids_raw[root_start..root_start + vector_dims as usize];
                for k in 0..vector_dims as usize { vec_res[k] -= root_vec[k]; }
                buckets[label].extend_from_slice(&vec_res);
            } else {
                // IVFFlat: Train on absolute values
                buckets[label].extend_from_slice(&vectors[start..start + vector_dims as usize]);
            }
        }
    }
    let p2_elapsed = p2_start.elapsed();
    debug1!("✅ [Phase 2] Data partitioned in {:.2?}", p2_elapsed);

    // Phase 3: Leaves
    let p3_start = Instant::now();
    debug1!("[Phase 3] Training {} leaf nodes per bucket...", leaf_k_per_root);
    let mut results: Vec<(Vec<f32>, i32)> = Vec::with_capacity((root_k + total_leaf_k) as usize);

    // Add Roots (Level 1: Still absolute from Super Root)
    for i in 0..root_k as usize {
        let start = i * vector_dims as usize;
        results.push((root_centroids_raw[start..start + vector_dims as usize].to_vec(), -1));
    }

    for root_id in 0..root_k as usize {
        let bucket_data = &buckets[root_id];
        if bucket_data.is_empty() {
            for _ in 0..leaf_k_per_root { results.push((vec![0.0; vector_dims as usize], root_id as i32)); }
        } else {
            // Because bucket_data contains residuals (if RQ=true), these results will be residuals!
            let (leaf_centroids, _) = run_clustering_batch(bucket_data.clone(), vector_dims, leaf_k_per_root, kmeans_iterations, kmeans_nredo, distance_operator, spherical_centroids, false);
            for j in 0..leaf_k_per_root as usize {
                let start = j * vector_dims as usize;
                results.push((leaf_centroids[start..start + vector_dims as usize].to_vec(), root_id as i32));
            }
        }
        if (root_id + 1) % 20 == 0 {
            debug1!("   -> Progress: {}/{} buckets. P3 Elapsed: {:.1?}", root_id + 1, root_k, p3_start.elapsed());
        }
    }
    let p3_elapsed = p3_start.elapsed();

    // --- FINAL SUMMARY BLOCK ---
    debug1!("==========================================================");
    debug1!("📊 HIERARCHICAL CLUSTERING COMPLETE (RESIDUAL MODE)");
    debug1!("----------------------------------------------------------");
    debug1!("Phase 1 (Root Training):    {:>12.2?}", p1_elapsed);
    debug1!("Phase 2 (Residual Prep):    {:>12.2?}", p2_elapsed);
    debug1!("Phase 3 (Leaf Training):    {:>12.2?}", p3_elapsed);
    debug1!("----------------------------------------------------------");
    debug1!("Total Padded Buckets:       {:>12}", empty_buckets);
    debug1!("Total Points Clustered:     {:>12}", num_vectors);
    debug1!("Total GPU Clusters:         {:>12}", results.len());
    debug1!("OVERALL EXECUTION TIME:     {:>12.2?}", global_start.elapsed());
    debug1!("==========================================================");

    results
}