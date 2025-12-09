use crate::util::{distance_type_from_str, normalize_vectors};
use cuvs::cluster::kmeans;
use cuvs::distance_type::DistanceType;
use cuvs::{ManagedTensor, Resources};
use ndarray::{Array1, Array2};
use pgrx::{info, warning};
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
    info!("⏱️ copied vec to gpu at: {:.2?}", start_time.elapsed());
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

    info!(
        "⏱️ preparing/transferring data done at: {:.2?}",
        start_time.elapsed()
    );

    info!("running kemans");
    let (inertia, n_iter) = kmeans::fit(&res, &kmeans_params, &dataset, &None, &mut centroids_gpu)
        .expect("kmeans training failed");
    info!("kmeans done with inertia: {inertia}, n_iter: {n_iter}");
    info!(
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
    info!(
        "⏱️ kmeans predict data done at: {:.2?}",
        start_time.elapsed()
    );

    info!("retrieve results from GPU");
    labels_gpu
        .to_host(&res, &mut labels_host)
        .expect("labels->host transfer failed");

    centroids_gpu
        .to_host(&res, &mut centroids_host)
        .expect("centroids->host transfer failed");
    info!(
        "⏱️ retrieved data from GPU at: {:.2?}",
        start_time.elapsed()
    );

    // calculate weights
    let mut counts = vec![0.0; num_clusters as usize];
    for &label in labels_host.iter() {
        counts[label as usize] += 1.0;
    }
    warning!("counts: {:#?}", counts);

    if spherical_centroids {
        info!("normalizing centroids");
        normalize_vectors(&mut centroids_host);
        info!("⏱️ normlaized centroids at: {:.2?}", start_time.elapsed());
    }

    let centroids_owned: Vec<f32> = centroids_host.into_raw_vec().into();

    info!(
        "\tClustering (k-means) done in: {:.2?}",
        start_time.elapsed()
    );
    (centroids_owned, counts)
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

    info!("⏱️ preparing VEC done at: {:.2?}", start_time.elapsed());

    let dataset = ManagedTensor::from(&vectors_array)
        .to_device(&res)
        .expect("vectors->tensor transformation failed");
    info!("⏱️ copied vec to gpu at: {:.2?}", start_time.elapsed());
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

    info!(
        "⏱️ preparing/transferring data done at: {:.2?}",
        start_time.elapsed()
    );

    info!("running kemans");
    let (inertia, n_iter) = kmeans::fit(
        &res,
        &kmeans_params,
        &dataset,
        &Some(weights),
        &mut centroids_gpu,
    )
    .expect("kmeans training failed");
    info!("kmeans done with inertia: {inertia}, n_iter: {n_iter}");
    info!(
        "⏱️ kmeans training data done at: {:.2?}",
        start_time.elapsed()
    );

    info!("retrieve results from GPU");

    centroids_gpu
        .to_host(&res, &mut centroids_host)
        .expect("centroids->host transfer failed");
    info!(
        "⏱️ retrieved data from GPU at: {:.2?}",
        start_time.elapsed()
    );

    if spherical_centroids {
        info!("normalizing centroids");
        normalize_vectors(&mut centroids_host);
        info!("⏱️ normlaized centroids at: {:.2?}", start_time.elapsed());
    }

    let centroids_owned: Vec<f32> = centroids_host.into_raw_vec().into();

    info!(
        "\tClustering (k-means) done in: {:.2?}",
        start_time.elapsed()
    );
    centroids_owned
}
