use crate::util::distance_type_from_str;
use cuvs::cluster::kmeans;
use cuvs::distance_type::DistanceType;
use cuvs::{ManagedTensor, Resources};
use ndarray::{Array, Array1, Array2, Axis};

use pgrx::{info, warning};
use std::time::Instant;
use log::warn;

pub fn run_clustering(
    vectors: Vec<f32>,
    vector_dims: u32,
    cluster_count: u32,
    kmeans_iterations: u32,
    kmeans_nredo: u32,
    distance_operator: &str,
    spherical_centroids: bool,
) -> Vec<f32> {
    info!("Clustering vectors on GPU");
    let start_time = Instant::now();
    let num_vectors = vectors.len() / vector_dims as usize;
    // cuvs setup
    let res = Resources::new().expect("GPU Resource creation failed");
    // shape is (rows, cols). rows is determined by the length of the vector input; so we divide by dimensions to get that value
    let vectors_array = Array2::from_shape_vec(
        (num_vectors, vector_dims as usize),
        vectors.to_vec(),
    )
        .expect("shaping vectors failed");

    info!(
        "⏱️ preparing VEC done at: {:.2?}",
        start_time.elapsed()
    );

    let dataset = ManagedTensor::from(&vectors_array)
        .to_device(&res)
        .expect("vectors->tensor transformation failed");
    info!(
        "⏱️ copied vec to gpu at: {:.2?}",
        start_time.elapsed()
    );
    let mut centroids_host =
        Array2::<f32>::zeros((cluster_count as usize, vector_dims as usize));
    let mut centroids_gpu = ManagedTensor::from(&centroids_host)
        .to_device(&res)
        .expect("centroids(empty)->GPU transfer failed");

    let mut labels_host = Array1::<i32>::zeros(num_vectors);
    let mut labels_gpu = ManagedTensor::from(&labels_host).to_device(&res).expect("labels(empty)->GPU transfer failed");

    let distance_operator_cuvs = distance_type_from_str(&distance_operator)
        .expect(format!("invalid distance operator: {distance_operator}").as_str());
    let balanced_kmeans = match distance_operator_cuvs {
        DistanceType::InnerProduct | DistanceType::CosineExpanded => true,
        _ => false,
    };

    info!(
        "using distance {:#?} with balanced kmeans: {:?}",
        distance_operator_cuvs,
        balanced_kmeans
    );

    let kmeans_params = kmeans::Params::new()
        .expect("kmeans params create failed")
        .set_n_clusters(cluster_count as i32)
        .set_max_iter(kmeans_iterations as i32)
        .set_n_init(kmeans_nredo as i32)
        .set_metric(distance_operator_cuvs)
        .set_hierarchical(balanced_kmeans)
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
    let _inertia_pred = kmeans::predict(&res, &kmeans_params, &dataset, &None, &centroids_gpu, &mut labels_gpu, true).expect("kmeans prediction failed");
    info!(
        "⏱️ kmeans predict data done at: {:.2?}",
        start_time.elapsed()
    );

    info!("retrieve results from GPU");
    labels_gpu.to_host(&res, &mut labels_host).expect("labels->host transfer failed");


    //let mut centroids_host_result =
     //   ndarray::Array::<f32, _>::zeros((cluster_count as usize, vector_dims as usize));
    //warning!("got labels: {:#?}", labels_host);
    centroids_gpu
        .to_host(&res, &mut centroids_host)
        .expect("centroids->host transfer failed");
    info!(
        "⏱️ retrieved data from GPU at: {:.2?}",
        start_time.elapsed()
    );

    if spherical_centroids {
        info!("normalizing centroids");
        // 1. Iterate over the rows (Axis 0) mutably
        for mut row in centroids_host.axis_iter_mut(Axis(0)) {
            // 2. Calculate L2 norm
            let norm = row.dot(&row).sqrt();
            // 3. Normalize in-place if there is normalization to be done
            if norm > f32::EPSILON {
                row /= norm;
            }
        }
        info!(
        "⏱️ normlaized centroids at: {:.2?}",
        start_time.elapsed()
    );
    }

    let centroids_owned: Vec<f32> = centroids_host.into_raw_vec().into();

    info!(
        "\tClustering (k-means) done in: {:.2?}",
        start_time.elapsed()
    );
    centroids_owned
}
