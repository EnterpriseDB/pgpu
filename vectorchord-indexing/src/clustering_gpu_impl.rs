use crate::util::distance_type_from_str;
use cuvs::cluster::kmeans;
use cuvs::distance_type::DistanceType;
use cuvs::{ManagedTensor, Resources};
use ndarray::{Array, Axis};

use pgrx::warning;
use std::time::Instant;

pub fn run_clustering(
    vectors: Vec<f32>,
    vector_dims: u32,
    cluster_count: u32,
    kmeans_iterations: u32,
    kmeans_nredo: u32,
    distance_operator: &str,
    spherical_centroids: bool,
) -> Vec<f32> {
    warning!("Clustering vectors on GPU");
    let start_time = Instant::now();

    // cuvs setup
    let res = Resources::new().expect("GPU Resource creation failed");
    // shape is (rows, cols). rows is determined by the length of the vector input; so we divide by dimensions to get that value
    let vectors_array = Array::from_shape_vec(
        (vectors.len() / vector_dims as usize, vector_dims as usize),
        vectors.to_vec(),
    )
        .expect("shaping vectors failed");
    let dataset = ManagedTensor::from(&vectors_array)
        .to_device(&res)
        .expect("vectors->tensor transformation failed");

    let centroids_host =
        ndarray::Array::<f32, _>::zeros((cluster_count as usize, vector_dims as usize));
    let mut centroids = ManagedTensor::from(&centroids_host)
        .to_device(&res)
        .expect("centroids(empty)->tensor transformation failed");

    let distance_operator_cuvs = distance_type_from_str(&distance_operator)
        .expect(format!("invalid distance operator: {distance_operator}").as_str());
    let balanced_kmeans = match distance_operator_cuvs {
        DistanceType::InnerProduct | DistanceType::CosineExpanded => true,
        _ => false,
    };

    warning!(
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

    warning!(
        "\tpreparing/transferring data done in: {:.2?}",
        start_time.elapsed()
    );

    warning!("running kemans");
    let (inertia, n_iter) = kmeans::fit(&res, &kmeans_params, &dataset, &None, &mut centroids)
        .expect("kmeans training failed");
    warning!("kmeans done with inertia: {inertia}, n_iter: {n_iter}");

    warning!("retrieve results from GPU");
    let mut centroids_host_result =
        ndarray::Array::<f32, _>::zeros((cluster_count as usize, vector_dims as usize));
    centroids
        .to_host(&res, &mut centroids_host_result)
        .expect("centroids->host transfer failed");

    if spherical_centroids {
        warning!("normalizing centroids");
        // 1. Iterate over the rows (Axis 0) mutably
        for mut row in centroids_host_result.axis_iter_mut(Axis(0)) {
            // 2. Calculate L2 norm
            let norm = row.dot(&row).sqrt();
            // 3. Normalize in-place if there is normalization to be done
            if norm > f32::EPSILON {
                row /= norm;
            }
        }
    }

    let centroids_owned: Vec<f32> = centroids_host_result.into_raw_vec().into();

    warning!(
        "\tClustering (k-means) done in: {:.2?}",
        start_time.elapsed()
    );
    centroids_owned
}
