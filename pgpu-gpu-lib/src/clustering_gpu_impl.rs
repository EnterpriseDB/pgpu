use abi_stable::std_types::RVec;
use faiss::{
    cluster::{Clustering, ClusteringParameters},
    gpu::StandardGpuResources,
    index_factory, GpuResources, MetricType,
};
use std::time::Instant;

// NOTICE: update the ABI version in abi_version.rs when changing this function signature
#[no_mangle]
pub extern "C" fn run_clustering(
    vectors: RVec<f32>,
    vector_dims: u32,
    cluster_count: u32,
    kmeans_iterations: u32,
    kmeans_nredo: u32,
    spherical_centroids: bool,
) -> RVec<f32> {
    println!("Clustering vectors on GPU");
    let start_time = Instant::now();

    // Set up clustering parameters
    let mut params = ClusteringParameters::new();
    params.set_niter(kmeans_iterations); // number of k-means iterations - vchord on GPU uses 25; on CPU they only use 10
    params.set_nredo(kmeans_nredo); // number of times to redo and keep best
    params.set_verbose(true); // print progress
    params.set_update_index(true); // update the index after each iteration for better results
    params.set_spherical(spherical_centroids);

    // spherical_centroids should be used when the distance metric for the dataset is not L2
    let index_metric_type = if spherical_centroids {
        MetricType::InnerProduct
    } else {
        MetricType::L2
    };

    // Create the clustering object with parameters
    let mut clustering = Clustering::new_with_params(vector_dims, cluster_count, &params)
        .expect("Clustering creation failed");

    let mut gpu_res = StandardGpuResources::new().unwrap();
    gpu_res.no_temp_memory().unwrap(); // TODO: we have a GPU memory leak when temp_memory is being used; not sure why. AFAICT this shouldn't be the case. Disabling temp memory seems to cost a little performance
    let mut index = index_factory(vector_dims, "Flat", index_metric_type)
        .unwrap()
        .into_gpu(&gpu_res, 0)
        .expect("Flat index creation failed");

    // Run the clustering algorithm
    clustering
        .train(&vectors, &mut index)
        .expect("training failed");

    // Retrieve centroids (k x vector_dims floats)
    let centroids = clustering.centroids().expect("centroids not found");
    let centroids_owned: RVec<f32> = centroids.into_iter().flatten().cloned().collect();

    println!(
        "\tClustering (k-means) done in: {:.2?}",
        start_time.elapsed()
    );
    centroids_owned
}
