use pgrx::{default, pg_extern};

#[allow(clippy::too_many_arguments)]
#[pg_extern]
pub fn create_vector_index_on_gpu(
    table_name: String,
    column_name: String,
    cluster_count: default!(i64, 1000),
    sampling_factor: default!(i64, 256),
    batch_size: default!(i64, 100000),
    kmeans_iterations: default!(i64, 10),
    kmeans_nredo: default!(i64, 1),
    distance_operator: default!(String, "'ip'"),
    skip_index_build: default!(bool, false),
    spherical_centroids: default!(bool, false),
) {
    vectorchord_indexing::index(
        table_name,
        column_name,
        cluster_count as u32,
        sampling_factor as u32,
        batch_size as u64,
        kmeans_iterations as u32,
        kmeans_nredo as u32,
        distance_operator,
        skip_index_build,
        spherical_centroids,
    );
}
