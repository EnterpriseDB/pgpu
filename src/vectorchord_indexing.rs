use pgrx::{default, pg_extern, warning};

#[allow(clippy::too_many_arguments)]
#[pg_extern]
pub fn create_vector_index_on_gpu(
    table_name: String,
    column_name: String,
    lists: default!(Option<Vec<i32>>, "NULL"),
    sampling_factor: default!(i64, 256),
    batch_size: default!(i64, 100000),
    kmeans_iterations: default!(i64, 10),
    kmeans_nredo: default!(i64, 1),
    distance_operator: default!(String, "'ip'"),
    skip_index_build: default!(bool, false),
    spherical_centroids: default!(bool, false),
    residual_quantization: default!(bool, false),
) {
    let auto_lists: Vec<u32> = match lists {
        None => {vec![1000]}
        Some(l) => {l.iter().map(|i| (*i).try_into().expect("value for lists can't be negative")).collect()}
    };
    warning!("auto_lists: {:?}", auto_lists);

    vectorchord_indexing::index(
        table_name,
        column_name,
        auto_lists,
        sampling_factor.try_into().expect("value for sampling_factor can't be negative"),
        batch_size.try_into().expect("value for batch_size can't be negative"),
        kmeans_iterations.try_into().expect("value for kmeans_iterations can't be negative"),
        kmeans_nredo.try_into().expect("value for kmeans_nredo can't be negative"),
        distance_operator,
        skip_index_build,
        spherical_centroids,
        residual_quantization,
    );
}
