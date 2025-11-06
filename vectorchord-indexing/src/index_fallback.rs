#[allow(clippy::too_many_arguments)]
pub fn index(
    _table_name: String,
    _column_name: String,
    _cluster_count: u32,
    _sampling_factor: u32,
    _batch_size: u64,
    _kmeans_iterations: u32,
    _kmeans_nredo: u32,
    _distance_operator: String,
    _skip_index_build: bool,
) {
    panic!("GPU acceleration is not available in this build of pgpu. Please check if your platform is supported: https://www.enterprisedb.com/docs/edb-postgres-ai/ai-factory/pipeline/compatibility/")
}
