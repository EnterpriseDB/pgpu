use crate::clustering_gpu_impl::run_clustering;
use crate::guc::use_gpu_acceleration;
use crate::vector_index_read::VectorReadBatcher;
use crate::vectorchord_index;
use crate::{centroids_table, util};
use pgrx::info;
use pgrx::{info, warning};
use pgrx::spi::quote_qualified_identifier;
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
pub fn index(
    table_name: String,
    column_name: String,
    cluster_count: u32,
    sampling_factor: u32,
    batch_size: u64,
    kmeans_iterations: u32,
    kmeans_nredo: u32,
    distance_operator: String,
    skip_index_build: bool,
    spherical_centroids: bool,
) {
    if !use_gpu_acceleration() {
        panic!("GPU acceleration is not enabled. Ensure that your system is compatible and then configure: \"SET pgpu.gpu_acceleration = 'enable';\"");
    }

    if sampling_factor < 40 {
        warning!("sampling factor {sampling_factor} is very low; consider increasing to at least 40 to achieve useful clustering results");
    }

    let (schema, table) = crate::parse_table_identifier(&table_name);
    let schema_table = quote_qualified_identifier(schema.clone(), table.clone());

    util::assert_valid_distance_operator(&distance_operator);
    let spherical_centroids = distance_operator == "ip";
    let centroid_table_name = quote_qualified_identifier(schema, format!("{table}_centroids"));
    assert!(centroid_table_name.len() <= 63, "generated centroid table name \"{centroid_table_name}\" is too long to use as a postgres identifier. Use a source table name that is shorter than 53 characters");

    let start_time = Instant::now();
    // this mimics the `sampling_factor` behavior of vectorchord; so we can use the same args for comparison
    let samples = (cluster_count as u64).saturating_mul(sampling_factor as u64);

    info!("running GPU accelerated index build for {schema_table}.{column_name} using {samples} samples (cluster_count * sampling_factor). Reading and processing vectors in batches of {batch_size}");

    let num_samples = (cluster_count as u64).saturating_mul(sampling_factor as u64);

    let mut batcher =
        VectorReadBatcher::new(schema_table.clone(), column_name, num_samples, batch_size);
    let num_batches = batcher.num_batches();

    // the intermediate batch runs need to produce enough output clusters so that the final consolidation run has enough input
    // we use the same num_samples as configured by the user
    let num_clusters_per_intermediate_batch: u32 = (num_samples / num_batches) as u32;

    info!("clustering properties:\n\t num_clusters_per_intermediate_batch: {num_clusters_per_intermediate_batch}");


    let mut centroids_all: Vec<f32> = Vec::new();
    let mut dims: u32 = 0;

    batcher.start_scan();
    while let Some((vecs, batch_dims)) = batcher.next_batch() {
        info!("processing batch...");
        dims = batch_dims; // this is not expected to change

        crate::print_memory(&vecs, "batch training vectors");

        let centroids_batch = run_clustering(
            vecs,
            dims,
            num_clusters_per_intermediate_batch,
            kmeans_iterations,
            kmeans_nredo,
            spherical_centroids,
        );

        centroids_all.extend_from_slice(&centroids_batch);
        crate::print_memory(&centroids_batch, "centroids from this batch");
        crate::print_memory(&centroids_all, "centroids from all batches");
    }
    batcher.end_scan();
    info!("getting data finished in {:.2?}", start_time.elapsed());

    let centroids_result_flat = if centroids_all.is_empty() {
        info!("No vectors to cluster");
        return;
    } else if centroids_all.len() == (cluster_count * dims) as usize {
        info!("All centroids computed in one batch, skipping re-clusting");
        centroids_all
    } else {
        info!("All centroids computed in multiple batches, starting re-clusting of {} centroids into {cluster_count} clusters", centroids_all.len()/(dims as usize));
        run_clustering(
            centroids_all,
            dims,
            cluster_count,
            kmeans_iterations,
            kmeans_nredo,
            spherical_centroids,
        )
    };

    let centroids_result = centroids_result_flat
        .chunks(dims as usize)
        .map(|x| x.to_vec())
        .collect();

    centroids_table::store_centroids(centroids_result, centroid_table_name.clone(), dims);
    if !skip_index_build {
        info!(
            "clustering all samples finished in {:.2?}. Calling vectorchord index creation",
            start_time.elapsed()
        );
        vectorchord_index::create_vectorchord_index(
            table,
            schema_table,
            centroid_table_name,
            distance_operator,
        );
    } else {
        info!(
        "clustering all samples finished in {:.2?}. SKIPPING vectorchord index creation; skip_index_build=true is set",
        start_time.elapsed()
    );
    }
}
