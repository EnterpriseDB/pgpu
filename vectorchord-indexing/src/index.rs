use crate::clustering_gpu_impl::{run_clustering, run_clustering_consolidate};
use crate::guc::use_gpu_acceleration;
use crate::vector_index_read::VectorReadBatcher;
use crate::vectorchord_index;
use crate::{centroids_table, util};
use pgrx::{info, warning};
use pgrx::spi::quote_qualified_identifier;
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
pub fn index(
    table_name: String,
    column_name: String,
    num_clusters: u32,
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
    let qualified_table = quote_qualified_identifier(schema.clone(), table.clone());

    util::assert_valid_distance_operator(&distance_operator);
    let centroid_table_name = quote_qualified_identifier(schema, format!("{table}_centroids"));
    assert!(centroid_table_name.len() <= 63, "generated centroid table name \"{centroid_table_name}\" is too long to use as a postgres identifier. Use a source table name that is shorter than 53 characters");

    let start_time = Instant::now();
    info!("running GPU accelerated index build for {qualified_table}.{column_name}");

    let num_samples = (num_clusters as u64).saturating_mul(sampling_factor as u64);

    let mut batcher =
        VectorReadBatcher::new(qualified_table.clone(), column_name, num_samples, batch_size);
    let num_batches = batcher.num_batches();

    // the intermediate batch runs need to produce enough output clusters so that the final consolidation run has enough input
    // we use the same num_samples as configured by the user
    // Note: typically, you'll want 30-50 data points per cluster. But here, we're just stiching together the pre-trained centroids from the intermediate batches
    // so a much lower points/clusters ration can be used
    let desired_intermediate_batch_clusters = num_clusters * 4; // * 40;
    let num_clusters_per_intermediate_batch: u32 = desired_intermediate_batch_clusters / num_batches;

    info!("clustering properties:\n\t num_clusters_per_intermediate_batch: {num_clusters_per_intermediate_batch}\n\t desired_intermediate_batch_clusters: {desired_intermediate_batch_clusters}\n\t num_clusters: {num_clusters}");


    let mut centroids_all: Vec<f32> = Vec::new();
    let mut weights_all: Vec<f32> = Vec::new();
    let mut dims: u32 = 0;


    while let Some((vecs, batch_dims)) = batcher.next_batch() {
        info!("processing batch...");
        dims = batch_dims; // this is not expected to change

        crate::print_memory(&vecs, "batch training vectors");

        let (centroids_batch, weights_batch) = run_clustering(
            vecs,
            dims,
            num_clusters_per_intermediate_batch,
            kmeans_iterations,
            kmeans_nredo,
            &distance_operator,
            spherical_centroids,
        );

        centroids_all.extend_from_slice(&centroids_batch);
        weights_all.extend_from_slice(&weights_batch);
        crate::print_memory(&centroids_batch, "centroids from this batch");
        crate::print_memory(&centroids_all, "centroids from all batches");
        crate::print_memory(&weights_all, "weights from all batches");
    }
    batcher.end_scan();
    info!("getting data finished in {:.2?}", start_time.elapsed());

    let centroids_result_flat = if centroids_all.is_empty() {
        info!("No vectors to cluster");
        return;
    } else if centroids_all.len() == (num_clusters * dims) as usize {
        info!("All centroids computed in one batch, skipping re-clusting");
        centroids_all
    } else {
        info!("All centroids computed in multiple batches, starting re-clusting of {} centroids into {num_clusters} clusters", centroids_all.len()/(dims as usize));
        run_clustering_consolidate(
            centroids_all,
            weights_all,
            dims,
            num_clusters,
            kmeans_iterations,
            kmeans_nredo,
            &distance_operator,
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
            qualified_table,
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
