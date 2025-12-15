use crate::clustering_gpu_impl::{
    run_clustering_batch, run_clustering_consolidate, run_clustering_multilevel,
};
use crate::guc::use_gpu_acceleration;
use crate::vector_index_read::VectorReadBatcher;
use crate::vectorchord_index;
use crate::{centroids_table, util};
use pgrx::spi::quote_qualified_identifier;
use pgrx::{info, warning};
use std::time::Instant;

#[allow(clippy::too_many_arguments)]
pub fn index(
    table_name: String,
    column_name: String,
    lists: Vec<u32>,
    sampling_factor: u32,
    batch_size: u64,
    kmeans_iterations: u32,
    kmeans_nredo: u32,
    distance_operator: String,
    skip_index_build: bool,
    spherical_centroids: bool,
    residual_quantization: bool,
) {
    let (num_clusters_top_option, num_clusters_leaf) = match lists.len() {
        1 => (None, lists[0]),
        2 => (Some(lists[0]), lists[1]),
        _ => {
            pgrx::error!("invalid lists parameter: {lists:?}. Must be either [n] or [n, m]")
        }
    };
    if !use_gpu_acceleration() {
        pgrx::error!("GPU acceleration is not enabled. Ensure that your system is compatible and then configure: \"SET pgpu.gpu_acceleration = 'enable';\"");
    }
    let (schema, table) = crate::util::parse_table_identifier(&table_name);
    let qualified_table = quote_qualified_identifier(schema.clone(), table.clone());
    info!("running GPU accelerated index build for {qualified_table}.{column_name}");

    if sampling_factor < 40 {
        warning!("sampling factor {sampling_factor} is very low; consider increasing to at least 40 to achieve useful clustering results");
    }

    util::assert_valid_distance_operator(&distance_operator);
    let centroid_table_name = quote_qualified_identifier(schema, format!("{table}_centroids"));
    assert!(centroid_table_name.len() <= 63, "generated centroid table name \"{centroid_table_name}\" is too long to use as a postgres identifier. Use a source table name that is shorter than 53 characters");

    let start_time = Instant::now();

    let num_samples = (num_clusters_leaf as u64).saturating_mul(sampling_factor as u64);
    let num_batches = num_samples.div_ceil(batch_size) as u32;

    // the intermediate batch runs need to produce enough output clusters so that the final consolidation run has enough input
    // we use the same num_samples as configured by the user
    // Note: typically, you'll want 30-50 data points per cluster. But here, we're just stiching together the pre-trained centroids from the intermediate batches
    // so a much lower points/clusters ration can be used
    let num_clusters_per_intermediate_batch: u32 = match num_batches {
        1 => {
            info!(
                "clustering properties:\n\t uses_batching: false\n\t num_clusters: {num_clusters_leaf}"
            );
            num_clusters_leaf
        }
        _ => {
            let desired_intermediate_batch_clusters = num_clusters_leaf * 4; // * 40;
            let n = desired_intermediate_batch_clusters / num_batches;
            info!("clustering properties:\n\t uses_batching: true\n\t num_clusters_per_intermediate_batch: {n}\n\t desired_intermediate_batch_clusters: {desired_intermediate_batch_clusters}\n\t num_clusters: {num_clusters_leaf}");
            n
        }
    };
    assert!(num_clusters_leaf > 2, "cluster count must be larger than 2");
    assert!(
        num_clusters_per_intermediate_batch > 2,
        "batch size is too small for clustering"
    );
    let mut batcher = VectorReadBatcher::new(
        qualified_table.clone(),
        column_name,
        num_samples,
        batch_size,
        num_clusters_per_intermediate_batch as u64,
    );

    let mut centroids_all: Vec<f32> = Vec::new();
    let mut weights_all: Vec<f32> = Vec::new();
    let mut dims: u32 = 0;

    let mut batch_count = 0;
    while let Some((vecs, batch_dims)) = batcher.next_batch() {
        batch_count += 1;
        info!("processing batch ({batch_count}/{num_batches})");
        dims = batch_dims; // this is not expected to change

        util::print_memory(&vecs, "batch training vectors");

        let (centroids_batch, weights_batch) = run_clustering_batch(
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
        util::print_memory(&centroids_batch, "centroids from this batch");
        util::print_memory(&centroids_all, "centroids from all batches");
        util::print_memory(&weights_all, "weights from all batches");
    }
    batcher.end_scan();
    info!("batches finished in {:.2?}", start_time.elapsed());

    let (centroids_leaf, weights_leaf) = if centroids_all.is_empty() {
        warning!("empty result from kmeans clustering");
        return;
    } else if centroids_all.len() == (num_clusters_leaf * dims) as usize {
        info!("All centroids computed in one batch, skipping re-clusting");
        (centroids_all, weights_all)
    } else {
        info!("All centroids computed in multiple batches, starting re-clusting of {} centroids into {num_clusters_leaf} clusters", centroids_all.len()/(dims as usize));
        run_clustering_consolidate(
            centroids_all,
            weights_all,
            dims,
            num_clusters_leaf,
            kmeans_iterations,
            kmeans_nredo,
            spherical_centroids,
        )
    };

    // elements are (centroid, parent_id)
    // the IDs of the centroids are their vector index
    let centroids_result: Vec<(Vec<f32>, i32)> = match num_clusters_top_option {
        None => {
            // No parent ID if we don't have a top-level list
            centroids_leaf
                .chunks(dims as usize)
                // -1 indicates NULL parent
                .map(|x| (x.to_vec(), -1))
                .collect()
        }
        Some(num_clusters_top) => {
            let (centroids_top, parents_leaf) = run_clustering_multilevel(
                &centroids_leaf,
                weights_leaf,
                dims,
                num_clusters_top,
                kmeans_iterations,
                kmeans_nredo,
                spherical_centroids,
            );
            let centroids_leaf_chunked: Vec<Vec<f32>> = centroids_leaf
                .chunks(dims as usize)
                .map(|x| x.to_vec())
                .collect();

            // we'll add the chunked top centroids first and then append the leaf ones
            let mut centroids_all_chunked: Vec<Vec<f32>> = centroids_top
                .chunks(dims as usize)
                .map(|x| x.to_vec())
                .collect();
            // init the NULL parents for the top centroids then append the leaf ones
            let mut parents_all = vec![-1; centroids_all_chunked.len()];

            centroids_all_chunked.extend(centroids_leaf_chunked);
            parents_all.extend(parents_leaf);
            assert_eq!(
                centroids_all_chunked.len(),
                parents_all.len(),
                "number of centroids and parents must match"
            );

            centroids_all_chunked
                .into_iter()
                .zip(parents_all.into_iter())
                .collect()
        }
    };

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
            residual_quantization,
        );
    } else {
        info!(
        "clustering all samples finished in {:.2?}. SKIPPING vectorchord index creation; skip_index_build=true is set",
        start_time.elapsed()
    );
    }
}
