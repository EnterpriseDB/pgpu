use crate::clustering_gpu_impl::{
    run_clustering_batch, run_clustering_consolidate, run_clustering_multilevel, run_clustering_hierarchical,
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
    info!("🚀 running GPU accelerated index build for {qualified_table}.{column_name}");

    let centroid_table_name = quote_qualified_identifier(schema, format!("{table}_centroids"));
    assert!(centroid_table_name.len() <= 63, "centroid table name too long");

    if sampling_factor < 40 {
        warning!("sampling factor {sampling_factor} is very low; consider increasing to at least 40 to achieve useful clustering results");
    }

    util::assert_valid_distance_operator(&distance_operator);

    let start_time = Instant::now();
    let num_samples = (num_clusters_leaf as u64).saturating_mul(sampling_factor as u64);
    let num_batches = num_samples.div_ceil(batch_size) as u32;

    let num_clusters_per_intermediate_batch: u32 = match num_batches {
        1 => {
            info!("clustering properties:\n\t uses_batching: false\n\t lists: {lists:?}");
            num_clusters_leaf
        }
        _ => {
            let desired_intermediate_batch_clusters = num_clusters_leaf * 4;
            let n = desired_intermediate_batch_clusters / num_batches;
            info!("clustering properties:\n\t uses_batching: true\n\t num_clusters_per_intermediate_batch: {n}\n\t desired_intermediate_batch_clusters: {desired_intermediate_batch_clusters}\n\t lists: {lists:?}");
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

    let mut dims: u32 = 0;

    // =========================================================================================
    // UNIFIED CLUSTERING LOGIC
    // =========================================================================================

    let centroids_result: Vec<(Vec<f32>, i32)> = if lists.len() == 2 {
        // --- PATH A: Top-Down Hierarchical (Optimized for 100M Scale) ---
        info!("Detected 2-level hierarchy {lists:?}. Using Optimized Top-Down GPU Clustering.");

        let mut all_vectors: Vec<f32> = Vec::with_capacity((num_samples * 96) as usize);
        let mut batch_count = 0;

        while let Some((vecs, batch_dims)) = batcher.next_batch() {
            batch_count += 1;
            dims = batch_dims;
            all_vectors.extend_from_slice(&vecs);
            if batch_count % 5 == 0 {
                info!("Loaded batch {batch_count}/{num_batches} into RAM...");
            }
        }
        batcher.end_scan();
        info!("batches finished loading in {:.2?}", start_time.elapsed());

        if all_vectors.is_empty() {
            warning!("No vectors found for clustering!");
            return;
        }

        run_clustering_hierarchical(
            all_vectors,
            dims,
            lists.clone(),
            kmeans_iterations,
            kmeans_nredo,
            &distance_operator,
            spherical_centroids,
            residual_quantization, // CRITICAL: Fixes 5% recall by training on residuals
        )

    } else {
        // --- PATH B: Bottom-Up Batch Logic ---
        let mut centroids_all: Vec<f32> = Vec::new();
        let mut weights_all: Vec<f32> = Vec::new();
        let mut batch_count = 0;

        while let Some((vecs, batch_dims)) = batcher.next_batch() {
            batch_count += 1;
            info!("processing batch ({batch_count}/{num_batches})");
            dims = batch_dims;

            let (centroids_batch, weights_batch) = run_clustering_batch(
                vecs,
                dims,
                num_clusters_per_intermediate_batch,
                kmeans_iterations,
                kmeans_nredo,
                &distance_operator,
                spherical_centroids,
                false, // No internal hierarchy for manual batches
            );

            centroids_all.extend_from_slice(&centroids_batch);
            weights_all.extend_from_slice(&weights_batch);
            util::print_memory(&centroids_batch, "centroids from this batch");
            util::print_memory(&centroids_all, "centroids from all batches");
            util::print_memory(&weights_all, "weights from all batches");
        }
        batcher.end_scan();
        info!("batches finished in {:.2?}", start_time.elapsed());

        let centroids_leaf = if centroids_all.is_empty() {
            warning!("empty result from kmeans clustering");
            return;
        } else if centroids_all.len() == (num_clusters_leaf * dims) as usize {
            info!("All centroids computed in one batch, skipping re-clusting");
            centroids_all
        } else {
            info!("All centroids computed in multiple batches, starting re-clusting of {} centroids into {num_clusters_leaf} clusters", centroids_all.len()/(dims as usize));
            run_clustering_consolidate(
                centroids_all,
                weights_all,
                dims,
                num_clusters_leaf,
                kmeans_iterations,
                kmeans_nredo,
                &distance_operator, // Updated signature
                spherical_centroids,
            )
        };

        match num_clusters_top_option {
            None => {
                centroids_leaf.chunks(dims as usize).map(|x| (x.to_vec(), -1)).collect()
            }
            Some(num_clusters_top) => {
                let (centroids_top, parents_leaf) = run_clustering_multilevel(
                    &centroids_leaf,
                    dims,
                    num_clusters_top,
                    kmeans_iterations,
                    kmeans_nredo,
                    &distance_operator, // Updated signature
                    spherical_centroids,
                );

                let mut results: Vec<(Vec<f32>, i32)> = centroids_top
                    .chunks(dims as usize).map(|x| (x.to_vec(), -1)).collect();

                let leaves: Vec<(Vec<f32>, i32)> = centroids_leaf
                    .chunks(dims as usize).zip(parents_leaf.into_iter())
                    .map(|(v, p)| (v.to_vec(), p)).collect();

                results.extend(leaves);
                results
            }
        }
    };

    // =========================================================================================
    // FINAL STORAGE AND INDEX CREATION
    // =========================================================================================

    centroids_table::store_centroids(
        centroids_result,
        centroid_table_name.clone(),
        dims,
        residual_quantization
    );

    if !skip_index_build {
        info!("✨ clustering finished in {:.2?}. Creating vectorchord index", start_time.elapsed());
        vectorchord_index::create_vectorchord_index(
            table,
            qualified_table,
            centroid_table_name,
            distance_operator,
            residual_quantization,
        );
    } else {
        info!("✅ clustering finished. skip_index_build=true");
    }
}