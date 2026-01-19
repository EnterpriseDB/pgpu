use crate::clustering_gpu_impl::{
    assign_to_roots_gpu, run_clustering_batch, run_clustering_consolidate,
    train_leaves_for_bucket_gpu, train_roots_gpu,
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
    // 1. Validate Inputs & Hardware
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

    // ========================================================================================
    // PATH A: TOP-DOWN HIERARCHICAL BUILD (New Logic)
    // Triggered if `lists` has 2 elements (e.g. [400, 160000])
    // ========================================================================================

    if let Some(num_roots) = num_clusters_top_option {
        let num_leaves = num_clusters_leaf;
        let num_leaves_per_root = num_leaves / num_roots;

        info!("🏗️ [HIERARCHICAL DETECTED] Starting Top-Down Build (in-memory)");
        info!("📊 Target: {} Roots | {} Leaves ({} per root)", num_roots, num_leaves, num_leaves_per_root);

        // 1. Load ALL Data into RAM (as requested)
        // We use the batcher to drain the table into a single Vec<f32>
        // NOTE: For 100M vectors @ 768 dims, this requires ~300GB RAM.
        let num_samples_total = 100_000_000; // Large upper bound to read everything
        let mut batcher = VectorReadBatcher::new(
            qualified_table.clone(),
            column_name.clone(),
            num_samples_total,
            batch_size,
            1, // Unused for simple read
        );

        let mut full_dataset: Vec<f32> = Vec::new();
        let mut vector_dims = 0;
        let mut loaded_count = 0;

        info!("📥 Loading dataset into RAM...");
        while let Some((vecs, dims)) = batcher.next_batch() {
            if vector_dims == 0 { vector_dims = dims; }
            loaded_count += vecs.len() / dims as usize;
            full_dataset.extend(vecs);

            if loaded_count % 5_000_000 == 0 {
                info!("... Loaded {} vectors", loaded_count);
            }
        }
        batcher.end_scan();
        info!("✅ Dataset Loaded: {} vectors. Time: {:.2?}", loaded_count, start_time.elapsed());

        // 2. Phase 1: Train Roots
        let root_centroids = train_roots_gpu(
            &full_dataset,
            vector_dims,
            num_roots,
            kmeans_iterations
        );

        // 3. Phase 2: Partition
        let assignments = assign_to_roots_gpu(
            &full_dataset,
            &root_centroids,
            vector_dims,
            num_roots
        );

        // 4. Phase 3: Train Leaves
        info!("🚀 [PHASE 3] Scattering vectors & Training Leaves...");
        let mut buckets: Vec<Vec<f32>> = vec![Vec::new(); num_roots as usize];

        // Scatter
        for (idx, &label) in assignments.iter().enumerate() {
            if label >= 0 && (label as usize) < num_roots as usize {
                let start = idx * vector_dims as usize;
                let end = start + vector_dims as usize;
                buckets[label as usize].extend_from_slice(&full_dataset[start..end]);
            }
        }

        let mut final_results: Vec<(Vec<f32>, i32)> = Vec::new();

        // Store Roots (Parent = -1)
        for root_vec in root_centroids.chunks(vector_dims as usize) {
            final_results.push((root_vec.to_vec(), -1));
        }

        // Train & Store Leaves (Parent = root_idx)
        let mut total_leaves_trained = 0;
        for (root_idx, bucket_vecs) in buckets.iter().enumerate() {
            let n_vecs = bucket_vecs.len() / vector_dims as usize;
            if n_vecs == 0 { continue; }

            if root_idx % 20 == 0 {
                info!("🌿 Root {}/{} | Size: {} | Training Leaves...", root_idx, num_roots, n_vecs);
            }

            let leaf_centroids_flat = train_leaves_for_bucket_gpu(
                bucket_vecs,
                vector_dims,
                num_leaves_per_root,
                15
            );

            for leaf_vec in leaf_centroids_flat.chunks(vector_dims as usize) {
                final_results.push((leaf_vec.to_vec(), root_idx as i32));
            }
            total_leaves_trained += num_leaves_per_root;
        }

        info!("🏁 [HIERARCHY COMPLETE] Trained {} Leaves. Saving...", total_leaves_trained);
        centroids_table::store_centroids(final_results, centroid_table_name.clone(), vector_dims);

    // ========================================================================================
    // PATH B: FLAT / LEGACY BUILD
    // Triggered if `lists` has 1 element (e.g. [2000])
    // ========================================================================================
    } else {
        info!("🏗️ [FLAT DETECTED] Running Bottom-Up Batch Clustering");

        let num_samples = (num_clusters_leaf as u64).saturating_mul(sampling_factor as u64);
        let num_batches = num_samples.div_ceil(batch_size) as u32;

        let num_clusters_per_intermediate_batch: u32 = match num_batches {
            1 => num_clusters_leaf,
            _ => {
                let target = num_clusters_leaf * 4;
                std::cmp::max(target / num_batches, 3) // Ensure at least 3
            }
        };

        let mut batcher = VectorReadBatcher::new(
            qualified_table.clone(),
            column_name.clone(),
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
            dims = batch_dims;

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
        }
        batcher.end_scan();

        // Consolidate Results
        let centroids_leaf = if centroids_all.is_empty() {
            warning!("empty result from kmeans clustering");
            return;
        } else if centroids_all.len() == (num_clusters_leaf * dims) as usize {
            centroids_all
        } else {
            info!("Consolidating {} centroids into {}...", centroids_all.len() / dims as usize, num_clusters_leaf);
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

        // Format for Storage (Parent = -1 for flat)
        let centroids_result: Vec<(Vec<f32>, i32)> = centroids_leaf
            .chunks(dims as usize)
            .map(|x| (x.to_vec(), -1))
            .collect();

        centroids_table::store_centroids(centroids_result, centroid_table_name.clone(), dims);
    }

    // ========================================================================================
    // CREATE INDEX (Common Step)
    // ========================================================================================
    if !skip_index_build {
        info!("💾 Training complete ({:.2?}). Building VectorChord Index...", start_time.elapsed());
        vectorchord_index::create_vectorchord_index(
            table,
            qualified_table,
            centroid_table_name,
            distance_operator,
            residual_quantization,
        );
    } else {
        info!("🛑 Skipping index build (skip_index_build=true). Centroids saved.");
    }
}
}