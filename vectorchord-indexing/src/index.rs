use crate::clustering_gpu_impl::{
    bottom_up, run_clustering_batch, run_clustering_consolidate, BottomUpResult,
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
    _: bool,
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
    let global_start = Instant::now();

    // ========================================================================================
    // PATH A: BOTTOM-UP HIERARCHICAL BUILD (GPU Optimized)
    // Triggered if `lists` has 2 elements (e.g. [400, 160000])
    //
    // Strategy: ONE GPU k-means for all leaves, then cluster leaves into roots.
    // This is much faster than top-down (which made N separate GPU calls).
    // ========================================================================================

    if let Some(num_roots) = num_clusters_top_option {
        let num_leaves = num_clusters_leaf;
        let num_samples_target = (num_leaves as u64).saturating_mul(sampling_factor as u64);

        info!("🏗️ [BOTTOM-UP BUILD] GPU-Accelerated Hierarchical Clustering");
        info!("📊 Target: {} Roots | {} Leaves", num_roots, num_leaves);
        info!("📉 Sampling: Factor={} -> Reading ~{} vectors", sampling_factor, num_samples_target);

        info!("⚙️ Configuration:\n\
           \t• Leaves:            {}\n\
           \t• Roots:             {}\n\
           \t• Sampling Factor:   {}\n\
           \t• Batch Size:        {}\n\
           \t• KMeans Iterations: {}",
           num_leaves, num_roots, sampling_factor, batch_size, kmeans_iterations);

        // ================================================================================
        // STEP 1: Load training data
        // ================================================================================
        let t_load_start = Instant::now();

        let mut batcher = VectorReadBatcher::new(
            qualified_table.clone(),
            column_name.clone(),
            num_leaves,
            sampling_factor,
            batch_size,
        );

        let mut training_dataset: Vec<f32> = Vec::with_capacity((num_samples_target as usize) * 768);
        let mut vector_dims = 0;
        let mut loaded_count = 0;

        info!("📥 Loading training samples into RAM...");

        let mut last_log_time = Instant::now();
        let mut last_log_count: usize = 0;

        while let Some((vecs, dims)) = batcher.next_batch() {
            if vector_dims == 0 {
                vector_dims = dims;
                let total_bytes = (num_samples_target as u64) * (dims as u64) * 4;
                let gb_usage = total_bytes as f64 / 1_073_741_824.0;

                info!("📝 Detected Vector Dims: {}", dims);
                info!("💾 Estimated RAM for Training Data: {:.2} GB", gb_usage);

                if gb_usage > 64.0 {
                    warning!("⚠️ High RAM usage! Ensure {:.0} GB free.", gb_usage * 1.2);
                }
            }

            loaded_count += vecs.len() / dims as usize;
            training_dataset.extend(vecs);

            // Progress logging every 5 seconds
            let elapsed_since_log = last_log_time.elapsed().as_secs_f64();
            if elapsed_since_log >= 5.0 {
                let vectors_since_log = loaded_count - last_log_count;
                let rate = vectors_since_log as f64 / elapsed_since_log;
                let percent = (loaded_count as f64 / num_samples_target as f64) * 100.0;
                let remaining = num_samples_target as usize - loaded_count;
                let eta_secs = if rate > 0.0 { remaining as f64 / rate } else { 0.0 };

                info!(
                    "⏳ Loaded {}/{} ({:.1}%) | {:.0} vec/s | ETA: {:.0}s",
                    loaded_count, num_samples_target, percent, rate, eta_secs
                );

                last_log_time = Instant::now();
                last_log_count = loaded_count;
            }
        }
        batcher.end_scan();
        let d_load = t_load_start.elapsed();
        info!("✅ Loaded {} vectors in {:.2?}", loaded_count, d_load);

        if loaded_count == 0 || vector_dims == 0 {
            pgrx::error!("❌ No training data found! Table might be empty.");
        }

        // ================================================================================
        // STEP 2: Run bottom-up GPU clustering
        // ================================================================================
        let t_cluster_start = Instant::now();

        let BottomUpResult {
            root_centroids,
            leaf_centroids,
            leaf_to_root,
        } = bottom_up(
            training_dataset,
            vector_dims,
            num_leaves,
            num_roots,
            kmeans_iterations,
            spherical_centroids,
        );

        let d_cluster = t_cluster_start.elapsed();

        // ================================================================================
        // STEP 3: Build centroids table
        // ================================================================================
        let t_store_start = Instant::now();

        let mut final_results: Vec<(Vec<f32>, i32)> = Vec::new();

        // Add root centroids (parent_id = -1)
        let num_roots_actual = root_centroids.len() / vector_dims as usize;
        for root_vec in root_centroids.chunks(vector_dims as usize) {
            final_results.push((root_vec.to_vec(), -1));
        }

        // Add leaf centroids with their parent assignments
        let num_leaves_actual = leaf_centroids.len() / vector_dims as usize;
        for (leaf_idx, leaf_vec) in leaf_centroids.chunks(vector_dims as usize).enumerate() {
            let parent_id = leaf_to_root[leaf_idx];
            final_results.push((leaf_vec.to_vec(), parent_id));
        }

        info!("💾 Storing {} centroids ({} roots + {} leaves)...",
              final_results.len(), num_roots_actual, num_leaves_actual);

        centroids_table::store_centroids(final_results, centroid_table_name.clone(), vector_dims);
        let d_store = t_store_start.elapsed();

        // ================================================================================
        // SUMMARY
        // ================================================================================
        info!(
            "\n⏱️  [TIMING SUMMARY - BOTTOM-UP]\n\
            \t• 📥 Data Loading:    {:.2?}\n\
            \t• 🚀 GPU Clustering:  {:.2?}\n\
            \t• 💾 Storage:         {:.2?}\n\
            \t-----------------------------\n\
            \t📊 [RESULTS]\n\
            \t• Vectors Sampled:    {}\n\
            \t• Root Centroids:     {}\n\
            \t• Leaf Centroids:     {}\n\
            \t-----------------------------\n\
            \t👉 TOTAL TIME:        {:.2?}",
            d_load,
            d_cluster,
            d_store,
            loaded_count,
            num_roots_actual,
            num_leaves_actual,
            global_start.elapsed()
        );

    // ========================================================================================
    // PATH B: FLAT BUILD (single level, no hierarchy)
    // ========================================================================================
    } else {
        info!("🏗️ [FLAT DETECTED] Running Bottom-Up Batch Clustering");

        // --- SAMPLER INITIALIZATION ---
        // New call format
        let mut batcher = VectorReadBatcher::new(
            qualified_table.clone(),
            column_name.clone(),
            num_clusters_leaf, // num_clusters
            sampling_factor,   // factor
            batch_size,
        );

        let mut centroids_all: Vec<f32> = Vec::new();
        let mut weights_all: Vec<f32> = Vec::new();
        let mut dims: u32 = 0;
        let mut batch_count = 0;

        let num_samples_target = (num_clusters_leaf as u64).saturating_mul(sampling_factor as u64);
        let num_batches = num_samples_target.div_ceil(batch_size) as u32; // FIXED: Variable defined

        let num_clusters_per_intermediate_batch: u32 = match num_batches {
            1 => num_clusters_leaf,
            _ => {
                let target = num_clusters_leaf * 4;
                std::cmp::max(target / num_batches, 3)
            }
        };

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

        let centroids_result: Vec<(Vec<f32>, i32)> = centroids_leaf
            .chunks(dims as usize)
            .map(|x| (x.to_vec(), -1))
            .collect();

        centroids_table::store_centroids(centroids_result, centroid_table_name.clone(), dims);
    }

    if !skip_index_build {
        info!("💾 Training complete ({:.2?}). Building VectorChord Index...", global_start.elapsed());
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