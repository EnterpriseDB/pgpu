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
    let global_start = Instant::now();

    // ========================================================================================
    // PATH A: TOP-DOWN HIERARCHICAL BUILD (New Logic)
    // Triggered if `lists` has 2 elements (e.g. [400, 160000])
    // ========================================================================================

    if let Some(num_roots) = num_clusters_top_option {
        let num_leaves = num_clusters_leaf;
        let num_leaves_per_root = num_leaves / num_roots;

        // 1. Calculate Sample Size based on Sampling Factor
        let num_samples_to_read = (num_leaves as u64).saturating_mul(sampling_factor as u64);

        info!("🏗️ [HIERARCHICAL DETECTED] Starting Top-Down Build");
        info!("📊 Target: {} Roots | {} Leaves ({} per root)", num_roots, num_leaves, num_leaves_per_root);
        info!("📉 Sampling: Factor={} -> Reading {} vectors for training", sampling_factor, num_samples_to_read);
        info!("⚙️ Clustering Configuration:\n\
           \t• Target Lists (Leaf):  {}\n\
           \t• Sampling Factor:      {}\n\
           \t• Batch Size:           {}\n\
           \t• KMeans Iterations:    {}\n\
           \t• KMeans N-Redo:        {} (Best of N runs)",
           num_clusters_leaf, sampling_factor, batch_size, kmeans_iterations, kmeans_nredo);

        // 2. Load Samples into RAM
        // We use the batcher to read exactly 'num_samples_to_read'
        let t_load_start = Instant::now();
        let mut batcher = VectorReadBatcher::new(
            qualified_table.clone(),
            column_name.clone(),
            num_samples_to_read,
            batch_size,
            1, // Unused for simple read
        );

        let mut training_dataset: Vec<f32> = Vec::with_capacity((num_samples_to_read as usize) * 768);
        let mut vector_dims = 0;
        let mut loaded_count = 0;

        info!("📥 Loading training samples into RAM...");
        while let Some((vecs, dims)) = batcher.next_batch() {
            if vector_dims == 0 { vector_dims = dims; }
            loaded_count += vecs.len() / dims as usize;
            training_dataset.extend(vecs);

            if loaded_count % 5_000_000 == 0 {
                info!("... Loaded {}/{} samples", loaded_count, num_samples_to_read);
            }
        }
        batcher.end_scan();
        let d_load = t_load_start.elapsed();
        info!("✅ Training Dataset Loaded: {} vectors. Time: {:.2?}", loaded_count, d_load);

        // --- PHASE 0: SUPER ROOT (Global Mean) --- Not needed
        // Level 0: The single center of the entire dataset.
        /*let t_p0_start = Instant::now();
        info!("🌍 [PHASE 0] Calculating Super Root (Global Mean)...");
        let mut super_root = vec![0.0f32; vector_dims as usize];
        for chunk in training_dataset.chunks(vector_dims as usize) {
            for (i, val) in chunk.iter().enumerate() {
                super_root[i] += val;
            }
        }


        let total_vecs_f32 = loaded_count as f32;
        for val in super_root.iter_mut() {
            *val /= total_vecs_f32;
        }
        let d_p0 = t_p0_start.elapsed();
        */

        // ========================================================================================
        // PHASE 1: MANUAL QUALITY-CONTROLLED ROOTS
        // ========================================================================================
        let t_p1_start = Instant::now();
        let num_attempts = kmeans_nredo;
        info!("🏗️ [PHASE 1] Starting Manual Quality-Control (Attempts: {}) ", num_attempts);

        // Initialize trackers OUTSIDE the loop scope
        let mut best_roots: Vec<f32> = Vec::new();
        let mut best_assignments: Vec<i32> = Vec::new();
        let mut lowest_max_bucket: usize = usize::MAX;

        for attempt in 1..=num_attempts {
            let attempt_start = Instant::now();

            // 1. Train candidate roots on GPU
            let candidate_roots = train_roots_gpu(
                &training_dataset,
                vector_dims,
                num_roots,
                100, // iterations
                1    // single redo (loop handles the rest)
            );

            // 2. PARTITION: Find assignments to evaluate this attempt
            let candidate_assignments = assign_to_roots_gpu(
                &training_dataset,
                &candidate_roots,
                vector_dims,
                num_roots
            );

            // 3. ANALYZE: Count bucket sizes
            let mut counts = vec![0usize; num_roots as usize];
            for &label in &candidate_assignments {
                if label >= 0 && (label as usize) < num_roots as usize {
                    counts[label as usize] += 1;
                }
            }

            let current_max = *counts.iter().max().unwrap_or(&usize::MAX);
            info!("  ↳ Attempt {}: Max Bucket = {} (Time: {:.2?})  ", attempt, current_max, attempt_start.elapsed());

            // 4. UPDATE: If this is the best distribution we've seen, save it
            if current_max < lowest_max_bucket {
                lowest_max_bucket = current_max;
                best_roots = candidate_roots;
                best_assignments = candidate_assignments;
            }
        }

        // Move the best results into the final variables
        let root_centroids = best_roots;
        let assignments = best_assignments;
        let d_p1 = t_p1_start.elapsed();
        info!("✅ Phase 1 Done. Selected Roots with Max Bucket: {}  ", lowest_max_bucket);

        // ========================================================================================
        // PHASE 2: PARTITION (Placeholder)
        // ========================================================================================
        // Assignments were already computed during the winning attempt of Phase 1
        let d_p2 = std::time::Duration::from_secs(0);
        info!("🔮 [PHASE 2] Reusing best assignments from Phase 1. Skipped 95s redundant call.");


        // --- PHASE 3: SCATTER & RESIDUAL TRAINING ---
        let t_p3_start = Instant::now();
        info!("🚀 [PHASE 3] Training Leaves on RESIDUALS (Matching CPU Logic)");

        let mut buckets: Vec<Vec<f32>> = vec![Vec::new(); num_roots as usize];

        // [RESIDUAL CHANGE]: Store residuals in buckets instead of raw vectors
        for (idx, &label) in assignments.iter().enumerate() {
            if label >= 0 && (label as usize) < num_roots as usize {
                let start = idx * vector_dims as usize;
                let end = start + vector_dims as usize;

                let root_start = label as usize * vector_dims as usize;
                let root_vec = &root_centroids[root_start..root_start + vector_dims as usize];
                let raw_vec = &training_dataset[start..end];

                // Compute: Vector - RootCentroid
                for j in 0..vector_dims as usize {
                    buckets[label as usize].push(raw_vec[j] - root_vec[j]);
                }
            }
        }

        let mut final_results: Vec<(Vec<f32>, i32)> = Vec::new();

        // 1. Store Roots (ID 0..399, Parent -1)
        for root_vec in root_centroids.chunks(vector_dims as usize) {
            final_results.push((root_vec.to_vec(), -1));
        }

        // 2. Train on Residuals & Convert back to Absolute
        let mut total_leaves_trained = 0;
        let mut missing_leaves = 0;
        let mut min_bucket = usize::MAX;
        let mut max_bucket = 0;

        let leaves_per_vector_ratio = num_leaves as f64 / loaded_count as f64;

        for (i, bucket_residuals) in buckets.iter().enumerate() {
            let n_vecs = bucket_residuals.len() / vector_dims as usize;
            let parent_id = i as i32;

            if n_vecs == 0 {
                missing_leaves += num_leaves / num_roots;
                continue;
            }
            if n_vecs < min_bucket { min_bucket = n_vecs; }
            if n_vecs > max_bucket { max_bucket = n_vecs; }

            let raw_target = n_vecs as f64 * leaves_per_vector_ratio;
            let mut target_leaves = raw_target.round() as u32;
            target_leaves = target_leaves.clamp(1, n_vecs as u32);

            // Train on the residuals
            let leaf_residuals = train_leaves_for_bucket_gpu(
                bucket_residuals,
                vector_dims,
                target_leaves,
                15
            );

            // [RESIDUAL CHANGE]: Add Root back to Residual to store absolute position
            let root_start = i * vector_dims as usize;
            let root_vec = &root_centroids[root_start..root_start + vector_dims as usize];

            for leaf_res_chunk in leaf_residuals.chunks(vector_dims as usize) {
                let mut absolute_leaf = vec![0.0f32; vector_dims as usize];
                for j in 0..vector_dims as usize {
                    absolute_leaf[j] = root_vec[j] + leaf_res_chunk[j];
                }
                final_results.push((absolute_leaf, parent_id));
            }
            total_leaves_trained += leaf_residuals.len() / vector_dims as usize;
        }
        let d_p3 = t_p3_start.elapsed();

        info!("🏁 [HIERARCHY COMPLETE] Trained {} Leaves. Total Centroids: {}  ", total_leaves_trained, final_results.len());
        info!("💾 Total Centroids (Roots+Leaves): {}  ", final_results.len());

        let t_store_start = Instant::now();
        centroids_table::store_centroids(final_results, centroid_table_name.clone(), vector_dims);
        let d_store = t_store_start.elapsed();

        // --- SUMMARY LOG ---
        info!(
            "\n⏱️  [TIMING SUMMARY]\n\
            \t• 📥 Data Loading:     {:.2?}  \n\
            \t• 🏗️ Phase 1 (Roots):  {:.2?}  \n\
            \t• 🔮 Phase 2 (Part.):  {:.2?}  \n\
            \t• 🌿 Phase 3 (Leaves): {:.2?}  \n\
            \t• 💾 Storage:          {:.2?}  \n\
            \t-----------------------------\n\
            \t📊 [SKEW & QUALITY REPORT]\n\
            \t• Min Bucket Size:     {}\n\
            \t• Max Bucket Size:     {}\n\
            \t• Leaves Trained:      {}\n\
            \t• Leaves Dropped:      {}\n\
            \t-----------------------------\n\
            \t👉 TOTAL TIME:         {:.2?}  ",
            d_load, d_p1, d_p2, d_p3, d_store,
            min_bucket, max_bucket, total_leaves_trained, missing_leaves,
            global_start.elapsed()
        );

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
