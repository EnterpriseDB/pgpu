use crate::clustering_gpu_impl::{
    create_gpu_resources, run_clustering_batch, run_clustering_consolidate,
    train_leaves_for_bucket_gpu, train_roots_and_assign_gpu,
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
    // PATH A: TOP-DOWN HIERARCHICAL BUILD
    // Triggered if `lists` has 2 elements (e.g. [400, 160000])
    //
    // Strategy:
    // 1. Train root centroids from sample
    // 2. Assign all vectors to their nearest root
    // 3. Train leaf centroids for each root's bucket
    // ========================================================================================

    if let Some(num_roots) = num_clusters_top_option {
        let num_leaves = num_clusters_leaf;
        let num_samples_target = (num_leaves as u64).saturating_mul(sampling_factor as u64);

        info!("🏗️ [TOP-DOWN BUILD] GPU-Accelerated Hierarchical Clustering");
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

            // Progress logging every 20 seconds
            let elapsed_since_log = last_log_time.elapsed().as_secs_f64();
            if elapsed_since_log >= 20.0 {
                let vectors_since_log = loaded_count - last_log_count;
                let rate = vectors_since_log as f64 / elapsed_since_log;
                let percent = (loaded_count as f64 / num_samples_target as f64) * 100.0;
                let remaining = num_samples_target as usize - loaded_count;
                let eta_secs = if rate > 0.0 { remaining as f64 / rate } else { 0.0 };

                // Build progress bar (30 chars wide)
                let bar_width = 30;
                let filled = ((percent / 100.0) * bar_width as f64) as usize;
                let empty = bar_width - filled;
                let bar: String = "█".repeat(filled) + &"░".repeat(empty);

                // Format rate (K or M)
                let rate_str = if rate >= 1_000_000.0 {
                    format!("{:.1}M", rate / 1_000_000.0)
                } else {
                    format!("{:.0}K", rate / 1_000.0)
                };

                // Format ETA (m:ss or just seconds)
                let eta_str = if eta_secs >= 60.0 {
                    format!("{}m {:02.0}s", (eta_secs / 60.0) as u32, eta_secs % 60.0)
                } else {
                    format!("{:.0}s", eta_secs)
                };

                // Format counts (M suffix for millions)
                let loaded_str = if loaded_count >= 1_000_000 {
                    format!("{:.1}M", loaded_count as f64 / 1_000_000.0)
                } else {
                    format!("{}K", loaded_count / 1000)
                };
                let target_str = if num_samples_target >= 1_000_000 {
                    format!("{:.1}M", num_samples_target as f64 / 1_000_000.0)
                } else {
                    format!("{}K", num_samples_target / 1000)
                };

                info!(
                    "[{}] {:.1}% ({}/{}) | {} vec/s | ETA: {}",
                    bar, percent, loaded_str, target_str, rate_str, eta_str
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
        // STEP 2: Normalize training data (if spherical)
        // ================================================================================
        // NOTE: cuVS k-means only supports L2 distance, so we normalize beforehand
        let t_norm_start = Instant::now();
        if spherical_centroids {
            info!("📐 Normalizing {} training vectors...", loaded_count);
            for chunk in training_dataset.chunks_mut(vector_dims as usize) {
                let mut norm_sq = 0.0f32;
                for x in chunk.iter() { norm_sq += x * x; }
                let norm = norm_sq.sqrt();
                if norm > 1e-6 {
                    for x in chunk.iter_mut() { *x /= norm; }
                }
            }
        }
        let d_norm = t_norm_start.elapsed();

        // ================================================================================
        // STEP 3+4: Train root centroids AND assign vectors (combined for GPU stability)
        // ================================================================================
        let (root_centroids, assignments, d_roots, d_assign) = train_roots_and_assign_gpu(
            &training_dataset,
            vector_dims,
            num_roots,
            kmeans_iterations,
            spherical_centroids,
        );

        // Analyze bucket distribution
        let mut bucket_counts = vec![0usize; num_roots as usize];
        for &label in &assignments {
            if label >= 0 && (label as usize) < num_roots as usize {
                bucket_counts[label as usize] += 1;
            }
        }
        let min_bucket = *bucket_counts.iter().min().unwrap_or(&0);
        let max_bucket = *bucket_counts.iter().max().unwrap_or(&0);
        info!("   Bucket distribution: min={}, max={}, ratio={:.2}x",
              min_bucket, max_bucket, max_bucket as f64 / (min_bucket.max(1) as f64));

        // ================================================================================
        // STEP 5: Build buckets and train leaves
        // ================================================================================
        let t_leaves_start = Instant::now();
        info!("🚀 [PHASE 3] Training leaves for {} buckets...", num_roots);

        // Build index-based buckets (store indices, not vectors - saves ~120GB RAM)
        let mut bucket_indices: Vec<Vec<usize>> = vec![Vec::new(); num_roots as usize];
        for (idx, &label) in assignments.iter().enumerate() {
            if label >= 0 && (label as usize) < num_roots as usize {
                bucket_indices[label as usize].push(idx);
            }
        }

        // Calculate leaves per bucket proportionally
        let leaves_per_vector_ratio = if loaded_count > 0 {
            num_leaves as f64 / loaded_count as f64
        } else {
            0.0
        };

        let mut final_results: Vec<(Vec<f32>, i32)> = Vec::new();

        // Add root centroids (parent_id = -1)
        for root_vec in root_centroids.chunks(vector_dims as usize) {
            final_results.push((root_vec.to_vec(), -1));
        }

        // Prepare work items for parallel processing
        let work_items: Vec<(usize, u32)> = bucket_indices
            .iter()
            .enumerate()
            .filter_map(|(bucket_idx, indices)| {
                let n_vecs = indices.len();
                if n_vecs == 0 {
                    return None;
                }
                let raw_target = n_vecs as f64 * leaves_per_vector_ratio;
                let target_leaves = (raw_target.round() as u32).clamp(1, n_vecs as u32);
                Some((bucket_idx, target_leaves))
            })
            .collect();

        // Parallel leaf training with multiple GPU streams
        let num_workers = 8; // 8 concurrent GPU streams
        let work_counter = std::sync::atomic::AtomicUsize::new(0);
        let completed_counter = std::sync::atomic::AtomicUsize::new(0);
        let total_work = work_items.len();

        info!("   Using {} parallel GPU workers for {} buckets", num_workers, total_work);

        let all_leaf_results: Vec<(usize, Vec<f32>)> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..num_workers)
                .map(|_worker_id| {
                    s.spawn(|| {
                        // Each worker gets its own GPU Resources (own CUDA stream)
                        let gpu_res = create_gpu_resources();
                        let mut results: Vec<(usize, Vec<f32>)> = Vec::new();

                        loop {
                            // Atomically grab next work item
                            let idx = work_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            if idx >= total_work {
                                break;
                            }

                            let (bucket_idx, target_leaves) = work_items[idx];
                            let indices = &bucket_indices[bucket_idx];

                            let leaf_centroids = train_leaves_for_bucket_gpu(
                                &gpu_res,
                                &training_dataset,
                                indices,
                                vector_dims,
                                target_leaves,
                                kmeans_iterations / 2,
                                spherical_centroids,
                            );

                            results.push((bucket_idx, leaf_centroids));

                            // Progress logging (eprintln is thread-safe, pgrx macros are not)
                            let done = completed_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                            if done % 50 == 0 || done == total_work {
                                eprintln!("   ... trained {}/{} buckets ({:.0}%)",
                                    done, total_work, 100.0 * done as f64 / total_work as f64);
                            }
                        }

                        results
                    })
                })
                .collect();

            // Collect results from all workers
            let mut all_results = Vec::new();
            for (worker_id, h) in handles.into_iter().enumerate() {
                match h.join() {
                    Ok(results) => all_results.extend(results),
                    Err(e) => {
                        // Worker panicked - log error after threads complete
                        eprintln!("Worker {} panicked: {:?}", worker_id, e);
                    }
                }
            }
            all_results
        });

        // Sort by bucket_idx and add to final results
        let mut sorted_results = all_leaf_results;
        sorted_results.sort_by_key(|(idx, _)| *idx);

        let mut total_leaves_trained = 0;
        for (bucket_idx, leaf_centroids) in sorted_results {
            let parent_id = bucket_idx as i32;
            for leaf_vec in leaf_centroids.chunks(vector_dims as usize) {
                final_results.push((leaf_vec.to_vec(), parent_id));
            }
            total_leaves_trained += leaf_centroids.len() / vector_dims as usize;
        }
        let d_leaves = t_leaves_start.elapsed();

        // ================================================================================
        // STEP 6: Store centroids
        // ================================================================================
        let t_store_start = Instant::now();
        info!("💾 Storing {} centroids ({} roots + {} leaves)...",
              final_results.len(), num_roots, total_leaves_trained);

        centroids_table::store_centroids(final_results, centroid_table_name.clone(), vector_dims);
        let d_store = t_store_start.elapsed();

        // ================================================================================
        // SUMMARY
        // ================================================================================
        info!(
            "\n⏱️  [TIMING SUMMARY - TOP-DOWN]\n\
            \t• 📥 Data Loading:     {:.2?}\n\
            \t• 📐 Normalization:    {:.2?}\n\
            \t• 🌳 Root Training:    {:.2?}\n\
            \t• 📍 Assignment:       {:.2?}\n\
            \t• 🌿 Leaf Training:    {:.2?}\n\
            \t• 💾 Storage:          {:.2?}\n\
            \t-----------------------------\n\
            \t📊 [RESULTS]\n\
            \t• Vectors Sampled:     {}\n\
            \t• Root Centroids:      {}\n\
            \t• Leaf Centroids:      {}\n\
            \t• Min/Max Bucket:      {}/{}\n\
            \t-----------------------------\n\
            \t👉 TOTAL TIME:         {:.2?}",
            d_load,
            d_norm,
            d_roots,
            d_assign,
            d_leaves,
            d_store,
            loaded_count,
            num_roots,
            total_leaves_trained,
            min_bucket, max_bucket,
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