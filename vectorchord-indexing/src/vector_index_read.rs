use crate::vector_type;
use pgrx::{debug1, info, warning, Spi};
use std::time::{SystemTime, UNIX_EPOCH};
use std::time::Instant;

pub struct VectorReadBatcher {
    qualified_table_name: String,
    column_name: String,

    // Sampling State
    target_samples: u64,
    vectors_read: u64,

    // The list of Block Numbers we intend to read
    blocks_to_read: Vec<u64>,
    current_block_idx: usize,

    // Constants
    blocks_per_query: usize,
    active: bool,
}

impl VectorReadBatcher {
    pub fn new(
        qualified_table_name: String,
        column_name: String,
        num_clusters: u32,
        sampling_factor: u32,
        _requested_batch_size: u64,
    ) -> Self {
        // 1. Calculate Target
        let target_samples = (num_clusters as u64).saturating_mul(sampling_factor as u64);

        // 2. Get Physical Size (Robust, No ANALYZE needed)
        let table_bytes: i64 = Spi::get_one(&format!(
            "SELECT pg_relation_size('{}'::regclass)",
            qualified_table_name
        )).expect("SPI failed to look up table size").unwrap_or(0);

        let total_blocks = (table_bytes / 8192).max(1) as u64;

        // 3. Estimate Density (Vectors per Block)
        let est_vectors_per_block = 100;

        let blocks_needed = (target_samples / est_vectors_per_block) + 1;

        // Safety Buffer: Read 20% extra blocks
        let blocks_to_queue = (blocks_needed as f64 * 1.2) as u64;
        let blocks_to_queue = blocks_to_queue.clamp(1, total_blocks);

        info!("🔍 [SAMPLER] Physical Table: {} MB ({} Blocks)", table_bytes / 1024 / 1024, total_blocks);
        info!("🔍 [SAMPLER] Target: {} vectors. Plan: Read {} random blocks (approx {:.2}%)",
            target_samples, blocks_to_queue, (blocks_to_queue as f64 / total_blocks as f64) * 100.0);

        // 4. Generate Random Block List (Manual Shuffle)
        let mut all_blocks: Vec<u64> = (0..total_blocks).collect();

        let mut seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let len = all_blocks.len();
        if len > 1 {
            for i in (1..len).rev() {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;

                let rnd = seed as usize;
                let j = rnd % (i + 1);
                all_blocks.swap(i, j);
            }
        }

        let blocks_to_read = all_blocks.into_iter().take(blocks_to_queue as usize).collect();

        VectorReadBatcher {
            qualified_table_name,
            column_name,
            target_samples,
            vectors_read: 0,
            blocks_to_read,
            current_block_idx: 0,
            blocks_per_query: 50,
            active: true,
        }
    }

    pub(crate) fn next_batch(&mut self) -> Option<(Vec<f32>, u32)> {
        // Stop if we have enough vectors OR run out of blocks
        if !self.active || self.vectors_read >= self.target_samples {
            return None;
        }
        if self.current_block_idx >= self.blocks_to_read.len() {
            info!("⚠️ [SAMPLER] Exhausted all queued blocks. (Read: {} / Target: {})", self.vectors_read, self.target_samples);
            self.active = false;
            return None;
        }

        let _start_time = Instant::now();

        // 1. Get the next chunk of Block IDs
        let end_idx = (self.current_block_idx + self.blocks_per_query).min(self.blocks_to_read.len());
        let batch_blocks = &self.blocks_to_read[self.current_block_idx..end_idx];
        self.current_block_idx = end_idx;

        // 2. Build the Direct Block Access Query (TID Scan)
        let values_list = batch_blocks.iter()
            .map(|b| format!("({})", b))
            .collect::<Vec<_>>()
            .join(",");

        // FIX: Added parenthesis concat for valid TID syntax: '(' || blk || ',0)'
        let query = format!(
            "WITH target_blocks(blk) AS (VALUES {}) \
             SELECT t.{} \
             FROM {} t \
             JOIN target_blocks b \
             ON t.ctid >= ('(' || b.blk::text || ',0)')::tid \
             AND t.ctid < ('(' || (b.blk + 1)::text || ',0)')::tid",
             values_list,
             self.column_name,
             self.qualified_table_name
        );

        // 3. Execute
        let (all_vectors, dims, read_count) = Spi::connect(|client| {
            let mut vectors: Vec<f32> = Vec::new();
            let mut dims: u32 = 0;
            let mut count = 0;

            let table = client.select(&query, None, &[])
                .expect("Failed to execute TID block scan");

            for row in table {
                if let Ok(entry) = row.get_datum_by_ordinal(1) {
                    if let Ok(Some(datum)) = entry.value::<pgrx::pg_sys::Datum>() {
                        unsafe {
                            let raw_ptr = datum.cast_mut_ptr();
                            let detoasted_ptr = pgrx::pg_sys::pg_detoast_datum(raw_ptr);
                            let byte_slice = pgrx::varlena_to_byte_slice(detoasted_ptr);
                            let (vec_vals, vec_dims) = vector_type::decode_pgvector_vector(byte_slice);

                            if dims == 0 { dims = vec_dims; }
                            vectors.extend_from_slice(&vec_vals);
                            count += 1;

                            if detoasted_ptr != raw_ptr {
                                pgrx::pg_sys::pfree(detoasted_ptr as *mut std::ffi::c_void);
                            }
                        }
                    }
                }
            }
            (vectors, dims, count)
        });

        self.vectors_read += read_count as u64;

        debug1!("✅ Scanned {} blocks -> {} vectors ({:.2?})", batch_blocks.len(), read_count, _start_time.elapsed());

        Some((all_vectors, dims))
    }

    pub(crate) fn end_scan(&mut self) {
        self.active = false;
    }
}