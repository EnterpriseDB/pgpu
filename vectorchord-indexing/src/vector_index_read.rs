use crate::vector_type;
use pgrx::{info, Spi};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct VectorReadBatcher {
    qualified_table_name: String,
    column_name: String,
    target_samples: u64,
    vectors_read: u64,
    blocks_to_read: Vec<u64>,
    current_block_idx: usize,
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

        // 2. Metadata & Setup (Fixed Datum Types)
        let (total_blocks, dims) = Spi::connect(|client| {
            // A. Enable Bitmap Prefetching (Critical for Speed)
            let _ = client.select("SET LOCAL effective_io_concurrency = 100", None, None);

            // B. Get Table Size (in Bytes)
            let size_query = format!("SELECT pg_relation_size('{}'::regclass)", qualified_table_name);
            let table_bytes: i64 = match client.select(&size_query, None, None) {
                Ok(table) => {
                    if let Some(row) = table.first().table {
                         row.get_datum_by_ordinal(1).unwrap().value::<i64>().unwrap().unwrap_or(0)
                    } else { 0 }
                },
                Err(_) => 0,
            };

            // C. Get Column Dimensions
            let dim_query = format!(
                "SELECT a.atttypmod \
                 FROM pg_attribute a \
                 WHERE a.attrelid = '{}'::regclass AND a.attname = '{}'",
                qualified_table_name, column_name
            );

            let mut found_dims = 0;
            if let Ok(table) = client.select(&dim_query, None, None) {
                if let Some(row) = table.first().table {
                    if let Ok(Some(val)) = row.get_datum_by_ordinal(1).unwrap().value::<i32>() {
                        if val > 0 { found_dims = val as u64; }
                    }
                }
            }

            ((table_bytes / 8192).max(1) as u64, found_dims)
        });

        // 3. Density Calc
        // Constants
        const BLOCK_SIZE: u64 = 8192;
        const PAGE_HEADER: u64 = 24;
        const TUPLE_HEADER: u64 = 24;
        const LINE_POINTER: u64 = 4;
        const TOAST_PTR_SIZE: u64 = 18;

        let vec_byte_size = dims * 4;
        let is_toasted = vec_byte_size > 2000 || vec_byte_size == 0;
        let data_size = if is_toasted { TOAST_PTR_SIZE } else { vec_byte_size };
        let row_footprint = TUPLE_HEADER + data_size + LINE_POINTER;
        let density = (BLOCK_SIZE - PAGE_HEADER) / row_footprint;

        info!("🔍 [SAMPLER] Table Blocks: {}. Mode: {}. Density: {} rows/blk",
            total_blocks, if is_toasted { "TOAST" } else { "INLINE" }, density);

        // 4. Plan the Read
        let blocks_needed = (target_samples / density) + 1;
        let blocks_to_queue = ((blocks_needed as f64 * 1.1) as u64).clamp(1, total_blocks);

        info!("🔍 [SAMPLER] Target: {} vectors. Plan: Read {} blocks ({:.2}%)",
            target_samples, blocks_to_queue, (blocks_to_queue as f64 / total_blocks as f64) * 100.0);

        // 5. Generate Random Block List
        let blocks_to_read = generate_shuffled_blocks(total_blocks, blocks_to_queue);

        VectorReadBatcher {
            qualified_table_name,
            column_name,
            target_samples,
            vectors_read: 0,
            blocks_to_read,
            current_block_idx: 0,
            // 5000 blocks = ~100MB-200MB per batch.
            // This is large enough to make SQL parsing overhead negligible.
            blocks_per_query: 5000,
            active: true,
        }
    }

    pub(crate) fn next_batch(&mut self) -> Option<(Vec<f32>, u32)> {
        if !self.active || self.vectors_read >= self.target_samples {
            return None;
        }
        if self.current_block_idx >= self.blocks_to_read.len() {
             info!("⚠️ [SAMPLER] Exhausted queued blocks.");
             self.active = false;
             return None;
        }

        // 1. Get batch of block IDs
        let end_idx = (self.current_block_idx + self.blocks_per_query).min(self.blocks_to_read.len());
        let batch_blocks = &self.blocks_to_read[self.current_block_idx..end_idx];
        self.current_block_idx = end_idx;

        // 2. Format for ARRAY construction
        let block_array_str = batch_blocks.iter()
            .map(|b| b.to_string())
            .collect::<Vec<_>>()
            .join(",");

        // 3. Optimized Query
        // JOIN UNNEST(ARRAY) is effectively a Bitmap Scan on TID ranges.
        // With effective_io_concurrency=100, Postgres will prefetch these pages.
        let query = format!(
            "SELECT t.{} \
             FROM {} t \
             JOIN UNNEST(ARRAY[{}]) AS blk_id \
             ON t.ctid >= ('(' || blk_id::text || ',0)')::tid \
             AND t.ctid < ('(' || (blk_id + 1)::text || ',0)')::tid",
             self.column_name,
             self.qualified_table_name,
             block_array_str
        );

        let (all_vectors, dims, read_count) = Spi::connect(|client| {
            // Pre-allocate to avoid reallocations
            let mut vectors: Vec<f32> = Vec::with_capacity(batch_blocks.len() * 150 * 768);
            let mut dims: u32 = 0;
            let mut count = 0;

            if let Ok(table) = client.select(&query, None, None) {
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
            }
            (vectors, dims, count)
        });

        self.vectors_read += read_count as u64;
        Some((all_vectors, dims))
    }

    pub(crate) fn end_scan(&mut self) {
        self.active = false;
    }
}

// --- Helper: Dependency-Free Shuffle ---
fn generate_shuffled_blocks(total_blocks: u64, limit: u64) -> Vec<u64> {
    let mut all_blocks: Vec<u64> = (0..total_blocks).collect();
    let mut seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64;
    let len = all_blocks.len();
    if len > 1 {
        for i in (1..len).rev() {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            let rnd = seed as usize;
            all_blocks.swap(i, rnd % (i + 1));
        }
    }
    all_blocks.into_iter().take(limit as usize).collect()
}