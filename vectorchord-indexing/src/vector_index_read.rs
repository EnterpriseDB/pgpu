use crate::vector_type;
use pgrx::{debug1, info, warning, pg_sys, Spi}; // Added pg_sys
use std::time::{SystemTime, UNIX_EPOCH};
use std::ptr;

pub struct VectorReadBatcher {
    // We hold the Relation pointer open (Unsafe C Pointer)
    relation: pg_sys::Relation,

    target_samples: u64,
    vectors_read: u64,

    blocks_to_read: Vec<u64>,
    current_block_idx: usize,

    // The exact column index (1-based) for the vector data
    vector_att_num: i16,

    prefetch_distance: usize,
    active: bool,
}

impl VectorReadBatcher {
    pub fn new(
        qualified_table_name: String,
        column_name: String,
        num_clusters: u32,
        sampling_factor: u32,
        _batch_size: u64,
    ) -> Self {
        // 1. SAFE SETUP: Use SQL to get Metadata (Robustness)
        let (rel_oid_res, total_blocks_res, att_num_res) = Spi::connect(|client| {
            // A. Resolve Table OID
            let oid: i64 = client.select(&format!("SELECT '{}'::regclass::oid", qualified_table_name), None, &[])
                .unwrap().first().get_datum_by_ordinal(1).unwrap().unwrap();

            // B. Get Physical Block Count
            let bytes: i64 = client.select(&format!("SELECT pg_relation_size({})", oid), None, &[])
                .unwrap().first().get_datum_by_ordinal(1).unwrap().unwrap();

            // C. Get Attribute Number (1-based index of column)
            let att: i16 = client.select(&format!(
                    "SELECT attnum FROM pg_attribute WHERE attrelid = {} AND attname = '{}'",
                    oid, column_name), None, &[])
                .expect("Column lookup failed").first().get_datum_by_ordinal(1).unwrap().unwrap();

            (oid as u32, (bytes / 8192).max(1) as u64, att)
        });

        let target_samples = (num_clusters as u64).saturating_mul(sampling_factor as u64);

        info!("🔍 [FAST-SCAN] Opening Table OID: {}. Column '{}' is Attribute #{}.", rel_oid_res, column_name, att_num_res);

        unsafe {
            // 2. UNSAFE OPEN: Open relation directly in C
            // AccessShareLock is the correct lock for reading (blocks DROP but allows Writes)
            let relation = pg_sys::table_open(rel_oid_res, pg_sys::AccessShareLock as i32);

            // 3. Plan the Read
            // Density estimate: 150 rows/block
            let blocks_needed = (target_samples / 150) + 1;
            let blocks_to_queue = (blocks_needed as f64 * 1.1) as u64;
            let blocks_to_queue = blocks_to_queue.clamp(1, total_blocks_res);

            info!("🔍 [FAST-SCAN] Plan: Scanning {} / {} blocks ({:.2}%)",
                blocks_to_queue, total_blocks_res, (blocks_to_queue as f64 / total_blocks_res as f64) * 100.0);

            let blocks_to_read = generate_shuffled_blocks(total_blocks_res, blocks_to_queue);

            // Log first 5 blocks for sanity check
            if !blocks_to_read.is_empty() {
                let sample: Vec<String> = blocks_to_read.iter().take(5).map(|x| x.to_string()).collect();
                info!("🔍 [DEBUG] First 5 Blocks: {:?}", sample);
            }

            VectorReadBatcher {
                relation,
                target_samples,
                vectors_read: 0,
                blocks_to_read,
                current_block_idx: 0,
                vector_att_num: att_num_res,
                prefetch_distance: 20,
                active: true,
            }
        }
    }

    pub(crate) fn next_batch(&mut self) -> Option<(Vec<f32>, u32)> {
        if !self.active || self.vectors_read >= self.target_samples {
            if self.active { self.end_scan(); }
            return None;
        }
        if self.current_block_idx >= self.blocks_to_read.len() {
             info!("⚠️ [FAST-SCAN] Exhausted queued blocks.");
             self.end_scan();
             return None;
        }

        unsafe {
            let mut vectors: Vec<f32> = Vec::with_capacity(100_000 * 768);
            let mut dims: u32 = 0;
            let mut vectors_in_batch = 0;

            // Process up to 500 blocks per call
            let batch_limit_blocks = 500;
            let mut blocks_processed = 0;

            debug1!("🚀 [DEBUG] Starting batch. Current Block Idx: {}", self.current_block_idx);

            while blocks_processed < batch_limit_blocks && self.current_block_idx < self.blocks_to_read.len() {

                // --- PREFETCHING (Async I/O) ---
                let prefetch_idx = self.current_block_idx + self.prefetch_distance;
                if prefetch_idx < self.blocks_to_read.len() {
                    let blk_prefetch = self.blocks_to_read[prefetch_idx] as u32;
                    // Check if PrefetchBuffer exists in your binding (Standard PG >= 9.x)
                    // If this fails to compile, comment it out.
                    pg_sys::PrefetchBuffer(self.relation, 0, blk_prefetch);
                }

                let blk_num = self.blocks_to_read[self.current_block_idx] as u32;
                self.current_block_idx += 1;
                blocks_processed += 1;

                // 1. Read Buffer (Load Page)
                let buffer = pg_sys::ReadBuffer(self.relation, blk_num);

                // 2. Lock Buffer (Share)
                pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_SHARE as i32);

                // 3. Get Page
                let page = pg_sys::BufferGetPage(buffer);
                let max_off = pg_sys::PageGetMaxOffsetNumber(page);

                // 4. Iterate Items
                for off in 1..=max_off {
                    let item_id = pg_sys::PageGetItemId(page, off);

                    // ItemIdIsUsed checks if the line pointer is not empty/dead
                    if pg_sys::ItemIdIsUsed(item_id) {
                        let item = pg_sys::PageGetItem(page, item_id);

                        // Access Tuple Descriptor from Relation
                        let tup_desc = (*self.relation).rd_att;

                        let mut is_null = false;

                        // getattr is a Postgres C function that handles tuple offset math
                        // It uses the TupleDesc to know where column N starts
                        let datum = pg_sys::getattr(
                            item, // pointer to tuple data
                            self.vector_att_num.into(),
                            tup_desc,
                            &mut is_null
                        );

                        if !is_null {
                            // Decode directly from raw pointer
                            let raw_ptr = datum as *mut pg_sys::varlena;
                            let detoasted_ptr = pg_sys::pg_detoast_datum(raw_ptr);
                            let byte_slice = pgrx::varlena_to_byte_slice(detoasted_ptr);

                            let (vec_vals, vec_dims) = vector_type::decode_pgvector_vector(byte_slice);

                            if dims == 0 { dims = vec_dims; }
                            vectors.extend_from_slice(&vec_vals);
                            vectors_in_batch += 1;

                            // Memory cleanup for TOAST
                            if detoasted_ptr != raw_ptr {
                                pg_sys::pfree(detoasted_ptr as *mut std::ffi::c_void);
                            }
                        }
                    }
                }

                // 5. Release Buffer
                pg_sys::UnlockReleaseBuffer(buffer);
            }

            self.vectors_read += vectors_in_batch;
            // debug1!("✅ Batch Done. Processed {} blocks. Vectors read: {}", blocks_processed, vectors_in_batch);

            Some((vectors, dims))
        }
    }

    pub(crate) fn end_scan(&mut self) {
        if self.active {
            unsafe {
                // Release the table lock
                pg_sys::table_close(self.relation, pg_sys::AccessShareLock as i32);
            }
            self.active = false;
            info!("🔒 [FAST-SCAN] Relation closed.");
        }
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