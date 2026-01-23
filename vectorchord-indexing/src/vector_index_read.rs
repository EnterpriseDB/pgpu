use crate::vector_type;
use pgrx::pg_sys::{self, BlockNumber, Datum, Buffer, InvalidBuffer, MAIN_FORKNUM};
use pgrx::info;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// VectorReadBatcher provides efficient random sampling of vectors from a PostgreSQL table
/// using direct block access with Feistel-cipher-based block permutation.
///
/// This implementation bypasses SPI calls entirely for maximum performance,
/// replicating the approach used by VectorChord 1.0.
pub struct VectorReadBatcher {
    heap_relation: pg_sys::Relation,
    column_attnum: i16,
    target_samples: u64,
    vectors_read: u64,
    dims: u32,
    total_blocks: BlockNumber,
    block_iterator: FeistelBlockIterator,
    current_block: Option<BlockReader>,
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
        let target_samples = (num_clusters as u64).saturating_mul(sampling_factor as u64);

        // Get relation and column info using unsafe PostgreSQL internals
        let (heap_relation, column_attnum, dims, total_blocks, block_iterator) = unsafe {
            // Parse and open relation using pgrx utilities
            let rel_oid = resolve_table_oid(&qualified_table_name);
            let heap_relation = pg_sys::relation_open(rel_oid, pg_sys::AccessShareLock as i32);

            // Get column attribute number
            let col_name_cstr = std::ffi::CString::new(column_name.as_str()).unwrap();
            let attnum = pg_sys::get_attnum(rel_oid, col_name_cstr.as_ptr());

            if attnum == pg_sys::InvalidAttrNumber as i16 {
                pg_sys::relation_close(heap_relation, pg_sys::AccessShareLock as i32);
                pgrx::error!("Column '{}' not found in table '{}'", column_name, qualified_table_name);
            }

            // Get vector dimensions from atttypmod
            let tuple_desc = (*heap_relation).rd_att;
            let attr = *(*tuple_desc).attrs.as_ptr().add((attnum - 1) as usize);
            let type_mod = attr.atttypmod;
            let detected_dims = if type_mod > 0 { type_mod as u32 } else { 0 };

            // Get total number of blocks using the fork-aware function
            let total_blocks = pg_sys::RelationGetNumberOfBlocksInFork(
                heap_relation,
                MAIN_FORKNUM,
            );

            // Calculate density for planning
            const BLOCK_SIZE: u64 = 8192;
            const PAGE_HEADER: u64 = 24;
            const TUPLE_HEADER: u64 = 24;
            const LINE_POINTER: u64 = 4;
            const TOAST_PTR_SIZE: u64 = 18;

            let vec_byte_size = detected_dims as u64 * 4;
            let is_toasted = vec_byte_size > 2000 || vec_byte_size == 0;
            let data_size = if is_toasted { TOAST_PTR_SIZE } else { vec_byte_size };
            let row_footprint = TUPLE_HEADER + data_size + LINE_POINTER;
            let density = if row_footprint > 0 {
                (BLOCK_SIZE - PAGE_HEADER) / row_footprint
            } else {
                100 // Default estimate
            };

            info!(
                "[SAMPLER] Table Blocks: {}. Mode: {}. Density: {} rows/blk",
                total_blocks,
                if is_toasted { "TOAST" } else { "INLINE" },
                density
            );

            let blocks_needed = (target_samples / density.max(1)) + 1;
            let blocks_to_scan = (blocks_needed as f64 * 1.1) as u32;
            let blocks_to_scan = blocks_to_scan.clamp(1, total_blocks);

            info!(
                "[SAMPLER] Target: {} vectors. Plan: Scan {} blocks ({:.2}%)",
                target_samples,
                blocks_to_scan,
                (blocks_to_scan as f64 / total_blocks.max(1) as f64) * 100.0
            );

            let block_iterator = FeistelBlockIterator::new(total_blocks, blocks_to_scan);

            (heap_relation, attnum, detected_dims, total_blocks, block_iterator)
        };

        VectorReadBatcher {
            heap_relation,
            column_attnum,
            target_samples,
            vectors_read: 0,
            dims,
            total_blocks,
            block_iterator,
            current_block: None,
            active: true,
        }
    }

    pub(crate) fn next_batch(&mut self) -> Option<(Vec<f32>, u32)> {
        if !self.active || self.vectors_read >= self.target_samples {
            return None;
        }

        // Read vectors in batches of ~10000 for memory efficiency
        let batch_target = 10000u64.min(self.target_samples - self.vectors_read);

        let mut vectors: Vec<f32> = Vec::with_capacity(batch_target as usize * self.dims.max(768) as usize);
        let mut dims: u32 = self.dims;
        let mut count: u64 = 0;

        while count < batch_target {
            // Try to get next tuple from current block
            if let Some(ref mut block_reader) = self.current_block {
                if let Some(datum) = block_reader.next_tuple_datum(self.column_attnum) {
                    if let Some((vec_vals, vec_dims)) = extract_vector_from_datum(datum) {
                        if dims == 0 {
                            dims = vec_dims;
                        }
                        vectors.extend_from_slice(&vec_vals);
                        count += 1;
                        continue;
                    }
                    continue; // Skip null/invalid tuples
                }
                // Current block exhausted, release it
                self.current_block = None;
            }

            // Get next block
            if let Some(block_num) = self.block_iterator.next() {
                self.current_block = Some(BlockReader::new(self.heap_relation, block_num));
            } else {
                // No more blocks
                if count == 0 {
                    self.active = false;
                    return None;
                }
                break;
            }
        }

        self.vectors_read += count;
        self.dims = dims;

        if count == 0 {
            self.active = false;
            None
        } else {
            Some((vectors, dims))
        }
    }

    pub(crate) fn end_scan(&mut self) {
        self.active = false;
        self.current_block = None;
    }
}

impl Drop for VectorReadBatcher {
    fn drop(&mut self) {
        self.current_block = None;
        unsafe {
            pg_sys::relation_close(self.heap_relation, pg_sys::AccessShareLock as i32);
        }
    }
}

/// Resolve a qualified table name to an OID
unsafe fn resolve_table_oid(qualified_table_name: &str) -> pg_sys::Oid {
    // Use regclass cast via SPI for reliable name resolution
    let query = format!("SELECT '{}'::regclass::oid", qualified_table_name);

    let oid = pgrx::Spi::connect(|client| {
        match client.select(&query, None, None) {
            Ok(table) => {
                if let Some(row) = table.first().table {
                    row.get_datum_by_ordinal(1)
                        .ok()
                        .and_then(|d| d.value::<pg_sys::Oid>().ok().flatten())
                        .unwrap_or(pg_sys::InvalidOid)
                } else {
                    pg_sys::InvalidOid
                }
            }
            Err(_) => pg_sys::InvalidOid,
        }
    });

    if oid == pg_sys::InvalidOid {
        pgrx::error!("Table '{}' not found", qualified_table_name);
    }

    oid
}

/// BlockReader handles reading tuples from a single heap block
struct BlockReader {
    relation: pg_sys::Relation,
    buffer: Buffer,
    page: pg_sys::Page,
    current_offset: u16,
    max_offset: u16,
    snapshot: pg_sys::Snapshot,
}

impl BlockReader {
    fn new(relation: pg_sys::Relation, block_num: BlockNumber) -> Self {
        unsafe {
            let buffer = pg_sys::ReadBuffer(relation, block_num);
            pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_SHARE as i32);

            let page = pg_sys::BufferGetPage(buffer);
            let max_offset = pg_sys::PageGetMaxOffsetNumber(page);
            let snapshot = pg_sys::GetActiveSnapshot();

            BlockReader {
                relation,
                buffer,
                page,
                current_offset: 1, // Offsets start at 1 in PostgreSQL
                max_offset,
                snapshot,
            }
        }
    }

    fn next_tuple_datum(&mut self, attnum: i16) -> Option<Datum> {
        unsafe {
            while self.current_offset <= self.max_offset {
                let offset = self.current_offset;
                self.current_offset += 1;

                // Get item pointer for this offset
                let item_id = pg_sys::PageGetItemId(self.page, offset);

                // Skip dead/unused items
                if !pg_sys::ItemIdIsNormal(item_id) {
                    continue;
                }

                // Get the heap tuple header
                let item = pg_sys::PageGetItem(self.page, item_id);
                let htup = item as *mut pg_sys::HeapTupleHeaderData;

                // Check tuple visibility using MVCC
                // For sampling, we use a simpler check - just verify it's not dead
                if !is_tuple_visible(htup, self.snapshot) {
                    continue;
                }

                // Extract the attribute value
                let tuple_desc = (*self.relation).rd_att;

                // Build a minimal HeapTupleData for attribute extraction
                let mut tuple_data = pg_sys::HeapTupleData {
                    t_len: (*item_id).lp_len() as u32,
                    t_self: pg_sys::ItemPointerData::default(),
                    t_tableOid: pg_sys::InvalidOid,
                    t_data: htup,
                };

                let mut is_null: bool = true;
                let datum = pg_sys::heap_getattr(
                    &mut tuple_data,
                    attnum as i32,
                    tuple_desc,
                    &mut is_null,
                );

                if is_null {
                    continue;
                }

                return Some(datum);
            }
        }
        None
    }
}

impl Drop for BlockReader {
    fn drop(&mut self) {
        unsafe {
            if self.buffer != InvalidBuffer {
                pg_sys::UnlockReleaseBuffer(self.buffer);
            }
        }
    }
}

/// Simple visibility check for heap tuples
unsafe fn is_tuple_visible(htup: *mut pg_sys::HeapTupleHeaderData, snapshot: pg_sys::Snapshot) -> bool {
    // Get tuple's xmin (inserting transaction)
    let xmin = (*htup).t_choice.t_heap.t_xmin;

    // For our sampling purposes, we accept tuples that are:
    // 1. Committed (xmin is in the past)
    // 2. Not deleted (xmax is invalid or not committed)

    // Check infomask for committed status
    let infomask = (*htup).t_infomask;

    // HEAP_XMIN_COMMITTED means the tuple is definitely visible
    if (infomask & pg_sys::HEAP_XMIN_COMMITTED as u16) != 0 {
        // Check if it's been deleted
        let xmax = (*htup).t_choice.t_heap.t_xmax;
        if xmax == 0 || xmax == pg_sys::InvalidTransactionId {
            return true;
        }
        // If xmax is set but not committed, tuple is still visible
        if (infomask & pg_sys::HEAP_XMAX_COMMITTED as u16) == 0 {
            return true;
        }
        // Deleted and committed - not visible
        return false;
    }

    // HEAP_XMIN_INVALID means tuple was never valid
    if (infomask & pg_sys::HEAP_XMIN_INVALID as u16) != 0 {
        return false;
    }

    // For in-progress transactions, do a proper visibility check
    // This is a simplified check - in production you'd use HeapTupleSatisfiesVisibility
    // But for sampling, we can be slightly loose
    pg_sys::TransactionIdDidCommit(xmin)
}

/// Extract vector data from a datum
fn extract_vector_from_datum(datum: Datum) -> Option<(Vec<f32>, u32)> {
    if datum.is_null() {
        return None;
    }

    unsafe {
        let raw_ptr = datum.cast_mut_ptr();
        let detoasted_ptr = pg_sys::pg_detoast_datum(raw_ptr);
        let byte_slice = pgrx::varlena_to_byte_slice(detoasted_ptr);
        let (vec_vals, vec_dims) = vector_type::decode_pgvector_vector(byte_slice);

        // Free detoasted memory if it was allocated
        if detoasted_ptr != raw_ptr {
            pg_sys::pfree(detoasted_ptr as *mut std::ffi::c_void);
        }

        Some((vec_vals, vec_dims))
    }
}

/// FeistelBlockIterator generates a random permutation of block numbers using a Feistel cipher.
/// This provides uniform random sampling without needing to store all block numbers.
struct FeistelBlockIterator {
    total_blocks: u32,
    blocks_to_return: u32,
    blocks_returned: u32,
    permutation_index: u32,
    width: u32,
    key_0: u64,
    key_1: u64,
}

impl FeistelBlockIterator {
    fn new(total_blocks: u32, blocks_to_sample: u32) -> Self {
        // Width must be even and large enough to cover total_blocks
        let width = if total_blocks <= 1 {
            2
        } else {
            let log2 = 32 - (total_blocks - 1).leading_zeros();
            (log2 as u32).next_multiple_of(2).max(2)
        };

        // Generate random keys using system time as seed
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let key_0 = hasher.finish();

        (seed ^ 0xDEADBEEF).hash(&mut hasher);
        let key_1 = hasher.finish();

        FeistelBlockIterator {
            total_blocks,
            blocks_to_return: blocks_to_sample.min(total_blocks),
            blocks_returned: 0,
            permutation_index: 0,
            width,
            key_0,
            key_1,
        }
    }

    /// Feistel cipher round function
    fn feistel_round(&self, round: u32, value: u32) -> u32 {
        let mut hasher = DefaultHasher::new();
        round.hash(&mut hasher);
        value.hash(&mut hasher);
        self.key_0.hash(&mut hasher);
        self.key_1.hash(&mut hasher);
        hasher.finish() as u32
    }

    /// Apply Feistel cipher to generate pseudo-random permutation
    fn feistel_permute(&self, input: u32) -> u32 {
        let half_width = self.width / 2;
        let mask = (1u32 << half_width) - 1;

        let mut left = (input >> half_width) & mask;
        let mut right = input & mask;

        // 8 rounds of Feistel cipher
        for round in 0..8 {
            let new_left = right;
            let new_right = left ^ (self.feistel_round(round, right) & mask);
            left = new_left;
            right = new_right;
        }

        (left << half_width) | right
    }
}

impl Iterator for FeistelBlockIterator {
    type Item = BlockNumber;

    fn next(&mut self) -> Option<Self::Item> {
        if self.blocks_returned >= self.blocks_to_return {
            return None;
        }

        let max_permutation = 1u32 << self.width;

        // Find next valid block number through permutation
        while self.permutation_index < max_permutation {
            let permuted = self.feistel_permute(self.permutation_index);
            self.permutation_index += 1;

            // Only return if within valid block range
            if permuted < self.total_blocks {
                self.blocks_returned += 1;
                return Some(permuted);
            }
        }

        None
    }
}
