use crate::vector_type;
use pgrx::pg_sys::{self, BlockNumber, Datum, Buffer};
use pgrx::info;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// PostgreSQL constants not exposed by pgrx
const MAIN_FORKNUM: i32 = 0; // pg_sys::ForkNumber::MAIN_FORKNUM
const INVALID_BUFFER: Buffer = 0;

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
fn resolve_table_oid(qualified_table_name: &str) -> pg_sys::Oid {
    // Use regclass cast via SPI for reliable name resolution
    let query = format!("SELECT '{}'::regclass::oid", qualified_table_name);

    let oid: Option<pg_sys::Oid> = pgrx::Spi::connect(|client| {
        let result = client.select(&query, None, &[]);
        match result {
            Ok(table) => {
                table.first().get_one::<pg_sys::Oid>().ok().flatten()
            }
            Err(_) => None,
        }
    });

    match oid {
        Some(o) if o != pg_sys::InvalidOid => o,
        _ => pgrx::error!("Table '{}' not found", qualified_table_name),
    }
}

/// BlockReader handles reading tuples from a single heap block
struct BlockReader {
    relation: pg_sys::Relation,
    buffer: Buffer,
    page: pg_sys::Page,
    current_offset: u16,
    max_offset: u16,
}

impl BlockReader {
    fn new(relation: pg_sys::Relation, block_num: BlockNumber) -> Self {
        unsafe {
            let buffer = pg_sys::ReadBuffer(relation, block_num);
            pg_sys::LockBuffer(buffer, pg_sys::BUFFER_LOCK_SHARE as i32);

            let page = pg_sys::BufferGetPage(buffer);
            let max_offset = page_get_max_offset_number(page);

            BlockReader {
                relation,
                buffer,
                page,
                current_offset: 1, // Offsets start at 1 in PostgreSQL
                max_offset,
            }
        }
    }

    fn next_tuple_datum(&mut self, attnum: i16) -> Option<Datum> {
        unsafe {
            while self.current_offset <= self.max_offset {
                let offset = self.current_offset;
                self.current_offset += 1;

                // Get item pointer for this offset
                let item_id = page_get_item_id(self.page, offset);

                // Skip dead/unused items
                if !item_id_is_normal(item_id) {
                    continue;
                }

                // Get the heap tuple header
                let item = page_get_item(self.page, item_id);
                let htup = item as *mut pg_sys::HeapTupleHeaderData;

                // Check tuple visibility - simplified check for sampling
                if !is_tuple_visible(htup) {
                    continue;
                }

                // Extract the attribute value
                let tuple_desc = (*self.relation).rd_att;

                // Build a minimal HeapTupleData for attribute extraction
                let mut tuple_data = pg_sys::HeapTupleData {
                    t_len: item_id_get_length(item_id),
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
            if self.buffer != INVALID_BUFFER {
                pg_sys::UnlockReleaseBuffer(self.buffer);
            }
        }
    }
}

// =============================================================================
// PostgreSQL Page Access Macros (reimplemented in Rust)
// These are macros in PostgreSQL C code, not exposed as functions in pgrx
// =============================================================================

/// PageGetMaxOffsetNumber - get the last valid offset on a page
/// Equivalent to: ((PageHeader)(page))->pd_lower <= SizeOfPageHeaderData ? 0 :
///                (((PageHeader)(page))->pd_lower - SizeOfPageHeaderData) / sizeof(ItemIdData)
#[inline]
unsafe fn page_get_max_offset_number(page: pg_sys::Page) -> u16 {
    let header = page as *const pg_sys::PageHeaderData;
    let pd_lower = (*header).pd_lower as usize;

    // SizeOfPageHeaderData = 24 bytes
    const SIZE_OF_PAGE_HEADER_DATA: usize = 24;
    // sizeof(ItemIdData) = 4 bytes
    const SIZE_OF_ITEM_ID_DATA: usize = 4;

    if pd_lower <= SIZE_OF_PAGE_HEADER_DATA {
        0
    } else {
        ((pd_lower - SIZE_OF_PAGE_HEADER_DATA) / SIZE_OF_ITEM_ID_DATA) as u16
    }
}

/// PageGetItemId - get pointer to an ItemId on a page
/// Equivalent to: &((PageHeader)(page))->pd_linp[(offsetNumber) - 1]
#[inline]
unsafe fn page_get_item_id(page: pg_sys::Page, offset_number: u16) -> *const pg_sys::ItemIdData {
    let header = page as *const pg_sys::PageHeaderData;
    let linp = std::ptr::addr_of!((*header).pd_linp) as *const pg_sys::ItemIdData;
    linp.add((offset_number - 1) as usize)
}

/// PageGetItem - get pointer to the actual item on a page
/// Equivalent to: (Item)(((char *)(page)) + ItemIdGetOffset(itemId))
#[inline]
unsafe fn page_get_item(page: pg_sys::Page, item_id: *const pg_sys::ItemIdData) -> pg_sys::Item {
    let offset = item_id_get_offset(item_id);
    (page as *mut u8).add(offset as usize) as pg_sys::Item
}

/// ItemIdGetOffset - extract offset from ItemIdData
/// The offset is stored in bits 0-14 of lp_off_flags
#[inline]
unsafe fn item_id_get_offset(item_id: *const pg_sys::ItemIdData) -> u16 {
    // In PostgreSQL, ItemIdData is a 32-bit value with:
    // - lp_off: 15 bits (offset)
    // - lp_flags: 2 bits
    // - lp_len: 15 bits (length)
    // The structure uses bitfields, but we can access via the raw u32
    let raw = std::ptr::read_unaligned(item_id as *const u32);
    (raw & 0x7FFF) as u16 // Lower 15 bits are offset
}

/// ItemIdGetLength - extract length from ItemIdData
#[inline]
unsafe fn item_id_get_length(item_id: *const pg_sys::ItemIdData) -> u32 {
    let raw = std::ptr::read_unaligned(item_id as *const u32);
    (raw >> 17) & 0x7FFF // Upper 15 bits (after 2 flag bits) are length
}

/// ItemIdGetFlags - extract flags from ItemIdData
#[inline]
unsafe fn item_id_get_flags(item_id: *const pg_sys::ItemIdData) -> u8 {
    let raw = std::ptr::read_unaligned(item_id as *const u32);
    ((raw >> 15) & 0x3) as u8 // Bits 15-16 are flags
}

/// ItemIdIsNormal - check if item is normal (in use, not dead/redirect)
/// LP_NORMAL = 1
#[inline]
unsafe fn item_id_is_normal(item_id: *const pg_sys::ItemIdData) -> bool {
    item_id_get_flags(item_id) == 1 // LP_NORMAL
}

/// Simple visibility check for heap tuples
/// For sampling purposes, we accept tuples that appear committed
unsafe fn is_tuple_visible(htup: *mut pg_sys::HeapTupleHeaderData) -> bool {
    // Check infomask for committed status
    let infomask = (*htup).t_infomask;

    // HEAP_XMIN_COMMITTED (0x0100) means the inserting transaction committed
    const HEAP_XMIN_COMMITTED: u16 = 0x0100;
    // HEAP_XMIN_INVALID (0x0200) means the tuple was never valid
    const HEAP_XMIN_INVALID: u16 = 0x0200;
    // HEAP_XMAX_COMMITTED (0x0400) means the deleting transaction committed
    const HEAP_XMAX_COMMITTED: u16 = 0x0400;
    // HEAP_XMAX_INVALID (0x0800) means no delete in progress
    const HEAP_XMAX_INVALID: u16 = 0x0800;

    // If xmin is marked as committed
    if (infomask & HEAP_XMIN_COMMITTED) != 0 {
        // Check if it's been deleted
        if (infomask & HEAP_XMAX_INVALID) != 0 {
            // Not deleted - visible
            return true;
        }
        if (infomask & HEAP_XMAX_COMMITTED) == 0 {
            // Delete not committed yet - still visible
            return true;
        }
        // Deleted and committed - not visible
        return false;
    }

    // If xmin is marked as invalid, tuple was never valid
    if (infomask & HEAP_XMIN_INVALID) != 0 {
        return false;
    }

    // For tuples without hint bits set, we need to check transaction status
    // For sampling, we'll optimistically include them (they're likely committed)
    // This is acceptable since sampling doesn't need perfect accuracy
    let xmin = (*htup).t_choice.t_heap.t_xmin;

    // Check if it's a frozen transaction (always visible)
    // FrozenTransactionId = 2
    if xmin == pg_sys::FrozenTransactionId {
        return true;
    }

    // For other transactions, check if committed
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
