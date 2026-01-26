use crate::vector_type;
use pgrx::pg_sys::{self, BlockNumber, Datum, OffsetNumber};
use pgrx::info;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ptr::NonNull;

// Configuration
const VECTORS_PER_BATCH: u64 = 50000;

// =============================================================================
// Reimplementation of PostgreSQL's inline table sampling functions
// These are static inline in tableam.h, not exposed by pgrx
// =============================================================================

/// Reimplementation of table_scan_sample_next_block from tableam.h
/// The tableam's scan_sample_next_block will call our TSM callback internally
#[inline]
unsafe fn table_scan_sample_next_block(
    scan: pg_sys::TableScanDesc,
    scanstate: *mut pg_sys::SampleScanState,
) -> bool {
    let scan_ref = &*scan;

    // Call the tableam's scan_sample_next_block directly
    // It will call our TSM NextSampleBlock callback internally
    let relation = scan_ref.rs_rd;
    let tableam = (*relation).rd_tableam;

    if let Some(scan_sample_next_block_fn) = (*tableam).scan_sample_next_block {
        return scan_sample_next_block_fn(scan, scanstate);
    }

    false
}

/// Reimplementation of table_scan_sample_next_tuple from tableam.h
/// The tableam's scan_sample_next_tuple will call our TSM callback internally
#[inline]
unsafe fn table_scan_sample_next_tuple(
    scan: pg_sys::TableScanDesc,
    scanstate: *mut pg_sys::SampleScanState,
    slot: *mut pg_sys::TupleTableSlot,
) -> bool {
    let scan_ref = &*scan;

    // Call the tableam's scan_sample_next_tuple directly
    // It will call our TSM NextSampleTuple callback internally
    let relation = scan_ref.rs_rd;
    let tableam = (*relation).rd_tableam;

    if let Some(scan_sample_next_tuple_fn) = (*tableam).scan_sample_next_tuple {
        return scan_sample_next_tuple_fn(scan, scanstate, slot);
    }

    false
}

/// Reimplementation of table_endscan from tableam.h
#[inline]
unsafe fn table_endscan(scan: pg_sys::TableScanDesc) {
    let scan_ref = &*scan;
    let relation = scan_ref.rs_rd;
    let tableam = (*relation).rd_tableam;

    if let Some(scan_end_fn) = (*tableam).scan_end {
        scan_end_fn(scan);
    }
}

/// Reimplementation of table_beginscan_sampling from tableam.h
/// This sets up a table scan for TABLESAMPLE operations
#[inline]
unsafe fn table_beginscan_sampling(
    rel: pg_sys::Relation,
    snapshot: pg_sys::Snapshot,
    nkeys: std::ffi::c_int,
    key: *mut pg_sys::ScanKeyData,
    allow_strat: bool,
    allow_sync: bool,
    allow_pagemode: bool,
) -> pg_sys::TableScanDesc {
    use pg_sys::ScanOptions;

    // Build flags for the scan
    let mut flags: u32 = ScanOptions::SO_TYPE_SAMPLESCAN as u32;

    if allow_strat {
        flags |= ScanOptions::SO_ALLOW_STRAT as u32;
    }
    if allow_sync {
        flags |= ScanOptions::SO_ALLOW_SYNC as u32;
    }
    if allow_pagemode {
        flags |= ScanOptions::SO_ALLOW_PAGEMODE as u32;
    }

    // Call the tableam's scan_begin function
    let tableam = (*rel).rd_tableam;

    if let Some(scan_begin_fn) = (*tableam).scan_begin {
        return scan_begin_fn(rel, snapshot, nkeys, key, std::ptr::null_mut(), flags);
    }

    std::ptr::null_mut()
}

/// Reimplementation of table_slot_create from tableam.h
/// Creates a TupleTableSlot suitable for the given relation
#[inline]
unsafe fn table_slot_create(
    rel: pg_sys::Relation,
    _reglist: *mut *mut pg_sys::List,
) -> *mut pg_sys::TupleTableSlot {
    let tableam = (*rel).rd_tableam;

    if let Some(slot_callbacks_fn) = (*tableam).slot_callbacks {
        let callbacks = slot_callbacks_fn(rel);
        return pg_sys::MakeSingleTupleTableSlot((*rel).rd_att, callbacks);
    }

    // Fallback - shouldn't happen with valid heap relations
    pg_sys::MakeSingleTupleTableSlot((*rel).rd_att, std::ptr::null())
}

/// VectorReadBatcher provides efficient random sampling of vectors from a PostgreSQL table.
/// Uses PostgreSQL's native table sampling infrastructure with Feistel-based block permutation,
/// matching VectorChord's implementation for optimal performance.
pub struct VectorReadBatcher {
    sampler: Option<HeapSampler>,
    sample: Option<HeapSample>,
    column_attnum: i16,
    target_samples: u64,
    vectors_read: u64,
    dims: u32,
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

        let (sampler, column_attnum, dims) = unsafe {
            // Parse and open relation
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

            // Get total number of blocks
            let total_blocks = pg_sys::RelationGetNumberOfBlocksInFork(heap_relation, 0);

            // Calculate density for logging
            let vec_byte_size = detected_dims as u64 * 4;
            let is_toasted = vec_byte_size > 2000 || vec_byte_size == 0;
            let density = if is_toasted { 177 } else { (8192 - 24) / (24 + vec_byte_size + 4) };
            let estimated_rows = total_blocks as u64 * density;

            let num_chunks = (total_blocks + CHUNK_SIZE_BLOCKS - 1) / CHUNK_SIZE_BLOCKS;
            info!(
                "[SAMPLER] Table: {} blocks (~{} rows). Target: {} samples. Dims: {}",
                total_blocks, estimated_rows, target_samples, detected_dims
            );
            info!(
                "[SAMPLER] Using chunked I/O: {} chunks of {} blocks ({}MB sequential reads)",
                num_chunks, CHUNK_SIZE_BLOCKS, (CHUNK_SIZE_BLOCKS * 8) / 1024
            );

            // Get snapshot
            let snapshot = pg_sys::GetActiveSnapshot();

            // Create sampler
            let sampler = HeapSampler::new(heap_relation, snapshot, total_blocks);

            (Some(sampler), attnum, detected_dims)
        };

        let mut batcher = VectorReadBatcher {
            sampler,
            sample: None,
            column_attnum,
            target_samples,
            vectors_read: 0,
            dims,
            active: true,
        };

        // Initialize the sample
        if let Some(ref sampler) = batcher.sampler {
            batcher.sample = Some(sampler.sample());
        }

        batcher
    }

    pub(crate) fn next_batch(&mut self) -> Option<(Vec<f32>, u32)> {
        if !self.active || self.vectors_read >= self.target_samples {
            return None;
        }

        let sample = self.sample.as_mut()?;
        let batch_target = VECTORS_PER_BATCH.min(self.target_samples - self.vectors_read);

        let mut vectors: Vec<f32> = Vec::with_capacity(batch_target as usize * self.dims.max(768) as usize);
        let mut dims: u32 = self.dims;
        let mut count: u64 = 0;

        while count < batch_target {
            match sample.next(self.column_attnum) {
                Some(datum) => {
                    if let Some(vec_dims) = extract_vector_to_buffer(datum, &mut vectors) {
                        if dims == 0 {
                            dims = vec_dims;
                        }
                        count += 1;
                    }
                }
                None => {
                    if count == 0 {
                        self.active = false;
                        return None;
                    }
                    break;
                }
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
        self.sample = None;
        self.sampler = None;
    }
}

impl Drop for VectorReadBatcher {
    fn drop(&mut self) {
        self.sample = None;
        self.sampler = None;
    }
}

/// Resolve a qualified table name to an OID
fn resolve_table_oid(qualified_table_name: &str) -> pg_sys::Oid {
    let query = format!("SELECT '{}'::regclass::oid", qualified_table_name);

    let oid: Option<pg_sys::Oid> = pgrx::Spi::connect(|client| {
        let result = client.select(&query, None, &[]);
        match result {
            Ok(table) => table.first().get_one::<pg_sys::Oid>().ok().flatten(),
            Err(_) => None,
        }
    });

    match oid {
        Some(o) if o != pg_sys::InvalidOid => o,
        _ => pgrx::error!("Table '{}' not found", qualified_table_name),
    }
}

/// Extract vector data from a datum directly into a destination buffer.
/// Returns the dimension of the vector, or None if datum is null.
fn extract_vector_to_buffer(datum: Datum, dest: &mut Vec<f32>) -> Option<u32> {
    if datum.is_null() {
        return None;
    }

    unsafe {
        let raw_ptr = datum.cast_mut_ptr();
        let detoasted_ptr = pg_sys::pg_detoast_datum(raw_ptr);
        let byte_slice = pgrx::varlena_to_byte_slice(detoasted_ptr);
        let dims = vector_type::decode_pgvector_vector_into(byte_slice, dest);

        if detoasted_ptr != raw_ptr {
            pg_sys::pfree(detoasted_ptr as *mut std::ffi::c_void);
        }

        Some(dims)
    }
}

// =============================================================================
// PostgreSQL Table Sampling Infrastructure (matches VectorChord's approach)
// =============================================================================

/// HeapSampler manages the sampling scan setup
struct HeapSampler {
    heap_relation: pg_sys::Relation,
    snapshot: pg_sys::Snapshot,
    total_blocks: BlockNumber,
}

impl HeapSampler {
    unsafe fn new(
        heap_relation: pg_sys::Relation,
        snapshot: pg_sys::Snapshot,
        total_blocks: BlockNumber,
    ) -> Self {
        Self {
            heap_relation,
            snapshot,
            total_blocks,
        }
    }

    fn sample(&self) -> HeapSample {
        unsafe {
            // Create sampler state with chunked block iteration for sequential I/O
            let state = NonNull::new_unchecked(Box::into_raw(Box::new(SamplerState {
                blocks_iter: Some(Box::new(ChunkedBlockIterator::new(
                    self.total_blocks,
                    CHUNK_SIZE_BLOCKS,
                ))),
                tuples_iter: None,
            })));

            // Create sample scan state with our custom TSM routine
            let sample_scan_state = NonNull::new_unchecked(Box::into_raw(Box::new(
                pg_sys::SampleScanState {
                    tsmroutine: (&raw const TSM_ROUTINE.0).cast_mut().cast(),
                    tsm_state: state.as_ptr().cast(),
                    ..core::mem::zeroed()
                },
            )));

            // Begin sampling scan using our reimplemented function
            let table_scan_desc = table_beginscan_sampling(
                self.heap_relation,
                self.snapshot,
                0,
                std::ptr::null_mut(),
                true,  // allow_strat
                false, // allow_sync
                true,  // allow_pagemode
            );

            // Create executor state
            let estate = pg_sys::CreateExecutorState();
            let econtext = pg_sys::MakePerTupleExprContext(estate);

            // Create tuple slot using our reimplemented function
            let slot = table_slot_create(self.heap_relation, std::ptr::null_mut());

            HeapSample {
                heap_relation: self.heap_relation,
                estate,
                econtext,
                slot,
                state,
                sample_scan_state,
                table_scan_desc,
                done: false,
                have_block: false,
            }
        }
    }
}

impl Drop for HeapSampler {
    fn drop(&mut self) {
        unsafe {
            pg_sys::relation_close(self.heap_relation, pg_sys::AccessShareLock as i32);
        }
    }
}

/// HeapSample handles the actual tuple iteration
struct HeapSample {
    #[allow(dead_code)]
    heap_relation: pg_sys::Relation,
    estate: *mut pg_sys::EState,
    econtext: *mut pg_sys::ExprContext,
    slot: *mut pg_sys::TupleTableSlot,
    state: NonNull<SamplerState>,
    sample_scan_state: NonNull<pg_sys::SampleScanState>,
    table_scan_desc: pg_sys::TableScanDesc,
    done: bool,
    have_block: bool,
}

impl HeapSample {
    fn next(&mut self, attnum: i16) -> Option<Datum> {
        if self.done {
            return None;
        }

        unsafe {
            loop {
                if !self.have_block {
                    // Get next sample block using our reimplemented function
                    if !table_scan_sample_next_block(
                        self.table_scan_desc,
                        self.sample_scan_state.as_ptr(),
                    ) {
                        self.have_block = false;
                        self.done = true;
                        return None;
                    }
                    self.have_block = true;
                }

                // Get next tuple using our reimplemented function
                if !table_scan_sample_next_tuple(
                    self.table_scan_desc,
                    self.sample_scan_state.as_ptr(),
                    self.slot,
                ) {
                    self.have_block = false;
                    continue;
                }

                // Reset per-tuple memory context (important for performance!)
                pg_sys::MemoryContextReset((*self.econtext).ecxt_per_tuple_memory);

                // Extract attribute from slot
                let mut is_null: bool = true;
                let datum = pg_sys::slot_getattr(self.slot, attnum as i32, &mut is_null);

                if is_null {
                    continue;
                }

                return Some(datum);
            }
        }
    }
}

impl Drop for HeapSample {
    fn drop(&mut self) {
        unsafe {
            // Reset memory context
            if !self.econtext.is_null() {
                pg_sys::MemoryContextReset((*self.econtext).ecxt_per_tuple_memory);
            }

            // End the table scan using our reimplemented function
            table_endscan(self.table_scan_desc);

            // Free our state boxes
            let _ = Box::from_raw(self.sample_scan_state.as_ptr());
            let _ = Box::from_raw(self.state.as_ptr());

            // Free tuple slot
            if !self.slot.is_null() {
                pg_sys::ExecDropSingleTupleTableSlot(self.slot);
            }

            // Free executor state
            if !self.estate.is_null() {
                pg_sys::FreeExecutorState(self.estate);
            }
        }
    }
}

// =============================================================================
// Sampler State and Block Iterators
// =============================================================================

// Number of consecutive blocks to read per random seek
// 25600 blocks = 200MB sequential read per seek (optimized for NVMe SSDs)
const CHUNK_SIZE_BLOCKS: u32 = 25600;

struct SamplerState {
    blocks_iter: Option<Box<ChunkedBlockIterator>>,
    tuples_iter: Option<std::ops::RangeInclusive<u16>>,
}

/// Feistel cipher-based block permutation for random sampling
struct FeistelBlockIterator {
    total_blocks: u32,
    permutation_index: u32,
    width: u32,
    key_0: u64,
    key_1: u64,
}

impl FeistelBlockIterator {
    fn new(total_blocks: u32) -> Self {
        let width = if total_blocks <= 1 {
            2
        } else {
            let log2 = 32 - (total_blocks - 1).leading_zeros();
            (log2 as u32).next_multiple_of(2).max(2)
        };

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
            permutation_index: 0,
            width,
            key_0,
            key_1,
        }
    }

    fn feistel_round(&self, round: u32, value: u32) -> u32 {
        let mut hasher = DefaultHasher::new();
        round.hash(&mut hasher);
        value.hash(&mut hasher);
        self.key_0.hash(&mut hasher);
        self.key_1.hash(&mut hasher);
        hasher.finish() as u32
    }

    fn feistel_permute(&self, input: u32) -> u32 {
        let half_width = self.width / 2;
        let mask = (1u32 << half_width) - 1;

        let mut left = (input >> half_width) & mask;
        let mut right = input & mask;

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
        let max_permutation = 1u32 << self.width;

        while self.permutation_index < max_permutation {
            let permuted = self.feistel_permute(self.permutation_index);
            self.permutation_index += 1;

            if permuted < self.total_blocks {
                return Some(permuted);
            }
        }

        None
    }
}

/// Chunked block iterator for sequential I/O optimization.
/// Divides the table into chunks of consecutive blocks, then visits chunks
/// in random order (using Feistel permutation). Within each chunk, blocks
/// are read sequentially, maximizing I/O throughput on SSDs.
struct ChunkedBlockIterator {
    total_blocks: u32,
    chunk_size: u32,
    /// Feistel iterator over chunk indices (not block indices)
    chunk_feistel: FeistelBlockIterator,
    /// Current chunk's starting block
    current_chunk_start: Option<u32>,
    /// Current offset within the current chunk
    current_offset: u32,
}

impl ChunkedBlockIterator {
    fn new(total_blocks: u32, chunk_size: u32) -> Self {
        // Number of chunks (rounded up)
        let num_chunks = (total_blocks + chunk_size - 1) / chunk_size;

        // Use Feistel to randomize chunk order
        let chunk_feistel = FeistelBlockIterator::new(num_chunks);

        ChunkedBlockIterator {
            total_blocks,
            chunk_size,
            chunk_feistel,
            current_chunk_start: None,
            current_offset: 0,
        }
    }
}

impl Iterator for ChunkedBlockIterator {
    type Item = BlockNumber;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we have a current chunk, try to emit the next block from it
            if let Some(chunk_start) = self.current_chunk_start {
                let block = chunk_start + self.current_offset;

                // Check if we're still within the chunk AND within total blocks
                if self.current_offset < self.chunk_size && block < self.total_blocks {
                    self.current_offset += 1;
                    return Some(block);
                }

                // Chunk exhausted, need to get next chunk
                self.current_chunk_start = None;
            }

            // Get next chunk from Feistel
            match self.chunk_feistel.next() {
                Some(chunk_idx) => {
                    let chunk_start = chunk_idx * self.chunk_size;
                    self.current_chunk_start = Some(chunk_start);
                    self.current_offset = 0;
                    // Continue loop to emit first block of new chunk
                }
                None => return None,
            }
        }
    }
}

// =============================================================================
// TSM (Table Sampling Method) Callbacks
// =============================================================================

#[pgrx::pg_guard]
unsafe extern "C-unwind" fn tsm_next_sample_block(
    node: *mut pg_sys::SampleScanState,
    _nblocks: BlockNumber,
) -> BlockNumber {
    let state: &mut SamplerState = &mut *((*node).tsm_state as *mut SamplerState);

    if let Some(ref mut iter) = state.blocks_iter {
        if let Some(block) = iter.next() {
            return block;
        }
    }

    pg_sys::InvalidBlockNumber
}

#[pgrx::pg_guard]
unsafe extern "C-unwind" fn tsm_next_sample_tuple(
    node: *mut pg_sys::SampleScanState,
    _blockno: BlockNumber,
    maxoffset: OffsetNumber,
) -> OffsetNumber {
    let state: &mut SamplerState = &mut *((*node).tsm_state as *mut SamplerState);

    let iter = state.tuples_iter.get_or_insert(1..=maxoffset);

    if let Some(offset) = iter.next() {
        offset
    } else {
        state.tuples_iter = None;
        pg_sys::InvalidOffsetNumber
    }
}

// TSM Routine structure
struct SyncTsmRoutine(TsmRoutine);
unsafe impl Sync for SyncTsmRoutine {}

// Field names match PostgreSQL's C struct TsmRoutine
#[repr(C)]
#[allow(non_snake_case)]
struct TsmRoutine {
    type_: pg_sys::NodeTag,
    parameterTypes: *mut pg_sys::List,
    repeatable_across_queries: bool,
    repeatable_across_scans: bool,
    SampleScanGetSampleSize: Option<unsafe extern "C-unwind" fn(*mut pg_sys::PlannerInfo, *mut pg_sys::RelOptInfo, *mut pg_sys::List, *mut BlockNumber, *mut f64)>,
    InitSampleScan: Option<unsafe extern "C-unwind" fn(*mut pg_sys::SampleScanState, std::ffi::c_int)>,
    BeginSampleScan: Option<unsafe extern "C-unwind" fn(*mut pg_sys::SampleScanState, *mut Datum, std::ffi::c_int, u32)>,
    NextSampleBlock: Option<unsafe extern "C-unwind" fn(*mut pg_sys::SampleScanState, BlockNumber) -> BlockNumber>,
    NextSampleTuple: Option<unsafe extern "C-unwind" fn(*mut pg_sys::SampleScanState, BlockNumber, OffsetNumber) -> OffsetNumber>,
    EndSampleScan: Option<unsafe extern "C-unwind" fn(*mut pg_sys::SampleScanState)>,
}

static TSM_ROUTINE: SyncTsmRoutine = SyncTsmRoutine(TsmRoutine {
    type_: pg_sys::NodeTag::T_TsmRoutine,
    parameterTypes: std::ptr::null_mut(),
    repeatable_across_queries: false,
    repeatable_across_scans: false,
    SampleScanGetSampleSize: None,
    InitSampleScan: None,
    BeginSampleScan: None,
    NextSampleBlock: Some(tsm_next_sample_block),
    NextSampleTuple: Some(tsm_next_sample_tuple),
    EndSampleScan: None,
});
