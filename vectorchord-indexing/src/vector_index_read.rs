use crate::vector_type;
use pgrx::pg_sys::{self, BlockNumber, Datum, OffsetNumber};
use pgrx::{info, PgRelation};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ptr::NonNull;

/// VectorReadBatcher provides efficient random sampling of vectors from a PostgreSQL table
/// using direct heap access with Feistel-cipher-based block permutation.
///
/// This implementation bypasses SPI calls entirely for maximum performance,
/// replicating the approach used by VectorChord 1.0.
pub struct VectorReadBatcher {
    sampler: Option<HeapSampler>,
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

        // Get relation and column info using unsafe PostgreSQL internals
        let (sampler, column_attnum, dims) = unsafe {
            // Parse table name and open relation
            let table_name_cstr = std::ffi::CString::new(qualified_table_name.as_str()).unwrap();
            let relname_list = pg_sys::stringToQualifiedNameList(table_name_cstr.as_ptr());
            let range_var = pg_sys::makeRangeVarFromNameList(relname_list);
            let rel_oid = pg_sys::RangeVarGetRelid(range_var, pg_sys::AccessShareLock as i32, false);

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
            let total_blocks = pg_sys::RelationGetNumberOfBlocks(heap_relation);

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
                (blocks_to_scan as f64 / total_blocks as f64) * 100.0
            );

            // Get snapshot
            let snapshot = pg_sys::GetActiveSnapshot();

            let sampler = HeapSampler::new(heap_relation, snapshot, total_blocks, blocks_to_scan);

            (Some(sampler), attnum, detected_dims)
        };

        VectorReadBatcher {
            sampler,
            column_attnum,
            target_samples,
            vectors_read: 0,
            dims,
            active: true,
        }
    }

    pub(crate) fn next_batch(&mut self) -> Option<(Vec<f32>, u32)> {
        if !self.active || self.vectors_read >= self.target_samples {
            return None;
        }

        let sampler = self.sampler.as_mut()?;

        // Read vectors in batches of ~10000 for memory efficiency
        let batch_target = 10000u64.min(self.target_samples - self.vectors_read);

        let mut vectors: Vec<f32> = Vec::with_capacity(batch_target as usize * self.dims.max(768) as usize);
        let mut dims: u32 = self.dims;
        let mut count: u64 = 0;

        while count < batch_target {
            match sampler.next_tuple() {
                Some(tuple_data) => {
                    // Extract the vector column datum
                    if let Some((vec_vals, vec_dims)) = self.extract_vector(&tuple_data) {
                        if dims == 0 {
                            dims = vec_dims;
                        }
                        vectors.extend_from_slice(&vec_vals);
                        count += 1;
                    }
                }
                None => {
                    // No more tuples available
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

    fn extract_vector(&self, tuple_data: &TupleData) -> Option<(Vec<f32>, u32)> {
        let attnum_idx = (self.column_attnum - 1) as usize;

        if attnum_idx >= 32 || tuple_data.is_nulls[attnum_idx] {
            return None;
        }

        let datum = tuple_data.values[attnum_idx];
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

    pub(crate) fn end_scan(&mut self) {
        self.active = false;
        self.sampler = None;
    }
}

impl Drop for VectorReadBatcher {
    fn drop(&mut self) {
        self.sampler = None;
    }
}

/// HeapSampler implements direct heap access with Feistel-based random block selection.
/// This replicates the VectorChord 1.0 sampling approach.
struct HeapSampler {
    heap_relation: pg_sys::Relation,
    snapshot: pg_sys::Snapshot,
    state: NonNull<SamplerState>,
    sample_scan_state: NonNull<pg_sys::SampleScanState>,
    table_scan_desc: pg_sys::TableScanDesc,
    slot: *mut pg_sys::TupleTableSlot,
    estate: *mut pg_sys::EState,
    econtext: *mut pg_sys::ExprContext,
    values: [Datum; 32],
    is_nulls: [bool; 32],
    done: bool,
    have_block: bool,
}

impl HeapSampler {
    unsafe fn new(
        heap_relation: pg_sys::Relation,
        snapshot: pg_sys::Snapshot,
        total_blocks: BlockNumber,
        blocks_to_sample: u32,
    ) -> Self {
        // Create sampler state with Feistel permutation
        let state = NonNull::new_unchecked(Box::into_raw(Box::new(SamplerState {
            blocks_iter: Some(Box::new(FeistelBlockIterator::new(
                total_blocks,
                blocks_to_sample,
            ))),
            tuples_iter: None,
        })));

        // Create sample scan state with our custom TSM routine
        let sample_scan_state =
            NonNull::new_unchecked(Box::into_raw(Box::new(pg_sys::SampleScanState {
                tsmroutine: &raw const TSM_ROUTINE as *const _ as *mut pg_sys::TsmRoutine,
                tsm_state: state.as_ptr() as *mut std::ffi::c_void,
                ..core::mem::zeroed()
            })));

        // Begin sampling scan
        let table_scan_desc = pg_sys::table_beginscan_sampling(
            heap_relation,
            snapshot,
            0,
            std::ptr::null_mut(),
            true,  // allow_strat
            false, // allow_sync
            true,  // allow_pagemode
        );

        // Create executor state for tuple processing
        let estate = pg_sys::CreateExecutorState();
        let econtext = pg_sys::MakePerTupleExprContext(estate);

        // Create tuple slot for storing scanned tuples
        let slot = pg_sys::table_slot_create(heap_relation, std::ptr::null_mut());

        HeapSampler {
            heap_relation,
            snapshot,
            state,
            sample_scan_state,
            table_scan_desc,
            slot,
            estate,
            econtext,
            values: [Datum::null(); 32],
            is_nulls: [true; 32],
            done: false,
            have_block: false,
        }
    }

    fn next_tuple(&mut self) -> Option<TupleData> {
        if self.done {
            return None;
        }

        unsafe {
            loop {
                if !self.have_block {
                    // Get next sample block
                    if !pg_sys::table_scan_sample_next_block(
                        self.table_scan_desc,
                        self.sample_scan_state.as_ptr(),
                    ) {
                        self.have_block = false;
                        self.done = true;
                        return None;
                    }
                    self.have_block = true;
                }

                // Get next tuple from current block
                if !pg_sys::table_scan_sample_next_tuple(
                    self.table_scan_desc,
                    self.sample_scan_state.as_ptr(),
                    self.slot,
                ) {
                    self.have_block = false;
                    continue;
                }

                // Extract values from tuple slot
                self.extract_slot_values();

                return Some(TupleData {
                    values: self.values,
                    is_nulls: self.is_nulls,
                });
            }
        }
    }

    unsafe fn extract_slot_values(&mut self) {
        // Reset per-tuple memory context
        pg_sys::MemoryContextReset((*self.econtext).ecxt_per_tuple_memory);

        // Use slot_getallattrs to get all attributes
        pg_sys::slot_getallattrs(self.slot);

        let slot = &*self.slot;
        let natts = (*slot.tts_tupleDescriptor).natts as usize;
        let natts = natts.min(32);

        // Copy values and nulls from the slot
        for i in 0..natts {
            self.values[i] = *slot.tts_values.add(i);
            self.is_nulls[i] = *slot.tts_isnull.add(i);
        }

        // Clear remaining slots
        for i in natts..32 {
            self.values[i] = Datum::null();
            self.is_nulls[i] = true;
        }
    }
}

impl Drop for HeapSampler {
    fn drop(&mut self) {
        unsafe {
            // End the table scan
            pg_sys::table_endscan(self.table_scan_desc);

            // Free executor state
            if !self.estate.is_null() {
                pg_sys::FreeExecutorState(self.estate);
            }

            // Free tuple slot
            if !self.slot.is_null() {
                pg_sys::ExecDropSingleTupleTableSlot(self.slot);
            }

            // Free our state boxes
            let _ = Box::from_raw(self.sample_scan_state.as_ptr());
            let _ = Box::from_raw(self.state.as_ptr());

            // Close the relation
            pg_sys::relation_close(self.heap_relation, pg_sys::AccessShareLock as i32);
        }
    }
}

/// TupleData holds the extracted values from a heap tuple
struct TupleData {
    values: [Datum; 32],
    is_nulls: [bool; 32],
}

/// SamplerState maintains the iteration state for block and tuple sampling
struct SamplerState {
    blocks_iter: Option<Box<FeistelBlockIterator>>,
    tuples_iter: Option<std::ops::RangeInclusive<u16>>,
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
        let width = if total_blocks == 0 {
            2
        } else {
            ((total_blocks.ilog2() + 1) as u32).next_multiple_of(2).max(2)
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

// =============================================================================
// PostgreSQL Table Sampling Method (TSM) Callbacks
// =============================================================================

/// TSM callback: returns the next block to sample
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

/// TSM callback: returns the next tuple offset within the current block
#[pgrx::pg_guard]
unsafe extern "C-unwind" fn tsm_next_sample_tuple(
    node: *mut pg_sys::SampleScanState,
    _blockno: BlockNumber,
    maxoffset: OffsetNumber,
) -> OffsetNumber {
    let state: &mut SamplerState = &mut *((*node).tsm_state as *mut SamplerState);

    // Initialize tuple iterator if needed
    let iter = state.tuples_iter.get_or_insert(1..=maxoffset);

    if let Some(offset) = iter.next() {
        offset
    } else {
        state.tuples_iter = None;
        pg_sys::InvalidOffsetNumber
    }
}

// =============================================================================
// TSM Routine Definition
// =============================================================================

/// Wrapper to make TsmRoutine Sync (it's a static read-only structure)
struct SyncTsmRoutine(TsmRoutine);
unsafe impl Sync for SyncTsmRoutine {}

/// Custom TsmRoutine structure compatible with PostgreSQL's expected layout
#[repr(C)]
struct TsmRoutine {
    type_: pg_sys::NodeTag,
    parameterTypes: *mut pg_sys::List,
    repeatable_across_queries: bool,
    repeatable_across_scans: bool,
    SampleScanGetSampleSize: Option<
        unsafe extern "C-unwind" fn(
            root: *mut pg_sys::PlannerInfo,
            baserel: *mut pg_sys::RelOptInfo,
            paramexprs: *mut pg_sys::List,
            pages: *mut BlockNumber,
            tuples: *mut f64,
        ),
    >,
    InitSampleScan: Option<
        unsafe extern "C-unwind" fn(node: *mut pg_sys::SampleScanState, eflags: std::ffi::c_int),
    >,
    BeginSampleScan: Option<
        unsafe extern "C-unwind" fn(
            node: *mut pg_sys::SampleScanState,
            params: *mut Datum,
            nparams: std::ffi::c_int,
            seed: u32,
        ),
    >,
    NextSampleBlock: Option<
        unsafe extern "C-unwind" fn(
            node: *mut pg_sys::SampleScanState,
            nblocks: BlockNumber,
        ) -> BlockNumber,
    >,
    NextSampleTuple: Option<
        unsafe extern "C-unwind" fn(
            node: *mut pg_sys::SampleScanState,
            blockno: BlockNumber,
            maxoffset: OffsetNumber,
        ) -> OffsetNumber,
    >,
    EndSampleScan: Option<unsafe extern "C-unwind" fn(node: *mut pg_sys::SampleScanState)>,
}

/// Static TSM routine with our custom callbacks
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
