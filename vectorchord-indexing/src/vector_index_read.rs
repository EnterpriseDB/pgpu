use crate::vector_type;
use pgrx::pg_sys::{format_type_be, SysScanDesc};
use pgrx::{debug1, heap_getattr_raw, info, pg_sys, warning, PgRelation, Spi};
use std::ffi::CStr;
use std::time::Instant;
use std::num::NonZero; // Explicit import for cleaner code


pub struct VectorReadBatcher {
    table_name: String,
    column_name: String,
    num_tuples_in_table: Option<u64>,
    num_samples: u64,
    num_samples_per_batch: u64,
    min_samples_per_batch: u64,
    vectors_read: u64,
    table_scan: Option<SysScanDesc>,
    pg_rel: Option<PgRelation>,
    col_num: Option<usize>,
    random_sampling: bool,
}

impl VectorReadBatcher {
    pub fn new(
        table_name: String,
        column_name: String,
        num_samples: u64,
        num_samples_per_batch: u64,
        min_samples_per_batch: u64,
    ) -> Self {
        // --- 🧪 TEST CONFIGURATION ---
        // Change this to 'false' to revert to sequential scanning
        let use_random_sampling = true;
        // -----------------------------

        let mut vbr = VectorReadBatcher {
            table_name,
            column_name,
            num_samples,
            num_samples_per_batch,
            min_samples_per_batch,
            num_tuples_in_table: None,
            vectors_read: 0,
            table_scan: None,
            pg_rel: None,
            col_num: None,
            random_sampling: use_random_sampling,
        };

        vbr.initialize();

        // Optional: Log the random plan if enabled
        if vbr.random_sampling {
            vbr.random_batch_sampling();
        }


        let table_size = (vbr).num_tuples();
        assert!(num_samples <= table_size as u64, "The table has fewer records ({table_size}) than the desired number of samples ({num_samples}) based on cluster_count*sampling_factor. Unable to continue");
        let rem = num_samples % num_samples_per_batch;
        if rem != 0 && rem < min_samples_per_batch {
            warning!("batch size {num_samples_per_batch} will lead to a remainder of {rem} samples in the last batch; which is too small for clustering. The last batch will be enlarged to {0} to contain this remainder", vbr.num_samples_per_batch + rem)
        }
        // TODO: calculate this from a new input "max memory GB"
        //info!("vector batch read properties:\n\t num_samples: {num_samples}\n\t num_samples_per_batch: {num_samples_per_batch}\n\t num_batches: {nb}\n\t table_size: {table_size}", nb=vbr.num_batches(), num_samples=vbr.num_samples, num_samples_per_batch=vbr.num_samples_per_batch, table_size=table_size);
        vbr
    }

    pub(crate) fn next_batch(&mut self) -> Option<(Vec<f32>, u32)> {
        // Stop if we are done
        if self.vectors_read >= self.num_samples {
            return None;
        }

        // Calculate needed samples
        let mut samples_to_read = self.num_samples_per_batch;
        let size_next_batch = self.num_samples
            .saturating_sub(self.vectors_read)
            .saturating_sub(self.num_samples_per_batch);

        if size_next_batch < self.min_samples_per_batch {
            samples_to_read += size_next_batch;
        }

        // Dispatch based on strategy
        if self.random_sampling {
            self.next_batch_random(samples_to_read)
        } else {
            // SEQUENTIAL: Just read from current position
            self.read_and_decode_rows(samples_to_read)
        }
    }

    // -------------------------------------------------------------------------
    // 2. THE NEW RANDOM STRATEGY (Self-Contained)
    // -------------------------------------------------------------------------
    fn next_batch_random(&mut self, samples_to_read: u64) -> Option<(Vec<f32>, u32)> {
        // A. Reset scan to the start
        self.restart_scan();

        // B. Calculate Random Jump
        let total_rows = self.num_tuples();
        let max_start = total_rows.saturating_sub(samples_to_read);

        let random_offset = Spi::get_one::<i64>(&format!("SELECT (random() * {})::bigint", max_start))
            .ok().flatten().unwrap_or(0) as u64;

        debug1!("🎲 [Random] Skipping {} rows...", random_offset);

        // C. The "Burn" Loop (Seek)
        unsafe {
            let scan = self.table_scan.expect("scan not active");
            for _ in 0..random_offset {
                let tuple = pg_sys::systable_getnext(scan);
                if tuple.is_null() { break; }
            }
        }

        // D. Hand off to the standard reader to get the actual data
        self.read_and_decode_rows(samples_to_read)
    }

    // -------------------------------------------------------------------------
    // 3. THE SHARED DECODER (Extracted from your original code)
    // -------------------------------------------------------------------------
    fn read_and_decode_rows(&mut self, count: u64) -> Option<(Vec<f32>, u32)> {
        let start_time = Instant::now();

        unsafe {
            let scan = self.table_scan.expect("scan not initialized");
            let pg_rel = self.pg_rel.clone().expect("rel not initialized");
            let tup_desc = pg_rel.tuple_desc();
            let col_num_nonzero = NonZero::new(self.col_num.unwrap()).expect("column number cannot be zero");

            let mut all_vectors: Vec<f32> = Vec::new();
            let mut dims: u32 = 0;
            let mut read_count = 0;

            for _ in 0..count {
                // 1. Get Tuple
                let tuple = pg_sys::systable_getnext(scan);
                if tuple.is_null() { break; }

                // 2. Get Datum
                let datum = heap_getattr_raw(tuple, col_num_nonzero, tup_desc.as_ptr())
                    .expect("unable to get datum");

                // 3. Detoast & Decode (Your original logic)
                let raw_ptr = datum.cast_mut_ptr();
                let detoasted_ptr = pg_sys::pg_detoast_datum(raw_ptr);

                let byte_slice = pgrx::varlena_to_byte_slice(detoasted_ptr);
                let (vector_values, vector_dims) = vector_type::decode_pgvector_vector(byte_slice);

                if dims == 0 { dims = vector_dims; }
                all_vectors.extend_from_slice(&vector_values);
                read_count += 1;

                // 4. Cleanup
                if detoasted_ptr != raw_ptr {
                    pg_sys::pfree(detoasted_ptr as *mut std::ffi::c_void);
                }
            }

            self.vectors_read += read_count;

            info!("✅ Batch Loaded: {} vectors in {:.2?} (Random: {})",
                read_count, start_time.elapsed(), self.random_sampling);

            if all_vectors.is_empty() { None } else { Some((all_vectors, dims)) }
        }
    }

    // -------------------------------------------------------------------------
    // HELPERS
    // -------------------------------------------------------------------------

    fn restart_scan(&mut self) {
        unsafe {
            if let Some(scan) = self.table_scan {
                pg_sys::systable_endscan(scan);
            }
            let pg_rel = self.pg_rel.as_ref().expect("PgRelation not open");
            let scan = pg_sys::systable_beginscan(
                pg_rel.as_ptr(), pg_sys::InvalidOid, false,
                pg_sys::GetTransactionSnapshot(), 0, std::ptr::null_mut(),
            );
            self.table_scan = Some(scan);
        }
    }

    fn initialize(&mut self) {
        let pg_rel = PgRelation::open_with_name_and_share_lock(&self.table_name)
            .expect("unable to open table");

        {
            let tup_desc = pg_rel.tuple_desc();
            let mut col_num_found: Option<i32> = None;
            for attr in tup_desc.iter().filter(|a| !a.attisdropped) {
                let col_name = pgrx::name_data_to_str(&attr.attname);
                unsafe {
                    if col_name == self.column_name {
                         let type_name = CStr::from_ptr(format_type_be(attr.atttypid)).to_str().unwrap();
                         if type_name != "vector" { pgrx::error!("column type is not vector"); }
                         col_num_found = Some(attr.attnum.into());
                    }
                }
            }
            let col_num = col_num_found.expect("column not found");
            self.col_num = Some(col_num as usize);
        }

        let scan = unsafe {
            pg_sys::systable_beginscan(
                pg_rel.as_ptr(), pg_sys::InvalidOid, false,
                pg_sys::GetTransactionSnapshot(), 0, std::ptr::null_mut(),
            )
        };
        self.table_scan = Some(scan);
        self.pg_rel = Some(pg_rel);
        debug1!("systable scan initialized");
    }

    pub(crate) fn num_tuples(&mut self) -> u64 {
        match self.num_tuples_in_table {
            None => {
                let tuples: i64 = Spi::get_one(format!("SELECT COUNT(1) FROM {}", self.table_name).as_str())
                        .unwrap().unwrap();
                self.num_tuples_in_table = Some(tuples as u64);
                tuples as u64
            }
            Some(tuples) => tuples,
        }
    }

    pub(crate) fn end_scan(self) {
        let scan = self.table_scan.expect("systable scan not initialized");
        unsafe { pg_sys::systable_endscan(scan); }
    }

    fn random_batch_sampling(&mut self) {
        let total_rows = self.num_tuples();
        let total_needed = self.num_samples;
        let batch_size = self.num_samples_per_batch;
        let num_batches = (total_needed + batch_size - 1) / batch_size;

        info!("📊 [Random Batch Sampling] Plan: Batches={}  ", num_batches);

        for i in 0..num_batches {
            let samples_so_far = i * batch_size;
            let remaining = total_needed.saturating_sub(samples_so_far);
            let this_batch_size = std::cmp::min(batch_size, remaining);
            if this_batch_size == 0 { break; }

            let max_start_index = total_rows - this_batch_size;
            let random_offset = Spi::get_one::<i64>(&format!("SELECT (random() * {})::bigint", max_start_index))
                .ok().flatten().unwrap_or(0);

            info!("   Batch #{}: Needs {} | Random range ~ [ {} .. {} ]",
                i + 1, this_batch_size, random_offset, random_offset as u64 + this_batch_size);
        }
    }
}
