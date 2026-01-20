use crate::vector_type;
use pgrx::pg_sys;
use pgrx::info;
use std::ffi::CString;
use std::time::Instant;

pub struct VectorReadBatcher {
    num_samples: u64,
    num_samples_per_batch: u64,
    min_samples_per_batch: u64,
    vectors_read: u64,
    cached_vectors: Vec<f32>,
    dims: u32,
}

impl VectorReadBatcher {
    pub fn new(
        table_name: String,
        _column_name: String, // Unused for count(*)
        num_samples: u64,
        num_samples_per_batch: u64,
        min_samples_per_batch: u64,
    ) -> Self {
        let start_time = Instant::now();
        let mut row_count = 0;

        unsafe {
            // 1. VISIBILITY: Force "Dirty Read" snapshot
            // We need this to see data created before this process started.
            let mut snapshot = pg_sys::GetLatestSnapshot();
            if snapshot.is_null() {
                snapshot = pg_sys::GetTransactionSnapshot();
            }
            if !snapshot.is_null() {
                pg_sys::PushActiveSnapshot(snapshot);
            }

            // 2. OID RESOLUTION
            let c_table = CString::new(table_name.clone()).unwrap();

            // Resolve Table OID (Standard C-API)
            let text_ptr = pg_sys::cstring_to_text(c_table.as_ptr());
            let list_ptr = pg_sys::textToQualifiedNameList(text_ptr);
            let range_var = pg_sys::makeRangeVarFromNameList(list_ptr);

            let rel_oid = pg_sys::RangeVarGetRelidExtended(
                range_var,
                pg_sys::AccessShareLock as i32,
                0, None, std::ptr::null_mut()
            );

            info!("🧬 [Count Check] OID Resolved: {}", rel_oid);

            // 3. OPEN RELATION
            let rel = pg_sys::table_open(rel_oid, pg_sys::AccessShareLock as i32);

            // 4. BEGIN HEAP SCAN
            // We scan the table, but we DO NOT ask for any columns.
            let scan_desc = pg_sys::heap_beginscan(
                rel,
                snapshot,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0
            );

            // 5. COUNT LOOP (No Data Reading)
            // Use '1' for ForwardScanDirection
            let direction = std::mem::transmute(1 as i32);

            loop {
                // Just get the next tuple handle.
                let tuple = pg_sys::heap_getnext(scan_desc, direction);

                // If null, we hit the end of the table.
                if tuple.is_null() { break; }

                // If not null, the row exists and is visible!
                row_count += 1;
            }

            // 6. CLEANUP
            pg_sys::heap_endscan(scan_desc);
            pg_sys::table_close(rel, pg_sys::AccessShareLock as i32);

            if !snapshot.is_null() {
                pg_sys::PopActiveSnapshot();
            }
        }

        info!("✅ [Count Check] Table: {} | Rows Found: {} | Time: {:.2?}",
              table_name, row_count, start_time.elapsed());

        // Stop here. Do not crash. Just report the count.
        if row_count == 0 {
             panic!("FATAL: Visibility Check Failed. Table appears empty despite being 5GB on disk.");
        }

        // Return a dummy batcher so the test doesn't crash immediately,
        // allowing you to see the log output.
        VectorReadBatcher {
            num_samples,
            num_samples_per_batch,
            min_samples_per_batch,
            vectors_read: 0,
            cached_vectors: vec![], // Empty
            dims: 1,
        }
    }

    pub fn next_batch(&mut self) -> Option<(Vec<f32>, u32)> {
        // Return None immediately to stop the training gracefully
        None
    }

    pub(crate) fn end_scan(self) {}
}