use crate::vector_type;
use pgrx::pg_sys::{self, varlena};
use pgrx::info;
use std::ffi::{CStr, CString};
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
        column_name: String,
        num_samples: u64,
        num_samples_per_batch: u64,
        min_samples_per_batch: u64,
    ) -> Self {
        let start_time = Instant::now();
        let mut vecs = Vec::new();
        let mut detected_dims = 0;

        unsafe {
            // 1. SETUP SNAPSHOT (CRITICAL FIX)
            // We push the snapshot BEFORE doing anything else.
            // GetLatestSnapshot() ignores transaction rules and sees disk data directly.
            let mut snapshot = pg_sys::GetLatestSnapshot();
            if snapshot.is_null() {
                snapshot = pg_sys::GetTransactionSnapshot();
            }
            if !snapshot.is_null() {
                pg_sys::PushActiveSnapshot(snapshot);
            }

            // 2. RESOLVE TABLE (Pure C-API, No SPI)
            // We construct a RangeVar (internal representation of "schema.table")
            // and ask Postgres to open it.
            let c_name = CString::new(table_name.clone()).expect("Invalid table name");

            // makeRangeVarFromNameList expects a List*, which is hard to construct manually.
            // Instead, we try the string-to-text path which is standard.
            let text_ptr = pg_sys::cstring_to_text(c_name.as_ptr());
            let list_ptr = pg_sys::textToQualifiedNameList(text_ptr);
            let range_var = pg_sys::makeRangeVarFromNameList(list_ptr);

            // table_openrv works like table_open but takes the RangeVar we just made.
            // If table_openrv is missing, we fall back to finding OID manually.
            // Based on PG17, RangeVarGetRelid is the safe bet.
            let rel_oid = pg_sys::RangeVarGetRelid(
                range_var,
                pg_sys::NoLock as i32,
                false // missing_ok? False = Panic if missing
            );

            info!("🧬 [System Scan] Resolved OID: {}", rel_oid);

            // 3. OPEN RELATION
            let rel = pg_sys::table_open(rel_oid, pg_sys::AccessShareLock as i32);

            // 4. FIND COLUMN ATTRIBUTE
            let tup_desc = (*rel).rd_att;
            let mut attnum = 0;

            for i in 0..(*tup_desc).natts {
                // Pointer arithmetic to iterate C-array
                let attr = *(*tup_desc).attrs.as_ptr().add(i as usize);
                let name_ptr = attr.attname.data.as_ptr();
                let name = CStr::from_ptr(name_ptr).to_string_lossy();

                if name == column_name {
                    attnum = attr.attnum;
                    break;
                }
            }

            if attnum == 0 {
                pg_sys::table_close(rel, pg_sys::AccessShareLock as i32);
                panic!("FATAL: Column '{}' not found in table schema.", column_name);
            }

            // 5. BEGIN HEAP SCAN
            // Using the 6-arg signature confirmed by grep
            let scan_desc = pg_sys::heap_beginscan(
                rel,
                snapshot,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(), // Parallel Scan Desc (NULL)
                0                     // Flags
            );

            // 6. SCAN LOOP
            let mut vectors_loaded = 0;

            // Use the constant found in your grep
            let direction = pg_sys::ScanDirection::ForwardScanDirection;

            loop {
                // heap_getnext (Legacy API)
                let tuple = pg_sys::heap_getnext(scan_desc, direction);

                if tuple.is_null() { break; }

                let mut is_null = false;

                // heap_getattr (Legacy API)
                let datum = pg_sys::heap_getattr(
                    tuple,
                    attnum as i32, // Explicit cast i16 -> i32
                    tup_desc,
                    &mut is_null
                );

                if !is_null {
                    // FIX: Transmute Datum to usize to bypass struct wrapper
                    let val: usize = std::mem::transmute(datum);
                    let ptr = val as *mut varlena;

                    let byte_slice = pgrx::varlena_to_byte_slice(ptr);
                    let (vals, d_dims) = vector_type::decode_pgvector_vector(byte_slice);

                    if detected_dims == 0 { detected_dims = d_dims; }
                    vecs.extend(vals);
                    vectors_loaded += 1;

                    if vectors_loaded >= num_samples {
                        break;
                    }
                }
            }

            // 7. CLEANUP
            pg_sys::heap_endscan(scan_desc);
            pg_sys::table_close(rel, pg_sys::AccessShareLock as i32);

            if !snapshot.is_null() {
                pg_sys::PopActiveSnapshot();
            }
        }

        let safe_dims = if detected_dims == 0 { 1 } else { detected_dims };
        let count_loaded = vecs.len() / (safe_dims as usize);

        if count_loaded == 0 {
             panic!("FATAL: [System Scan] Read 0 vectors. Disk is empty or visibility failed.");
        }

        info!("✅ [System Scan] Success: Loaded {} vectors in {:.2?}", count_loaded, start_time.elapsed());

        VectorReadBatcher {
            num_samples,
            num_samples_per_batch,
            min_samples_per_batch,
            vectors_read: 0,
            cached_vectors: vecs,
            dims: safe_dims,
        }
    }

    pub fn next_batch(&mut self) -> Option<(Vec<f32>, u32)> {
        if self.cached_vectors.is_empty() || self.vectors_read >= self.num_samples { return None; }

        let mut to_read = self.num_samples_per_batch as usize;
        let remaining = (self.num_samples - self.vectors_read) as usize;
        if remaining < (self.num_samples_per_batch + self.min_samples_per_batch) as usize {
            to_read = remaining;
        }

        let start = (self.vectors_read as usize) * (self.dims as usize);
        let end = start + (to_read * self.dims as usize);

        if end > self.cached_vectors.len() { return None; }

        let batch = self.cached_vectors[start..end].to_vec();
        self.vectors_read += to_read as u64;
        Some((batch, self.dims))
    }

    pub(crate) fn end_scan(self) {}
}