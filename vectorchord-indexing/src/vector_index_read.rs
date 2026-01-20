use crate::vector_type;
use pgrx::pg_sys::{self, Datum, varlena};
use pgrx::info; // info! macro
use std::ffi::{CStr, CString}; // <--- FIXED IMPORTS
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
            // 1. RESOLVE TABLE OID (System Level)
            // Convert Rust String -> CString -> Postgres Text* -> QualifiedName List
            let c_name = CString::new(table_name.clone()).expect("Invalid table name");
            let text_ptr = pg_sys::cstring_to_text(c_name.as_ptr());
            let raw_name_list = pg_sys::textToQualifiedNameList(text_ptr);

            // Create RangeVar (abstract table reference)
            let range_var = pg_sys::makeRangeVarFromNameList(raw_name_list);

            // Get OID.
            // args: RangeVar, LockMode (NoLock), missing_ok (false)
            let rel_oid = pg_sys::RangeVarGetRelid(
                range_var,
                pg_sys::NoLock as i32,
                false
            );

            info!("🧬 [System Scan] Resolved Table '{}' -> OID: {}", table_name, rel_oid);

            // 2. OPEN RELATION
            // AccessShareLock lets us read while others read
            let rel = pg_sys::table_open(rel_oid, pg_sys::AccessShareLock as i32);

            // 3. FIND COLUMN ATTRIBUTE
            let tup_desc = (*rel).rd_att;
            let mut attnum = 0;

            // Iterate over table columns (attributes)
            for i in 0..(*tup_desc).natts {
                let attr = *(*tup_desc).attrs.add(i as usize);
                let name_ptr = attr.attname.data.as_ptr();
                let name = CStr::from_ptr(name_ptr).to_string_lossy(); // <--- Standard CStr usage

                if name == column_name {
                    attnum = attr.attnum;
                    break;
                }
            }

            if attnum == 0 {
                pg_sys::table_close(rel, pg_sys::AccessShareLock as i32);
                panic!("FATAL: Column '{}' not found in table schema.", column_name);
            }

            // 4. BEGIN SCAN (Using GetLatestSnapshot for Dirty Read)
            // We use GetLatestSnapshot() to see the raw disk state
            let snapshot = pg_sys::GetLatestSnapshot();

            let scan_desc = pg_sys::table_beginscan(
                rel,
                snapshot,
                0,
                std::ptr::null_mut()
            );

            // 5. ITERATE HEAP TUPLES
            let mut vectors_loaded = 0;
            // Optional: You can implement a counter here to skip N rows for "offset"
            // let skip_count = 0;

            loop {
                // Get next raw tuple from disk
                let tuple = pg_sys::heap_getnext(scan_desc, pg_sys::ForwardScanDirection);

                // If tuple is null, we reached the end of the file
                if tuple.is_null() {
                    break;
                }

                // Extract the specific column (Datum)
                let mut is_null = false;
                let datum = pg_sys::heap_getattr(
                    tuple,
                    attnum,
                    tup_desc,
                    &mut is_null
                );

                if !is_null {
                    // Cast Datum -> varlena* -> Byte Slice
                    let byte_slice = pgrx::varlena_to_byte_slice(datum as *mut varlena);
                    let (vals, d_dims) = vector_type::decode_pgvector_vector(byte_slice);

                    if detected_dims == 0 { detected_dims = d_dims; }
                    vecs.extend(vals);
                    vectors_loaded += 1;

                    if vectors_loaded >= num_samples {
                        break;
                    }