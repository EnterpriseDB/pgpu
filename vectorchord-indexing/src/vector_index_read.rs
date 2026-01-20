use crate::vector_type;
use pgrx::pg_sys::{self, varlena};
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
        column_name: String,
        num_samples: u64,
        num_samples_per_batch: u64,
        min_samples_per_batch: u64,
    ) -> Self {
        let start_time = Instant::now();
        let mut vecs = Vec::new();
        let mut detected_dims = 0;

        unsafe {
            // 1. SETUP SNAPSHOT
            // Crucial for visibility in background workers
            let mut snapshot = pg_sys::GetLatestSnapshot();
            if snapshot.is_null() {
                snapshot = pg_sys::GetTransactionSnapshot();
            }
            if !snapshot.is_null() {
                pg_sys::PushActiveSnapshot(snapshot);
            }

            // 2. RESOLVE TABLE OID
            let c_table = CString::new(table_name.clone()).expect("Invalid table name");
            let text_ptr = pg_sys::cstring_to_text(c_table.as_ptr());
            let list_ptr = pg_sys::textToQualifiedNameList(text_ptr);
            let range_var = pg_sys::makeRangeVarFromNameList(list_ptr);

            let rel_oid = pg_sys::RangeVarGetRelidExtended(
                range_var,
                pg_sys::AccessShareLock as i32,
                0,
                None,
                std::ptr::null_mut()
            );

            // 3. RESOLVE ATTRIBUTE NUMBER (Column ID)
            let c_col = CString::new(column_name.clone()).expect("Invalid column name");
            let attnum = pg_sys::get_attnum(rel_oid, c_col.as_ptr());

            if attnum <= 0 {
                panic!("FATAL: Column '{}' not found in table '{}' (Attnum: {}).", column_name, table_name, attnum);
            }

            info!("🧬 [Safe Scan] OID: {} | AttNum: {}", rel_oid, attnum);

            // 4. OPEN RELATION
            let rel = pg_sys::table_open(rel_oid, pg_sys::AccessShareLock as i32);
            let tup_desc = (*rel).rd_att;

            // 5. BEGIN HEAP SCAN
            let scan_desc = pg_sys::heap_beginscan(
                rel,
                snapshot,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0
            );

            // 6. SCAN LOOP
            let mut vectors_loaded = 0;
            let direction = pg_sys::ScanDirection::ForwardScanDirection;

            loop {
                let tuple = pg_sys::heap_getnext(scan_desc, direction);
                if tuple.is_null() { break; }

                let mut is_null = false;
                let datum = pg_sys::heap_getattr(
                    tuple,
                    attnum as i32,
                    tup_desc,
                    &mut is_null
                );

                if !is_null {
                    // --- FIX 1: Transmute Datum Wrapper to Primitive ---
                    // pgrx Datum is a struct wrapping a usize/pointer. We unwrap it.
                    let val: usize = std::mem::transmute(datum);
                    let raw_ptr = val as *mut varlena;

                    // --- FIX 2: Detoast to prevent Segfaults ---
                    // pg_detoast_datum ensures the data is in memory and decompressed.
                    let safe_ptr = pg_sys::pg_detoast_datum(raw_ptr);

                    // Decode
                    let byte_slice = pgrx::varlena_to_byte_slice(safe_ptr);
                    let (vals, d_dims) = vector_type::decode_pgvector_vector(byte_slice);

                    if detected_dims == 0 { detected_dims = d_dims; }
                    vecs.extend(vals);
                    vectors_loaded += 1;

                    if vectors_loaded >= num_samples {
                        break;