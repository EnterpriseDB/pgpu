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
             panic!("FATAL: [Safe Scan] Read 0 vectors. The disk is empty.");
        }

        info!("✅ [Safe Scan] Success: Loaded {} vectors in {:.2?}", count_loaded, start_time.elapsed());

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