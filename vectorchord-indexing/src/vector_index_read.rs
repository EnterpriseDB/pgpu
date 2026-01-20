use crate::vector_type;
use pgrx::pg_sys::{self, Datum, varlena};
use pgrx::{info, Spi};
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

        // 1. RESOLVE OID SAFELY (Via SPI to avoid C string complexity)
        let rel_oid = Spi::connect(|client| {
            client.select(&format!("SELECT '{}'::regclass::oid", table_name), None, &[])
                .and_then(|t| t.get_one::<pg_sys::Oid>())
        })
        .expect("SPI Connect Failed")
        .expect("Table OID not found");

        info!("🧬 [Raw Heap Scan] OID: {} | Target: {}", rel_oid, table_name);

        unsafe {
            // 2. OPEN RELATION
            // AccessShareLock (1)
            let rel = pg_sys::table_open(rel_oid, pg_sys::AccessShareLock as i32);

            // 3. FIND ATTRIBUTE (Column)
            let tup_desc = (*rel).rd_att;
            let mut attnum = 0;

            for i in 0..(*tup_desc).natts {
                // Pointer arithmetic to get the attribute
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
                panic!("FATAL: Column '{}' not found.", column_name);
            }

            // 4. BEGIN HEAP SCAN
            // We use heap_beginscan directly. If this fails to link, we are in deep trouble,
            // but it is the standard non-inline way to scan in older versions (and often wrapped).
            // We use GetLatestSnapshot() to ensure we see data on disk.
            let snapshot = pg_sys::GetLatestSnapshot();
            if snapshot.is_null() {
                // Fallback if Latest is null (rare)
                 pg_sys::PushActiveSnapshot(pg_sys::GetTransactionSnapshot());
            }

            // Note: If heap_beginscan is missing in your specific pg_sys,
            // the compiler will complain, but it's our best bet over table_beginscan.
            let scan_desc = pg_sys::heap_beginscan(
                rel,
                pg_sys::GetLatestSnapshot(),
                0,
                std::ptr::null_mut()
            );

            // 5. ITERATE (Using heap_getnext)
            // We hardcode ForwardScanDirection = 1 to avoid the missing Enum constant error.
            let forward_scan: i32 = 1;
            // We cast it to the Enum type if required by Rust, or pass as i32 if bindings are loose.
            // In strict Rust bindings, this might need: std::mem::transmute(1) or similar.
            // But usually the binding expects the Enum type.
            let direction = std::mem::transmute::<i32, pg_sys::ScanDirection>(forward_scan);

            let mut vectors_loaded = 0;

            loop {
                let tuple = pg_sys::heap_getnext(scan_desc, direction);

                if tuple.is_null() { break; }

                let mut is_null = false;

                // Use heap_getattr (which your logs said EXISTS)
                let datum = pg_sys::heap_getattr(
                    tuple,
                    attnum,
                    tup_desc,
                    &mut is_null
                );

                if !is_null {
                    // Extract data
                    let ptr = datum as *mut varlena; // Direct cast for pgrx Datum
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

            // 6. CLEANUP
            pg_sys::heap_endscan(scan_desc);
            pg_sys::table_close(rel, pg_sys::AccessShareLock as i32);
        }

        // 7. CHECK
        let safe_dims = if detected_dims == 0 { 1 } else { detected_dims };
        let count_loaded = vecs.len() / (safe_dims as usize);

        if count_loaded == 0 {
             panic!("FATAL: [Raw Heap Scan] Read 0 vectors. The disk is empty or snapshot is invalid.");
        }

        info!("✅ [Raw Heap Scan] Success: Loaded {} vectors in {:.2?}", count_loaded, start_time.elapsed());

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
        if remaining < (self.num_samples_per_batch + self.min_samples_per_batch) as usize { to_read = remaining; }
        let start = (self.vectors_read as usize) * (self.dims as usize);
        let end = start + (to_read * self.dims as usize);
        if end > self.cached_vectors.len() { return None; }
        let batch = self.cached_vectors[start..end].to_vec();
        self.vectors_read += to_read as u64;
        Some((batch, self.dims))
    }

    pub(crate) fn end_scan(self) {}
}