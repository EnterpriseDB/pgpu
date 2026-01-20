use crate::vector_type;
use pgrx::pg_sys::{self, Datum, varlena};
use pgrx::{info, Spi};
use std::ffi::CStr;
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

        // 1. RESOLVE OID VIA SPI (Safe & Easy)
        let rel_oid = Spi::connect(|client| {
            // We use regclass to safely resolve schema.table to an OID
            let query = format!("SELECT '{}'::regclass::oid", table_name);
            client.select(&query, None, &[])
                .and_then(|t| t.get_one::<pg_sys::Oid>())
        })
        .expect("SPI Connection Failed")
        .expect("Table not found (OID resolution failed)");

        info!("🧬 [System Scan] Table '{}' -> OID: {}", table_name, rel_oid);

        unsafe {
            // 2. OPEN RELATION
            // AccessShareLock (1) allows concurrent reads
            let rel = pg_sys::table_open(rel_oid, pg_sys::AccessShareLock as i32);

            // 3. FIND ATTRIBUTE NUMBER (Column Index)
            let tup_desc = (*rel).rd_att;
            let mut attnum = 0;

            // Fix: Use .as_ptr().add() for correct pointer arithmetic on flexible array
            for i in 0..(*tup_desc).natts {
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

            // 4. PREPARE SCAN (PG17+ Compatible)
            // We use GetLatestSnapshot to bypass MVCC visibility rules (Dirty Read)
            let snapshot = pg_sys::GetLatestSnapshot();
            if snapshot.is_null() {
                // If null, we try to grab the transaction snapshot as fallback
                pg_sys::PushActiveSnapshot(pg_sys::GetTransactionSnapshot());
            }

            // table_beginscan is inline; use table_beginscan_strat
            let scan_desc = pg_sys::table_beginscan_strat(
                rel,
                pg_sys::GetLatestSnapshot(), // Use Latest to see "everything"
                0,
                std::ptr::null_mut(),
                true,
                false
            );

            // 5. CREATE TUPLE SLOT (Mandatory in PG17)
            // heap_getnext is gone. We must use slots.
            let slot = pg_sys::MakeSingleTupleTableSlot(
                tup_desc,
                &pg_sys::TTSOpsHeapTuple
            );

            // 6. EXECUTE SCAN
            let mut vectors_loaded = 0;
            let forward = pgrx::pg_sys::ScanDirection::ForwardScanDirection;

            loop {
                // Modern scanning: Get next slot
                let has_data = pg_sys::table_scan_getnextslot(scan_desc, forward, slot);
                if !has_data { break; } // End of table

                let mut is_null = false;

                // Extract Datum from Slot
                let datum = pg_sys::slot_getattr(
                    slot,
                    attnum,
                    &mut is_null
                );

                if !is_null {
                    // Fix: Datum cast using .value() for pgrx's Datum wrapper or direct cast
                    // In pgrx raw bindings, Datum is often a usize/uintptr_t
                    let ptr = datum.value() as *mut varlena;
                    let byte_slice = pgrx::varlena_to_byte_slice(ptr);

                    let (vals, d_dims) = vector_type::decode_pgvector_vector(byte_slice);
                    if detected_dims == 0 { detected_dims = d_dims; }
                    vecs.extend(vals);
                    vectors_loaded += 1;

                    if vectors_loaded >= num_samples {
                        break;
                    }
                }

                // Clear slot for next iteration
                pg_sys::ExecClearTuple(slot);
            }

            // 7. CLEANUP
            pg_sys::ExecDropSingleTupleTableSlot(slot);
            pg_sys::table_endscan(scan_desc);
            pg_sys::table_close(rel, pg_sys::AccessShareLock as i32);
        }

        // 8. FINAL SAFETY CHECK
        let safe_dims = if detected_dims == 0 { 1 } else { detected_dims };
        let count_loaded = vecs.len() / (safe_dims as usize);

        if count_loaded == 0 {
             panic!("FATAL: [System Scan] Read 0 vectors. The physical table file is truly empty.");
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