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
            // 1. SETUP SNAPSHOT (Dirty Read)
            // Essential for background workers starting early in the lifecycle.
            let mut snapshot = pg_sys::GetLatestSnapshot();
            if snapshot.is_null() {
                snapshot = pg_sys::GetTransactionSnapshot();
            }
            if !snapshot.is_null() {
                pg_sys::PushActiveSnapshot(snapshot);
            }

            // 2. RESOLVE OID (Using RangeVarGetRelidExtended)
            let c_name = CString::new(table_name.clone()).expect("Invalid table name");
            let text_ptr = pg_sys::cstring_to_text(c_name.as_ptr());
            let list_ptr = pg_sys::textToQualifiedNameList(text_ptr);
            let range_var = pg_sys::makeRangeVarFromNameList(list_ptr);

            // PG17 Signature: (relation, lockmode, flags, callback, callback_arg)
            // flags=0 (Default), callback=None, callback_arg=NULL
            let rel_oid = pg_sys::RangeVarGetRelidExtended(
                range_var,
                pg_sys::AccessShareLock as i32,
                0,                  // flags (0 = Panic if missing)
                None,               // callback
                std::ptr::null_mut()// callback_arg
            );

            info!("🧬 [System Scan] Resolved OID: {} for table '{}'", rel_oid, table_name);

            // 3. OPEN RELATION
            let rel = pg_sys::table_open(rel_oid, pg_sys::AccessShareLock as i32);

            // 4. FIND ATTRIBUTE
            let tup_desc = (*rel).rd_att;
            let mut attnum = 0;

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

            // 5. BEGIN HEAP SCAN
            // 6-Argument signature for PG17
            let scan_desc = pg_sys::heap_beginscan(
                rel,
                snapshot,
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(), // ParallelTableScanDesc
                0                     // flags
            );

            // 6. SCAN LOOP
            let mut vectors_loaded = 0;

            // Use the ScanDirection module path found in your grep
            let direction = pg_sys::ScanDirection::ForwardScanDirection;

            loop {
                let tuple = pg_sys::heap_getnext(scan_desc, direction);
                if tuple.is_null() { break; }

                let mut is_null = false;

                // Explicit cast i16 -> i32 for attnum
                let datum = pg_sys::heap_getattr(
                    tuple,
                    attnum as i32,
                    tup_desc,
                    &mut is_null
                );

                if !is_null {
                    // Transmute Datum wrapper to usize -> pointer
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
             panic!("FATAL: [System Scan] Read 0 vectors. The disk appears empty.");
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