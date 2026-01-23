use crate::vector_type;
use pgrx::{debug1, info, warning, Spi};
use std::time::{SystemTime, UNIX_EPOCH};
use std::time::Instant;

pub struct VectorReadBatcher {
    cursor_name: String,
    target_samples: u64,
    internal_batch_size: u64, // Separated from external batch_size
    vectors_read: u64,
    active: bool,
}

impl VectorReadBatcher {
    pub fn new(
        qualified_table_name: String,
        column_name: String,
        num_clusters: u32,
        sampling_factor: u32,
        requested_batch_size: u64,
    ) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let cursor_name = format!("pgpu_cursor_{}", nanos);

        // 1. Calculate Target
        let target_samples = (num_clusters as u64).saturating_mul(sampling_factor as u64);

        // 2. Physical Size Check
        let table_bytes: i64 = Spi::get_one(&format!(
            "SELECT pg_relation_size('{}'::regclass)",
            qualified_table_name
        )).expect("SPI failed to look up table size").unwrap_or(0);

        // 3. Robust Estimation (Based on your ps output: 157 rows/block)
        // We use 160 as the standard density for TOASTed vector tables (768d).
        let estimated_rows: u64 = if table_bytes > 0 {
            ((table_bytes / 8192) * 160) as u64
        } else {
            0
        };

        info!("🔍 [DEBUG] Table Size: {} MB | Est Rows: {} (Density: 160/block) | Target: {}",
            table_bytes / 1024 / 1024, estimated_rows, target_samples);

        // 4. Strategy Selection
        // Use Block Sampling if target is less than 90% of our estimate
        let can_use_tablesample = table_bytes > 0;
        let use_block_sampling = can_use_tablesample && (target_samples < (estimated_rows as f64 * 0.9) as u64);

        let query = if use_block_sampling {
            let ratio = target_samples as f64 / estimated_rows as f64;
            // 1.15x buffer to be safe
            let percent = (ratio * 100.0 * 1.15).clamp(0.0001, 100.0);

            info!("📉 Sampling Strategy: Block Sampling (TABLESAMPLE SYSTEM)");
            info!("   ↳ Reading: {:.4}% of blocks to get ~{} vectors", percent, target_samples);

            format!(
                "SELECT {} FROM {} TABLESAMPLE SYSTEM({:.4})",
                column_name, qualified_table_name, percent
            )
        } else {
            info!("📉 Sampling Strategy: Full Sequential Scan");
            info!("   ↳ Reason: Target Sample size is nearly the full table size.");

            format!("SELECT {} FROM {}", column_name, qualified_table_name)
        };

        Spi::run(&format!("DECLARE \"{}\" NO SCROLL CURSOR FOR {}", cursor_name, query))
             .expect("failed to declare sampling cursor");

        // CRITICAL FIX: Cap internal fetch size to 50k to prevent OOM crash.
        // Your code still gets 'requested_batch_size' eventually, but we fetch from DB in small sips.
        let internal_batch_size = requested_batch_size.clamp(1000, 50_000);

        VectorReadBatcher {
            cursor_name,
            target_samples,
            internal_batch_size,
            vectors_read: 0,
            active: true,
        }
    }

    pub(crate) fn next_batch(&mut self) -> Option<(Vec<f32>, u32)> {
        if !self.active || self.vectors_read >= self.target_samples {
            if self.active { self.end_scan(); }
            return None;
        }

        let start_time = Instant::now();
        // Use the capped internal_batch_size (50k), NOT the huge 10M one
        let fetch_sql = format!("FETCH FORWARD {} FROM \"{}\"", self.internal_batch_size, self.cursor_name);

        let (all_vectors, dims, read_count) = Spi::connect(|client| {
            let mut vectors: Vec<f32> = Vec::new();
            let mut dims: u32 = 0;
            let mut count = 0;

            let table = client.select(&fetch_sql, None, &[])
                .expect("SPI SELECT failed inside next_batch");

            for row in table {
                if let Ok(entry) = row.get_datum_by_ordinal(1) {
                    if let Ok(Some(datum)) = entry.value::<pgrx::pg_sys::Datum>() {
                        unsafe {
                            let raw_ptr = datum.cast_mut_ptr();
                            let detoasted_ptr = pgrx::pg_sys::pg_detoast_datum(raw_ptr);
                            let byte_slice = pgrx::varlena_to_byte_slice(detoasted_ptr);
                            let (vec_vals, vec_dims) = vector_type::decode_pgvector_vector(byte_slice);

                            if dims == 0 { dims = vec_dims; }
                            vectors.extend_from_slice(&vec_vals);
                            count += 1;

                            if detoasted_ptr != raw_ptr {
                                pgrx::pg_sys::pfree(detoasted_ptr as *mut std::ffi::c_void);
                            }
                        }
                    }
                }
            }
            (vectors, dims, count)
        });

        if read_count == 0 {
            info!("🔍 [DEBUG] Cursor exhausted. (Read: {} / Target: {})", self.vectors_read, self.target_samples);
            self.end_scan();
            return None;
        }

        self.vectors_read += read_count as u64;
        // Comment out debug1 to reduce log spam if fetching small chunks
        // debug1!("✅ Batch Loaded: {} vectors in {:.2?}", read_count, start_time.elapsed());

        Some((all_vectors, dims))
    }

    pub(crate) fn end_scan(&mut self) {
        if self.active {
            let _ = Spi::run(&format!("CLOSE \"{}\"", self.cursor_name));
            self.active = false;
        }
    }
}