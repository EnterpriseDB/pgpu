use crate::vector_type;
use pgrx::{debug1, info, warning, Spi};
use std::time::{SystemTime, UNIX_EPOCH};
use std::time::Instant;

pub struct VectorReadBatcher {
    cursor_name: String,
    target_samples: u64,
    batch_size: u64,
    vectors_read: u64,
    active: bool,
}

impl VectorReadBatcher {
    pub fn new(
        qualified_table_name: String,
        column_name: String,
        num_clusters: u32,
        sampling_factor: u32,
        batch_size: u64,
    ) -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let cursor_name = format!("pgpu_cursor_{}", nanos);

        // 1. Calculate Target
        let target_samples = (num_clusters as u64).saturating_mul(sampling_factor as u64);
        info!("🔍 [DEBUG] Initializing Batcher. Target Samples: {} ({} clusters * {})", target_samples, num_clusters, sampling_factor);

        // 2. Size Estimation (Debug Logs Added)
        let table_bytes: i64 = Spi::get_one(&format!(
            "SELECT pg_relation_size('{}'::regclass)",
            qualified_table_name
        )).expect("SPI failed to look up table size").unwrap_or(0);

        let reltuples: i64 = Spi::get_one(&format!(
            "SELECT reltuples::bigint FROM pg_class WHERE oid = '{}'::regclass",
            qualified_table_name
        )).unwrap_or(Some(0)).unwrap_or(0);

        info!("🔍 [DEBUG] Table Stats -> Physical Size: {} bytes | pg_class.reltuples: {}", table_bytes, reltuples);

        // 3. Fallback Logic
        let estimated_rows: u64 = if table_bytes > 0 {
            // Assume 50 rows per 8KB block (conservative for vectors)
            ((table_bytes / 8192) * 50) as u64
        } else {
            reltuples.max(0) as u64
        };

        info!("🔍 [DEBUG] Final Estimated Rows (Conservative): {}", estimated_rows);

        // 4. Determine Strategy
        let can_use_tablesample = table_bytes > 0; // Cannot use tablesample on size 0 (partition parents/views)
        let use_block_sampling = can_use_tablesample && (target_samples < estimated_rows);

        let query = if use_block_sampling {
            let ratio = target_samples as f64 / estimated_rows as f64;
            let percent = (ratio * 100.0 * 1.2).clamp(0.0001, 100.0);

            info!("📉 Sampling Strategy: Block Sampling (TABLESAMPLE SYSTEM)");
            info!("   ↳ Calculated Percent: {:.6}% (Ratio: {:.6})", percent, ratio);

            format!(
                "SELECT {} FROM {} TABLESAMPLE SYSTEM({:.4})",
                column_name, qualified_table_name, percent
            )
        } else {
            info!("📉 Sampling Strategy: Full Sequential Scan");
            if !can_use_tablesample {
                info!("   ↳ Reason: Physical size is 0 (Partitioned Table or View?)");
            } else {
                info!("   ↳ Reason: Target ({}) >= Est Rows ({})", target_samples, estimated_rows);
            }

            format!("SELECT {} FROM {}", column_name, qualified_table_name)
        };

        info!("🔍 [DEBUG] Declaring Cursor: \"{}\"", cursor_name);
        info!("🔍 [DEBUG] Query: {}", query);

        Spi::run(&format!("DECLARE \"{}\" NO SCROLL CURSOR FOR {}", cursor_name, query))
             .expect("failed to declare sampling cursor");

        VectorReadBatcher {
            cursor_name,
            target_samples,
            batch_size,
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
        let fetch_sql = format!("FETCH FORWARD {} FROM \"{}\"", self.batch_size, self.cursor_name);

        // info!("🔍 [DEBUG] Executing Fetch: {}", fetch_sql); // Uncomment if very verbose needed

        let (all_vectors, dims, read_count) = Spi::connect(|client| {
            let mut vectors: Vec<f32> = Vec::new();
            let mut dims: u32 = 0;
            let mut count = 0;

            // Use .expect to force panic with message if cursor is gone
            let table = client.select(&fetch_sql, None, &[])
                .expect("SPI SELECT failed inside next_batch (Cursor may be closed or invalid)");

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
                    } else {
                         // Very verbose: log if we hit NULLs
                         // warning!("🔍 [DEBUG] Row found but datum was NULL or invalid");
                    }
                }
            }
            (vectors, dims, count)
        });

        if read_count == 0 {
            info!("🔍 [DEBUG] Cursor exhausted. Closing scan. (Total Read: {})", self.vectors_read);
            self.end_scan();
            return None;
        }

        self.vectors_read += read_count as u64;
        debug1!("✅ Batch Loaded: {} vectors in {:.2?}", read_count, start_time.elapsed());

        Some((all_vectors, dims))
    }

    pub(crate) fn end_scan(&mut self) {
        if self.active {
            let _ = Spi::run(&format!("CLOSE \"{}\"", self.cursor_name));
            self.active = false;
        }
    }
}