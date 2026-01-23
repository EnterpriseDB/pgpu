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
        // Fix: Use SystemTime for a unique cursor suffix instead of missing pg_sys function
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let cursor_name = format!("pgpu_cursor_{}", nanos);

        // 1. Calculate Target Count
        let target_samples = (num_clusters as u64).saturating_mul(sampling_factor as u64);

        // 2. Get Total Row Estimate
        let total_rows: i64 = Spi::get_one(&format!(
            "SELECT reltuples::bigint FROM pg_class WHERE oid = '{}'::regclass",
            qualified_table_name
        )).expect("SPI failed to look up table size").unwrap_or(1_000_000);

        let total_rows = total_rows.max(1) as u64;

        // 3. Determine Query Strategy
        let query = if target_samples < total_rows {
            // A. FAST BLOCK SAMPLING (TABLESAMPLE SYSTEM)
            let ratio = target_samples as f64 / total_rows as f64;
            // Request 10% extra (1.1x) to cover empty pages/dead tuples
            let percent = (ratio * 100.0 * 1.1).clamp(0.0001, 100.0);

            info!("📉 Sampling Strategy: Block Sampling (TABLESAMPLE SYSTEM)");
            info!("   ↳ Target: {} vectors ({} clusters * {} factor)", target_samples, num_clusters, sampling_factor);
            info!("   ↳ Reading: {:.4}% of table blocks (Est. Total Rows: {})", percent, total_rows);

            format!(
                "SELECT {} FROM {} TABLESAMPLE SYSTEM({:.4})",
                column_name, qualified_table_name, percent
            )
        } else {
            // B. FULL SEQUENTIAL SCAN (Fallback)
            info!("📉 Sampling Strategy: Full Sequential Scan (Target > Total Table Size)");
            format!("SELECT {} FROM {}", column_name, qualified_table_name)
        };

        // 4. Declare the Cursor
        Spi::run(&format!("DECLARE \"{}\" CURSOR FOR {}", cursor_name, query))
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
        let fetch_sql = format!("FETCH {} FROM \"{}\"", self.batch_size, self.cursor_name);

        let (all_vectors, dims, read_count) = Spi::connect(|client| {
            let mut vectors: Vec<f32> = Vec::new();
            let mut dims: u32 = 0;
            let mut count = 0;

            if let Ok(table) = client.select(&fetch_sql, None, &[]) {
                for row in table {
                    if let Ok(entry) = row.get_datum_by_ordinal(1) {
                        // Explicitly ask for pg_sys::Datum
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
            }
            (vectors, dims, count)
        });

        if read_count == 0 {
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