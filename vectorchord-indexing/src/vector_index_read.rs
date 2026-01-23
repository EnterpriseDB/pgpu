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
        // Unique cursor name
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let cursor_name = format!("pgpu_cursor_{}", nanos);

        // 1. Calculate Target Sample Count
        let target_samples = (num_clusters as u64).saturating_mul(sampling_factor as u64);

        // 2. Robust Size Estimation (No ANALYZE dependency)
        // We check the physical file size on disk.
        let table_bytes: i64 = Spi::get_one(&format!(
            "SELECT pg_relation_size('{}'::regclass)",
            qualified_table_name
        )).expect("SPI failed to look up table size").unwrap_or(0);

        let table_bytes = table_bytes.max(0) as u64;
        let total_blocks = table_bytes / 8192; // Standard Postgres Page Size

        // Conservative Estimate: Assume 50 rows per block (for TOASTed vectors).
        // This is safe: If real density is higher (200 rows), we just read 4x more data (fast).
        // If we assumed high density and it was low, we would run out of data.
        let est_rows_from_disk = total_blocks * 50;

        // 3. Determine Query Strategy
        // We use Block Sampling if the estimated rows are significantly larger than our target.
        let use_block_sampling = total_blocks > 100 && target_samples < (est_rows_from_disk / 2);

        let query = if use_block_sampling {
            // A. FAST BLOCK SAMPLING (TABLESAMPLE SYSTEM)

            // Calculate ratio based on our conservative disk estimate
            let ratio = target_samples as f64 / est_rows_from_disk as f64;

            // Multiply by 1.2x safety factor
            let percent = (ratio * 100.0 * 1.2).clamp(0.0001, 100.0);

            info!("📉 Sampling Strategy: Block Sampling (TABLESAMPLE SYSTEM)");
            info!("   ↳ Table Size: {} MB ({} Blocks)", table_bytes / 1024 / 1024, total_blocks);
            info!("   ↳ Target: {} vectors | Reading: {:.4}% of blocks", target_samples, percent);

            format!(
                "SELECT {} FROM {} TABLESAMPLE SYSTEM({:.4})",
                column_name, qualified_table_name, percent
            )
        } else {
            // B. FULL SEQUENTIAL SCAN (Fallback)
            // Used for small tables or when we need a huge % of data.
            info!("📉 Sampling Strategy: Full Sequential Scan");
            info!("   ↳ Reason: Table too small (<100 blocks) or Target Sample > 50% of table");

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
                        // Explicitly ask for pg_sys::Datum to resolve type inference error
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