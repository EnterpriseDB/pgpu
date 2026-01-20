use crate::vector_type;
use pgrx::pg_sys::{self, Datum, varlena};
use pgrx::{info, Spi, warning};
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
        raw_table_name: String,
        column_name: String,
        num_samples: u64,
        num_samples_per_batch: u64,
        min_samples_per_batch: u64,
    ) -> Self {
        let start_time = Instant::now();

        // 1. FORCE "DIRTY READ" VISIBILITY (Crucial for Cold Start)
        // We use GetLatestSnapshot() to bypass transaction isolation and see
        // data that exists on disk immediately after DB startup.
        unsafe {
            pg_sys::SetCurrentStatementStartTimestamp();
            if !pg_sys::ActiveSnapshotSet() {
                let snap = pg_sys::GetLatestSnapshot();
                if snap.is_null() {
                    // If even Latest is null, we are too early, but we try anyway.
                    warning!("⚠️ Critical: GetLatestSnapshot() returned NULL. Reading might fail.");
                } else {
                    pg_sys::PushActiveSnapshot(snap);
                }
            }
        }

        // 2. STRICT QUOTING (Fixes "public.table" -> "\"public\".\"table\"")
        let quoted_table = if raw_table_name.contains('.') {
            raw_table_name.split('.')
                .map(|part| format!("\"{}\"", part))
                .collect::<Vec<_>>()
                .join(".")
        } else {
            format!("\"{}\"", raw_table_name)
        };

        // 3. TRY COUNT (But Ignore Failure)
        let total_rows = Spi::connect(|client| {
            client.select(&format!("SELECT count(*) FROM {}", quoted_table), None, &[])
                .and_then(|t| t.get_one::<i64>())
                .unwrap_or(Some(0))
                .unwrap_or(0)
        });

        // 4. CALCULATE OFFSET (Fallback to 0 if count failed)
        let offset = if total_rows > num_samples as i64 {
             Spi::connect(|client| {
                let max_off = total_rows - num_samples as i64;
                client.select(&format!("SELECT (random() * {})::bigint", max_off), None, &[])
                    .and_then(|t| t.get_one::<i64>()).unwrap_or(Some(0)).unwrap_or(0)
            })
        } else {
            // This is the fix for "0 rows found": We ignore the count and read anyway.
            info!("⚠️ Count was 0/Null. Defaulting to OFFSET 0 to force-read data.");
            0
        };

        info!("🎯 [VectorChord] Target: {} | Offset: {} | Attempting to read {} vectors", quoted_table, offset, num_samples);

        // 5. EXECUTE BLIND LOAD
        let (cached_vectors, dims) = Spi::connect(|client| {
            let mut vecs = Vec::new();
            let mut detected_dims = 0;

            // Debug Log: Prove we are connected to the right DB
            let db = client.select("SELECT current_database()", None, &[])
                .and_then(|t| t.get_one::<String>())
                .unwrap_or(Some("?".to_string()))
                .unwrap_or("?".to_string());

            if vecs.is_empty() {
                info!("🌍 Internal DB Context: [{}]", db);
            }

            // The Query: Sequential Block Read
            let query = format!(
                "SELECT \"{}\" FROM {} OFFSET {} LIMIT {}",
                column_name, quoted_table, offset, num_samples
            );

            let table = client.select(&query, None, &[]).expect("Select Failed");

            for row in table {
                let datum = row.get_datum_by_ordinal(1).expect("Col missing").value::<Datum>();
                if let Ok(Some(d)) = datum {
                    let byte_slice = unsafe {
                        pgrx::varlena_to_byte_slice(d.cast_mut_ptr::<varlena>())
                    };
                    let (vals, d_dims) = vector_type::decode_pgvector_vector(byte_slice);
                    if detected_dims == 0 { detected_dims = d_dims; }
                    vecs.extend(vals);
                }
            }
            Ok::<(Vec<f32>, u32), pgrx::spi::Error>((vecs, detected_dims))
        }).expect("SPI Load Failed");

        // 6. CLEANUP SNAPSHOT
        unsafe {
            if pg_sys::ActiveSnapshotSet() {
                pg_sys::PopActiveSnapshot();
            }
        }

        let safe_dims = if dims == 0 { 1 } else { dims };
        let count_loaded = cached_vectors.len() / (safe_dims as usize);

        if count_loaded == 0 {
             pgrx::error!("FATAL: Still read 0 rows. Confirm 'VACUUM ANALYZE' was run and table '{}' is in DB.", quoted_table);
        }

        info!("✅ [VectorChord] Success: Loaded {} vectors in {:.2?}", count_loaded, start_time.elapsed());

        VectorReadBatcher {
            num_samples,
            num_samples_per_batch,
            min_samples_per_batch,
            vectors_read: 0,
            cached_vectors,
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