use crate::vector_type;
use pgrx::pg_sys::{self, Datum, varlena};
use pgrx::{info, Spi};
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

        // 1. FORCE VISIBILITY (The "Dirty Read" Fix)
        // We use GetLatestSnapshot() instead of GetTransactionSnapshot() to bypass
        // stale transaction contexts.
        unsafe {
            pg_sys::SetCurrentStatementStartTimestamp();
            if !pg_sys::ActiveSnapshotSet() {
                let snap = pg_sys::GetLatestSnapshot();
                if !snap.is_null() {
                    pg_sys::PushActiveSnapshot(snap);
                } else {
                    info!("⚠️ Warning: Could not acquire LatestSnapshot.");
                }
            }
        }

        // 2. STRICT QUOTING: Handle "public.table" -> "public"."table"
        let quoted_table = if raw_table_name.contains('.') {
            raw_table_name.split('.')
                .map(|part| format!("\"{}\"", part))
                .collect::<Vec<_>>()
                .join(".")
        } else {
            format!("\"{}\"", raw_table_name)
        };

        // 3. VECTORCHORD: Count -> Offset -> Block Read
        let (total_rows, offset) = Spi::connect(|client| {
            // A. Get Count
            let count = client.select(&format!("SELECT count(*) FROM {}", quoted_table), None, &[])
                .and_then(|t| t.get_one::<i64>())
                .unwrap_or(Some(0))
                .unwrap_or(0);

            // B. Calculate Random Offset
            let off = if count > num_samples as i64 {
                let max_off = count - num_samples as i64;
                client.select(&format!("SELECT (random() * {})::bigint", max_off), None, &[])
                    .and_then(|t| t.get_one::<i64>())
                    .unwrap_or(Some(0))
                    .unwrap_or(0)
            } else {
                0
            };

            Ok::<(i64, i64), pgrx::spi::Error>((count, off))
        }).expect("SPI Setup Failed");

        info!("🎯 [VectorChord] Target: {} | Found: {} rows | Offset: {}", quoted_table, total_rows, offset);

        // 4. LOAD DATA BLOCK
        let (cached_vectors, dims) = Spi::connect(|client| {
            let mut vecs = Vec::new();
            let mut detected_dims = 0;

            // We force a sequential scan of the block using OFFSET/LIMIT
            let query = format!(
                "SELECT \"{}\" FROM {} OFFSET {} LIMIT {}",
                column_name, quoted_table, offset, num_samples
            );

            let table = client.select(&query, None, &[]).expect("Block Read Failed");

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

        // 5. CLEANUP SNAPSHOT
        unsafe {
            if pg_sys::ActiveSnapshotSet() {
                pg_sys::PopActiveSnapshot();
            }
        }

        // 6. VALIDATION (Prevent Divide-by-Zero)
        let safe_dims = if dims == 0 { 1 } else { dims };
        let count_loaded = cached_vectors.len() / (safe_dims as usize);

        if count_loaded == 0 {
             pgrx::error!("FATAL: Still read 0 rows from {}. Check permissions or database connection.", quoted_table);
        }

        info!("✅ [VectorChord] Loaded {} vectors in {:.2?}", count_loaded, start_time.elapsed());

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