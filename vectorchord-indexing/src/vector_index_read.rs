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
        table_name: String,
        column_name: String,
        num_samples: u64,
        num_samples_per_batch: u64,
        min_samples_per_batch: u64,
    ) -> Self {
        let start_time = Instant::now();

        // --- THE SNAPSHOT SYNC ---
        // We ensure the internal SPI can see the 100M rows even if the catalog is stale.
        unsafe {
            if !pg_sys::ActiveSnapshotSet() {
                pg_sys::PushActiveSnapshot(pg_sys::GetTransactionSnapshot());
            }
        }

        // 1. GET TOTAL ROWS (The Jump Range)
        let total: i64 = Spi::connect(|client| {
            client.select(&format!("SELECT count(*) FROM {}", table_name), None, &[])
                .and_then(|t| t.get_one::<i64>())
                .unwrap_or(Some(0))
                .unwrap_or(0)
        });

        // 2. CALCULATE JUMP OFFSET
        let offset: i64 = if total > num_samples as i64 {
            Spi::connect(|client| {
                let max_off = total - num_samples as i64;
                client.select(&format!("SELECT (random() * {})::bigint", max_off), None, &[])
                    .and_then(|t| t.get_one::<i64>())
                    .unwrap_or(Some(0))
                    .unwrap_or(0)
            })
        } else {
            0
        };

        info!("🎯 [VectorChord] Found {} rows. Jumping to offset: {}", total, offset);

        // 3. CONTINUOUS BLOCK READ
        let (cached_vectors, dims) = Spi::connect(|client| {
            let mut vecs = Vec::new();
            let mut detected_dims = 0;

            let query = format!(
                "SELECT \"{}\" FROM {} OFFSET {} LIMIT {}",
                column_name, table_name, offset, num_samples
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
        }).expect("SPI Connection Error");

        // 4. CLEANUP
        unsafe {
            if pg_sys::ActiveSnapshotSet() {
                pg_sys::PopActiveSnapshot();
            }
        }

        let safe_dims = if dims == 0 { 1 } else { dims };
        let count_loaded = (cached_vectors.len() / (safe_dims as usize)) as u64;

        info!("✅ [VectorChord] Data Block Loaded: {} vectors in {:.2?}", count_loaded, start_time.elapsed());

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
        if self.cached_vectors.is_empty() || self.vectors_read >= self.num_samples {
            return None;
        }

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