use crate::vector_type;
use pgrx::{debug1, info, Spi};
use std::time::Instant;
use pgrx::pg_sys::Datum;

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

        // 1. Get total rows AND a random offset in one SQL call
        let (total_rows, random_offset) = Spi::connect(|client| {
            let count_query = format!("SELECT COUNT(*) FROM {}", table_name);
            let total = client.select(&count_query, None, &[])
                .and_then(|t| t.get_one::<i64>())
                .unwrap_or(Some(0))
                .unwrap_or(0);

            let mut offset = 0i64;
            if total > num_samples as i64 {
                let max_off = total - num_samples as i64;
                // Use Postgres SQL to generate the random offset
                let off_query = format!("SELECT (random() * {})::bigint", max_off);
                offset = client.select(&off_query, None, &[])
                    .and_then(|t| t.get_one::<i64>())
                    .unwrap_or(Some(0))
                    .unwrap_or(0);
            }
            Ok::<(i64, i64), pgrx::spi::Error>((total, offset))
        }).expect("Failed to calculate offset");

        // 2. Execute the Block-Offset Query
        let query = format!(
            "SELECT {column_name} FROM {table_name} OFFSET {random_offset} LIMIT {num_samples}"
        );

        info!("🚀 [PHASE 1] Random Sampler Initialized");
        info!("🎲 Random Start: row {} | Total: {} | Samples: {}  ", random_offset, total_rows, num_samples);

        let (cached_vectors, dims) = Spi::connect(|client| {
            let mut all_vecs = Vec::new();
            let mut detected_dims = 0;
            let mut row_count = 0;

            let tuple_table = client.select(&query, None, &[]).expect("Failed to fetch samples");

            for row in tuple_table {
                let entry = row.get_datum_by_ordinal(1).expect("Column not found");
                if let Ok(Some(raw_datum)) = entry.value::<Datum>() {
                    let byte_slice = unsafe { pgrx::varlena_to_byte_slice(raw_datum.cast_mut_ptr()) };
                    let (vec_vals, v_dims) = vector_type::decode_pgvector_vector(byte_slice);

                    if detected_dims == 0 { detected_dims = v_dims; }
                    all_vecs.extend(vec_vals);
                    row_count += 1;
                }
            }

            if row_count == 0 && total_rows > 0 {
                pgrx::error!("SQL returned 0 rows. Verify permissions for table: {}", table_name);
            }

            Ok::<(Vec<f32>, u32), pgrx::spi::Error>((all_vecs, detected_dims))
        }).expect("SPI Error");

        let safe_dims = if dims == 0 { 768 } else { dims };

        info!(
            "✅ Loaded {} vectors in {:.2?}",
            cached_vectors.len() / (safe_dims as usize),
            start_time.elapsed()
        );

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
        if self.vectors_read >= self.num_samples {
            return None;
        }

        let mut samples_to_read = self.num_samples_per_batch as usize;
        let remaining = (self.num_samples - self.vectors_read) as usize;

        if remaining < (self.num_samples_per_batch + self.min_samples_per_batch) as usize {
            samples_to_read = remaining;
        }

        debug1!(
            "📦 Feeding Batch: {}-{} of {}  ",
            self.vectors_read,
            self.vectors_read + (samples_to_read as u64),
            self.num_samples
        );

        let start_idx = (self.vectors_read as usize) * (self.dims as usize);
        let end_idx = (self.vectors_read as usize + samples_to_read) * (self.dims as usize);

        if end_idx > self.cached_vectors.len() {
            return None;
        }

        let batch = self.cached_vectors[start_idx..end_idx].to_vec();
        self.vectors_read += samples_to_read as u64;

        Some((batch, self.dims))
    }

    pub(crate) fn end_scan(self) {
        info!("🏁 Training sample scan complete.");
    }
}