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

        // 1. Get the total row count to determine the "skip" range
        let total_rows: i64 = Spi::get_one(&format!("SELECT COUNT(1) FROM {}", table_name))
            .expect("SQL error fetching table size")
            .unwrap_or(0);

        // 2. Calculate a random starting point (The VectorChord Strategy)
        // We ensure that (offset + num_samples) does not exceed the table size.
        let max_offset = if total_rows > num_samples as i64 {
            total_rows - num_samples as i64
        } else {
            0
        };

        // Generate a random offset using Postgres's internal random() function
        let random_offset: i64 = if max_offset > 0 {
            unsafe { (pgrx::pg_sys::random() % max_offset).abs() }
        } else {
            0
        };

        // 3. Use OFFSET/LIMIT for a guaranteed sequential block read
        let query = format!(
            "SELECT {column_name} FROM {table_name} OFFSET {random_offset} LIMIT {num_samples}"
        );

        info!("🚀 [PHASE 1] Initializing Block-Offset Sampler (VectorChord Approach)");
        info!("🎲 Choosing random start at row: {} of {}  ", random_offset, total_rows);

        let (cached_vectors, dims) = Spi::connect(|client| {
            let mut all_vecs = Vec::new();
            let mut detected_dims = 0;
            let mut row_count = 0;

            let tuple_table = client.select(&query, None, &[]).expect("Failed to fetch samples");

            let decode_start = Instant::now();
            for row in tuple_table {
                let entry = row.get_datum_by_ordinal(1).expect("Column not found");

                // Using the Ok(Some(..)) pattern for Result<Option<Datum>>
                if let Ok(Some(raw_datum)) = entry.value::<Datum>() {
                    let byte_slice = unsafe { pgrx::varlena_to_byte_slice(raw_datum.cast_mut_ptr()) };
                    let (vec_vals, v_dims) = vector_type::decode_pgvector_vector(byte_slice);

                    if detected_dims == 0 { detected_dims = v_dims; }
                    all_vecs.extend(vec_vals);
                    row_count += 1;
                }
            }

            if row_count == 0 && total_rows > 0 {
                pgrx::error!("SQL returned 0 rows despite table having data. Check permissions for {}", table_name);
            }

            info!("✅ Decoding complete. Loaded {} vectors in {:?}", row_count, decode_start.elapsed());
            Ok::<(Vec<f32>, u32), pgrx::spi::Error>((all_vecs, detected_dims))
        }).expect("SPI Error");

        // Guard against divide-by-zero in logs
        let safe_dims = if dims == 0 { 768 } else { dims };

        info!(
            "📊 Sampler Ready: Loaded {} total vectors in {:.2?}  ",
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

        // Safety slice check
        if end_idx > self.cached_vectors.len() {
            return None;
        }

        let batch = self.cached_vectors[start_idx..end_idx].to_vec();
        self.vectors_read += samples_to_read as u64;

        Some((batch, self.dims))
    }

    pub(crate) fn end_scan(self) {
        info!("🏁 Training sample session ended. Memory released.");
    }
}