use crate::vector_type;
use pgrx::{info, Spi};
use pgrx::pg_sys::Datum;
use rand::Rng;
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
    ) -> (Self, Vec<f32>) {
        let start_time = Instant::now();

        let quoted_table = if table_name.contains('.') {
            table_name.split('.')
                .map(|part| format!("\"{}\"", part))
                .collect::<Vec<_>>()
                .join(".")
        } else {
            format!("\"{}\"", table_name)
        };

        // Count total rows in the table
        let total_rows: u64 = Spi::connect(|client| {
            let count_query = format!("SELECT COUNT(1) FROM {}", quoted_table);
            let result = client.select(&count_query, None, &[])?;
            let count: i64 = result.first().get_datum_by_ordinal(1)?.unwrap_or(Datum).into();
            Ok::<u64, pgrx::spi::Error>(count as u64)
        }).expect("FATAL: Failed to count rows");

        if total_rows == 0 {
            panic!("FATAL: Table '{}' is empty.", quoted_table);
        }

        // Generate a random offset
        let random_offset: u64 = rand::rng().random_range(0..total_rows.saturating_sub(num_samples));

        info!("🚀 [SQL Load] Reading {} samples from '{}' with random offset {}", num_samples, quoted_table, random_offset);

        // Fetch the sample dataset
        let (vecs, detected_dims) = Spi::connect(|client| {
            let query = format!(
                "SELECT \"{}\" FROM {} LIMIT {} OFFSET {}",
                column_name, quoted_table, num_samples, random_offset
            );

            let mut internal_vecs = Vec::new();
            let mut internal_dims = 0;

            let tup_table = client.select(&query, None, &[])?;

            for row in tup_table {
                if let Some(datum) = row.get_datum_by_ordinal(1)? {
                    let byte_slice = unsafe {
                        pgrx::varlena_to_byte_slice(datum.cast_mut_ptr::<pgrx::pg_sys::varlena>())
                    };

                    let (vals, d_dims) = vector_type::decode_pgvector_vector(byte_slice);

                    if internal_dims == 0 { internal_dims = d_dims; }
                    internal_vecs.extend(vals);
                }
            }

            Ok::<(Vec<f32>, u32), pgrx::spi::Error>((internal_vecs, internal_dims))
        }).expect("FATAL: SQL Query Failed");

        let safe_dims = if detected_dims == 0 { 1 } else { detected_dims };
        let count_loaded = vecs.len() / (safe_dims as usize);

        info!("✅ [SQL Load] Loaded {} vectors in {:.2?}", count_loaded, start_time.elapsed());

        if count_loaded == 0 {
            panic!("FATAL: Query returned 0 rows. Is the table '{}' empty?", quoted_table);
        }

        let batcher = VectorReadBatcher {
            num_samples,
            num_samples_per_batch,
            min_samples_per_batch,
            vectors_read: 0,
            cached_vectors: vecs.clone(),
            dims: safe_dims,
        };

        (batcher, vecs)
    }

    pub(crate) fn next_batch(&mut self) -> Option<(Vec<f32>, u32)> {
        let start_time = Instant::now();

        // Calculate the remaining samples to read
        let remaining_samples = self.num_samples.saturating_sub(self.vectors_read);
        if remaining_samples == 0 {
            return None; // No more samples to read
        }

        // Determine the batch size
        let mut samples_to_read = self.num_samples_per_batch.min(remaining_samples);
        let size_next_batch = remaining_samples.saturating_sub(self.num_samples_per_batch);
        if size_next_batch < self.min_samples_per_batch {
            samples_to_read += size_next_batch;
        }

        // Generate a random offset within bounds
        let max_offset = self.num_samples.saturating_sub(samples_to_read);
        let random_offset = rand::rng().random_range(0..=max_offset);

        info!(
            "({vectors_read}/{num_samples}) Reading next batch of {samples_to_read} vectors with random offset {random_offset}...",
            vectors_read = self.vectors_read,
            num_samples = self.num_samples,
            samples_to_read = samples_to_read,
            random_offset = random_offset
        );

        // Fetch the batch using SQL with LIMIT and OFFSET
        let query = format!(
            "SELECT \"{}\" FROM {} LIMIT {} OFFSET {}",
            self.cached_vectors, self.dims, samples_to_read, random_offset
        );

        let (all_vectors, dims) = Spi::connect(|client| {
            let mut internal_vecs = Vec::new();
            let mut internal_dims = 0;

            let tup_table = client.select(&query, None, &[])?;

            for row in tup_table {
                if let Some(datum) = row.get_datum_by_ordinal(1)? {
                    let byte_slice = unsafe {
                        pgrx::varlena_to_byte_slice(datum.cast_mut_ptr::<pgrx::pg_sys::varlena>())
                    };

                    let (vector_values, vector_dims) = vector_type::decode_pgvector_vector(byte_slice);
                    internal_vecs.extend_from_slice(&vector_values);
                    internal_dims = vector_dims;
                }
            }

            Ok::<(Vec<f32>, u32), pgrx::spi::Error>((internal_vecs, internal_dims))
        }).expect("FATAL: SQL Query Failed");

        self.vectors_read += samples_to_read;

        info!(
            "Read {} vectors in: {:.2?}",
            all_vectors.len(),
            start_time.elapsed()
        );

        match all_vectors.is_empty() {
            true => None,
            false => Some((all_vectors, dims)),
        }
    }
}