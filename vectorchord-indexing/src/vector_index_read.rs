use crate::vector_type;
use pgrx::{info, Spi};
use pgrx::pg_sys::Datum; // Explicit import for Datum
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

        // 1. Quoting Logic (Crucial for "public.table" vs "public"."table")
        // If the name has a dot, we assume it's schema.table and quote parts.
        // Otherwise we quote the whole thing.
        let quoted_table = if table_name.contains('.') {
            table_name.split('.')
                .map(|part| format!("\"{}\"", part))
                .collect::<Vec<_>>()
                .join(".")
        } else {
            format!("\"{}\"", table_name)
        };

        info!("🚀 [SQL Load] Reading from: {}", quoted_table);

        // 2. Execute Standard SQL Query
        // We let Postgres handle ALL the complexity (TOAST, Visibility, Snapshots)
        let (vecs, detected_dims) = Spi::connect(|client| {
            let query = format!(
                "SELECT \"{}\" FROM {} LIMIT {}",
                column_name, quoted_table, num_samples
            );

            let mut internal_vecs = Vec::new();
            let mut internal_dims = 0;

            // Execute the query
            let tup_table = client.select(&query, None, None)?;

            for row in tup_table {
                // Get the vector column (Ordinal 1)
                // We use get_datum_by_ordinal which handles NULLs safely
                let datum_opt = row.get_datum_by_ordinal(1)?;

                if let Some(datum) = datum_opt {
                    // Convert pgrx::Datum to raw byte slice safely
                    // pgrx handles the detoasting automatically here via the helper
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

        // 3. Validation
        let safe_dims = if detected_dims == 0 { 1 } else { detected_dims };
        let count_loaded = vecs.len() / (safe_dims as usize);

        info!("✅ [SQL Load] Loaded {} vectors in {:.2?}", count_loaded, start_time.elapsed());

        if count_loaded == 0 {
            panic!("FATAL: Query returned 0 rows. Is the table '{}' empty?", quoted_table);
        }

        VectorReadBatcher {
            num_samples,
            num_samples_per_batch,
            min_samples_per_batch,
            vectors_read: 0,
            cached_vectors: vecs,
            dims: safe_dims,
        }
    }

    pub fn next_batch(&mut self) -> Option<(Vec<f32>, u32)> {
        if self.cached_vectors.is_empty() || self.vectors_read >= self.num_samples { return None; }

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