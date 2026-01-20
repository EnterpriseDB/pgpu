use crate::vector_type;
use pgrx::{info, Spi};
use pgrx::pg_sys::Datum;
use std::time::Instant;
use rand::Rng;

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
            let result = client.select(&count_query, None, None)?;
            let count: i64 = result.first().get_datum_by_ordinal(1)?.unwrap_or(0);
            Ok::<u64, pgrx::spi::Error>(count as u64)
        }).expect("FATAL: Failed to count rows");

        if total_rows == 0 {
            panic!("FATAL: Table '{}' is empty.", quoted_table);
        }

        // Generate a random offset
        let random_offset: u64 = rand::thread_rng().gen_range(0..total_rows.saturating_sub(num_samples));

        info!("🚀 [SQL Load] Reading {} samples from '{}' with random offset {}", num_samples, quoted_table, random_offset);

        // Fetch the sample dataset
        let (vecs, detected_dims) = Spi::connect(|client| {
            let query = format!(
                "SELECT \"{}\" FROM {} LIMIT {} OFFSET {}",
                column_name, quoted_table, num_samples, random_offset
            );

            let mut internal_vecs = Vec::new();
            let mut internal_dims = 0;

            let tup_table = client.select(&query, None, None)?;

            for row in tup_table {
                let datum_opt = row.get_datum_by_ordinal(1)?;

                if let Some(datum) = datum_opt {
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
}