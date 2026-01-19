use crate::vector_type;
use pgrx::pg_sys::Datum; // Fixed import path
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
        table_name: String,
        column_name: String,
        num_samples: u64,
        num_samples_per_batch: u64,
        min_samples_per_batch: u64,
    ) -> Self {
        let start_time = Instant::now();

        // 1. Get Count and Offset with proper pgrx arguments (&[])
        let (total, offset) = Spi::connect(|client| {
            let total = client
                .select(&format!("SELECT count(*) FROM {}", table_name), None, &[])
                .expect("SQL Count Failed")
                .get_one::<i64>()
                .expect("Result was not i64")
                .unwrap_or(0);

            let mut offset = 0i64;
            if total > num_samples as i64 {
                let max_off = total - num_samples as i64;
                offset = client
                    .select(
                        &format!("SELECT (random() * {})::bigint", max_off),
                        None,
                        &[],
                    )
                    .expect("SQL Offset Failed")
                    .get_one::<i64>()
                    .expect("Result was not i64")
                    .unwrap_or(0);
            }
            Ok::<(i64, i64), pgrx::spi::Error>((total, offset))
        })
        .expect("SPI Connection Failed");

        info!(
            "📊 Table: {} | Total: {} | Sampling: {} from offset {}  ",
            table_name, total, num_samples, offset
        );

        if total == 0 {
            panic!(
                "FATAL: Table '{}' reports 0 rows. This session cannot see the data.",
                table_name
            );
        }

        // 2. Load Data using &[] for parameters
        let (cached_vectors, dims) = Spi::connect(|client| {
            let mut vecs = Vec::with_capacity((num_samples * 768) as usize);
            let mut detected_dims = 0;

            let query = format!(
                "SELECT {} FROM {} OFFSET {} LIMIT {}",
                column_name, table_name, offset, num_samples
            );
            let table = client.select(&query, None, &[]).expect("Data load query failed");

            for row in table {
                let datum = row
                    .get_datum_by_ordinal(1)
                    .expect("Column 1 missing")
                    .value::<Datum>();

                if let Ok(Some(d)) = datum {
                    // Fixed: Added explicit type annotation *mut pgrx::pg_sys::varlena
                    let byte_slice = unsafe {
                        pgrx::varlena_to_byte_slice(d.cast_mut_ptr::<pgrx::pg_sys::varlena>())
                    };
                    let (vals, d_dims) = vector_type::decode_pgvector_vector(byte_slice);
                    if detected_dims == 0 {
                        detected_dims = d_dims;
                    }
                    vecs.extend(vals);
                }
            }
            Ok::<(Vec<f32>, u32), pgrx::spi::Error>((vecs, detected_dims))
        })
        .expect("SPI Data Connection Failed");

        let safe_dims = if dims == 0 { 1 } else { dims };

        info!(
            "✅ Load Complete: {} vectors in {:.2?}",
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
        if self.vectors_read >= self.num_samples || self.dims <= 1 {
            return None;
        }

        let mut to_read = self.num_samples_per_batch as usize;
        let remaining = (self.num_samples - self.vectors_read) as usize;
        if remaining < (self.num_samples_per_batch + self.min_samples_per_batch) as usize {
            to_read = remaining;
        }

        let start = (self.vectors_read as usize) * (self.dims as usize);
        let end = start + (to_read * self.dims as usize);

        if end > self.cached_vectors.len() {
            return None;
        }

        let batch = self.cached_vectors[start..end].to_vec();
        self.vectors_read += to_read as u64;
        Some((batch, self.dims))
    }

    pub(crate) fn end_scan(self) {}
}