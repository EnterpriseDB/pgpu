use crate::vector_type;
use pgrx::pg_sys::{format_type_be, SysScanDesc};
use pgrx::{debug1, heap_getattr_raw, info, pg_sys, warning, PgRelation, Spi};
use std::ffi::CStr;
use std::time::Instant;

pub struct VectorReadBatcher {
    table_name: String,
    column_name: String,
    num_tuples_in_table: Option<u64>,
    num_samples: u64,
    num_samples_per_batch: u64,
    min_samples_per_batch: u64,
    vectors_read: u64,
    //table_scan: Option<SysScanDesc>,
    //pg_rel: Option<PgRelation>,
    //col_num: Option<usize>,
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

        // We use Bernoulli sampling to get a random distribution across the whole table.
        // 2.5% is a safe bet for a 1M sample from 40M rows.
        let query = format!(
            "SELECT {column_name} FROM {table_name} TABLESAMPLE BERNOULLI (2.5) LIMIT {num_samples}"
        );

        info!("🚀 [PHASE 1] Initializing Random Sampler...");
        info!("🔍 Executing SQL: {}  ", query);

        let (cached_vectors, dims) = Spi::connect(|client| {
            let mut all_vecs = Vec::new();
            let mut detected_dims = 0;
            let mut row_count = 0;

            let tuple_table = client.select(&query, None, None).expect("Failed to fetch samples");

            let decode_start = Instant::now();
            for row in tuple_table {
                let datum = row.get_datum_by_ordinal(1).expect("Column not found");
                if let Some(raw_ptr) = datum {
                    let byte_slice = unsafe { pgrx::varlena_to_byte_slice(raw_ptr.cast_mut_ptr()) };
                    let (vec_vals, v_dims) = vector_type::decode_pgvector_vector(byte_slice);
                    all_vecs.extend(vec_vals);
                    detected_dims = v_dims;
                    row_count += 1;
                }

                // Log progress every 250k rows so you know it hasn't crashed
                if row_count % 250_000 == 0 {
                    debug1!("   ... decoded {}/{} vectors", row_count, num_samples);
                }
            }

            info!("✅ Decoding complete. Processed {} rows in {:?}  ", row_count, decode_start.elapsed());
            Ok::<(Vec<f32>, u32), pgrx::spi::Error>((all_vecs, detected_dims))
        }).expect("SPI Error");

        let total_init_time = start_time.elapsed();
        info!(
            "📊 Sampler Ready: Loaded {} vectors ({} dims) in {:.2?}",
            cached_vectors.len() / (dims as usize),
            dims,
            total_init_time
        );

        VectorReadBatcher {
            table_name,
            column_name,
            num_samples,
            num_samples_per_batch,
            min_samples_per_batch,
            vectors_read: 0,
            cached_vectors,
            dims,
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

        // Detailed logging for the batching process
        debug1!(
            "📦 Batching: Rows {}-{} of {}  ",
            self.vectors_read,
            self.vectors_read + samples_to_read as u64,
            self.num_samples
        );

        let start_idx = self.vectors_read as usize * self.dims as usize;
        let end_idx = (self.vectors_read as usize + samples_to_read) * self.dims as usize;

        let batch = self.cached_vectors[start_idx..end_idx].to_vec();
        self.vectors_read += samples_to_read as u64;

        Some((batch, self.dims))
    }

    pub(crate) fn num_tuples(&mut self) -> u64 {
        Spi::get_one::<i64>(&format!("SELECT COUNT(1) FROM {}", self.table_name))
            .expect("SQL error")
            .unwrap_or(0) as u64
    }

    pub(crate) fn end_scan(self) {
        info!("🏁 Scan session ended. Memory released.");
    }
}
