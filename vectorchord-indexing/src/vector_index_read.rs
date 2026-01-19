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

        // SQL using the fixed 2.5% rate as requested
        let query = format!(
            "SELECT {column_name} FROM {table_name} TABLESAMPLE BERNOULLI (2.5) LIMIT {num_samples}"
        ); // Still not fully random as we use LIMIT, but better than SeqScan

        info!("🚀 [PHASE 1] Initializing Sampler (Fixed 2.5% Rate)");
        info!("⚡ Executing SQL: {}", query);

        let (cached_vectors, dims) = Spi::connect(|client| {
            let mut all_vecs = Vec::new();
            let mut detected_dims = 0;
            let mut row_count = 0;

            // Fixed: Added Some(&[]) for arguments to match pgrx signature
            let tuple_table = client.select(&query, None, Some(&[])).expect("Failed to fetch samples");

            let decode_start = Instant::now();
            for row in tuple_table {
                // Fixed: get_datum_by_ordinal(1) returns an entry we must extract carefully
                let entry = row.get_datum_by_ordinal(1).expect("Column not found");

                // Fixed: Extract the internal Datum pointer safely
                if let Some(raw_datum) = entry.value::<Datum>() {
                    let byte_slice = unsafe { pgrx::varlena_to_byte_slice(raw_datum.cast_mut_ptr()) };
                    let (vec_vals, v_dims) = vector_type::decode_pgvector_vector(byte_slice);
                    all_vecs.extend(vec_vals);
                    detected_dims = v_dims;
                    row_count += 1;
                }

                if row_count % 250_000 == 0 {
                    info!("   ... decoded {}/{} vectors", row_count, num_samples);
                }
            }

            info!("✅ Decoding complete. Processed {} rows in {:?}", row_count, decode_start.elapsed());
            Ok::<(Vec<f32>, u32), pgrx::spi::Error>((all_vecs, detected_dims))
        }).expect("SPI Error");

        VectorReadBatcher {
            table_name,
            column_name,
            num_tuples_in_table: None,
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

        debug1!(
            "📦 Batching: {}-{} of {}  ",
            self.vectors_read,
            self.vectors_read + samples_to_read as u64,
            self.num_samples
        );

        let start_idx = self.vectors_read as usize * self.dims as usize;
        let end_idx = (self.vectors_read as usize + samples_to_read) * self.dims as usize;

        // Ensure we don't overflow if the cache is smaller than num_samples
        if end_idx > self.cached_vectors.len() {
            return None;
        }

        let batch = self.cached_vectors[start_idx..end_idx].to_vec();
        self.vectors_read += samples_to_read as u64;

        Some((batch, self.dims))
    }

    pub(crate) fn num_tuples(&mut self) -> u64 {
        match self.num_tuples_in_table {
            Some(count) => count,
            None => {
                let count: i64 = Spi::get_one(&format!("SELECT COUNT(1) FROM {}  ", self.table_name))
                    .expect("SQL error")
                    .unwrap_or(0);
                self.num_tuples_in_table = Some(count as u64);
                count as u64
            }
        }
    }

    pub(crate) fn end_scan(self) {
        info!("🏁 Training sample session ended.");
    }
}