use crate::vector_type;
use pgrx::pg_sys::Datum;
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
        full_table_name: String,
        column_name: String,
        num_samples: u64,
        num_samples_per_batch: u64,
        min_samples_per_batch: u64,
    ) -> Self {
        let start_time = Instant::now();
        info!("Table name{}." , full_table_name);
        // 1. STRIP SCHEMA: Extract just the table name if "public." or "schema." is present
        let table_name = if let Some(pos) = full_table_name.rfind('.') {
            &full_table_name[pos + 1..]
        } else {
            &full_table_name
        };

        info!("🎯 [PHASE 1] Target: {} | Column: {}  ", table_name, column_name);

        // 2. GET COUNT: Using the clean table name
        let total: i64 = Spi::connect(|client| {
            client.select(&format!("SELECT COUNT(*) FROM {}", table_name), None, &[])
                .expect("Failed to execute COUNT query")
                .get_one::<i64>()
                .unwrap_or(Some(0))
                .unwrap_or(0)
        });

        // 3. CALCULATE RANDOM OFFSET
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

        info!("📊 Statistics: {} rows found | Random Offset: {}  ", total, offset);

        // 4. DATA LOAD: Fetch vectors into memory
        let (cached_vectors, dims) = Spi::connect(|client| {
            let mut vecs = Vec::new();
            let mut detected_dims = 0;

            let query = format!(
                "SELECT {} FROM {} OFFSET {} LIMIT {}",
                column_name, table_name, offset, num_samples
            );

            let table = client.select(&query, None, &[]).expect("Data load query failed");

            for row in table {
                let datum = row.get_datum_by_ordinal(1).expect("Col 1 missing").value::<Datum>();

                if let Ok(Some(d)) = datum {
                    let byte_slice = unsafe {
                        pgrx::varlena_to_byte_slice(d.cast_mut_ptr::<pgrx::pg_sys::varlena>())
                    };
                    let (vals, d_dims) = vector_type::decode_pgvector_vector(byte_slice);

                    if detected_dims == 0 { detected_dims = d_dims; }
                    vecs.extend(vals);
                }
            }
            Ok::<(Vec<f32>, u32), pgrx::spi::Error>((vecs, detected_dims))
        }).expect("SPI Data Connection Failed");

        // 5. DIVISION SHIELD: Ensure we don't crash if loading failed
        let safe_dims = if dims == 0 {
            warning!("⚠️ Warning: Dimensions are 0. Using 1 to prevent crash.");
            1
        } else {
            dims
        };

        let loaded_count = cached_vectors.len() / (safe_dims as usize);
        info!("✅ Success: Loaded {} vectors for training in {:.2?}", loaded_count, start_time.elapsed());

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
        if self.vectors_read >= self.num_samples || self.dims <= 1 || self.cached_vectors.is_empty() {
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

    pub(crate) fn end_scan(self) {
        info!("🏁 Training sample session ended. Cache cleared.");
    }
}