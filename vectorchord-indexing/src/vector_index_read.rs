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

        // 1. RE-ENABLE FULL NAME WITH QUOTING
        // We wrap the table name in quotes to handle schemas correctly: "public"."table_name"
        let quoted_table = if full_table_name.contains('.') {
            full_table_name.split('.')
                .map(|s| format!("\"{}\"", s))
                .collect::<Vec<String>>()
                .join(".")
        } else {
            format!("\"{}\"", full_table_name)
        };

        // 2. LOG THE ENVIRONMENT
        Spi::connect(|client| {
            let db: String = client.select("SELECT current_database()", None, &[])
                .and_then(|t| t.get_one()).unwrap_or(Some("?".into())).unwrap();
            let schema: String = client.select("SELECT current_schema()", None, &[])
                .and_then(|t| t.get_one()).unwrap_or(Some("?".into())).unwrap();
            info!("🌍 ENV: Database=[{}] | Active Schema=[{}] | Targeting=[{}]", db, schema, quoted_table);
            Ok::<(), pgrx::spi::Error>(())
        }).ok();

        // 3. DIRECT COUNT
        let total: i64 = Spi::connect(|client| {
            client.select(&format!("SELECT COUNT(*) FROM {}", quoted_table), None, &[])
                .expect("Failed to execute COUNT query")
                .get_one::<i64>()
                .unwrap_or(Some(0))
                .unwrap_or(0)
        });

        // 4. RANDOM OFFSET
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

        info!("📊 Stats: {} rows | Offset: {} | Column: {}", total, offset, column_name);

        // 5. DATA LOAD
        let (cached_vectors, dims) = Spi::connect(|client| {
            let mut vecs = Vec::new();
            let mut detected_dims = 0;

            // Note the use of "{}" for the column name - usually columns don't need quotes
            // unless they have spaces, but let's keep it simple for now.
            let query = format!(
                "SELECT \"{}\" FROM {} OFFSET {} LIMIT {}",
                column_name, quoted_table, offset, num_samples
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

        let safe_dims = if dims == 0 { 1 } else { dims };
        let loaded_count = cached_vectors.len() / (safe_dims as usize);

        info!("✅ Phase 1 Done: Loaded {} vectors in {:.2?}", loaded_count, start_time.elapsed());

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

        if end > self.cached_vectors.len() { return None; }

        let batch = self.cached_vectors[start..end].to_vec();
        self.vectors_read += to_read as u64;
        Some((batch, self.dims))
    }

    pub(crate) fn end_scan(self) {}
}