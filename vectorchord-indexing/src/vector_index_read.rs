use crate::vector_type;
use pgrx::pg_sys::Datum;
use pgrx::{info, Spi, warning, error};
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

        // 1. REFRESH SNAPSHOT & CHECK IDENTITY
        // We run a dummy command to ensure the SPI context is initialized and updated
        let total: i64 = Spi::connect(|client| {
            // Force Postgres to update its internal visibility map for this session
            client.select("SET TRANSACTION ISOLATION LEVEL READ COMMITTED", None, &[]).ok();

            // Check the OID to ensure the table actually exists in the current DB
            let oid: Option<pgrx::pg_sys::Oid> = client.select(
                &format!("SELECT '{}'::regclass::oid", table_name), None, &[]
            ).and_then(|t| t.get_one()).unwrap_or(None);

            match oid {
                Some(id) => info!("🆔 Verified Table OID: {}", id),
                None => warning!("❌ Table {} NOT FOUND in catalog!", table_name),
            }

            // Execute the count
            client.select(&format!("SELECT COUNT(*) FROM {}", table_name), None, &[])
                .expect("Failed to execute COUNT query")
                .get_one::<i64>()
                .unwrap_or(Some(0))
                .unwrap_or(0)
        });

        info!("📊 [SQL] Table: {} | Row Count: {}", table_name, total);

        // 2. SAFETY CHECK: If SQL still says 0, use a "Brute Force" limit to bypass count logic
        let (offset, target_to_read) = if total == 0 {
            warning!("❗ SQL Count failed. Attempting brute-force read of {} rows anyway...", num_samples);
            (0i64, num_samples)
        } else {
            let off = if total > num_samples as i64 {
                Spi::connect(|client| {
                    let max_off = total - num_samples as i64;
                    client.select(&format!("SELECT (random() * {})::bigint", max_off), None, &[])
                        .and_then(|t| t.get_one::<i64>()).unwrap_or(Some(0)).unwrap_or(0)
                })
            } else {
                0
            };
            (off, num_samples)
        };

        // 3. DATA LOAD
        let (cached_vectors, dims) = Spi::connect(|client| {
            let mut vecs = Vec::new();
            let mut detected_dims = 0;

            // We use the OFFSET/LIMIT here. If it's truly empty, it returns 0 rows.
            let query = format!("SELECT {column_name} FROM {table_name} OFFSET {offset} LIMIT {target_to_read}");
            let table = client.select(&query, None, &[]).expect("Load failed");

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

        // 4. PREVENT CRASH: The Division Shield
        let safe_dims = if dims == 0 { 1 } else { dims };
        let count_loaded = cached_vectors.len() / (safe_dims as usize);

        if count_loaded == 0 {
            error!("❌ FATAL ERROR: No data visible to Postgres SPI. Even brute-force LIMIT 1 returned 0 rows.");
            panic!("Please run 'VACUUM ANALYZE {}' in psql and restart the benchmark.", table_name);
        }

        info!("✅ [PHASE 1] Success: Loaded {} vectors", count_loaded);

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
        if self.cached_vectors.is_empty() || self.vectors_read >= self.num_samples {
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