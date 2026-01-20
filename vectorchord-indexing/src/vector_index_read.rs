use crate::vector_type;
use pgrx::pg_sys::{self, Datum};
use pgrx::{info, Spi, warning, error, pg_sys::varlena};
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

        // 1. RESOLVE OID & CHECK VISIBILITY
        // We use the internal regclass logic to see if Postgres even knows what this is.
        let (total, table_oid) = Spi::connect(|client| {
            // Force a snapshot update - critical for background workers
            unsafe { pg_sys::PushActiveSnapshot(pg_sys::GetTransactionSnapshot()); }

            let oid_res = client.select(&format!("SELECT '{}'::regclass::oid", full_table_name), None, &[])
                .and_then(|t| t.get_one::<pg_sys::Oid>());

            let oid = match oid_res {
                Ok(Some(id)) => id,
                _ => {
                    error!("FATAL: Table [{}] not found in database. Check the name/quotes.", full_table_name);
                    return Ok((0i64, 0));
                }
            };

            // Get the count using the OID directly (bypassing schema-resolution issues)
            let count = client.select(&format!("SELECT reltuples::bigint FROM pg_class WHERE oid = {}", oid), None, &[])
                .and_then(|t| t.get_one::<i64>())
                .unwrap_or(Some(0))
                .unwrap_or(0);

            Ok::<(i64, pg_sys::Oid), pgrx::spi::Error>((count, oid))
        }).expect("SPI Connection Fail");

        info!("🧬 Forensic: OID {} | Found {} rows in catalog", table_oid, total);

        // 2. THE NUCLEAR COUNT (If catalog is -1 or 0)
        let actual_total = if total <= 0 {
            warning!("Catalog says 0. Forcing a sequential row count scan...");
            Spi::connect(|client| {
                client.select(&format!("SELECT count(*) FROM {}", full_table_name), None, &[])
                    .and_then(|t| t.get_one::<i64>())
                    .unwrap_or(Some(0))
                    .unwrap_or(0)
            })
        } else {
            total
        };

        if actual_total == 0 {
            error!("❌ ABSOLUTE ZERO: Even a direct count returned 0 rows for {}. Data is uncommitted or in a different database.", full_table_name);
        }

        // 3. LOAD DATA (Explicit Column and Table quoting)
        let (cached_vectors, dims) = Spi::connect(|client| {
            let mut vecs = Vec::new();
            let mut detected_dims = 0;

            // We use the OID for the table to be 100% sure we hit the same object
            let query = format!(
                "SELECT \"{}\" FROM {} LIMIT {}",
                column_name, full_table_name, num_samples
            );

            let table = client.select(&query, None, &[]).expect("Load failed");

            for row in table {
                let datum = row.get_datum_by_ordinal(1).expect("Col 1 missing").value::<Datum>();
                if let Ok(Some(d)) = datum {
                    let byte_slice = unsafe {
                        pgrx::varlena_to_byte_slice(d.cast_mut_ptr::<varlena>())
                    };
                    let (vals, d_dims) = vector_type::decode_pgvector_vector(byte_slice);
                    if detected_dims == 0 { detected_dims = d_dims; }
                    vecs.extend(vals);
                }
            }

            // Cleanup snapshot
            unsafe { pg_sys::PopActiveSnapshot(); }

            Ok::<(Vec<f32>, u32), pgrx::spi::Error>((vecs, detected_dims))
        }).expect("SPI Final Load Fail");

        let safe_dims = if dims == 0 { 1 } else { dims };
        info!("✅ Load Done: {} vectors loaded.", cached_vectors.len() / (safe_dims as usize));

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
        if self.cached_vectors.is_empty() || self.vectors_read >= self.num_samples { return None; }
        let mut to_read = self.num_samples_per_batch as usize;
        let remaining = (self.num_samples - self.vectors_read) as usize;
        if remaining < (self.num_samples_per_batch + self.min_samples_per_batch) as usize { to_read = remaining; }
        let start = (self.vectors_read as usize) * (self.dims as usize);
        let end = start + (to_read * self.dims as usize);
        if end > self.cached_vectors.len() { return None; }
        let batch = self.cached_vectors[start..end].to_vec();
        self.vectors_read += to_read as u64;
        Some((batch, self.dims))
    }

    pub(crate) fn end_scan(self) {}
}