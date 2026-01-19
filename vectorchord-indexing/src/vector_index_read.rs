use crate::vector_type;
use pgrx::pg_sys::Datum;
use pgrx::{info, warning, Spi};
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

        // --- MULTI-STRATEGY COUNT ---
        let total = Spi::connect(|client| {
            // Strategy 1: Explicit COUNT(*)
            if let Ok(Some(count)) = client.select(&format!("SELECT COUNT(*) FROM {table_name}"), None, &[])
                .and_then(|t| t.get_one::<i64>()) {
                if count > 0 { return Ok(count); }
            }

            // Strategy 2: Fast Catalog Lookup (Estimate)
            // Useful if the table is locked or COUNT is failing for session reasons
            if let Ok(Some(estimate)) = client.select(
                &format!("SELECT reltuples::bigint FROM pg_class WHERE oid = '{table_name}'::regclass"),
                None, &[]
            ).and_then(|t| t.get_one::<i64>()) {
                if estimate > 0 {
                    warning!("⚠️ Strategy 1 failed. Using Catalog Estimate: {}", estimate);
                    return Ok(estimate);
                }
            }

            // Strategy 3: Information Schema
            if let Ok(Some(info_count)) = client.select(
                &format!("SELECT n_live_tup FROM pg_stat_user_tables WHERE relname = '{table_name}'"),
                None, &[]
            ).and_then(|t| t.get_one::<i64>()) {
                if info_count > 0 { return Ok(info_count); }
            }

            // Final Fallback: If we can't find a count, assume it's large enough for our sample
            warning!("❗ All count strategies failed for {table_name}. Defaulting to sample size.");
            Ok(num_samples as i64)
        }).expect("SPI Connection Failed during count phase");

        // Calculate offset safely
        let offset = if total > num_samples as i64 {
            Spi::connect(|client| {
                client.select(&format!("SELECT (random() * {})::bigint", total - num_samples as i64), None, &[])
                    .and_then(|t| t.get_one::<i64>())
            }).unwrap_or(Some(0)).unwrap_or(0)
        } else {
            0
        };

        info!("📊 Dataset Detected: {} rows | Offset: {} | Target: {}", total, offset, num_samples);

        // --- DATA LOAD ---
        let (cached_vectors, dims) = Spi::connect(|client| {
            let mut vecs = Vec::new();
            let mut detected_dims = 0;

            // Note: If the table is truly empty, this simply returns an empty iterator (no panic)
            let query = format!("SELECT {column_name} FROM {table_name} OFFSET {offset} LIMIT {num_samples}");
            let table = client.select(&query, None, &[]).expect("Data load query failed");

            for row in table {
                let datum = row.get_datum_by_ordinal(1).expect("Column 1 missing").value::<Datum>();

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

        if loaded_count == 0 {
            warning!("🛑 LOADED 0 VECTORS. Check if table {table_name} is in the same database.");
        } else {
            info!("✅ Success: Loaded {} vectors in {:.2?}", loaded_count, start_time.elapsed());
        }

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
        if self.vectors_read >= self.num_samples || self.dims <= 1 { return None; }
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