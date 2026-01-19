use crate::vector_type;
use pgrx::{debug1, info, Spi};
use std::time::Instant;
use pgrx::pg_sys::Datum;

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

        // --- STEP 1: RUN DEBUGGER ---
        Self::debug_session_context(&table_name);

        // --- STEP 2: CALCULATE OFFSET ---
        let (_total_rows, random_offset) = Spi::connect(|client| {
            let count_sql = format!("SELECT count(*) FROM {}", table_name);
            info!("🛠️ [SQL CHECK] Count query: {}  ", count_sql);

            let total = client.select(&count_sql, None, &[])
                .and_then(|t| t.get_one::<i64>())
                .unwrap_or(Some(0))
                .unwrap_or(0);

            let mut offset = 0i64;
            if total > num_samples as i64 {
                let max_off = total - num_samples as i64;
                let off_sql = format!("SELECT (random() * {})::bigint", max_off);
                info!("🛠️ [SQL CHECK] Offset query: {}  ", off_sql);

                offset = client.select(&off_sql, None, &[])
                    .and_then(|t| t.get_one::<i64>())
                    .unwrap_or(Some(0))
                    .unwrap_or(0);
            }
            Ok::<(i64, i64), pgrx::spi::Error>((total, offset))
        }).expect("SPI Error during pre-scan");

        // --- STEP 3: EXECUTE DATA LOAD ---
        let query = format!(
            "SELECT {column_name} FROM {table_name} OFFSET {random_offset} LIMIT {num_samples}"
        );
        info!("🛠️ [SQL CHECK] Main Data Query: {}  ", query);

        let (cached_vectors, dims) = Spi::connect(|client| {
            let mut all_vecs = Vec::new();
            let mut detected_dims = 0;
            let mut row_count = 0;

            let tuple_table = client.select(&query, None, &[]).expect("Query failed");

            for row in tuple_table {
                let entry = row.get_datum_by_ordinal(1).expect("Column not found");
                if let Ok(Some(raw_datum)) = entry.value::<Datum>() {
                    let byte_slice = unsafe { pgrx::varlena_to_byte_slice(raw_datum.cast_mut_ptr()) };
                    let (vec_vals, v_dims) = vector_type::decode_pgvector_vector(byte_slice);
                    if detected_dims == 0 { detected_dims = v_dims; }
                    all_vecs.extend(vec_vals);
                    row_count += 1;
                }
            }

            if row_count == 0 {
                debug1!("❌ FATAL: 0 vectors returned. Check the 'Main Data Query' above in psql.");
            }

            Ok::<(Vec<f32>, u32), pgrx::spi::Error>((all_vecs, detected_dims))
        }).expect("SPI Error during data load");

        let safe_dims = if dims == 0 { 1 } else { dims };

        info!("✅ Phase 1 Ready: Loaded {} vectors in {:.2?}  ", cached_vectors.len() / (safe_dims as usize), start_time.elapsed());

        VectorReadBatcher {
            num_samples,
            num_samples_per_batch,
            min_samples_per_batch,
            vectors_read: 0,
            cached_vectors,
            dims: safe_dims,
        }
    }

    /// Isolated forensic function to check why Postgres might think a table is empty
    fn debug_session_context(table_name: &str) {
        Spi::connect(|client| {
            let user: String = client.select("SELECT current_user", None, &[]).and_then(|t| t.get_one()).unwrap_or(Some("?".into())).unwrap();
            let db: String = client.select("SELECT current_database()", None, &[]).and_then(|t| t.get_one()).unwrap_or(Some("?".into())).unwrap();

            // Checking the global system catalog for physical presence
            let stats_query = format!("SELECT relpages, reltuples FROM pg_class WHERE oid = '{}'::regclass", table_name);
            let stats = client.select(&stats_query, None, &[]);

            info!("--- [DIAGNOSTIC] ---");
            info!("👤 User: {} | 📂 DB: {}", user, db);

            if let Ok(table) = stats {
                for row in table {
                    // Safe extraction of i32 (pages) and f32 (tuples)
                    let pages: i32 = row.get_datum_by_ordinal(1).unwrap().value().unwrap().unwrap_or(0);
                    let tuples: f32 = row.get_datum_by_ordinal(2).unwrap().value().unwrap().unwrap_or(0.0);
                    info!("📦 Disk Pages: {} | 📈 Catalog Tuples: {}", pages, tuples);
                }
            } else {
                warning!("❌ Table [{}] not found in pg_class. Check schema or quotes.", table_name);
            }
            info!("--- [END DIAGNOSTIC] ---");
            Ok::<(), pgrx::spi::Error>(())
        }).ok();
    }

    pub fn next_batch(&mut self) -> Option<(Vec<f32>, u32)> {
        if self.vectors_read >= self.num_samples || self.dims <= 1 { return None; }
        let mut samples_to_read = self.num_samples_per_batch as usize;
        let remaining = (self.num_samples - self.vectors_read) as usize;
        if remaining < (self.num_samples_per_batch + self.min_samples_per_batch) as usize { samples_to_read = remaining; }
        let start_idx = (self.vectors_read as usize) * (self.dims as usize);
        let end_idx = (self.vectors_read as usize + samples_to_read) * (self.dims as usize);
        if end_idx > self.cached_vectors.len() { return None; }
        let batch = self.cached_vectors[start_idx..end_idx].to_vec();
        self.vectors_read += samples_to_read as u64;
        Some((batch, self.dims))
    }

    pub(crate) fn end_scan(self) { info!("🏁 Training scan complete."); }
}