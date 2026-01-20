use crate::vector_type;
use pgrx::pg_sys;
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
        _column_name: String, // Unused for this sanity check
        num_samples: u64,
        num_samples_per_batch: u64,
        min_samples_per_batch: u64,
    ) -> Self {
        let start_time = Instant::now();

        // 1. DIAGNOSTICS: Check Transaction State
        // SPI requires an active transaction. If this is false, we found the root cause.
        let is_txn = unsafe { pg_sys::IsTransactionState() };
        info!("🔍 [Sanity Check] Is Transaction Active? {}", is_txn);

        // 2. SNAPSHOT: Ensure we can see data
        // Even in standard envs, background workers sometimes need this push.
        unsafe {
            let snap = pg_sys::GetTransactionSnapshot();
            if !snap.is_null() {
                pg_sys::PushActiveSnapshot(snap);
            }
        }

        // 3. EXECUTE SIMPLE SQL
        // We use Spi::connect manually to catch the exact error if it fails.
        let result = Spi::connect(|client| {
            // A. Check simple connectivity
            let db_name = client.select("SELECT current_database()", None, None)
                .map(|t| t.get_one::<String>().unwrap_or(Some("?".to_string())))
                .unwrap_or(Some("ERROR".to_string()));

            info!("🌍 [Sanity Check] Connected to DB: {:?}", db_name);

            // B. Run the count
            // We quote the table name manually to be safe
            let query = format!("SELECT count(1) FROM \"{}\"", table_name.replace("\"", ""));

            info!("🚀 [Sanity Check] Running: {}", query);

            let count = client.select(&query, None, None)
                .and_then(|t| t.get_one::<i64>());

            Ok::<Option<i64>, pgrx::spi::Error>(count.ok().flatten())
        });

        // 4. CLEANUP SNAPSHOT
        unsafe {
            let snap = pg_sys::GetTransactionSnapshot();
            if !snap.is_null() {
                pg_sys::PopActiveSnapshot();
            }
        }

        // 5. REPORT RESULTS
        match result {
            Ok(Some(c)) => {
                info!("✅ [Sanity Check] Success! Count: {}", c);
                if c == 0 {
                    info!("⚠️ Table exists but is empty (or rows are invisible).");
                }
            },
            Ok(None) => info!("⚠️ [Sanity Check] Query ran but returned NULL."),
            Err(e) => info!("❌ [Sanity Check] SPI Failed: {:?}", e), // This prints the real error
        }

        // Return dummy to prevent crash, allowing you to read logs
        VectorReadBatcher {
            num_samples,
            num_samples_per_batch,
            min_samples_per_batch,
            vectors_read: 0,
            cached_vectors: vec![],
            dims: 1,
        }
    }

    pub fn next_batch(&mut self) -> Option<(Vec<f32>, u32)> {
        None
    }

    pub(crate) fn end_scan(self) {}
}