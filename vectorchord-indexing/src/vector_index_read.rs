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
        _column_name: String, // Unused for count(*)
        num_samples: u64,
        num_samples_per_batch: u64,
        min_samples_per_batch: u64,
    ) -> Self {
        let start_time = Instant::now();

        // 1. DIAGNOSTICS: Check Transaction State
        let is_txn = unsafe { pg_sys::IsTransactionState() };
        info!("🔍 [Sanity Check] Is Transaction Active? {}", is_txn);

        // 2. SNAPSHOT: Ensure we can see data
        unsafe {
            let snap = pg_sys::GetTransactionSnapshot();
            if !snap.is_null() {
                pg_sys::PushActiveSnapshot(snap);
            }
        }

        // 3. EXECUTE SIMPLE SQL (Fixed Syntax)
        // We use Spi::connect manually to catch the exact error if it fails.
        let result = Spi::connect(|client| {
            // A. Check connectivity
            // Fix: Pass &[] instead of None for arguments
            let db_name = client.select("SELECT current_database()", None, &[])
                .map(|t| t.get_one::<String>().unwrap_or(Some("?".to_string())))
                .unwrap_or(Some("ERROR".to_string()));

            info!("🌍 [Sanity Check] Connected to DB: {:?}", db_name);

            // B. Run the count
            // We quote the table name manually to be safe
            let query = format!("SELECT count(1) FROM \"{}\"", table_name.replace("\"", ""));

            info!("🚀 [Sanity Check] Running: {}", query);

            // Fix: Pass &[] for arguments
            let count = client.select(&query, None, &[])
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
                    // If count is 0, the table is empty or invisible to this transaction.
                    // This confirms the "Cold Start" theory.
                    panic!("FATAL: Table exists but count(1) returned 0. Visibility issue confirmed.");
                }
            },
            Ok(None) => info!("⚠️ [Sanity Check] Query ran but returned NULL."),
            Err(e) => {
                // If this prints, we know exactly why SPI is failing (e.g. "Relation does not exist")
                info!("❌ [Sanity Check] SPI Failed with Error: {:?}", e);
                panic!("FATAL: SPI Connection Failed. See logs above.");
            }
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