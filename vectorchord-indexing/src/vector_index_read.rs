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

        // 1. TRANSACTION STATE CHECK
        let is_txn = unsafe { pg_sys::IsTransactionState() };
        info!("🔍 [Sanity Check] Is Transaction Active? {}", is_txn);

        // 2. SNAPSHOT SETUP
        unsafe {
            let snap = pg_sys::GetTransactionSnapshot();
            if !snap.is_null() {
                pg_sys::PushActiveSnapshot(snap);
            }
        }

        // 3. CORRECT QUOTING LOGIC
        // "public.table" -> "public"."table"
        let quoted_table = if table_name.contains('.') {
            table_name.split('.')
                .map(|part| format!("\"{}\"", part))
                .collect::<Vec<_>>()
                .join(".")
        } else {
            format!("\"{}\"", table_name)
        };

        // 4. EXECUTE SQL
        let result = Spi::connect(|client| {
            let query = format!("SELECT count(1) FROM {}", quoted_table);
            info!("🚀 [Sanity Check] Running: {}", query);

            let count = client.select(&query, None, &[])
                .and_then(|t| t.get_one::<i64>());

            Ok::<Option<i64>, pgrx::spi::Error>(count.ok().flatten())
        });

        // 5. CLEANUP
        unsafe {
            let snap = pg_sys::GetTransactionSnapshot();
            if !snap.is_null() {
                pg_sys::PopActiveSnapshot();
            }
        }

        // 6. REPORT
        match result {
            Ok(Some(c)) => {
                info!("✅ [Sanity Check] Success! Count: {}", c);
                if c == 0 {
                    panic!("FATAL: Table exists but is empty (visibility issue).");
                }
            },
            Ok(None) => info!("⚠️ [Sanity Check] Query returned NULL."),
            Err(e) => {
                info!("❌ [Sanity Check] SPI Failed: {:?}", e);
                panic!("FATAL: SPI Failed");
            }
        }

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