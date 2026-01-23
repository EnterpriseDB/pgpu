use crate::vector_type;
use pgrx::{debug1, info, warning, Spi};
use std::time::{SystemTime, UNIX_EPOCH};
use std::time::Instant;

pub struct VectorReadBatcher {
    qualified_table_name: String,
    column_name: String,

    // Sampling State
    target_samples: u64,
    vectors_read: u64,

    // The list of Block Numbers we intend to read
    blocks_to_read: Vec<u64>,
    current_block_idx: usize,

    // Constants
    blocks_per_query: usize,
    active: bool,
}

impl VectorReadBatcher {
    pub fn new(
        qualified_table_name: String,
        column_name: String,
        num_clusters: u32,
        sampling_factor: u32,
        _requested_batch_size: u64,
    ) -> Self {
        // 1. Calculate Target
        let target_samples = (num_clusters as u64).saturating_mul(sampling_factor as u64);

        // 2. Get Physical Size
        let table_bytes: i64 = Spi::get_one(&format!(
            "SELECT pg_relation_size('{}'::regclass)",
            qualified_table_name
        )).expect("SPI failed to look up table size").unwrap_or(0);

        let total_blocks = (table_bytes / 8192).max(1) as u64;

        // 3. DYNAMIC DENSITY CALCULATION
        info!("🔍 [DEBUG] Inspecting column '{}' for density estimation...", column_name);

        // Get column info
        let (typname, atttypmod): (String, i32) = Spi::get_two(&format!(
            "SELECT t.typname, a.atttypmod \
             FROM pg_attribute a \
             JOIN pg_type t ON a.atttypid = t.oid \
             WHERE a.attrelid = '{}'::regclass AND a.attname = '{}'",
            qualified_table_name, column_name
        )).expect("Failed to get column info").unwrap_or(("vector".to_string(), -1));

        // Parse Dims
        let dims = if atttypmod > 0 { atttypmod as u64 } else { 0 };
        let vec_size = dims * 4;

        info!("🔍 [DEBUG] Column Info -> Type: '{}', Mod: {}, Dims: {}, Vector Bytes: {}",
            typname, atttypmod, dims, vec_size);

        // Density Formula
        let est_rows_per_