use pgrx::{debug1, info, warning, Spi};
use std::time::Instant;

pub fn store_centroids(
    centroids: Vec<(Vec<f32>, i32)>,
    table_name: String,
    vector_dimensions: u32,
    residual_quantization: bool, // Add this parameter to the function signature
) {
    info!("Storing {} centroids in {table_name}", centroids.len());
    let start_time = Instant::now();

    // 1. Prepare the table
    Spi::run(&format!("DROP TABLE IF EXISTS {table_name}")).ok();
    Spi::run(&format!(
        "CREATE TABLE {table_name} (id INT, parent INT, vector vector({vector_dimensions}))"
    )).expect("Failed to create centroids table");

    let is_hierarchical = centroids.iter().any(|(_, p)| *p != -1);

    // 2. Insert Super Root (ID 0)
    // For IP/RQ, we MUST use a zero vector for ID 0.
    // This ensures Level 1 (Roots) are treated as absolute offsets.
    let root_vec = vec![0.0; vector_dimensions as usize];
    Spi::run_with_args(
        &format!("INSERT INTO {table_name} (id, parent, vector) VALUES (0, NULL, $1)"),
        &[root_vec.into()],
    ).expect("Unable to insert ID 0 root");

    // 3. Prepare Insert Query (Using owned String to resolve table name)
    let query = format!("INSERT INTO {table_name} (id, parent, vector) VALUES ($1, $2, $3)");

    Spi::connect_mut(|client| {
        let mut i: i32 = 1;

        // Note: For RQ math, we need the absolute root vectors to calculate residuals
        // only if they weren't already residuals in the input
        for (vec, parent) in centroids {
            // Parent mapping:
            // parent -1 (Root) -> points to 0 (Super Root)
            // parent 0..399 (Leaf) -> points to 1..400 (Root)
            let pg_parent = Some(parent + 1);

            // FIX: Use &query to satisfy pgrx SpiClient::update's Query trait bound
            client
                .update(&query, None, &[i.into(), pg_parent.into(), vec.into()])
                .expect("Error inserting centroid");

            i += 1;
        }
    });

    debug1!("\tStoring centroids took: {:.2?}", start_time.elapsed());
}


/// calcluates the mean vector of all vectors with the given target ID
fn _mean_filtered(data: &[(Vec<f32>, i32)], target_id: i32) -> Option<Vec<f32>> {
    let mut sum_vec: Option<Vec<f32>> = None;
    let mut count = 0.0;

    for (vec, id) in data {
        if *id == target_id {
            if let Some(sums) = &mut sum_vec {
                // Add current vector to existing sums
                for (i, val) in vec.iter().enumerate() {
                    sums[i] += val;
                }
            } else {
                // First match: initialize sums with this vector's values
                sum_vec = Some(vec.clone());
            }
            count += 1.0;
        }
    }

    // Divide by count
    sum_vec.map(|sums| sums.iter().map(|s| s / count).collect())
}
