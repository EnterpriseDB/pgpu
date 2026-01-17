use pgrx::{debug1, info, warning, Spi};
use std::time::Instant;

pub fn store_centroids(
    centroids: Vec<(Vec<f32>, i32)>,
    table_name: String,
    vector_dimensions: u32,
) {
    info!("Storing {} centroids in {table_name}", centroids.len());
    let start_time = Instant::now();

    pgrx::notice!("Deleting old centroids table (if it exists from a previous run)");
    Spi::run(&format!("DROP TABLE IF EXISTS {table_name}"))
        .expect("error deleting old centroids table");
    Spi::run(&format!(
        "CREATE TABLE {table_name} (id INT, parent INT, vector vector({vector_dimensions}))"
    ))
    .expect("error creating centroids table");

    if centroids.is_empty() {
        warning!("\tNo centroids to store, skipping");
        return;
    }

    // --- NEW LOGIC: Determine if we are Hierarchical ---
    // If any centroid has a parent that is NOT -1, we are in a multi-level hierarchy.
    let is_hierarchical = centroids.iter().any(|(_, p)| *p != -1);

    let mut start_index: i32 = 0;

    if !is_hierarchical {
        // FLAT INDEX PATH: Keep original logic of adding a "Super Root" at ID 0
        let root = vec![0.0; vector_dimensions as usize];
        Spi::run_with_args(
            &format!("INSERT INTO {table_name} (id, parent, vector) VALUES (0, NULL, $1)"),
            &[root.into()],
        )
        .expect("unable to insert root centroid");
        start_index = 1; // Start user centroids at ID 1
    } else {
        // HIERARCHICAL PATH: Do NOT insert a root at ID 0.
        // Let the hierarchical function's own roots be the Level 0 nodes.
        start_index = 0; // Start user centroids at ID 0
    }

    let query = &format!("INSERT INTO {table_name} (id, parent, vector) VALUES ($1, $2, $3)");

    Spi::connect_mut(|client| {
        let mut i: i32 = start_index;
        for (vec, parent) in centroids {
            let pg_parent: Option<i32>;

            if is_hierarchical {
                // In hierarchical mode, parent -1 means NULL (Top level)
                pg_parent = if parent == -1 { None } else { Some(parent) };
            } else {
                // In flat mode, parent -1 points to our Super Root (ID 0)
                pg_parent = Some(parent + 1);
            }

            client
                .update(query, None, &[i.into(), pg_parent.into(), vec.into()])
                .expect("error inserting centroid");
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
