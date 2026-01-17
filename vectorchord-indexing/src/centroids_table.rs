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

    // --- NEW: Detect Hierarchy and Calculate Single Root ---
    let is_hierarchical = centroids.iter().any(|(_, p)| *p != -1);

    let root_vec = if is_hierarchical {
        // For 2-layer hierarchical: The Super Root (ID 0) is the average of all Roots (-1)
        let mut sum_vec = vec![0.0; vector_dimensions as usize];
        let mut count = 0.0;
        for (vec, parent) in &centroids {
            if *parent == -1 {
                for (i, val) in vec.iter().enumerate() {
                    sum_vec[i] += val;
                }
                count += 1.0;
            }
        }
        if count > 0.0 {
            sum_vec.iter().map(|v| v / count).collect()
        } else {
            vec![0.0; vector_dimensions as usize]
        }
    } else {
        // For Flat: Just use a zero vector as the Super Root
        vec![0.0; vector_dimensions as usize]
    };

    // Insert the ONLY NULL parent allowed
    Spi::run_with_args(
        &format!("INSERT INTO {table_name} (id, parent, vector) VALUES (0, NULL, $1)"),
        &[root_vec.into()],
    ).expect("unable to insert root centroid");

    let query = format!("INSERT INTO {table_name} (id, parent, vector) VALUES ($1, $2, $3)");

    Spi::connect_mut(|client| {
        let mut i: i32 = 1;
        for (vec, parent) in centroids {
            // parent + 1 mapping:
            // parent -1 (Roots) -> 0 (points to Super Root ID 0)
            // parent 0-399 (Leaves) -> 1-400 (points to their specific Root ID)
            let pg_parent = Some(parent + 1);

            client
                .update(&query, None, &[i.into(), pg_parent.into(), vec.into()])
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
