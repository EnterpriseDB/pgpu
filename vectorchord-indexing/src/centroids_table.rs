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

    // Note: benchmarks show no improvement from using the "mean root" so we just use 0 here
    //let root = mean_filtered(&centroids, -1).expect("no centroids with parent -1 found");
    let root = vec![0.0; vector_dimensions as usize];
    Spi::run_with_args(
        &format!("INSERT INTO {table_name} (id, parent, vector) VALUES (0, NULL, $1)"),
        &[root.into()],
    )
    .expect("unable to insert root centroid");

    let query = &format!(
        // note: parent ID is an option so it will be NULL for the top level of the voronoi tree
        // i.e. in a single-level case every centroid will have NULL parent, and in multi/2-level the top level (non leaves) will have NULL
        "INSERT INTO {table_name} (id, parent, vector)
            VALUES ($1, $2, $3)",
    );

    // TODO: can we do a more efficient bulk insert? The only other way I found is to
    // use string formatting and prepare a long statement.
    Spi::connect_mut(|client| {
        // note: we explicitly control the IDs since they must match the parent IDs in multi-level case
        let mut i: i32 = 1;
        for (vec, parent) in centroids {
            // parent + 1 serves two purposes:
            // the cluster labels start at 0 but our table has the root at ID 0 and the actual parent clusters start at 1; so need to add 1
            // centroids without parent get assigned parent ID -1 in our upstream code. So by adding one we get 0; the root
            client
                .update(query, None, &[i.into(), (parent + 1).into(), vec.into()])
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
