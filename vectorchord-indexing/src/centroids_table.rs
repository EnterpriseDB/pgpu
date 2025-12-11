use pgrx::{info, warning, Spi};
use std::time::Instant;

pub fn store_centroids(centroids: Vec<(Vec<f32>, i32)>, table_name: String, vector_dimensions: u32) {
    info!("Storing {} centroids in {table_name}", centroids.len());
    let start_time = Instant::now();

    pgrx::notice!("Deleting old centroids table (if it exists from a previous run)");
    Spi::run(&format!("DROP TABLE IF EXISTS {table_name}"))
        .expect("error deleting old centroids table");
    Spi::run(&format!("CREATE TABLE {table_name} (id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, parent INT, vector vector({vector_dimensions}))")).expect("error creating centroids table");

    if centroids.is_empty() {
        warning!("\tNo centroids to store, skipping");
        return;
    }

    // TODO: vchord told us we don't need the root; just use NULL as parent for all entries
    // I'm leaving it in for now since this was extensively tested, but we can remove it if we want
    let root = centroids
        .first()
        .expect("expected at least one centroid; found empty list")
        .clone();
    Spi::run_with_args(&format!("INSERT INTO {table_name} (id, parent, vector) OVERRIDING SYSTEM VALUE VALUES (0, NULL, $1)"), &[root.into()]).expect("unable to insert root centroid");

    let query = &format!(
        // note: parent ID is always 0 since we only support one lvl in the voronoi tree
        "INSERT INTO {table_name} (parent, vector)
            VALUES ($1, $2)",
    );

    // TODO: can we do a more efficient bulk insert? The only other way I found is to
    // use string formatting and prepare a long statement.
    Spi::connect_mut(|client| {
        for (vec, parent) in centroids {
            client
                .update(query, None, &[parent.into(), vec.into()])
                .expect("error inserting centroid");
        }
    });

    info!("\tStoring centroids took: {:.2?}", start_time.elapsed());
}
