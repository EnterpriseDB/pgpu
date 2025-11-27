use pgrx::{info, Spi};
use crate::util;

pub fn create_vectorchord_index(
    table: String,
    schema_table: String,
    centroids_table_name: String,
    distance_operator: String,
) {
    util::assert_valid_distance_operator(&distance_operator);
    let index_name = format!("{table}_pgpu_ext");
    let index_metric_type = format!("vector_{distance_operator}_ops");

    info!("running \"CREATE INDEX {index_name} ON {schema_table} USING vchordrq\" with external centroids");

    Spi::run(&format!("DROP INDEX IF EXISTS {index_name};"))
        .expect("error deleting old centroids table");

    Spi::run(&format!(
        "CREATE INDEX {index_name} ON {schema_table} USING vchordrq (embedding {index_metric_type}) WITH (options = $$
residual_quantization = true
build.pin = true
[build.external]
table = '{centroids_table_name}'
$$);"
    ))
    .expect("error creating vectorchord index");
}
