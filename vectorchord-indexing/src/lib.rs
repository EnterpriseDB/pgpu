mod centroids_table;
mod clustering_gpu_impl;
pub mod guc;
pub mod index;
mod vector_index_read;
mod vector_type;
mod vectorchord_index;
mod util;

pub use guc::*;
pub use index::*;
use pgrx::Spi;

fn print_memory(v: &Vec<f32>, message: &str) {
    let heap_size = calculate_vec_size(v) as f64 / 1024.0 / 1024.0 / 1024.0;
    pgrx::info!("Data length in main memory ({message}): {heap_size:.2} GB");
}

fn calculate_vec_size<T>(v: &Vec<T>) -> usize {
    // 1. Get the size of a single element (e.g., f32 is 4 bytes)
    let elem_size = std::mem::size_of::<T>();

    // 2. Get the current allocated capacity of the Vec
    let capacity = v.capacity();

    // 3. Calculate the total heap memory size in bytes
    capacity * elem_size
}

/// Parse a fully qualified table identifier using Postgres' built-in `parse_ident`.
///
/// # Panics
/// Panics if the identifier does not contain exactly `schema.table`.
pub fn parse_table_identifier(ident: &str) -> (String, String) {
    let parts: Vec<String> =
        Spi::get_one_with_args("SELECT pg_catalog.parse_ident($1)", &[ident.into()])
            .expect("parse_ident() returned NULL")
            .unwrap();

    if parts.len() != 2 {
        panic!(
            "Invalid identifier '{ident}': must contain schema and table name (e.g \"public.table\"), got {parts:?}",
        );
    }

    (parts[0].clone(), parts[1].clone())
}
