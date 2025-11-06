#[cfg(feature = "cuda_gpu")]
mod centroids_table;
#[cfg(feature = "cuda_gpu")]
pub mod guc;
#[cfg(feature = "cuda_gpu")]
pub mod index;
#[cfg(feature = "cuda_gpu")]
mod vector_index_read;
#[cfg(feature = "cuda_gpu")]
mod vectorchord_index;
#[cfg(feature = "cuda_gpu")]
mod wrapper_gpu_lib;

#[cfg(feature = "cuda_gpu")]
pub use guc::*;
#[cfg(feature = "cuda_gpu")]
pub use index::*;

#[cfg(not(feature = "cuda_gpu"))]
pub mod guc_fallback;
#[cfg(not(feature = "cuda_gpu"))]
pub mod index_fallback;

#[cfg(not(feature = "cuda_gpu"))]
pub use guc_fallback::*;
#[cfg(not(feature = "cuda_gpu"))]
pub use index_fallback::*;
use pgrx::Spi;

#[cfg(feature = "cuda_gpu")]
fn print_memory(vectors: &[f32], message: &str) {
    let bytes_len = std::mem::size_of_val(vectors);
    let kb_len = bytes_len as f64 / 1024.0 / 1024.0 / 1024.0;
    pgrx::info!("Data length in main memory ({message}): {:.2} GB", kb_len);
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
