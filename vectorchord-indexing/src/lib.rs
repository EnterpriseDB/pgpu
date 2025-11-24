mod centroids_table;
mod clustering_gpu_impl;
pub mod guc;
pub mod index;
mod vector_index_read;
mod vectorchord_index;
mod vector_type;

pub use guc::*;
pub use index::*;
use pgrx::spi::SpiError;
use pgrx::{debug1, pg_extern, warning, PgMemoryContexts, Spi};
use std::ffi::c_long;
use std::time::Instant;

fn print_memory(v: &Vec<f32>, message: &str) {
    let bytes_len = std::mem::size_of_val(v);
    let kb_len = bytes_len as f64 / 1024.0 / 1024.0 / 1024.0;
    let heap_size = calculate_vec_size(v) as f64 / 1024.0 / 1024.0 / 1024.0;
    pgrx::info!("Data length in main memory ({message}): {kb_len:.2} GB, heap {heap_size:.2} GB");
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

pub fn dump_mem_ctx() {
    Spi::run(
        "DO $$
DECLARE
    current_pid INTEGER;
BEGIN
    -- 1. Get the PID of the current session.
    SELECT pg_backend_pid() INTO current_pid;

    RAISE NOTICE 'Dumping memory contexts for PID % to the server log.', current_pid;

    -- 2. Call the dump function using the retrieved PID.
    -- This function returns TRUE on success.
    PERFORM pg_log_backend_memory_contexts(current_pid);

    RAISE NOTICE 'Memory contexts have been logged. Check the PostgreSQL server logs.';

END
$$ LANGUAGE plpgsql;",
    )
    .unwrap();
}

#[pg_extern]
pub fn apples() {
    warning!("running memory test");
    let query = "SELECT
            g.row_num,
            (
                SELECT ARRAY_AGG(i ORDER BY i)
                FROM GENERATE_SERIES(1, 10000) AS t(i)
            ) AS increasing_numbers_array
        FROM GENERATE_SERIES(1, 10000) AS g(row_num);
            ";
    for i in 0..1000000000 {
        let vecs = unsafe {
            PgMemoryContexts::switch_to(
                &mut PgMemoryContexts::Transient {
                    parent: PgMemoryContexts::CurrentMemoryContext.value(),
                    name: "per-call",
                    min_context_size: 4096,
                    initial_block_size: 4096,
                    max_block_size: 4096,
                },
                |_| {
                    // get one batch using offset
                    let vecs = Spi::connect(|client| {
                        let res = client
                            .select(query, None, &[])
                            .expect("unable to get result")
                            .map(|row| {
                                Ok::<Vec<i32>, SpiError>(
                                    row.get_by_name("increasing_numbers_array")?.unwrap(),
                                )
                            })
                            .collect::<Result<Vec<_>, SpiError>>()
                            .expect("");
                        res.to_owned()
                    });
                    vecs
                },
            )
        };
        dump_mem_ctx();
        let used_md: i64 = Spi::get_one("select used_bytes/1024/1024 as \"used MB\" from pg_backend_memory_contexts where name = 'TopTransactionContext';").expect("unable to get memory usage").unwrap();
        warning!(
            "iteration {i}, received {} vectors - TopTransactionContext used memory: {:.2} MB",
            vecs.len(),
            used_md
        );
    }
}

#[pg_extern]
pub fn bananas() {
    warning!("running memory test");
    let query = "select public.increasing_numbers_array()";
    for i in 0..1000 {
        let vecs = unsafe {
            PgMemoryContexts::switch_to(
                &mut PgMemoryContexts::Transient {
                    parent: PgMemoryContexts::CurrentMemoryContext.value(),
                    name: "per-call",
                    min_context_size: 4096,
                    initial_block_size: 4096,
                    max_block_size: 4096,
                },
                |_| {
                    // get one batch using offset
                    let vecs = Spi::connect(|client| {
                        let res = client
                            .select(query, None, &[])
                            .expect("unable to get result")
                            .map(|row| {
                                Ok::<Vec<i32>, SpiError>(
                                    row.get_by_name("increasing_numbers_array")?.unwrap(),
                                )
                            })
                            .collect::<Result<Vec<_>, SpiError>>()
                            .expect("");
                        res.to_owned()
                    });
                    vecs
                },
            )
        };
        let used_md: i64 = Spi::get_one("select used_bytes/1024/1024 as \"used MB\" from pg_backend_memory_contexts where name = 'TopTransactionContext';").expect("unable to get memory usage").unwrap();
        warning!(
            "iteration {i}, received {} vectors - TopTransactionContext used memory: {:.2} MB",
            vecs.len(),
            used_md
        );
    }
}
