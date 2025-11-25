mod vectorchord_indexing;

use pgrx::prelude::*;

pg_module_magic!(name, version);

/// For basic CI testing; just to see if the extension will run
#[pg_extern]
fn ping() -> &'static str {
    "pong"
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
