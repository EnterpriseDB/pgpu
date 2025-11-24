use crate::{dump_mem_ctx, vector_type};
use pgrx::datum::PgVarlena;
use pgrx::pg_sys::{varlena, SysScanDesc};
use pgrx::prelude::PgHeapTuple;
use pgrx::spi::SpiError;
use pgrx::{debug1, debug2, debug3, debug5, heap_getattr_raw, info, pg_sys, spi, warning, Array, PgBox, PgMemoryContexts, PgRelation, PgTupleDesc, Spi};
use std::ffi::{c_char, c_long};
use std::time::Instant;

pub struct VectorReadBatcher {
    table_name: String,
    column_name: String,
    vectors_total: u64,
    vectors_per_batch: u64,
    cursor_name: Option<String>,
    table_scan: Option<SysScanDesc>,
    pg_rel: Option<PgRelation>,
    col_num: Option<usize>,
}

impl VectorReadBatcher {
    pub fn new(
        table_name: String,
        column_name: String,
        vectors_total: u64,
        vectors_per_batch: u64,
    ) -> Self {
        VectorReadBatcher {
            table_name,
            column_name,
            vectors_total,
            vectors_per_batch,
            cursor_name: None,
            table_scan: None,
            pg_rel: None,
            col_num: None,
        }
    }

    pub(crate) fn get_batch_cursor(&mut self) -> Option<(Vec<f32>, u32)> {
        debug1!("(SPI/SQL Cursor) Reading a batch of {vectors_per_batch} vectors (total vectors to read: {vectors_total}) from {table_name}.{column_name}...",
        vectors_total = self.vectors_total,
        vectors_per_batch = self.vectors_per_batch,
        table_name = self.table_name,
        column_name = self.column_name,);

        let start_time = Instant::now();

        // Open the cursor on the first use. We will reference it by name
        if self.cursor_name.is_none() {
            // TODO: use pgvector libs to get access to the native vector type and remove casting
            let source_table_query = &format!(
                "SELECT {column_name}::float4[] AS raw_embedding FROM {table_name} LIMIT {vectors_total}",
                column_name = self.column_name,
                vectors_total = self.vectors_total,
                table_name = self.table_name,
            );
            let cursor_name = Spi::connect_mut(|client| {
                let cursor = client.open_cursor(source_table_query, &[]);
                Ok::<_, spi::Error>(cursor.detach_into_name())
            })
            .expect("error opening cursor");
            self.cursor_name = Some(cursor_name);
        }

        // get one batch from the cursor
        let vecs = Spi::connect_mut(|client| {
            let mut cursor = client
                .find_cursor(self.cursor_name.as_ref().unwrap())
                .expect("unable to find cursor");

            let res = cursor
                .fetch(self.vectors_per_batch as c_long)
                .expect("unable to fetch vectors")
                .map(|row| Ok::<Vec<f32>, SpiError>(row["raw_embedding"].value()?.unwrap()))
                .collect::<Result<Vec<_>, SpiError>>()
                .expect("error reading vectors from table");

            // The cursor needs to be detached explicitly if we want to use it again. Otherwise, it will be dropped
            // once this function returns
            if !res.is_empty() {
                self.cursor_name = Some(cursor.detach_into_name());
            }
            res
        });

        let dims = match vecs.first() {
            Some(vector) => vector.len(),
            None => 0,
        };

        debug1!(
            "Read {} vectors in: {:.2?}",
            vecs.len(),
            start_time.elapsed()
        );
        match vecs.is_empty() {
            true => None,
            false => Some((vecs.into_iter().flatten().collect(), dims as u32)),
        }
    }

    pub(crate) fn get_batch_native(&mut self) -> Option<(Vec<f32>, u32)> {
        debug1!("(TABLE SCAN) Reading a batch of {vectors_per_batch} vectors, (total vectors to read: {vectors_total}) from {table_name}.{column_name}...",
        vectors_total = self.vectors_total,
        vectors_per_batch = self.vectors_per_batch,
        table_name = self.table_name,
        column_name = self.column_name);
        unsafe {
            let start_time = Instant::now();

            // Open the cursor on the first use. We will reference it by name
            if self.table_scan.is_none() {
                let pg_rel = PgRelation::open_with_name_and_share_lock(&*self.table_name)
                    .expect("unable to open table");
                if !pg_rel.is_table() {
                    panic!(
                        "table {} is not a table; only regular tables are supported",
                        self.table_name
                    );
                }
                debug2!(
                    "table {} is a table and has {:?} tuples",
                    self.table_name,
                    pg_rel.reltuples()
                );

                // look for the column number
                let tup_desc = pg_rel.tuple_desc();
                let mut col_num_found: Option<i32> = None;
                for (_, attr) in tup_desc.iter().filter(|a| !a.attisdropped).enumerate() {
                    let col_name = pgrx::name_data_to_str(&attr.attname);
                    if col_name == self.column_name {
                        col_num_found = Some(attr.attnum.into());
                    }
                }
                let col_num = col_num_found.expect(
                    format!(
                        "column {} not found in table {}",
                        self.column_name, self.table_name
                    )
                    .as_str(),
                );
                self.col_num = Some(col_num as usize);

                // initialize a simple sequential table scan; "systable_beginscan" is just called "systable" for historical reasons
                // very common to use this wrapper on user tables
                let scan = pg_sys::systable_beginscan(
                    pg_rel.as_ptr(),
                    pg_sys::InvalidOid,               // no index
                    false,                            // no idex use
                    pg_sys::GetTransactionSnapshot(), // we need to use our snapshot to not violate MVCC
                    0,                                // number of scan keys
                    std::ptr::null_mut(),             // no key; no filter needed
                );
                self.table_scan = Some(scan);
                self.pg_rel = Some(pg_rel.clone());
                debug1!("systable scan initialized");
                warning!("systable scan initialized");
            }

            let scan = self.table_scan.clone().expect("systable scan not initialized");
            let pg_rel = self.pg_rel.clone().expect("tuple descriptor not initialized");
            let tup_desc = pg_rel.tuple_desc();
            let col_num = self.col_num.expect("column number not initialized");


            let mut all_vectors: Vec<f32> = Vec::new();
            let mut dims: u32 = 0;
            for i in 0..self.vectors_per_batch {
                let tuple = pg_sys::systable_getnext(scan);
                if tuple.is_null() {
                    pg_sys::systable_endscan(scan);
                    break;
                }
                debug3!("({i}) got a tuple");

                let pgtup = PgHeapTuple::from_heap_tuple(tup_desc.clone(), tuple);
                // Extract the raw Datum
                let pg_attr = pgtup
                    .get_attribute_by_index(std::num::NonZero::new(col_num).unwrap())
                    //.get_by_index::<pgrx::datum::PgVarlena<f32>>(std::num::NonZero::new(col_num as usize).unwrap())
                    .unwrap();

                debug3!("({i}) got a pg_attr: {:?}", pg_attr);

                let datum = heap_getattr_raw(
                    tuple,
                    std::num::NonZero::new(col_num).unwrap(),
                    tup_desc.as_ptr(),
                )
                .expect("unable to get datum");

                debug3!("({i}) got a datum: {:?}", datum);

                let raw_ptr = datum.cast_mut_ptr() as *mut pg_sys::varlena;
                let detoasted_ptr = pg_sys::pg_detoast_datum(raw_ptr);

                let byte_slice = pgrx::varlena_to_byte_slice(detoasted_ptr);

                debug5!("({i}) got bytes: {:?}", byte_slice);

                let (vector_values, dims) = vector_type::decode_pgvector_vector(byte_slice);
                all_vectors.extend_from_slice(&vector_values);

                debug5!("({i}) got the vector: {:?}", vector_values);

                // CLEANUP: run after data was copied
                if detoasted_ptr != raw_ptr {
                    pg_sys::pfree(detoasted_ptr as *mut std::ffi::c_void);
                }
            }

            //pg_sys::systable_endscan(scan);
            info!(
            "Read vectors in: {:.2?}",
            start_time.elapsed()
        );
            match all_vectors.is_empty() {
                true => None,
                false => Some((all_vectors, dims)),
            }
        }
    }

    pub fn get_batch_test(&self) -> Option<(Vec<f32>, u32)> {
        let dims = 2000;
        let inner_vec: Vec<f32> = std::iter::repeat(0.01234).take(dims).collect();
        let matrix: Vec<Vec<f32>> = std::iter::repeat(inner_vec.clone())
            .take(self.vectors_per_batch as usize)
            .collect();
        let matrix_flat: Vec<f32> = matrix.into_iter().flatten().collect();
        Some((matrix_flat, dims as u32))
    }
}

impl Iterator for VectorReadBatcher {
    type Item = (Vec<f32>, u32);

    fn next(&mut self) -> Option<Self::Item> {
        self.get_batch_cursor()
        //self.get_batch_native()
    }
}
