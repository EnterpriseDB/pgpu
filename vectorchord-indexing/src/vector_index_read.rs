use crate::vector_type;
use pgrx::pg_sys::{format_type_be, SysScanDesc};
use pgrx::{debug1, heap_getattr_raw, info, pg_sys, warning, PgRelation, Spi};
use std::ffi::CStr;
use std::time::Instant;

pub struct VectorReadBatcher {
    table_name: String,
    column_name: String,
    num_tuples_in_table: Option<u64>,
    num_samples: u64,
    num_samples_per_batch: u64,
    min_samples_per_batch: u64,
    vectors_read: u64,
    table_scan: Option<SysScanDesc>,
    pg_rel: Option<PgRelation>,
    col_num: Option<usize>,
}

impl VectorReadBatcher {
    pub fn new(
        table_name: String,
        column_name: String,
        num_samples: u64,
        num_samples_per_batch: u64,
        min_samples_per_batch: u64,
    ) -> Self {
        let mut vbr = VectorReadBatcher {
            table_name,
            column_name,
            num_samples,
            num_samples_per_batch,
            min_samples_per_batch,
            num_tuples_in_table: None,
            vectors_read: 0,
            table_scan: None,
            pg_rel: None,
            col_num: None,
        };
        vbr.initialize();
        let table_size = (vbr).num_tuples();
        assert!(num_samples <= table_size as u64, "The table has fewer records ({table_size}) than the desired number of samples ({num_samples}) based on cluster_count*sampling_factor. Unable to continue");
        let rem = num_samples % num_samples_per_batch;
        if rem != 0 && rem < min_samples_per_batch {
            warning!("batch size {num_samples_per_batch} will lead to a remainder of {rem} samples in the last batch; which is too small for clustering. The last batch will be enlarged to {0} to contain this remainder", vbr.num_samples_per_batch + rem)
        }
        // TODO: calculate this from a new input "max memory GB"
        info!("vector batch read properties:\n\t num_samples: {num_samples}\n\t num_samples_per_batch: {num_samples_per_batch}\n\t num_batches: {nb}\n\t table_size: {table_size}", nb=vbr.num_batches(), num_samples=vbr.num_samples, num_samples_per_batch=vbr.num_samples_per_batch, table_size=table_size);
        vbr
    }

    pub fn num_batches(&self) -> u32 {
        self.num_samples.div_ceil(self.num_samples_per_batch) as u32
    }

    // original SQL/SPI based implementation. Unused because of memory "leak": https://github.com/pgcentralfoundation/pgrx/issues/2211
    // left-in for reference
    /*    pub(crate) fn get_batch_cursor(&mut self) -> Option<(Vec<f32>, u32)> {
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

        info!(
            "Read {} vectors in: {:.2?}",
            vecs.len(),
            start_time.elapsed()
        );
        match vecs.is_empty() {
            true => None,
            false => Some((vecs.into_iter().flatten().collect(), dims as u32)),
        }
    }*/

    fn initialize(&mut self) {
        let pg_rel = PgRelation::open_with_name_and_share_lock(&self.table_name)
            .expect("unable to open table");
        if !pg_rel.is_table() {
            pgrx::error!(
                "table {} is not a table; only regular tables are supported",
                self.table_name
            );
        }

        // look for the column number
        let tup_desc = pg_rel.tuple_desc();
        let mut col_num_found: Option<i32> = None;
        for attr in tup_desc.iter().filter(|a| !a.attisdropped) {
            let col_name = pgrx::name_data_to_str(&attr.attname);
            unsafe {
                if col_name == self.column_name {
                    let type_name = CStr::from_ptr(format_type_be(attr.atttypid))
                        .to_str()
                        .expect("invalid type name");
                    if type_name != "vector" {
                        pgrx::error!("column \"{}\" type is not \"vector\". Only pgvector/vector types are supported", self.column_name);
                    }
                    col_num_found = Some(attr.attnum.into());
                }
            }
        }
        let col_num = col_num_found.unwrap_or_else(|| {
            pgrx::error!(
                "column {} not found in table {}",
                self.column_name, self.table_name
            )
        });
        self.col_num = Some(col_num as usize);

        // initialize a simple sequential table scan; "systable_beginscan" is just called "systable" for historical reasons
        // very common to use this wrapper on user tables
        let scan = unsafe {
            pg_sys::systable_beginscan(
                pg_rel.as_ptr(),
                pg_sys::InvalidOid,               // no index
                false,                            // no idex use
                pg_sys::GetTransactionSnapshot(), // we need to use our snapshot to not violate MVCC
                0,                                // number of scan keys
                std::ptr::null_mut(),             // no key; no filter needed
            )
        };
        self.table_scan = Some(scan);
        self.pg_rel = Some(pg_rel.clone());
        debug1!("systable scan initialized");
    }

    pub(crate) fn num_tuples(&mut self) -> u64 {
        match self.num_tuples_in_table {
            None => {
                let tuples: i64 =
                    Spi::get_one(format!("SELECT COUNT(1) FROM {}", self.table_name).as_str())
                        .unwrap()
                        .unwrap();
                self.num_tuples_in_table = Some(tuples as u64);
                tuples as u64
            }
            Some(tuples) => tuples,
        }
    }

    pub(crate) fn end_scan(self) {
        let scan = self.table_scan.expect("systable scan not initialized");
        unsafe {
            pg_sys::systable_endscan(scan);
        }
    }

    pub(crate) fn next_batch(&mut self) -> Option<(Vec<f32>, u32)> {
        // take the remainder into this batch if it would be too small for clustering
        let mut samples_to_read = self.num_samples_per_batch;
        let size_next_batch = self
            .num_samples
            .saturating_sub(self.vectors_read)
            .saturating_sub(self.num_samples_per_batch);
        if size_next_batch < self.min_samples_per_batch {
            samples_to_read += size_next_batch;
        }
        debug1!("({vectors_read}/{num_samples}) Reading next batch of {num_samples_per_batch} from {table_name}.{column_name}...",
            num_samples = self.num_samples,
            vectors_read = self.vectors_read,
            num_samples_per_batch = samples_to_read,
            table_name = self.table_name,
            column_name = self.column_name
        );
        let start_time = Instant::now();
        unsafe {
            assert!(
                self.table_scan.is_some(),
                "systable scan not initialized; call start_scan() first"
            );

            let scan = self.table_scan.expect("systable scan not initialized");
            let pg_rel = self
                .pg_rel
                .clone()
                .expect("tuple descriptor not initialized");
            let tup_desc = pg_rel.tuple_desc();
            let col_num = self.col_num.expect("column number not initialized");
            let col_num_nonzero = std::num::NonZero::new(col_num).unwrap();

            let mut all_vectors: Vec<f32> = Vec::new();
            let mut dims: u32 = 0;
            for _i in 0..samples_to_read {
                if self.vectors_read >= self.num_samples {
                    break;
                }
                self.vectors_read += 1;
                let tuple = pg_sys::systable_getnext(scan);
                if tuple.is_null() {
                    break;
                }
                //debug3!("({i}) got a tuple");

                let datum = heap_getattr_raw(tuple, col_num_nonzero, tup_desc.as_ptr())
                    .expect("unable to get datum");
                //debug3!("({i}) got a datum: {:?}", datum);

                let raw_ptr = datum.cast_mut_ptr();
                let detoasted_ptr = pg_sys::pg_detoast_datum(raw_ptr);

                let byte_slice = pgrx::varlena_to_byte_slice(detoasted_ptr);
                //debug5!("({i}) got bytes: {:?}", byte_slice);

                let (vector_values, vector_dims) = vector_type::decode_pgvector_vector(byte_slice);
                all_vectors.extend_from_slice(&vector_values);
                dims = vector_dims;
                //debug5!("({i}) got the vector: {:?}", vector_values);

                if detoasted_ptr != raw_ptr {
                    pg_sys::pfree(detoasted_ptr as *mut std::ffi::c_void);
                }
            }

            debug1!("Read vectors in: {:.2?}", start_time.elapsed());
            match all_vectors.is_empty() {
                true => None,
                false => Some((all_vectors, dims)),
            }
        }
    }

    /*    pub fn get_batch_test(&self) -> Option<(Vec<f32>, u32)> {
        let dims = 2000;
        let inner_vec: Vec<f32> = std::iter::repeat(0.01234).take(dims).collect();
        let matrix: Vec<Vec<f32>> = std::iter::repeat(inner_vec.clone())
            .take(self.vectors_per_batch as usize)
            .collect();
        let matrix_flat: Vec<f32> = matrix.into_iter().flatten().collect();
        Some((matrix_flat, dims as u32))
    }*/
}
