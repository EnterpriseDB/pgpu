use pgrx::spi::SpiError;
use pgrx::{debug1, spi, Spi};
use std::ffi::c_long;
use std::time::Instant;

pub struct VectorReadBatcher {
    table_name: String,
    column_name: String,
    vectors_total: u64,
    vectors_per_batch: u64,
    cursor_name: Option<String>,
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
        }
    }

    pub(crate) fn get_batch(&mut self) -> Option<(Vec<Vec<f32>>, u32)> {
        debug1!("Reading a batch of {vectors_per_batch} vectors (total vectors to read: {vectors_total}) from {table_name}.{column_name}...",
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
            false => Some((vecs, dims as u32)),
        }
    }
}

impl Iterator for VectorReadBatcher {
    type Item = (Vec<Vec<f32>>, u32);

    fn next(&mut self) -> Option<Self::Item> {
        self.get_batch()
    }
}
