# PGPU - GPU acceleration for PostgreSQL

PGPU is a postgres extension that can use NVIDIA GPUs with CUDA to accelerate certain operations in the database and/or to offload them
from the CPU to the GPU.

Right now, the extension implements GPU acceleration for vector index build using the vectorchord extension.

## GPU accelerated vector index build
The vectorchord extension uses an index type that is based on "vector centroids". These
centroids can be computed very quickly on a GPU and vectorchord can leverage such
externally computed centroids via an "external build" setting:  https://docs.vectorchord.ai/vectorchord/usage/external-index-precomputation.html

PGPU implement exactly this process:
- read data from the database
- compute centroids on the GPU
- write centroids to PG table
- call vectorchord indexing, passing the pre-computed centroids table

This work is based on scripts that vectorchord provides: https://github.com/tensorchord/VectorChord/tree/main/scripts#run-external-index-precomputation-toolkit

PGPU just simplifies this process by packaging everything into a single PG exntension and only requiring a single function call.
![pgpu process flow](docs/images/pgpu_flow.png)



### Usage examples
#### Set up test data and run PGPU
```sql
-- create test tables and generate data
-- 10K table
CREATE TABLE test_10k_vecs
(
   id        bigserial PRIMARY KEY,
   embedding vector(2000)
);


INSERT INTO test_10k_vecs (embedding)
SELECT arr.embedding
FROM generate_series(1, 10000) AS g(i)
        CROSS JOIN LATERAL (
   SELECT array_agg(((g.i - 1) * 3 + gs.j)::real)
   FROM generate_series(1, 2000) AS gs(j)
   ) AS arr(embedding);
```

#### Run PGPU
```sql
SELECT pgpu.create_vector_index_on_gpu(table_name => 'public.test_10k_vecs', 
                                       column_name => 'embedding', 
                                       batch_size => 1000, 
                                       cluster_count => 1000, 
                                       sampling_factor => 10, 
                                       kmeans_iterations=>10, 
                                       kmeans_nredo=>1, 
                                       distance_operator=>'ip',
                                       skip_index_build=>true,
                                       spherical_centroids=>true
       );
```

## Function reference
```sql
CREATE FUNCTION "create_vector_index_on_gpu"(
        "table_name" TEXT,
        "column_name" TEXT,
        "cluster_count" bigint DEFAULT 1000,
        "sampling_factor" bigint DEFAULT 256,
        "batch_size" bigint DEFAULT 100000,
        "kmeans_iterations" bigint DEFAULT 10,
        "kmeans_nredo" bigint DEFAULT 1,
        "distance_operator" TEXT DEFAULT 'ip',
        "skip_index_build" bool DEFAULT false,
        "spherical_centroids" bool DEFAULT false
) RETURNS void STRICT
```

- `table_name`: the fully qualified table name
  - example: `public.test_table`
- `column_name`: the vector column in the table that should be indexed
- `cluster_count`: how many centroids should be computed
  - default: `1000`
  - note: refer to vectorchord docs for more details on this parameter: https://docs.vectorchord.ai/vectorchord/usage/indexing.html#tuning this is effectively the `lists` parameter in vectorchord
- `sampling_factor`: how many samples to take per centroid/cluster
  - default: `256`
  - note: values below 40 are not recommended. More samples lead to more accurate indexes but also increase the clustering time
- `batch_size`: how many rows to process at once
  - default: `100000`
  - note: when this number is lower than cluster_count*sampling_factor, clustering will run in multiple batches. This is useful to reduce the overall amount of memory required for clustering
- `kmeans_iterations`: how many iterations to run during clustering
  - default: `10`
  - note: this rarely needs to be changed
- `kmeans_nredo`: how many times to rerun the clustering algorithm
  - default: `1`
  - note: this rarely needs to be changed
- `distance_operator`: what distance operator to use for clustering 
  - default: `'ip'`
  - valid values: `'ip'`, `'l2'`, `'cos'`
  - note: the index will be built for this specific distance operator. So it will only be used for queries with the same distance operator. Typically, this is determined by the dataset.
- `skip_index_build`: skip the index build step and only create the centroids table
  - note: useful for testing/benchmarking purposes
- `spherical_centroids`: whether to normalize centroids to unit sphere
  - default: `false`
  - note: this should be enabled when using `ip` distance operator and/or when using a dataset that is normalized to unit sphere


## Building and running
See script [scripts/setup_build.sh](scripts/setup_build.sh)

- PGPU uses NVIDIA cuVS for GPU accelerated k-means clustering https://github.com/rapidsai/cuvs/tree/main/rust
- `vectorchord` (aka. `vchord`) and `pgvector` (aka. `vector`) PG extensions need to be installed