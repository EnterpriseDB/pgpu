--NOTE: This SQL test script needs to be manually run right now. We need a worker node with GPU and that isn't available in CI yet


-- Prepare
create extension pgpu cascade;
create extension vchord cascade;
-- add to "/var/lib/postgresql/17/main/postgresql.auto.conf"
-- shared_preload_libraries = 'vchord'






CREATE TABLE test_vecs_normalized_small (
                              id SERIAL PRIMARY KEY,
                              embedding vector(2)
);
CREATE TABLE test_vecs_normalized_large (
                              id SERIAL PRIMARY KEY,
                              embedding vector(2)
);
CREATE TABLE test_vecs_small (
                              id SERIAL PRIMARY KEY,
                              embedding vector(2)
);

INSERT INTO test_vecs_small (embedding)
SELECT
    ARRAY[
        (random() * 2 - 1), -- X value
        (random() * 2 - 1)  -- Y value
        ]::vector
FROM generate_series(1, 1000);

INSERT INTO test_vecs_normalized_large (embedding)
SELECT
    l2_normalize(ARRAY[
        (random() * 2 - 1), -- X value
        (random() * 2 - 1)  -- Y value
        ]::vector)
FROM generate_series(1, 100000);

-- 4. Verify the data
-- This query checks the id, the raw vector, and proves the length is 1.0 (Normalized)
SELECT *
FROM test_vecs_normalized_small
LIMIT 10;

-- run the python script to visualize the vectors




-- sampling factor and cluster count guarantee that we execute in batches
-- NOTE: must use l2 distance to avoid normalizing vectors
SELECT pgpu.create_vector_index_on_gpu(table_name => 'public.test_vecs_normalized_small', column_name => 'embedding', batch_size => 1000,
                                       cluster_count => 10, sampling_factor => 100, kmeans_iterations=>10,
                                       kmeans_nredo=>1, distance_operator=> 'ip', skip_index_build=> true);




-- sampling factor and cluster count guarantee that we execute in batches
-- NOTE: must use l2 distance to avoid normalizing vectors
SELECT pgpu.create_vector_index_on_gpu(table_name => 'public.test_vecs_small', column_name => 'embedding', batch_size => 1000,
                                       cluster_count => 10, sampling_factor => 100, kmeans_iterations=>10,
                                       kmeans_nredo=>1, distance_operator=> 'ip', skip_index_build=> true);






-- check with:
select (vector::real[])[1]
from test_100k_vecs_ip_centroids
ORDER BY (vector::real[])[1];











-- compare against series. Check if centroids are within "1" distance of expected 1..100
WITH expected AS (
    SELECT g AS val
    FROM generate_series(1, 100) AS g
),
     produced AS (
         SELECT DISTINCT (vector::real[])[1]::double precision AS first_dim
         FROM test_100k_vecs_centroids
     ),
     matched AS (
         -- For each expected val, pick the nearest produced centroid within 0.5
         SELECT e.val,
                p.first_dim
         FROM expected e
                  LEFT JOIN LATERAL (
             SELECT p.first_dim
             FROM produced p
             WHERE abs(p.first_dim - e.val) < 1.0
             ORDER BY abs(p.first_dim - e.val)
             LIMIT 1
             ) p ON true
     )
SELECT
            COUNT(*) FILTER (WHERE first_dim IS NOT NULL) AS matches,
            COUNT(*) FILTER (WHERE first_dim IS NULL)  AS missing_count,
            array_agg(val ORDER BY val) FILTER (WHERE first_dim IS NULL) AS missing_values
FROM matched;


-- matches | missing_count | missing_values
-- ---------+---------------+----------------
--      100 |             0 |
-- (1 row)