--NOTE: This SQL test script needs to be manually run right now. We need a worker node with GPU and that isn't available in CI yet


-- Prepare
create extension pgpu cascade;
create extension vchord cascade;
-- add to "/var/lib/postgresql/17/main/postgresql.auto.conf"
-- shared_preload_libraries = 'vchord'

-- Setup
CREATE TABLE test_100k_vecs_ip
(
    id        bigserial PRIMARY KEY,
    embedding vector(2000)
);



-- we get 100k vectors but only 100 different ones.
-- the first vector is [1, 1, 1, 1, ...] and gets repeated 1000 times
-- the 1001 vector is [2, 2, 2, ...] and gets repeated 1000 times
--  this means kmeans SHOULD find 100 unique clusters close to those values
-- each vector is [k, k, k, ..., k] (length DIM)
DO
$$
    DECLARE
        DIM    int := 2000;
        K      int := 100; -- NOTE: not actually using this below; we're using floats here so it's awkward using counters like this
        COPIES int := 1000;
    BEGIN
        EXECUTE format($f$
        INSERT INTO test_100k_vecs_ip (embedding)
        SELECT array_fill(k::real, ARRAY[%1$s])::vector
        FROM generate_series(0.001, 0.1, 0.001) AS k
        CROSS JOIN generate_series(1, %3$s) AS r;
      $f$, DIM, K, COPIES);
    END
$$;



-- sampling factor and cluster count guarantee that we execute in batches
-- NOTE: must use l2 distance to avoid normalizing vectors
SELECT pgpu.create_vector_index_on_gpu(table_name => 'public.test_100k_vecs_ip', column_name => 'embedding', batch_size => 10000,
                                       cluster_count => 100, sampling_factor => 1000, kmeans_iterations=>10,
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