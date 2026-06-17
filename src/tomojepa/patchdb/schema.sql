-- patchdb DuckDB schema.
-- BLOB columns hold raw little-endian numpy bytes; dtype/shape are implied by
-- the owning model row (grid, k, embed_dim).

CREATE SEQUENCE IF NOT EXISTS seq_collections START 1;
CREATE SEQUENCE IF NOT EXISTS seq_images START 1;

CREATE TABLE IF NOT EXISTS collections (
    id          INTEGER PRIMARY KEY DEFAULT nextval('seq_collections'),
    name        VARCHAR UNIQUE NOT NULL,
    created_at  TIMESTAMP DEFAULT now(),
    params      JSON
);

-- One shared-basis model per collection.
CREATE TABLE IF NOT EXISTS models (
    collection_id  INTEGER PRIMARY KEY,
    ckpt_path      VARCHAR,
    k              INTEGER,        -- components per token
    embed_dim      INTEGER,        -- backbone token dim D (basis is [k, D])
    grid           INTEGER,        -- patch grid side G
    patch_size     INTEGER,        -- pixels per patch
    img_size       INTEGER,
    whiten_index   BOOLEAN,        -- whether stored FAISS vectors are whitened
    outlier_pct    DOUBLE,
    fg_thresh      DOUBLE,         -- foreground std threshold (NULL = no fg mask)
    basis          BLOB,           -- float32 [k, D]
    mean           BLOB,           -- float32 [1, D]
    ev             BLOB            -- float32 [k]
);

CREATE TABLE IF NOT EXISTS images (
    id             INTEGER PRIMARY KEY DEFAULT nextval('seq_images'),
    collection_id  INTEGER,
    ord            INTEGER,        -- contiguous position within collection
    dataset_index  INTEGER,        -- slice index in the source stack
    source_uri     VARCHAR,        -- e.g. "soild_stack.zarr#484"
    pattern        VARCHAR,
    backend        VARCHAR,
    data_dir       VARCHAR,
    dataset_key    VARCHAR,
    n_fg           INTEGER,        -- foreground token count
    created_at     TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS codes (
    image_id  INTEGER PRIMARY KEY,
    codes     BLOB,                -- float16 [G, G, k]
    fg        BLOB                 -- bool    [G, G]
);

-- FAISS index sidecar + the token map that resolves each vector back to its
-- patch. The map is stored as a numpy sidecar (int32 [ntotal, 3] = image ord,
-- gi, gj) rather than DB rows: at ~0.6M tokens a columnar row-insert is far too
-- slow, and the engine needs the map as a contiguous array anyway.
CREATE TABLE IF NOT EXISTS faiss_index (
    collection_id  INTEGER PRIMARY KEY,
    path           VARCHAR,
    token_path     VARCHAR,
    ntotal         BIGINT,
    dim            INTEGER,
    metric         VARCHAR,
    index_type     VARCHAR
);
