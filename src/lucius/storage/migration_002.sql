-- 002: media assets may be captions (timed narration of a demonstration video).
-- SQLite cannot change a CHECK constraint in place: the table is rebuilt (foreign keys are
-- switched off by the migration runner so rows referencing media keep their references).
CREATE TABLE media_assets_new (
    id TEXT PRIMARY KEY,
    demonstration_id TEXT REFERENCES demonstrations(id) ON DELETE SET NULL,
    kind TEXT NOT NULL CHECK (kind IN ('video','image','image_sequence','blender_project','text','screenshot',
                                       'captions')),
    role TEXT NOT NULL,                  -- demonstration, reference, target, before, after, intermediate, instruction, narration...
    filename TEXT NOT NULL,
    path TEXT NOT NULL,                  -- relative to data_dir/media
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    mime TEXT,
    width INTEGER,
    height INTEGER,
    duration_s REAL,
    fps REAL,
    frame_count INTEGER,
    policy TEXT NOT NULL,
    analysis TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
INSERT INTO media_assets_new (id, demonstration_id, kind, role, filename, path, sha256, size_bytes, mime, width,
                              height, duration_s, fps, frame_count, policy, analysis, created_at)
    SELECT id, demonstration_id, kind, role, filename, path, sha256, size_bytes, mime, width, height, duration_s,
           fps, frame_count, policy, analysis, created_at FROM media_assets;
DROP TABLE media_assets;
ALTER TABLE media_assets_new RENAME TO media_assets;
CREATE INDEX idx_media_sha ON media_assets(sha256);
CREATE INDEX idx_media_demo ON media_assets(demonstration_id);
