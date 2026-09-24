-- Lucius relational schema (migration 1).
-- SQLite is the index/relational layer; frames and media stay as external files.

CREATE TABLE users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at REAL NOT NULL
);

-- A session is one contiguous stream of experience: a live demonstration (WATCH ME),
-- an agent execution, a practice attempt, or a trajectory reconstructed from external media.
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    kind TEXT NOT NULL CHECK (kind IN ('live_demo','agent_execution','practice','external_media','benchmark','validation')),
    source TEXT NOT NULL,                -- learning source class (user_demo, external_video, agent_success, ...)
    status TEXT NOT NULL,                -- recording, finalized, interrupted, processing, processed, failed
    task_id TEXT,
    task_text TEXT,
    task_class TEXT,
    reference_ids TEXT NOT NULL DEFAULT '[]',
    blender_version TEXT,
    os_version TEXT,
    resolution TEXT,
    dpi_scale REAL,
    monitor_layout TEXT,
    start_time REAL NOT NULL,
    end_time REAL,
    outcome TEXT,                        -- success, failure, partial, unknown, executed_unverified
    recording_config TEXT NOT NULL DEFAULT '{}',
    policy TEXT NOT NULL,                -- DataPolicy JSON (consent, license, training eligibility)
    content_hash TEXT,
    processing TEXT NOT NULL DEFAULT '{}', -- per-stage processing status
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX idx_sessions_kind ON sessions(kind, start_time);
CREATE INDEX idx_sessions_status ON sessions(status);
CREATE INDEX idx_sessions_task_class ON sessions(task_class);

CREATE TABLE media_assets (
    id TEXT PRIMARY KEY,
    demonstration_id TEXT REFERENCES demonstrations(id) ON DELETE SET NULL,
    kind TEXT NOT NULL CHECK (kind IN ('video','image','image_sequence','blender_project','text','screenshot')),
    role TEXT NOT NULL,                  -- demonstration, reference, target, before, after, intermediate, instruction...
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
CREATE INDEX idx_media_sha ON media_assets(sha256);
CREATE INDEX idx_media_demo ON media_assets(demonstration_id);

CREATE TABLE demonstrations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_types TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL,                -- UPLOADED..READY / FAILED
    status_history TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    instructions TEXT NOT NULL DEFAULT '[]',
    task_text TEXT,
    source_class TEXT NOT NULL,
    policy TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE frames (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    ts REAL NOT NULL,
    path TEXT NOT NULL,                  -- relative to data_dir/frames (content addressed)
    sha256 TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    source TEXT NOT NULL,                -- screen, window, video, image, viewport_render
    media_asset_id TEXT REFERENCES media_assets(id) ON DELETE SET NULL,
    media_timestamp REAL,
    window_bounds TEXT,
    active_area TEXT,
    dhash TEXT,                          -- 64-bit difference hash for visual-change detection
    change_score REAL,                   -- visual change vs previous stored frame (0..1)
    meta TEXT NOT NULL DEFAULT '{}',
    UNIQUE(session_id, seq)
);
CREATE INDEX idx_frames_session_ts ON frames(session_id, ts);

-- Raw captured events. Never rewritten by downstream processing.
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    actor TEXT NOT NULL CHECK (actor IN ('human','agent','system')),
    blender_active INTEGER,
    payload TEXT NOT NULL DEFAULT '{}',
    UNIQUE(session_id, seq)
);
CREATE INDEX idx_events_session_kind ON events(session_id, kind);

CREATE TABLE segments (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    t_start REAL NOT NULL,
    t_end REAL NOT NULL,
    step_start INTEGER NOT NULL,
    step_end INTEGER NOT NULL,
    label TEXT NOT NULL,
    title TEXT,
    label_confidence REAL NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ('deterministic','model','human')),
    locked INTEGER NOT NULL DEFAULT 0,
    outcome TEXT NOT NULL DEFAULT 'unknown',
    boundary_reasons TEXT NOT NULL DEFAULT '[]',
    label_evidence TEXT NOT NULL DEFAULT '[]',
    representative_frame_ids TEXT NOT NULL DEFAULT '[]',
    summary TEXT,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX idx_segments_session ON segments(session_id, idx);

CREATE TABLE trajectory_steps (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    t_start REAL NOT NULL,
    t_end REAL NOT NULL,
    frame_before_id TEXT REFERENCES frames(id) ON DELETE SET NULL,
    frame_after_id TEXT REFERENCES frames(id) ON DELETE SET NULL,
    action_type TEXT NOT NULL,
    action_payload TEXT NOT NULL DEFAULT '{}',
    action_source TEXT NOT NULL CHECK (action_source IN ('observed','inferred','model_inferred','human_confirmed')),
    evidence_kind TEXT NOT NULL,
    action_confidence REAL NOT NULL,
    evidence TEXT NOT NULL DEFAULT '[]',
    candidate_actions TEXT NOT NULL DEFAULT '[]',
    window_title TEXT,
    window_bounds TEXT,
    mode_label TEXT,
    tool_label TEXT,
    selection_hint TEXT,
    undo_redo_flag TEXT,
    actor TEXT NOT NULL,
    segment_id TEXT REFERENCES segments(id) ON DELETE SET NULL,
    event_seq_start INTEGER,
    event_seq_end INTEGER,
    media_timestamp REAL,
    state_before TEXT,
    state_after TEXT,
    meta TEXT NOT NULL DEFAULT '{}',
    UNIQUE(session_id, idx)
);
CREATE INDEX idx_steps_segment ON trajectory_steps(segment_id);

-- Human edits to segments/intents are themselves learning data.
CREATE TABLE human_edits (
    id TEXT PRIMARY KEY,
    subject_kind TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    op TEXT NOT NULL,
    before TEXT,
    after TEXT,
    user_id TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX idx_edits_subject ON human_edits(subject_kind, subject_id);

CREATE TABLE intents (
    id TEXT PRIMARY KEY,
    segment_id TEXT NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    category TEXT NOT NULL,
    target TEXT,
    scope TEXT,
    confidence REAL NOT NULL,
    evidence_source TEXT NOT NULL,       -- rules, model, human
    evidence TEXT NOT NULL DEFAULT '[]',
    reason_code TEXT NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);
CREATE INDEX idx_intents_segment ON intents(segment_id, is_current);

CREATE TABLE taxonomy_terms (
    kind TEXT NOT NULL,                  -- segment_label, intent_category, intent_target, object_class, action_type
    term TEXT NOT NULL,
    description TEXT,
    parent TEXT,
    source TEXT NOT NULL,                -- system_seeded, human, model
    created_at REAL NOT NULL,
    PRIMARY KEY (kind, term)
);

CREATE TABLE skills (
    id TEXT PRIMARY KEY,                 -- stable key, e.g. hard_surface_blade_blockout
    name TEXT NOT NULL,
    status TEXT NOT NULL,                -- candidate_pattern, candidate_skill, validated, high_confidence, disabled, merged
    current_version INTEGER NOT NULL,
    source_class TEXT NOT NULL,
    origin_sources TEXT NOT NULL DEFAULT '[]',
    categories TEXT NOT NULL DEFAULT '[]',
    object_class TEXT,
    confidence REAL NOT NULL DEFAULT 0,
    confidence_breakdown TEXT NOT NULL DEFAULT '{}',
    usage_count INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    human_confirmations INTEGER NOT NULL DEFAULT 0,
    human_rejections INTEGER NOT NULL DEFAULT 0,
    last_used REAL,
    parent_skill_id TEXT REFERENCES skills(id) ON DELETE SET NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX idx_skills_status ON skills(status);

CREATE TABLE skill_versions (
    id TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    definition TEXT NOT NULL,
    change_note TEXT NOT NULL,
    created_by TEXT NOT NULL,            -- extractor, generalizer, human, rollback, merge
    parent_version INTEGER,
    created_at REAL NOT NULL,
    UNIQUE(skill_id, version)
);

-- Provenance: which evidence supports which skill.
CREATE TABLE skill_examples (
    id TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL REFERENCES skills(id) ON DELETE CASCADE,
    skill_version INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('demonstration','validation','execution','counterexample','correction')),
    source_class TEXT NOT NULL,
    session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    segment_ids TEXT NOT NULL DEFAULT '[]',
    media_asset_id TEXT REFERENCES media_assets(id) ON DELETE SET NULL,
    run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
    t_start REAL,
    t_end REAL,
    frame_ids TEXT NOT NULL DEFAULT '[]',
    outcome TEXT,
    evidence_weight REAL NOT NULL,
    instance_signature TEXT,             -- distinguishes instances for generalization evidence
    summary TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX idx_skill_examples_skill ON skill_examples(skill_id);

-- Learning/provenance graph between heterogeneous nodes (skills, failures, episodes, media...).
CREATE TABLE graph_edges (
    id TEXT PRIMARY KEY,
    src_kind TEXT NOT NULL,
    src_id TEXT NOT NULL,
    rel TEXT NOT NULL,
    dst_kind TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    meta TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(src_kind, src_id, rel, dst_kind, dst_id)
);
CREATE INDEX idx_edges_dst ON graph_edges(dst_kind, dst_id);

CREATE TABLE memory_episodes (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL UNIQUE REFERENCES sessions(id) ON DELETE CASCADE,
    task_text TEXT,
    task_class TEXT,
    source_class TEXT NOT NULL,
    outcome TEXT,
    summary TEXT NOT NULL,
    structured TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX idx_episodes_task_class ON memory_episodes(task_class);

CREATE TABLE semantic_memories (
    id TEXT PRIMARY KEY,
    statement TEXT NOT NULL,
    pattern_key TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT '{}',
    source_class TEXT NOT NULL,
    status TEXT NOT NULL,                -- candidate, validated, rejected
    support_count INTEGER NOT NULL,
    contradiction_count INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL,
    evidence TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE failure_records (
    id TEXT PRIMARY KEY,
    signature TEXT NOT NULL UNIQUE,
    task_class TEXT,
    phase TEXT,
    observed_problem TEXT NOT NULL,
    symptoms TEXT NOT NULL DEFAULT '[]',
    likely_cause TEXT,
    correction TEXT,
    future_rule TEXT,
    rule_status TEXT NOT NULL,           -- candidate, promoted, confirmed, rejected
    trigger_action TEXT,                 -- action type whose premature/incorrect use caused the failure
    guard_checkpoint TEXT,               -- checkpoint id that must pass before trigger_action
    evidence TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL,
    confidence_breakdown TEXT NOT NULL DEFAULT '{}',
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    correction_attempts INTEGER NOT NULL DEFAULT 0,
    correction_successes INTEGER NOT NULL DEFAULT 0,
    retrieval_priority REAL NOT NULL DEFAULT 1,
    skill_id TEXT REFERENCES skills(id) ON DELETE SET NULL,
    source_class TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_seen REAL NOT NULL
);
CREATE INDEX idx_failures_task ON failure_records(task_class, phase);

CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    task_text TEXT NOT NULL,
    task_class TEXT,
    mode TEXT NOT NULL,                  -- execute, practice, benchmark, validation
    state TEXT NOT NULL,
    status TEXT NOT NULL,                -- running, success, failure, executed_unverified, aborted
    backend TEXT NOT NULL,
    environment TEXT NOT NULL,           -- blender_live, blender_headless
    arm TEXT,                            -- experiment arm / strategy
    plan TEXT,
    retrieval_id TEXT,
    practice_task_id TEXT,
    benchmark_id TEXT,
    params TEXT NOT NULL DEFAULT '{}',
    metrics TEXT NOT NULL DEFAULT '{}',
    started_at REAL NOT NULL,
    ended_at REAL
);
CREATE INDEX idx_runs_mode ON runs(mode, started_at);

CREATE TABLE run_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    evidence TEXT NOT NULL DEFAULT '[]',
    confidence REAL,
    context TEXT NOT NULL DEFAULT '{}',
    ts REAL NOT NULL
);
CREATE INDEX idx_transitions_run ON run_transitions(run_id, seq);

CREATE TABLE corrections (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
    failure_id TEXT REFERENCES failure_records(id) ON DELETE SET NULL,
    kind TEXT NOT NULL,                  -- takeover, demo_undo, annotation
    reason TEXT,                         -- only when the human provided one
    start_time REAL NOT NULL,
    end_time REAL,
    before_state TEXT,
    after_state TEXT,
    before_frame_id TEXT REFERENCES frames(id) ON DELETE SET NULL,
    after_frame_id TEXT REFERENCES frames(id) ON DELETE SET NULL,
    agent_context TEXT NOT NULL DEFAULT '{}',
    correction_steps TEXT NOT NULL DEFAULT '[]',
    outcome TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX idx_corrections_failure ON corrections(failure_id);

CREATE TABLE retrieval_records (
    id TEXT PRIMARY KEY,
    run_id TEXT,
    query_text TEXT NOT NULL,
    query_meta TEXT NOT NULL DEFAULT '{}',
    strategy TEXT NOT NULL,
    results TEXT NOT NULL,
    feedback TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE TABLE evaluations (
    id TEXT PRIMARY KEY,
    subject_kind TEXT NOT NULL,          -- run, session, segment, skill_validation
    subject_id TEXT NOT NULL,
    run_id TEXT REFERENCES runs(id) ON DELETE CASCADE,
    level INTEGER NOT NULL CHECK (level BETWEEN 1 AND 5),
    checkpoint_id TEXT,
    passed INTEGER,
    score REAL,
    method TEXT NOT NULL,                -- execution, structural, visual_measured, visual_model, human, generalization
    subjective INTEGER NOT NULL DEFAULT 0,
    evaluator TEXT NOT NULL,
    evidence TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX idx_evaluations_subject ON evaluations(subject_kind, subject_id);

CREATE TABLE reference_constraints (
    id TEXT PRIMARY KEY,
    media_asset_id TEXT NOT NULL REFERENCES media_assets(id) ON DELETE CASCADE,
    constraint_type TEXT NOT NULL,
    target TEXT NOT NULL,
    value TEXT NOT NULL,
    source TEXT NOT NULL,                -- measured, model_inferred, human_confirmed
    confidence REAL NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE benchmarks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    definition TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    arms TEXT NOT NULL,
    benchmark_ids TEXT NOT NULL,
    status TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE TABLE benchmark_results (
    id TEXT PRIMARY KEY,
    benchmark_id TEXT NOT NULL REFERENCES benchmarks(id) ON DELETE CASCADE,
    experiment_id TEXT REFERENCES experiments(id) ON DELETE CASCADE,
    run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
    arm TEXT NOT NULL,
    variant TEXT,
    success INTEGER NOT NULL,
    metrics TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX idx_bench_results ON benchmark_results(benchmark_id, arm);

CREATE TABLE practice_tasks (
    id TEXT PRIMARY KEY,
    curriculum TEXT NOT NULL,
    stage INTEGER NOT NULL,
    name TEXT NOT NULL,
    definition TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE(curriculum, stage, name)
);

CREATE TABLE mastery_records (
    id TEXT PRIMARY KEY,
    curriculum TEXT NOT NULL,
    stage INTEGER NOT NULL,
    status TEXT NOT NULL,                -- locked, practicing, mastered, needs_demonstration
    metrics TEXT NOT NULL,
    mastery_score REAL,
    human_override TEXT,
    updated_at REAL NOT NULL,
    UNIQUE(curriculum, stage)
);

CREATE TABLE datasets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    filters TEXT NOT NULL,
    path TEXT NOT NULL,
    stats TEXT NOT NULL DEFAULT '{}',
    quality_report TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE TABLE dataset_samples (
    id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
    line_no INTEGER NOT NULL,
    quality_score REAL NOT NULL,
    issues TEXT NOT NULL DEFAULT '[]',
    provenance TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE TABLE embeddings (
    owner_kind TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (owner_kind, owner_id, provider)
);

CREATE TABLE preferences (
    user_id TEXT NOT NULL REFERENCES users(id),
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL NOT NULL,
    source TEXT NOT NULL,                -- learned, human
    evidence TEXT NOT NULL DEFAULT '[]',
    updated_at REAL NOT NULL,
    PRIMARY KEY (user_id, key)
);

CREATE TABLE processing_jobs (
    id TEXT PRIMARY KEY,
    subject_kind TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending','running','done','failed','skipped')),
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    result TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(subject_kind, subject_id, stage)
);
CREATE INDEX idx_jobs_status ON processing_jobs(status, created_at);

CREATE TABLE model_calls (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT,
    purpose TEXT NOT NULL,
    status TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    latency_s REAL,
    error TEXT,
    created_at REAL NOT NULL
);

-- Observability log of bus events (what the system decided and why).
CREATE TABLE event_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    ts REAL NOT NULL,
    subject_id TEXT,
    payload TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX idx_event_log_type ON event_log(type, ts);
