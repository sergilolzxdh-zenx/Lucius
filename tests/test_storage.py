from __future__ import annotations

import pytest
from PIL import Image

from lucius.events import EventType
from lucius.provenance import DataPolicy
from lucius.sessions import CapturedEvent, EventKind, SessionKind, SessionStatus
from lucius.storage.frames import hash_distance


def test_session_lifecycle_and_event_ordering(sessions):
    session = sessions.create(user_id="u1", kind=SessionKind.LIVE_DEMO, policy=DataPolicy.for_live_demo(),
                              task_text="model a sword")
    assert session.status == SessionStatus.RECORDING
    events = [CapturedEvent(seq=i, ts=1000.0 + i * 0.1, kind=EventKind.KEY_DOWN, payload={"key": "E"})
              for i in range(5)]
    sessions.append_events(session.id, list(reversed(events)))
    # Re-appending (journal replay) is idempotent.
    sessions.append_events(session.id, events[:2])
    stored = sessions.events(session.id)
    assert [e.seq for e in stored] == [0, 1, 2, 3, 4]
    assert all(a.ts <= b.ts for a, b in zip(stored, stored[1:]))
    finalized = sessions.finalize(session.id)
    assert finalized.status == SessionStatus.FINALIZED
    assert finalized.end_time is not None and finalized.content_hash


def test_frames_are_content_addressed_and_deleted_with_session(sessions, frames):
    s1 = sessions.create(user_id="u", kind=SessionKind.LIVE_DEMO, policy=DataPolicy.for_live_demo())
    s2 = sessions.create(user_id="u", kind=SessionKind.LIVE_DEMO, policy=DataPolicy.for_live_demo())
    image = Image.new("RGB", (64, 48), (40, 90, 200))
    f1 = sessions.add_frame(s1.id, image, seq=0, ts=1.0, source="screen")
    f2 = sessions.add_frame(s2.id, image, seq=0, ts=1.0, source="screen")
    assert f1.path == f2.path  # stored once
    other = Image.new("RGB", (64, 48), (0, 0, 0))
    other.paste((255, 255, 255), (0, 0, 32, 48))
    from lucius.storage import thumbnail
    f3 = sessions.add_frame(s1.id, other, seq=1, ts=2.0, source="screen", previous_thumb=thumbnail(image))
    assert f3.change_score is not None and f3.change_score > 0
    sessions.delete(s1.id)
    assert frames.path(f1.path).exists()  # still referenced by s2
    assert not frames.path(f3.path).exists()
    sessions.delete(s2.id)
    assert not frames.path(f1.path).exists()


def test_dhash_distance_bounds():
    a = "0" * 16
    assert hash_distance(a, a) == 0.0
    assert hash_distance(a, "f" * 16) == 1.0
    assert hash_distance(None, a) == 1.0


def test_bus_persists_decisions_but_not_capture_noise(bus, db):
    received = []
    bus.subscribe("*", received.append)
    bus.publish(EventType.FRAME_CAPTURED, "ses_x", frame_id="f")
    bus.publish(EventType.SKILL_PROMOTED, "blade", to="validated")
    assert len(received) == 2
    logged = bus.recent()
    assert [e["type"] for e in logged] == ["SKILL_PROMOTED"]


def test_bus_handler_failure_is_isolated(bus):
    def broken(_event):
        raise RuntimeError("boom")

    seen = []
    bus.subscribe(EventType.SEGMENT_CREATED, broken)
    bus.subscribe(EventType.SEGMENT_CREATED, seen.append)
    bus.publish(EventType.SEGMENT_CREATED, "seg")
    assert len(seen) == 1


def test_policy_defaults_are_training_opt_in():
    live = DataPolicy.for_live_demo()
    assert live.learning_allowed and not live.training_eligible
    consented = DataPolicy.for_live_demo(training_consent=True)
    assert consented.training_eligible
    from lucius.provenance import SourceClass

    external = DataPolicy.for_external(SourceClass.EXTERNAL_VIDEO)
    assert external.learning_allowed and not external.export_allowed and not external.training_eligible


def test_media_rebuild_migration_keeps_references(tmp_path, monkeypatch):
    """Migration 2 rebuilds media_assets; rows pointing at media (SET NULL and CASCADE) must survive."""
    import sqlite3

    import lucius.storage.db as dbmod

    all_migrations = dbmod._migrations()
    monkeypatch.setattr(dbmod, "_migrations", lambda: all_migrations[:1])
    path = tmp_path / "old.sqlite3"
    old = dbmod.Database(path)
    c = old.conn
    c.execute("INSERT INTO users(id, name, created_at) VALUES ('u', 'u', 0)")
    c.execute("INSERT INTO sessions(id, user_id, kind, source, status, start_time, policy) "
              "VALUES ('s', 'u', 'external_media', 'external_video', 'processed', 0, '{}')")
    c.execute("INSERT INTO media_assets(id, kind, role, filename, path, sha256, size_bytes, policy, created_at) "
              "VALUES ('m', 'video', 'demonstration', 'v.mp4', 'ab/v.mp4', 'ab', 1, '{}', 0)")
    c.execute("INSERT INTO frames(id, session_id, seq, ts, path, sha256, width, height, source, media_asset_id) "
              "VALUES ('f', 's', 0, 0, 'p', 'h', 1, 1, 'video', 'm')")
    c.execute("INSERT INTO reference_constraints(id, media_asset_id, constraint_type, target, value, source, "
              "confidence, created_at) VALUES ('r', 'm', 'aspect_ratio', 'object', '{}', 'measured', 1, 0)")
    with pytest.raises(sqlite3.IntegrityError):
        c.execute("INSERT INTO media_assets(id, kind, role, filename, path, sha256, size_bytes, policy, created_at) "
                  "VALUES ('c', 'captions', 'narration', 'v.json3', 'ab/v.json3', 'cd', 1, '{}', 0)")
    old.close()

    monkeypatch.setattr(dbmod, "_migrations", lambda: all_migrations)
    new = dbmod.Database(path)
    assert new.scalar("SELECT MAX(version) FROM schema_migrations") == 2
    assert new.scalar("SELECT media_asset_id FROM frames WHERE id = 'f'") == "m"          # not nulled
    assert new.scalar("SELECT COUNT(*) FROM reference_constraints WHERE media_asset_id = 'm'") == 1  # not deleted
    new.execute("INSERT INTO media_assets(id, kind, role, filename, path, sha256, size_bytes, policy, created_at) "
                "VALUES ('c', 'captions', 'narration', 'v.json3', 'ab/v.json3', 'cd', 1, '{}', 0)")
    assert new.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    new.execute("DELETE FROM media_assets WHERE id = 'm'")                                  # FKs still enforced
    assert new.scalar("SELECT COUNT(*) FROM reference_constraints") == 0
    assert new.scalar("SELECT media_asset_id FROM frames WHERE id = 'f'") is None
    new.close()
