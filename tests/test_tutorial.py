"""Narrated tutorials: captions, spoken actions, narration/video alignment, chapters and clip windows."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

from lucius.app import Lucius
from lucius.ingestion import DemoStatus, MediaInput, MediaRole
from lucius.ingestion.captions import CaptionTrack, estimate_lag, find_mentions
from lucius.ingestion.download import (
    Chapter,
    DownloadedVideo,
    _pick_captions,
    _run,
    parse_timestamp,
    select_chapters,
)
from lucius.ingestion.media import MediaKind
from lucius.ingestion.tutorial import TutorialImporter
from lucius.providers.base import ModelResult, Providers
from lucius.provenance import EvidenceKind


def _json3(path, phrases):
    """phrases: [(start_s, "words ...")] -> a YouTube-style json3 file with per-word offsets."""
    events = [{"tStartMs": 0, "dDurationMs": 600000, "id": 1}]
    for start, text in phrases:
        segs = [{"utf8": ("" if i == 0 else " ") + w, "tOffsetMs": i * 250, "acAsrConf": 0}
                for i, w in enumerate(text.split())]
        segs[0].pop("tOffsetMs")
        events.append({"tStartMs": int(start * 1000), "dDurationMs": 4000, "wWinId": 1, "segs": segs})
        events.append({"tStartMs": int(start * 1000) + 10, "dDurationMs": 2000, "wWinId": 1, "aAppend": 1,
                       "segs": [{"utf8": "\n"}]})
    path.write_text(json.dumps({"wireMagic": "pb3", "events": events}))
    return path


# -- parsing ---------------------------------------------------------------------------------------------

def test_json3_words_keep_their_own_times(tmp_path):
    track = CaptionTrack.from_file(_json3(tmp_path / "abc.es-orig.json3", [(10.0, "pulsamos la e para extruir"),
                                                                           (20.0, "y ahora escalamos")]))
    assert track.language == "es" and track.source == "auto"
    assert [w.text for w in track.words][:3] == ["pulsamos", "la", "e"]
    assert track.words[2].t == pytest.approx(10.5) and track.words[0].end == pytest.approx(10.25)
    assert track.text(19.0, 30.0) == "y ahora escalamos"
    assert len(track.window(15, 30).words) == 3


def test_vtt_rolling_duplicates_and_srt(tmp_path):
    vtt = tmp_path / "v.en.vtt"
    vtt.write_text("""WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:03.000 align:start position:0%

now<00:00:01.400><c> press</c><00:00:01.800><c> E</c>

00:00:03.000 --> 00:00:03.010 align:start position:0%
now press E

00:00:03.010 --> 00:00:05.000 align:start position:0%
now press E
to<00:00:03.500><c> extrude</c>
""")
    track = CaptionTrack.from_file(vtt)
    assert [w.text for w in track.words] == ["now", "press", "E", "to", "extrude"]  # repeated line dropped
    assert track.words[2].t == pytest.approx(1.8) and track.words[4].t == pytest.approx(3.5)
    srt = tmp_path / "s.srt"
    srt.write_text("1\n00:00:02,000 --> 00:00:04,000\nAdd a cube\n\n2\n00:00:05,000 --> 00:00:06,000\nthen bevel it\n")
    words = CaptionTrack.from_file(srt, language="en").words
    assert [w.text for w in words] == ["Add", "a", "cube", "then", "bevel", "it"] and words[3].t == 5.0


def test_spoken_actions_in_english_and_spanish(tmp_path):
    track = CaptionTrack.from_file(_json3(tmp_path / "t.json3", [
        (1.0, "press R then press X and it rotates on the x axis"),
        (10.0, "pulsamos la g para moverlo"),
        (20.0, "vamos a añadir un cilindro"),
        (30.0, "con control r hacemos un corte en bucle"),
        (40.0, "pulsa a la derecha del panel"),          # 'a' here is a preposition, not the A key
        (50.0, "we add a mirror modifier"),
    ]))
    mentions = find_mentions(track.cues)
    by_t = {round(m.t): m for m in mentions}
    assert by_t[1].action == "rotate" and by_t[1].params["axis"] == "x"  # X after R is an axis, not delete
    assert not any(m.action == "delete" for m in mentions)
    assert by_t[10].action == "translate" and by_t[10].params["hotkey"] == "g"
    assert by_t[20].action == "add_primitive" and by_t[20].params["kind"] == "cylinder"
    assert by_t[30].action == "loop_cut" and by_t[30].params["hotkey"] == "ctrl+r"
    assert 40 not in by_t
    assert any(m.action == "set_symmetry" and m.params.get("type") == "MIRROR" for m in mentions)


# -- alignment -------------------------------------------------------------------------------------------

def test_lag_is_recovered_only_when_it_beats_chance():
    rng = random.Random(7)
    events = sorted(rng.uniform(0, 600) for _ in range(120))
    said = [e - 1.5 + rng.uniform(-0.3, 0.3) for e in rng.sample(events, 30)]   # narrator speaks 1.5 s early
    noise = [rng.uniform(0, 600) for _ in range(10)]                             # talk about other things
    estimate = estimate_lag(said + noise, events)
    assert estimate.applied and estimate.lag_s == pytest.approx(1.5, abs=0.5) and estimate.z > 2
    unrelated = estimate_lag([rng.uniform(0, 600) for _ in range(40)], events)
    assert not unrelated.applied and unrelated.lag_s == 0.0
    assert estimate_lag([], events).reason == "no_mentions_or_events"


# -- ingestion of a narrated clip ------------------------------------------------------------------------

pytest.importorskip("cv2")


@pytest.fixture
def app(tmp_path):
    instance = Lucius(data_dir=tmp_path / "data", background_processing=False)
    yield instance
    instance.close()


def _narrated_video(tmp_path):
    from tests.fixtures.media import tutorial_script, write_video

    video = tmp_path / "tut.mp4"
    write_video(video, tutorial_script())   # changes at 2 (mode), 4 (scale x), 6 (scale z), 8 (orbit), 10 (extrude)
    captions = _json3(tmp_path / "tut.en.json3", [(1.0, "tab into edit mode"), (3.2, "now press S to scale it"),
                                                  (9.1, "and extrude it upwards")])
    return video, captions


@pytest.mark.media
def test_clip_with_narration(app, tmp_path):
    video, captions = _narrated_video(tmp_path)
    demo = app.ingestion.create(title="clip", task_text="scale and extrude", inputs=[
        MediaInput(path=str(video), clip_start=3.0, clip_end=12.0, source_url="https://example.invalid/v"),
        MediaInput(path=str(captions), role=MediaRole.NARRATION, kind=MediaKind.CAPTIONS, language="en")])
    result = app.ingestion.process(demo.id, validate=False)
    assert result.status == DemoStatus.READY, result.status_history[-1]
    steps = app.trajectories.for_session(result.session_id)
    stamps = [s.media_timestamp for s in steps]
    assert all(t >= 3.0 for t in stamps), stamps                  # the mode change at 2 s is outside the clip
    assert any(abs(t - 4.0) <= 0.8 for t in stamps) and any(abs(t - 10.0) <= 0.8 for t in stamps)
    assert min(s.t_start for s in steps) >= app.sessions.get(result.session_id).start_time  # clip starts at 0
    scale = next(s for s in steps if abs(s.media_timestamp - 4.0) <= 0.8)
    assert scale.action_type == "scale" and "now press S to scale it" in scale.meta["narration"]
    assert scale.action_payload["input"] == [{"hotkey": "S", "source": "narration"}]  # hint for GUI execution
    assert any(e.startswith("narration") for c in scale.candidate_actions if c.action_type == "scale"
               for e in c.evidence)
    extrude = next(s for s in steps if abs(s.media_timestamp - 10.0) <= 0.8)
    assert any(c.action_type == "extrude" for c in extrude.candidate_actions)
    frames = app.sessions.frames_for(result.session_id)
    assert all(3.0 <= f.media_timestamp <= 12.0 for f in frames)
    notes = [e.payload for e in app.sessions.events(result.session_id, kinds=["annotation"])]
    assert [n["label"] for n in notes] == ["narration", "narration"]  # cues inside the clip only
    video_asset = next(a for a in app.ingestion.media.list(demonstration_id=demo.id) if a.kind == MediaKind.VIDEO)
    assert video_asset.analysis["clip"] == [3.0, 12.0]
    assert video_asset.analysis["narration"]["alignment"]["reason"] == "too_few_matches"  # 2 mentions: no shift
    assert any(s.evidence_kind in (EvidenceKind.NARRATION, EvidenceKind.VISUAL_STATE_TRANSITION) for s in steps)


class ScriptedVision:
    name, model, supports_images = "scripted", "scripted-vlm", True

    def __init__(self):
        self.calls = []

    def complete_json(self, *, purpose, system, prompt, schema, images=(), max_tokens=8000):
        self.calls.append({"purpose": purpose, "prompt": prompt, "images": [i.label for i in images]})
        if purpose == "chapter_title_translation":
            return ModelResult({"titles": [{"index": 0, "english": "Pipe modelling"}]}, "scripted", self.model)
        indices = [int(label.split()[1]) for label in (i for i in self.calls[-1]["images"]) if "BEFORE" in label]
        return ModelResult({"transitions": [
            {"index": i, "change_category": "geometry", "description": "d", "visible_mode": None,
             "visible_tool": None, "visible_keystrokes": [],
             "candidate_operations": [{"operation": "scale", "confidence": 0.9, "evidence": "e"}]}
            for i in indices]}, "scripted", self.model)


@pytest.mark.media
def test_vision_budget_prefers_narrated_changes(app, tmp_path):
    video, captions = _narrated_video(tmp_path)
    vision = ScriptedVision()
    app.providers = Providers(llm=vision, vlm=vision, embeddings=app.providers.embeddings)
    app.ingestion.vision.providers = app.providers
    app.config.processing.max_vision_pairs = 2
    demo = app.ingestion.create(title="budget", task_text="scale", inputs=[
        MediaInput(path=str(video)), MediaInput(path=str(captions), role=MediaRole.NARRATION, kind=MediaKind.CAPTIONS)])
    result = app.ingestion.process(demo.id, validate=False)
    assert result.status == DemoStatus.READY
    vision_calls = [c for c in vision.calls if c["purpose"] == "media_transition_analysis"]
    assert len(vision_calls) == 1 and len(vision_calls[0]["images"]) == 4          # 2 pairs, one call
    assert "Narration around pair" in vision_calls[0]["prompt"]
    info = next(h for h in result.status_history if h["status"] == "INFERRING_ACTIONS")["vision"]
    assert info["sent"] == 2 and info["answered"] == 2 and info["transitions"] >= 4
    steps = app.trajectories.for_session(result.session_id)
    sent = {int(label.split()[1]) for label in vision_calls[0]["images"] if "BEFORE" in label}
    narrated = [i for i, s in enumerate(sorted(steps, key=lambda s: s.t_start)) if s.meta.get("narration")
                and any(c.action_type in ("scale", "extrude") and any("narration" in e for e in c.evidence)
                        for c in s.candidate_actions)]
    assert len(narrated) > len(sent) and sent <= set(narrated)  # the budget went to changes someone talked about


# -- chapters, windows and translation -------------------------------------------------------------------

def _video(chapters, language="es", duration=3600.0):
    return DownloadedVideo(video_id="vid", url="https://example.invalid/v", title="Guía", duration=duration,
                           language=language, video_path=None, captions_path=None, captions_source=None,
                           info_path=None, chapters=chapters)


def test_plan_chapters_windows_and_translation(app):
    chapters = [Chapter(0, "Intro", 0, 150), Chapter(1, "Instalación Blender", 150, 270),
                Chapter(2, "Modelado tubería", 564, 1152), Chapter(3, "Texturizado", 1152, 1800)]
    importer = TutorialImporter(app)
    parts = importer.plan(_video(chapters), translate=False)
    assert [p.skipped is None for p in parts] == [False, False, True, True]
    assert parts[2].title == "Guía — 3. Modelado tubería" and (parts[2].start, parts[2].end) == (564, 1152)
    assert [p.chapter.index for p in importer.plan(_video(chapters), chapters="3-4", translate=False)] == [2, 3]
    window = importer.plan(_video(chapters), start=600.0, end=1200.0, translate=False)
    assert len(window) == 1 and window[0].task_text == "Modelado tubería" and window[0].end == 1200.0
    unchaptered = importer.plan(_video([], duration=2000.0), window_s=900.0, translate=False)
    assert [(p.start, p.end) for p in unchaptered] == [(0.0, 900.0), (900.0, 1800.0), (1800.0, 2000.0)]
    vision = ScriptedVision()
    app.providers = Providers(llm=vision, vlm=vision, embeddings=app.providers.embeddings)
    translated = importer.plan(_video(chapters), chapters="3")
    assert translated[0].task_text == "Pipe modelling (Modelado tubería)"
    english = importer.plan(_video(chapters, language="en"), chapters="3")
    assert english[0].task_text == "Modelado tubería" and len(vision.calls) == 1  # English needs no call


def test_timestamps_and_chapter_selection():
    assert parse_timestamp("1:02:03") == 3723 and parse_timestamp("9:24") == 564 and parse_timestamp("90.5") == 90.5
    chapters = [Chapter(i, f"c{i}", i * 10.0, i * 10.0 + 10) for i in range(6)]
    assert [c.index for c in select_chapters(chapters, "2,4-5")] == [1, 3, 4]
    assert len(select_chapters(chapters, None)) == 6


# -- download --------------------------------------------------------------------------------------------

def test_caption_choice_prefers_manual_then_original_speech_track():
    info = {"language": "es", "subtitles": {}, "automatic_captions": {"es-orig": [{}], "es": [{}], "en": [{}]}}
    assert _pick_captions(info, None) == (["es-orig"], "auto")
    info["subtitles"] = {"es-ES": [{"ext": "vtt"}]}
    assert _pick_captions(info, None) == (["es-ES"], "manual")
    assert _pick_captions({"automatic_captions": {"de-orig": [{}]}}, None) == (["de-orig"], "auto")
    assert _pick_captions({}, None) == ([], None)


def test_blocked_downloads_try_other_clients(monkeypatch):
    class DownloadError(Exception):
        pass

    attempts = []

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download):
            client = (self.opts.get("extractor_args") or {}).get("youtube", {}).get("player_client", ["default"])[0]
            attempts.append(client)
            if client in ("default", "android_vr"):
                raise DownloadError("ERROR: Sign in to confirm you're not a bot")
            return {"id": "x", "client": client}

        def sanitize_info(self, info):
            return info

    fake = SimpleNamespace(YoutubeDL=FakeYDL, utils=SimpleNamespace(DownloadError=DownloadError))
    monkeypatch.setattr("lucius.ingestion.download._yt_dlp", lambda: fake)
    monkeypatch.setattr("lucius.ingestion.download.time.sleep", lambda _s: None)
    assert _run("https://example.invalid/v", {}, download=False)["client"] == "tv"
    assert attempts == ["default", "default", "android_vr", "android_vr", "tv"]

    def not_a_block(self, url, download):
        raise DownloadError("ERROR: Video unavailable")

    monkeypatch.setattr(FakeYDL, "extract_info", not_a_block)
    with pytest.raises(Exception, match="download failed"):
        _run("https://example.invalid/v", {}, download=False)


# -- watching by URL with a video model ------------------------------------------------------------------

class ScriptedVideoModel:
    name, model, supports_images, supports_video = "scripted", "scripted-video", True, True

    def __init__(self, relative=False):
        self.calls = []
        self.relative = relative

    def complete_json(self, *, purpose, system, prompt, schema, images=(), max_tokens=8000, videos=()):
        self.calls.append({"purpose": purpose, "prompt": prompt, "videos": list(videos)})
        if purpose == "chapter_title_translation":
            return ModelResult({"titles": []}, "scripted", self.model)
        clip = videos[0]
        base = 0.0 if self.relative else clip.start_s

        def at(offset):
            t = base + offset
            return f"{int(t // 60):02d}:{t % 60:04.1f}"

        def op(offset, operation, evidence="visible_change", **extra):
            item = {"start": at(offset), "end": at(offset + 2), "operation": operation, "object": "Circle",
                    "mode": "Edit Mode", "axis": None, "value": None, "kind": None, "modifier_type": "NONE",
                    "evidence": evidence, "confidence": 0.9, "description": operation}
            item.update(extra)
            return item

        return ModelResult({"summary": "pipe", "operations": [
            op(10, "add_primitive", kind="circle", mode="Object Mode"),
            op(40, "extrude", evidence="visible_keystroke_overlay"),
            op(70, "select_all", evidence="narration_only"),
            op(100, "add_modifier", modifier_type="subsurf"),
            {**op(0, "scale"), "start": "99:00", "end": "99:02"},           # outside the clip: misread
        ]}, "scripted", self.model)


def _remote_app(app, model):
    app.providers = Providers(llm=model, vlm=model, embeddings=app.providers.embeddings)
    app.ingestion.vision.providers = app.providers
    app.ingestion.watcher.providers = app.providers
    return app


@pytest.mark.parametrize("relative", [False, True])
def test_video_model_watches_a_clip_by_url(app, tmp_path, relative):
    model = ScriptedVideoModel(relative=relative)
    _remote_app(app, model)
    captions = _json3(tmp_path / "v.es-orig.json3", [(639.5, "le damos a la e de extrude y arrastramos"),
                                                     (669.0, "con la a seleccionamos todo")])
    demo = app.ingestion.create(title="pipe", task_text="Model pipe", inputs=[
        MediaInput(url="https://www.youtube.com/watch?v=abc", kind=MediaKind.VIDEO, clip_start=600.0, clip_end=720.0),
        MediaInput(path=str(captions), role=MediaRole.NARRATION, kind=MediaKind.CAPTIONS, language="es")])
    assert "video_url" in demo.source_types and demo.source_class == "external_video"
    result = app.ingestion.process(demo.id, validate=False)
    assert result.status == DemoStatus.READY, result.status_history[-1]
    video = model.calls[0]["videos"][0]
    assert (video.uri, video.start_s, video.end_s) == ("https://www.youtube.com/watch?v=abc", 600.0, 720.0)
    assert "[10:39] le damos a la e de extrude" in model.calls[0]["prompt"]   # narration sent with the clip
    steps = sorted(app.trajectories.for_session(result.session_id), key=lambda s: s.t_start)
    assert [round(s.media_timestamp) for s in steps] == [610, 640, 670, 700]  # player time, misread one dropped
    add, extrude, select, modifier = steps
    assert add.params == {"kind": "circle"} and add.state_after["active_object"] == "Circle"
    assert add.mode_label == "OBJECT" and extrude.mode_label == "EDIT_MESH"
    assert extrude.evidence_kind == EvidenceKind.VISIBLE_SHORTCUT
    assert extrude.action_confidence > 0.9 and extrude.action_payload["input"][0]["hotkey"] == "E"  # narration agrees
    assert select.evidence_kind == EvidenceKind.NARRATION and select.action_type == "unknown_action"  # only heard
    assert modifier.params == {"type": "SUBSURF"} and "type" not in extrude.params  # 'NONE' ignored
    assert all(s.action_source.value == "model_inferred" and not s.frame_before_id for s in steps)
    asset = next(a for a in app.ingestion.media.list(demonstration_id=demo.id) if a.kind == MediaKind.VIDEO)
    assert asset.analysis["remote"] and asset.analysis["watch"]["answered"] == 1
    with pytest.raises(Exception, match="remote media has no local file"):
        app.ingestion.media.path(asset)


def test_watcher_chunks_long_clips_and_stops_on_exhausted_quota(app):
    from lucius.errors import ProviderError
    from lucius.ingestion.video_model import VideoWatcher

    class Exhausted(ScriptedVideoModel):
        def complete_json(self, **kwargs):
            self.calls.append(kwargs)
            raise ProviderError("daily quota for 'm' is exhausted", purpose="video_watch", transient=False)

    watcher = VideoWatcher(Providers(vlm=(model := Exhausted()), embeddings=app.providers.embeddings), chunk_s=300)
    assert watcher.chunks(0, 590) == [(0, 295), (295, 590)] and len(watcher.chunks(0, 1500)) == 5
    report = watcher.watch("https://example.invalid/v", 0, 1500, context="x")
    assert report.answered == 0 and len(model.calls) == 1 and "daily quota" in report.errors[0]


@pytest.mark.bpy
def test_validation_cannot_pass_a_skill_whose_values_nobody_showed(app, tmp_path):
    """Found live: a pipe skill from a video had 14 of its operations uncompilable (distances unknown); the
    run executed only 'add circle', its one check (circle exists) passed, and the skill was 'validated'."""
    from lucius.blender.headless import headless_available

    if headless_available() is None:
        pytest.skip("needs Blender")
    class PipeModel(ScriptedVideoModel):
        def complete_json(self, *, purpose, system, prompt, schema, images=(), max_tokens=8000, videos=()):
            if purpose != "video_watch":
                return super().complete_json(purpose=purpose, system=system, prompt=prompt, schema=schema)
            ops = [("add_primitive", "Object Mode", {"kind": "circle"}), ("mode_change", "Edit Mode", {}),
                   ("fill", "Edit Mode", {}), ("extrude", "Edit Mode", {}), ("scale", "Edit Mode", {}),
                   ("extrude", "Edit Mode", {}), ("inset", "Edit Mode", {}), ("mode_change", "Object Mode", {}),
                   ("shade_smooth", "Object Mode", {})]
            return ModelResult({"summary": "pipe", "operations": [
                {"start": f"10:{10 + 5 * i:02d}", "end": f"10:{12 + 5 * i:02d}", "operation": name, "object": "Circle",
                 "mode": mode, "axis": None, "value": None, "kind": extra.get("kind"), "modifier_type": None,
                 "evidence": "visible_change", "confidence": 0.9, "description": name}
                for i, (name, mode, extra) in enumerate(ops)]}, "scripted", self.model)

    _remote_app(app, PipeModel())
    demo = app.ingestion.create(title="pipe", task_text="Model pipe", inputs=[
        MediaInput(url="https://www.youtube.com/watch?v=abc", kind=MediaKind.VIDEO, clip_start=600.0, clip_end=720.0)])
    result = app.ingestion.process(demo.id, validate=True)
    assert result.status == DemoStatus.READY, result.status_history[-1]
    runs = next(h for h in result.status_history if h["status"] == "READY")["validation"]["runs"]
    assert runs
    for skill_id, run in runs.items():
        assert run["unresolved"] and run["verdict"] != "success"
        assert app.library.get(skill_id).status.value in ("candidate_pattern", "candidate_skill")


def test_an_object_of_unknown_kind_is_not_guessed_to_be_a_cube():
    """Found live: 'add an Area light' (reported without a kind) became a skill that adds a cube, and its only
    check ('Area exists') passed -- a validated skill that does something else entirely."""
    from lucius.planner.compile import CompileContext, Uncompilable, compile_template
    from lucius.skills.schema import ActionTemplate

    for kind in (None, "area", "point"):
        template = ActionTemplate(action_type="add_primitive", args={"kind": kind, "name": "{object_name}"})
        with pytest.raises(Uncompilable, match="unknown or not a mesh primitive"):
            compile_template(template, {"object_name": "Area"}, CompileContext())
    ok = compile_template(ActionTemplate(action_type="add_primitive", args={"kind": "monkey", "name": "{object_name}"}),
                          {"object_name": "Suzanne"}, CompileContext())
    assert ok[-1].args["kind"] == "monkey"


@pytest.mark.bpy
def test_validating_a_skill_credits_only_that_skill(app):
    """Found in the tutorial run: validation retrieved other skills into its plan, and a generic skill collected
    92 'uses' from other skills' validations."""
    from lucius.blender.headless import headless_available
    from tests.fixtures.demos import sword_blockout_demo
    from tests.test_e2e_learning import record

    if headless_available() is None:
        pytest.skip("needs Blender")
    session_id = record(app, sword_blockout_demo(blade_length=6.0, blade_width=0.3, variant="a", t0=1_700_000_000.0))
    record(app, sword_blockout_demo(blade_length=4.0, blade_width=0.24, variant="b", with_mistake=False,
                                    t0=1_700_100_000.0))
    uses = {s: app.library.get(s).usage_count for s in ("hard_surface_blade_blockout", "hard_surface_guard_blockout")}
    report = app.ingestion._validate(None, session_id, ["hard_surface_blade_blockout"])
    run = report["runs"]["hard_surface_blade_blockout"]
    plan = app.db.query_one("SELECT plan FROM runs WHERE id = ?", (run["run_id"],))["plan"]
    assert '"hard_surface_guard_blockout"' not in plan                      # only the skill under test is planned
    metrics = json.loads(app.db.query_one("SELECT metrics FROM runs WHERE id = ?", (run["run_id"],))["metrics"])
    credited = {c["skill_id"] for c in metrics.get("pending_credit", [])}  # the silhouette awaits a person here
    if run["verdict"] != "needs_human":
        credited.add("hard_surface_blade_blockout")
        assert app.library.get("hard_surface_blade_blockout").usage_count == uses["hard_surface_blade_blockout"] + 1
    assert credited == {"hard_surface_blade_blockout"}
    assert app.library.get("hard_surface_guard_blockout").usage_count == uses["hard_surface_guard_blockout"]
