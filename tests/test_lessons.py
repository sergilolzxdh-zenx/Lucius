"""Lessons (tutorial chapters as recipes), the maker, projects and the new Blender actions."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from lucius.app import Lucius
from lucius.ingestion.download import Chapter, DownloadedVideo
from lucius.ingestion.tutorial import TutorialPart
from lucius.lessons.catalogue import BASIC_ACTIONS, catalogue, normalize_args
from lucius.lessons.projects import ProjectStore, contact_sheet
from lucius.lessons.recipe import parse_recipe, recipe_schema
from lucius.providers.base import ModelResult, Providers

PIL = pytest.importorskip("PIL")


# -- recipes and arguments -----------------------------------------------------------------------------

def test_arguments_are_normalised_from_how_people_write_them():
    args, notes = normalize_args("rotate_selection", {"object": "Mug", "axis": "y", "angle_deg": 45})
    assert args["angle"] == pytest.approx(math.pi / 4) and "angle_deg" not in args and not notes
    args, notes = normalize_args("transform_object", {"object": "Mug", "rotation": [90, 0, 0]})
    assert args["rotation"][0] == pytest.approx(math.pi / 2) and notes == ["rotation read as degrees"]
    args, _ = normalize_args("transform_object", {"object": "Mug", "rotation": [1.2, 0, 0]})
    assert args["rotation"][0] == 1.2                       # already radians
    args, _ = normalize_args("set_material", {"object": "Mug", "base_color": "#FF8000", "roughness": 0.4})
    assert args["base_color"][0] == pytest.approx(1.0) and args["base_color"][1] == pytest.approx(0.2158, abs=1e-3)
    args, _ = normalize_args("add_modifier", {"object": "M", "type": "SUBSURF", "props": {"levels": 2}})
    assert args["props"] == {"levels": 2}
    args, _ = normalize_args("add_primitive", {"kind": "UV Sphere"})
    assert args["kind"] == "uv_sphere"


def test_model_recipes_are_parsed_and_checked():
    data = {"title": "Mug", "summary": "a mug", "objects": [{"name": "Mug", "description": "cup"}],
            "expected_result": "a mug", "steps": [
                {"action": "add_primitive", "args": '{"kind": "cylinder", "name": "Mug"}', "note": "", "video_time": "1:00"},
                {"action": "extrude", "args": "{not json", "note": "", "video_time": None},
                {"action": "bevel", "args": '{"object": "Mug", "offset": 0.01}', "note": "", "video_time": None},
                {"action": "run_python", "args": "{}", "note": "", "video_time": None}]}
    recipe, problems = parse_recipe(data, allowed=["add_primitive", "extrude"])
    assert [s.action for s in recipe.steps] == ["add_primitive"]
    assert len(problems) == 3 and "not valid JSON" in problems[0] and "not available" in problems[1]
    assert recipe.object_names() == ["Mug"]
    schema = recipe_schema(["add_primitive"])
    assert schema["properties"]["steps"]["items"]["properties"]["action"]["enum"] == ["add_primitive"]
    text = catalogue(BASIC_ACTIONS)
    assert "select_box" in text and "bevel" not in text


def test_contact_sheet_and_project_store(tmp_path):
    from PIL import Image

    a = tmp_path / "a.png"
    Image.new("RGB", (160, 90), (200, 50, 50)).save(a)
    sheet = contact_sheet([(a, "tutorial"), (None, "attempt 1: failed")], tmp_path / "sheet.png", title="Mug")
    assert Image.open(sheet).size[0] == 2 * 480
    store = ProjectStore(tmp_path / "projects")
    project = store.create("task", "a sword", task="a sword")
    project.add_attempt({"number": 1, "score": 5})
    assert store.list()[0]["attempts"] == 1
    (project.path("render.png")).write_bytes(a.read_bytes())
    assert store.file(project.id, "render.png").exists()
    with pytest.raises(Exception, match="not found"):
        store.file(project.id, "../../etc/passwd")
    with pytest.raises(Exception, match="not found"):
        store.get("../x")


def test_storyboard_frame_is_cropped_from_the_sprite_sheet(tmp_path, monkeypatch):
    from PIL import Image

    from lucius.lessons import references

    sheet = Image.new("RGB", (800, 450), (0, 0, 0))
    sheet.paste(Image.new("RGB", (160, 90), (0, 255, 0)), (160 * 2, 90 * 1))  # tile 7 (row 1, col 2)
    sheet_path = tmp_path / "sheet.jpg"
    sheet.save(sheet_path, quality=95)
    monkeypatch.setattr(references, "_fetch", lambda url, dest: sheet_path)
    info = {"id": "v", "duration": 500, "formats": [
        {"format_id": "sb0", "width": 159, "height": 90, "rows": 5, "columns": 5, "fps": 0.1,
         "fragments": [{"url": "https://i.ytimg.com/sb/v/M0.jpg"}, {"url": "https://i.ytimg.com/sb/v/M1.jpg"}]}]}
    frame = references.storyboard_frame(info, 72.0, tmp_path, tmp_path / "f.png")   # 72 s -> frame 7
    pixel = Image.open(frame).getpixel((80, 45))
    assert pixel[1] > 200 and pixel[0] < 60


def test_later_skills_do_not_reset_the_scene(tmp_path):
    from lucius.planner.model import TaskSpec
    from lucius.skills.schema import ActionTemplate, SkillDefinition, SkillPhase

    app = Lucius(data_dir=tmp_path / "data", background_processing=False)
    try:
        skills = []
        for name in ("first", "second"):
            definition = SkillDefinition(skill_id=f"s_{name}", name=name, purpose=name, phases=[SkillPhase(
                name="setup", actions=[ActionTemplate(action_type="reset_scene", object_ref=None),
                                       ActionTemplate(action_type="add_primitive", object_ref=None,
                                                      args={"kind": "cube", "name": name})])])
            skills.append(app.library.create(definition, created_by="test", change_note="test"))
        plan = app.planner.plan_skills(TaskSpec(text="two parts"), skills)
        names = [a.name for s in plan.steps for a in s.actions]
        assert names.count("reset_scene") == 1 and names[0] == "reset_scene"
        # A recipe step (a Blender action with concrete values) compiles to itself.
        from lucius.planner.compile import CompileContext, compile_template

        actions = compile_template(ActionTemplate(action_type="rotate_selection", object_ref=None,
                                                  args={"object": "M", "axis": "y", "angle_deg": 90}),
                                   {}, CompileContext())
        assert actions[0].name == "rotate_selection" and actions[0].args["angle"] == pytest.approx(math.pi / 2)
    finally:
        app.close()


# -- the learning loop, end to end with a scripted model --------------------------------------------------

MUG_OK = [
    {"action": "add_primitive", "args": {"kind": "cylinder", "name": "Mug", "radius": 0.4, "depth": 0.1,
                                         "location": [0, 0, 0.05]}},
    {"action": "select_box", "args": {"object": "Mug", "element": "FACE", "min": [None, None, 0.04],
                                      "space": "local", "facing": [0, 0, 1]}},
    {"action": "extrude", "args": {"object": "Mug", "offset": [0, 0, 0.9]}},
    {"action": "inset", "args": {"object": "Mug", "thickness": 0.05}},
    {"action": "extrude", "args": {"object": "Mug", "offset": [0, 0, -0.85]}},
    {"action": "add_modifier", "args": {"object": "Mug", "type": "SUBSURF", "props": {"levels": 2}}},
    {"action": "shade", "args": {"object": "Mug", "smooth": True}},
    {"action": "set_material", "args": {"object": "Mug", "base_color": "#F2EEE8", "roughness": 0.3}},
]
BROKEN = [MUG_OK[0], {**MUG_OK[1], "args": {**MUG_OK[1]["args"], "min": [None, None, 5.0]}}, *MUG_OK[2:]]


def _recipe(steps, title="Mug"):
    return {"title": title, "summary": "a ceramic mug", "objects": [{"name": "Mug", "description": "mug"}],
            "expected_result": "a white mug", "steps": [
                {"action": s["action"], "args": json.dumps(s["args"]), "note": "", "video_time": "32:10"}
                for s in steps]}


class ScriptedTeacher:
    """Plays the video model: notes, a recipe with one broken selection, comparisons that improve."""

    name = "scripted"
    model = "scripted-video"
    supports_images = True
    supports_video = True

    def __init__(self):
        self.calls = []
        self.compares = 0

    def complete_json(self, *, purpose, system, prompt, schema, images=(), max_tokens=8000, videos=()):
        self.calls.append({"purpose": purpose, "images": len(images), "videos": len(videos), "prompt": prompt})
        if purpose == "chapter_title_translation":
            return ModelResult({"titles": []}, "scripted", self.model)
        if purpose == "lesson_notes":
            return ModelResult({"summary": "models a mug", "operations": [
                {"time": "32:10", "object": "Mug", "operation": "Extrude", "selection": "top face",
                 "values": "0.9 m up", "value_source": "estimated", "effect": "taller"}],
                "objects_at_end": [{"name": "Mug", "shape": "mug", "size": "0.8 x 0.8 x 1", "location": "origin",
                                    "material": None}]}, "scripted", self.model)
        if purpose == "lesson_recipe":
            return ModelResult(_recipe(BROKEN), "scripted", self.model)
        if purpose in ("lesson_recipe_fix", "lesson_recipe_revise", "make_fix", "make_revise"):
            return ModelResult(_recipe(MUG_OK), "scripted", self.model)
        if purpose == "lesson_compare":
            self.compares += 1
            score = 5 if self.compares == 1 else 9
            return ModelResult({"score": score, "verdict": "close" if score < 8 else "good", "matches": ["mug body"],
                                "differences": [{"object": "Mug", "problem": "no handle", "fix": "add a handle"}],
                                "reference_time": "57:40"}, "scripted", self.model)
        if purpose == "make_plan":
            return ModelResult({**_recipe(MUG_OK, "Cup"), "missing_techniques": []}, "scripted", self.model)
        if purpose == "make_critique":
            return ModelResult({"score": 8, "verdict": "good", "looks_like": "a white cup", "matches": ["cup"],
                                "differences": []}, "scripted", self.model)
        raise AssertionError(f"unexpected call {purpose}")


def test_a_chapter_is_learned_rebuilt_compared_and_practised(tmp_path, headless_blender):
    app = Lucius(data_dir=tmp_path / "data", background_processing=False)
    teacher = ScriptedTeacher()
    app.providers = Providers(llm=teacher, vlm=teacher, embeddings=app.providers.embeddings)
    app.ingestion.watcher.providers = app.providers
    try:
        from lucius.lessons import LessonLearner, Maker

        video = DownloadedVideo(video_id="vid123", url="https://www.youtube.com/watch?v=vid123", title="Curso",
                                duration=3600.0, language="es", video_path=None, captions_path=None,
                                captions_source=None, info_path=tmp_path / "none",
                                chapters=[Chapter(index=3, title="Modelar la taza", start=1914.0, end=2100.0)])
        part = TutorialPart(title="Curso — 4. Modelar la taza", start=1914.0, end=2100.0,
                            task_text="Model the mug (Modelar la taza)", chapter=video.chapters[0])
        messages = []
        learner = LessonLearner(app, practice_rounds=2, render_samples=4)
        results = learner.learn(video, [part], on_progress=messages.append)
        assert len(results) == 1
        result = results[0]
        assert result.status == "learned" and result.score == 9 and result.attempts == 2
        project = learner.projects.get(result.project_id)
        attempts = project.data["attempts"]
        assert attempts[0]["fixes"] == 1 and attempts[0]["run"]["ok"] and attempts[0]["score"] == 5
        assert (project.dir / "sheet.png").exists() and (project.dir / "scene.blend").exists()
        assert (project.dir / "attempt1_three_quarter.png").exists()
        mug = next(o for o in attempts[-1]["run"]["scene"] if o["name"] == "Mug")
        assert mug["dimensions"][2] == pytest.approx(1.0, abs=0.05) and mug["materials"]
        purposes = [c["purpose"] for c in teacher.calls]
        assert purposes.count("lesson_notes") == 1 and "lesson_recipe_fix" in purposes
        compare = next(c for c in teacher.calls if c["purpose"] == "lesson_compare")
        assert compare["videos"] == 1 and compare["images"] >= 1
        skill = app.library.get(result.skill_id)
        assert skill.status.value == "validated" and "tutorial_recipe" in skill.definition.categories
        state = json.loads((app.config.data_dir / "lessons" / "vid123" / "state.json").read_text())
        assert state["chapters"]["1914-2100"]["status"] == "learned"

        # Resuming skips the learned chapter; the notes are cached.
        assert learner.learn(video, [part]) == []

        # The maker may use what the lesson taught (extrude, inset, modifiers...), plus basic actions.
        maker = Maker(app, iterations=2, render_samples=4)
        techniques = maker.techniques()
        assert {"extrude", "inset", "set_material"} <= techniques.actions and "SUBSURF" in techniques.modifiers
        assert "bridge_edge_loops" not in techniques.allowed(False)
        made = maker.make("a cup for coffee", on_progress=messages.append)
        assert made.status == "made" and made.score == 8 and made.skill_id
        assert app.library.get(made.skill_id).status.value != "validated"   # self-judged: not validated
        from lucius.lessons import rate_project

        rated = rate_project(app, made.project_id, good=True, note="looks like a cup")
        assert rated["rating"]["good"] is True
        assert app.library.get(made.skill_id).status.value == "validated"
        plan = next(c for c in teacher.calls if c["purpose"] == "make_plan")
        assert "Model the mug" in plan["prompt"] or "Mug" in plan["prompt"]
    finally:
        app.close()


def test_bridge_scene_actions_render(tmp_path, headless_blender):
    """Materials, lights, camera, box selection, bridging and a render, in headless Blender."""
    from lucius.blender.headless import HeadlessBlender

    out = tmp_path / "out"
    out.mkdir()
    with HeadlessBlender(allowed_save_dirs=[str(out)]) as bridge:
        def ex(action, **args):
            return bridge.execute(action, args, timeout=300)["result"]

        ex("reset_scene", keep_camera_light=False)
        ex("add_primitive", kind="cylinder", name="Mug", radius=0.4, depth=0.1, vertices=16, location=[0, 0, 0.05])
        ex("select_box", object="Mug", min=[None, None, 0.04], element="FACE", space="local", facing=[0, 0, 1])
        ex("extrude", object="Mug", offset=[0, 0, 0.9])
        ex("loop_cut_axis", object="Mug", axis="z", positions=[0.25, 0.5, 0.75])
        assert ex("select_box", object="Mug", min=[0.3, -0.12, 0.5], max=[None, 0.12, 0.65], element="FACE",
                  space="local", facing=[1, 0, 0])["selected"] >= 1
        ex("extrude", object="Mug", distance=0.2)
        ex("rotate_selection", object="Mug", axis="y", angle=math.radians(45))
        ex("delete_elements", object="Mug", what="ONLY_FACES")
        with pytest.raises(Exception, match="nothing inside the box"):
            ex("select_box", object="Mug", min=[None, None, 50.0], space="local")
        ex("add_primitive", kind="torus", name="Donut", major_radius=0.3, minor_radius=0.12, location=[1, 0, 0.12])
        ex("select_box", object="Donut", min=[None, None, 0.5], element="FACE")
        assert ex("separate_selection", object="Donut", new_name="Icing")["object"] == "Icing"
        ex("add_primitive", kind="cylinder", name="Sprinkle", radius=0.01, depth=0.04, vertices=6)
        assert ex("add_scatter", object="Icing", instance="Sprinkle", count=50)["count"] == 50
        ex("set_material", object="Icing", name="Pink", base_color=[0.9, 0.3, 0.5], roughness=0.3)
        ex("add_light", type="AREA", name="Key", location=[2, -2, 3], look_at=[0, 0, 0], power=300)
        ex("add_light", type="SUN", name="Key", location=[2, -2, 3], power=2)              # updates, no duplicate
        ex("add_camera", name="Shot", location=[3, -3, 2], look_at=[0.5, 0, 0.3], lens=40)
        summary = bridge.request("scene_summary")
        names = {o["name"]: o for o in summary["objects"]}
        assert names["Key"]["light"]["type"] == "SUN" and summary["camera"] == "Shot"
        assert names["Icing"]["materials"] == ["Pink"] and "Sprinkle" not in names   # moved out of the scene
        path = ex("render_image", path=str(out / "shot.png"), camera="scene", samples=2, width=160, height=120)["path"]
        assert Path(path).exists()
        ex("render_image", path=str(out / "auto.png"), view="front", frame=["Mug"], samples=2, width=160, height=120)
        with pytest.raises(Exception, match="outside the directories"):
            ex("render_image", path=str(tmp_path / "elsewhere.png"), samples=1, width=32, height=32)


def test_misspelled_arguments_are_matched_to_the_real_ones():
    from lucius.lessons.catalogue import fix_keys

    fixed, notes = fix_keys("select_box", {".max": [1, 1, 1], "elemnt": "FACE", "min": [0, 0, 0], "colour": 1},
                            {"object", "min", "max", "element", "space"})
    assert fixed == {"max": [1, 1, 1], "element": "FACE", "min": [0, 0, 0], "colour": 1}
    assert len(notes) == 2
    fixed, _ = fix_keys("rotate_selection", {"angle_deg": 45}, {"object", "axis", "angle"})
    assert fixed == {"angle_deg": 45}          # degree spellings are converted later, not renamed
