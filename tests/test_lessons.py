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
        if purpose in ("lesson_recipe_fix", "make_fix"):
            # The broken selection (step 1) is replaced; everything else is kept.
            return ModelResult({"explanation": "the top face is at z 0.05", "edits": [
                {"op": "replace", "index": 1, "action": "select_box", "args": json.dumps(MUG_OK[1]["args"]),
                 "note": "top face"}]}, "scripted", self.model)
        if purpose in ("lesson_recipe_revise", "make_revise"):
            return ModelResult({"explanation": "rounder", "edits": [
                {"op": "insert_before", "index": 7, "action": "add_modifier",
                 "args": json.dumps({"object": "Mug", "type": "BEVEL", "props": {"width": 0.01}}), "note": ""}]},
                "scripted", self.model)
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
        # Relearning practises from the best recipe so far instead of writing a new one.
        again = learner.learn(video, [part], redo=True)
        assert [c["purpose"] for c in teacher.calls].count("lesson_recipe") == 1
        assert learner.projects.get(again[0].project_id).data["continued_from"]["score"] == 9

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


def test_extrude_keeps_the_bottom_of_a_lone_face_like_blender(headless_blender):
    def ex(action, **args):
        return headless_blender.execute(action, args, timeout=120)["result"]

    ex("reset_scene", keep_camera_light=False)
    ex("add_primitive", kind="circle", name="Base", vertices=8, fill=True)
    ex("select_all", object="Base")
    ex("extrude", object="Base", offset=[0, 0, 1])
    base = next(o for o in headless_blender.request("scene_summary")["objects"] if o["name"] == "Base")
    assert base["mesh"]["faces"] == 10                  # 8 walls, top and the kept bottom: a closed cup
    ex("select_box", object="Base", max=[None, None, 0.01], space="local", element="FACE")   # the bottom exists
    ex("add_primitive", kind="cylinder", name="Can", vertices=8, radius=1, depth=1)
    ex("select_box", object="Can", min=[None, None, 0.49], space="local", element="FACE", facing=[0, 0, 1])
    ex("extrude", object="Can", offset=[0, 0, 1])
    can = next(o for o in headless_blender.request("scene_summary")["objects"] if o["name"] == "Can")
    assert can["mesh"]["faces"] == 18 and can["mesh"]["edges"] == 40   # the old cap is gone, no loose edges


def test_patches_edit_a_recipe_by_step_index():
    from lucius.lessons.recipe import Recipe, RecipeStep, apply_patch

    recipe = Recipe(title="t", steps=[RecipeStep(action="add_primitive", args={"kind": "cube"}),
                                      RecipeStep(action="extrude", args={"offset": [0, 0, 1]}, video_time="1:00"),
                                      RecipeStep(action="shade", args={"smooth": True})])
    patched, problems = apply_patch(recipe, {"explanation": "x", "edits": [
        {"op": "replace", "index": 1, "action": "extrude", "args": '{"offset": [0, 0, 2]}', "note": ""},
        {"op": "insert_before", "index": 0, "action": "set_mode", "args": '{"mode": "OBJECT"}', "note": ""},
        {"op": "insert_before", "index": 3, "action": "fill", "args": "{}", "note": "end"},
        {"op": "delete", "index": 2, "action": None, "args": None, "note": ""},
        {"op": "replace", "index": 9, "action": "fill", "args": "{}", "note": ""},
        {"op": "replace", "index": 0, "action": "run_python", "args": "{}", "note": ""}]})
    assert [s.action for s in patched.steps] == ["set_mode", "add_primitive", "extrude", "fill"]
    assert patched.steps[2].args == {"offset": [0, 0, 2]} and patched.steps[2].video_time == "1:00"
    assert len(problems) == 2


def test_inset_works_on_a_lone_face_like_blender(headless_blender):
    def ex(action, **args):
        return headless_blender.execute(action, args, timeout=120)["result"]

    ex("reset_scene", keep_camera_light=False)
    ex("add_primitive", kind="circle", name="Dish", vertices=16, radius=2, fill=True)
    ex("select_all", object="Dish")
    assert ex("inset", object="Dish", thickness=0.4)["new_faces"] > 0
    assert ex("inset", object="Dish", thickness=0.4)["new_faces"] > 0     # the inner face stayed selected
    dish = next(o for o in headless_blender.request("scene_summary")["objects"] if o["name"] == "Dish")
    assert dish["mesh"]["verts"] == 48 and dish["mesh"]["faces"] == 33
    # Extruding the open rim and scaling it moves only the new loop, not the rim.
    ex("select_box", object="Dish", element="EDGE", space="local", boundary=True)
    ex("extrude", object="Dish", offset=[0, 0, -0.2])
    ex("scale_selection", object="Dish", factor=[0.5, 0.5, 1])
    dish = next(o for o in headless_blender.request("scene_summary")["objects"] if o["name"] == "Dish")
    assert dish["dimensions"][0] == pytest.approx(4.0, abs=0.01)


def test_scene_setup_for_a_final_shot(tmp_path):
    """Pattern and bump materials, light and camera updates that keep what is not given, working to scale, a
    particle system changed by name, and renders with the scene's own lights and settings."""
    pytest.importorskip("bpy")
    from lucius.blender.headless import HeadlessBlender

    out = tmp_path / "out"
    out.mkdir()
    with HeadlessBlender(allowed_save_dirs=[str(out)]) as bridge:
        def ex(action, **args):
            return bridge.execute(action, args, timeout=300)["result"]

        def objects():
            return {o["name"]: o for o in bridge.request("scene_summary")["objects"]}

        ex("reset_scene", keep_camera_light=False)
        ex("add_primitive", kind="plane", name="Cloth", size=20)
        made = ex("set_material", object="Cloth", name="Cloth", base_color=[1, 1, 1], pattern="brick",
                  pattern_color=[1, 1, 1], line_color=[0.01, 0.05, 0.31], pattern_scale=0.3, mortar_size=0.16,
                  bump="magic")
        assert made["set"]["pattern"] == "brick" and made["set"]["bump"] == "magic"
        ex("add_primitive", kind="torus", name="Donut", major_radius=1, minor_radius=0.4, location=[0, 0, 0.4])
        ex("add_primitive", kind="cube", name="Grain", size=0.05, location=[0, 0, -3])
        ex("add_scatter", object="Donut", instance="Grain", name="Sugar", count=100)
        ex("add_scatter", object="Donut", instance="Grain", name="Sugar", count=200, scale=1.5)   # changes it
        assert [m["type"] for m in objects()["Donut"]["modifiers"]] == ["PARTICLE_SYSTEM"]

        ex("add_light", type="SPOT", name="Key", location=[0, -6, 8], look_at=[0, 0, 0], power=5000,
           temperature=4500, spot_size=34, spot_blend=0.4)
        ex("add_light", name="Key", power=900)                          # only the power changes
        key = objects()["Key"]
        assert key["light"] == {"type": "SPOT", "energy": 900.0} and key["location"] == [0.0, -6.0, 8.0]
        ex("add_light", name="Key", type="AREA", size=3)                # a spot becomes a panel
        assert objects()["Key"]["light"]["type"] == "AREA"
        with pytest.raises(Exception, match="needs a location"):
            ex("add_light", name="Fill", power=10)

        ex("add_camera", name="Shot", location=[0, -9, 7], look_at=[0, 0, 0.4], lens=50)
        ex("add_camera", name="Shot", focus_object="Donut", fstop=1.0)   # focus only: the camera stays put
        shot = objects()["Shot"]
        assert shot["location"] == [0.0, -9.0, 7.0]
        assert shot["camera"] == {"lens": 50.0, "fstop": 1.0, "focus_object": "Donut"}

        ex("scale_scene", factor=0.05)
        scaled = objects()
        assert scaled["Donut"]["dimensions"][0] == pytest.approx(0.14, abs=1e-3)
        assert scaled["Shot"]["location"] == pytest.approx([0.0, -0.45, 0.35])
        assert scaled["Key"]["location"] == pytest.approx([0.0, -0.3, 0.4])

        lit = ex("render_image", path=str(out / "lit.png"), view="three_quarter", frame=["Donut"], samples=1,
                 width=48, height=48, lights="scene")
        assert Path(lit["path"]).exists()
        ex("set_render", samples=1, width=40, height=40, view_transform="Standard")
        final = ex("render_image", path=str(out / "final.png"), camera="scene", scene_settings=True)
        from PIL import Image

        assert Image.open(final["path"]).size == (40, 40)


def test_previews_frame_the_subject_not_the_ground():
    from lucius.lessons.teacher import subject_names

    objects = [{"name": "Cloth", "type": "MESH", "dimensions": [26, 26, 0]},
               {"name": "Plate", "type": "MESH", "dimensions": [6, 6, 0.4]},
               {"name": "Lamp", "type": "LIGHT", "dimensions": [0, 0, 0]}]
    assert subject_names(objects) == ["Plate"]
    assert subject_names(objects[:1]) == ["Cloth"]          # a plane alone is the subject


def test_a_course_pack_is_replayed_on_another_machine(tmp_path, headless_blender):
    """A pack of kept chapter recipes is rebuilt in order in an empty data folder, each chapter continuing
    from the one before, and kept with its teacher's score; a second replay skips what is already kept."""
    from PIL import Image

    from lucius.lessons.teacher import Teacher

    pack = tmp_path / "pack"
    (pack / "frames").mkdir(parents=True)
    Image.new("RGB", (32, 18), (90, 60, 30)).save(pack / "frames" / "01.jpg")
    (pack / "01_base.json").write_text(json.dumps({"title": "Base", "steps": [
        {"action": "add_primitive", "args": {"kind": "circle", "name": "Base", "vertices": 8, "fill": True}}]}))
    (pack / "02_cup.json").write_text(json.dumps({"title": "Cup", "steps": [
        {"action": "select_all", "args": {"object": "Base"}},
        {"action": "extrude", "args": {"object": "Base", "offset": [0, 0, 1]}},
        {"action": "set_material", "args": {"object": "Base", "name": "Clay", "base_color": "#C08040"}}]}))
    (pack / "course.json").write_text(json.dumps({
        "video_id": "vid456", "title": "Curso corto", "duration": 600, "teacher": "someone",
        "chapters": [{"title": "Base", "start_time": 0, "end_time": 300},
                     {"title": "Taza", "start_time": 300, "end_time": 600}],
        "lessons": [{"chapter": 1, "recipe": "01_base.json", "score": 9, "frame": "frames/01.jpg"},
                    {"chapter": 2, "recipe": "02_cup.json", "score": 7, "teacher": "another"}]}))
    app = Lucius(data_dir=tmp_path / "data", background_processing=False)
    try:
        teacher = Teacher(app, render_samples=2)
        results = teacher.install_course(pack)
        assert [r["status"] for r in results] == ["learned", "learned"], results
        assert results[0]["skill_id"] == "lesson_vid456_01"
        assert (teacher.projects.get(results[0]["project_id"]).dir / "tutorial_frame.png").exists()
        chapters = teacher.learner._state(teacher.video("vid456"))["chapters"]
        assert chapters["0-300"]["teacher"] == "someone" and chapters["300-600"]["teacher"] == "another"
        assert "Base (MESH)" in chapters["300-600"]["scene_after"]      # built on chapter 1's scene
        assert "Clay" in chapters["300-600"]["scene_after"] and teacher.name == "teacher"
        assert [r["status"] for r in teacher.install_course(pack)] == ["already learned"] * 2
        # A removed skill is forgotten with its chapter, so the next replay builds that chapter again.
        from lucius.cli import forget_lessons

        app.library.remove("lesson_vid456_02")
        forget_lessons(app.config.data_dir / "lessons", {"lesson_vid456_02"})
        assert not app.library.exists("lesson_vid456_02")
        assert [r["status"] for r in teacher.install_course(pack)] == ["already learned", "learned"]
        assert app.library.exists("lesson_vid456_02")
    finally:
        app.close()


def test_shade_smooth_sticks_when_given_in_edit_mode(headless_blender):
    def ex(action, **args):
        return headless_blender.execute(action, args, timeout=120)["result"]

    ex("reset_scene", keep_camera_light=False)
    ex("add_primitive", kind="cylinder", name="Can", vertices=12)
    ex("select_all", object="Can")
    ex("extrude", object="Can", offset=[0, 0, 1])      # leaves the object in edit mode
    ex("shade", object="Can", smooth=True)
    ex("set_mode", object="Can", mode="OBJECT")
    can = next(o for o in headless_blender.request("scene_summary")["objects"] if o["name"] == "Can")
    assert can["mesh"]["smooth_faces"] == can["mesh"]["faces"]


def test_proportional_editing_drop_and_modifier_updates(tmp_path):
    """O (proportional editing) makes neighbours follow a moved vertex; drop_object rests an object on what is below;
    a modifier added again by name changes instead of stacking; the dots pattern exists."""
    pytest.importorskip("bpy")
    from lucius.blender.headless import HeadlessBlender

    with HeadlessBlender() as bridge:
        def ex(action, **args):
            return bridge.execute(action, args, timeout=300)["result"]

        def obj(name):
            return next(o for o in bridge.request("scene_summary")["objects"] if o["name"] == name)

        ex("reset_scene", keep_camera_light=False)
        ex("add_primitive", kind="plane", name="Sheet", size=2)
        ex("select_all", object="Sheet")
        ex("subdivide", object="Sheet", cuts=9)
        ex("select_box", object="Sheet", element="VERT", min=[-0.05, -0.05, None], max=[0.05, 0.05, None], space="local")
        moved = ex("translate_selection", object="Sheet", offset=[0, 0, 1.0], proportional=0.6)
        assert moved["moved"] == 1 and moved["followed"] > 4
        sheet = obj("Sheet")
        assert sheet["dimensions"][2] == pytest.approx(1.0, abs=1e-3)          # the corners stayed on the ground
        ex("set_mode", object="Sheet", mode="OBJECT")

        ex("add_primitive", kind="cube", name="Box", size=1, location=[0, 0, 0.5])
        ex("add_primitive", kind="cube", name="Crate", size=0.4, location=[0.1, 0, 5])
        assert ex("drop_object", object="Crate", onto=["Box"])["z"] == pytest.approx(1.2, abs=1e-3)
        ex("add_primitive", kind="cube", name="Sunk", size=0.4, location=[3, 0, -0.1])
        assert ex("drop_object", object="Sunk")["z"] == pytest.approx(0.2, abs=1e-3)   # comes up onto the floor

        ex("add_modifier", object="Box", type="BEVEL", name="Bevel", props={"width": 0.05})
        ex("add_modifier", object="Box", type="BEVEL", name="Bevel", props={"width": 0.1, "segments": 3})
        assert [m["name"] for m in obj("Box")["modifiers"]] == ["Bevel"]
        made = ex("set_material", object="Box", name="Spots", base_color=[0.8, 0.05, 0.05], pattern="dots",
                  pattern_color=[0.8, 0.9, 0.9], pattern_scale=2.0, dot_size=0.3)
        assert made["set"]["pattern"] == "dots"


def test_taught_objects_are_rebuilt_without_a_model(tmp_path, headless_blender):
    from lucius.errors import ValidationError
    from lucius.lessons import Maker
    from lucius.lessons.teacher import Teacher, load_recipe, words

    assert words("Hazme una ESPADA bonita, por favor") == ["espada", "bonita"]
    path = tmp_path / "stool.json"
    path.write_text(json.dumps({"title": "Stool", "aliases": ["taburete"], "steps": [
        {"action": "add_primitive", "args": {"kind": "cylinder", "name": "Seat", "radius": 0.2, "depth": 0.05,
                                             "location": [0, 0, 0.5]}},
        {"action": "add_primitive", "args": {"kind": "cylinder", "name": "Leg", "radius": 0.03, "depth": 0.5,
                                             "location": [0, 0, 0.25]}}]}))
    app = Lucius(data_dir=tmp_path / "data", background_processing=False)
    try:
        teacher = Teacher(app, name="someone", render_samples=2)
        taught = teacher.teach_task(load_recipe(path), "a stool", score=9)
        assert taught.skill_id == "object_stool" and app.library.get("object_stool").status.value == "validated"
        maker = Maker(app, render_samples=2)
        assert maker.recall("un taburete alto")["skill_id"] == "object_stool"
        result = maker.make("un taburete alto", offline=True)
        assert result.status == "rebuilt" and result.skill_id == "object_stool" and result.final_render
        with pytest.raises(ValidationError, match="learned objects: Stool"):
            maker.make("a spaceship", offline=True)
    finally:
        app.close()


def test_edit_mode_building_and_joining(headless_blender):
    """Shift+A in edit mode joins the mesh; L grows a selection to the connected part; Shift+D copies it;
    Ctrl+J joins objects; auto smooth keeps right angles sharp."""
    def ex(action, **args):
        return headless_blender.execute(action, args, timeout=120)["result"]

    def obj(name):
        return next((o for o in headless_blender.request("scene_summary")["objects"] if o["name"] == name), None)

    ex("reset_scene", keep_camera_light=False)
    ex("add_primitive", kind="cube", name="Body", size=2, location=[0, 0, 1])
    assert ex("add_primitive", kind="cube", size=0.5, location=[0, 0, 3], into="Body")["added_verts"] == 8
    assert obj("Body")["mesh"]["verts"] == 16 and len(headless_blender.request("scene_summary")["objects"]) == 1
    ex("select_box", object="Body", element="VERT", min=[None, None, 2.2], max=[0.1, 0.1, 2.8], space="local")
    assert ex("select_linked", object="Body")["selected_verts"] == 8
    assert ex("duplicate_selection", object="Body", offset=[1, 0, 0])["copied_verts"] == 8
    assert obj("Body")["mesh"]["verts"] == 24
    assert ex("shade", object="Body", smooth=True, auto_smooth_deg=30)["sharp_edges"] == 36
    ex("add_primitive", kind="uv_sphere", name="Head", radius=0.5, location=[0, 0, 4])
    joined = ex("join_objects", names=["Head", "Body"], into="Body")
    assert joined["joined"] == ["Head"] and obj("Head") is None
    ex("set_world", sky=True, sun_elevation_deg=20, strength=0.1)
    assert ex("set_world", color=[0.1, 0.1, 0.1], strength=1.0)["sky"] is False


def test_animation_keyframes_shake_and_video(tmp_path):
    pytest.importorskip("bpy")
    from lucius.blender.headless import HeadlessBlender

    out = tmp_path / "out"
    out.mkdir()
    with HeadlessBlender(allowed_save_dirs=[str(out)]) as bridge:
        def ex(action, **args):
            return bridge.execute(action, args, timeout=600)["result"]

        ex("reset_scene", keep_camera_light=True)
        ex("add_primitive", kind="cube", name="Box", size=1)
        ex("insert_keyframe", object="Camera", frame=1, location=[6, -6, 3], look_at=[0, 0, 0])
        ex("insert_keyframe", object="Camera", frame=4, location=[0, -8, 2], look_at=[0, 0, 0], interpolation="LINEAR")
        assert ex("insert_keyframe", object="Camera", frame=1, focus_distance=5, fstop=2)["keyed"] == ["focus_distance",
                                                                                                        "fstop"]
        with pytest.raises(Exception, match="belong to a camera"):
            ex("insert_keyframe", object="Box", frame=1, lens=30)
        assert ex("set_frames", start=1, end=4, fps=12)["end"] == 4
        assert ex("add_shake", object="Camera", strength=0.05)["noisy_curves"] == 6
        animation = bridge.request("scene_summary")["animation"]
        assert animation["animated"] == ["Camera"] and animation["end"] == 4
        ex("set_render", samples=1, width=64, height=36, denoise=False)
        still = ex("render_image", path=str(out / "f4.png"), camera="scene", width=64, height=36, samples=1,
                   frame_number=4, lights="scene")
        assert Path(still["path"]).exists()
        video = ex("render_animation", path=str(out / "shot.mp4"), step=2, samples=1, percentage=33)   # odd -> even
        assert video["frames"] == 2 and video["bytes"] > 0
        with pytest.raises(Exception, match="must end with"):
            ex("render_animation", path=str(out / "shot.avi"))


def test_collections_and_scattered_collections_survive_the_next_chapter(tmp_path):
    """A chapter continues the previous chapter's saved scene: collections (even ones only a particle system uses)
    come back with the objects."""
    pytest.importorskip("bpy")
    from lucius.blender.headless import HeadlessBlender

    out = tmp_path / "out"
    out.mkdir()
    with HeadlessBlender(allowed_save_dirs=[str(out)], allowed_read_dirs=[str(out)]) as bridge:
        def ex(action, **args):
            return bridge.execute(action, args, timeout=300)["result"]

        ex("reset_scene", keep_camera_light=False)
        ex("add_primitive", kind="cone", name="Blade", radius=0.05, depth=0.4, location=[5, 0, 0])
        ex("add_primitive", kind="cone", name="Blade2", radius=0.08, depth=0.3, location=[6, 0, 0])
        assert ex("move_to_collection", names=["Blade", "Blade2"], collection="Grass")["objects"] == ["Blade", "Blade2"]
        ex("add_primitive", kind="plane", name="Ground", size=4)
        scattered = ex("add_scatter", object="Ground", collection="Grass", name="Lawn", count=50, children=5,
                       rotation_axis="OB_Y")
        assert scattered["children"] == 5
        ex("save_file", path=str(out / "chapter.blend"))
        ex("reset_scene", keep_camera_light=False)
        loaded = ex("import_blend", path=str(out / "chapter.blend"))
        assert "Ground" in loaded["objects"] and "Blade" not in loaded["objects"]   # hidden originals stay hidden
        ex("add_scatter", object="Ground", collection="Grass", name="Lawn", count=80)   # the collection is back


def test_rigging_bones_binding_constraints_and_drivers(tmp_path):
    pytest.importorskip("bpy")
    from lucius.blender.headless import HeadlessBlender

    out = tmp_path / "out"
    out.mkdir()
    with HeadlessBlender(allowed_save_dirs=[str(out)]) as bridge:
        def ex(action, **args):
            return bridge.execute(action, args, timeout=600)["result"]

        def obj(name):
            return next(o for o in bridge.request("scene_summary")["objects"] if o["name"] == name)

        ex("reset_scene", keep_camera_light=False)
        # a leg: a subdivided cylinder, two bones, automatic weights
        ex("add_primitive", kind="cylinder", name="Leg", radius=0.15, depth=2, vertices=12, location=[0.4, 0, 1])
        ex("loop_cut_axis", object="Leg", axis="z", positions=[0.2, 0.4, 0.5, 0.6, 0.8])
        ex("set_mode", object="Leg", mode="OBJECT")
        made = ex("add_armature", name="Rig", bones=[
            {"name": "thigh.L", "head": [0.4, 0, 2], "tail": [0.4, 0, 1]},
            {"name": "shin.L", "tail": [0.4, 0, 0], "parent": "thigh.L", "connect": True}])
        assert made["bones"] == ["thigh.L", "shin.L"]
        assert ex("symmetrize_bones", armature="Rig")["mirrored"] == ["thigh.R", "shin.R"]
        bound = ex("bind_to_armature", objects=["Leg"], armature="Rig", mode="AUTOMATIC")
        assert bound["vertex_groups"]["Leg"] >= 2
        before = obj("Leg")["dimensions"]
        ex("pose_bone", armature="Rig", bone="shin.L", rotation=[math.radians(80), 0, 0])
        bent = obj("Leg")["dimensions"]
        assert bent[1] > before[1] + 0.5          # the shin swung forward: the mesh bent with it
        # a rigid part on a bone keeps its place and then follows the bone
        ex("add_primitive", kind="cube", name="Knee", size=0.3, location=[0.4, 0, 1])
        ex("bind_to_armature", objects=["Knee"], armature="Rig", mode="BONE", bone="thigh.L")
        assert obj("Knee")["location"] == pytest.approx([0.4, 0, 1], abs=1e-4)
        assert obj("Knee")["parent_bone"] == "thigh.L"
        # empty groups + assigned weights
        ex("add_primitive", kind="cube", name="Box", size=0.5, location=[2, 0, 1])
        ex("bind_to_armature", objects=["Box"], armature="Rig", mode="EMPTY")
        ex("select_all", object="Box")
        assert ex("assign_weights", object="Box", group="thigh.L", weight=1.0, exclusive=True)["vertices"] == 8
        # an IK control bone and its constraint
        ex("add_armature", name="Rig", bones=[{"name": "ik.L", "head": [0.4, 0.5, 0], "tail": [0.4, 0.5, -0.3],
                                               "deform": False}])
        ex("pose_bone", armature="Rig", bone="shin.L", reset=True)
        con = ex("add_constraint", object="Rig", bone="shin.L", type="IK", target="Rig", subtarget="ik.L",
                 chain_count=2)
        assert con["type"] == "IK"
        ex("add_constraint", object="Rig", bone="ik.L", type="CHILD_OF", target="Rig", subtarget="thigh.L")
        # drivers: a cube's Z follows two spheres (one - two), and a bone's rotation drives a Z rotation
        ex("add_primitive", kind="uv_sphere", name="One", radius=0.2, location=[4, 0, 3])
        ex("add_primitive", kind="uv_sphere", name="Two", radius=0.2, location=[5, 0, 1])
        ex("add_primitive", kind="cube", name="Follower", size=0.3, location=[6, 0, 0])
        drv = ex("add_driver", object="Follower", path="location", index=2, expression="one - two", variables=[
            {"name": "one", "object": "One", "transform": "LOC_Z"}, {"name": "two", "object": "Two",
                                                                     "transform": "LOC_Z"}])
        assert drv["valid"]
        assert obj("Follower")["location"][2] == pytest.approx(2.0, abs=1e-3)
        ex("add_driver", object="Follower", path="rotation_euler", index=2, expression="var / 2", variables=[
            {"name": "var", "object": "Rig", "bone": "thigh.L", "transform": "ROT_X", "space": "TRANSFORM_SPACE"}])
        with pytest.raises(Exception, match="unknown name|may only use"):
            ex("add_driver", object="Follower", path="location", index=0, expression="__import__('os')",
               variables=[{"name": "var", "object": "One"}])
        with pytest.raises(Exception, match="property path"):
            ex("add_driver", object="Follower", path="location; import os", index=0, variables=[{"object": "One"}])
        shot = ex("render_image", path=str(out / "rig.png"), view="front", samples=1, width=64, height=48)
        assert Path(shot["path"]).exists()
        names = {o["name"] for o in bridge.request("scene_summary")["objects"]}
        assert not any(n.startswith("LuciusBoneShape") for n in names)   # stand-ins removed after the render


def test_ik_fk_switch_visual_pose_interpolation_and_retiming(tmp_path):
    pytest.importorskip("bpy")
    from lucius.blender.headless import HeadlessBlender

    with HeadlessBlender(allowed_save_dirs=[str(tmp_path)]) as bridge:
        def ex(action, **args):
            return bridge.execute(action, args, timeout=600)["result"]

        def tail(bone, frame):
            ex("set_frames", current=frame)
            return ex("pose_bone", armature="Rig", bone=bone)["tail"]

        ex("reset_scene", keep_camera_light=False)
        ex("add_armature", name="Rig", bones=[
            {"name": "thigh.L", "head": [0, 0, 2], "tail": [0, -0.05, 1]},
            {"name": "shin.L", "tail": [0, 0, 0], "parent": "thigh.L", "connect": True},
            {"name": "ik.L", "head": [0, 0, 0], "tail": [0, 0.3, 0], "deform": False}])
        ex("add_constraint", object="Rig", bone="shin.L", type="IK", target="Rig", subtarget="ik.L", chain_count=2)
        # the foot planted (IK) and lifted: the knee bends
        ex("pose_bone", armature="Rig", bone="ik.L", location=[0, 0, 0], frame=1)
        ex("pose_bone", armature="Rig", bone="ik.L", location=[0, -0.4, 0.8], frame=10)
        ex("key_constraint", object="Rig", bone="shin.L", constraint="IK", influence=1, frame=10)
        # FK matched to the IK pose (visual transform), then the switch to FK on the next frame
        for bone in ("thigh.L", "shin.L"):
            assert ex("pose_bone", armature="Rig", bone=bone, visual=True, frame=10)["keyed"]
            ex("pose_bone", armature="Rig", bone=bone, visual=True, frame=11)
        ex("key_constraint", object="Rig", bone="shin.L", constraint="IK", influence=0, frame=11)
        ik_pose = tail("shin.L", 10)
        assert ik_pose == pytest.approx([0, -0.4, 0.8], abs=0.02)
        assert tail("shin.L", 11) == pytest.approx(ik_pose, abs=0.02)   # no jump at the switch
        # FK from here: the thigh swings on its own, the IK handle no longer matters
        ex("pose_bone", armature="Rig", bone="thigh.L", rotation=[math.radians(-60), 0, 0], frame=20)
        ex("pose_bone", armature="Rig", bone="ik.L", location=[0, 0, 0], frame=20)
        assert tail("shin.L", 20) != pytest.approx(tail("shin.L", 11), abs=0.05)
        with pytest.raises(Exception, match="no constraint"):
            ex("key_constraint", object="Rig", bone="shin.L", constraint="Nope", influence=0, frame=12)
        # blocking: every key constant; then Bezier with vector handles in a range, and an ease
        assert ex("set_interpolation", object="Rig", interpolation="CONSTANT")["keys"] > 10
        assert tail("shin.L", 15) == pytest.approx(tail("shin.L", 11), abs=1e-3)   # held until the next key
        ex("set_interpolation", object="Rig", interpolation="BEZIER", handle="AUTO_CLAMPED")
        assert ex("set_interpolation", object="Rig", bones=["ik.L"], channels=["location"], axes="z",
                  start=5, end=30, handle="VECTOR")["keys"] == 2
        ex("set_interpolation", object="Rig", bones=["thigh.L"], ease_in=40, ease_out=40)
        # timing: the thigh's keys after frame 10 moved later (overlapping action), everything scaled in time
        moved = ex("retime_keys", object="Rig", bones=["thigh.L"], start=12, offset=4)
        assert moved["moved"] > 0 and moved["last"] == 24
        with pytest.raises(Exception, match="same frame"):
            ex("retime_keys", object="Rig", bones=["thigh.L"], start=24, end=24, offset=-13)
        with pytest.raises(Exception, match="same frame"):
            ex("retime_keys", object="Rig", scale=0.5, pivot=1)    # frames 10 and 11 would both become 6
        whole = ex("retime_keys", object="Rig", scale=0.8, pivot=1)
        assert whole["first"] == 1 and whole["last"] == 19     # 24 -> 1 + 23 * 0.8 = 19.4 -> 19 (snapped)
