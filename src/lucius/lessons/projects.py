"""Projects: everything Lucius builds, kept where a person can look at it and judge it.

Each lesson chapter and each task gets a folder under ``<data dir>/projects``: the recipe of every
attempt, renders of every attempt, the tutorial's own frame (lessons) or the reference image
(tasks), the final ``.blend`` (open it in Blender), a side-by-side ``sheet.png`` and ``project.json``
with scores, the model's comparison notes and the person's rating. Ratings feed back into the skill
the project produced (a "good" confirms it, a "bad" counts against it).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from lucius.errors import NotFoundError
from lucius.ids import new_id

SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]{1,120}$")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "project"


class Project:
    def __init__(self, directory: Path, data: dict[str, Any]) -> None:
        self.dir = directory
        self.data = data

    @property
    def id(self) -> str:
        return str(self.data["id"])

    def path(self, name: str) -> Path:
        return self.dir / name

    def save(self) -> None:
        self.data["updated_at"] = time.time()
        tmp = self.dir / "project.json.tmp"
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False, default=str))
        tmp.replace(self.dir / "project.json")

    def add_attempt(self, attempt: dict[str, Any]) -> None:
        self.data.setdefault("attempts", []).append(attempt)
        self.save()

    def summary(self) -> dict[str, Any]:
        d = self.data
        return {"id": d["id"], "kind": d.get("kind"), "title": d.get("title"), "status": d.get("status"),
                "score": d.get("score"), "attempts": len(d.get("attempts", [])), "rating": d.get("rating"),
                "skill_id": d.get("skill_id"), "sheet": d.get("sheet"), "final_render": d.get("final_render"),
                "created_at": d.get("created_at"), "dir": str(self.dir)}


class ProjectStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def create(self, kind: str, title: str, **meta: Any) -> Project:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        project_id = f"{stamp}-{_slug(title)}-{new_id('p')[-6:]}"
        directory = self.root / project_id
        directory.mkdir(parents=True, exist_ok=False)
        project = Project(directory, {"id": project_id, "kind": kind, "title": title, "status": "running",
                                      "created_at": time.time(), "attempts": [], **meta})
        project.save()
        return project

    def get(self, project_id: str) -> Project:
        if not SAFE_ID.match(project_id):
            raise NotFoundError(f"project {project_id!r} not found")
        path = self.root / project_id / "project.json"
        if not path.exists():
            raise NotFoundError(f"project {project_id!r} not found")
        return Project(path.parent, json.loads(path.read_text()))

    def list(self, kind: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        out = []
        if not self.root.exists():
            return out
        for path in sorted(self.root.glob("*/project.json"), reverse=True):
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if kind is None or data.get("kind") == kind:
                out.append(Project(path.parent, data).summary())
            if len(out) >= limit:
                break
        return out

    def file(self, project_id: str, name: str) -> Path:
        """A file of a project, refusing anything outside its folder."""
        project = self.get(project_id)
        path = (project.dir / name).resolve()
        if project.dir.resolve() not in path.parents or not path.is_file():
            raise NotFoundError(f"{name} not found in project {project_id}")
        return path


def contact_sheet(tiles: list[tuple[Path | None, str]], dest: Path, *, tile_w: int = 480, tile_h: int = 360,
                  title: str = "") -> Path:
    """Images side by side with a caption under each (a missing image is a grey tile)."""
    from PIL import Image, ImageDraw, ImageFont

    caption_h, title_h = 44, (34 if title else 0)
    columns = min(4, max(1, len(tiles)))
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_w, title_h + rows * (tile_h + caption_h)), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    if title:
        draw.text((12, 8), title[:110], fill=(235, 235, 235), font=font)
    for i, (path, caption) in enumerate(tiles):
        x, y = (i % columns) * tile_w, title_h + (i // columns) * (tile_h + caption_h)
        if path is not None and Path(path).exists():
            img = Image.open(path).convert("RGB")
            img.thumbnail((tile_w - 8, tile_h - 8), Image.Resampling.LANCZOS)
            if max(img.size) < min(tile_w, tile_h) // 2:   # tiny storyboard frames: enlarge for viewing
                factor = min((tile_w - 8) / img.width, (tile_h - 8) / img.height)
                img = img.resize((int(img.width * factor), int(img.height * factor)), Image.Resampling.LANCZOS)
            sheet.paste(img, (x + (tile_w - img.width) // 2, y + (tile_h - img.height) // 2))
        else:
            draw.rectangle((x + 4, y + 4, x + tile_w - 4, y + tile_h - 4), fill=(60, 60, 64))
            draw.text((x + 16, y + tile_h // 2), "no image", fill=(200, 200, 200), font=font)
        for line_no, line in enumerate(_wrap(caption, 46)[:2]):
            draw.text((x + 8, y + tile_h + 4 + line_no * 20), line, fill=(230, 230, 230), font=font)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest)
    return dest


def _wrap(text: str, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width and line:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines
